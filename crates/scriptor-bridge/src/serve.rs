//! The native OCR bridge HTTP server: the OCR contract over
//! HTTP/1.1 with one thread per connection, optional bearer-token
//! authorisation and optional TLS. The OCR work itself is a [`DocumentOcr`].
//!
//! Routes: `GET /health`, `GET /progress/<sha256>`, `POST /cancel/<sha256>`
//! and `POST /ocr` (a JSON `path` request or the raw source bytes with the
//! `Semantik-Source-SHA256` and `Semantik-Contract` headers).

use anyhow::{Result, anyhow, bail};
use serde::de::{self, Deserialize, MapAccess, SeqAccess, Visitor};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Largest accepted JSON request body.
pub const MAX_REQUEST_BYTES: u64 = 1024 * 1024;
/// Largest accepted source document.
pub const MAX_SOURCE_BYTES: u64 = 256 * 1024 * 1024;
/// Largest encoded reply.
pub const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
/// Documents converted at once when the environment does not say.
pub const DEFAULT_HTTP_CONCURRENCY: usize = 2;
/// Most documents the environment may allow at once.
pub const MAX_HTTP_CONCURRENCY: usize = 32;
/// Socket read and write timeout of one connection.
pub const HTTP_READ_TIMEOUT_SECS: u64 = 30;
/// Environment variable holding the bearer token of a non-loopback bind.
pub const DEFAULT_TOKEN_ENV: &str = "SCRIPTOR_BRIDGE_TOKEN";
/// Environment variable bounding concurrent documents.
pub const HTTP_CONCURRENCY_ENV: &str = "SCRIPTOR_BRIDGE_HTTP_CONCURRENCY";
/// Environment variable naming the `max_num_seqs` of the OCR model server;
/// without an explicit concurrency the bridge serves that many documents.
pub const MAX_NUM_SEQS_ENV: &str = "SCRIPTOR_BRIDGE_MAX_NUM_SEQS";
/// Port served when none is given.
pub const DEFAULT_PORT: u16 = 8765;
/// Value of the `Semantik-Contract` header of a bytes request.
pub const BYTES_CONTRACT_HEADER: &str = "pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1";
/// Contracts the health reply advertises.
pub const SUPPORTED_CONTRACTS: [&str; 2] = ["path", "bytes"];
/// Request header selecting the task of a bytes request.
pub const TEXT_MODE_HEADER: &str = "Semantik-Task";
/// Value of the task header (and of a path request's `mode`) that selects text.
pub const TEXT_MODE: &str = "text";

const BODY_CHUNK_BYTES: usize = 1024 * 1024;
const BODY_CONTENT_TYPES: [&str; 4] = ["application/pdf", "image/png", "image/jpeg", "image/webp"];
const MAX_HEAD_LINE_BYTES: u64 = 65_536;
const MAX_HEADERS: usize = 100;
const ACCEPT_POLL: Duration = Duration::from_millis(10);
const SERVER_NAME: &str = "semantik-glm-ocr";

/// What a request asks the converter to produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrTask {
    /// Layout regions with their text.
    Layout,
    /// Plain document text.
    Text,
}

/// Why a conversion produced no reply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OcrFailure {
    /// Cancelled through `POST /cancel/<sha256>`; answered with 409.
    Cancelled,
    /// The OCR service or model is unavailable; answered with 503. The
    /// diagnostics are `(category, count)` pairs written to the log.
    Unavailable { diagnostics: Vec<(String, u64)> },
    /// The source cannot be converted; answered with 422.
    Rejected,
    /// An unexpected failure; answered with 500.
    Internal,
}

/// The conversion behind the server.
pub trait DocumentOcr: Send + Sync {
    /// Converts `source` (a supported PDF or image whose digest the server
    /// has already verified) into the reply JSON of the OCR contract.
    /// `progress(pages_done, pages_total)` publishes progress for
    /// `GET /progress/<sha256>`; `cancel()` turns true once
    /// `POST /cancel/<sha256>` has been accepted, and the conversion then
    /// returns [`OcrFailure::Cancelled`] at its next safe point.
    /// `diagnostic(category, count)` records a counter of a conversion that
    /// succeeds; a failing conversion carries its counters in the failure.
    fn ocr(
        &self,
        source: &[u8],
        task: OcrTask,
        progress: &(dyn Fn(usize, usize) + Sync),
        cancel: &(dyn Fn() -> bool + Sync),
        diagnostic: &(dyn Fn(&str, u64) + Sync),
    ) -> Result<Value, OcrFailure>;

    /// [`DocumentOcr::ocr`] that also publishes which pages are being read:
    /// `pages(reading, read)` takes the one-based ascending numbers of the
    /// pages being read and of the pages read, for `GET /progress/<sha256>`.
    /// A conversion that cannot tell publishes none.
    fn ocr_with_pages(
        &self,
        source: &[u8],
        task: OcrTask,
        progress: &(dyn Fn(usize, usize) + Sync),
        _pages: &(dyn Fn(&[i64], &[i64]) + Sync),
        cancel: &(dyn Fn() -> bool + Sync),
        diagnostic: &(dyn Fn(&str, u64) + Sync),
    ) -> Result<Value, OcrFailure> {
        self.ocr(source, task, progress, cancel, diagnostic)
    }

    /// [`DocumentOcr::ocr_with_pages`] that also publishes the load:
    /// `load(requests, workers)` takes the region recognition requests of
    /// this conversion in flight now and its worker pool size, for
    /// `GET /progress/<sha256>`. A conversion that cannot tell publishes
    /// none.
    #[allow(clippy::too_many_arguments)]
    fn ocr_with_load(
        &self,
        source: &[u8],
        task: OcrTask,
        progress: &(dyn Fn(usize, usize) + Sync),
        pages: &(dyn Fn(&[i64], &[i64]) + Sync),
        _load: &(dyn Fn(usize, usize) + Sync),
        cancel: &(dyn Fn() -> bool + Sync),
        diagnostic: &(dyn Fn(&str, u64) + Sync),
    ) -> Result<Value, OcrFailure> {
        self.ocr_with_pages(source, task, progress, pages, cancel, diagnostic)
    }
}

/// A byte stream of one connection.
trait Conn: Read + Write + Send {}
impl<T: Read + Write + Send> Conn for T {}

/// Validated settings of a server.
#[derive(Clone)]
pub struct ServeConfig {
    pub address: IpAddr,
    pub port: u16,
    pub tls: Option<Arc<rustls::ServerConfig>>,
    pub token: Option<String>,
    pub max_requests: usize,
}

impl ServeConfig {
    /// Settings from the command-line options and `env`: the bind address
    /// defaults to loopback, a non-loopback address requires TLS and a bearer
    /// token (named by `token_env`, default [`DEFAULT_TOKEN_ENV`]), a token
    /// name on loopback turns authorisation on, the certificate and key are
    /// given together, and concurrency comes from [`HTTP_CONCURRENCY_ENV`].
    pub fn from_options(
        port: Option<u16>,
        listen: Option<&str>,
        tls_cert: Option<&Path>,
        tls_key: Option<&Path>,
        token_env: Option<&str>,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ServeConfig> {
        let port = match port {
            None | Some(0) => DEFAULT_PORT,
            Some(port) => port,
        };
        let address: IpAddr = listen
            .unwrap_or("127.0.0.1")
            .parse()
            .map_err(|_| anyhow!("HTTP bridge bind address is invalid"))?;
        let token = if !address.is_loopback() {
            if tls_cert.is_none() || tls_key.is_none() {
                bail!("a non-loopback bind address requires --tls-cert and --tls-key");
            }
            Some(bearer_token(env, token_env.unwrap_or(DEFAULT_TOKEN_ENV))?)
        } else {
            token_env.map(|name| bearer_token(env, name)).transpose()?
        };
        if tls_cert.is_some() != tls_key.is_some() {
            bail!("--tls-cert and --tls-key are used together");
        }
        let tls = match (tls_cert, tls_key) {
            (Some(cert), Some(key)) => Some(tls_config(cert, key)?),
            _ => None,
        };
        Ok(ServeConfig {
            address,
            port,
            tls,
            token,
            max_requests: http_concurrency(env)?,
        })
    }
}

fn bearer_token(env: &dyn Fn(&str) -> Option<String>, name: &str) -> Result<String> {
    match env(name) {
        Some(token) if !token.is_empty() => Ok(token),
        _ => bail!("bearer token environment variable {name} is not set"),
    }
}

fn http_concurrency(env: &dyn Fn(&str) -> Option<String>) -> Result<usize> {
    let invalid = || anyhow!("HTTP bridge concurrency is invalid");
    if env(HTTP_CONCURRENCY_ENV).is_none()
        && let Some(text) = env(MAX_NUM_SEQS_ENV)
    {
        return match text.trim().parse::<usize>() {
            Ok(served) if served >= 1 => Ok(served),
            _ => Err(anyhow!("OCR server max_num_seqs is invalid")),
        };
    }
    let value = match env(HTTP_CONCURRENCY_ENV) {
        Some(text) => text.trim().parse::<i64>().map_err(|_| invalid())?,
        None => DEFAULT_HTTP_CONCURRENCY as i64,
    };
    if (1..=MAX_HTTP_CONCURRENCY as i64).contains(&value) {
        Ok(value as usize)
    } else {
        Err(invalid())
    }
}

fn tls_config(cert: &Path, key: &Path) -> Result<Arc<rustls::ServerConfig>> {
    use rustls_pki_types::pem::PemObject;
    use rustls_pki_types::{CertificateDer, PrivateKeyDer};
    let unusable = || anyhow!("TLS certificate or key could not be loaded");
    let chain: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(cert)
        .map_err(|_| unusable())?
        .collect::<Result<_, _>>()
        .map_err(|_| unusable())?;
    if chain.is_empty() {
        return Err(unusable());
    }
    let key = PrivateKeyDer::from_pem_file(key).map_err(|_| unusable())?;
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let config = rustls::ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS12, &rustls::version::TLS13])
        .map_err(|_| unusable())?
        .with_no_client_auth()
        .with_single_cert(chain, key)
        .map_err(|_| unusable())?;
    Ok(Arc::new(config))
}

/// What a running source has reported: pages done of the total, and the
/// pages being read and read.
struct Progress {
    done: usize,
    total: usize,
    reading: Vec<i64>,
    read: Vec<i64>,
    /// Region requests in flight and the worker pool size, when reported.
    load: Option<(usize, usize)>,
}

#[derive(Default)]
struct Book {
    progress: HashMap<String, (usize, usize)>,
    pages: HashMap<String, (Vec<i64>, Vec<i64>)>,
    load: HashMap<String, (usize, usize)>,
    cancelled: HashSet<String>,
}

struct Slots(AtomicUsize);

impl Slots {
    fn try_acquire(&self) -> bool {
        self.0
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |free| {
                free.checked_sub(1)
            })
            .is_ok()
    }

    fn release(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct Slot<'a>(&'a Slots);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.0.release();
    }
}

struct Shared {
    ocr: Arc<dyn DocumentOcr>,
    token: Option<Vec<u8>>,
    slots: Slots,
    max_requests: usize,
    book: Mutex<Book>,
    diagnostics: Mutex<HashMap<String, u64>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Shared {
    fn authorized(&self, header: Option<&str>) -> bool {
        let Some(token) = &self.token else {
            return true;
        };
        let value = header.unwrap_or("");
        let (scheme, presented) = value.split_once(' ').unwrap_or((value, ""));
        scheme.eq_ignore_ascii_case("bearer")
            && constant_time_eq(presented.trim().as_bytes(), token)
    }

    fn cancel(&self, digest: &str) -> bool {
        let mut book = lock(&self.book);
        if !book.progress.contains_key(digest) {
            return false;
        }
        book.cancelled.insert(digest.to_string());
        true
    }

    fn progress_of(&self, digest: &str) -> Option<Progress> {
        let book = lock(&self.book);
        let (done, total) = book.progress.get(digest).copied()?;
        let (reading, read) = book.pages.get(digest).cloned().unwrap_or_default();
        Some(Progress {
            done,
            total,
            reading,
            read,
            load: book.load.get(digest).copied(),
        })
    }

    fn record_diagnostics(&self, diagnostics: &[(String, u64)]) {
        let unknown = [("sdk_failure".to_string(), 1u64)];
        let diagnostics = if diagnostics.is_empty() {
            &unknown[..]
        } else {
            diagnostics
        };
        for (category, count) in diagnostics {
            let valid = !category.is_empty()
                && category
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
                && *count >= 1;
            let (category, count) = if valid {
                (category.as_str(), *count)
            } else {
                ("sdk_failure", 1)
            };
            let total = {
                let mut totals = lock(&self.diagnostics);
                let total = totals.entry(category.to_string()).or_insert(0);
                *total += count;
                *total
            };
            let _ = writeln!(
                std::io::stderr(),
                "ocr-bridge: diagnostic category={category} count={count} total={total}"
            );
        }
    }

    fn convert(&self, source: &[u8], digest: &str, task: OcrTask) -> Result<Vec<u8>, OcrFailure> {
        let progress = |done: usize, total: usize| {
            lock(&self.book)
                .progress
                .insert(digest.to_string(), (done, total));
        };
        let pages = |reading: &[i64], read: &[i64]| {
            lock(&self.book)
                .pages
                .insert(digest.to_string(), (reading.to_vec(), read.to_vec()));
        };
        let load = |requests: usize, workers: usize| {
            lock(&self.book)
                .load
                .insert(digest.to_string(), (requests, workers));
        };
        let cancel = || lock(&self.book).cancelled.contains(digest);
        let diagnostic = |category: &str, count: u64| {
            self.record_diagnostics(&[(category.to_string(), count)]);
        };
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
            self.ocr
                .ocr_with_load(source, task, &progress, &pages, &load, &cancel, &diagnostic)
        }));
        {
            let mut book = lock(&self.book);
            book.progress.remove(digest);
            book.pages.remove(digest);
            book.load.remove(digest);
            book.cancelled.remove(digest);
        }
        let reply = match outcome {
            Ok(Ok(reply)) => reply,
            Ok(Err(failure)) => {
                if let OcrFailure::Unavailable { diagnostics } = &failure {
                    self.record_diagnostics(diagnostics);
                }
                return Err(failure);
            }
            Err(_) => return Err(OcrFailure::Internal),
        };
        let encoded = serde_json::to_vec(&reply).map_err(|_| OcrFailure::Internal)?;
        if encoded.len() > MAX_RESPONSE_BYTES {
            return Err(OcrFailure::Rejected);
        }
        Ok(encoded)
    }

    fn convert_request(&self, data: &[u8]) -> Result<Vec<u8>, OcrFailure> {
        let request = parse_request(data).ok_or(OcrFailure::Rejected)?;
        let source = read_source(&request.source, &request.digest).ok_or(OcrFailure::Rejected)?;
        let task = if request.text {
            OcrTask::Text
        } else {
            OcrTask::Layout
        };
        self.convert(&source, &request.digest, task)
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        let a = left.get(index).copied().unwrap_or(0);
        let b = right.get(index).copied().unwrap_or(0);
        difference |= usize::from(a ^ b);
    }
    difference == 0
}

fn is_digest(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Whether `data` starts with the signature of a PDF or a supported image.
pub fn supported_source(data: &[u8]) -> bool {
    data.starts_with(b"%PDF-")
        || data.starts_with(b"\x89PNG\r\n\x1a\n")
        || data.starts_with(b"\xff\xd8\xff")
        || data.starts_with(b"GIF87a")
        || data.starts_with(b"GIF89a")
        || (data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP".as_slice()))
        || data.starts_with(b"II*\x00")
        || data.starts_with(b"MM\x00*")
        || data.starts_with(b"BM")
}

struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(StrictVisitor)
    }
}

struct StrictVisitor;

impl<'de> Visitor<'de> for StrictVisitor {
    type Value = StrictValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value without duplicate fields")
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<StrictValue, E> {
        Ok(StrictValue(Value::Bool(value)))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<StrictValue, E> {
        Ok(StrictValue(Value::from(value)))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<StrictValue, E> {
        Ok(StrictValue(Value::from(value)))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<StrictValue, E> {
        serde_json::Number::from_f64(value)
            .map(|number| StrictValue(Value::Number(number)))
            .ok_or_else(|| E::custom("number is not finite"))
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<StrictValue, E> {
        Ok(StrictValue(Value::String(value.to_string())))
    }

    fn visit_unit<E: de::Error>(self) -> Result<StrictValue, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<StrictValue, A::Error> {
        let mut items = Vec::new();
        while let Some(StrictValue(item)) = seq.next_element()? {
            items.push(item);
        }
        Ok(StrictValue(Value::Array(items)))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<StrictValue, A::Error> {
        let mut fields = serde_json::Map::new();
        while let Some((key, StrictValue(value))) = map.next_entry::<String, StrictValue>()? {
            if fields.insert(key, value).is_some() {
                return Err(de::Error::custom("request contains duplicate JSON fields"));
            }
        }
        Ok(StrictValue(Value::Object(fields)))
    }
}

struct PathRequest {
    source: String,
    digest: String,
    text: bool,
}

/// The validated fields of a `path` request body, or `None` when the body is
/// not exactly the request object of the contract.
fn parse_request(data: &[u8]) -> Option<PathRequest> {
    if data.len() as u64 > MAX_REQUEST_BYTES {
        return None;
    }
    let StrictValue(Value::Object(request)) = serde_json::from_slice(data).ok()? else {
        return None;
    };
    let mut keys: Vec<&str> = request
        .keys()
        .map(String::as_str)
        .filter(|key| *key != "mode")
        .collect();
    keys.sort_unstable();
    if keys != ["contract", "source", "source_sha256", "stage"] {
        return None;
    }
    let text = match request.get("mode") {
        None => false,
        Some(Value::String(mode)) if mode == TEXT_MODE => true,
        Some(_) => return None,
    };
    if request.get("stage")?.as_str()? != "glm_ocr" {
        return None;
    }
    let source = request.get("source")?.as_str()?;
    if source.is_empty() || source.contains('\0') || !Path::new(source).is_absolute() {
        return None;
    }
    let digest = request.get("source_sha256")?.as_str()?;
    if !is_digest(digest) {
        return None;
    }
    let contract = json!({"pages": "one-based", "regions": "reading-order", "bbox": "x0,y0,x1,y1"});
    if *request.get("contract")? != contract {
        return None;
    }
    Some(PathRequest {
        source: source.to_string(),
        digest: digest.to_string(),
        text,
    })
}

/// The bytes of the regular file `path` when they fit the source limit,
/// hash to `digest` and carry a supported signature.
fn read_source(path: &str, digest: &str) -> Option<Vec<u8>> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path).ok()?;
    let metadata = file.metadata().ok()?;
    if !metadata.is_file() || metadata.len() > MAX_SOURCE_BYTES {
        return None;
    }
    let mut data = Vec::new();
    file.take(MAX_SOURCE_BYTES + 1)
        .read_to_end(&mut data)
        .ok()?;
    if data.len() as u64 > MAX_SOURCE_BYTES
        || sha256_hex(&data) != digest
        || !supported_source(&data)
    {
        return None;
    }
    Some(data)
}

struct Head {
    method: String,
    path: String,
    minor_version: u8,
    headers: Vec<(String, String)>,
}

impl Head {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

enum HeadError {
    Silent,
    Reject(u16),
}

type Reader = BufReader<Box<dyn Conn>>;

fn read_limited_line(reader: &mut Reader, too_long: u16) -> Result<String, HeadError> {
    let mut line = Vec::new();
    let read = reader
        .by_ref()
        .take(MAX_HEAD_LINE_BYTES + 1)
        .read_until(b'\n', &mut line)
        .map_err(|_| HeadError::Silent)?;
    if read == 0 {
        return Err(HeadError::Silent);
    }
    if line.len() as u64 > MAX_HEAD_LINE_BYTES {
        return Err(HeadError::Reject(too_long));
    }
    Ok(String::from_utf8_lossy(&line)
        .trim_end_matches(['\r', '\n'])
        .to_string())
}

fn read_head(reader: &mut Reader) -> Result<Head, HeadError> {
    let request_line = read_limited_line(reader, 414)?;
    let words: Vec<&str> = request_line.split_whitespace().collect();
    if words.is_empty() {
        return Err(HeadError::Silent);
    }
    let &[method, path, version] = words.as_slice() else {
        return Err(HeadError::Reject(400));
    };
    let Some(numbers) = version.strip_prefix("HTTP/") else {
        return Err(HeadError::Reject(400));
    };
    let Some((major, minor)) = numbers.split_once('.') else {
        return Err(HeadError::Reject(400));
    };
    let (Ok(major), Ok(minor)) = (major.parse::<u32>(), minor.parse::<u8>()) else {
        return Err(HeadError::Reject(400));
    };
    if major >= 2 {
        return Err(HeadError::Reject(505));
    }
    let mut headers = Vec::new();
    loop {
        let line = read_limited_line(reader, 431)?;
        if line.is_empty() {
            break;
        }
        if headers.len() >= MAX_HEADERS {
            return Err(HeadError::Reject(431));
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_string(), value.trim().to_string()));
        }
    }
    Ok(Head {
        method: method.to_string(),
        path: path.to_string(),
        minor_version: if major == 1 { minor } else { 0 },
        headers,
    })
}

struct Reply {
    status: u16,
    body: Vec<u8>,
    bearer_challenge: bool,
}

impl Reply {
    fn json(status: u16, payload: &Value) -> Reply {
        Reply {
            status,
            body: serde_json::to_vec(payload).unwrap_or_default(),
            bearer_challenge: false,
        }
    }

    fn rejected(status: u16) -> Reply {
        Reply::json(status, &json!({"error": "request rejected"}))
    }

    fn failure(failure: &OcrFailure) -> Reply {
        match failure {
            OcrFailure::Cancelled => Reply::json(409, &json!({"error": "OCR request cancelled"})),
            OcrFailure::Unavailable { .. } => {
                Reply::json(503, &json!({"error": "OCR service unavailable"}))
            }
            OcrFailure::Rejected => Reply::json(422, &json!({"error": "OCR request rejected"})),
            OcrFailure::Internal => Reply::json(500, &json!({"error": "OCR bridge unavailable"})),
        }
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        202 => "Accepted",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        408 => "Request Timeout",
        409 => "Conflict",
        411 => "Length Required",
        413 => "Payload Too Large",
        414 => "URI Too Long",
        415 => "Unsupported Media Type",
        417 => "Expectation Failed",
        422 => "Unprocessable Entity",
        431 => "Request Header Fields Too Large",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        505 => "HTTP Version Not Supported",
        _ => "Internal Server Error",
    }
}

fn http_date(now: SystemTime) -> String {
    const DAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let seconds = now
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs() as i64);
    let days = seconds.div_euclid(86_400);
    let time = seconds.rem_euclid(86_400);
    let weekday = (days + 4).rem_euclid(7) as usize;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!(
        "{}, {day:02} {} {year} {:02}:{:02}:{:02} GMT",
        DAYS[weekday],
        MONTHS[(month - 1) as usize],
        time / 3600,
        time % 3600 / 60,
        time % 60
    )
}

fn write_reply(stream: &mut dyn Write, reply: &Reply) -> std::io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nServer: {SERVER_NAME}\r\nDate: {}\r\n",
        reply.status,
        reason(reply.status),
        http_date(SystemTime::now())
    );
    if reply.bearer_challenge {
        head.push_str("WWW-Authenticate: Bearer\r\n");
    }
    head.push_str(&format!(
        "Content-Type: application/json\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        reply.body.len()
    ));
    let mut message = head.into_bytes();
    message.extend_from_slice(&reply.body);
    stream.write_all(&message)?;
    stream.flush()
}

fn unauthorized() -> Reply {
    Reply {
        bearer_challenge: true,
        ..Reply::rejected(401)
    }
}

fn get(shared: &Shared, head: &Head) -> Reply {
    if head.path == "/health" {
        return Reply::json(
            200,
            &json!({
                "status": "ready",
                "contracts": SUPPORTED_CONTRACTS,
                "max_requests": shared.max_requests,
            }),
        );
    }
    if !shared.authorized(head.header("Authorization")) {
        return unauthorized();
    }
    if let Some(digest) = head.path.strip_prefix("/progress/")
        && is_digest(digest)
        && let Some(progress) = shared.progress_of(digest)
    {
        let mut body = json!({
            "pages_done": progress.done,
            "pages_total": progress.total,
            "pages_reading": progress.reading,
            "pages_read": progress.read,
        });
        if let (Some((requests, workers)), Some(fields)) = (progress.load, body.as_object_mut()) {
            fields.insert("requests".to_string(), json!(requests));
            fields.insert("workers".to_string(), json!(workers));
        }
        return Reply::json(200, &body);
    }
    Reply::rejected(404)
}

fn read_body(reader: &mut Reader, length: u64) -> std::io::Result<Vec<u8>> {
    let mut data = Vec::new();
    let mut chunk = vec![0u8; BODY_CHUNK_BYTES.min(usize::try_from(length).unwrap_or(usize::MAX))];
    let mut remaining = length;
    while remaining > 0 {
        let want = chunk
            .len()
            .min(usize::try_from(remaining).unwrap_or(usize::MAX));
        let read = reader.read(&mut chunk[..want])?;
        if read == 0 {
            break;
        }
        data.extend_from_slice(&chunk[..read]);
        remaining -= read as u64;
    }
    Ok(data)
}

fn post(shared: &Shared, head: &Head, reader: &mut Reader) -> Reply {
    if !shared.authorized(head.header("Authorization")) {
        return unauthorized();
    }
    if let Some(digest) = head.path.strip_prefix("/cancel/") {
        return if is_digest(digest) && shared.cancel(digest) {
            Reply::json(202, &json!({"status": "cancelling"}))
        } else {
            Reply::rejected(404)
        };
    }
    if head.path != "/ocr" {
        return Reply::rejected(404);
    }
    if !shared.slots.try_acquire() {
        return Reply::json(503, &json!({"error": "OCR bridge busy"}));
    }
    let _slot = Slot(&shared.slots);
    post_ocr(shared, head, reader)
}

fn post_ocr(shared: &Shared, head: &Head, reader: &mut Reader) -> Reply {
    if head.header("Transfer-Encoding").is_some() {
        return Reply::rejected(400);
    }
    let content_type = head
        .header("Content-Type")
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim();
    let bytes_route = BODY_CONTENT_TYPES.contains(&content_type);
    if content_type != "application/json" && !bytes_route {
        return Reply::rejected(415);
    }
    let raw_length = head.header("Content-Length").unwrap_or("");
    if raw_length.is_empty() || !raw_length.bytes().all(|byte| byte.is_ascii_digit()) {
        return Reply::rejected(411);
    }
    let Ok(length) = raw_length.parse::<u64>() else {
        return Reply::rejected(413);
    };
    let limit = if bytes_route {
        MAX_SOURCE_BYTES
    } else {
        MAX_REQUEST_BYTES
    };
    if length > limit {
        return Reply::rejected(413);
    }
    let rejected = || Reply::json(422, &json!({"error": "OCR request rejected"}));
    let mut digest = String::new();
    if bytes_route {
        digest = head
            .header("Semantik-Source-SHA256")
            .unwrap_or("")
            .to_string();
        if !is_digest(&digest) || head.header("Semantik-Contract") != Some(BYTES_CONTRACT_HEADER) {
            return rejected();
        }
    }
    let data = match read_body(reader, length) {
        Ok(data) => data,
        Err(_) => return Reply::json(408, &json!({"error": "request timed out"})),
    };
    if data.len() as u64 != length {
        return Reply::rejected(400);
    }
    let outcome = if bytes_route {
        if sha256_hex(&data) != digest || !supported_source(&data) {
            Err(OcrFailure::Rejected)
        } else {
            let task = if head.header(TEXT_MODE_HEADER) == Some(TEXT_MODE) {
                OcrTask::Text
            } else {
                OcrTask::Layout
            };
            shared.convert(&data, &digest, task)
        }
    } else {
        shared.convert_request(&data)
    };
    match outcome {
        Ok(body) => Reply {
            status: 200,
            body,
            bearer_challenge: false,
        },
        Err(failure) => Reply::failure(&failure),
    }
}

fn handle_connection(shared: &Shared, stream: Box<dyn Conn>) {
    let mut reader: Reader = BufReader::new(stream);
    let head = match read_head(&mut reader) {
        Ok(head) => head,
        Err(HeadError::Silent) => return,
        Err(HeadError::Reject(status)) => {
            let _ = write_reply(reader.get_mut(), &Reply::rejected(status));
            return;
        }
    };
    if head.minor_version >= 1
        && let Some(expectation) = head.header("Expect")
    {
        if expectation.eq_ignore_ascii_case("100-continue") {
            let _ = reader.get_mut().write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
        } else {
            let _ = write_reply(reader.get_mut(), &Reply::rejected(417));
            return;
        }
    }
    let reply = match head.method.as_str() {
        "GET" => get(shared, &head),
        "POST" => post(shared, &head, &mut reader),
        _ => Reply::rejected(501),
    };
    let _ = write_reply(reader.get_mut(), &reply);
}

/// Stops a running [`OcrServer::serve`] loop.
#[derive(Clone)]
pub struct ShutdownHandle(Arc<AtomicBool>);

impl ShutdownHandle {
    /// Asks the serve loop to return once its in-flight requests finish.
    pub fn stop(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// A bound OCR bridge server.
pub struct OcrServer {
    listener: TcpListener,
    shared: Arc<Shared>,
    tls: Option<Arc<rustls::ServerConfig>>,
    stop: Arc<AtomicBool>,
}

impl OcrServer {
    /// Binds the address of `config`; a non-loopback address needs both a
    /// bearer token and TLS.
    pub fn bind(config: ServeConfig, ocr: Arc<dyn DocumentOcr>) -> Result<OcrServer> {
        if !config.address.is_loopback()
            && (config.token.as_deref().is_none_or(str::is_empty) || config.tls.is_none())
        {
            bail!("a non-loopback bind address requires a bearer token and TLS");
        }
        let listener = TcpListener::bind(SocketAddr::new(config.address, config.port))
            .map_err(|_| anyhow!("HTTP bridge failed to bind"))?;
        listener.set_nonblocking(true)?;
        Ok(OcrServer {
            listener,
            shared: Arc::new(Shared {
                ocr,
                token: config.token.map(String::into_bytes),
                slots: Slots(AtomicUsize::new(config.max_requests)),
                max_requests: config.max_requests,
                book: Mutex::new(Book::default()),
                diagnostics: Mutex::new(HashMap::new()),
            }),
            tls: config.tls,
            stop: Arc::new(AtomicBool::new(false)),
        })
    }

    /// The bound address.
    pub fn local_addr(&self) -> SocketAddr {
        self.listener
            .local_addr()
            .unwrap_or_else(|_| SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
    }

    /// A handle that stops [`OcrServer::serve`].
    pub fn shutdown_handle(&self) -> ShutdownHandle {
        ShutdownHandle(Arc::clone(&self.stop))
    }

    /// Accepts connections, one thread each, until stopped, then waits for
    /// the in-flight connections to finish.
    pub fn serve(&self) {
        let mut workers: Vec<std::thread::JoinHandle<()>> = Vec::new();
        while !self.stop.load(Ordering::SeqCst) {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    if stream.set_nonblocking(false).is_err() {
                        continue;
                    }
                    let timeout = Some(Duration::from_secs(HTTP_READ_TIMEOUT_SECS));
                    let _ = stream.set_read_timeout(timeout);
                    let _ = stream.set_write_timeout(timeout);
                    let shared = Arc::clone(&self.shared);
                    let tls = self.tls.clone();
                    workers.push(std::thread::spawn(move || {
                        let connection: Box<dyn Conn> = match tls {
                            Some(config) => match rustls::ServerConnection::new(config) {
                                Ok(session) => Box::new(rustls::StreamOwned::new(session, stream)),
                                Err(_) => return,
                            },
                            None => Box::new(stream),
                        };
                        handle_connection(&shared, connection);
                    }));
                    workers.retain(|worker| !worker.is_finished());
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(ACCEPT_POLL);
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => std::thread::sleep(ACCEPT_POLL),
            }
        }
        for worker in workers {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;
    use std::time::Instant;

    #[derive(Clone, Copy)]
    enum Mode {
        Reply,
        Fail(fn() -> OcrFailure),
        Panic,
        Hold,
    }

    struct Stub {
        mode: Mode,
        calls: Mutex<Vec<(OcrTask, Vec<u8>)>>,
        entered: AtomicBool,
        release: AtomicBool,
    }

    impl Stub {
        fn new(mode: Mode) -> Arc<Stub> {
            Arc::new(Stub {
                mode,
                calls: Mutex::new(Vec::new()),
                entered: AtomicBool::new(false),
                release: AtomicBool::new(false),
            })
        }

        fn call_count(&self) -> usize {
            lock(&self.calls).len()
        }
    }

    impl DocumentOcr for Stub {
        #[allow(clippy::too_many_arguments)]
        fn ocr_with_load(
            &self,
            source: &[u8],
            task: OcrTask,
            progress: &(dyn Fn(usize, usize) + Sync),
            pages: &(dyn Fn(&[i64], &[i64]) + Sync),
            load: &(dyn Fn(usize, usize) + Sync),
            cancel: &(dyn Fn() -> bool + Sync),
            diagnostic: &(dyn Fn(&str, u64) + Sync),
        ) -> Result<Value, OcrFailure> {
            if matches!(self.mode, Mode::Hold) {
                load(4, 32);
            }
            self.ocr_with_pages(source, task, progress, pages, cancel, diagnostic)
        }

        fn ocr_with_pages(
            &self,
            source: &[u8],
            task: OcrTask,
            progress: &(dyn Fn(usize, usize) + Sync),
            pages: &(dyn Fn(&[i64], &[i64]) + Sync),
            cancel: &(dyn Fn() -> bool + Sync),
            diagnostic: &(dyn Fn(&str, u64) + Sync),
        ) -> Result<Value, OcrFailure> {
            if matches!(self.mode, Mode::Hold) {
                pages(&[5, 7], &[1, 2, 3]);
            }
            self.ocr(source, task, progress, cancel, diagnostic)
        }

        fn ocr(
            &self,
            source: &[u8],
            task: OcrTask,
            progress: &(dyn Fn(usize, usize) + Sync),
            cancel: &(dyn Fn() -> bool + Sync),
            _diagnostic: &(dyn Fn(&str, u64) + Sync),
        ) -> Result<Value, OcrFailure> {
            lock(&self.calls).push((task, source.to_vec()));
            self.entered.store(true, Ordering::SeqCst);
            let reply = json!({"title": "Synthetic report", "regions": []});
            match self.mode {
                Mode::Reply => Ok(reply),
                Mode::Fail(make) => Err(make()),
                Mode::Panic => panic!("private-source-name and provider body"),
                Mode::Hold => {
                    progress(3, 9);
                    loop {
                        if cancel() {
                            return Err(OcrFailure::Cancelled);
                        }
                        if self.release.load(Ordering::SeqCst) {
                            return Ok(reply);
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                }
            }
        }
    }

    struct Running {
        port: u16,
        stop: ShutdownHandle,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl Drop for Running {
        fn drop(&mut self) {
            self.stop.stop();
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    fn start(ocr: Arc<dyn DocumentOcr>, token: Option<&str>, max_requests: usize) -> Running {
        let config = ServeConfig {
            address: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: 0,
            tls: None,
            token: token.map(str::to_string),
            max_requests,
        };
        let server = OcrServer::bind(config, ocr).unwrap();
        let port = server.local_addr().port();
        let stop = server.shutdown_handle();
        let thread = std::thread::spawn(move || server.serve());
        Running {
            port,
            stop,
            thread: Some(thread),
        }
    }

    struct Response {
        status: u16,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    }

    impl Response {
        fn json(&self) -> Value {
            serde_json::from_slice(&self.body).unwrap()
        }

        fn header(&self, name: &str) -> Option<&str> {
            self.headers
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case(name))
                .map(|(_, value)| value.as_str())
        }
    }

    fn exchange(port: u16, raw: &[u8]) -> Response {
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(20)))
            .unwrap();
        stream.write_all(raw).unwrap();
        stream.shutdown(std::net::Shutdown::Write).unwrap();
        let mut bytes = Vec::new();
        let _ = stream.read_to_end(&mut bytes);
        let split = bytes
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("response head");
        let head = String::from_utf8(bytes[..split].to_vec()).unwrap();
        let mut lines = head.lines();
        let status = lines
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let headers = lines
            .filter_map(|line| line.split_once(':'))
            .map(|(name, value)| (name.trim().to_string(), value.trim().to_string()))
            .collect();
        Response {
            status,
            headers,
            body: bytes[split + 4..].to_vec(),
        }
    }

    fn call(
        port: u16,
        method: &str,
        path: &str,
        headers: &[(&str, String)],
        body: &[u8],
    ) -> Response {
        let mut raw = format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\n");
        for (name, value) in headers {
            raw.push_str(&format!("{name}: {value}\r\n"));
        }
        raw.push_str("\r\n");
        let mut bytes = raw.into_bytes();
        bytes.extend_from_slice(body);
        exchange(port, &bytes)
    }

    fn get_path(port: u16, path: &str) -> Response {
        call(port, "GET", path, &[], b"")
    }

    fn post_bytes(port: u16, body: &[u8], overrides: &[(&str, Option<String>)]) -> Response {
        let mut headers = vec![
            ("Content-Type", Some("application/pdf".to_string())),
            ("Content-Length", Some(body.len().to_string())),
            ("Semantik-Source-SHA256", Some(sha256_hex(body))),
            ("Semantik-Contract", Some(BYTES_CONTRACT_HEADER.to_string())),
        ];
        for (name, value) in overrides {
            headers.retain(|(existing, _)| existing != name);
            headers.push((*name, value.clone()));
        }
        let headers: Vec<(&str, String)> = headers
            .into_iter()
            .filter_map(|(name, value)| value.map(|value| (name, value)))
            .collect();
        call(port, "POST", "/ocr", &headers, body)
    }

    fn pdf() -> Vec<u8> {
        b"%PDF-1.4\nsynthetic\n%%EOF\n".to_vec()
    }

    fn wait_until(condition: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while !condition() {
            assert!(Instant::now() < deadline, "condition never held");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "semantik-serve-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn health_lists_the_contracts_and_needs_no_token() {
        let server = start(Stub::new(Mode::Reply), Some("synthetic-token"), 1);
        let response = get_path(server.port, "/health");
        assert_eq!(response.status, 200);
        assert_eq!(
            response.json(),
            json!({"status": "ready", "contracts": ["path", "bytes"], "max_requests": 1})
        );
        assert_eq!(response.header("Cache-Control"), Some("no-store"));
        assert_eq!(response.header("Connection"), Some("close"));
        assert_eq!(response.header("Content-Type"), Some("application/json"));
    }

    #[test]
    fn bytes_request_with_matching_hash_and_contract_converts() {
        let stub = Stub::new(Mode::Reply);
        let server = start(stub.clone(), None, 1);
        let body = pdf();
        let response = post_bytes(server.port, &body, &[]);
        assert_eq!(response.status, 200);
        assert_eq!(response.json()["title"], "Synthetic report");
        assert_eq!(*lock(&stub.calls), vec![(OcrTask::Layout, body.clone())]);
        let text = post_bytes(
            server.port,
            &body,
            &[(TEXT_MODE_HEADER, Some(TEXT_MODE.to_string()))],
        );
        assert_eq!(text.status, 200);
        assert_eq!(lock(&stub.calls)[1].0, OcrTask::Text);
    }

    #[test]
    fn bytes_request_refuses_wrong_hash_and_missing_contract() {
        let stub = Stub::new(Mode::Reply);
        let server = start(stub.clone(), None, 1);
        let body = pdf();
        let wrong = post_bytes(
            server.port,
            &body,
            &[("Semantik-Source-SHA256", Some("0".repeat(64)))],
        );
        let missing = post_bytes(server.port, &body, &[("Semantik-Contract", None)]);
        let malformed = post_bytes(
            server.port,
            &body,
            &[("Semantik-Source-SHA256", Some("A".repeat(64)))],
        );
        for response in [&wrong, &missing, &malformed] {
            assert_eq!(response.status, 422);
            assert_eq!(response.json(), json!({"error": "OCR request rejected"}));
        }
        let unsupported = b"not a document".to_vec();
        assert_eq!(post_bytes(server.port, &unsupported, &[]).status, 422);
        assert_eq!(stub.call_count(), 0);
    }

    #[test]
    fn bytes_request_refuses_oversize_length_before_reading_the_body() {
        let server = start(Stub::new(Mode::Reply), None, 1);
        let response = call(
            server.port,
            "POST",
            "/ocr",
            &[
                ("Content-Type", "application/pdf".to_string()),
                ("Content-Length", (MAX_SOURCE_BYTES + 1).to_string()),
                ("Semantik-Source-SHA256", "a".repeat(64)),
                ("Semantik-Contract", BYTES_CONTRACT_HEADER.to_string()),
            ],
            b"",
        );
        assert_eq!(response.status, 413);
    }

    #[test]
    fn unbounded_content_length_is_refused_without_a_traceback() {
        let server = start(Stub::new(Mode::Reply), None, 1);
        let response = call(
            server.port,
            "POST",
            "/ocr",
            &[
                ("Content-Type", "application/json".to_string()),
                ("Content-Length", "9".repeat(5_000)),
            ],
            b"",
        );
        assert_eq!(response.status, 413);
        assert_eq!(response.json(), json!({"error": "request rejected"}));
    }

    #[test]
    fn malformed_post_requests_get_the_status_of_their_defect() {
        let server = start(Stub::new(Mode::Reply), None, 1);
        let json_type = ("Content-Type", "application/json".to_string());
        let unsupported = call(
            server.port,
            "POST",
            "/ocr",
            &[("Content-Type", "text/plain".to_string())],
            b"",
        );
        assert_eq!(unsupported.status, 415);
        let no_length = call(
            server.port,
            "POST",
            "/ocr",
            std::slice::from_ref(&json_type),
            b"",
        );
        assert_eq!(no_length.status, 411);
        let bad_length = call(
            server.port,
            "POST",
            "/ocr",
            &[json_type.clone(), ("Content-Length", "-1".to_string())],
            b"",
        );
        assert_eq!(bad_length.status, 411);
        let chunked = call(
            server.port,
            "POST",
            "/ocr",
            &[
                json_type.clone(),
                ("Transfer-Encoding", "chunked".to_string()),
            ],
            b"",
        );
        assert_eq!(chunked.status, 400);
        let short = call(
            server.port,
            "POST",
            "/ocr",
            &[json_type, ("Content-Length", "10".to_string())],
            b"{}",
        );
        assert_eq!(short.status, 400);
        assert_eq!(
            call(server.port, "POST", "/elsewhere", &[], b"").status,
            404
        );
        assert_eq!(get_path(server.port, "/elsewhere").status, 404);
        assert_eq!(call(server.port, "PUT", "/ocr", &[], b"").status, 501);
        assert_eq!(exchange(server.port, b"GARBAGE\r\n\r\n").status, 400);
        assert_eq!(
            exchange(server.port, b"GET /health HTTP/2.0\r\n\r\n").status,
            505
        );
    }

    #[test]
    fn path_request_reads_the_source_and_converts() {
        let stub = Stub::new(Mode::Reply);
        let server = start(stub.clone(), None, 1);
        let dir = scratch("path");
        let source = dir.join("synthetic.pdf");
        let body = pdf();
        std::fs::write(&source, &body).unwrap();
        let request = |source: &str, digest: &str, extra: Value| {
            let mut object = json!({
                "stage": "glm_ocr",
                "source": source,
                "source_sha256": digest,
                "contract": {"pages": "one-based", "regions": "reading-order", "bbox": "x0,y0,x1,y1"},
            });
            if let (Some(map), Value::Object(extra)) = (object.as_object_mut(), extra) {
                map.extend(extra);
            }
            let payload = serde_json::to_vec(&object).unwrap();
            call(
                server.port,
                "POST",
                "/ocr",
                &[
                    ("Content-Type", "application/json".to_string()),
                    ("Content-Length", payload.len().to_string()),
                ],
                &payload,
            )
        };
        let path = source.to_str().unwrap();
        let digest = sha256_hex(&body);
        let converted = request(path, &digest, json!({}));
        assert_eq!(converted.status, 200);
        assert_eq!(*lock(&stub.calls), vec![(OcrTask::Layout, body.clone())]);
        let text = request(path, &digest, json!({"mode": "text"}));
        assert_eq!(text.status, 200);
        assert_eq!(lock(&stub.calls)[1].0, OcrTask::Text);
        for refused in [
            request(path, &"0".repeat(64), json!({})),
            request("relative.pdf", &digest, json!({})),
            request(
                dir.join("missing.pdf").to_str().unwrap(),
                &digest,
                json!({}),
            ),
            request(path, &digest, json!({"mode": "layout"})),
            request(path, &digest, json!({"extra": 1})),
            request(path, "ABC", json!({})),
        ] {
            assert_eq!(refused.status, 422);
            assert_eq!(refused.json(), json!({"error": "OCR request rejected"}));
        }
        let duplicate = br#"{"stage":"glm_ocr","stage":"glm_ocr"}"#;
        let response = call(
            server.port,
            "POST",
            "/ocr",
            &[
                ("Content-Type", "application/json".to_string()),
                ("Content-Length", duplicate.len().to_string()),
            ],
            duplicate,
        );
        assert_eq!(response.status, 422);
        assert_eq!(stub.call_count(), 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn progress_reports_a_running_source_only_and_cancel_stops_it() {
        let stub = Stub::new(Mode::Hold);
        let server = start(stub.clone(), None, 2);
        let body = pdf();
        let digest = sha256_hex(&body);
        let port = server.port;
        assert_eq!(get_path(port, &format!("/progress/{digest}")).status, 404);
        assert_eq!(get_path(port, "/progress/not-a-digest").status, 404);
        assert_eq!(
            call(port, "POST", &format!("/cancel/{digest}"), &[], b"").status,
            404
        );
        let running = {
            let body = body.clone();
            std::thread::spawn(move || post_bytes(port, &body, &[]))
        };
        wait_until(|| get_path(port, &format!("/progress/{digest}")).status == 200);
        let progress = get_path(port, &format!("/progress/{digest}"));
        assert_eq!(
            progress.json(),
            json!({
                "pages_done": 3,
                "pages_total": 9,
                "pages_reading": [5, 7],
                "pages_read": [1, 2, 3],
                "requests": 4,
                "workers": 32,
            })
        );
        assert_eq!(
            call(
                port,
                "POST",
                &format!("/cancel/{}", "b".repeat(64)),
                &[],
                b""
            )
            .status,
            404
        );
        let cancelled = call(port, "POST", &format!("/cancel/{digest}"), &[], b"");
        assert_eq!(cancelled.status, 202);
        assert_eq!(cancelled.json(), json!({"status": "cancelling"}));
        let finished = running.join().unwrap();
        assert_eq!(finished.status, 409);
        assert_eq!(finished.json(), json!({"error": "OCR request cancelled"}));
        assert_eq!(get_path(port, &format!("/progress/{digest}")).status, 404);
        assert_eq!(
            call(port, "POST", &format!("/cancel/{digest}"), &[], b"").status,
            404
        );
    }

    #[test]
    fn a_second_document_is_refused_as_busy_while_one_converts() {
        let stub = Stub::new(Mode::Hold);
        let server = start(stub.clone(), None, 1);
        let port = server.port;
        let body = pdf();
        let first = {
            let body = body.clone();
            std::thread::spawn(move || post_bytes(port, &body, &[]))
        };
        wait_until(|| stub.entered.load(Ordering::SeqCst));
        let busy = post_bytes(port, &body, &[]);
        assert_eq!(busy.status, 503);
        assert_eq!(busy.json(), json!({"error": "OCR bridge busy"}));
        stub.release.store(true, Ordering::SeqCst);
        assert_eq!(first.join().unwrap().status, 200);
        assert_eq!(post_bytes(port, &body, &[]).status, 200);
    }

    #[test]
    fn conversion_failures_map_to_fixed_statuses_without_reflecting_details() {
        let cases: [(Mode, u16, &str); 4] = [
            (
                Mode::Fail(|| OcrFailure::Unavailable {
                    diagnostics: vec![("layout_failure".into(), 2)],
                }),
                503,
                "OCR service unavailable",
            ),
            (
                Mode::Fail(|| OcrFailure::Rejected),
                422,
                "OCR request rejected",
            ),
            (
                Mode::Fail(|| OcrFailure::Internal),
                500,
                "OCR bridge unavailable",
            ),
            (Mode::Panic, 500, "OCR bridge unavailable"),
        ];
        for (mode, status, message) in cases {
            let server = start(Stub::new(mode), None, 1);
            let response = post_bytes(server.port, &pdf(), &[]);
            assert_eq!(response.status, status);
            assert_eq!(response.json(), json!({"error": message}));
            assert!(!String::from_utf8_lossy(&response.body).contains("private-source-name"));
        }
    }

    #[test]
    fn bearer_token_is_required_and_never_echoed() {
        let token = "synthetic-bearer-token";
        let stub = Stub::new(Mode::Reply);
        let server = start(stub.clone(), Some(token), 1);
        let port = server.port;
        let body = pdf();
        let denied = post_bytes(port, &body, &[]);
        let wrong = post_bytes(
            port,
            &body,
            &[("Authorization", Some("Bearer other-token".into()))],
        );
        let allowed = post_bytes(
            port,
            &body,
            &[("Authorization", Some(format!("Bearer {token}")))],
        );
        let lowercase = post_bytes(
            port,
            &body,
            &[("Authorization", Some(format!("bearer {token}")))],
        );
        let cancel = call(
            port,
            "POST",
            &format!("/cancel/{}", "0".repeat(64)),
            &[],
            b"",
        );
        let progress = get_path(port, &format!("/progress/{}", "0".repeat(64)));
        assert_eq!(denied.status, 401);
        assert_eq!(wrong.status, 401);
        assert_eq!(cancel.status, 401);
        assert_eq!(progress.status, 401);
        assert_eq!(denied.header("WWW-Authenticate"), Some("Bearer"));
        assert_eq!(allowed.status, 200);
        assert_eq!(lowercase.status, 200);
        assert_eq!(stub.call_count(), 2);
        for response in [&denied, &wrong, &allowed] {
            assert!(!String::from_utf8_lossy(&response.body).contains(token));
        }
        let authorised_cancel = call(
            port,
            "POST",
            &format!("/cancel/{}", "0".repeat(64)),
            &[("Authorization", format!("Bearer {token}"))],
            b"",
        );
        assert_eq!(authorised_cancel.status, 404);
    }

    #[test]
    fn non_loopback_bind_requires_token_and_tls() {
        let open = ServeConfig {
            address: "0.0.0.0".parse().unwrap(),
            port: 0,
            tls: None,
            token: Some("synthetic-token".into()),
            max_requests: 1,
        };
        assert!(OcrServer::bind(open, Stub::new(Mode::Reply)).is_err());
        let options =
            |listen: Option<&str>, cert: Option<&str>, key: Option<&str>, token: Option<&str>| {
                ServeConfig::from_options(
                    None,
                    listen,
                    cert.map(Path::new),
                    key.map(Path::new),
                    token,
                    &no_env,
                )
                .err()
                .map(|error| error.to_string())
            };
        assert_eq!(
            options(Some("0.0.0.0"), None, None, None).unwrap(),
            "a non-loopback bind address requires --tls-cert and --tls-key"
        );
        assert!(
            options(
                Some("0.0.0.0"),
                Some("synthetic.pem"),
                Some("synthetic.key"),
                None
            )
            .unwrap()
            .contains(DEFAULT_TOKEN_ENV)
        );
        assert_eq!(
            options(Some("localhost"), None, None, None).unwrap(),
            "HTTP bridge bind address is invalid"
        );
        assert_eq!(
            options(None, Some("synthetic.pem"), None, None).unwrap(),
            "--tls-cert and --tls-key are used together"
        );
        assert_eq!(
            options(None, Some("synthetic.pem"), Some("synthetic.key"), None).unwrap(),
            "TLS certificate or key could not be loaded"
        );
        assert_eq!(
            options(None, None, None, Some("MISSING_TOKEN")).unwrap(),
            "bearer token environment variable MISSING_TOKEN is not set"
        );
    }

    #[test]
    fn served_max_num_seqs_sets_concurrency_unless_explicit() {
        let served = |name: &str| (name == MAX_NUM_SEQS_ENV).then(|| "48".to_string());
        assert_eq!(http_concurrency(&served).unwrap(), 48);
        let explicit = |name: &str| match name {
            MAX_NUM_SEQS_ENV => Some("48".to_string()),
            HTTP_CONCURRENCY_ENV => Some("3".to_string()),
            _ => None,
        };
        assert_eq!(http_concurrency(&explicit).unwrap(), 3);
        let bad = |name: &str| (name == MAX_NUM_SEQS_ENV).then(|| "0".to_string());
        assert!(http_concurrency(&bad).is_err());
    }

    #[test]
    fn loopback_defaults_and_environment_settings() {
        let config = ServeConfig::from_options(None, None, None, None, None, &no_env).unwrap();
        assert!(config.address.is_loopback());
        assert_eq!(config.port, DEFAULT_PORT);
        assert_eq!(config.max_requests, DEFAULT_HTTP_CONCURRENCY);
        assert!(config.token.is_none() && config.tls.is_none());
        let env = |name: &str| match name {
            "BRIDGE_TOKEN" => Some("synthetic".to_string()),
            HTTP_CONCURRENCY_ENV => Some("4".to_string()),
            _ => None,
        };
        let config =
            ServeConfig::from_options(Some(18002), None, None, None, Some("BRIDGE_TOKEN"), &env)
                .unwrap();
        assert_eq!(config.port, 18002);
        assert_eq!(config.token.as_deref(), Some("synthetic"));
        assert_eq!(config.max_requests, 4);
        for bad in ["0", "33", "many"] {
            let env = |name: &str| (name == HTTP_CONCURRENCY_ENV).then(|| bad.to_string());
            let error = ServeConfig::from_options(None, None, None, None, None, &env)
                .err()
                .unwrap();
            assert_eq!(error.to_string(), "HTTP bridge concurrency is invalid");
        }
    }

    #[test]
    fn http_dates_follow_the_rfc_format() {
        assert_eq!(http_date(UNIX_EPOCH), "Thu, 01 Jan 1970 00:00:00 GMT");
        assert_eq!(
            http_date(UNIX_EPOCH + Duration::from_secs(1_700_000_000)),
            "Tue, 14 Nov 2023 22:13:20 GMT"
        );
    }
}
