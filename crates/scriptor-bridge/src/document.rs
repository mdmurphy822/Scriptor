//! The native OCR conversion behind the bridge server: settings
//! read from the environment, and a
//! [`DocumentOcr`] that renders, lays out, recognises and assembles one
//! document.

use crate::assemble::{AssembleError, Assembled};
use crate::client::{ClientConfig, OcrClient, RegionError, RegionRetryPolicy};
use crate::crop::region_data_url;
use crate::prompt::{Task, request_body};
use crate::recover::pdf_metadata_title;
use crate::recovery_finisher::RecoveryFinisher;
use crate::render::PageSource;
use crate::schedule::{ScheduleConfig, ScheduleError, run_document_observed};
use crate::serve::{
    DEFAULT_TOKEN_ENV, DocumentOcr, HTTP_CONCURRENCY_ENV, MAX_NUM_SEQS_ENV, OcrFailure, OcrTask,
};
use reqwest::Url;
use scriptor_layout::LayoutError;
pub use scriptor_layout::LayoutModel;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fmt;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// A layout model shared by every conversion of a server.
pub type SharedLayout = Arc<dyn LayoutModel + Send + Sync>;

/// Environment variable naming the recognition engine's chat-completions URL.
pub const URL_ENV: &str = "SCRIPTOR_BRIDGE_ENGINE_URL";
/// Environment variable naming the recognition model.
pub const MODEL_ENV: &str = "SCRIPTOR_BRIDGE_MODEL";
/// Environment variable holding the engine's bearer token.
pub const API_KEY_ENV: &str = "SCRIPTOR_BRIDGE_API_KEY";
/// Environment variable naming the layout compute device.
pub const LAYOUT_DEVICE_ENV: &str = "SCRIPTOR_BRIDGE_LAYOUT_DEVICE";
/// Environment variable naming the page checkpoint directory.
pub const CHECKPOINT_DIR_ENV: &str = "SCRIPTOR_BRIDGE_CHECKPOINT_DIR";
/// Environment variable naming the page render resolution.
pub const FIGURE_DPI_ENV: &str = "SCRIPTOR_BRIDGE_FIGURE_DPI";
/// Environment variable naming the recognition worker concurrency; unset,
/// it follows the engine's `max_num_seqs` when that is set.
pub const WORKERS_ENV: &str = "SCRIPTOR_BRIDGE_WORKERS";
/// Environment variable naming the layout batch size.
pub const LAYOUT_BATCH_SIZE_ENV: &str = "SCRIPTOR_BRIDGE_LAYOUT_BATCH_SIZE";
/// Environment variable naming the connect timeout in seconds.
pub const CONNECT_TIMEOUT_ENV: &str = "SCRIPTOR_BRIDGE_CONNECT_TIMEOUT_SECS";
/// Environment variable naming the request timeout in seconds.
pub const REQUEST_TIMEOUT_ENV: &str = "SCRIPTOR_BRIDGE_REQUEST_TIMEOUT_SECS";
/// Environment variable naming the layout runtime library.
pub const LAYOUT_RUNTIME_ENV: &str = "SCRIPTOR_BRIDGE_LAYOUT_RUNTIME";
/// Environment variable naming the layout weights file.
pub const LAYOUT_WEIGHTS_ENV: &str = "SCRIPTOR_BRIDGE_LAYOUT_WEIGHTS";

/// Every bridge setting as `(SCRIPTOR_ name, legacy SEMANTIK_ name)`. The
/// `SCRIPTOR_` name wins; the legacy name is read when it is unset.
pub const SETTING_NAMES: [(&str, &str); 15] = [
    (URL_ENV, "SEMANTIK_GLM_OCR_URL"),
    (MODEL_ENV, "SEMANTIK_GLM_OCR_MODEL"),
    (API_KEY_ENV, "SEMANTIK_GLM_OCR_API_KEY"),
    (LAYOUT_DEVICE_ENV, "SEMANTIK_GLM_OCR_LAYOUT_DEVICE"),
    (CHECKPOINT_DIR_ENV, "SEMANTIK_GLM_OCR_CHECKPOINT_DIR"),
    (FIGURE_DPI_ENV, "SEMANTIK_FIGURE_DPI"),
    (WORKERS_ENV, "SEMANTIK_GLM_OCR_WORKERS"),
    (LAYOUT_BATCH_SIZE_ENV, "SEMANTIK_GLM_OCR_LAYOUT_BATCH_SIZE"),
    (CONNECT_TIMEOUT_ENV, "SEMANTIK_GLM_OCR_CONNECT_TIMEOUT_SECS"),
    (REQUEST_TIMEOUT_ENV, "SEMANTIK_GLM_OCR_REQUEST_TIMEOUT_SECS"),
    (LAYOUT_RUNTIME_ENV, "SEMANTIK_LAYOUT_RUNTIME"),
    (LAYOUT_WEIGHTS_ENV, "SEMANTIK_LAYOUT_WEIGHTS"),
    (DEFAULT_TOKEN_ENV, "SEMANTIK_BRIDGE_TOKEN"),
    (HTTP_CONCURRENCY_ENV, "SEMANTIK_GLM_OCR_HTTP_CONCURRENCY"),
    (MAX_NUM_SEQS_ENV, "SEMANTIK_GLM_OCR_MAX_NUM_SEQS"),
];

/// The legacy name of a `SCRIPTOR_` setting, or `None` for any other name.
fn legacy_name(name: &str) -> Option<&'static str> {
    SETTING_NAMES
        .iter()
        .find(|(current, _)| *current == name)
        .map(|(_, legacy)| *legacy)
}

/// An environment view over `env` in which each `SCRIPTOR_` setting that is
/// unset reads as its legacy `SEMANTIK_` name. A set `SCRIPTOR_` value wins
/// even when the legacy name is also set; any other name passes through.
pub fn with_legacy_fallback<'a>(
    env: &'a dyn Fn(&str) -> Option<String>,
) -> impl Fn(&str) -> Option<String> + 'a {
    move |name: &str| env(name).or_else(|| legacy_name(name).and_then(env))
}

/// The default figure resolution in dots per inch.
const DEFAULT_FIGURE_DPI: i64 = 300;
/// The highest accepted figure resolution in dots per inch.
const MAX_FIGURE_DPI: i64 = 600;
/// The requests held per running sequence of the OCR server.
const SERVED_QUEUE_FACTOR: i64 = 2;
/// The default recognition worker count.
const DEFAULT_WORKERS: i64 = 32;
/// The highest accepted recognition worker count.
const MAX_WORKERS: i64 = 128;
/// The default layout batch size.
const DEFAULT_LAYOUT_BATCH_SIZE: i64 = 4;
/// The highest accepted layout batch size.
const MAX_LAYOUT_BATCH_SIZE: i64 = 64;
/// The default connect timeout in seconds.
const DEFAULT_CONNECT_TIMEOUT_SECS: i64 = 30;
/// The highest accepted connect timeout in seconds.
const MAX_CONNECT_TIMEOUT_SECS: i64 = 300;
/// The highest accepted request timeout in seconds.
const MAX_REQUEST_TIMEOUT_SECS: i64 = 1_200;
/// The longest accepted model name, in characters.
const MAX_MODEL_CHARS: usize = 512;

/// A setting that is missing or invalid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError {
    /// The environment variable at fault.
    pub setting: &'static str,
    /// What is wrong with it.
    pub message: &'static str,
}

impl ConfigError {
    fn new(setting: &'static str, message: &'static str) -> Self {
        Self { setting, message }
    }

    /// The `(category, count)` diagnostic naming the setting.
    #[must_use]
    pub fn diagnostic(&self) -> (String, u64) {
        (format!("setting_{}", self.setting.to_ascii_lowercase()), 1)
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.message, self.setting)
    }
}

impl std::error::Error for ConfigError {}

/// The settings of the native conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentConfig {
    /// The seat's chat-completions URL: a loopback `http` or `https` URL.
    pub endpoint: String,
    /// The `model` field of every request.
    pub model: String,
    /// The bearer token sent to the seat.
    pub api_key: Option<String>,
    /// The layout compute device name, when one is requested.
    pub layout_device: Option<String>,
    /// The render resolution of PDF pages.
    pub figure_dpi: u32,
    /// The recognition worker concurrency.
    pub workers: usize,
    /// The most pages the layout model sees in one call.
    pub layout_batch_size: usize,
    pub connect_timeout: Duration,
    /// The opted-in total request deadline; `None` imposes no deadline.
    pub request_timeout: Option<Duration>,
    /// An absolute directory holding per-document page checkpoints.
    pub checkpoint_dir: Option<PathBuf>,
    /// The layout runtime library path.
    pub layout_runtime: Option<PathBuf>,
    /// The layout weights file path.
    pub layout_weights: Option<PathBuf>,
}

fn bounded(
    env: &dyn Fn(&str) -> Option<String>,
    name: &'static str,
    default: i64,
    maximum: i64,
    message: &'static str,
) -> Result<i64, ConfigError> {
    let value = match env(name) {
        None => default,
        Some(text) => text
            .trim()
            .parse::<i64>()
            .map_err(|_| ConfigError::new(name, message))?,
    };
    if (1..=maximum).contains(&value) {
        Ok(value)
    } else {
        Err(ConfigError::new(name, message))
    }
}

/// The worker count for a server that runs `served` sequences at once:
/// twice as many, so that requests wait queued at the server beyond its
/// running slots and a finished sequence is replaced at the next step.
fn served_workers(served: &str) -> Result<i64, ConfigError> {
    match served.trim().parse::<i64>() {
        Ok(count) if count >= 1 => Ok(count.saturating_mul(SERVED_QUEUE_FACTOR)),
        _ => Err(ConfigError::new(
            MAX_NUM_SEQS_ENV,
            "OCR server max_num_seqs is invalid",
        )),
    }
}

fn endpoint_is_valid(endpoint: &str) -> bool {
    if endpoint
        .chars()
        .any(|character| character.is_whitespace() || u32::from(character) < 32)
    {
        return false;
    }
    let Ok(url) = Url::parse(endpoint) else {
        return false;
    };
    let loopback = url.host_str().is_some_and(|host| {
        host == "localhost"
            || host
                .trim_matches(['[', ']'])
                .parse::<IpAddr>()
                .is_ok_and(|address| address.is_loopback())
    });
    let authority_only = endpoint
        .split_once("://")
        .is_some_and(|(_, rest)| !rest.contains('/'));
    matches!(url.scheme(), "http" | "https")
        && loopback
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
        && !authority_only
}

impl DocumentConfig {
    /// The settings `env` holds, with their names, defaults and bounds. The
    /// layout runtime and weights are optional here; see
    /// [`DocumentConfig::layout_files`].
    ///
    /// # Errors
    ///
    /// Returns the first missing or invalid setting.
    pub fn from_env(env: &dyn Fn(&str) -> Option<String>) -> Result<Self, ConfigError> {
        let endpoint = env(URL_ENV).unwrap_or_default().trim().to_owned();
        let model = env(MODEL_ENV).unwrap_or_default().trim().to_owned();
        if endpoint.is_empty() {
            return Err(ConfigError::new(
                URL_ENV,
                "self-hosted OCR endpoint is not configured",
            ));
        }
        if model.is_empty()
            || model.chars().count() > MAX_MODEL_CHARS
            || model.chars().any(|character| u32::from(character) < 32)
        {
            return Err(ConfigError::new(
                MODEL_ENV,
                "self-hosted OCR model is not configured",
            ));
        }
        if !endpoint_is_valid(&endpoint) {
            return Err(ConfigError::new(
                URL_ENV,
                "self-hosted OCR endpoint is invalid",
            ));
        }
        let figure_dpi = bounded(
            env,
            FIGURE_DPI_ENV,
            DEFAULT_FIGURE_DPI,
            MAX_FIGURE_DPI,
            "figure extraction DPI is invalid",
        )?;
        let workers = match (env(WORKERS_ENV), env(MAX_NUM_SEQS_ENV)) {
            (None, Some(text)) => served_workers(&text)?,
            _ => bounded(
                env,
                WORKERS_ENV,
                DEFAULT_WORKERS,
                MAX_WORKERS,
                "GLM-OCR worker concurrency is invalid",
            )?,
        };
        let layout_batch_size = bounded(
            env,
            LAYOUT_BATCH_SIZE_ENV,
            DEFAULT_LAYOUT_BATCH_SIZE,
            MAX_LAYOUT_BATCH_SIZE,
            "GLM-OCR layout batch size is invalid",
        )?;
        let connect = bounded(
            env,
            CONNECT_TIMEOUT_ENV,
            DEFAULT_CONNECT_TIMEOUT_SECS,
            MAX_CONNECT_TIMEOUT_SECS,
            "GLM-OCR connect timeout is invalid",
        )?;
        let request = match env(REQUEST_TIMEOUT_ENV) {
            None => None,
            Some(_) => Some(bounded(
                env,
                REQUEST_TIMEOUT_ENV,
                MAX_REQUEST_TIMEOUT_SECS,
                MAX_REQUEST_TIMEOUT_SECS,
                "GLM-OCR request timeout is invalid",
            )?),
        };
        let present = |name: &str| env(name).filter(|value| !value.is_empty());
        let checkpoint_dir = present(CHECKPOINT_DIR_ENV).map(PathBuf::from);
        if checkpoint_dir
            .as_deref()
            .is_some_and(|dir| !dir.is_absolute())
        {
            return Err(ConfigError::new(
                CHECKPOINT_DIR_ENV,
                "GLM-OCR checkpoint directory must be an absolute path",
            ));
        }
        let seconds = |value: i64| Duration::from_secs(u64::try_from(value).unwrap_or(0));
        Ok(Self {
            endpoint,
            model,
            api_key: present(API_KEY_ENV),
            layout_device: present(LAYOUT_DEVICE_ENV),
            figure_dpi: u32::try_from(figure_dpi).unwrap_or(300),
            workers: usize::try_from(workers).unwrap_or(1),
            layout_batch_size: usize::try_from(layout_batch_size).unwrap_or(1),
            connect_timeout: seconds(connect),
            request_timeout: request.map(seconds),
            checkpoint_dir,
            layout_runtime: present(LAYOUT_RUNTIME_ENV).map(PathBuf::from),
            layout_weights: present(LAYOUT_WEIGHTS_ENV).map(PathBuf::from),
        })
    }

    /// The layout runtime and weights paths, each of which must be set and
    /// name an existing file.
    ///
    /// # Errors
    ///
    /// Returns the setting that is unset or names no file.
    pub fn layout_files(&self) -> Result<(&Path, &Path), ConfigError> {
        fn file<'a>(
            path: &'a Option<PathBuf>,
            name: &'static str,
        ) -> Result<&'a Path, ConfigError> {
            match path {
                None => Err(ConfigError::new(name, "the layout setting is not set")),
                Some(path) if path.is_file() => Ok(path.as_path()),
                Some(_) => Err(ConfigError::new(name, "the layout setting names no file")),
            }
        }
        Ok((
            file(&self.layout_runtime, LAYOUT_RUNTIME_ENV)?,
            file(&self.layout_weights, LAYOUT_WEIGHTS_ENV)?,
        ))
    }
}

/// The conversion of whole documents through the layout model and the
/// recognition seat.
pub struct NativeDocumentOcr {
    config: DocumentConfig,
    layout: SharedLayout,
}

impl NativeDocumentOcr {
    /// A conversion with `config`, detecting layout through `layout` and
    /// finishing every recognised page with the text-layer, figure-text,
    /// blank-band and form-field recoveries.
    #[must_use]
    pub fn new(config: DocumentConfig, layout: SharedLayout) -> Self {
        Self { config, layout }
    }

    fn client(&self) -> Result<(OcrClient, Arc<RegionRetryPolicy>), OcrFailure> {
        let mut client = ClientConfig::new(self.config.endpoint.clone());
        client.api_key = self.config.api_key.clone();
        client.connect_timeout = self.config.connect_timeout;
        client.request_timeout = self.config.request_timeout;
        let policy = Arc::new(RegionRetryPolicy::default());
        let client =
            OcrClient::new(client, Arc::clone(&policy)).map_err(|_| OcrFailure::Internal)?;
        Ok((client, policy))
    }

    fn layout_reply(
        &self,
        source: &[u8],
        progress: &(dyn Fn(usize, usize) + Sync),
        reading: &(dyn Fn(&[i64], &[i64]) + Sync),
        load: &(dyn Fn(usize, usize) + Sync),
        cancel: &(dyn Fn() -> bool + Sync),
        diagnostic: &(dyn Fn(&str, u64) + Sync),
    ) -> Result<Value, OcrFailure> {
        let pages = PageSource::open_at(source, f64::from(self.config.figure_dpi))
            .map_err(|_| OcrFailure::Rejected)?;
        let (client, policy) = self.client()?;
        let mut schedule = ScheduleConfig::new(self.config.workers, self.config.layout_batch_size);
        schedule.model.clone_from(&self.config.model);
        schedule.metadata_title = pdf_metadata_title(source);
        let checkpoint = self.config.checkpoint_dir.as_ref().map(|base| {
            base.join(
                Sha256::digest(source)
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>(),
            )
        });
        schedule.checkpoint_dir.clone_from(&checkpoint);
        let finisher = RecoveryFinisher::new(
            &client,
            &self.config.model,
            Arc::new(source.to_vec()),
            &pages,
        );
        let outcome = run_document_observed(
            &pages,
            &*self.layout,
            &client,
            &finisher,
            &schedule,
            cancel,
            progress,
            reading,
            load,
        );
        match outcome {
            Ok(assembled) => {
                if let Some(directory) = checkpoint {
                    let _ = std::fs::remove_dir_all(directory);
                }
                if assembled.successful_blank_omitted > 0 {
                    diagnostic(
                        "successful_blank_omitted",
                        assembled.successful_blank_omitted as u64,
                    );
                }
                for (category, count) in &assembled.diagnostics {
                    diagnostic(category, *count as u64);
                }
                for (category, count) in policy.take_diagnostics() {
                    diagnostic(&category, count);
                }
                layout_json(assembled)
            }
            Err(error) => Err(failure_of(error, policy.take_diagnostics())),
        }
    }

    fn text_reply(
        &self,
        source: &[u8],
        progress: &(dyn Fn(usize, usize) + Sync),
        reading: &(dyn Fn(&[i64], &[i64]) + Sync),
        cancel: &(dyn Fn() -> bool + Sync),
        diagnostic: &(dyn Fn(&str, u64) + Sync),
    ) -> Result<Value, OcrFailure> {
        if source.starts_with(b"%PDF-") {
            return Err(OcrFailure::Rejected);
        }
        let page = PageSource::open(source)
            .and_then(|pages| pages.render_page(0))
            .map_err(|_| OcrFailure::Rejected)?;
        let url = region_data_url(&page.image).map_err(|_| OcrFailure::Rejected)?;
        let body = request_body(&self.config.model, Task::Text, &url);
        let (client, policy) = self.client()?;
        progress(0, 1);
        reading(&[1], &[]);
        let content = match client.recognize(&body, cancel) {
            Ok(content) => content,
            Err(error) => return Err(region_failure(&error, policy.take_diagnostics())),
        };
        progress(1, 1);
        reading(&[], &[1]);
        for (category, count) in policy.take_diagnostics() {
            diagnostic(&category, count);
        }
        Ok(text_json(&content))
    }
}

impl DocumentOcr for NativeDocumentOcr {
    fn ocr(
        &self,
        source: &[u8],
        task: OcrTask,
        progress: &(dyn Fn(usize, usize) + Sync),
        cancel: &(dyn Fn() -> bool + Sync),
        diagnostic: &(dyn Fn(&str, u64) + Sync),
    ) -> Result<Value, OcrFailure> {
        self.ocr_with_pages(source, task, progress, &|_, _| {}, cancel, diagnostic)
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
        self.ocr_with_load(
            source,
            task,
            progress,
            pages,
            &|_, _| {},
            cancel,
            diagnostic,
        )
    }

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
        match task {
            OcrTask::Layout => self.layout_reply(source, progress, pages, load, cancel, diagnostic),
            OcrTask::Text => self.text_reply(source, progress, pages, cancel, diagnostic),
        }
    }
}

fn layout_json(assembled: Assembled) -> Result<Value, OcrFailure> {
    let regions = serde_json::to_value(&assembled.regions).map_err(|_| OcrFailure::Internal)?;
    Ok(json!({
        "title": assembled.title,
        "regions": regions,
        "empty_pages": assembled.empty_pages,
    }))
}

/// The reply for one image read in text mode: one paragraph region per
/// non-empty recognised line, with no geometry.
fn text_json(content: &str) -> Value {
    let lines = content
        .split(|character| {
            matches!(
                character,
                '\n' | '\r'
                    | '\u{0b}'
                    | '\u{0c}'
                    | '\u{1c}'
                    | '\u{1d}'
                    | '\u{1e}'
                    | '\u{85}'
                    | '\u{2028}'
                    | '\u{2029}'
            )
        })
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let regions: Vec<Value> = lines
        .enumerate()
        .map(|(index, line)| {
            json!({
                "id": format!("p0001-r{:06}", index + 1),
                "page": 1,
                "kind": "paragraph",
                "text": line,
            })
        })
        .collect();
    json!({"title": "", "regions": regions, "empty_pages": []})
}

fn unavailable(mut diagnostics: Vec<(String, u64)>, fallback: &str) -> OcrFailure {
    if diagnostics.is_empty() {
        diagnostics.push((fallback.to_owned(), 1));
    }
    OcrFailure::Unavailable { diagnostics }
}

/// The failure of a text-mode region request, with the seat's `seat`
/// diagnostics.
fn region_failure(error: &RegionError, seat: Vec<(String, u64)>) -> OcrFailure {
    match error {
        RegionError::Cancelled => OcrFailure::Cancelled,
        RegionError::Failed(failure) => unavailable(
            seat,
            &format!("region_final_{}", failure.class.diagnostic_name()),
        ),
    }
}

/// The failure a document run's error is answered with, `seat` being the
/// seat diagnostics the run collected.
#[must_use]
pub fn failure_of(error: ScheduleError, mut seat: Vec<(String, u64)>) -> OcrFailure {
    match error {
        ScheduleError::Cancelled => OcrFailure::Cancelled,
        ScheduleError::Render(_) => OcrFailure::Rejected,
        ScheduleError::Layout(LayoutError::Backend(_)) => {
            unavailable(Vec::new(), "layout_backend_failed")
        }
        ScheduleError::Layout(LayoutError::Shape(_)) | ScheduleError::Finish(_) => {
            OcrFailure::Internal
        }
        ScheduleError::Assemble(AssembleError::Recognition { diagnostics }) => {
            seat.extend(
                diagnostics.into_iter().map(|(name, count)| {
                    (name.to_owned(), u64::try_from(count).unwrap_or(u64::MAX))
                }),
            );
            unavailable(seat, "recognition_failed")
        }
        ScheduleError::Assemble(AssembleError::Rejected(_)) => OcrFailure::Rejected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::fixtures::pdf;
    use scriptor_layout::{MockLayout, NUM_CLASSES, RawOutputs};
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();
        move |name| map.get(name).cloned()
    }

    const URL: &str = "http://127.0.0.1:8000/v1/chat/completions";

    fn base() -> Vec<(&'static str, &'static str)> {
        vec![(URL_ENV, URL), (MODEL_ENV, "glm-ocr")]
    }

    fn rejected(pairs: &[(&str, &str)]) -> ConfigError {
        DocumentConfig::from_env(&env_of(pairs)).unwrap_err()
    }

    #[test]
    fn the_table_names_every_setting_once_with_a_legacy_name() {
        let mut current: Vec<&str> = SETTING_NAMES.iter().map(|(name, _)| *name).collect();
        let mut legacy: Vec<&str> = SETTING_NAMES.iter().map(|(_, name)| *name).collect();
        assert!(current.iter().all(|name| name.starts_with("SCRIPTOR_")));
        assert!(legacy.iter().all(|name| name.starts_with("SEMANTIK_")));
        current.sort_unstable();
        legacy.sort_unstable();
        current.dedup();
        legacy.dedup();
        assert_eq!(current.len(), SETTING_NAMES.len());
        assert_eq!(legacy.len(), SETTING_NAMES.len());
    }

    #[test]
    fn a_legacy_name_is_read_when_the_scriptor_name_is_unset() {
        for (current, legacy) in SETTING_NAMES {
            let raw = env_of(&[(legacy, "old")]);
            let view = with_legacy_fallback(&raw);
            assert_eq!(view(current).as_deref(), Some("old"), "{current}");
        }
    }

    #[test]
    fn the_scriptor_name_wins_over_the_legacy_name() {
        for (current, legacy) in SETTING_NAMES {
            let raw = env_of(&[(current, "new"), (legacy, "old")]);
            let view = with_legacy_fallback(&raw);
            assert_eq!(view(current).as_deref(), Some("new"), "{current}");
        }
    }

    #[test]
    fn the_scriptor_name_alone_is_read_and_unrelated_names_pass_through() {
        let raw = env_of(&[(URL_ENV, "new"), ("OTHER", "x")]);
        let view = with_legacy_fallback(&raw);
        assert_eq!(view(URL_ENV).as_deref(), Some("new"));
        assert_eq!(view(MODEL_ENV), None);
        assert_eq!(view("OTHER").as_deref(), Some("x"));
        assert_eq!(view("SEMANTIK_GLM_OCR_URL"), None);
    }

    #[test]
    fn document_settings_resolve_through_the_legacy_fallback() {
        let raw = env_of(&[
            ("SEMANTIK_GLM_OCR_URL", URL),
            ("SEMANTIK_GLM_OCR_MODEL", "old-model"),
            (MODEL_ENV, "new-model"),
        ]);
        let config = DocumentConfig::from_env(&with_legacy_fallback(&raw)).unwrap();
        assert_eq!(config.endpoint, URL);
        assert_eq!(config.model, "new-model");
    }

    #[test]
    fn unset_workers_follow_the_served_sequences() {
        let mut pairs = base();
        pairs.push((MAX_NUM_SEQS_ENV, "32"));
        assert_eq!(
            DocumentConfig::from_env(&env_of(&pairs)).unwrap().workers,
            64
        );
        pairs.push((WORKERS_ENV, "8"));
        assert_eq!(
            DocumentConfig::from_env(&env_of(&pairs)).unwrap().workers,
            8
        );
        let mut bad = base();
        bad.push((MAX_NUM_SEQS_ENV, "0"));
        assert_eq!(rejected(&bad).setting, MAX_NUM_SEQS_ENV);
    }

    #[test]
    fn defaults_follow_the_python_bridge() {
        let config = DocumentConfig::from_env(&env_of(&base())).unwrap();
        assert_eq!(config.endpoint, URL);
        assert_eq!(config.model, "glm-ocr");
        assert_eq!(config.figure_dpi, 300);
        assert_eq!(config.workers, 32);
        assert_eq!(config.layout_batch_size, 4);
        assert_eq!(config.connect_timeout, Duration::from_secs(30));
        assert_eq!(config.request_timeout, None);
        assert_eq!(config.api_key, None);
        assert_eq!(config.checkpoint_dir, None);
        assert_eq!(config.layout_runtime, None);
    }

    #[test]
    fn every_setting_is_read_by_its_name() {
        let mut pairs = base();
        pairs.extend([
            (API_KEY_ENV, "secret"),
            (LAYOUT_DEVICE_ENV, "cuda"),
            (CHECKPOINT_DIR_ENV, "/var/ckpt"),
            (FIGURE_DPI_ENV, "200"),
            (WORKERS_ENV, "8"),
            (LAYOUT_BATCH_SIZE_ENV, "2"),
            (CONNECT_TIMEOUT_ENV, "5"),
            (REQUEST_TIMEOUT_ENV, "60"),
            (LAYOUT_RUNTIME_ENV, "/opt/rt.so"),
            (LAYOUT_WEIGHTS_ENV, "/opt/w.onnx"),
        ]);
        let config = DocumentConfig::from_env(&env_of(&pairs)).unwrap();
        assert_eq!(config.api_key.as_deref(), Some("secret"));
        assert_eq!(config.layout_device.as_deref(), Some("cuda"));
        assert_eq!(config.checkpoint_dir, Some(PathBuf::from("/var/ckpt")));
        assert_eq!(
            (config.figure_dpi, config.workers, config.layout_batch_size),
            (200, 8, 2)
        );
        assert_eq!(config.connect_timeout, Duration::from_secs(5));
        assert_eq!(config.request_timeout, Some(Duration::from_secs(60)));
        assert_eq!(config.layout_runtime, Some(PathBuf::from("/opt/rt.so")));
        assert_eq!(config.layout_weights, Some(PathBuf::from("/opt/w.onnx")));
    }

    #[test]
    fn missing_and_invalid_settings_name_the_setting() {
        assert_eq!(rejected(&[(MODEL_ENV, "m")]).setting, URL_ENV);
        assert_eq!(rejected(&[(URL_ENV, URL)]).setting, MODEL_ENV);
        assert_eq!(
            rejected(&[(URL_ENV, URL), (MODEL_ENV, "a\nb")]).setting,
            MODEL_ENV
        );
        for bad in [
            "ftp://127.0.0.1/x",
            "http://example.com/v1",
            "http://127.0.0.1:8000",
            "http://user:pw@127.0.0.1/x",
            "http://127.0.0.1/x?a=1",
            "http://127.0.0.1/x#f",
            "http://127.0.0.1/ x",
        ] {
            assert_eq!(
                rejected(&[(URL_ENV, bad), (MODEL_ENV, "m")]).setting,
                URL_ENV,
                "{bad}"
            );
        }
        for good in ["http://localhost:1/x", "https://[::1]:2/x"] {
            assert!(
                DocumentConfig::from_env(&env_of(&[(URL_ENV, good), (MODEL_ENV, "m")])).is_ok()
            );
        }
        for (name, value) in [
            (FIGURE_DPI_ENV, "601"),
            (WORKERS_ENV, "0"),
            (WORKERS_ENV, "129"),
            (LAYOUT_BATCH_SIZE_ENV, "65"),
            (CONNECT_TIMEOUT_ENV, "301"),
            (REQUEST_TIMEOUT_ENV, "x"),
            (REQUEST_TIMEOUT_ENV, ""),
            (CHECKPOINT_DIR_ENV, "relative/dir"),
        ] {
            let mut pairs = base();
            pairs.push((name, value));
            assert_eq!(rejected(&pairs).setting, name, "{name}={value}");
        }
    }

    #[test]
    fn layout_files_must_be_set_and_exist() {
        let mut config = DocumentConfig::from_env(&env_of(&base())).unwrap();
        assert_eq!(
            config.layout_files().unwrap_err().setting,
            LAYOUT_RUNTIME_ENV
        );
        let dir = std::env::temp_dir().join(format!("semantik-doc-layout-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let runtime = dir.join("rt");
        std::fs::write(&runtime, b"x").unwrap();
        config.layout_runtime = Some(runtime.clone());
        config.layout_weights = Some(dir.join("missing"));
        assert_eq!(
            config.layout_files().unwrap_err().setting,
            LAYOUT_WEIGHTS_ENV
        );
        let weights = dir.join("w");
        std::fs::write(&weights, b"x").unwrap();
        config.layout_weights = Some(weights.clone());
        assert_eq!(
            config.layout_files().unwrap(),
            (runtime.as_path(), weights.as_path())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn schedule_errors_map_to_failures() {
        let seat = vec![("region_final_request_exception".to_owned(), 2)];
        assert_eq!(
            failure_of(ScheduleError::Cancelled, vec![]),
            OcrFailure::Cancelled
        );
        assert_eq!(
            failure_of(ScheduleError::Render(anyhow::anyhow!("x")), vec![]),
            OcrFailure::Rejected
        );
        assert_eq!(
            failure_of(ScheduleError::Finish("x".into()), vec![]),
            OcrFailure::Internal
        );
        assert_eq!(
            failure_of(
                ScheduleError::Layout(LayoutError::Shape("x".into())),
                vec![]
            ),
            OcrFailure::Internal
        );
        assert!(matches!(
            failure_of(
                ScheduleError::Layout(LayoutError::Backend("x".into())),
                vec![]
            ),
            OcrFailure::Unavailable { .. }
        ));
        let OcrFailure::Unavailable { diagnostics } = failure_of(
            ScheduleError::Assemble(AssembleError::Recognition {
                diagnostics: vec![("missing_text", 3)],
            }),
            seat,
        ) else {
            panic!("expected unavailable");
        };
        assert_eq!(
            diagnostics,
            vec![
                ("region_final_request_exception".to_owned(), 2),
                ("missing_text".to_owned(), 3)
            ]
        );
        assert_eq!(
            failure_of(
                ScheduleError::Assemble(AssembleError::Rejected("x")),
                vec![]
            ),
            OcrFailure::Rejected
        );
    }

    #[test]
    fn text_mode_makes_one_paragraph_per_nonempty_line() {
        let reply = text_json("first\r\n  \n second \u{2028}third");
        let texts: Vec<&str> = reply["regions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|region| region["text"].as_str().unwrap())
            .collect();
        assert_eq!(texts, ["first", "second", "third"]);
        assert_eq!(reply["regions"][2]["id"], "p0001-r000003");
        assert_eq!(reply["regions"][0]["page"], 1);
        assert_eq!(reply["title"], "");
        assert_eq!(reply["empty_pages"], json!([]));
    }

    /// A local seat answering every request with `content` and counting them.
    fn seat(content: &'static str) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&count);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let seen = Arc::clone(&seen);
                std::thread::spawn(move || {
                    let mut received = Vec::new();
                    loop {
                        let mut buffer = [0; 8192];
                        let Ok(count) = stream.read(&mut buffer) else {
                            return;
                        };
                        if count == 0 {
                            return;
                        }
                        received.extend_from_slice(&buffer[..count]);
                        let Some(end) = received.windows(4).position(|part| part == b"\r\n\r\n")
                        else {
                            continue;
                        };
                        let head = String::from_utf8_lossy(&received[..end]).to_ascii_lowercase();
                        let length = head
                            .lines()
                            .find_map(|line| line.strip_prefix("content-length:"))
                            .map_or(0, |value| value.trim().parse::<usize>().unwrap_or(0));
                        if received.len() >= end + 4 + length {
                            break;
                        }
                    }
                    seen.fetch_add(1, Ordering::SeqCst);
                    let body = json!({"choices": [{"message": {"content": content}}]}).to_string();
                    let _ = stream.write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    );
                });
            }
        });
        (format!("http://{address}/v1/chat/completions"), count)
    }

    fn mock() -> MockLayout {
        let mut raw = RawOutputs::blank(300, NUM_CLASSES);
        raw.set_query(0, 22, 4.0, [0.5, 0.25, 0.5, 0.125]);
        raw.set_query(1, 6, 4.0, [0.5, 0.75, 0.5, 0.125]);
        raw.set_order(0, 1, 6.0);
        MockLayout::new(vec![raw; 4])
    }

    fn conversion(endpoint: &str) -> NativeDocumentOcr {
        let mut config = DocumentConfig::from_env(&env_of(&base())).unwrap();
        config.endpoint = endpoint.to_owned();
        config.workers = 2;
        config.layout_batch_size = 2;
        config.request_timeout = Some(Duration::from_secs(20));
        NativeDocumentOcr::new(config, Arc::new(mock()))
    }

    #[test]
    fn a_synthetic_pdf_converts_through_a_local_seat() {
        let (endpoint, requests) = seat("Recognised words");
        let document = pdf(&[("[0 0 100 100]", "")]);
        let progress = std::sync::Mutex::new(Vec::new());
        let reply = conversion(&endpoint)
            .ocr(
                &document,
                OcrTask::Layout,
                &|done, total| progress.lock().unwrap().push((done, total)),
                &|| false,
                &|_, _| {},
            )
            .unwrap();
        assert_eq!(reply["empty_pages"], json!([]));
        assert!(reply["title"].is_string());
        let regions = reply["regions"].as_array().unwrap();
        assert!(!regions.is_empty());
        assert!(regions.iter().any(|region| {
            region["text"]
                .as_str()
                .is_some_and(|t| t.contains("Recognised"))
        }));
        assert!(requests.load(Ordering::SeqCst) >= 1);
        let progress = progress.into_inner().unwrap();
        assert_eq!(progress.first(), Some(&(0, 1)));
        assert_eq!(progress.last(), Some(&(1, 1)));
    }

    #[test]
    fn an_unreachable_seat_is_unavailable_and_cancellation_wins() {
        let closed = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!(
            "http://{}/v1/chat/completions",
            closed.local_addr().unwrap()
        );
        drop(closed);
        let document = pdf(&[("[0 0 100 100]", "")]);
        let outcome = conversion(&endpoint).ocr(
            &document,
            OcrTask::Layout,
            &|_, _| {},
            &|| false,
            &|_, _| {},
        );
        assert!(
            matches!(outcome, Err(OcrFailure::Unavailable { .. })),
            "{outcome:?}"
        );
        let outcome =
            conversion(&endpoint).ocr(&document, OcrTask::Layout, &|_, _| {}, &|| true, &|_, _| {});
        assert_eq!(outcome, Err(OcrFailure::Cancelled));
    }

    #[test]
    fn unreadable_sources_are_rejected_and_pdfs_have_no_text_mode() {
        let conversion = conversion(URL);
        assert_eq!(
            conversion.ocr(
                b"not a document",
                OcrTask::Layout,
                &|_, _| {},
                &|| false,
                &|_, _| {}
            ),
            Err(OcrFailure::Rejected)
        );
        let document = pdf(&[("[0 0 100 100]", "")]);
        assert_eq!(
            conversion.ocr(&document, OcrTask::Text, &|_, _| {}, &|| false, &|_, _| {}),
            Err(OcrFailure::Rejected)
        );
    }
}
