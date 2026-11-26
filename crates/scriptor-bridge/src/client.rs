//! The blocking chat client of the recognition seat: one region request with
//! in-call retries, one serialized tail retry under a per-document budget, and
//! typed failures.

use crate::request_policy::{Jitter, SeatGate, ThreadJitter, seat_gate, sleep_unless};
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, RETRY_AFTER};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// How often an in-flight attempt re-checks for cancellation.
const CANCEL_POLL: Duration = Duration::from_millis(100);

/// The retries made inside one call after the first attempt.
pub const SDK_RETRY_MAX_ATTEMPTS: u32 = 2;

/// The first retry's backoff.
pub const SDK_BACKOFF_BASE: Duration = Duration::from_millis(500);

/// The longest backoff, `Retry-After` included.
pub const SDK_BACKOFF_MAX: Duration = Duration::from_secs(8);

/// The jitter applied to a backoff, as a fraction of it either way.
pub const SDK_JITTER_RATIO: f64 = 0.2;

/// The statuses retried inside a call.
pub const SDK_RETRY_STATUSES: [u16; 5] = [429, 500, 502, 503, 504];

/// The wait before the tail retry.
pub const TAIL_BACKOFF: Duration = Duration::from_millis(250);

/// The tail retries one document may issue.
pub const MAX_TAIL_RETRIES: usize = 32;

/// The tail retries that run at once.
pub const TAIL_CONCURRENCY: usize = 1;

/// The connect timeout.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// The characters of an error body kept in a failure's detail.
const BODY_PREVIEW_CHARS: usize = 500;

/// The in-call retry timing.
#[derive(Clone, Debug)]
pub struct RetryTiming {
    pub max_retries: u32,
    pub backoff_base: Duration,
    pub backoff_max: Duration,
    pub jitter_ratio: f64,
}

impl Default for RetryTiming {
    fn default() -> Self {
        Self {
            max_retries: SDK_RETRY_MAX_ATTEMPTS,
            backoff_base: SDK_BACKOFF_BASE,
            backoff_max: SDK_BACKOFF_MAX,
            jitter_ratio: SDK_JITTER_RATIO,
        }
    }
}

/// How to reach the recognition seat.
#[derive(Clone, Debug)]
pub struct ClientConfig {
    /// The chat-completions URL.
    pub endpoint: String,
    /// Sent as a bearer token when set.
    pub api_key: Option<String>,
    pub connect_timeout: Duration,
    /// The total deadline of one request; `None` lets a request run as long
    /// as its generation takes.
    pub request_timeout: Option<Duration>,
    pub retry: RetryTiming,
}

impl ClientConfig {
    /// A configuration for `endpoint` with the default timeouts and retries.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: None,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            request_timeout: None,
            retry: RetryTiming::default(),
        }
    }
}

/// The class of a region request that did not produce text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureClass {
    /// A 200 whose body lacks `choices[0].message.content`.
    MalformedSuccess,
    /// A 200 whose body is not JSON.
    ResponseDecode,
    /// A 200 whose content is neither a string nor null.
    InvalidContentType,
    /// The request failed before a response: connect, timeout, or reset.
    RequestException,
    /// The last attempt was answered 429.
    RateLimited,
    /// The last attempt was answered 5xx.
    ProviderFailure,
    /// The seat answered a 4xx other than 429.
    PermanentClient,
    /// Any other status.
    Unexpected,
}

impl FailureClass {
    /// The suffix of the `region_final_*` diagnostic of the class.
    pub const fn diagnostic_name(self) -> &'static str {
        match self {
            Self::MalformedSuccess => "malformed_success",
            Self::ResponseDecode => "response_decode",
            Self::InvalidContentType => "invalid_content_type",
            Self::RequestException => "request_exception",
            Self::RateLimited => "rate_limited",
            Self::ProviderFailure => "provider_failure",
            Self::PermanentClient => "permanent_client",
            Self::Unexpected => "unexpected",
        }
    }

    /// Whether the tail retry applies to the class.
    pub const fn retryable(self) -> bool {
        matches!(
            self,
            Self::MalformedSuccess
                | Self::ResponseDecode
                | Self::RequestException
                | Self::RateLimited
                | Self::ProviderFailure
        )
    }
}

/// What became of the tail retry of a failed request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TailOutcome {
    /// The class is not retried.
    NotRetryable,
    /// The document's tail retries were spent.
    BudgetExhausted,
    /// The tail retry ran and failed too.
    Failed,
}

/// A region request that ended without text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionFailure {
    /// The class of the last attempt.
    pub class: FailureClass,
    /// The status of the last attempt when it had one.
    pub status: Option<u16>,
    pub detail: String,
    pub tail: TailOutcome,
}

/// What the seat read for one request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recognition {
    /// The reply text.
    pub text: String,
    /// Whether the seat's repetition stop ended the reply.
    pub repetition_stopped: bool,
}

/// Why a region request returned no text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegionError {
    /// The cancellation check held.
    Cancelled,
    Failed(RegionFailure),
}

impl fmt::Display for RegionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => f.write_str("the region request was cancelled"),
            Self::Failed(failure) => write!(
                f,
                "region recognition failed ({}{}): {}",
                failure.class.diagnostic_name(),
                failure
                    .status
                    .map(|status| format!(", status {status}"))
                    .unwrap_or_default(),
                failure.detail
            ),
        }
    }
}

impl std::error::Error for RegionError {}

/// The tail-retry budget and counters shared by every request of a document.
pub struct RegionRetryPolicy {
    max_tail_retries: usize,
    backoff: Duration,
    issued: AtomicUsize,
    tail_slot: SeatGate,
    counters: Mutex<BTreeMap<String, u64>>,
}

impl Default for RegionRetryPolicy {
    fn default() -> Self {
        Self::new(MAX_TAIL_RETRIES, TAIL_BACKOFF)
    }
}

impl RegionRetryPolicy {
    /// A policy allowing `max_tail_retries` tail retries per document, each
    /// after `backoff`, [`TAIL_CONCURRENCY`] at a time.
    pub fn new(max_tail_retries: usize, backoff: Duration) -> Self {
        let tail_slot = SeatGate::default();
        tail_slot.set_ceiling(TAIL_CONCURRENCY);
        Self {
            max_tail_retries,
            backoff,
            issued: AtomicUsize::new(0),
            tail_slot,
            counters: Mutex::new(BTreeMap::new()),
        }
    }

    /// Restores the budget and clears the counters.
    pub fn begin_document(&self) {
        self.issued.store(0, Ordering::SeqCst);
        self.lock_counters().clear();
    }

    /// The nonzero counters, by name, and resets them.
    pub fn take_diagnostics(&self) -> Vec<(String, u64)> {
        std::mem::take(&mut *self.lock_counters())
            .into_iter()
            .filter(|(_, count)| *count > 0)
            .collect()
    }

    fn lock_counters(&self) -> std::sync::MutexGuard<'_, BTreeMap<String, u64>> {
        self.counters
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn count(&self, name: &str) {
        *self.lock_counters().entry(name.to_owned()).or_insert(0) += 1;
    }

    fn count_final(&self, class: FailureClass) {
        self.count(&format!("region_final_{}", class.diagnostic_name()));
    }

    fn take_budget(&self) -> bool {
        let mut current = self.issued.load(Ordering::SeqCst);
        loop {
            if current >= self.max_tail_retries {
                return false;
            }
            match self.issued.compare_exchange(
                current,
                current + 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(seen) => current = seen,
            }
        }
    }
}

/// A blocking client for one recognition seat endpoint.
pub struct OcrClient {
    http: Client,
    config: ClientConfig,
    gate: std::sync::Arc<SeatGate>,
    policy: std::sync::Arc<RegionRetryPolicy>,
}

fn failure(class: FailureClass, status: Option<u16>, detail: impl Into<String>) -> RegionError {
    RegionError::Failed(RegionFailure {
        class,
        status,
        detail: detail.into(),
        tail: TailOutcome::NotRetryable,
    })
}

/// The seconds a backoff waits: the server's `Retry-After` when positive,
/// else `base * 2^attempt`, either capped at the maximum, then moved by up to
/// the jitter ratio either way (`unit` in `[0, 1)` picks the position).
fn backoff_seconds(timing: &RetryTiming, attempt: u32, retry_after: Option<f64>, unit: f64) -> f64 {
    let max = timing.backoff_max.as_secs_f64();
    let wait = match retry_after {
        Some(seconds) if seconds > 0.0 => seconds.min(max),
        _ => {
            let growth = 2f64.powi(i32::try_from(attempt).unwrap_or(i32::MAX));
            (timing.backoff_base.as_secs_f64() * growth).min(max)
        }
    };
    let spread = wait * timing.jitter_ratio;
    if spread > 0.0 {
        (wait + (unit * 2.0 - 1.0) * spread).max(0.0)
    } else {
        wait
    }
}

/// The `Retry-After` header parsed as a seconds float.
fn retry_after_seconds(headers: &HeaderMap) -> Option<f64> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?.trim();
    value.parse::<f64>().ok()
}

/// The `finish_reason` of a 200 body's first choice.
fn finish_reason(body: &Value) -> Option<&str> {
    body.get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("finish_reason"))
        .and_then(Value::as_str)
}

/// Whether a 200 body's first choice carries `stop_reason` `"repetition"`,
/// the seat's report that its repetition stop ended the reply. A body
/// without the field, as every standard server sends, reads as not stopped.
fn repetition_stopped(body: &Value) -> bool {
    body.get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("stop_reason"))
        .and_then(Value::as_str)
        == Some("repetition")
}

/// The reply of a 200 body: `choices[0].message.content` trimmed, null read
/// as empty, with whether the seat's repetition stop ended it. A reply the seat cut at its length limit is reported on standard
/// error.
fn reply(body: &Value) -> Result<Recognition, RegionError> {
    if finish_reason(body) == Some("length") {
        eprintln!(
            "semantik ocr warning: the reply was cut at the length limit (finish_reason=length)"
        );
    }
    let content = body
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"));
    match content {
        Some(Value::String(text)) => Ok(Recognition {
            text: text.trim().to_owned(),
            repetition_stopped: repetition_stopped(body),
        }),
        Some(Value::Null) => Ok(Recognition {
            text: String::new(),
            repetition_stopped: repetition_stopped(body),
        }),
        Some(other) => Err(failure(
            FailureClass::InvalidContentType,
            Some(200),
            format!("the reply content is {}", json_kind(other)),
        )),
        None => Err(failure(
            FailureClass::MalformedSuccess,
            Some(200),
            "the reply lacks choices[0].message.content",
        )),
    }
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

impl OcrClient {
    /// A client for `config` whose tail retries draw on `policy`.
    pub fn new(
        config: ClientConfig,
        policy: std::sync::Arc<RegionRetryPolicy>,
    ) -> Result<Self, crate::http::ClientError> {
        let http = crate::http::http_client_for(
            &config.endpoint,
            crate::http::HttpProfile {
                timeout: config.request_timeout,
                connect_timeout: Some(config.connect_timeout),
            },
        )?;
        let gate = seat_gate(&config.endpoint);
        Ok(Self {
            http,
            config,
            gate,
            policy,
        })
    }

    /// The text the seat reads for `body`; see [`Self::recognize_detailed`].
    pub fn recognize(
        &self,
        body: &Value,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<String, RegionError> {
        self.recognize_detailed(body, cancelled)
            .map(|recognition| recognition.text)
    }

    /// The text the seat reads for `body` and whether its repetition stop
    /// ended the reply. A failed call whose class is retryable gets one tail
    /// retry, after the tail backoff, while the document's budget lasts;
    /// `cancelled` is checked before every attempt and during every wait.
    pub fn recognize_detailed(
        &self,
        body: &Value,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Recognition, RegionError> {
        let payload = serde_json::to_vec(body)
            .map_err(|error| failure(FailureClass::Unexpected, None, error.to_string()))?;
        let first = match self.call(&payload, cancelled) {
            Ok(text) => return Ok(text),
            Err(RegionError::Cancelled) => return Err(RegionError::Cancelled),
            Err(RegionError::Failed(first)) => first,
        };
        self.policy.count_final(first.class);
        if !first.class.retryable() {
            return Err(RegionError::Failed(first));
        }
        if !self.policy.take_budget() {
            self.policy.count("region_tail_retry_budget_exhausted");
            return Err(RegionError::Failed(RegionFailure {
                tail: TailOutcome::BudgetExhausted,
                ..first
            }));
        }
        let Some(_slot) = self.policy.tail_slot.acquire(cancelled) else {
            return Err(RegionError::Cancelled);
        };
        if !sleep_unless(self.policy.backoff, cancelled) {
            return Err(RegionError::Cancelled);
        }
        match self.call(&payload, cancelled) {
            Ok(text) => {
                self.policy.count("region_tail_retry_recovered");
                Ok(text)
            }
            Err(RegionError::Cancelled) => Err(RegionError::Cancelled),
            Err(RegionError::Failed(second)) => {
                self.policy.count("region_tail_retry_exhausted");
                self.policy.count_final(second.class);
                Err(RegionError::Failed(RegionFailure {
                    tail: TailOutcome::Failed,
                    ..second
                }))
            }
        }
    }

    fn wait(
        &self,
        attempt: u32,
        retry_after: Option<f64>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(), RegionError> {
        let unit = ThreadJitter::new().unit();
        let seconds = backoff_seconds(&self.config.retry, attempt, retry_after, unit);
        if sleep_unless(Duration::from_secs_f64(seconds), cancelled) {
            Ok(())
        } else {
            Err(RegionError::Cancelled)
        }
    }

    /// One attempt on a detached thread, awaited in slices so `cancelled`
    /// ends the wait while the request is still in flight; the abandoned
    /// attempt finishes on its own and its result is discarded.
    fn attempt(
        &self,
        payload: &[u8],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Recognition, (RegionError, Option<f64>)> {
        let http = self.http.clone();
        let endpoint = self.config.endpoint.clone();
        let api_key = self.config.api_key.clone();
        let payload = payload.to_vec();
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = sender.send(send_and_read(&http, &endpoint, api_key.as_deref(), payload));
        });
        loop {
            match receiver.recv_timeout(CANCEL_POLL) {
                Ok(outcome) => return outcome,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if cancelled() {
                        return Err((RegionError::Cancelled, None));
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    return Err((
                        failure(
                            FailureClass::RequestException,
                            None,
                            "the request thread ended without an outcome",
                        ),
                        None,
                    ));
                }
            }
        }
    }

    /// One region call: the first attempt and up to `max_retries` retries of
    /// transport errors, undecodable 200 bodies, and retryable statuses.
    fn call(
        &self,
        payload: &[u8],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Recognition, RegionError> {
        let last = self.config.retry.max_retries;
        for attempt in 0..=last {
            if cancelled() {
                return Err(RegionError::Cancelled);
            }
            let Some(permit) = self.gate.acquire(cancelled) else {
                return Err(RegionError::Cancelled);
            };
            let outcome = self.attempt(payload, cancelled);
            drop(permit);
            match outcome {
                Ok(text) => {
                    self.gate.succeeded();
                    return Ok(text);
                }
                Err((error, retry_after)) => {
                    let retryable_here = match &error {
                        RegionError::Failed(found) => match found.class {
                            FailureClass::RequestException | FailureClass::ResponseDecode => true,
                            _ => found
                                .status
                                .is_some_and(|status| SDK_RETRY_STATUSES.contains(&status)),
                        },
                        RegionError::Cancelled => false,
                    };
                    if let RegionError::Failed(found) = &error
                        && found.status == Some(429)
                    {
                        self.gate.throttled();
                    }
                    if retryable_here && attempt < last {
                        self.wait(attempt, retry_after, cancelled)?;
                        continue;
                    }
                    return Err(error);
                }
            }
        }
        Err(failure(
            FailureClass::Unexpected,
            None,
            "the retry loop ended without an outcome",
        ))
    }
}

/// Posts `payload` to `endpoint` and reads the reply.
fn send_and_read(
    http: &Client,
    endpoint: &str,
    api_key: Option<&str>,
    payload: Vec<u8>,
) -> Result<Recognition, (RegionError, Option<f64>)> {
    let mut request = http
        .post(endpoint)
        .header(CONTENT_TYPE, "application/json")
        .body(payload);
    if let Some(key) = api_key {
        request = request.header(AUTHORIZATION, format!("Bearer {key}"));
    }
    match request.send() {
        Err(error) => Err((
            failure(FailureClass::RequestException, None, error.to_string()),
            None,
        )),
        Ok(response) => read(response),
    }
}

/// The text of `response`, or the failure it is with the `Retry-After`
/// seconds it carried.
fn read(response: reqwest::blocking::Response) -> Result<Recognition, (RegionError, Option<f64>)> {
    let status = response.status().as_u16();
    let retry_after = retry_after_seconds(response.headers());
    if status == 200 {
        let bytes = response.bytes().map_err(|error| {
            (
                failure(FailureClass::RequestException, Some(200), error.to_string()),
                None,
            )
        })?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
            (
                failure(FailureClass::ResponseDecode, Some(200), error.to_string()),
                None,
            )
        })?;
        return reply(&value).map_err(|error| (error, None));
    }
    let preview: String = response
        .text()
        .unwrap_or_default()
        .chars()
        .take(BODY_PREVIEW_CHARS)
        .collect();
    let class = match status {
        429 => FailureClass::RateLimited,
        400..=499 => FailureClass::PermanentClient,
        500..=599 => FailureClass::ProviderFailure,
        _ => FailureClass::Unexpected,
    };
    Err((failure(class, Some(status), preview), retry_after))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, mpsc};
    use std::thread;
    use std::time::Instant;

    #[test]
    fn a_choice_reports_the_repetition_stop_only_when_it_carries_the_field() {
        let choice = |extra: Value| {
            let mut choice = json!({"finish_reason": "stop", "message": {"content": " text "}});
            if let (Some(map), Some(extra)) = (choice.as_object_mut(), extra.as_object()) {
                map.extend(extra.clone());
            }
            json!({ "choices": [choice] })
        };
        let stopped = reply(&choice(json!({"stop_reason": "repetition"}))).unwrap();
        assert_eq!(stopped.text, "text");
        assert!(stopped.repetition_stopped);
        assert!(!reply(&choice(json!({}))).unwrap().repetition_stopped);
        assert!(
            !reply(&choice(json!({"stop_reason": "other"})))
                .unwrap()
                .repetition_stopped
        );
        assert!(
            !reply(&choice(json!({"stop_reason": null})))
                .unwrap()
                .repetition_stopped
        );
    }

    enum Reply {
        Status(u16, String),
        Hangup,
    }

    fn ok(text: &str) -> Reply {
        Reply::Status(
            200,
            json!({"choices": [{"message": {"content": text}}]}).to_string(),
        )
    }

    fn status(code: u16) -> Reply {
        Reply::Status(code, "busy".to_owned())
    }

    /// Serves `replies` one connection each; every request's lowercased head
    /// and body go to the returned channel.
    fn stub(replies: Vec<Reply>) -> (String, mpsc::Receiver<(String, Vec<u8>)>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            for reply in replies {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let mut received = Vec::new();
                let (head, payload) = loop {
                    let mut buffer = [0; 4096];
                    let count = stream.read(&mut buffer).unwrap();
                    received.extend_from_slice(&buffer[..count]);
                    let Some(end) = received.windows(4).position(|part| part == b"\r\n\r\n") else {
                        continue;
                    };
                    let head = String::from_utf8_lossy(&received[..end]).to_ascii_lowercase();
                    let length = head
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length:"))
                        .map_or(0, |value| value.trim().parse::<usize>().unwrap());
                    if received.len() >= end + 4 + length {
                        break (head, received[end + 4..end + 4 + length].to_vec());
                    }
                };
                sender.send((head, payload)).unwrap();
                if let Reply::Status(code, body) = reply {
                    let _ = stream.write_all(
                        format!(
                            "HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    );
                }
            }
        });
        (format!("http://{address}/v1/chat/completions"), receiver)
    }

    fn fast_client(url: &str, budget: usize) -> (OcrClient, Arc<RegionRetryPolicy>) {
        let mut config = ClientConfig::new(url);
        config.api_key = Some("secret".to_owned());
        config.request_timeout = Some(Duration::from_secs(10));
        config.retry.backoff_base = Duration::from_millis(1);
        config.retry.backoff_max = Duration::from_millis(5);
        let policy = Arc::new(RegionRetryPolicy::new(budget, Duration::from_millis(1)));
        (OcrClient::new(config, policy.clone()).unwrap(), policy)
    }

    fn never() -> bool {
        false
    }

    fn body() -> Value {
        json!({"messages": [], "model": "m"})
    }

    fn failed(result: Result<String, RegionError>) -> RegionFailure {
        match result {
            Err(RegionError::Failed(found)) => found,
            other => panic!("expected a failure, got {other:?}"),
        }
    }

    #[test]
    fn backoff_follows_the_sdk_formula() {
        let timing = RetryTiming::default();
        assert_eq!(backoff_seconds(&timing, 0, None, 0.5), 0.5);
        assert_eq!(backoff_seconds(&timing, 1, None, 0.5), 1.0);
        assert_eq!(backoff_seconds(&timing, 5, None, 0.5), 8.0);
        assert_eq!(backoff_seconds(&timing, 0, Some(3.0), 0.5), 3.0);
        assert_eq!(backoff_seconds(&timing, 0, Some(100.0), 0.5), 8.0);
        assert_eq!(backoff_seconds(&timing, 0, Some(0.0), 0.5), 0.5);
        assert!((backoff_seconds(&timing, 0, None, 0.0) - 0.4).abs() < 1e-9);
        assert!((backoff_seconds(&timing, 0, None, 1.0) - 0.6).abs() < 1e-9);
    }

    #[test]
    fn sends_the_body_with_a_bearer_token_and_returns_trimmed_text() {
        let (url, requests) = stub(vec![ok("  hello \n")]);
        let (client, _) = fast_client(&url, 4);
        assert_eq!(client.recognize(&body(), &never).unwrap(), "hello");
        let (head, payload) = requests.try_recv().unwrap();
        assert!(head.contains("authorization: bearer secret"));
        assert!(head.contains("application/json"));
        assert_eq!(serde_json::from_slice::<Value>(&payload).unwrap(), body());
    }

    #[test]
    fn transient_statuses_are_retried_inside_the_call() {
        let (url, requests) = stub(vec![status(503), status(429), ok("done")]);
        let (client, policy) = fast_client(&url, 4);
        assert_eq!(client.recognize(&body(), &never).unwrap(), "done");
        assert_eq!(requests.try_iter().count(), 3);
        assert!(policy.take_diagnostics().is_empty());
    }

    #[test]
    fn an_exhausted_call_gets_one_tail_retry() {
        let (url, requests) = stub(vec![status(503), status(503), status(503), ok("late")]);
        let (client, policy) = fast_client(&url, 4);
        assert_eq!(client.recognize(&body(), &never).unwrap(), "late");
        assert_eq!(requests.try_iter().count(), 4);
        assert_eq!(
            policy.take_diagnostics(),
            vec![
                ("region_final_provider_failure".to_owned(), 1),
                ("region_tail_retry_recovered".to_owned(), 1),
            ]
        );
    }

    #[test]
    fn a_failing_tail_retry_reports_the_last_class() {
        let mut replies = vec![status(503), status(503), status(503)];
        replies.extend([status(429), status(429), status(429)]);
        let (url, requests) = stub(replies);
        let (client, policy) = fast_client(&url, 4);
        let found = failed(client.recognize(&body(), &never));
        assert_eq!(found.class, FailureClass::RateLimited);
        assert_eq!(found.status, Some(429));
        assert_eq!(found.tail, TailOutcome::Failed);
        assert_eq!(requests.try_iter().count(), 6);
        let counters = policy.take_diagnostics();
        assert!(counters.contains(&("region_tail_retry_exhausted".to_owned(), 1)));
        assert!(counters.contains(&("region_final_rate_limited".to_owned(), 1)));
    }

    #[test]
    fn permanent_client_errors_are_not_retried() {
        let (url, requests) = stub(vec![status(400), ok("unused")]);
        let (client, _) = fast_client(&url, 4);
        let found = failed(client.recognize(&body(), &never));
        assert_eq!(found.class, FailureClass::PermanentClient);
        assert_eq!(found.tail, TailOutcome::NotRetryable);
        assert_eq!(requests.try_iter().count(), 1);
    }

    #[test]
    fn a_malformed_success_skips_the_sdk_retries_and_takes_the_tail() {
        let malformed = || Reply::Status(200, r#"{"choices":[]}"#.to_owned());
        let (url, requests) = stub(vec![malformed(), ok("fixed")]);
        let (client, _) = fast_client(&url, 4);
        assert_eq!(client.recognize(&body(), &never).unwrap(), "fixed");
        assert_eq!(requests.try_iter().count(), 2);

        let (url, requests) = stub(vec![malformed(), malformed()]);
        let (client, _) = fast_client(&url, 4);
        let found = failed(client.recognize(&body(), &never));
        assert_eq!(found.class, FailureClass::MalformedSuccess);
        assert_eq!(found.tail, TailOutcome::Failed);
        assert_eq!(requests.try_iter().count(), 2);
    }

    #[test]
    fn null_content_is_empty_text_and_other_types_fail_closed() {
        let null = Reply::Status(
            200,
            r#"{"choices":[{"message":{"content":null}}]}"#.to_owned(),
        );
        let (url, _requests) = stub(vec![null]);
        let (client, _) = fast_client(&url, 4);
        assert_eq!(client.recognize(&body(), &never).unwrap(), "");

        let number = Reply::Status(200, r#"{"choices":[{"message":{"content":5}}]}"#.to_owned());
        let (url, requests) = stub(vec![number, ok("unused")]);
        let (client, _) = fast_client(&url, 4);
        let found = failed(client.recognize(&body(), &never));
        assert_eq!(found.class, FailureClass::InvalidContentType);
        assert_eq!(found.tail, TailOutcome::NotRetryable);
        assert_eq!(requests.try_iter().count(), 1);
    }

    #[test]
    fn undecodable_bodies_are_retried_then_classed_response_decode() {
        let garbage = || Reply::Status(200, "not json".to_owned());
        let (url, requests) = stub((0..6).map(|_| garbage()).collect());
        let (client, _) = fast_client(&url, 4);
        let found = failed(client.recognize(&body(), &never));
        assert_eq!(found.class, FailureClass::ResponseDecode);
        assert_eq!(requests.try_iter().count(), 6);
    }

    #[test]
    fn dropped_connections_are_request_exceptions() {
        let hangups = (0..6).map(|_| Reply::Hangup).collect();
        let (url, requests) = stub(hangups);
        let (client, _) = fast_client(&url, 4);
        let found = failed(client.recognize(&body(), &never));
        assert_eq!(found.class, FailureClass::RequestException);
        assert_eq!(requests.try_iter().count(), 6);
    }

    #[test]
    fn the_tail_budget_is_spent_per_document() {
        let malformed = || Reply::Status(200, r#"{"choices":[]}"#.to_owned());
        let (url, requests) = stub(vec![malformed(), malformed(), malformed()]);
        let (client, policy) = fast_client(&url, 1);
        assert_eq!(
            failed(client.recognize(&body(), &never)).tail,
            TailOutcome::Failed
        );
        let found = failed(client.recognize(&body(), &never));
        assert_eq!(found.tail, TailOutcome::BudgetExhausted);
        assert_eq!(requests.try_iter().count(), 3);
        assert!(
            policy
                .take_diagnostics()
                .contains(&("region_tail_retry_budget_exhausted".to_owned(), 1))
        );
        policy.begin_document();
        assert!(policy.take_diagnostics().is_empty());
    }

    #[test]
    fn cancellation_ends_a_backoff_wait() {
        let (url, requests) = stub(vec![status(503)]);
        let mut config = ClientConfig::new(url);
        config.retry.backoff_base = Duration::from_secs(30);
        config.retry.backoff_max = Duration::from_secs(30);
        let policy = Arc::new(RegionRetryPolicy::default());
        let client = OcrClient::new(config, policy).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let started = Instant::now();
        let worker = thread::spawn(move || {
            client.recognize(&body(), &|| worker_stop.load(Ordering::SeqCst))
        });
        requests.recv().unwrap();
        stop.store(true, Ordering::SeqCst);
        crate::request_policy::notify_stop();
        assert_eq!(worker.join().unwrap(), Err(RegionError::Cancelled));
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn a_cancelled_request_returns_while_the_seat_is_still_answering() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!(
            "http://{}/v1/chat/completions",
            listener.local_addr().unwrap()
        );
        let held = thread::spawn(move || listener.accept().map(|(stream, _)| stream));
        let (client, _) = fast_client(&url, 4);
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(300));
            flag.store(true, Ordering::SeqCst);
        });
        let (done, result) = mpsc::channel();
        thread::spawn(move || {
            let _ = done.send(client.recognize(&body(), &|| stop.load(Ordering::SeqCst)));
        });
        let outcome = result
            .recv_timeout(Duration::from_secs(8))
            .expect("recognize must return once cancelled");
        assert_eq!(outcome, Err(RegionError::Cancelled));
        drop(held);
    }

    #[test]
    fn a_stop_before_the_first_attempt_sends_nothing() {
        let (url, requests) = stub(vec![ok("unused")]);
        let (client, _) = fast_client(&url, 4);
        assert_eq!(
            client.recognize(&body(), &|| true),
            Err(RegionError::Cancelled)
        );
        assert!(requests.try_recv().is_err());
    }
}
