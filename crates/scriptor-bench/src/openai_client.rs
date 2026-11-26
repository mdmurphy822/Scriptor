//! A blocking client for OpenAI-compatible chat endpoints that posts the exact
//! body the Semantik OCR bridge sends.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::regions::Task;
use crate::{BenchError, Result};

/// The reply limit of a region request.
pub const MAX_TOKENS: u32 = 8_192;
/// The sampling temperature.
pub const TEMPERATURE: f64 = 0.0;
/// The nucleus mass.
pub const TOP_P: f64 = 0.00001;
/// The candidate count.
pub const TOP_K: u32 = 1;
/// The repetition penalty.
pub const REPETITION_PENALTY: f64 = 1.1;

/// The chat-completions body of one region request: the image part then the prompt
/// part in a single user message, the sampling fields, and `model` when not empty.
pub fn request_body(model: &str, task: Task, image_data_url: &str) -> Value {
    let mut body = json!({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": task.prompt()},
            ],
        }],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
    });
    if !model.is_empty() {
        body["model"] = json!(model);
    }
    body
}

/// Adds `logprobs: true, top_logprobs: 2` to a body.
pub fn with_logprobs(mut body: Value) -> Value {
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(2);
    body
}

/// The chat-completions URL of an endpoint given as a base URL or a full URL.
pub fn chat_url(endpoint: &str) -> String {
    let e = endpoint.trim_end_matches('/');
    if e.ends_with("/chat/completions") {
        e.to_owned()
    } else if e.ends_with("/v1") {
        format!("{e}/chat/completions")
    } else {
        format!("{e}/v1/chat/completions")
    }
}

/// The models URL of an endpoint given as a base URL or a full URL.
pub fn models_url(endpoint: &str) -> String {
    let chat = chat_url(endpoint);
    let base = chat.trim_end_matches("/chat/completions");
    format!("{base}/models")
}

/// What one successful chat reply carries.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatReply {
    pub content: String,
    pub finish_reason: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Per generated token, the log-probability gap between the two most likely
    /// candidates; `None` when the endpoint returned no logprobs.
    pub margins: Option<Vec<f64>>,
    pub latency: Duration,
}

/// Parses a chat-completions reply body.
pub fn parse_reply(body: &Value, latency: Duration) -> Result<ChatReply> {
    let choice = body
        .pointer("/choices/0")
        .ok_or_else(|| BenchError("reply has no choices[0]".into()))?;
    let content = choice
        .pointer("/message/content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let finish_reason = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let usage = |k: &str| {
        body.pointer(&format!("/usage/{k}"))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    let margins = choice
        .pointer("/logprobs/content")
        .and_then(Value::as_array)
        .map(|toks| {
            toks.iter()
                .map(|t| {
                    let mut lps: Vec<f64> = t
                        .get("top_logprobs")
                        .and_then(Value::as_array)
                        .map(|a| {
                            a.iter()
                                .filter_map(|c| c.get("logprob")?.as_f64())
                                .collect()
                        })
                        .unwrap_or_default();
                    lps.sort_by(|a, b| b.total_cmp(a));
                    match lps.as_slice() {
                        [a, b, ..] => a - b,
                        _ => f64::NAN,
                    }
                })
                .collect()
        });
    Ok(ChatReply {
        content,
        finish_reason,
        prompt_tokens: usage("prompt_tokens"),
        completion_tokens: usage("completion_tokens"),
        margins,
        latency,
    })
}

/// A chat endpoint a run can post to.
pub trait ChatEndpoint: Sync {
    /// Posts one body and returns the parsed reply.
    fn chat(&self, body: &Value) -> Result<ChatReply>;
    /// Whether the endpoint has accepted logprobs requests so far.
    fn logprobs_supported(&self) -> bool;
}

/// Retry behaviour for 429 and 5xx replies.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    pub attempts: usize,
    pub backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            attempts: 5,
            backoff: Duration::from_millis(200),
        }
    }
}

/// A blocking HTTP client bound to one endpoint.
pub struct Client {
    agent: ureq::Agent,
    chat_url: String,
    models_url: String,
    bearer: Option<String>,
    retry: RetryPolicy,
    logprobs_ok: AtomicBool,
}

impl Client {
    /// A client for `endpoint` (base URL or full chat URL).
    pub fn new(endpoint: &str, bearer: Option<String>, retry: RetryPolicy) -> Self {
        let config = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(None)
            .build();
        Self {
            agent: ureq::Agent::new_with_config(config),
            chat_url: chat_url(endpoint),
            models_url: models_url(endpoint),
            bearer,
            retry,
            logprobs_ok: AtomicBool::new(true),
        }
    }

    /// The chat-completions URL.
    pub fn url(&self) -> &str {
        &self.chat_url
    }

    /// The ids listed by `GET /v1/models`.
    pub fn models(&self) -> Result<Vec<String>> {
        let mut req = self.agent.get(&self.models_url);
        if let Some(t) = &self.bearer {
            req = req.header("Authorization", format!("Bearer {t}"));
        }
        let mut resp = req.call().map_err(|e| BenchError(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp
            .body_mut()
            .read_to_string()
            .map_err(|e| BenchError(e.to_string()))?;
        if status != 200 {
            return Err(BenchError(format!("models: HTTP {status}")));
        }
        let v: Value = serde_json::from_str(&text)?;
        Ok(v.get("data")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|m| m.get("id")?.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default())
    }

    fn post(&self, body: &Value) -> Result<(u16, String, Option<u64>)> {
        let mut req = self
            .agent
            .post(&self.chat_url)
            .header("Content-Type", "application/json");
        if let Some(t) = &self.bearer {
            req = req.header("Authorization", format!("Bearer {t}"));
        }
        let mut resp = req
            .send(body.to_string())
            .map_err(|e| BenchError(e.to_string()))?;
        let status = resp.status().as_u16();
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.trim().parse::<u64>().ok());
        let text = resp
            .body_mut()
            .read_to_string()
            .map_err(|e| BenchError(e.to_string()))?;
        Ok((status, text, retry_after))
    }

    fn send_with_retries(&self, body: &Value) -> Result<(u16, String, Instant)> {
        let start = Instant::now();
        let mut attempt = 0;
        loop {
            let (status, text, retry_after) = self.post(body)?;
            let transient = status == 429 || status >= 500;
            if transient && attempt < self.retry.attempts {
                attempt += 1;
                let wait = retry_after
                    .map(Duration::from_secs)
                    .unwrap_or(self.retry.backoff * attempt as u32);
                std::thread::sleep(wait);
                continue;
            }
            return Ok((status, text, start));
        }
    }
}

impl ChatEndpoint for Client {
    fn chat(&self, body: &Value) -> Result<ChatReply> {
        let wants_logprobs = body.get("logprobs").and_then(Value::as_bool) == Some(true);
        let mut body = body.clone();
        if wants_logprobs && !self.logprobs_ok.load(Ordering::Relaxed) {
            strip_logprobs(&mut body);
        }
        let (mut status, mut text, mut start) = self.send_with_retries(&body)?;
        if wants_logprobs && (status == 400 || status == 422) {
            self.logprobs_ok.store(false, Ordering::Relaxed);
            strip_logprobs(&mut body);
            (status, text, start) = self.send_with_retries(&body)?;
        }
        if status != 200 {
            return Err(BenchError(format!(
                "HTTP {status}: {}",
                text.chars().take(300).collect::<String>()
            )));
        }
        parse_reply(&serde_json::from_str(&text)?, start.elapsed())
    }

    fn logprobs_supported(&self) -> bool {
        self.logprobs_ok.load(Ordering::Relaxed)
    }
}

fn strip_logprobs(body: &mut Value) {
    if let Some(o) = body.as_object_mut() {
        o.remove("logprobs");
        o.remove("top_logprobs");
    }
}
