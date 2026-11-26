//! Sampling parameters carried by a chat-completion request: wire shape,
//! defaults, validation with vLLM-worded errors, and greedy normalisation.

use serde::Deserialize;

/// Temperatures below this are treated as greedy decoding.
pub const GREEDY_EPSILON: f32 = 1e-5;

/// Stop strings as sent on the wire: one string or a list.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum StopSpec {
    /// A single stop string.
    One(String),
    /// Several stop strings.
    Many(Vec<String>),
}

/// Sampling fields of a request body; every field is optional on the wire.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct SamplingRequest {
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus mass.
    pub top_p: Option<f32>,
    /// Number of candidates kept.
    pub top_k: Option<i32>,
    /// Repetition penalty.
    pub repetition_penalty: Option<f32>,
    /// Presence penalty.
    pub presence_penalty: Option<f32>,
    /// Frequency penalty.
    pub frequency_penalty: Option<f32>,
    /// Maximum generated tokens.
    pub max_tokens: Option<i64>,
    /// Alias of `max_tokens`.
    pub max_completion_tokens: Option<i64>,
    /// Stop strings.
    pub stop: Option<StopSpec>,
    /// Generator seed.
    pub seed: Option<i64>,
    /// Whether to return log-probabilities.
    pub logprobs: Option<bool>,
    /// Number of alternatives per position.
    pub top_logprobs: Option<i64>,
    /// Whether generation ends once the text is a repetition loop; off unless set.
    pub stop_on_repetition: Option<bool>,
}

/// Validated sampling parameters of one sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// Sampling temperature; `0.0` is greedy.
    pub temperature: f32,
    /// Nucleus mass in `(0, 1]`.
    pub top_p: f32,
    /// Candidates kept; `0` disables.
    pub top_k: i32,
    /// Repetition penalty, positive.
    pub repetition_penalty: f32,
    /// Presence penalty in `[-2, 2]`.
    pub presence_penalty: f32,
    /// Frequency penalty in `[-2, 2]`.
    pub frequency_penalty: f32,
    /// Generation limit; `None` runs to the context limit.
    pub max_tokens: Option<u32>,
    /// Stop strings, in request order.
    pub stop: Vec<String>,
    /// Generator seed; `None` draws one per request.
    pub seed: Option<u64>,
    /// Alternatives returned per position; `None` disables log-probabilities.
    pub logprobs: Option<u32>,
    /// When set, ends generation with a `stop` once the generated text repeats a unit of
    /// at least ten characters ten times in a row.
    pub stop_on_repetition: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            max_tokens: None,
            stop: Vec::new(),
            seed: None,
            logprobs: None,
            stop_on_repetition: false,
        }
    }
}

/// A rejected sampling field; `message` is the user-facing 400 text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamError {
    /// Offending field name.
    pub param: &'static str,
    /// vLLM-worded message.
    pub message: String,
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ParamError {}

fn reject(param: &'static str, message: String) -> ParamError {
    ParamError { param, message }
}

impl SamplingParams {
    /// Builds validated parameters from the request fields, applying vLLM
    /// defaults. Greedy requests (temperature below [`GREEDY_EPSILON`]) drop
    /// `top_p` and `top_k`.
    pub fn from_request(req: &SamplingRequest) -> Result<Self, ParamError> {
        let d = Self::default();
        let temperature = req.temperature.unwrap_or(d.temperature);
        if temperature.is_nan() || temperature < 0.0 {
            return Err(reject(
                "temperature",
                format!("temperature must be non-negative, got {temperature}."),
            ));
        }
        let top_p = req.top_p.unwrap_or(d.top_p);
        if !(top_p > 0.0 && top_p <= 1.0) {
            return Err(reject(
                "top_p",
                format!("top_p must be in (0, 1], got {top_p}."),
            ));
        }
        let top_k = req.top_k.unwrap_or(d.top_k);
        if top_k < -1 {
            return Err(reject(
                "top_k",
                format!("top_k must be -1 (disable), 0 (disable), or at least 1, got {top_k}."),
            ));
        }
        let top_k = top_k.max(0);
        let repetition_penalty = req.repetition_penalty.unwrap_or(d.repetition_penalty);
        if repetition_penalty.is_nan() || repetition_penalty <= 0.0 {
            return Err(reject(
                "repetition_penalty",
                format!("repetition_penalty must be greater than zero, got {repetition_penalty}."),
            ));
        }
        let presence_penalty = req.presence_penalty.unwrap_or(d.presence_penalty);
        if !(-2.0..=2.0).contains(&presence_penalty) {
            return Err(reject(
                "presence_penalty",
                format!("presence_penalty must be in [-2, 2], got {presence_penalty}."),
            ));
        }
        let frequency_penalty = req.frequency_penalty.unwrap_or(d.frequency_penalty);
        if !(-2.0..=2.0).contains(&frequency_penalty) {
            return Err(reject(
                "frequency_penalty",
                format!("frequency_penalty must be in [-2, 2], got {frequency_penalty}."),
            ));
        }
        let max_tokens = match req.max_completion_tokens.or(req.max_tokens) {
            None => None,
            Some(n) if n >= 1 => Some(u32::try_from(n).unwrap_or(u32::MAX)),
            Some(n) => {
                return Err(reject(
                    "max_tokens",
                    format!("max_tokens must be at least 1, got {n}."),
                ));
            }
        };
        let stop = match &req.stop {
            None => Vec::new(),
            Some(StopSpec::One(s)) => vec![s.clone()],
            Some(StopSpec::Many(v)) => v.clone(),
        }
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect();
        let seed = req.seed.map(|s| s as u64);
        let logprobs = match (req.logprobs.unwrap_or(false), req.top_logprobs) {
            (_, Some(n)) if n < 0 => {
                return Err(reject(
                    "top_logprobs",
                    format!("top_logprobs must be non-negative, got {n}."),
                ));
            }
            (false, Some(n)) if n > 0 => {
                return Err(reject(
                    "top_logprobs",
                    "when using `top_logprobs`, `logprobs` must be set to true.".to_string(),
                ));
            }
            (true, n) => Some(u32::try_from(n.unwrap_or(0)).unwrap_or(u32::MAX)),
            (false, _) => None,
        };
        let greedy = temperature < GREEDY_EPSILON;
        Ok(Self {
            temperature: if greedy { 0.0 } else { temperature },
            top_p: if greedy { 1.0 } else { top_p },
            top_k: if greedy { 0 } else { top_k },
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            max_tokens,
            stop,
            seed,
            logprobs,
            stop_on_repetition: req.stop_on_repetition.unwrap_or(d.stop_on_repetition),
        })
    }

    /// Parses the sampling fields from a JSON request body.
    pub fn from_json(body: &serde_json::Value) -> Result<Self, ParamError> {
        let req: SamplingRequest = serde_json::from_value(body.clone())
            .map_err(|e| reject("body", format!("invalid sampling parameters: {e}")))?;
        Self::from_request(&req)
    }

    /// Whether decoding is argmax.
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }
}
