//! HTTP error responses in the vLLM error-body shape, and the engine failures that map to them.

use axum::Json;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// An error returned to the client as `{"error": {...}}`.
#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    /// HTTP status.
    pub status: StatusCode,
    /// Error class name placed in `error.type`.
    pub kind: &'static str,
    /// User-facing message.
    pub message: String,
    /// Offending request parameter, if any.
    pub param: Option<String>,
    /// Seconds for the `Retry-After` header on 429/503 responses.
    pub retry_after: Option<u64>,
}

impl ApiError {
    /// A 400 response.
    pub fn bad_request(message: impl Into<String>, param: Option<&str>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "BadRequestError",
            message: message.into(),
            param: param.map(str::to_owned),
            retry_after: None,
        }
    }

    /// A 404 response.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            kind: "NotFoundError",
            message: message.into(),
            param: None,
            retry_after: None,
        }
    }

    /// A 413 response.
    pub fn payload_too_large(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            kind: "PayloadTooLarge",
            message: message.into(),
            param: None,
            retry_after: None,
        }
    }

    /// A 503 response for load the intake cannot hold, carrying `Retry-After`.
    pub fn overloaded(message: impl Into<String>, retry_after: u64) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            kind: "ServiceUnavailable",
            message: message.into(),
            param: None,
            retry_after: Some(retry_after),
        }
    }

    /// A 429 response carrying `Retry-After`.
    pub fn too_many_requests(retry_after: u64) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            kind: "TooManyRequests",
            message: "The admission queue is full; retry after the indicated delay.".to_owned(),
            param: None,
            retry_after: Some(retry_after),
        }
    }

    /// A 503 response for an engine that is not ready.
    pub fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            kind: "ServiceUnavailable",
            message: message.into(),
            param: None,
            retry_after: Some(1),
        }
    }

    /// A 503 response for a request the engine cancelled before it finished.
    pub fn cancelled(reason: &str) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            kind: "RequestCancelled",
            message: format!("The request was cancelled before it finished: {reason}."),
            param: None,
            retry_after: Some(1),
        }
    }

    /// A 500 response.
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "InternalServerError",
            message: message.into(),
            param: None,
            retry_after: None,
        }
    }

    /// The JSON body of the response.
    pub fn body(&self) -> serde_json::Value {
        json!({"error": {
            "message": self.message,
            "type": self.kind,
            "param": self.param,
            "code": self.status.as_u16(),
        }})
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (self.status, Json(self.body())).into_response();
        if let Some(secs) = self.retry_after
            && let Ok(value) = HeaderValue::from_str(&secs.to_string())
        {
            response.headers_mut().insert(header::RETRY_AFTER, value);
        }
        response
    }
}

/// A request the engine refused or could not serve.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineError {
    /// Prompt plus requested output does not fit the model context.
    ContextOverflow {
        /// Prompt tokens, including image tokens.
        prompt_tokens: u64,
        /// Requested output tokens, when the request set a limit.
        max_tokens: Option<u64>,
        /// Model context length.
        max_model_len: u64,
    },
    /// An image or other media part cannot be processed.
    UnsupportedMedia(String),
    /// More images than the per-prompt limit.
    TooManyImages {
        /// Per-prompt image limit.
        limit: usize,
        /// Images in the request.
        got: usize,
    },
    /// A request part exceeds a derived size bound; the message names the bound.
    PayloadTooLarge(String),
    /// The admission queue is full; retry after the given seconds.
    QueueFull {
        /// Suggested delay.
        retry_after: u64,
    },
    /// The engine has not finished starting.
    NotReady,
    /// The engine is draining for shutdown and accepts no new requests.
    Draining,
    /// The engine has failed fatally; the text names the cause.
    Dead(String),
    /// The request is invalid for another reason.
    Invalid(String),
    /// An internal failure.
    Internal(String),
}

impl From<EngineError> for ApiError {
    fn from(err: EngineError) -> Self {
        match err {
            EngineError::ContextOverflow {
                prompt_tokens,
                max_tokens,
                max_model_len,
            } => ApiError::bad_request(
                context_overflow_message(prompt_tokens, max_tokens, max_model_len),
                Some("input_tokens"),
            ),
            EngineError::PayloadTooLarge(detail) => ApiError::payload_too_large(detail),
            EngineError::UnsupportedMedia(detail) => ApiError::bad_request(detail, None),
            EngineError::TooManyImages { limit, .. } => {
                ApiError::bad_request(too_many_images_message("image", limit), None)
            }
            EngineError::QueueFull { retry_after } => ApiError::too_many_requests(retry_after),
            EngineError::NotReady => ApiError::unavailable("The engine is still starting."),
            EngineError::Draining => ApiError::unavailable(
                "The engine is draining for shutdown and accepts no requests.",
            ),
            EngineError::Dead(reason) => ApiError::unavailable(format!(
                "The engine has failed and is shutting down: {reason}"
            )),
            EngineError::Invalid(detail) => ApiError::bad_request(detail, None),
            EngineError::Internal(detail) => ApiError::internal(detail),
        }
    }
}

/// vLLM-worded text for a prompt that does not fit the context window.
pub fn context_overflow_message(
    prompt_tokens: u64,
    max_tokens: Option<u64>,
    max_model_len: u64,
) -> String {
    match max_tokens {
        Some(out) => format!(
            "You passed {prompt_tokens} input tokens and requested {out} output tokens. \
             However, the model's context length is only {max_model_len} tokens, resulting in a \
             maximum input length of {} tokens. Please reduce the length of the input prompt.",
            max_model_len.saturating_sub(out)
        ),
        None => format!(
            "The decoder prompt (length {prompt_tokens}) is longer than the maximum model length \
             of {max_model_len}. Make sure that `max_model_len` is no smaller than the number of \
             text tokens plus multimodal tokens."
        ),
    }
}

/// vLLM-worded text for a prompt carrying more media items than `--limit-mm-per-prompt` allows.
pub fn too_many_images_message(modality: &str, limit: usize) -> String {
    format!("At most {limit} {modality}(s) may be provided in one prompt.")
}
