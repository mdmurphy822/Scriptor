//! `POST /v1/chat/completions`: request validation, admission and the non-streaming reply.

use std::time::Instant;

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};
use scriptor_chat::{ChatMessage, Content, ContentPart, RenderOptions};
use scriptor_engine::request::{SamplingParams, SamplingRequest};
use scriptor_spec::SamplingDefaults;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::app::{AppState, GenEvent, GenerateRequest, count_images, unix_now};
use crate::errors::{ApiError, too_many_images_message};
use crate::health::all_done;
use crate::intake::{Refusal, Reservation, image_charge};
use crate::metrics::Metrics;
use crate::sse;
use crate::usage::Usage;

/// Streaming options of a chat request.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct StreamOptions {
    /// Append a final chunk carrying `usage`.
    pub include_usage: bool,
}

/// The wire body of a chat-completion request.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    /// Model name; must match the served name when present.
    #[serde(default)]
    pub model: Option<String>,
    /// Conversation messages.
    pub messages: Vec<ChatMessage>,
    /// Reply as server-sent events.
    #[serde(default)]
    pub stream: bool,
    /// Streaming options.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Tool declarations passed to the chat template.
    #[serde(default)]
    pub tools: Option<Value>,
    /// Tool selection; accepted and not interpreted.
    #[serde(default)]
    pub tool_choice: Option<Value>,
    /// Extra chat-template variables.
    #[serde(default)]
    pub chat_template_kwargs: Option<Value>,
    /// Number of choices; only 1 is served.
    #[serde(default)]
    pub n: Option<u32>,
    /// Output-format constraint; only absent, null or `{"type": "text"}` is served.
    #[serde(default)]
    pub response_format: Option<Value>,
    /// Per-token logit offsets; only absent, null or an empty object is served.
    #[serde(default)]
    pub logit_bias: Option<Value>,
    /// Sampling fields.
    #[serde(flatten)]
    pub sampling: SamplingRequest,
}

/// Rejects request features the server does not implement with a 400 naming the parameter;
/// requests that use none of them pass unchanged.
pub fn check_supported_features(req: &ChatRequest) -> Result<(), ApiError> {
    if req.sampling.logprobs == Some(true) {
        return Err(ApiError::bad_request(
            "logprobs are not supported: `logprobs` must be false or absent.",
            Some("logprobs"),
        ));
    }
    if req.sampling.top_logprobs.is_some_and(|n| n > 0) {
        return Err(ApiError::bad_request(
            "logprobs are not supported: `top_logprobs` must be 0 or absent.",
            Some("top_logprobs"),
        ));
    }
    if let Some(format) = req.response_format.as_ref().filter(|v| !v.is_null()) {
        let is_text = format.get("type").and_then(Value::as_str) == Some("text");
        if !is_text {
            return Err(ApiError::bad_request(
                "`response_format` is not supported: only {\"type\": \"text\"} is accepted.",
                Some("response_format"),
            ));
        }
    }
    if let Some(bias) = req.logit_bias.as_ref().filter(|v| !v.is_null()) {
        let is_empty = bias.as_object().is_some_and(|o| o.is_empty());
        if !is_empty {
            return Err(ApiError::bad_request(
                "`logit_bias` is not supported: it must be empty or absent.",
                Some("logit_bias"),
            ));
        }
    }
    Ok(())
}

/// Validates the media parts of a request: only base64 image data URLs, at most `limit_images`.
pub fn check_media(messages: &[ChatMessage], limit_images: usize) -> Result<(), ApiError> {
    let mut images = 0usize;
    for message in messages {
        let Some(Content::Parts(parts)) = &message.content else {
            continue;
        };
        for part in parts {
            match part {
                ContentPart::Text { .. } => {}
                ContentPart::ImageUrl { image_url, .. } => {
                    images += 1;
                    let url = image_url.url.as_str();
                    let is_data_image = url.starts_with("data:image/")
                        && url
                            .split_once(',')
                            .is_some_and(|(head, _)| head.ends_with(";base64"));
                    if !is_data_image {
                        return Err(ApiError::bad_request(
                            "Unsupported image URL: only base64 data URLs of the form \
                             data:image/<type>;base64,<data> are accepted.",
                            Some("messages"),
                        ));
                    }
                }
                ContentPart::Other(value) => {
                    let kind = value.get("type").and_then(Value::as_str).unwrap_or("");
                    let modality = match kind {
                        "input_audio" | "audio_url" => "audio",
                        "video_url" => "video",
                        other => {
                            return Err(ApiError::bad_request(
                                format!("Unknown content part type: {other:?}."),
                                Some("messages"),
                            ));
                        }
                    };
                    return Err(ApiError::bad_request(
                        too_many_images_message(modality, 0),
                        Some("messages"),
                    ));
                }
            }
        }
    }
    if images > limit_images {
        return Err(ApiError::bad_request(
            too_many_images_message("image", limit_images),
            Some("messages"),
        ));
    }
    Ok(())
}

/// Fills every sampling field the client omitted from the served defaults; explicit fields win.
pub fn with_defaults(
    mut sampling: SamplingRequest,
    defaults: &SamplingDefaults,
) -> SamplingRequest {
    sampling.temperature = sampling.temperature.or(defaults.temperature);
    sampling.top_p = sampling.top_p.or(defaults.top_p);
    sampling.top_k = sampling.top_k.or(defaults.top_k);
    sampling.repetition_penalty = sampling.repetition_penalty.or(defaults.repetition_penalty);
    sampling
}

/// Tracks text deliveries of one request and records time-to-first-token and
/// inter-delivery gaps.
pub(crate) struct DeliveryClock {
    started: Instant,
    last: Option<Instant>,
    first: Option<Instant>,
}

impl DeliveryClock {
    pub(crate) fn new(started: Instant) -> Self {
        Self {
            started,
            last: None,
            first: None,
        }
    }

    /// Notes one text delivery and records its latency series.
    pub(crate) fn delivered(&mut self, metrics: &Metrics) {
        let now = Instant::now();
        match self.last {
            None => {
                self.first = Some(now);
                metrics.observe_first_token(now.duration_since(self.started).as_secs_f64());
            }
            Some(previous) => {
                metrics.observe_inter_token(now.duration_since(previous).as_secs_f64());
            }
        }
        self.last = Some(now);
    }

    /// The instant of the first delivery.
    pub(crate) fn first(&self) -> Option<Instant> {
        self.first
    }
}

fn render_options(req: &ChatRequest) -> Result<RenderOptions, ApiError> {
    let mut options = match &req.chat_template_kwargs {
        Some(kwargs) => RenderOptions::from_kwargs(kwargs)
            .map_err(|e| ApiError::bad_request(e.to_string(), Some("chat_template_kwargs")))?,
        None => RenderOptions::default(),
    };
    let sets_generation_prompt = req
        .chat_template_kwargs
        .as_ref()
        .and_then(|k| k.get("add_generation_prompt"))
        .is_some();
    if !sets_generation_prompt {
        options.add_generation_prompt = true;
    }
    if let Some(tools) = &req.tools {
        options.tools = Some(tools.clone());
    }
    Ok(options)
}

/// Handles a chat-completion request in JSON or SSE form.
pub async fn chat_completions(State(state): State<AppState>, request: Request) -> Response {
    match read_body(&state, request).await {
        Ok(body) => match handle(state, body).await {
            Ok(response) => response,
            Err(err) => err.into_response(),
        },
        Err(err) => err.into_response(),
    }
}

async fn read_body(state: &AppState, request: Request) -> Result<Bytes, ApiError> {
    let limit = state.config.max_body_bytes();
    let too_large = || {
        ApiError::payload_too_large(format!(
            "The request body exceeds the {limit} byte limit, derived as {} image(s) of base64 \
             PNG at max_pixels {} plus {} bytes of text.",
            state.config.limit_images,
            state.config.max_pixels,
            crate::app::TEXT_BODY_BYTES
        ))
    };
    let declared = request
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    if declared.is_some_and(|n| n > limit) {
        return Err(too_large());
    }
    axum::body::to_bytes(request.into_body(), limit)
        .await
        .map_err(|_| too_large())
}

/// Reserves the bytes a request holds while in flight (its body plus each image at its
/// decoded payload); `None` when the intake is unbounded.
fn reserve_intake(
    state: &AppState,
    body_bytes: usize,
    messages: &[ChatMessage],
) -> Result<Option<Reservation>, ApiError> {
    let Some(intake) = &state.intake else {
        return Ok(None);
    };
    let resize = state.config.resize_config();
    let images = count_images(messages);
    let payload = messages
        .iter()
        .filter_map(|m| match &m.content {
            Some(Content::Parts(parts)) => Some(parts),
            _ => None,
        })
        .flatten()
        .filter_map(|p| match p {
            ContentPart::ImageUrl { image_url, .. } => Some(image_charge(&image_url.url, &resize)),
            _ => None,
        })
        .fold(0u64, u64::saturating_add);
    let cost = (body_bytes as u64).saturating_add(payload);
    let slots = state.engine.stats().slots;
    match intake.reserve(cost, slots, state.config.retry_after_secs) {
        Ok(reservation) => Ok(Some(reservation)),
        Err(Refusal::TooLarge { cost, budget }) => Err(ApiError::payload_too_large(format!(
            "The request needs {cost} bytes of intake ({body_bytes} body bytes plus {images} \
             image(s) of {payload} decoded bytes in total at max_pixels {}), more than the \
             {budget} byte intake budget.",
            state.config.max_pixels
        ))),
        Err(Refusal::Full {
            cost,
            free,
            budget,
            retry_after,
        }) => {
            state.metrics.record_rejected_busy();
            Err(ApiError::overloaded(
                format!(
                    "The intake is full: this request needs {cost} bytes and {free} of the \
                     {budget} byte intake budget are free; retry after {retry_after}s."
                ),
                retry_after,
            ))
        }
    }
}

async fn handle(state: AppState, body: Bytes) -> Result<Response, ApiError> {
    let req: ChatRequest = serde_json::from_slice(&body)
        .map_err(|e| ApiError::bad_request(format!("Invalid request body: {e}"), Some("body")))?;
    let served = state.engine.served_name().to_owned();
    if let Some(model) = &req.model
        && model != &served
    {
        return Err(ApiError::not_found(format!(
            "The model `{model}` does not exist."
        )));
    }
    if !all_done(&state.engine.health()) {
        return Err(ApiError::unavailable("The engine is still starting."));
    }
    if req.n.is_some_and(|n| n != 1) {
        return Err(ApiError::bad_request("Only n = 1 is supported.", Some("n")));
    }
    check_supported_features(&req)?;
    check_media(&req.messages, state.config.limit_images)?;
    let sampling_request = with_defaults(req.sampling.clone(), &state.config.sampling_defaults);
    let sampling = SamplingParams::from_request(&sampling_request)
        .map_err(|e| ApiError::bad_request(e.message, Some(e.param)))?;
    let render = render_options(&req)?;
    #[cfg(feature = "kv-demand-diagnostic")]
    if req.stream && state.engine.refuses_streaming() {
        return Err(ApiError::bad_request(
            "streaming is unsupported under kv_policy=demand-diagnostic",
            Some("stream"),
        ));
    }
    let reservation = reserve_intake(&state, body.len(), &req.messages)?;
    let id = state.next_request_id();
    let started = Instant::now();
    let rx = state.engine.submit(GenerateRequest {
        request_id: id.clone(),
        messages: req.messages,
        render,
        sampling,
        #[cfg(feature = "kv-demand-diagnostic")]
        restartable: !req.stream,
    })?;
    if req.stream {
        let include_usage = req.stream_options.is_some_and(|o| o.include_usage);
        return Ok(sse::stream_response(
            state,
            rx,
            id,
            served,
            include_usage,
            started,
            reservation,
        ));
    }
    let _reservation = reservation;
    collect(state, rx, id, served, started).await
}

/// Records a finished request in the metrics.
pub(crate) fn record_finished(
    state: &AppState,
    usage: &Usage,
    finish_reason: &str,
    started: Instant,
    first_token: Option<Instant>,
) {
    let now = Instant::now();
    let first = first_token.unwrap_or(now);
    state.metrics.record_request(
        usage.prompt_tokens,
        usage.completion_tokens,
        finish_reason,
        first.duration_since(started).as_secs_f64(),
        now.duration_since(first).as_secs_f64(),
    );
}

async fn collect(
    state: AppState,
    mut rx: tokio::sync::mpsc::Receiver<GenEvent>,
    id: String,
    model: String,
    started: Instant,
) -> Result<Response, ApiError> {
    let mut text = String::new();
    let mut clock = DeliveryClock::new(started);
    while let Some(event) = rx.recv().await {
        match event {
            GenEvent::Delta(piece) => {
                clock.delivered(&state.metrics);
                text.push_str(&piece);
            }
            GenEvent::Queued(seconds) => state.metrics.observe_queue_time(seconds),
            GenEvent::Finished {
                finish_reason,
                stop_reason,
                prompt_tokens,
                completion_tokens,
            } => {
                let usage = Usage::new(prompt_tokens, completion_tokens);
                record_finished(&state, &usage, &finish_reason, started, clock.first());
                let body = json!({
                    "id": id,
                    "object": "chat.completion",
                    "created": unix_now(),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "logprobs": null,
                        "finish_reason": finish_reason,
                        "stop_reason": stop_reason,
                    }],
                    "usage": usage,
                });
                return Ok(Json(body).into_response());
            }
            GenEvent::Failed(message) => return Err(ApiError::internal(message)),
            GenEvent::QueueFull { retry_after } => {
                return Err(ApiError::too_many_requests(retry_after));
            }
            #[cfg(feature = "kv-demand-diagnostic")]
            GenEvent::Restarted => {
                text.clear();
                clock = DeliveryClock::new(started);
            }
            GenEvent::Cancelled { reason } => {
                state.metrics.record_cancelled(&reason);
                return Err(ApiError::cancelled(&reason));
            }
        }
    }
    Err(ApiError::internal(
        "The engine closed the stream without finishing the request.",
    ))
}
