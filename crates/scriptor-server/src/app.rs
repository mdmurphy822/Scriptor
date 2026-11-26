//! Shared server state, the engine contract, the router, and a scripted engine for tests.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use scriptor_chat::{ChatMessage, Content, ContentPart, RenderOptions};
use scriptor_engine::request::SamplingParams;
use scriptor_spec::SamplingDefaults;
use tokio::sync::mpsc;

use crate::errors::EngineError;
use crate::health::{self, Dead, Stage};
use crate::intake::Intake;
use crate::metrics::Metrics;
use crate::{chat, models};

/// Server-side limits and behaviour switches.
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// Maximum images per prompt (`--limit-mm-per-prompt` for images).
    pub limit_images: usize,
    /// Bytes of request bodies and decoded image payloads the intake may hold in flight;
    /// `None` leaves the intake unbounded.
    pub intake_bytes: Option<u64>,
    /// Smallest pixel count the image preprocessor resizes to; sizes the per-image payload
    /// reservation.
    pub min_pixels: u64,
    /// Largest pixel count the image preprocessor admits; sizes the body limit and bounds
    /// the per-image payload reservation.
    pub max_pixels: u64,
    /// Seconds advertised in `Retry-After` on 429 responses.
    pub retry_after_secs: u64,
    /// Sampling values applied to the fields a request omits.
    pub sampling_defaults: SamplingDefaults,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            limit_images: 1,
            min_pixels: scriptor_image::MIN_PIXELS,
            intake_bytes: None,
            max_pixels: scriptor_image::MAX_PIXELS,
            retry_after_secs: 1,
            sampling_defaults: SamplingDefaults::default(),
        }
    }
}

/// Bytes of request text allowed beside the images.
pub(crate) const TEXT_BODY_BYTES: usize = 1 << 20;

impl AppConfig {
    /// The resize parameters the image preprocessor applies.
    pub fn resize_config(&self) -> scriptor_image::SmartResizeConfig {
        scriptor_image::SmartResizeConfig {
            min_pixels: self.min_pixels,
            max_pixels: self.max_pixels,
            ..scriptor_image::SmartResizeConfig::default()
        }
    }

    /// Bytes one worst-case request reserves: the largest body plus the decoded payload of
    /// every image it may carry.
    pub fn max_request_bytes(&self) -> u64 {
        (self.max_body_bytes() as u64).saturating_add(
            (self.limit_images as u64)
                .saturating_mul(crate::intake::image_payload_bytes(self.max_pixels)),
        )
    }

    /// Largest accepted request body: `limit_images` base64 images, each sized for an
    /// uncompressed RGB PNG at the preprocessor's maximum pixel count, plus the text allowance.
    pub fn max_body_bytes(&self) -> usize {
        let side = (self.max_pixels as f64).sqrt().ceil() as usize;
        let png = 3 * side * side + side + side * side / 100 + 4096;
        let base64 = png.div_ceil(3) * 4;
        self.limit_images
            .saturating_mul(base64)
            .saturating_add(TEXT_BODY_BYTES)
    }
}

/// A validated chat request handed to the engine.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// Request identifier, unique per server run.
    pub request_id: String,
    /// Conversation messages in request order.
    pub messages: Vec<ChatMessage>,
    /// Chat-template options.
    pub render: RenderOptions,
    /// Sampling parameters.
    pub sampling: SamplingParams,
    /// Whether the engine may discard the output and run the request again from its prompt:
    /// true for requests that answer in one piece.
    #[cfg(feature = "kv-demand-diagnostic")]
    pub restartable: bool,
}

/// An event on a generation stream.
#[derive(Debug, Clone, PartialEq)]
pub enum GenEvent {
    /// Newly decoded text.
    Delta(String),
    /// Seconds the request waited in the engine before admission.
    Queued(f64),
    /// Generation ended.
    Finished {
        /// `stop` or `length`.
        finish_reason: String,
        /// Why a `stop` finish happened when it was not an end-of-sequence token or a stop
        /// string; `repetition` when the opt-in repetition stop ended generation.
        stop_reason: Option<String>,
        /// Prompt tokens, including image tokens.
        prompt_tokens: u64,
        /// Generated tokens.
        completion_tokens: u64,
    },
    /// Generation failed after admission.
    Failed(String),
    /// The engine's admission queue refused the request before it produced output; retry after
    /// the given seconds.
    QueueFull {
        /// Suggested delay.
        retry_after: u64,
    },
    /// The engine evicted the request and starts it over: every delta delivered so far is void
    /// and the full output follows again.
    #[cfg(feature = "kv-demand-diagnostic")]
    Restarted,
    /// The engine ended the request before it completed; `reason` is `drain` when the engine
    /// was shutting down and `client` when the consumer asked for it.
    Cancelled {
        /// Why the request was cancelled.
        reason: String,
    },
}

/// Where the engine is in its life.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Lifecycle {
    /// Accepting requests (or still booting; the stage list says which).
    #[default]
    Running,
    /// Shutting down: running requests finish, new ones are refused.
    Draining,
    /// Failed fatally; the text names the cause.
    Dead(String),
}

impl Lifecycle {
    /// Lowercase state name used in `/health` and metrics.
    pub fn label(&self) -> &'static str {
        match self {
            Lifecycle::Running => "running",
            Lifecycle::Draining => "draining",
            Lifecycle::Dead(_) => "dead",
        }
    }
}

/// A snapshot of live engine counters used to render `/metrics` and admission control.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EngineStats {
    /// Sequences currently decoding.
    pub running: u64,
    /// Requests admitted but not yet running.
    pub waiting: u64,
    /// Fraction of the KV pool in use, in `[0, 1]`.
    pub kv_cache_usage: f64,
    /// Configured stream count: sequences the engine runs at once.
    pub slots: u64,
    /// Fraction of the KV pool promised to running sequences, in `[0, 1]`.
    pub kv_reserved_perc: f64,
    /// Tokens the KV pool holds.
    pub kv_cache_tokens: u64,
    /// Speculative drafts issued.
    pub spec_drafts: u64,
    /// Draft tokens proposed.
    pub spec_draft_tokens: u64,
    /// Draft tokens accepted.
    pub spec_accepted_tokens: u64,
    /// Accepted draft tokens per draft position.
    pub spec_accepted_per_pos: Vec<u64>,
    /// Decode graphs held in device memory.
    pub graphs_captured: u64,
    /// Decode steps served by graph replay.
    pub graph_replays: u64,
    /// Decode steps served by replaying a graph that uses the two-stage target selector.
    pub graph_split_replays: u64,
    /// Decode steps that ran eagerly for lack of a covering graph.
    pub graph_fallbacks: u64,
    /// Slot occupancy of the device-resident decode steps.
    pub slot_occupancy: scriptor_engine::SlotOccupancy,
    /// Host-span and vision-flight counters; `None` unless `SCRIPTOR_L1_COUNTERS=1`.
    pub l1: Option<scriptor_engine::l1_counters::L1Stats>,
    /// Embedding-cache hits.
    pub embedding_cache_hits: u64,
    /// Host-pressure tier: 0 normal, 1 watch, 2 shed, 3 protect.
    pub pressure_tier: u8,
    /// Host `MemAvailable` at the last pressure sample, in bytes.
    pub host_mem_available_bytes: u64,
    /// Host full memory-stall average over ten seconds at the last sample, in percent.
    pub host_psi_memory_full_avg10: f64,
    /// Shed actions the pressure monitor issued.
    pub load_shed_total: u64,
    /// Protect actions the pressure monitor issued.
    pub load_protect_total: u64,
    /// Requests refused because host memory was under pressure.
    pub admission_rejected_pressure: u64,
    /// Requests cancelled to relieve host memory pressure.
    pub requests_shed_total: u64,
    /// Sequences preempted, by recompute or by restart (yields excluded).
    pub preemptions: u64,
    /// Sequences preempted for recompute.
    pub preemptions_recompute: u64,
    /// Sequences evicted and restarted from their prompt.
    pub preemptions_restart: u64,
    /// Sequences in the vision or prefill phase returned to the queue to free blocks; not
    /// part of `preemptions`.
    pub preemptions_yield: u64,
    /// Generated tokens discarded by restarts.
    pub restart_discarded_tokens: u64,
    /// Whether KV blocks are committed on demand rather than reserved.
    pub kv_demand: bool,
    /// KV blocks held by sequences.
    pub kv_used_blocks: u64,
    /// Blocks the running sequences needed to cover their next decode step at the last pre-step.
    pub kv_step_growth_blocks: u64,
    /// Requests the engine has taken on: running plus evicted ones awaiting re-admission.
    pub accepted_requests: u64,
    /// Device bytes of image embeddings held by live work.
    pub pinned_embedding_bytes: u64,
    /// Device bytes of image embeddings, including cache-only entries.
    pub retained_embedding_bytes: u64,
    /// Steps that blocked admission, by reason: slots, watermark, hysteresis, prompt blocks,
    /// accepted, embedding.
    pub admission_blocked: [u64; 6],
    /// Submissions refused because the waiting queue was full.
    pub requests_rejected_queue: u64,
    /// Wall seconds of each completed boot stage, in stage order.
    pub boot_stage_seconds: Vec<(String, f64)>,
    /// Cumulative prefill seconds of the scheduler.
    pub prefill_seconds: f64,
    /// Cumulative decode seconds of the scheduler.
    pub decode_seconds: f64,
    /// Cumulative engine-thread seconds spent preparing decode steps: sequence selection, KV
    /// reservation and row assembly. Reported apart from `decode_seconds`, which still
    /// includes the executor's own input preparation.
    pub decode_prep_seconds: f64,
    /// Cumulative vision seconds of the scheduler.
    pub vision_seconds: f64,
    /// Engine-thread seconds spent submitting or running vision batches.
    pub vision_host_seconds: f64,
    /// Vision batches by image count: `(images, batches)` ascending.
    pub vision_batch_sizes: Vec<(u64, u64)>,
    /// Lifecycle state of the engine.
    pub lifecycle: Lifecycle,
    /// Engine-thread loop iterations completed.
    pub heartbeat_beats: u64,
    /// Seconds since the engine thread last completed an iteration while it has work; zero
    /// when idle.
    pub heartbeat_age_seconds: f64,
    /// Vision batches by patch count rounded up to a power of two: `(upper bound, batches)`
    /// ascending.
    pub vision_batch_patch_sizes: Vec<(u64, u64)>,
    /// Admitted sequences waiting for their image embeddings.
    pub active_vision: u64,
    /// Admitted sequences waiting for or in a prefill pass.
    pub active_prefill: u64,
    /// Admitted sequences decoding.
    pub active_running: u64,
    /// Image buffers released after prefill.
    pub embedding_released: u64,
    /// Device bytes of the released image buffers.
    pub embedding_released_bytes: u64,
    /// Images submitted.
    pub images_submitted: u64,
    /// Patches of the submitted images.
    pub image_patches_submitted: u64,
    /// Submitted images by patch count rounded up to a power of two: `(upper bound, images)`
    /// ascending.
    pub image_patch_sizes: Vec<(u64, u64)>,
    /// Submitted images whose exact content key had not been seen before.
    pub image_key_unique: u64,
    /// Submitted images whose exact content key had been seen before.
    pub image_key_repeat: u64,
    /// Patches of the submitted images counted in `image_key_repeat`.
    pub image_key_repeat_patches: u64,
    /// Submitted images whose key was not compared because the key ledger was full.
    pub image_key_untracked: u64,
    /// Prefill passes by token count rounded up to a power of two: `(upper bound, passes)`
    /// ascending.
    pub prefill_pass_token_sizes: Vec<(u64, u64)>,
    /// Tokens summed over all prefill passes.
    pub prefill_pass_tokens: u64,
    /// Prefill passes below the tensor-pipe ridge size.
    pub prefill_passes_below_ridge: u64,
    /// Tokens summed over the passes counted in `prefill_passes_below_ridge`.
    pub prefill_pass_tokens_below_ridge: u64,
    /// Images embedded by the vision tower.
    pub vision_images: u64,
    /// Patches embedded by the vision tower.
    pub vision_patches: u64,
}

/// The inference engine as the HTTP layer sees it.
pub trait Engine: Send + Sync + 'static {
    /// Model name clients address.
    fn served_name(&self) -> &str;
    /// Context length in tokens.
    fn max_model_len(&self) -> u64;
    /// Startup stages; the engine is ready when all are done.
    fn health(&self) -> Vec<Stage>;
    /// Set once the engine has stopped serving for good; health then reports `dead`.
    fn dead(&self) -> Option<Dead> {
        None
    }
    /// Current counters.
    fn stats(&self) -> EngineStats;
    /// Lifecycle state; engines that never drain or die stay `Running`.
    fn lifecycle(&self) -> Lifecycle {
        Lifecycle::Running
    }
    /// Whether streaming responses are refused because the engine may restart requests.
    #[cfg(feature = "kv-demand-diagnostic")]
    fn refuses_streaming(&self) -> bool {
        false
    }
    /// Admits a request and returns its event stream.
    fn submit(&self, request: GenerateRequest) -> Result<mpsc::Receiver<GenEvent>, EngineError>;
}

/// State shared by all handlers.
#[derive(Clone)]
pub struct AppState {
    /// The engine behind the API.
    pub engine: Arc<dyn Engine>,
    /// Metric registry, also given to the engine for native series.
    pub metrics: Arc<Metrics>,
    /// Limits and switches.
    pub config: AppConfig,
    /// Byte ledger of in-flight requests, when the intake is bounded.
    pub intake: Option<Intake>,
    /// Unix time the server started, reported as the model `created` time.
    pub started_unix: u64,
    next_id: Arc<AtomicU64>,
}

impl AppState {
    /// Builds state over an engine and a metric registry.
    pub fn new(engine: Arc<dyn Engine>, metrics: Arc<Metrics>, config: AppConfig) -> Self {
        Self {
            engine,
            metrics,
            intake: config.intake_bytes.map(Intake::new),
            config,
            started_unix: unix_now(),
            next_id: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Allocates the next request identifier.
    pub fn next_request_id(&self) -> String {
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("chatcmpl-{:x}{n:08x}", unix_now())
    }
}

/// Seconds since the Unix epoch.
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Builds the router for `/v1/models`, `/v1/chat/completions`, `/metrics` and `/health`.
pub fn router(state: AppState) -> Router {
    let body_limit = state.config.max_body_bytes();
    Router::new()
        .route("/v1/models", get(models::list_models))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/metrics", get(metrics_text))
        .route("/debug/active-steps", get(active_steps))
        .route("/health", get(health::health))
        .layer(DefaultBodyLimit::max(body_limit))
        .layer(tower_http::catch_panic::CatchPanicLayer::custom(
            panic_response,
        ))
        .with_state(state)
}

fn panic_response(_: Box<dyn std::any::Any + Send + 'static>) -> axum::response::Response {
    crate::errors::ApiError::internal("The request handler failed internally.").into_response()
}

/// `GET /debug/active-steps`: the active-count telemetry as JSON, 404 while it is off.
async fn active_steps() -> axum::response::Response {
    match scriptor_engine::active_telemetry::full_snapshot() {
        Some(t) => ([(header::CONTENT_TYPE, "application/json")], t.dump_json()).into_response(),
        None => axum::http::StatusCode::NOT_FOUND.into_response(),
    }
}

async fn metrics_text(State(state): State<AppState>) -> impl IntoResponse {
    let body = state
        .metrics
        .render(state.engine.served_name(), &state.engine.stats());
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

/// An engine that replays a fixed token script, for exercising the HTTP surface.
pub struct ScriptedEngine {
    name: String,
    max_model_len: u64,
    text_tokens: u64,
    image_tokens: u64,
    pieces: Vec<String>,
    ready: AtomicBool,
    waiting: AtomicU64,
    stages: Mutex<Vec<Stage>>,
    failure: Mutex<Option<EngineError>>,
    dead: Mutex<Option<Dead>>,
    lifecycle: Mutex<Lifecycle>,
    cancel_with: Mutex<Option<String>>,
    stop_reason: Option<String>,
    received: Mutex<Vec<GenerateRequest>>,
    #[cfg(feature = "kv-demand-diagnostic")]
    refuse_streaming: AtomicBool,
    #[cfg(feature = "kv-demand-diagnostic")]
    restart_after: Mutex<Option<usize>>,
    refuse_at_intake: Mutex<Option<u64>>,
}

impl ScriptedEngine {
    /// Creates a ready engine that answers every request with `pieces`, one token each.
    pub fn new(name: &str, max_model_len: u64, pieces: &[&str]) -> Self {
        Self {
            name: name.to_owned(),
            max_model_len,
            text_tokens: 12,
            image_tokens: 235,
            pieces: pieces.iter().map(|p| (*p).to_owned()).collect(),
            ready: AtomicBool::new(true),
            waiting: AtomicU64::new(0),
            stages: Mutex::new(Vec::new()),
            failure: Mutex::new(None),
            dead: Mutex::new(None),
            lifecycle: Mutex::new(Lifecycle::Running),
            cancel_with: Mutex::new(None),
            stop_reason: None,
            received: Mutex::new(Vec::new()),
            #[cfg(feature = "kv-demand-diagnostic")]
            refuse_streaming: AtomicBool::new(false),
            #[cfg(feature = "kv-demand-diagnostic")]
            restart_after: Mutex::new(None),
            refuse_at_intake: Mutex::new(None),
        }
    }

    /// Sets the prompt tokens contributed by text and by each image.
    pub fn with_prompt_tokens(mut self, text_tokens: u64, image_tokens: u64) -> Self {
        self.text_tokens = text_tokens;
        self.image_tokens = image_tokens;
        self
    }

    /// Makes every finished stream report `stop_reason`.
    pub fn with_stop_reason(mut self, stop_reason: &str) -> Self {
        self.stop_reason = Some(stop_reason.to_owned());
        self
    }

    /// Sets readiness; a not-ready engine reports an unfinished `weights` stage.
    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::SeqCst);
    }

    /// Marks the engine as stopped for good; `/health` then reports `dead`.
    pub fn set_dead(&self, dead: Dead) {
        *self.dead.lock().unwrap_or_else(|e| e.into_inner()) = Some(dead);
    }

    /// Sets the reported waiting-queue length.
    pub fn set_waiting(&self, waiting: u64) {
        self.waiting.store(waiting, Ordering::SeqCst);
    }

    /// Replaces the reported stages; when empty the stage list follows readiness.
    pub fn set_stages(&self, stages: Vec<Stage>) {
        *self.stages.lock().unwrap_or_else(|e| e.into_inner()) = stages;
    }

    /// Makes every following `submit` fail with `error`.
    pub fn fail_with(&self, error: EngineError) {
        *self.failure.lock().unwrap_or_else(|e| e.into_inner()) = Some(error);
    }

    /// Sets the reported lifecycle state.
    pub fn set_lifecycle(&self, state: Lifecycle) {
        *self.lifecycle.lock().unwrap_or_else(|e| e.into_inner()) = state;
    }

    /// Makes every following stream end with a cancellation for `reason` instead of finishing.
    pub fn cancel_with(&self, reason: &str) {
        *self.cancel_with.lock().unwrap_or_else(|e| e.into_inner()) = Some(reason.to_owned());
    }

    /// Makes the engine refuse streaming responses, as an engine that restarts requests does.
    #[cfg(feature = "kv-demand-diagnostic")]
    pub fn refuse_streaming(&self) {
        self.refuse_streaming.store(true, Ordering::SeqCst);
    }

    /// Makes each restartable request deliver `pieces` deltas, restart, and then deliver the
    /// whole script again.
    #[cfg(feature = "kv-demand-diagnostic")]
    pub fn restart_after(&self, pieces: usize) {
        *self.restart_after.lock().unwrap_or_else(|e| e.into_inner()) = Some(pieces);
    }

    /// Makes every following request pass `submit` and then be refused by the engine's intake:
    /// its stream ends with [`GenEvent::QueueFull`] carrying `retry_after`.
    pub fn refuse_at_intake(&self, retry_after: u64) {
        *self
            .refuse_at_intake
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(retry_after);
    }

    /// Requests received so far.
    pub fn received(&self) -> Vec<GenerateRequest> {
        self.received
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

pub(crate) fn count_images(messages: &[ChatMessage]) -> u64 {
    messages
        .iter()
        .filter_map(|m| match &m.content {
            Some(Content::Parts(parts)) => Some(parts),
            _ => None,
        })
        .flatten()
        .filter(|p| matches!(p, ContentPart::ImageUrl { .. }))
        .count() as u64
}

impl Engine for ScriptedEngine {
    fn served_name(&self) -> &str {
        &self.name
    }

    fn max_model_len(&self) -> u64 {
        self.max_model_len
    }

    #[cfg(feature = "kv-demand-diagnostic")]
    fn refuses_streaming(&self) -> bool {
        self.refuse_streaming.load(Ordering::SeqCst)
    }

    fn dead(&self) -> Option<Dead> {
        self.dead.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn health(&self) -> Vec<Stage> {
        let stages = self.stages.lock().unwrap_or_else(|e| e.into_inner());
        if !stages.is_empty() {
            return stages.clone();
        }
        vec![Stage::new("weights", self.ready.load(Ordering::SeqCst))]
    }

    fn stats(&self) -> EngineStats {
        EngineStats {
            waiting: self.waiting.load(Ordering::SeqCst),
            lifecycle: self.lifecycle(),
            ..EngineStats::default()
        }
    }

    fn lifecycle(&self) -> Lifecycle {
        self.lifecycle
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn submit(&self, request: GenerateRequest) -> Result<mpsc::Receiver<GenEvent>, EngineError> {
        if !self.ready.load(Ordering::SeqCst) {
            return Err(EngineError::NotReady);
        }
        match self.lifecycle() {
            Lifecycle::Running => {}
            Lifecycle::Draining => return Err(EngineError::Draining),
            Lifecycle::Dead(reason) => return Err(EngineError::Dead(reason)),
        }
        if let Some(err) = self
            .failure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
        {
            return Err(err);
        }
        let prompt_tokens = self.text_tokens + self.image_tokens * count_images(&request.messages);
        let max_tokens = request.sampling.max_tokens.map(u64::from);
        let overflow = match max_tokens {
            Some(out) => prompt_tokens + out > self.max_model_len,
            None => prompt_tokens >= self.max_model_len,
        };
        if overflow {
            return Err(EngineError::ContextOverflow {
                prompt_tokens,
                max_tokens,
                max_model_len: self.max_model_len,
            });
        }
        let budget = max_tokens.unwrap_or(u64::MAX);
        let take = (self.pieces.len() as u64).min(budget) as usize;
        let finish_reason = if take < self.pieces.len() {
            "length"
        } else {
            "stop"
        };
        let (tx, rx) = mpsc::channel(2 * take + 4);
        let refused = *self
            .refuse_at_intake
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(retry_after) = refused {
            let _ = tx.try_send(GenEvent::QueueFull { retry_after });
            self.received
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(request);
            return Ok(rx);
        }
        let _ = tx.try_send(GenEvent::Queued(0.0));
        #[cfg(feature = "kv-demand-diagnostic")]
        if request.restartable
            && let Some(n) = *self.restart_after.lock().unwrap_or_else(|e| e.into_inner())
        {
            for piece in &self.pieces[..take.min(n)] {
                let _ = tx.try_send(GenEvent::Delta(piece.clone()));
            }
            let _ = tx.try_send(GenEvent::Restarted);
        }
        for piece in &self.pieces[..take] {
            let _ = tx.try_send(GenEvent::Delta(piece.clone()));
        }
        let cancel = self
            .cancel_with
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let _ = tx.try_send(match cancel {
            Some(reason) => GenEvent::Cancelled { reason },
            None => GenEvent::Finished {
                finish_reason: finish_reason.to_owned(),
                stop_reason: self.stop_reason.clone(),
                prompt_tokens,
                completion_tokens: take as u64,
            },
        });
        self.received
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(request);
        Ok(rx)
    }
}
