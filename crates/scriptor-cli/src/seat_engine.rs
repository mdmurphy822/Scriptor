//! The HTTP-facing engine: turns chat requests into scheduler requests, streams decoded text
//! back, and reports readiness and counters.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use scriptor_cache::{ImageGrid as CacheGrid, embedding_key};
use scriptor_chat::{
    ChatMessage, ChatTemplate, Content, ContentPart, ImageGrid, PromptBuilder, RenderOptions,
    Tokenizer,
};
use scriptor_engine::boot::{BootStatus, StageState};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::sampling::{FinishReason, RepeatTail, find_stop};
use scriptor_engine::supervise::serve_jobs;
use scriptor_engine::{EngineEvent, EngineHandle, EngineRequest, ImageInput};
use scriptor_image::{
    GridThw, MediaPolicy, RgbImage, SmartResizeConfig, load_media, patchify_bf16, preprocess,
};
use scriptor_server::{
    Engine, EngineError, EngineStats, GenEvent, GenerateRequest, Lifecycle, Next, Stage, next_event,
};
use scriptor_spec::{ImageProcessorConfig, ModelSpec};
use tokio::sync::mpsc;

/// Text and image preparation shared by served requests and the boot self-test.
pub struct Frontend {
    builder: PromptBuilder,
    resize: SmartResizeConfig,
    policy: MediaPolicy,
}

/// A prepared request plus the prompt length it will occupy.
pub struct Prepared {
    pub request: EngineRequest,
}

impl Frontend {
    /// Loads the tokenizer and chat template from the model directory.
    pub fn load(
        model_dir: &Path,
        spec: &ModelSpec,
        image: &ImageProcessorConfig,
    ) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| format!("tokenizer: {e}"))?;
        let template = ChatTemplate::from_file(model_dir.join("chat_template.jinja"))
            .map_err(|e| format!("chat template: {e}"))?;
        let name = tokenizer
            .id_token(spec.image_token_id)
            .ok_or("the tokenizer does not know the image token id")?;
        let merge = image.merge_size;
        let builder = PromptBuilder::new(tokenizer, template, &name, merge)
            .map_err(|e| format!("prompt builder: {e}"))?;
        Ok(Self {
            builder,
            resize: SmartResizeConfig {
                min_pixels: image.min_pixels as u64,
                max_pixels: image.max_pixels as u64,
                ..SmartResizeConfig::default()
            },
            policy: MediaPolicy::default(),
        })
    }

    fn image_input(grid: GridThw, pixels: &[half::bf16]) -> ImageInput {
        let mut bytes = Vec::with_capacity(pixels.len() * 2);
        for p in pixels {
            bytes.extend_from_slice(&p.to_bits().to_le_bytes());
        }
        let grid = CacheGrid {
            t: grid.t,
            h: grid.h,
            w: grid.w,
        };
        ImageInput {
            key: embedding_key(&bytes, grid),
            grid,
            payload: Arc::new(bytes),
        }
    }

    fn assemble(
        &self,
        messages: &[ChatMessage],
        render: &RenderOptions,
        images: Vec<ImageInput>,
        params: SamplingParams,
    ) -> Result<Prepared, EngineError> {
        let prepared = self
            .builder
            .prepare(messages, render)
            .map_err(|e| EngineError::Invalid(e.to_string()))?;
        let grids: Vec<ImageGrid> = images
            .iter()
            .map(|i| ImageGrid {
                t: i.grid.t as usize,
                h: i.grid.h as usize,
                w: i.grid.w as usize,
            })
            .collect();
        let prompt = self
            .builder
            .expand(&prepared, &grids)
            .map_err(|e| EngineError::Invalid(e.to_string()))?;
        Ok(Prepared {
            request: EngineRequest {
                prompt: prompt.token_ids,
                images,
                params,
            },
        })
    }

    /// Decodes every image reference of the request and builds the scheduler request.
    pub fn prepare(&self, request: &GenerateRequest) -> Result<Prepared, EngineError> {
        let mut images = Vec::new();
        for message in &request.messages {
            if let Some(Content::Parts(parts)) = &message.content {
                for part in parts {
                    if let ContentPart::ImageUrl { image_url, .. } = part {
                        let bytes = load_media(&image_url.url, &self.policy)
                            .map_err(|e| EngineError::UnsupportedMedia(e.to_string()))?;
                        let pre = preprocess(&bytes, &self.resize).map_err(|e| match e {
                            scriptor_image::ImageError::TooLarge { .. } => {
                                EngineError::PayloadTooLarge(e.to_string())
                            }
                            other => EngineError::UnsupportedMedia(other.to_string()),
                        })?;
                        images.push(Self::image_input(pre.grid_thw, &pre.pixels));
                    }
                }
            }
        }
        self.assemble(
            &request.messages,
            &request.render,
            images,
            request.sampling.clone(),
        )
    }

    /// Builds the self-test request: a synthetic image and the recognition prompt.
    pub fn self_test_request(&self, max_tokens: u32) -> Result<Prepared, EngineError> {
        let side = scriptor_engine::boot::readiness::SELF_TEST_SIDE;
        let image = RgbImage {
            width: side,
            height: side,
            data: scriptor_engine::boot::readiness::self_test_pixels(),
        };
        let (grid, pixels) = patchify_bf16(&image);
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: Some(Content::Parts(vec![
                ContentPart::image_url("data:image/png;base64,AA==", None),
                ContentPart::text(scriptor_engine::boot::readiness::SELF_TEST_PROMPT),
            ])),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: Some(max_tokens),
            ..SamplingParams::default()
        };
        self.assemble(
            &messages,
            &RenderOptions {
                add_generation_prompt: true,
                ..RenderOptions::default()
            },
            vec![Self::image_input(grid, &pixels)],
            params,
        )
    }
}

enum PrepKind {
    Chat(Box<GenerateRequest>),
    SelfTest(u32),
}

struct PrepJob {
    kind: PrepKind,
    reply: std::sync::mpsc::Sender<Result<Prepared, EngineError>>,
}

struct Inner {
    prep: Mutex<std::sync::mpsc::Sender<PrepJob>>,
    tokenizer: Tokenizer,
    handle: OnceLock<EngineHandle>,
    fatal: scriptor_engine::FatalSlot,
    status: Arc<BootStatus>,
    name: String,
    max_model_len: u64,
    draining: AtomicBool,
}

/// The engine behind the HTTP surface; requests are refused until [`SeatEngine::open`].
#[derive(Clone)]
pub struct SeatEngine {
    inner: Arc<Inner>,
}

impl SeatEngine {
    /// Creates an engine that reports `status` and refuses requests until opened. Request
    /// preparation (template, tokenizer, image decoding) runs on `workers` threads.
    pub fn new(
        model_dir: &Path,
        spec: &ModelSpec,
        image: &ImageProcessorConfig,
        workers: usize,
        status: Arc<BootStatus>,
        name: String,
        max_model_len: u64,
    ) -> Result<Self, String> {
        Frontend::load(model_dir, spec, image)?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| format!("tokenizer: {e}"))?;
        let (tx, rx) = std::sync::mpsc::channel::<PrepJob>();
        let rx = Arc::new(Mutex::new(rx));
        for n in 0..workers.max(1) {
            let rx = Arc::clone(&rx);
            let dir = model_dir.to_path_buf();
            let spec = spec.clone();
            let image = image.clone();
            std::thread::Builder::new()
                .name(format!("scriptor-prep-{n}"))
                .spawn(move || {
                    serve_jobs(
                        &rx,
                        || Frontend::load(&dir, &spec, &image).ok(),
                        |frontend, job: PrepJob| {
                            let result = match job.kind {
                                PrepKind::Chat(req) => frontend.prepare(&req),
                                PrepKind::SelfTest(max) => frontend.self_test_request(max),
                            };
                            let _ = job.reply.send(result);
                        },
                    );
                })
                .map_err(|e| format!("preparation thread: {e}"))?;
        }
        Ok(Self {
            inner: Arc::new(Inner {
                prep: Mutex::new(tx),
                tokenizer,
                handle: OnceLock::new(),
                fatal: scriptor_engine::fatal_slot(),
                status,
                name,
                max_model_len,
                draining: AtomicBool::new(false),
            }),
        })
    }

    fn prepare(&self, kind: PrepKind) -> Result<Prepared, EngineError> {
        let (reply, answer) = std::sync::mpsc::channel();
        self.inner
            .prep
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .send(PrepJob { kind, reply })
            .map_err(|_| EngineError::Internal("request preparation is unavailable".into()))?;
        answer.recv().map_err(|_| {
            EngineError::Internal("request preparation failed; the worker was restarted".into())
        })?
    }

    /// Builds the boot self-test request.
    pub fn self_test_request(&self, max_tokens: u32) -> Result<Prepared, EngineError> {
        self.prepare(PrepKind::SelfTest(max_tokens))
    }

    /// Stops the scheduler thread when the engine was opened; returns whether it ended
    /// without a panic.
    pub fn shutdown(&self) -> bool {
        self.inner.handle.get().is_none_or(EngineHandle::shutdown)
    }

    /// Starts draining: health reports `draining` and new requests are refused; running
    /// requests continue.
    pub fn begin_drain(&self) {
        self.inner.draining.store(true, Ordering::Release);
        if let Some(h) = self.inner.handle.get() {
            h.begin_drain();
        }
    }

    /// Slot the engine thread publishes a sticky device fault to; pass it to
    /// [`EngineHandle::spawn_with`].
    pub fn fatal_slot(&self) -> scriptor_engine::FatalSlot {
        Arc::clone(&self.inner.fatal)
    }

    /// Resolves with the sticky device fault that stopped the engine, never when none occurs.
    pub async fn wait_fatal(&self) -> scriptor_engine::EngineError {
        let mut rx = self.inner.fatal.subscribe();
        loop {
            if let Some(err) = rx.borrow_and_update().clone() {
                return err;
            }
            if rx.changed().await.is_err() {
                std::future::pending::<()>().await;
            }
        }
    }

    /// Cancels all running and waiting requests with the drain reason.
    pub fn interrupt(&self) {
        self.inner.draining.store(true, Ordering::Release);
        if let Some(h) = self.inner.handle.get() {
            h.interrupt();
        }
    }

    /// Time the drain may last: see [`EngineHandle::drain_deadline`]; zero before the engine
    /// opens, when nothing is in flight.
    pub fn drain_deadline(&self, bound: Option<Duration>) -> Duration {
        self.inner
            .handle
            .get()
            .map_or(Duration::ZERO, |h| h.drain_deadline(bound))
    }

    /// Starts accepting requests on `handle`.
    pub fn open(&self, handle: EngineHandle) {
        let _ = self.inner.handle.set(handle);
    }
}

fn map_engine_error(e: scriptor_engine::EngineError) -> EngineError {
    use scriptor_engine::EngineError as E;
    match e {
        E::Closed => EngineError::NotReady,
        E::Draining => EngineError::Draining,
        E::Pressure { retry_after_secs } => EngineError::QueueFull {
            retry_after: retry_after_secs,
        },
        #[cfg(feature = "kv-demand-diagnostic")]
        E::QueueFull { retry_after_secs } => EngineError::QueueFull {
            retry_after: retry_after_secs,
        },
        E::ContextOverflow(m) => EngineError::Invalid(m),
        E::ImageTooLarge { .. }
        | E::EmbeddingBudget { .. }
        | E::InvalidImage { .. }
        | E::PlaceholderMismatch { .. }
        | E::EmptyPrompt
        | E::PromptExceedsCache { .. } => EngineError::Invalid(e.to_string()),
        other => EngineError::Internal(other.to_string()),
    }
}

/// The stream event for an engine failure: a capacity refusal keeps its retry delay so the
/// server answers it as a throttle, any other failure carries its message.
fn failure_event(e: scriptor_engine::EngineError) -> GenEvent {
    match map_engine_error(e.clone()) {
        EngineError::QueueFull { retry_after } => GenEvent::QueueFull { retry_after },
        _ => GenEvent::Failed(e.to_string()),
    }
}

impl Engine for SeatEngine {
    fn served_name(&self) -> &str {
        &self.inner.name
    }

    fn max_model_len(&self) -> u64 {
        self.inner.max_model_len
    }

    fn lifecycle(&self) -> Lifecycle {
        if let Some(reason) = self.inner.handle.get().and_then(EngineHandle::dead_reason) {
            Lifecycle::Dead(reason)
        } else if self.inner.draining.load(Ordering::Acquire) {
            Lifecycle::Draining
        } else {
            Lifecycle::Running
        }
    }

    fn health(&self) -> Vec<Stage> {
        self.inner
            .status
            .snapshot()
            .into_iter()
            .map(|r| {
                Stage::new(
                    r.name,
                    matches!(r.state, StageState::Done | StageState::Skipped),
                )
            })
            .collect()
    }

    fn dead(&self) -> Option<scriptor_server::Dead> {
        match self.inner.fatal.borrow().as_ref()? {
            scriptor_engine::EngineError::Fatal { code, message } => Some(scriptor_server::Dead {
                cuda: Some(*code),
                message: message.clone(),
            }),
            other => Some(scriptor_server::Dead {
                cuda: None,
                message: other.to_string(),
            }),
        }
    }

    fn stats(&self) -> EngineStats {
        let boot_stage_seconds = self
            .inner
            .status
            .snapshot()
            .into_iter()
            .filter(|r| matches!(r.state, StageState::Done))
            .map(|r| (r.name.to_owned(), r.seconds))
            .collect();
        let lifecycle = self.lifecycle();
        let Some(handle) = self.inner.handle.get() else {
            return EngineStats {
                boot_stage_seconds,
                lifecycle,
                ..EngineStats::default()
            };
        };
        let m = handle.metrics();
        let beat = handle.heartbeat();
        let pressure = handle.pressure().snapshot();
        let phase = phase_stats(&m);
        EngineStats {
            pressure_tier: pressure.tier as u8,
            host_mem_available_bytes: pressure.host_mem_available_bytes,
            host_psi_memory_full_avg10: pressure.psi_full_avg10,
            load_shed_total: pressure.shed_actions,
            load_protect_total: pressure.protect_actions,
            admission_rejected_pressure: pressure.admission_rejected,
            requests_shed_total: m.shed,
            running: m.running as u64,
            waiting: m.waiting as u64,
            kv_cache_usage: m.pool.usage(),
            slots: m.slots as u64,
            kv_reserved_perc: if m.pool.total_blocks == 0 {
                0.0
            } else {
                m.reserved_blocks as f64 / m.pool.total_blocks as f64
            },
            kv_cache_tokens: m.kv_tokens as u64,
            spec_drafts: m.spec_drafts,
            spec_draft_tokens: m.spec_draft_tokens,
            spec_accepted_tokens: m.spec_accepted_tokens,
            spec_accepted_per_pos: m.spec_accepted_per_pos,
            graphs_captured: m.graphs_captured as u64,
            graph_replays: m.graph_replays,
            graph_split_replays: m.graph_split_replays,
            graph_fallbacks: m.graph_fallbacks,
            slot_occupancy: m.slot_occupancy,
            l1: m.l1,
            embedding_cache_hits: m.embedding_cache.hits,
            preemptions: m.pool.preemptions + m.restarts,
            preemptions_recompute: m.pool.preemptions,
            preemptions_restart: m.restarts,
            preemptions_yield: m.yields,
            restart_discarded_tokens: m.restart_discarded_tokens,
            kv_demand: m.kv_policy != scriptor_spec::KvPolicy::Reserve,
            kv_used_blocks: m.pool.used_blocks as u64,
            kv_step_growth_blocks: m.step_growth_blocks as u64,
            accepted_requests: m.accepted as u64,
            pinned_embedding_bytes: m.pinned_embedding_bytes as u64,
            retained_embedding_bytes: m.retained_embedding_bytes as u64,
            admission_blocked: [
                m.admission_blocked.slots,
                m.admission_blocked.watermark,
                m.admission_blocked.hysteresis,
                m.admission_blocked.prompt_blocks,
                m.admission_blocked.accepted,
                m.admission_blocked.embedding,
            ],
            requests_rejected_queue: m.rejected_queue,
            boot_stage_seconds,
            vision_batch_sizes: m.vision_batch_sizes,
            lifecycle,
            heartbeat_beats: beat.beats,
            heartbeat_age_seconds: beat.age.as_secs_f64(),
            vision_batch_patch_sizes: m.vision_batch_patch_sizes,
            prefill_pass_token_sizes: m.prefill_pass_stats.sizes,
            prefill_pass_tokens: m.prefill_pass_stats.tokens,
            prefill_passes_below_ridge: m.prefill_pass_stats.below_ridge_passes,
            prefill_pass_tokens_below_ridge: m.prefill_pass_stats.below_ridge_tokens,
            vision_images: m.vision_images,
            vision_patches: m.vision_patches,
            active_vision: m.active_vision as u64,
            active_prefill: m.active_prefill as u64,
            active_running: m.active_running as u64,
            embedding_released: m.embedding_released,
            embedding_released_bytes: m.embedding_released_bytes,
            images_submitted: m.images.images,
            image_patches_submitted: m.images.patches,
            image_patch_sizes: m.images.patch_sizes,
            image_key_unique: m.images.unique,
            image_key_repeat: m.images.repeat,
            image_key_repeat_patches: m.images.repeat_patches,
            image_key_untracked: m.images.untracked,
            ..phase
        }
    }

    #[cfg(feature = "kv-demand-diagnostic")]
    fn refuses_streaming(&self) -> bool {
        self.inner
            .handle
            .get()
            .is_some_and(|h| h.kv_policy() == scriptor_spec::KvPolicy::DemandDiagnostic)
    }

    fn submit(&self, request: GenerateRequest) -> Result<mpsc::Receiver<GenEvent>, EngineError> {
        match self.lifecycle() {
            Lifecycle::Running => {}
            Lifecycle::Draining => return Err(EngineError::Draining),
            Lifecycle::Dead(reason) => return Err(EngineError::Dead(reason)),
        }
        let Some(handle) = self.inner.handle.get() else {
            return Err(EngineError::NotReady);
        };
        #[cfg(feature = "kv-demand-diagnostic")]
        let restartable = request.restartable;
        let stops = request.sampling.stop.clone();
        let stop_on_repetition = request.sampling.stop_on_repetition;
        let max_tokens = request.sampling.max_tokens.map(u64::from);
        let prepared = self.prepare(PrepKind::Chat(Box::new(request)))?;
        let prompt_tokens = prepared.request.prompt.len() as u64;
        let overflow = match max_tokens {
            Some(out) => prompt_tokens + out > self.inner.max_model_len,
            None => prompt_tokens >= self.inner.max_model_len,
        };
        if overflow {
            return Err(EngineError::ContextOverflow {
                prompt_tokens,
                max_tokens,
                max_model_len: self.inner.max_model_len,
            });
        }
        #[cfg(feature = "kv-demand-diagnostic")]
        let stream = if restartable {
            handle.submit_restartable(prepared.request)
        } else {
            handle.submit(prepared.request)
        }
        .map_err(map_engine_error)?;
        #[cfg(not(feature = "kv-demand-diagnostic"))]
        let stream = handle.submit(prepared.request).map_err(map_engine_error)?;
        let (tx, rx) = mpsc::channel(64);
        let inner = Arc::clone(&self.inner);
        let handle = handle.clone();
        tokio::spawn(async move {
            relay(
                inner,
                handle,
                stream,
                stops,
                stop_on_repetition,
                prompt_tokens,
                tx,
            )
            .await;
        });
        Ok(rx)
    }
}

/// Moves engine events to the client stream as decoded text, applying stop strings.
async fn relay(
    inner: Arc<Inner>,
    handle: EngineHandle,
    mut stream: scriptor_engine::RequestStream,
    stops: Vec<String>,
    stop_on_repetition: bool,
    prompt_tokens: u64,
    tx: mpsc::Sender<GenEvent>,
) {
    let mut repeat = stop_on_repetition.then(RepeatTail::new);
    let mut decoder = inner.tokenizer.incremental(true);
    let hold = stops.iter().map(String::len).max().unwrap_or(0);
    let mut pending = String::new();
    let mut produced = 0u64;
    let id = stream.id;
    loop {
        let event = match next_event(&mut stream.events, &tx).await {
            Next::Event(e) => e,
            Next::Ended => return,
            Next::ConsumerGone => {
                handle.cancel(id);
                return;
            }
        };
        match event {
            EngineEvent::Token(t) => {
                produced += 1;
                let piece = match decoder.push(t.token) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = tx.send(GenEvent::Failed(e.to_string())).await;
                        handle.cancel(id);
                        return;
                    }
                };
                pending.push_str(&piece);
                if let Some(at) = find_stop(&pending, 0, &stops) {
                    let head = pending[..at].to_owned();
                    if !head.is_empty() && tx.send(GenEvent::Delta(head)).await.is_err() {
                        handle.cancel(id);
                        return;
                    }
                    let _ = tx
                        .send(GenEvent::Finished {
                            finish_reason: "stop".into(),
                            stop_reason: None,
                            prompt_tokens,
                            completion_tokens: produced,
                        })
                        .await;
                    handle.cancel(id);
                    return;
                }
                if repeat.as_mut().is_some_and(|r| r.push_str(&piece)) {
                    if !pending.is_empty()
                        && tx
                            .send(GenEvent::Delta(std::mem::take(&mut pending)))
                            .await
                            .is_err()
                    {
                        handle.cancel(id);
                        return;
                    }
                    let _ = tx
                        .send(GenEvent::Finished {
                            finish_reason: "stop".into(),
                            stop_reason: Some("repetition".into()),
                            prompt_tokens,
                            completion_tokens: produced,
                        })
                        .await;
                    handle.cancel(id);
                    return;
                }
                let mut cut = pending.len().saturating_sub(hold);
                while !pending.is_char_boundary(cut) {
                    cut -= 1;
                }
                if cut > 0 {
                    let out: String = pending.drain(..cut).collect();
                    if tx.send(GenEvent::Delta(out)).await.is_err() {
                        handle.cancel(id);
                        return;
                    }
                }
            }
            EngineEvent::Finished(done) => {
                pending.push_str(&decoder.finish().unwrap_or_default());
                if !pending.is_empty() {
                    let _ = tx.send(GenEvent::Delta(std::mem::take(&mut pending))).await;
                }
                let reason = match done.reason {
                    FinishReason::Stop => "stop",
                    FinishReason::Length => "length",
                };
                let _ = tx
                    .send(GenEvent::Finished {
                        finish_reason: reason.into(),
                        stop_reason: None,
                        prompt_tokens: done.prompt_tokens as u64,
                        completion_tokens: done.completion_tokens as u64,
                    })
                    .await;
                return;
            }
            #[cfg(feature = "kv-demand-diagnostic")]
            EngineEvent::Restarted => {
                decoder = inner.tokenizer.incremental(true);
                pending.clear();
                produced = 0;
                repeat = stop_on_repetition.then(RepeatTail::new);
                if tx.send(GenEvent::Restarted).await.is_err() {
                    handle.cancel(id);
                    return;
                }
            }
            EngineEvent::Aborted => {
                let _ = tx
                    .send(GenEvent::Cancelled {
                        reason: "client".into(),
                    })
                    .await;
                return;
            }
            EngineEvent::Cancelled(reason) => {
                let _ = tx
                    .send(GenEvent::Cancelled {
                        reason: reason.label().into(),
                    })
                    .await;
                return;
            }
            EngineEvent::Failed(e) => {
                let _ = tx.send(failure_event(e)).await;
                return;
            }
        }
    }
}

#[cfg(all(test, feature = "kv-demand-diagnostic"))]
mod failure_event_tests {
    use super::*;

    #[test]
    fn an_intake_refusal_keeps_its_retry_delay() {
        let refused = scriptor_engine::EngineError::QueueFull {
            retry_after_secs: 4,
        };
        assert_eq!(
            failure_event(refused),
            GenEvent::QueueFull { retry_after: 4 }
        );
    }

    #[test]
    fn other_failures_carry_their_message() {
        let e = scriptor_engine::EngineError::Cache("full".into());
        assert_eq!(failure_event(e.clone()), GenEvent::Failed(e.to_string()));
    }
}
/// Server stats carrying the scheduler's cumulative phase seconds. `decode_prep_seconds` is
/// reported separately from `decode_seconds` and is never added to it.
fn phase_stats(m: &scriptor_engine::scheduler::EngineMetrics) -> EngineStats {
    EngineStats {
        prefill_seconds: m.prefill_seconds,
        decode_seconds: m.decode_seconds,
        decode_prep_seconds: m.decode_prep_seconds,
        vision_seconds: m.vision_seconds,
        vision_host_seconds: m.vision_host_seconds,
        ..EngineStats::default()
    }
}

#[cfg(test)]
mod phase_stats_tests {
    use super::*;

    #[test]
    fn decode_prep_seconds_maps_exactly_and_apart_from_decode() {
        let m = scriptor_engine::scheduler::EngineMetrics {
            decode_seconds: 3.5,
            decode_prep_seconds: 0.25,
            ..Default::default()
        };
        let stats = phase_stats(&m);
        assert_eq!(stats.decode_prep_seconds, 0.25);
        assert_eq!(stats.decode_seconds, 3.5);
    }
}
