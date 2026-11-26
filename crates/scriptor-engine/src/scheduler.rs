//! Step scheduler: admission against concurrency and the KV pool, then per step the vision,
//! prefill and decode stages over the executor boundary.

use crate::error::EngineError;
use crate::sampling::{FinishReason, Sampled, SequenceSampler};
use crate::sequence::{
    CancelReason, EngineEvent, EngineRequest, FinishedEvent, Phase, RequestId, RequestTimings,
    Sequence, TokenEvent,
};
use crate::vision_queue::SharedEmbedding;
use scriptor_cache::{
    BlockId, BlockPool, DeviceBuffer, EmbeddingCache, EmbeddingCacheStats, ImageGrid, KvLayout,
    PoolStats,
};
use scriptor_kernels::sampling::{RowParams, TokenMasks};
use scriptor_spec::KvPolicy;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

/// Runtime limits and token ids the scheduler works with.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Sequences admitted at once.
    pub max_concurrency: usize,
    /// Tokens a sequence may occupy (prompt plus generated).
    pub max_context: usize,
    /// Patches one vision batch may hold; also the largest admissible single image.
    pub vision_batch_patches: usize,
    /// Whether vision batches run concurrently with prefill and decode on executors that
    /// support it.
    pub vision_overlap: bool,
    /// Whether a sequence's image embeddings and their embedding-budget charge are released
    /// once its prefill completes. A sequence prefilled again re-resolves its images from the
    /// embedding cache, or embeds them anew when the cache no longer holds them.
    pub embedding_release_after_prefill: bool,
    /// Side of the patch merge window (patches per token = `spatial_merge`²).
    pub spatial_merge: usize,
    /// Vocabulary size of the logits rows.
    pub vocab_size: usize,
    /// Ids that end a sequence.
    pub eos_token_ids: Vec<u32>,
    /// Id of the image placeholder token.
    pub image_token_id: u32,
    /// Byte budget of the image-embedding cache; zero disables retention.
    pub embedding_cache_bytes: usize,
    /// Embedding width in elements.
    pub embedding_hidden: usize,
    /// Bytes per embedding element.
    pub embedding_elem_bytes: usize,
    /// Seed for requests that carry none; offset by the request id.
    pub fallback_seed: u64,
}

/// Names a vision batch started with [`ModelExecutor::vision_begin`].
pub type VisionToken = u64;

/// One image to embed.
#[derive(Debug, Clone)]
pub struct VisionJob {
    pub seq: RequestId,
    /// Index of the image within its request.
    pub image: usize,
    pub key: scriptor_cache::EmbeddingKey,
    pub grid: ImageGrid,
    pub patches: usize,
    pub payload: Arc<Vec<u8>>,
}

/// One sequence to prefill: all tokens from position zero, with the KV blocks to fill.
pub struct PrefillRow<B> {
    pub seq: RequestId,
    pub tokens: Vec<u32>,
    /// Embeddings of the sequence's images in prompt order.
    pub images: Vec<Arc<B>>,
    pub blocks: Vec<BlockId>,
    /// Whether the executor returns the logits of the row's last token. When false the row
    /// continues on the device-resident step: the executor fills its KV (and draft state),
    /// returns an empty logits row, and the step's first row is the last token of `tokens`.
    pub logits: bool,
}

/// One sequence to advance by a token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeRow {
    pub seq: RequestId,
    /// Last accepted token, fed as input.
    pub token: u32,
    /// Position of `token` in the sequence.
    pub position: usize,
    pub blocks: Vec<BlockId>,
}

/// One sequence of a device-resident decode step: the row to advance plus the sampling state the
/// device needs the first time it sees the sequence.
#[derive(Debug, Clone, Copy)]
pub struct DeviceRow<'a> {
    pub row: &'a DecodeRow,
    /// Sampling parameters of the next token; `step` is the tokens generated so far.
    pub params: RowParams,
    /// The sequence's prompt and output membership bitsets and counts, uploaded when the
    /// executor starts tracking the sequence.
    pub masks: &'a TokenMasks,
}

/// Tokens a sequence emitted in one device-resident step with the top-2 margin of each.
#[derive(Debug, Clone, PartialEq)]
pub struct StepRowResult {
    /// Emitted token ids in order: the accepted drafts, then one verifier token.
    pub tokens: Vec<u32>,
    /// Best minus second-best penalised logit at each emitted token.
    pub margins: Vec<f32>,
}

/// Graph counters an executor reports for the metrics endpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphStats {
    /// Graphs held in device memory.
    pub captured: usize,
    /// Decode steps served by replaying a graph.
    pub replays: u64,
    /// Decode steps served by replaying a graph that uses the two-stage target selector.
    pub split_replays: u64,
    /// Decode steps that ran eagerly because no graph covers their shape.
    pub fallbacks: u64,
    /// Slot occupancy of the device-resident decode steps.
    pub occupancy: crate::SlotOccupancy,
}

/// The model side of the scheduler: vision tower, prefill and decode over paged KV.
pub trait ModelExecutor: Send {
    /// Embedding buffer type produced by the vision tower.
    type Embedding: DeviceBuffer + Send + Sync + 'static;

    /// Embeds the images of one vision batch; one buffer per job, in job order, holding
    /// `patches / merge²` rows of the embedding width.
    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<Self::Embedding>, EngineError>;

    /// Patches per vision batch from which the tower's throughput no longer improves with batch
    /// size; zero means batches always run as soon as they are pending.
    fn vision_saturation_patches(&self) -> usize {
        0
    }

    /// Tokens per prefill pass from which the text model's matrix multiplications fill the
    /// device; zero means prefill rows always run as soon as their embeddings are ready.
    fn prefill_saturation_tokens(&self) -> usize {
        0
    }

    /// Lets the executor build the host-side inputs of a vision batch the scheduler expects to
    /// run next, while the device is busy. A later [`ModelExecutor::vision`] or
    /// [`ModelExecutor::vision_begin`] over the same jobs may reuse the result.
    fn vision_prepare(&mut self, _jobs: &[VisionJob]) -> Result<(), EngineError> {
        Ok(())
    }

    /// Whether [`ModelExecutor::vision_begin`] runs a batch concurrently with the other stages.
    fn supports_async_vision(&self) -> bool {
        false
    }

    /// Starts embedding a vision batch without waiting for it and returns the token naming it.
    /// Only called when [`ModelExecutor::supports_async_vision`] is true; at most one batch is
    /// outstanding.
    fn vision_begin(&mut self, _jobs: &[VisionJob]) -> Result<VisionToken, EngineError> {
        Err(EngineError::Executor(
            "the executor does not support asynchronous vision".into(),
        ))
    }

    /// Whether the batch named by `token` has finished.
    fn vision_ready(&mut self, _token: VisionToken) -> Result<bool, EngineError> {
        Ok(true)
    }

    /// Waits for the batch named by `token` and returns its embeddings in job order.
    fn vision_finish(&mut self, _token: VisionToken) -> Result<Vec<Self::Embedding>, EngineError> {
        Err(EngineError::Executor(
            "the executor does not support asynchronous vision".into(),
        ))
    }

    /// Prefills the rows and returns the fp32 logits of each row's last token; a row without
    /// [`PrefillRow::logits`] yields an empty row.
    fn prefill(
        &mut self,
        rows: &[PrefillRow<Self::Embedding>],
    ) -> Result<Vec<Vec<f32>>, EngineError>;

    /// Advances the rows by one token and returns each row's fp32 logits.
    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError>;

    /// Whether [`ModelExecutor::decode_device`] is available.
    fn supports_device_step(&self) -> bool {
        false
    }

    /// Advances the rows by one device-resident step: draft, verify, accept and sample entirely
    /// on the device, returning only the emitted tokens of each row. Only called when
    /// [`ModelExecutor::supports_device_step`] is true.
    fn decode_device(
        &mut self,
        _rows: &[DeviceRow<'_>],
    ) -> Result<Vec<StepRowResult>, EngineError> {
        Err(EngineError::Executor(
            "the executor does not support device-resident decoding".into(),
        ))
    }

    /// Enqueues a device-resident step without waiting for its tokens and returns whether it
    /// did; [`ModelExecutor::decode_device_finish`] then returns the step's results. When false
    /// nothing was enqueued and the caller runs [`ModelExecutor::decode_device`] instead.
    fn decode_device_begin(&mut self, _rows: &[DeviceRow<'_>]) -> Result<bool, EngineError> {
        Ok(false)
    }

    /// Waits for the step enqueued by [`ModelExecutor::decode_device_begin`] and returns each
    /// row's emitted tokens in row order.
    fn decode_device_finish(&mut self) -> Result<Vec<StepRowResult>, EngineError> {
        Err(EngineError::Executor(
            "no device-resident step is in flight".into(),
        ))
    }

    /// Host timing of the step last collected by [`ModelExecutor::decode_device_finish`]; taken
    /// once. Executors that do not measure it return `None`.
    fn take_host_sample(&mut self) -> Option<crate::l1_counters::StepHostSample> {
        None
    }

    /// Device and host timing of the finished vision batch named by `token`; taken once.
    /// Executors that do not measure it return `None`.
    fn take_vision_timing(
        &mut self,
        _token: VisionToken,
    ) -> Option<crate::l1_counters::VisionTiming> {
        None
    }

    /// Tokens past a sequence's current length the executor may write KV for in one decode
    /// step; at least [`ModelExecutor::draft_depth`].
    fn lookahead_tokens(&self) -> usize {
        self.draft_depth()
    }

    /// Graph counters of the executor.
    fn graph_stats(&self) -> GraphStats {
        GraphStats::default()
    }

    /// Drops executor-side state of a sequence that finished, failed, was cancelled or evicted.
    fn release(&mut self, _seq: RequestId) {}

    /// Returns device memory the executor holds beyond its plan to the system; called when the
    /// host is under memory pressure.
    fn trim_memory(&mut self) {}

    /// Draft tokens proposed per sequence per decode step; zero selects plain decoding.
    fn draft_depth(&self) -> usize {
        0
    }

    /// Advances the rows by one speculative step: drafts [`ModelExecutor::draft_depth`] tokens
    /// per row, verifies them in one pass, and returns each row's drafts with the fp32 logits
    /// of the `drafts + 1` verified positions. Only called when the depth is non-zero.
    fn decode_spec(&mut self, _rows: &[DecodeRow]) -> Result<Vec<SpecRowOutput>, EngineError> {
        Err(EngineError::Executor(
            "the executor does not support speculative decoding".into(),
        ))
    }
}

/// Drafts of one sequence and the verifier logits at the positions they cover.
#[derive(Debug, Clone)]
pub struct SpecRowOutput {
    /// Draft tokens, in order.
    pub drafts: Vec<u32>,
    /// fp32 logits at the last accepted token and after each draft: `drafts.len() + 1` rows.
    pub logits: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Counters {
    pub(crate) submitted: u64,
    pub(crate) finished: u64,
    pub(crate) aborted: u64,
    pub(crate) failed: u64,
    pub(crate) shed: u64,
    pub(crate) prompt_tokens: u64,
    pub(crate) generation_tokens: u64,
    pub(crate) steps: u64,
    pub(crate) vision_batches: u64,
    pub(crate) vision_images: u64,
    pub(crate) vision_patches: u64,
    pub(crate) vision_seconds: f64,
    pub(crate) vision_host_seconds: f64,
    pub(crate) prefill_passes: u64,
    pub(crate) prefill_seconds: f64,
    pub(crate) decode_steps: u64,
    pub(crate) decode_seconds: f64,
    pub(crate) decode_prep_seconds: f64,
    pub(crate) queue_seconds: f64,
    pub(crate) spec_drafts: u64,
    pub(crate) spec_draft_tokens: u64,
    pub(crate) spec_accepted_tokens: u64,
    pub(crate) embedding_released: u64,
    pub(crate) embedding_released_bytes: u64,
    pub(crate) restarts: u64,
    pub(crate) yields: u64,
    pub(crate) restart_discarded_tokens: u64,
    pub(crate) rejected_queue: u64,
    pub(crate) blocked: AdmissionBlocked,
}

/// Scheduling steps in which the front of the waiting queue could not be admitted, or an
/// embedding reservation could not be made, by the first condition that stopped it. The
/// reserve policy leaves every field but `embedding` at zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AdmissionBlocked {
    /// Every slot was taken.
    pub slots: u64,
    /// The free blocks left after the front's first step did not cover the running sequences'
    /// next-step growth plus the margin, or the reservation guard for non-restartable
    /// sequences refused it.
    pub watermark: u64,
    /// As `watermark`, and the doubled margin of the step after an eviction was the only
    /// reason it failed.
    pub hysteresis: u64,
    /// The front's prompt and first-step blocks were not free.
    pub prompt_blocks: u64,
    /// The accepted-request bound was reached.
    pub accepted: u64,
    /// The embedding budget could not take the oldest unreserved request's embeddings.
    pub embedding: u64,
}

/// Snapshot of scheduler gauges and lifetime counters for the metrics endpoint.
#[derive(Debug, Clone, Default)]
pub struct EngineMetrics {
    /// Sequences admitted (holding blocks).
    pub running: usize,
    /// Output tokens the waiting and running requests may still generate.
    pub tokens_in_flight: u64,
    /// Slots the scheduler admits into.
    pub slots: usize,
    /// Blocks promised to running sequences.
    pub reserved_blocks: usize,
    /// Tokens the KV pool holds.
    pub kv_tokens: usize,
    /// Sequences waiting for admission, including evicted ones.
    pub waiting: usize,
    /// Admitted sequences waiting for their image embeddings.
    pub active_vision: usize,
    /// Admitted sequences waiting for or in a prefill pass.
    pub active_prefill: usize,
    /// Admitted sequences decoding.
    pub active_running: usize,
    /// Image buffers released after prefill under `embedding_release_after_prefill`.
    pub embedding_released: u64,
    /// Device bytes of the released image buffers.
    pub embedding_released_bytes: u64,
    /// Per-image patch counts and duplicate-key counters of the submitted images.
    pub images: crate::image_ledger::ImageLedgerStats,
    /// Waiting sequences that were never admitted.
    #[cfg(feature = "kv-demand-diagnostic")]
    pub waiting_new: usize,
    /// Policy the scheduler commits KV blocks under.
    pub kv_policy: KvPolicy,
    /// Requests the engine has taken on: running plus evicted ones awaiting re-admission.
    pub accepted: usize,
    /// Blocks the running sequences needed to cover their next decode step at the last
    /// pre-step phase; zero under the reserve policy.
    pub step_growth_blocks: usize,
    /// Device bytes of image embeddings held by live work, counting shared buffers once.
    pub pinned_embedding_bytes: usize,
    /// Device bytes of image embeddings: pinned plus cache-only entries.
    pub retained_embedding_bytes: usize,
    /// Scheduling steps that could not admit or reserve, by reason.
    pub admission_blocked: AdmissionBlocked,
    /// Sequences evicted and restarted from their prompt.
    pub restarts: u64,
    /// Sequences still in the vision or prefill phase returned to the waiting queue to free
    /// their blocks, with nothing generated lost.
    pub yields: u64,
    /// Generated tokens discarded by restarts.
    pub restart_discarded_tokens: u64,
    /// Submissions the scheduler refused because the waiting queue was full; the handle's own
    /// refusals are added by [`crate::handle::EngineHandle::metrics`].
    pub rejected_queue: u64,
    pub submitted: u64,
    pub finished: u64,
    pub aborted: u64,
    pub failed: u64,
    /// Requests cancelled to relieve host memory pressure.
    pub shed: u64,
    pub prompt_tokens: u64,
    pub generation_tokens: u64,
    pub steps: u64,
    pub vision_batches: u64,
    pub vision_images: u64,
    pub vision_patches: u64,
    pub vision_seconds: f64,
    /// Engine-thread seconds spent submitting or running vision batches.
    pub vision_host_seconds: f64,
    /// Vision batches by image count: `(images, batches)` ascending.
    pub vision_batch_sizes: Vec<(u64, u64)>,
    /// Vision batches by patch count rounded up to a power of two: `(upper bound, batches)`
    /// ascending.
    pub vision_batch_patch_sizes: Vec<(u64, u64)>,
    pub prefill_passes: u64,
    /// Prefill pass sizes: power-of-two histogram and token totals.
    pub prefill_pass_stats: crate::pass_ledger::PassStats,
    pub prefill_seconds: f64,
    pub decode_steps: u64,
    pub decode_seconds: f64,
    /// Engine-thread seconds spent preparing decode steps (sequence selection, KV
    /// reservation, row assembly), outside `decode_seconds`.
    pub decode_prep_seconds: f64,
    /// Summed submission-to-admission time of admitted requests.
    pub queue_seconds: f64,
    /// Speculative verification steps summed over sequences.
    pub spec_drafts: u64,
    /// Draft tokens proposed.
    pub spec_draft_tokens: u64,
    /// Draft tokens accepted.
    pub spec_accepted_tokens: u64,
    /// Draft tokens accepted at each draft position.
    pub spec_accepted_per_pos: Vec<u64>,
    /// Graphs the executor holds in device memory.
    pub graphs_captured: usize,
    /// Decode steps served by graph replay.
    pub graph_replays: u64,
    /// Decode steps served by replaying a graph that uses the two-stage target selector.
    pub graph_split_replays: u64,
    /// Decode steps that ran eagerly for lack of a covering graph.
    pub graph_fallbacks: u64,
    /// Slot occupancy of the device-resident decode steps.
    pub slot_occupancy: crate::SlotOccupancy,
    /// Host-span and vision-flight counters; `None` unless `SCRIPTOR_L1_COUNTERS=1`.
    pub l1: Option<crate::l1_counters::L1Stats>,
    pub pool: PoolStats,
    pub embedding_cache: EmbeddingCacheStats,
}

/// What one scheduler step did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepReport {
    pub admitted: usize,
    pub vision_images: usize,
    pub prefill_rows: usize,
    pub decode_rows: usize,
    pub finished: usize,
    pub preempted: usize,
}

/// vLLM-worded refusal text when a prompt (plus requested output) exceeds the context.
fn overflow_message(max_context: usize, prompt: usize, max_tokens: Option<u32>) -> String {
    match max_tokens {
        Some(out) => format!(
            "This model's maximum context length is {max_context} tokens. However, you requested \
             {out} output tokens and your prompt contains at least {prompt} input tokens, for a \
             total of at least {} tokens. Please reduce the length of the input prompt or the \
             number of requested output tokens.",
            prompt + out as usize
        ),
        None => format!(
            "This model's maximum context length is {max_context} tokens. However, your request \
             has {prompt} input tokens. Please reduce the length of the input messages."
        ),
    }
}

/// Checks a request against the context window, the image budget, the placeholder count and
/// the KV pool capacity; the error text is the user-facing refusal.
pub fn validate_request(
    config: &SchedulerConfig,
    layout: &KvLayout,
    req: &EngineRequest,
) -> Result<(), EngineError> {
    if req.prompt.is_empty() {
        return Err(EngineError::EmptyPrompt);
    }
    let group = config.spatial_merge * config.spatial_merge;
    let mut expected = 0usize;
    for img in &req.images {
        let patches = img.patches();
        if patches == 0 || patches % group != 0 {
            return Err(EngineError::InvalidImage { patches, group });
        }
        if patches > config.vision_batch_patches {
            return Err(EngineError::ImageTooLarge {
                patches,
                budget: config.vision_batch_patches,
            });
        }
        expected += patches / group;
    }
    if let Some(budget) = crate::embedding_budget::embedding_bound(config) {
        let required: usize = req
            .images
            .iter()
            .map(|i| crate::embedding_budget::embedding_bytes(config, i.patches()))
            .sum();
        if required > budget {
            return Err(EngineError::EmbeddingBudget { required, budget });
        }
    }
    let placeholders = req
        .prompt
        .iter()
        .filter(|&&t| t == config.image_token_id)
        .count();
    if placeholders != expected {
        return Err(EngineError::PlaceholderMismatch {
            placeholders,
            expected,
        });
    }
    let prompt = req.prompt.len();
    let out = req.params.max_tokens.map_or(0, |m| m as usize);
    if prompt + out > config.max_context
        || (req.params.max_tokens.is_none() && prompt >= config.max_context)
    {
        return Err(EngineError::ContextOverflow(overflow_message(
            config.max_context,
            prompt,
            req.params.max_tokens,
        )));
    }
    if layout.blocks_for_tokens(prompt) > layout.num_blocks {
        return Err(EngineError::PromptExceedsCache {
            blocks: layout.num_blocks,
            tokens: prompt,
        });
    }
    Ok(())
}

/// The scheduler core; driven by calling [`Scheduler::step`].
pub struct Scheduler<E: ModelExecutor> {
    pub(crate) config: SchedulerConfig,
    pub(crate) executor: E,
    pub(crate) pool: BlockPool,
    pub(crate) embedding_cache: EmbeddingCache<SharedEmbedding<E::Embedding>>,
    pub(crate) waiting: VecDeque<Sequence<E::Embedding>>,
    pub(crate) active: BTreeMap<RequestId, Sequence<E::Embedding>>,
    pub(crate) counters: Counters,
    pub(crate) disconnected: Vec<RequestId>,
    pub(crate) spec_accepted_per_pos: Vec<u64>,
    pub(crate) pass_ledger: crate::pass_ledger::PassLedger,
    pub(crate) spec_log: crate::spec_log::SpecLog,
    pub(crate) l1: crate::l1_counters::L1Ledger,
    pub(crate) vision_flight: Option<crate::vision_queue::VisionFlight>,
    pub(crate) embedding_reserved: std::collections::BTreeSet<RequestId>,
    pub(crate) image_ledger: crate::image_ledger::ImageLedger,
    pub(crate) vision_sizes: BTreeMap<u64, u64>,
    pub(crate) vision_patch_sizes: BTreeMap<u64, u64>,
    pub(crate) vision_deferred_since: Option<Instant>,
    pub(crate) prefill_deferred_since: Option<Instant>,
    pub(crate) staged_vision: Vec<(RequestId, usize)>,
    pub(crate) last_step_work: std::time::Duration,
    fatal: Option<EngineError>,
    fatal_sink: Option<crate::handle::FatalSlot>,
    pub(crate) interrupt: Arc<AtomicBool>,
    next_id: RequestId,
    next_rank: u64,
    kv_policy: KvPolicy,
    pub(crate) last_step_growth: usize,
    pub(crate) decode_scratch: crate::decode_step::DecodeScratch,
    /// Whether the previous step evicted a running sequence; consumed by the next demand
    /// admission pass.
    #[cfg(feature = "kv-demand-diagnostic")]
    pub(crate) evicted_since_admit: bool,
    /// Whether a decode step is between launch and collect.
    #[cfg(feature = "kv-demand-diagnostic")]
    pub(crate) decode_in_flight: bool,
}

impl<E: ModelExecutor> Scheduler<E> {
    pub fn new(config: SchedulerConfig, layout: KvLayout, executor: E) -> Self {
        let embedding_cache = EmbeddingCache::new(
            config.embedding_cache_bytes,
            config.embedding_hidden,
            config.embedding_elem_bytes,
        );
        Self {
            config,
            executor,
            pool: BlockPool::new(layout),
            embedding_cache,
            waiting: VecDeque::new(),
            active: BTreeMap::new(),
            counters: Counters::default(),
            disconnected: Vec::new(),
            spec_accepted_per_pos: Vec::new(),
            pass_ledger: crate::pass_ledger::PassLedger::default(),
            spec_log: crate::spec_log::SpecLog::from_env(),
            l1: crate::l1_counters::L1Ledger::new(
                crate::l1_counters::switches_from_env().unwrap_or_default(),
            ),
            vision_flight: None,
            embedding_reserved: std::collections::BTreeSet::new(),
            image_ledger: Default::default(),
            vision_sizes: BTreeMap::new(),
            vision_patch_sizes: BTreeMap::new(),
            vision_deferred_since: None,
            prefill_deferred_since: None,
            staged_vision: Vec::new(),
            last_step_work: std::time::Duration::ZERO,
            fatal: None,
            fatal_sink: None,
            interrupt: Arc::new(AtomicBool::new(false)),
            next_id: 0,
            next_rank: 0,
            kv_policy: KvPolicy::default(),
            last_step_growth: 0,
            decode_scratch: Default::default(),
            #[cfg(feature = "kv-demand-diagnostic")]
            evicted_since_admit: false,
            #[cfg(feature = "kv-demand-diagnostic")]
            decode_in_flight: false,
        }
    }

    /// Selects how admission commits KV blocks; [`KvPolicy::Reserve`] unless set.
    pub fn set_kv_policy(&mut self, policy: KvPolicy) {
        self.kv_policy = policy;
    }

    /// The active KV commitment policy.
    pub fn kv_policy(&self) -> KvPolicy {
        self.kv_policy
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    pub fn executor(&self) -> &E {
        &self.executor
    }

    pub fn executor_mut(&mut self) -> &mut E {
        &mut self.executor
    }

    /// The sticky device fault that stopped the scheduler, if any. Once set every request has
    /// been failed with it and no further executor call is made.
    pub fn fatal(&self) -> Option<&EngineError> {
        self.fatal.as_ref()
    }

    /// Sets the slot a recorded fatal fault is published to, before any request is failed
    /// with it.
    pub fn set_fatal_sink(&mut self, sink: crate::handle::FatalSlot) {
        self.fatal_sink = Some(sink);
    }

    /// Records a fatal `err`.
    pub(crate) fn note_fatal(&mut self, err: &EngineError) {
        if err.is_fatal() && self.fatal.is_none() {
            self.fatal = Some(err.clone());
            if let Some(sink) = &self.fatal_sink {
                sink.send_replace(Some(err.clone()));
            }
        }
    }

    /// When a fatal fault is recorded, fails every running and waiting request with it without
    /// touching the device and returns true.
    fn abort_on_fatal(&mut self) -> bool {
        let Some(err) = self.fatal.clone() else {
            return false;
        };
        self.abandon_all(&err);
        self.vision_flight = None;
        true
    }

    /// Replaces the flag that makes [`Scheduler::step`] stop between its stages; once it is
    /// set, the owner is expected to call [`Scheduler::cancel_all`].
    pub fn set_interrupt_flag(&mut self, flag: Arc<AtomicBool>) {
        self.interrupt = flag;
    }

    /// Whether the interrupt flag is set.
    pub(crate) fn interrupted(&self) -> bool {
        self.interrupt.load(Ordering::Acquire)
    }

    /// Whether no request is waiting or running.
    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.active.is_empty()
    }

    /// Validates and enqueues a request; events arrive on `tx`. Returns its id.
    pub fn submit(
        &mut self,
        request: EngineRequest,
        tx: UnboundedSender<EngineEvent>,
    ) -> Result<RequestId, EngineError> {
        let id = self.next_id;
        self.submit_with_id(id, request, tx)?;
        Ok(id)
    }

    /// As [`Scheduler::submit`] with a caller-assigned id, which must be unused.
    pub fn submit_with_id(
        &mut self,
        id: RequestId,
        request: EngineRequest,
        tx: UnboundedSender<EngineEvent>,
    ) -> Result<(), EngineError> {
        validate_request(&self.config, self.pool.layout(), &request)?;
        #[cfg(feature = "kv-demand-diagnostic")]
        if self.kv_policy == KvPolicy::DemandDiagnostic && self.waiting_new() >= self.queue_bound()
        {
            self.counters.rejected_queue += 1;
            return Err(EngineError::QueueFull {
                retry_after_secs: 1,
            });
        }
        self.next_id = self.next_id.max(id + 1);
        let EngineRequest {
            prompt,
            images,
            params,
        } = request;
        let sampler = SequenceSampler::new(
            self.config.vocab_size,
            &prompt,
            params,
            self.config.fallback_seed.wrapping_add(id),
        );
        for img in &images {
            self.image_ledger.observe(img.key, img.patches());
        }
        let embeds = images.iter().map(|_| None).collect();
        let rank = self.next_rank;
        self.next_rank += 1;
        self.counters.submitted += 1;
        self.counters.prompt_tokens += prompt.len() as u64;
        self.waiting.push_back(Sequence {
            id,
            rank,
            prompt,
            images,
            embeds,
            sampler,
            generated: Vec::new(),
            tx,
            timings: RequestTimings::default(),
            submitted: Instant::now(),
            admitted_once: false,
            preemptions: 0,
            phase: Phase::Vision,
            device_pending: false,
            #[cfg(feature = "kv-demand-diagnostic")]
            restartable: false,
        });
        Ok(())
    }

    /// Cancels a request wherever it is; its channel receives [`EngineEvent::Aborted`].
    pub fn cancel(&mut self, id: RequestId) -> bool {
        if let Some(pos) = self.waiting.iter().position(|s| s.id == id) {
            let seq = self.waiting.remove(pos).expect("position is valid");
            let _ = seq.tx.send(EngineEvent::Aborted);
            self.counters.aborted += 1;
            return true;
        }
        if let Some(seq) = self.active.remove(&id) {
            self.release_resources(id);
            let _ = seq.tx.send(EngineEvent::Aborted);
            self.counters.aborted += 1;
            return true;
        }
        false
    }

    /// Cancels every waiting and running request with `reason`; returns how many were cancelled.
    pub fn cancel_all(&mut self, reason: CancelReason) -> usize {
        let ids: Vec<RequestId> = self
            .waiting
            .iter()
            .map(|s| s.id)
            .chain(self.active.keys().copied())
            .collect();
        let mut n = 0;
        for id in ids {
            if let Some(pos) = self.waiting.iter().position(|s| s.id == id) {
                let seq = self.waiting.remove(pos).expect("position is valid");
                let _ = seq.tx.send(EngineEvent::Cancelled(reason));
                self.counters.aborted += 1;
                n += 1;
            } else if let Some(seq) = self.active.remove(&id) {
                self.release_resources(id);
                let _ = seq.tx.send(EngineEvent::Cancelled(reason));
                self.counters.aborted += 1;
                n += 1;
            }
        }
        n
    }

    /// Fails every waiting and running request with `err` without touching the executor or the
    /// block pool, for use when their state can no longer be trusted.
    pub fn abandon_all(&mut self, err: &EngineError) -> usize {
        let mut n = 0;
        for seq in std::mem::take(&mut self.waiting) {
            let _ = seq.tx.send(EngineEvent::Failed(err.clone()));
            n += 1;
        }
        for (_, seq) in std::mem::take(&mut self.active) {
            let _ = seq.tx.send(EngineEvent::Failed(err.clone()));
            n += 1;
        }
        self.counters.failed += n as u64;
        n
    }

    /// Output tokens the waiting and running requests may still generate: each request's
    /// `max_tokens` (the remaining context when unbounded) less what it has generated.
    pub fn tokens_in_flight(&self) -> u64 {
        self.waiting
            .iter()
            .chain(self.active.values())
            .map(|s| {
                let cap = s.sampler.params().max_tokens.map_or_else(
                    || self.config.max_context.saturating_sub(s.prompt.len()),
                    |m| m as usize,
                );
                cap.saturating_sub(s.generated.len()) as u64
            })
            .sum()
    }

    /// Fails the youngest running request with [`EngineError::Shed`], releasing its blocks;
    /// returns whether one was running.
    pub fn shed_youngest(&mut self) -> bool {
        let Some(&id) = self.active.keys().next_back() else {
            return false;
        };
        let seq = self.active.remove(&id).expect("key exists");
        self.release_resources(id);
        let _ = seq.tx.send(EngineEvent::Failed(EngineError::Shed));
        self.counters.shed += 1;
        true
    }

    /// Fails every running and waiting request with [`EngineError::Shed`]; returns how many.
    pub fn shed_all(&mut self) -> usize {
        let mut n = 0;
        while self.shed_youngest() {
            n += 1;
        }
        for seq in std::mem::take(&mut self.waiting) {
            let _ = seq.tx.send(EngineEvent::Failed(EngineError::Shed));
            self.counters.shed += 1;
            n += 1;
        }
        n
    }

    /// Replaces the L1 counter switches, discarding counters recorded so far.
    pub fn set_l1_switches(&mut self, switches: crate::l1_counters::L1Switches) {
        self.l1 = crate::l1_counters::L1Ledger::new(switches);
    }

    /// Drops every cached image embedding.
    pub fn clear_embedding_cache(&mut self) {
        self.embedding_cache.clear();
    }

    /// Runs one scheduling step: admit, vision, prefill, decode. Stops after any stage once
    /// the interrupt flag is set.
    pub fn step(&mut self) -> StepReport {
        if self.abort_on_fatal() {
            return StepReport::default();
        }
        let before = self.counters;
        self.l1.on_scheduler_step();
        let step_started = Instant::now();
        let pre_preempt = self.pool.stats().preemptions;
        let mut report = StepReport {
            admitted: self.admit(),
            ..StepReport::default()
        };
        if self.interrupted() {
            return report;
        }
        self.run_vision();
        report.vision_images = (self.counters.vision_images - before.vision_images) as usize;
        if self.abort_on_fatal() || self.interrupted() {
            return report;
        }
        let prefill_started = self.l1.now();
        let prefilled = self.run_prefill();
        if self.l1.enabled() {
            let spent = self.l1.now().saturating_sub(prefill_started);
            self.l1.exclude_prefill(spent);
        }
        if self.abort_on_fatal() {
            return report;
        }
        report.prefill_rows = prefilled.len();
        if self.interrupted() {
            return report;
        }
        let sampled: Vec<RequestId> = prefilled
            .iter()
            .copied()
            .filter(|id| self.active.get(id).is_some_and(|s| !s.device_pending))
            .collect();
        report.decode_rows = self.run_decode(&sampled);
        if self.abort_on_fatal() {
            return report;
        }
        if self.vision_flight.is_some()
            && (self.is_idle() || (report.prefill_rows == 0 && report.decode_rows == 0))
        {
            let blocked_from = self.l1.now();
            self.poll_vision(true);
            if self.l1.enabled() {
                let spent = self.l1.now().saturating_sub(blocked_from);
                self.l1.exclude_vision_block(spent);
            }
            if self.abort_on_fatal() {
                return report;
            }
        }
        let gone = std::mem::take(&mut self.disconnected);
        for id in gone {
            self.cancel(id);
        }
        report.finished = (self.counters.finished - before.finished) as usize;
        report.preempted = (self.pool.stats().preemptions - pre_preempt) as usize;
        if !self.is_idle() {
            self.counters.steps += 1;
        }
        let vision_spent = self.counters.vision_host_seconds - before.vision_host_seconds;
        if report.decode_rows + report.prefill_rows > 0 {
            self.last_step_work = step_started
                .elapsed()
                .saturating_sub(std::time::Duration::from_secs_f64(vision_spent.max(0.0)));
        }
        for line in self.l1.drain_lines() {
            eprintln!("{line}");
        }
        report
    }

    /// Steps until no request is waiting or running.
    pub fn run_until_idle(&mut self) {
        while !self.is_idle() {
            self.step();
        }
    }

    /// Admitted sequences in `phase`.
    fn active_in(&self, phase: Phase) -> usize {
        self.active.values().filter(|s| s.phase == phase).count()
    }

    /// Gauges and counters for the metrics endpoint.
    pub fn metrics(&self) -> EngineMetrics {
        let c = &self.counters;
        let (pinned_embedding_bytes, retained_embedding_bytes) = self.embedding_usage();
        EngineMetrics {
            kv_policy: self.kv_policy,
            accepted: self.accepted(),
            step_growth_blocks: self.last_step_growth,
            pinned_embedding_bytes,
            retained_embedding_bytes,
            admission_blocked: c.blocked,
            restarts: c.restarts,
            yields: c.yields,
            restart_discarded_tokens: c.restart_discarded_tokens,
            rejected_queue: c.rejected_queue,
            #[cfg(feature = "kv-demand-diagnostic")]
            waiting_new: self.waiting_new(),
            running: self.active.len(),
            tokens_in_flight: self.tokens_in_flight(),
            slots: self.config.max_concurrency,
            reserved_blocks: self.reserved_blocks(),
            kv_tokens: self.pool.layout().num_blocks * self.pool.layout().block_tokens,
            waiting: self.waiting.len(),
            active_vision: self.active_in(Phase::Vision),
            active_prefill: self.active_in(Phase::Prefill),
            active_running: self.active_in(Phase::Running),
            embedding_released: c.embedding_released,
            embedding_released_bytes: c.embedding_released_bytes,
            images: self.image_ledger.snapshot(),
            submitted: c.submitted,
            finished: c.finished,
            aborted: c.aborted,
            failed: c.failed,
            shed: c.shed,
            prompt_tokens: c.prompt_tokens,
            generation_tokens: c.generation_tokens,
            steps: c.steps,
            vision_batches: c.vision_batches,
            vision_images: c.vision_images,
            vision_patches: c.vision_patches,
            vision_seconds: c.vision_seconds,
            vision_host_seconds: c.vision_host_seconds,
            vision_batch_sizes: self.vision_sizes.iter().map(|(&k, &v)| (k, v)).collect(),
            vision_batch_patch_sizes: self
                .vision_patch_sizes
                .iter()
                .map(|(&k, &v)| (k, v))
                .collect(),
            prefill_passes: c.prefill_passes,
            prefill_pass_stats: self.pass_ledger.stats(),
            prefill_seconds: c.prefill_seconds,
            decode_steps: c.decode_steps,
            decode_seconds: c.decode_seconds,
            decode_prep_seconds: c.decode_prep_seconds,
            queue_seconds: c.queue_seconds,
            spec_drafts: c.spec_drafts,
            spec_draft_tokens: c.spec_draft_tokens,
            spec_accepted_tokens: c.spec_accepted_tokens,
            spec_accepted_per_pos: self.spec_accepted_per_pos.clone(),
            graphs_captured: self.executor.graph_stats().captured,
            graph_replays: self.executor.graph_stats().replays,
            graph_split_replays: self.executor.graph_stats().split_replays,
            graph_fallbacks: self.executor.graph_stats().fallbacks,
            slot_occupancy: self.executor.graph_stats().occupancy,
            l1: self.l1.stats(),
            pool: self.pool.stats(),
            embedding_cache: self.embedding_cache.stats(),
        }
    }

    /// KV blocks a sequence may come to hold: its prompt plus the tokens it can still generate
    /// (all remaining context when unbounded), clamped to the context window, plus the rows a
    /// decode step reserves ahead, clamped to the pool.
    pub(crate) fn reservation_blocks(&self, seq: &Sequence<E::Embedding>) -> usize {
        let ahead = self
            .executor
            .lookahead_tokens()
            .max(self.executor.draft_depth());
        let out = seq
            .sampler
            .params()
            .max_tokens
            .map_or(self.config.max_context, |m| m as usize);
        let tokens = (seq.prompt.len() + out).min(self.config.max_context) + ahead;
        let layout = self.pool.layout();
        layout.blocks_for_tokens(tokens).min(layout.num_blocks)
    }

    /// KV blocks promised to the running sequences.
    pub(crate) fn reserved_blocks(&self) -> usize {
        self.active
            .values()
            .map(|s| self.reservation_blocks(s))
            .sum()
    }

    /// Admits waiting sequences under the selected [`KvPolicy`].
    fn admit(&mut self) -> usize {
        match self.kv_policy {
            KvPolicy::Reserve => self.admit_reserved(),
            #[cfg(feature = "kv-demand-diagnostic")]
            KvPolicy::DemandDiagnostic => self.admit_demand(),
        }
    }

    /// Fills each unresolved image of `seq` from the embedding cache when its key is cached.
    fn resolve_cached_embeddings(&mut self, seq: &mut Sequence<E::Embedding>) {
        for (i, img) in seq.images.iter().enumerate() {
            if seq.embeds[i].is_none()
                && let Some(entry) = self.embedding_cache.get(&img.key)
            {
                seq.embeds[i] = Some(Arc::clone(&entry.buffer.0));
            }
        }
    }

    /// Drops the embeddings of a prefilled sequence and its budget reservation, counting the
    /// released buffers; the cache keeps the buffers until eviction.
    pub(crate) fn release_embeddings(&mut self, id: RequestId) {
        let Some(seq) = self.active.get_mut(&id) else {
            return;
        };
        let mut buffers = 0u64;
        let mut bytes = 0u64;
        for embed in seq.embeds.iter_mut() {
            if let Some(buf) = embed.take() {
                buffers += 1;
                bytes += DeviceBuffer::byte_len(&*buf) as u64;
            }
        }
        self.embedding_reserved.remove(&id);
        self.counters.embedding_released += buffers;
        self.counters.embedding_released_bytes += bytes;
    }

    /// Admits waiting sequences in order while concurrency allows and the pool can cover every
    /// admitted sequence's full reservation, so a running sequence never waits for blocks.
    /// Blocks are attached for the current context and resolved cached image embeddings.
    fn admit_reserved(&mut self) -> usize {
        let mut admitted = 0;
        let mut reserved = self.reserved_blocks();
        while let Some(front) = self.waiting.front() {
            if self.active.len() >= self.config.max_concurrency {
                break;
            }
            let bound = self.reservation_blocks(front);
            if reserved + bound > self.pool.layout().num_blocks {
                break;
            }
            let need = self.pool.layout().blocks_for_tokens(front.len());
            if self.pool.free_blocks() < need {
                break;
            }
            reserved += bound;
            let mut seq = self.waiting.pop_front().expect("front exists");
            self.pool
                .admit_with_key(seq.id, seq.rank)
                .expect("waiting ids are unregistered");
            self.pool
                .grow_to(seq.id, seq.len())
                .expect("free blocks checked");
            let first_admission = !seq.admitted_once;
            if first_admission {
                seq.admitted_once = true;
                seq.timings.queue = seq.submitted.elapsed();
                self.counters.queue_seconds += seq.timings.queue.as_secs_f64();
            }
            if first_admission || self.config.embedding_release_after_prefill {
                self.resolve_cached_embeddings(&mut seq);
            }
            seq.phase = if seq.embeddings_ready() {
                Phase::Prefill
            } else {
                Phase::Vision
            };
            self.active.insert(seq.id, seq);
            admitted += 1;
        }
        admitted
    }

    /// Blocks a sequence still needs for its next decode step: the blocks covering its length
    /// plus the lookahead rows (clamped to the pool) beyond those it holds.
    #[cfg(feature = "kv-demand-diagnostic")]
    pub(crate) fn step_growth_blocks(&self, id: RequestId, len: usize, ahead: usize) -> usize {
        let layout = self.pool.layout();
        let want = layout.blocks_for_tokens(len + ahead).min(layout.num_blocks);
        let held = self.pool.block_table(id).map_or(0, <[_]>::len);
        want.saturating_sub(held)
    }

    /// Demand admission: the front of the queue is admitted while concurrency allows, its
    /// prompt and first-step blocks are free now, and the free blocks left over still cover
    /// the next-step growth of every running sequence plus a margin of
    /// `W = max(max_concurrency, ceil(blocks / 50))` blocks (`2W` in the step after an
    /// eviction). Nothing is promised beyond the next step.
    ///
    /// A front whose client has disconnected is cancelled instead of admitted, so it is never
    /// prefilled.
    ///
    /// A non-restartable front is admitted while other sequences are active only if the
    /// reservations of the non-restartable ones, its own included, plus one full context fit
    /// the pool, so blocks nothing can reclaim never leave the oldest sequence short.
    ///
    /// When no sequence is running, nothing can drain, so the front is admitted without the
    /// margin; the hard requirement that its prompt and first-step blocks fit the free blocks
    /// still applies. A front that does not fit the free blocks waits: this policy has no
    /// eviction yet, so it never frees blocks by stopping a running sequence.
    #[cfg(feature = "kv-demand-diagnostic")]
    fn admit_demand(&mut self) -> usize {
        let ahead = self
            .executor
            .lookahead_tokens()
            .max(self.executor.draft_depth());
        let blocks = self.pool.layout().num_blocks;
        let base_margin = self.config.max_concurrency.max(blocks.div_ceil(50));
        let margin = if self.evicted_since_admit {
            2 * base_margin
        } else {
            base_margin
        };
        let mut admitted = 0;
        while let Some(front) = self.waiting.front() {
            if front.tx.is_closed() {
                let seq = self.waiting.pop_front().expect("front exists");
                let _ = seq.tx.send(EngineEvent::Aborted);
                self.counters.aborted += 1;
                continue;
            }
            if self.active.len() >= self.config.max_concurrency {
                self.counters.blocked.slots += 1;
                break;
            }
            if !front.admitted_once && self.accepted() >= self.accepted_bound() {
                self.counters.blocked.accepted += 1;
                break;
            }
            let need = self.pool.layout().blocks_for_tokens(front.len());
            let first_step = self
                .pool
                .layout()
                .blocks_for_tokens(front.len() + ahead)
                .min(blocks)
                .max(need);
            let free = self.pool.free_blocks();
            if first_step > free {
                self.counters.blocked.prompt_blocks += 1;
                break;
            }
            if !self.active.is_empty() {
                let running: usize = self
                    .active
                    .values()
                    .map(|s| self.step_growth_blocks(s.id, s.len(), ahead))
                    .sum();
                if free - first_step < running + margin {
                    if free - first_step >= running + base_margin {
                        self.counters.blocked.hysteresis += 1;
                    } else {
                        self.counters.blocked.watermark += 1;
                    }
                    break;
                }
            }
            if !front.restartable && !self.active.is_empty() {
                let pinned: usize = self
                    .active
                    .values()
                    .filter(|s| !s.restartable)
                    .map(|s| self.reservation_blocks(s))
                    .sum();
                let context = blocks.min(
                    self.pool
                        .layout()
                        .blocks_for_tokens(self.config.max_context + ahead),
                );
                if pinned + self.reservation_blocks(front) + context > blocks {
                    self.counters.blocked.watermark += 1;
                    break;
                }
            }
            let mut seq = self.waiting.pop_front().expect("front exists");
            self.pool
                .admit_with_key(seq.id, seq.rank)
                .expect("waiting ids are unregistered");
            self.pool
                .grow_to(seq.id, seq.len())
                .expect("free blocks checked");
            let first_admission = !seq.admitted_once;
            if first_admission {
                seq.admitted_once = true;
                seq.timings.queue = seq.submitted.elapsed();
                self.counters.queue_seconds += seq.timings.queue.as_secs_f64();
            }
            if first_admission || self.config.embedding_release_after_prefill {
                self.resolve_cached_embeddings(&mut seq);
            }
            seq.phase = if seq.embeddings_ready() {
                Phase::Prefill
            } else {
                Phase::Vision
            };
            self.active.insert(seq.id, seq);
            admitted += 1;
        }
        self.evicted_since_admit = false;
        admitted
    }

    /// Waiting requests that were never admitted; evicted requests awaiting re-admission are
    /// not counted.
    #[cfg(feature = "kv-demand-diagnostic")]
    pub(crate) fn waiting_new(&self) -> usize {
        self.waiting.iter().filter(|s| !s.admitted_once).count()
    }

    /// Most never-admitted requests that may wait: twice the concurrency.
    #[cfg(feature = "kv-demand-diagnostic")]
    pub(crate) fn queue_bound(&self) -> usize {
        2 * self.config.max_concurrency
    }

    /// Most requests that may be accepted (running plus evicted and awaiting re-admission):
    /// twice the concurrency.
    #[cfg(feature = "kv-demand-diagnostic")]
    pub(crate) fn accepted_bound(&self) -> usize {
        2 * self.config.max_concurrency
    }

    /// Requests the engine has taken on: running ones plus evicted ones awaiting re-admission.
    pub(crate) fn accepted(&self) -> usize {
        self.active.len() + self.waiting.iter().filter(|s| s.admitted_once).count()
    }

    /// As [`Scheduler::submit_with_id`], marking the request restartable when `restartable`: an
    /// eviction may then discard its output and run it again from the prompt.
    #[cfg(feature = "kv-demand-diagnostic")]
    pub fn submit_restartable_with_id(
        &mut self,
        id: RequestId,
        request: EngineRequest,
        tx: UnboundedSender<EngineEvent>,
        restartable: bool,
    ) -> Result<(), EngineError> {
        self.submit_with_id(id, request, tx)?;
        if restartable && let Some(seq) = self.waiting.back_mut() {
            seq.restartable = true;
        }
        Ok(())
    }

    /// Whether the sequence's tokens are chosen on the device step: the executor supports it
    /// and the request asks for no log-probabilities.
    pub(crate) fn device_bound(&self, seq: &Sequence<E::Embedding>) -> bool {
        self.executor.supports_device_step() && seq.sampler.params().logprobs.is_none()
    }

    fn release_resources(&mut self, id: RequestId) {
        let _ = self.pool.release(id);
        self.executor.release(id);
    }

    /// Delivers a sampled token and applies the finish rules; returns whether it finished.
    pub(crate) fn accept_token(&mut self, id: RequestId, logits: &[f32]) -> bool {
        self.accept_sampled(id, logits).1
    }

    /// As [`Scheduler::accept_token`], also returning the sampled token id.
    pub(crate) fn accept_sampled(&mut self, id: RequestId, logits: &[f32]) -> (u32, bool) {
        let seq = self.active.get_mut(&id).expect("sequence is active");
        let sampled = seq.sampler.sample(logits);
        self.deliver(id, sampled)
    }

    /// Delivers a token the device already selected; returns whether the sequence finished.
    pub(crate) fn accept_emitted(&mut self, id: RequestId, token: u32, margin: f32) -> bool {
        self.deliver(
            id,
            Sampled {
                token,
                margin,
                logprob: None,
                top_logprobs: Vec::new(),
            },
        )
        .1
    }

    /// Records a selected token, streams it and applies the finish rules.
    fn deliver(&mut self, id: RequestId, sampled: Sampled) -> (u32, bool) {
        let seq = self.active.get_mut(&id).expect("sequence is active");
        seq.sampler.accept(sampled.token);
        let index = seq.generated.len() as u32;
        seq.generated.push(sampled.token);
        self.counters.generation_tokens += 1;
        let is_eos = self.config.eos_token_ids.contains(&sampled.token);
        let mut reason = seq
            .sampler
            .finish_after(sampled.token, &self.config.eos_token_ids);
        if reason.is_none() && seq.len() >= self.config.max_context {
            reason = Some(FinishReason::Length);
        }
        if !is_eos
            && seq
                .tx
                .send(EngineEvent::Token(TokenEvent {
                    id,
                    index,
                    token: sampled.token,
                    margin: sampled.margin,
                    logprob: sampled.logprob,
                    top_logprobs: sampled.top_logprobs,
                }))
                .is_err()
        {
            self.disconnected.push(id);
        }
        match reason {
            Some(r) => {
                self.finish(id, r);
                (sampled.token, true)
            }
            None => (sampled.token, false),
        }
    }

    pub(crate) fn finish(&mut self, id: RequestId, reason: FinishReason) {
        let Some(seq) = self.active.remove(&id) else {
            return;
        };
        self.release_resources(id);
        self.counters.finished += 1;
        if let Some(line) =
            self.spec_log
                .finish(id, seq.prompt.len(), seq.images.len(), seq.generated.len())
        {
            eprintln!("{line}");
        }
        let _ = seq.tx.send(EngineEvent::Finished(FinishedEvent {
            id,
            reason,
            prompt_tokens: seq.prompt.len(),
            completion_tokens: seq.generated.len(),
            timings: seq.timings,
            preemptions: seq.preemptions,
        }));
    }

    pub(crate) fn fail(&mut self, id: RequestId, err: EngineError) {
        self.note_fatal(&err);
        if let Some(seq) = self.active.remove(&id) {
            self.release_resources(id);
            self.counters.failed += 1;
            let _ = seq.tx.send(EngineEvent::Failed(err));
        }
    }

    pub(crate) fn fail_many(&mut self, ids: impl IntoIterator<Item = RequestId>, err: EngineError) {
        let mut seen = Vec::new();
        for id in ids {
            if !seen.contains(&id) {
                seen.push(id);
                self.fail(id, err.clone());
            }
        }
    }

    /// Returns an evicted sequence to the waiting queue for recompute, ahead of every request
    /// of higher submission rank and behind every request of lower rank.
    pub(crate) fn requeue_preempted(&mut self, id: RequestId) {
        if let Some(mut seq) = self.active.remove(&id) {
            self.executor.release(id);
            seq.preemptions += 1;
            if self.config.embedding_release_after_prefill {
                self.resolve_cached_embeddings(&mut seq);
            }
            seq.phase = if seq.embeddings_ready() {
                Phase::Prefill
            } else {
                Phase::Vision
            };
            let at = self.waiting.partition_point(|w| w.rank < seq.rank);
            self.waiting.insert(at, seq);
        }
    }

    /// Returns a sequence still in the vision or prefill phase to the waiting queue at its
    /// rank, freeing its executor state; its embeddings, their budget charge and any generated
    /// tokens are kept, and no event is sent.
    #[cfg(feature = "kv-demand-diagnostic")]
    pub(crate) fn requeue_unstarted(&mut self, id: RequestId) {
        let Some(mut seq) = self.active.remove(&id) else {
            return;
        };
        self.executor.release(id);
        self.counters.yields += 1;
        seq.preemptions += 1;
        seq.phase = Phase::Prefill;
        let at = self.waiting.partition_point(|w| w.rank < seq.rank);
        self.waiting.insert(at, seq);
        self.evicted_since_admit = true;
    }

    /// Returns an evicted sequence to the waiting queue for a restart from its prompt: its
    /// generated tokens and sampler state are discarded, its rank and embeddings are kept, and
    /// it re-enters ahead of every request of higher rank. The consumer is told with
    /// [`EngineEvent::Restarted`].
    #[cfg(feature = "kv-demand-diagnostic")]
    pub(crate) fn requeue_restart(&mut self, id: RequestId) {
        let Some(mut seq) = self.active.remove(&id) else {
            return;
        };
        self.executor.release(id);
        self.counters.restarts += 1;
        self.counters.restart_discarded_tokens += seq.generated.len() as u64;
        seq.preemptions += 1;
        seq.generated.clear();
        seq.device_pending = false;
        seq.sampler = SequenceSampler::new(
            self.config.vocab_size,
            &seq.prompt,
            seq.sampler.params().clone(),
            self.config.fallback_seed.wrapping_add(id),
        );
        seq.phase = Phase::Prefill;
        let _ = seq.tx.send(EngineEvent::Restarted);
        let at = self.waiting.partition_point(|w| w.rank < seq.rank);
        self.waiting.insert(at, seq);
        self.evicted_since_admit = true;
    }
}
