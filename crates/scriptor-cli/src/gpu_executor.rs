//! The scheduler's model executor on the GPU: vision tower, paged prefill and decode, and the
//! MTP draft chain with batched verification. Text steps run on the main stream; vision
//! batches run on their own stream so they overlap decode.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::sys::{CUevent_flags, cudaError_enum};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream};
use scriptor_cache::DeviceBuffer;
use scriptor_chat::{ImageSpan, mrope_positions};
use scriptor_cuda::{Device, DeviceArena, GreenPartitions, SmSplit};
use scriptor_engine::{
    DecodeRow, DeviceRow, EngineError, GraphStats, ModelExecutor, PrefillRow, RequestId,
    SpecRowOutput, StepRowResult, VisionJob, VisionToken,
};
use scriptor_kernels::attention::DECODE_MAX_ROWS;
use scriptor_kernels::cublaslt::{GemmEngine, GemmOptions};
use scriptor_kernels::heuristics::{ShapeInventory, TuningTable};
use scriptor_kernels::reference::sampling::argmax_with_margin;
use scriptor_model::gpu::{
    DecodeSeq, GpuDecoderWeights, GpuKvPool, GpuMtpWeights, GpuVision, GpuVisionParams, PrefillSeq,
    RowOps, StepInput, TextExecutor, TextGeometry, VisionWorkspace,
};
use scriptor_model::reference::TensorSource;
use scriptor_model::{Merger, VisionTower};
use scriptor_spec::{MemoryPlan, ModelSpec};
use scriptor_weights::SafetensorsDir;

use crate::device_step::{ActiveRow, CaptureReport, DeviceStep, StepLimits, StepWeights};

/// Elements of one patch row fed to the vision tower.
const PATCH_ELEMS: usize = 1176;

/// Largest draft depth whose verification rows fit one paged-attention launch.
pub fn max_draft_depth(heads: usize, kv_heads: usize) -> usize {
    (DECODE_MAX_ROWS / (heads / kv_heads).max(1)).saturating_sub(1)
}

/// Maps a failure to an executor error; a sticky CUDA fault becomes [`EngineError::Fatal`].
/// A raw driver error, a [`scriptor_cuda::CudaError`] and a `GpuTextError::Cuda` carrying the
/// same `CUresult` classify identically.
pub(crate) fn exec_err<E: std::fmt::Display + 'static>(e: E) -> EngineError {
    let any: &dyn std::any::Any = &e;
    let raw;
    let cuda = if let Some(c) = any.downcast_ref::<scriptor_cuda::CudaError>() {
        Some(c)
    } else if let Some(scriptor_model::gpu::GpuTextError::Cuda(c)) =
        any.downcast_ref::<scriptor_model::gpu::GpuTextError>()
    {
        Some(c)
    } else if let Some(d) = any.downcast_ref::<cudarc::driver::DriverError>() {
        raw = scriptor_cuda::CudaError::from(*d);
        Some(&raw)
    } else {
        None
    };
    match cuda {
        Some(c) if c.is_fatal() => {
            let code = c.code().unwrap_or_default();
            let name = scriptor_cuda::fatal_code_name(code).unwrap_or("unclassified");
            EngineError::fatal(code, format!("{name}: {c}"))
        }
        _ => EngineError::Executor(e.to_string()),
    }
}

/// How the image embeddings of one prefill reach the text forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImageFeed {
    /// The batch references no image.
    None,
    /// The batch references one embedding, which the forward pass reads in place.
    Direct,
    /// The embeddings are copied, in prompt order, into one zeroed buffer.
    Packed { sources: usize },
}

/// Device work the image embeddings cost one prefill.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FeedCost {
    /// Temporary device allocations.
    pub allocations: usize,
    /// Zero-fill operations over a temporary buffer.
    pub zero_fills: usize,
    /// Device-to-device copies.
    pub copies: usize,
    /// Bytes moved by those copies.
    pub copied_bytes: usize,
    /// Device bytes held only for the duration of the prefill.
    pub temp_bytes: usize,
}

impl ImageFeed {
    /// Chooses the feed from the row count of each embedding reference, in prompt order. Only
    /// a single reference is read in place; repeated references to one embedding count once
    /// per occurrence and stay packed.
    pub(crate) fn plan(source_rows: &[usize], direct: bool) -> Self {
        match source_rows.len() {
            0 => Self::None,
            1 if direct => Self::Direct,
            n => Self::Packed { sources: n },
        }
    }

    /// The device work this feed costs for `source_rows` and a hidden width of `hidden`.
    #[cfg(test)]
    pub(crate) fn cost(self, source_rows: &[usize], hidden: usize) -> FeedCost {
        match self {
            Self::None | Self::Direct => FeedCost {
                allocations: 0,
                zero_fills: 0,
                copies: 0,
                copied_bytes: 0,
                temp_bytes: 0,
            },
            Self::Packed { .. } => {
                let bytes = source_rows.iter().sum::<usize>() * hidden * 2;
                FeedCost {
                    allocations: 1,
                    zero_fills: 1,
                    copies: source_rows.len(),
                    copied_bytes: bytes,
                    temp_bytes: bytes,
                }
            }
        }
    }
}

/// Image embeddings on the device, `rows x hidden` bf16, with the merged token grid they cover.
pub struct GpuEmbedding {
    buf: CudaSlice<u16>,
    rows: usize,
    hidden: usize,
    grid: (usize, usize, usize),
}

#[cfg(test)]
impl GpuEmbedding {
    /// Language-model tokens the embedding covers.
    pub(crate) fn row_count(&self) -> usize {
        self.rows
    }

    /// Hidden width of one row.
    pub(crate) fn width(&self) -> usize {
        self.hidden
    }
}

impl DeviceBuffer for GpuEmbedding {
    fn byte_len(&self) -> usize {
        self.rows * self.hidden * 2
    }
}

/// Weights resident on the device.
pub struct GpuWeights {
    decoder: GpuDecoderWeights,
    mtp: Option<GpuMtpWeights>,
    vision: GpuVisionParams,
}

impl GpuWeights {
    #[cfg(test)]
    pub(crate) fn from_parts(
        decoder: GpuDecoderWeights,
        mtp: Option<GpuMtpWeights>,
        vision: GpuVisionParams,
    ) -> Self {
        Self {
            decoder,
            mtp,
            vision,
        }
    }

    /// Uploads the decoder, the vision tower and, when `with_mtp`, the MTP head.
    pub fn load(
        stream: &Arc<CudaStream>,
        dir: &SafetensorsDir,
        spec: &ModelSpec,
        with_mtp: bool,
    ) -> Result<Self, EngineError> {
        let decoder = GpuDecoderWeights::load(stream, dir, &spec.text).map_err(exec_err)?;
        let mtp = if with_mtp {
            if spec.text.num_mtp_layers == 0 {
                return Err(EngineError::Executor(
                    "the checkpoint has no MTP layer for speculative decoding".into(),
                ));
            }
            Some(GpuMtpWeights::load_sharing(stream, dir, &spec.text, &decoder).map_err(exec_err)?)
        } else {
            None
        };
        let source = TensorSource::new(dir);
        let tower = VisionTower::load(&source, &spec.vision).map_err(exec_err)?;
        let merger = Merger::load(&source, &spec.vision).map_err(exec_err)?;
        let vision = GpuVisionParams::from_reference(stream, &tower, &merger).map_err(exec_err)?;
        Ok(Self {
            decoder,
            mtp,
            vision,
        })
    }
}

/// A reference into a hidden-state buffer: `row` is the first of the sequence's rows.
#[derive(Clone)]
struct HiddenRef {
    buf: Arc<CudaSlice<u16>>,
    row: usize,
}

/// What the MTP layer still has to consume for a sequence.
enum Feed {
    /// Right after prefill: the prompt rows, with the shifted prompt ids and their positions.
    Boot {
        hidden: HiddenRef,
        next_tokens: Vec<u32>,
        positions: Vec<[i32; 3]>,
    },
    /// Right after a prefill whose next token the device step chooses: the MTP KV already
    /// holds every prompt position but the last.
    Prompt,
    /// After a verification pass that started at sequence index `start`.
    Verified {
        hidden: HiddenRef,
        start: usize,
        drafts: Vec<u32>,
    },
}

struct SeqState {
    /// Added to a sequence index to get its rotary position after the last image.
    delta: i64,
    feed: Option<Feed>,
}

/// Device events and host stamps bracketing one vision batch for the L1 counters.
struct VisionProbe {
    copy_from: CudaEvent,
    copy_to: CudaEvent,
    start: Arc<AtomicU64>,
    ready: Arc<AtomicU64>,
}

/// Host function run on a stream: stores the epoch time into the stamp it was launched with.
unsafe extern "C" fn stamp_callback(data: *mut std::ffi::c_void) {
    // SAFETY: `data` is the `Box<Arc<AtomicU64>>` leaked by `launch_stamp`, and the driver runs
    // each host function exactly once.
    let stamp = unsafe { Box::from_raw(data.cast::<Arc<AtomicU64>>()) };
    stamp.store(
        scriptor_engine::l1_counters::epoch_ns().max(1),
        Ordering::Release,
    );
}

/// Enqueues a host function on `stream` that stores the epoch time into `stamp` once all
/// earlier work on the stream has finished.
fn launch_stamp(stream: &CudaStream, stamp: &Arc<AtomicU64>) -> Result<(), String> {
    stream
        .context()
        .bind_to_thread()
        .map_err(|e| e.to_string())?;
    let data = Box::into_raw(Box::new(Arc::clone(stamp)));
    // SAFETY: the stream handle is live, the callback only reads `data`, and the box is
    // reclaimed here when the launch is refused.
    let rc = unsafe {
        cudarc::driver::sys::cuLaunchHostFunc(
            stream.cu_stream(),
            Some(stamp_callback),
            data.cast::<std::ffi::c_void>(),
        )
    };
    if rc != cudaError_enum::CUDA_SUCCESS {
        // SAFETY: the driver did not take ownership of `data`.
        drop(unsafe { Box::from_raw(data) });
        return Err(format!("{rc:?}"));
    }
    Ok(())
}

/// A vision batch enqueued on the vision stream: its inputs stay alive until the batch ends.
struct VisionPending {
    done: CudaEvent,
    l1: Option<VisionProbe>,
    outs: Vec<CudaSlice<u16>>,
    grids: Vec<(usize, usize, usize)>,
    _input: CudaSlice<u16>,
}

/// The GPU model executor.
pub struct GpuExecutor {
    stream: Arc<CudaStream>,
    vision_stream: Arc<CudaStream>,
    vision_pending: HashMap<VisionToken, VisionPending>,
    next_vision: VisionToken,
    l1_vision: bool,
    l1_callbacks: bool,
    l1_timing: HashMap<VisionToken, scriptor_engine::l1_counters::VisionTiming>,
    text: TextExecutor,
    vision: GpuVision,
    vision_ws: VisionWorkspace,
    vision_gemm: GemmEngine,
    decoder: GpuDecoderWeights,
    mtp: Option<GpuMtpWeights>,
    kv: GpuKvPool,
    mtp_kv: Option<GpuKvPool>,
    ops: RowOps,
    hidden: usize,
    vision_saturation: usize,
    prefill_saturation: usize,
    staged: Option<StagedVision>,
    vocab: usize,
    image_token_id: u32,
    depth: usize,
    seqs: HashMap<RequestId, SeqState>,
    device_step: DeviceStep,
    ctx: Arc<cudarc::driver::CudaContext>,
    plan_total: u64,
    _arena: DeviceArena,
    decode_stream: Option<Arc<CudaStream>>,
    shape_log_path: Option<PathBuf>,
    // Declared last so the partitions are destroyed after every stream user above.
    partitions: Option<GreenPartitions>,
}

/// Host-side patch inputs built ahead for the vision batch named by `keys`.
struct StagedVision {
    keys: Vec<(RequestId, usize)>,
    bits: Vec<u16>,
}

/// Concatenates the jobs' patch payloads as the little-endian u16 words the tower reads.
fn stage_patches(jobs: &[VisionJob]) -> Result<Vec<u16>, EngineError> {
    let total: usize = jobs.iter().map(|j| j.patches).sum();
    let mut bits: Vec<u16> = Vec::with_capacity(total * PATCH_ELEMS);
    for job in jobs {
        if job.payload.len() != job.patches * PATCH_ELEMS * 2 {
            return Err(EngineError::Executor(format!(
                "image payload holds {} bytes for {} patches",
                job.payload.len(),
                job.patches
            )));
        }
        bits.extend(
            job.payload
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]])),
        );
    }
    Ok(bits)
}

/// Logs the boot-autotune choices that differ from the first candidate.
fn log_autotune(label: &str, report: &scriptor_model::gpu::TuneReport) {
    for t in report.gemm.iter().filter(|t| t.algo_index != 0) {
        eprintln!(
            "scriptor: autotune {label} gemm m{} n{} k{} {}: candidate {} of {} ({} not bit-identical), {:.0}us -> {:.0}us",
            t.m,
            t.key.n,
            t.key.k,
            t.key.epilogue.name(),
            t.algo_index,
            t.candidates,
            t.rejected,
            t.first_micros,
            t.micros,
        );
    }
}

impl Drop for GpuExecutor {
    /// Writes the recorded GEMM inventory when [`GpuExecutor::record_gemm_shapes`] enabled it,
    /// headed by the total device-step graph replay count.
    fn drop(&mut self) {
        let Some(path) = self.shape_log_path.take() else {
            return;
        };
        let inv = self.take_gemm_shapes();
        let stats = self.device_step.stats();
        let text = format!(
            "# device-step graph replays (all buckets): {}; eager fallbacks: {}\n{}",
            stats.replays,
            stats.fallbacks,
            inv.to_toml()
        );
        match std::fs::write(&path, text) {
            Ok(()) => eprintln!(
                "scriptor: wrote {} GEMM shape records to {}",
                inv.len(),
                path.display()
            ),
            Err(e) => eprintln!(
                "scriptor: writing GEMM shape records to {} failed: {e}",
                path.display()
            ),
        }
    }
}

impl GpuExecutor {
    #[allow(clippy::too_many_arguments)]
    /// Builds the executors and KV pools sized by `plan`. `depth` is the MTP draft depth.
    pub fn new(
        device: &Device,
        stream: &Arc<CudaStream>,
        weights: GpuWeights,
        spec: &ModelSpec,
        plan: &MemoryPlan,
        depth: usize,
        tuning: TuningTable,
        limits: StepLimits,
    ) -> Result<Self, EngineError> {
        let geometry = TextGeometry::from_spec(&spec.text).map_err(exec_err)?;
        let limit = max_draft_depth(geometry.heads, geometry.kv_heads);
        if depth > limit {
            return Err(EngineError::Executor(format!(
                "draft depth {depth} exceeds the supported maximum of {limit}"
            )));
        }
        if depth > 0 && weights.mtp.is_none() {
            return Err(EngineError::Executor(
                "speculative decoding needs the MTP weights".into(),
            ));
        }
        let ctx = device.context();
        // SAFETY: cross-stream ordering is explicit: vision outputs are allocated on the main
        // stream, the vision stream waits on an event recorded after the allocation, and the
        // host waits on the vision stream's completion event before the main stream reads them.
        unsafe { ctx.disable_event_tracking() };
        let sm_count = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .map_err(exec_err)? as usize;
        let options = GemmOptions::default();
        let mut arena = device
            .arena(3 * options.workspace_bytes + (4 << 20), stream)
            .map_err(exec_err)?;
        let mut text = TextExecutor::new(
            ctx,
            stream,
            &mut arena,
            geometry,
            sm_count,
            tuning.clone(),
            options,
        )
        .map_err(exec_err)?;
        let vision_stream = device.new_stream().map_err(exec_err)?;
        let mut vision_gemm =
            GemmEngine::new(ctx, &vision_stream, &mut arena, tuning.clone(), options)
                .map_err(exec_err)?;
        let GpuWeights {
            decoder,
            mtp,
            vision: vision_params,
        } = weights;
        let vision = GpuVision::new(ctx, vision_params).map_err(exec_err)?;
        let tune_started = std::time::Instant::now();
        let vision_report = vision
            .autotune(&mut vision_gemm, plan.vision_batch_patches)
            .map_err(exec_err)?;
        let text_report = text.autotune(limits.max_context).map_err(exec_err)?;
        vision_stream.synchronize().map_err(exec_err)?;
        stream.synchronize().map_err(exec_err)?;
        log_autotune("vision", &vision_report);
        log_autotune("text", &text_report);
        eprintln!(
            "scriptor: boot autotune took {:.1}s",
            tune_started.elapsed().as_secs_f64()
        );
        let vision_ws =
            VisionWorkspace::new(&vision_stream, vision.params(), plan.vision_batch_patches)
                .map_err(exec_err)?;
        let blocks = plan.kv_blocks as usize;
        scriptor_cuda::fault::check_alloc("kv_pool").map_err(exec_err)?;
        let kv = GpuKvPool::new(
            stream,
            geometry.layers,
            blocks,
            plan.kv_block_tokens,
            &geometry,
        )
        .map_err(exec_err)?;
        let mtp_kv = if depth > 0 {
            Some(
                GpuKvPool::new(stream, 1, blocks, plan.kv_block_tokens, &geometry)
                    .map_err(exec_err)?,
            )
        } else {
            None
        };
        let device_step = DeviceStep::new(
            ctx,
            stream,
            &mut arena,
            geometry,
            sm_count,
            tuning.clone(),
            options,
            &kv,
            depth,
            limits,
        )?;
        stream.synchronize().map_err(exec_err)?;
        vision_stream.synchronize().map_err(exec_err)?;
        let vision_saturation =
            scriptor_engine::vision_queue::saturation_patches(sm_count, vision.params().hidden);
        let prefill_saturation =
            scriptor_engine::vision_queue::saturation_patches(sm_count, geometry.hidden);
        Ok(Self {
            stream: stream.clone(),
            vision_stream,
            vision_pending: HashMap::new(),
            next_vision: 0,
            l1_vision: scriptor_engine::l1_counters::switches_from_env()
                .unwrap_or_default()
                .counters,
            l1_callbacks: true,
            l1_timing: HashMap::new(),
            text,
            vision,
            vision_ws,
            vision_gemm,
            decoder,
            mtp,
            kv,
            mtp_kv,
            ops: RowOps::load(ctx).map_err(exec_err)?,
            hidden: geometry.hidden,
            vision_saturation,
            prefill_saturation,
            staged: None,
            vocab: geometry.vocab,
            image_token_id: spec.image_token_id,
            depth,
            seqs: HashMap::new(),
            device_step,
            ctx: Arc::clone(ctx),
            plan_total: plan.total,
            _arena: arena,
            decode_stream: None,
            shape_log_path: None,
            partitions: None,
        })
    }

    /// Moves vision batches and the decode graphs onto SM partitions of the device. Call it
    /// after construction and before [`Self::capture_graphs`]: construction has already chosen
    /// every launch geometry, split plan and cuBLASLt algorithm on the full device, and those
    /// choices are kept. Eager prefill, eager decode and the MTP draft chain stay on the
    /// original full-device stream, which has no SM limit; the decode partition runs only the
    /// replayed step graphs.
    pub fn partition(&mut self, split: SmSplit) -> Result<(), EngineError> {
        self.stream.synchronize().map_err(exec_err)?;
        self.vision_stream.synchronize().map_err(exec_err)?;
        let greatest = cudarc::driver::result::stream::get_priority_range().map_or(0, |(_, g)| g);
        let partitions =
            GreenPartitions::create(&self.ctx, split, (0, greatest)).map_err(exec_err)?;
        for line in partitions.grant_log() {
            eprintln!("scriptor: SM split request {line}");
        }
        self.vision_gemm.set_stream(partitions.vision_stream());
        self.vision_stream = partitions.vision_stream().clone();
        self.device_step.set_stream(partitions.decode_stream());
        self.decode_stream = Some(partitions.decode_stream().clone());
        self.partitions = Some(partitions);
        Ok(())
    }

    /// The stream vision batches run on.
    #[cfg(test)]
    pub(crate) fn vision_stream(&self) -> &Arc<CudaStream> {
        &self.vision_stream
    }

    /// Turns the L1 vision events, host callbacks and step host samples on or off.
    #[cfg(test)]
    pub(crate) fn set_l1_probes(&mut self, on: bool) {
        self.l1_vision = on;
        self.l1_callbacks = true;
        self.device_step.set_host_probe(on);
    }

    /// The embedding's bf16 words, downloaded after the main stream drains.
    #[cfg(test)]
    pub(crate) fn embedding_bits(&self, e: &GpuEmbedding) -> Result<Vec<u16>, EngineError> {
        self.stream
            .clone_dtoh(&e.buf.slice(0..e.rows * e.hidden))
            .map_err(exec_err)
    }

    /// The hidden-state buffer the last prefill kept for `seq`, as bf16 words.
    #[cfg(test)]
    pub(crate) fn kept_hidden_bits(&self, seq: RequestId) -> Result<Option<Vec<u16>>, EngineError> {
        match self.seqs.get(&seq).and_then(|s| s.feed.as_ref()) {
            Some(Feed::Boot { hidden, .. }) => Ok(Some(
                self.stream.clone_dtoh(&*hidden.buf).map_err(exec_err)?,
            )),
            _ => Ok(None),
        }
    }

    /// The executor's driver context.
    #[cfg(test)]
    pub(crate) fn context(&self) -> &Arc<cudarc::driver::CudaContext> {
        &self.ctx
    }

    /// The vision and decode partition context handles, when partitioned.
    #[cfg(test)]
    pub(crate) fn partition_contexts(&self) -> Option<(usize, usize)> {
        self.partitions
            .as_ref()
            .map(|p| (p.vision_context(), p.decode_context()))
    }

    /// The execution-context handle of every captured graph kernel node.
    #[cfg(test)]
    pub(crate) fn graph_node_contexts(&self) -> Result<Vec<usize>, EngineError> {
        self.device_step.graph_node_contexts()
    }

    /// The raw device-step buffers of the last step.
    #[cfg(test)]
    pub(crate) fn step_bits(&self) -> Result<crate::device_step::StateBits, EngineError> {
        self.device_step.state_bits()
    }

    /// Starts copying the bf16 verification logits and drafts of each eagerly run device step.
    #[cfg(test)]
    pub(crate) fn start_verification_tap(&mut self, rows: usize) -> Result<(), EngineError> {
        self.device_step.start_tap(rows)
    }

    /// The tapped bf16 logits of the first `rows` rows and the drafts of the last eager step.
    #[cfg(test)]
    pub(crate) fn verification_tap(
        &self,
        rows: usize,
    ) -> Result<(Vec<u16>, Vec<u32>), EngineError> {
        self.device_step.tapped(rows)
    }

    /// Arms draft overwrites consumed by the next device step; see
    /// [`DeviceStep::overwrite_drafts`].
    #[cfg(test)]
    pub(crate) fn overwrite_drafts(&mut self, patches: Vec<(RequestId, Vec<u32>)>) {
        self.device_step.overwrite_drafts(patches);
    }

    /// The device slot a tracked sequence occupies.
    #[cfg(test)]
    pub(crate) fn device_slot(&self, id: RequestId) -> Option<usize> {
        self.device_step.slot(id)
    }

    /// Makes work later enqueued on the decode partition wait for everything already enqueued
    /// on the main stream.
    fn order_decode_after_main(&self) -> Result<(), EngineError> {
        if let Some(decode) = &self.decode_stream {
            let done = self.stream.record_event(None).map_err(exec_err)?;
            decode.wait(&done).map_err(exec_err)?;
        }
        Ok(())
    }

    /// Records the shape and route of every GEMM dispatch on the vision, text and device-step
    /// engines without timing them, counting dispatches made while a graph is captured apart
    /// from executed ones; the merged inventory is written to `path` as TOML when the
    /// executor is dropped. Recording never changes which algorithm runs.
    pub fn record_gemm_shapes(&mut self, path: PathBuf) {
        self.vision_gemm.set_shape_phase("vision");
        self.vision_gemm.set_shape_log(Some(false));
        self.text.gemm_engine().set_shape_log(Some(false));
        self.device_step.gemm_engine().set_shape_log(Some(false));
        self.shape_log_path = Some(path);
    }

    /// The dispatches recorded by every engine since recording started, merged.
    pub fn take_gemm_shapes(&mut self) -> ShapeInventory {
        let mut inv = self.vision_gemm.take_shape_log();
        inv.merge(&self.text.gemm_engine().take_shape_log());
        inv.merge(&self.device_step.gemm_engine().take_shape_log());
        inv
    }

    /// Captures the device-step graphs and returns what they cost.
    pub fn capture_graphs(&mut self) -> Result<CaptureReport, EngineError> {
        self.device_step.capture(
            StepWeights {
                decoder: &self.decoder,
                mtp: self.mtp.as_ref(),
            },
            &mut self.kv,
            self.mtp_kv.as_mut(),
        )
    }

    fn rope(&self, id: RequestId, index: usize) -> [i32; 3] {
        let delta = self.seqs.get(&id).map_or(0, |s| s.delta);
        let p = (index as i64 + delta) as i32;
        [p, p, p]
    }

    /// Prefills `rows`. A prompt whose image runs reference exactly one embedding hands that
    /// embedding's device buffer straight to the text forward pass; with `direct` false, or
    /// with several references, the embeddings are first packed into one zeroed buffer.
    pub(crate) fn prefill_images(
        &mut self,
        rows: &[PrefillRow<GpuEmbedding>],
        direct: bool,
    ) -> Result<Vec<Vec<f32>>, EngineError> {
        let hidden = self.hidden;

        let mut positions: Vec<Vec<[i32; 3]>> = Vec::with_capacity(rows.len());
        let mut deltas: Vec<i64> = Vec::with_capacity(rows.len());
        let mut image_rows: Vec<u32> = Vec::new();
        let mut sources: Vec<(&GpuEmbedding, usize)> = Vec::new();
        let mut offset = 0usize;
        for row in rows {
            let mut spans: Vec<ImageSpan> = Vec::new();
            let mut i = 0usize;
            while i < row.tokens.len() {
                if row.tokens[i] != self.image_token_id {
                    i += 1;
                    continue;
                }
                let start = i;
                while i < row.tokens.len() && row.tokens[i] == self.image_token_id {
                    i += 1;
                }
                let n = spans.len();
                let emb = row.images.get(n).ok_or_else(|| {
                    EngineError::Executor("more image runs than image embeddings".into())
                })?;
                if emb.rows != i - start {
                    return Err(EngineError::Executor(format!(
                        "image run of {} tokens for an embedding of {} rows",
                        i - start,
                        emb.rows
                    )));
                }
                spans.push(ImageSpan {
                    range: start..i,
                    llm_grid: emb.grid,
                });
                image_rows.extend((start..i).map(|r| (offset + r) as u32));
                sources.push((emb.as_ref(), offset + start));
            }
            if spans.len() != row.images.len() {
                return Err(EngineError::Executor(
                    "fewer image runs than image embeddings".into(),
                ));
            }
            let m = mrope_positions(row.tokens.len(), &spans).map_err(exec_err)?;
            positions.push(
                (0..row.tokens.len())
                    .map(|r| {
                        [
                            m.rows[0][r] as i32,
                            m.rows[1][r] as i32,
                            m.rows[2][r] as i32,
                        ]
                    })
                    .collect(),
            );
            deltas.push(m.delta);
            offset += row.tokens.len();
        }
        let source_rows: Vec<usize> = sources.iter().map(|(e, _)| e.rows).collect();
        let feed_plan = ImageFeed::plan(&source_rows, direct);
        let packed = match feed_plan {
            ImageFeed::Packed { .. } => {
                let mut cat = self
                    .stream
                    .alloc_zeros::<u16>(image_rows.len() * hidden)
                    .map_err(exec_err)?;
                let mut at = 0usize;
                for (emb, _) in &sources {
                    let n = emb.rows * hidden;
                    let src = emb.buf.slice(0..n);
                    let mut dst = cat.slice_mut(at..at + n);
                    self.stream.memcpy_dtod(&src, &mut dst).map_err(exec_err)?;
                    at += n;
                }
                Some(cat)
            }
            ImageFeed::None | ImageFeed::Direct => None,
        };
        let embeds: Option<&CudaSlice<u16>> = match feed_plan {
            ImageFeed::None => None,
            ImageFeed::Direct => Some(&sources[0].0.buf),
            ImageFeed::Packed { .. } => packed.as_ref(),
        };
        let prefill: Vec<PrefillSeq<'_>> = rows
            .iter()
            .zip(&positions)
            .map(|(r, p)| PrefillSeq {
                tokens: &r.tokens,
                positions: p,
                block_table: &r.blocks,
            })
            .collect();
        let mut sample = Vec::with_capacity(rows.len());
        let mut end = 0usize;
        for r in rows {
            end += r.tokens.len();
            if r.logits {
                sample.push((end - 1) as u32);
            }
        }
        let keep = self.depth > 0;
        let out = self
            .text
            .forward(
                &self.decoder,
                &mut self.kv,
                &StepInput {
                    prefill: &prefill,
                    decode: &[],
                    image_rows: &image_rows,
                    sample_rows: &sample,
                    keep_hidden: keep,
                },
                embeds,
            )
            .map_err(exec_err)?;
        let mut sampled = if sample.is_empty() {
            Vec::new().into_iter()
        } else {
            self.download_rows(&out.logits, sample.len())?.into_iter()
        };
        let shared = out.hidden.map(Arc::new);
        if let Some(buf) = &shared {
            self.prefill_prompt_layer(rows, &positions, buf)?;
        }
        let mut logits: Vec<Vec<f32>> = Vec::with_capacity(rows.len());
        let mut start = 0usize;
        for (i, row) in rows.iter().enumerate() {
            let feed = shared.as_ref().map(|buf| {
                if row.logits {
                    Feed::Boot {
                        hidden: HiddenRef {
                            buf: Arc::clone(buf),
                            row: start,
                        },
                        next_tokens: row.tokens.iter().skip(1).copied().collect(),
                        positions: positions[i].clone(),
                    }
                } else {
                    Feed::Prompt
                }
            });
            logits.push(if row.logits {
                sampled.next().unwrap_or_default()
            } else {
                Vec::new()
            });
            self.device_step.release(row.seq);
            self.seqs.insert(
                row.seq,
                SeqState {
                    delta: deltas[i],
                    feed,
                },
            );
            start += row.tokens.len();
        }
        Ok(logits)
    }

    fn download_rows(
        &self,
        logits: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<Vec<Vec<f32>>, EngineError> {
        let all = self.stream.clone_dtoh(logits).map_err(exec_err)?;
        Ok(all
            .chunks_exact(self.vocab)
            .take(rows)
            .map(<[f32]>::to_vec)
            .collect())
    }

    fn copy_hidden(
        &self,
        src: &HiddenRef,
        dst: &mut CudaSlice<u16>,
        dst_row: usize,
        rows: usize,
    ) -> Result<(), EngineError> {
        self.ops
            .copy_rows(
                &self.stream,
                &src.buf,
                src.row * self.hidden,
                self.hidden,
                dst,
                dst_row * self.hidden,
                self.hidden,
                rows,
                self.hidden,
            )
            .map_err(exec_err)
    }

    /// Runs the MTP draft chain for the rows and returns each row's drafts.
    fn draft_chain(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<u32>>, EngineError> {
        let b = rows.len();
        let q = self.depth + 1;
        let hidden = self.hidden;
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| EngineError::Executor("missing MTP weights".into()))?;

        // Pass 0: the MTP layer consumes every position it has not seen, boot sequences by
        // prefill from an empty cache and the others as `q` decode rows.
        let mut boot: Vec<usize> = Vec::new();
        let mut steady: Vec<usize> = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            match self.seqs.get(&row.seq).and_then(|s| s.feed.as_ref()) {
                Some(Feed::Boot { .. }) => boot.push(i),
                Some(Feed::Verified { .. }) => steady.push(i),
                Some(Feed::Prompt) | None => {
                    return Err(EngineError::Executor(format!(
                        "sequence {} has no draft state",
                        row.seq
                    )));
                }
            }
        }
        let mut boot_tokens: Vec<Vec<u32>> = Vec::new();
        let mut boot_positions: Vec<Vec<[i32; 3]>> = Vec::new();
        let mut steady_tokens: Vec<Vec<u32>> = Vec::new();
        let mut steady_positions: Vec<Vec<[i32; 3]>> = Vec::new();
        let mut accepted_rows: Vec<usize> = Vec::new();
        for &i in &boot {
            let Some(Feed::Boot {
                next_tokens,
                positions,
                ..
            }) = self.seqs.get(&rows[i].seq).and_then(|s| s.feed.as_ref())
            else {
                unreachable!("classified above");
            };
            let mut t = next_tokens.clone();
            t.push(rows[i].token);
            boot_tokens.push(t);
            boot_positions.push(positions.clone());
        }
        for &i in &steady {
            let Some(Feed::Verified { start, drafts, .. }) =
                self.seqs.get(&rows[i].seq).and_then(|s| s.feed.as_ref())
            else {
                unreachable!("classified above");
            };
            let e = rows[i].position.saturating_sub(*start);
            if e == 0 || e > q {
                return Err(EngineError::Executor(format!(
                    "sequence {} advanced {e} positions in one step (draft depth {})",
                    rows[i].seq, self.depth
                )));
            }
            let mut t = Vec::with_capacity(q);
            for j in 0..q {
                t.push(if j + 1 < e {
                    drafts[j]
                } else if j + 1 == e {
                    rows[i].token
                } else {
                    drafts.get(j).copied().unwrap_or(rows[i].token)
                });
            }
            steady_tokens.push(t);
            steady_positions.push((0..q).map(|j| self.rope(rows[i].seq, start + j)).collect());
            accepted_rows.push(e);
        }
        let boot_rows: usize = boot_tokens.iter().map(Vec::len).sum();
        let total_rows = boot_rows + steady.len() * q;
        let mut hbuf = self
            .stream
            .alloc_zeros::<u16>((total_rows * hidden).max(1))
            .map_err(exec_err)?;
        let mut at = 0usize;
        for (n, &i) in boot.iter().enumerate() {
            let Some(Feed::Boot { hidden: h, .. }) =
                self.seqs.get(&rows[i].seq).and_then(|s| s.feed.as_ref())
            else {
                unreachable!("classified above");
            };
            self.copy_hidden(h, &mut hbuf, at, boot_tokens[n].len())?;
            at += boot_tokens[n].len();
        }
        for &i in &steady {
            let Some(Feed::Verified { hidden: h, .. }) =
                self.seqs.get(&rows[i].seq).and_then(|s| s.feed.as_ref())
            else {
                unreachable!("classified above");
            };
            self.copy_hidden(h, &mut hbuf, at, q)?;
            at += q;
        }
        let tables: Vec<&[u32]> = boot
            .iter()
            .chain(steady.iter())
            .map(|&i| rows[i].blocks.as_slice())
            .collect();
        let prefill: Vec<PrefillSeq<'_>> = boot
            .iter()
            .enumerate()
            .map(|(n, _)| PrefillSeq {
                tokens: &boot_tokens[n],
                positions: &boot_positions[n],
                block_table: tables[n],
            })
            .collect();
        let decode: Vec<DecodeSeq<'_>> = steady
            .iter()
            .enumerate()
            .map(|(n, &i)| {
                let Some(Feed::Verified { start, .. }) =
                    self.seqs.get(&rows[i].seq).and_then(|s| s.feed.as_ref())
                else {
                    unreachable!("classified above");
                };
                DecodeSeq {
                    tokens: &steady_tokens[n],
                    positions: &steady_positions[n],
                    block_table: tables[boot.len() + n],
                    cached: *start,
                }
            })
            .collect();
        let mut sample: Vec<u32> = Vec::with_capacity(b);
        let mut offset = 0usize;
        for t in &boot_tokens {
            sample.push((offset + t.len() - 1) as u32);
            offset += t.len();
        }
        for e in &accepted_rows {
            sample.push((offset + e - 1) as u32);
            offset += q;
        }
        let order: Vec<usize> = boot.iter().chain(steady.iter()).copied().collect();
        let need_hidden = self.depth > 1;
        let mtp_kv = self
            .mtp_kv
            .as_mut()
            .ok_or_else(|| EngineError::Executor("missing MTP KV pool".into()))?;
        let out = self
            .text
            .mtp_forward(
                mtp,
                mtp_kv,
                &StepInput {
                    prefill: &prefill,
                    decode: &decode,
                    image_rows: &[],
                    sample_rows: &sample,
                    keep_hidden: need_hidden,
                },
                &hbuf,
            )
            .map_err(exec_err)?;
        let logits = self.download_rows(&out.logits, b)?;
        let mut drafts: Vec<Vec<u32>> = vec![Vec::with_capacity(self.depth); b];
        for (n, &i) in order.iter().enumerate() {
            drafts[i].push(argmax_with_margin(&logits[n]).token);
        }
        // Carry buffer in `rows` order: the MTP layer output at each sequence's sampled row.
        let mut carry = if need_hidden {
            let mut c = self
                .stream
                .alloc_zeros::<u16>((b * hidden).max(1))
                .map_err(exec_err)?;
            let full = out
                .hidden
                .as_ref()
                .ok_or_else(|| EngineError::Executor("MTP forward returned no hidden".into()))?;
            for (n, &i) in order.iter().enumerate() {
                self.ops
                    .copy_rows(
                        &self.stream,
                        full,
                        sample[n] as usize * hidden,
                        hidden,
                        &mut c,
                        i * hidden,
                        hidden,
                        1,
                        hidden,
                    )
                    .map_err(exec_err)?;
            }
            Some(c)
        } else {
            None
        };
        drop(out);

        // Passes 1..depth: one row per sequence, the previous draft with the layer's own output.
        for pass in 1..self.depth {
            let tokens: Vec<[u32; 1]> = drafts.iter().map(|d| [d[pass - 1]]).collect();
            let positions: Vec<[[i32; 3]; 1]> = rows
                .iter()
                .map(|r| [self.rope(r.seq, r.position - 1 + pass)])
                .collect();
            let decode: Vec<DecodeSeq<'_>> = rows
                .iter()
                .enumerate()
                .map(|(i, r)| DecodeSeq {
                    tokens: &tokens[i],
                    positions: &positions[i],
                    block_table: &r.blocks,
                    cached: r.position - 1 + pass,
                })
                .collect();
            let sample: Vec<u32> = (0..b as u32).collect();
            let keep = pass + 1 < self.depth;
            let mtp_kv = self
                .mtp_kv
                .as_mut()
                .ok_or_else(|| EngineError::Executor("missing MTP KV pool".into()))?;
            let out = self
                .text
                .mtp_forward(
                    mtp,
                    mtp_kv,
                    &StepInput {
                        prefill: &[],
                        decode: &decode,
                        image_rows: &[],
                        sample_rows: &sample,
                        keep_hidden: keep,
                    },
                    carry
                        .as_ref()
                        .ok_or_else(|| EngineError::Executor("missing draft carry".into()))?,
                )
                .map_err(exec_err)?;
            let logits = self.download_rows(&out.logits, b)?;
            for (i, row) in logits.iter().enumerate() {
                drafts[i].push(argmax_with_margin(row).token);
            }
            carry = out.hidden;
        }
        Ok(drafts)
    }

    /// Runs the MTP draft chain and the verification pass for the rows.
    fn speculate(&mut self, rows: &[DecodeRow]) -> Result<Vec<SpecRowOutput>, EngineError> {
        let b = rows.len();
        let q = self.depth + 1;
        let mut drafts = self.draft_chain(rows)?;

        // Verification: the last token and its drafts through the main model.
        let tokens: Vec<Vec<u32>> = rows
            .iter()
            .zip(&drafts)
            .map(|(r, d)| std::iter::once(r.token).chain(d.iter().copied()).collect())
            .collect();
        let positions: Vec<Vec<[i32; 3]>> = rows
            .iter()
            .map(|r| (0..q).map(|j| self.rope(r.seq, r.position + j)).collect())
            .collect();
        let decode: Vec<DecodeSeq<'_>> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| DecodeSeq {
                tokens: &tokens[i],
                positions: &positions[i],
                block_table: &r.blocks,
                cached: r.position,
            })
            .collect();
        let sample: Vec<u32> = (0..(b * q) as u32).collect();
        let out = self
            .text
            .forward(
                &self.decoder,
                &mut self.kv,
                &StepInput {
                    prefill: &[],
                    decode: &decode,
                    image_rows: &[],
                    sample_rows: &sample,
                    keep_hidden: true,
                },
                None,
            )
            .map_err(exec_err)?;
        let flat = self.download_rows(&out.logits, b * q)?;
        let verified = Arc::new(
            out.hidden
                .ok_or_else(|| EngineError::Executor("verify returned no hidden".into()))?,
        );
        let mut results = Vec::with_capacity(b);
        let mut flat = flat.into_iter();
        for (i, row) in rows.iter().enumerate() {
            let logits: Vec<Vec<f32>> = flat.by_ref().take(q).collect();
            if let Some(state) = self.seqs.get_mut(&row.seq) {
                state.feed = Some(Feed::Verified {
                    hidden: HiddenRef {
                        buf: Arc::clone(&verified),
                        row: i * q,
                    },
                    start: row.position,
                    drafts: drafts[i].clone(),
                });
            }
            results.push(SpecRowOutput {
                drafts: std::mem::take(&mut drafts[i]),
                logits,
            });
        }
        Ok(results)
    }
}

impl GpuExecutor {
    /// Runs the MTP layer over the prompt positions of the rows that continue on the device
    /// step, filling its KV for every position except the last; no draft head runs and nothing
    /// returns to the host. `hidden` holds the main model's final hidden state of every prefill
    /// row.
    fn prefill_prompt_layer(
        &mut self,
        rows: &[PrefillRow<GpuEmbedding>],
        positions: &[Vec<[i32; 3]>],
        hidden: &Arc<CudaSlice<u16>>,
    ) -> Result<(), EngineError> {
        let width = self.hidden;
        let mut picked: Vec<(usize, usize)> = Vec::new();
        let mut start = 0usize;
        let mut total = 0usize;
        for (i, row) in rows.iter().enumerate() {
            if !row.logits && row.tokens.len() > 1 {
                picked.push((i, start));
                total += row.tokens.len() - 1;
            }
            start += row.tokens.len();
        }
        if picked.is_empty() {
            return Ok(());
        }
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| EngineError::Executor("missing MTP weights".into()))?;
        let mut hbuf = self
            .stream
            .alloc_zeros::<u16>((total * width).max(1))
            .map_err(exec_err)?;
        let mut at = 0usize;
        for &(i, first) in &picked {
            let n = rows[i].tokens.len() - 1;
            let src = HiddenRef {
                buf: Arc::clone(hidden),
                row: first,
            };
            self.copy_hidden(&src, &mut hbuf, at, n)?;
            at += n;
        }
        let prefill: Vec<PrefillSeq<'_>> = picked
            .iter()
            .map(|&(i, _)| PrefillSeq {
                tokens: &rows[i].tokens[1..],
                positions: &positions[i][..rows[i].tokens.len() - 1],
                block_table: &rows[i].blocks,
            })
            .collect();
        let mtp_kv = self
            .mtp_kv
            .as_mut()
            .ok_or_else(|| EngineError::Executor("missing MTP KV pool".into()))?;
        self.text
            .mtp_forward(
                mtp,
                mtp_kv,
                &StepInput {
                    prefill: &prefill,
                    decode: &[],
                    image_rows: &[],
                    sample_rows: &[],
                    keep_hidden: false,
                },
                &hbuf,
            )
            .map_err(exec_err)?;
        Ok(())
    }
}

impl ModelExecutor for GpuExecutor {
    fn trim_memory(&mut self) {
        if self.stream.synchronize().is_ok() {
            let _ = scriptor_cuda::device::trim_context_pool(&self.ctx, self.plan_total as usize);
        }
    }

    type Embedding = GpuEmbedding;

    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<GpuEmbedding>, EngineError> {
        let token = self.vision_begin(jobs)?;
        self.vision_finish(token)
    }

    fn vision_saturation_patches(&self) -> usize {
        self.vision_saturation
    }

    fn prefill_saturation_tokens(&self) -> usize {
        self.prefill_saturation
    }

    fn vision_prepare(&mut self, jobs: &[VisionJob]) -> Result<(), EngineError> {
        let bits = stage_patches(jobs)?;
        self.staged = Some(StagedVision {
            keys: jobs.iter().map(|j| (j.seq, j.image)).collect(),
            bits,
        });
        Ok(())
    }

    fn supports_async_vision(&self) -> bool {
        true
    }

    fn vision_begin(&mut self, jobs: &[VisionJob]) -> Result<VisionToken, EngineError> {
        let keys: Vec<(RequestId, usize)> = jobs.iter().map(|j| (j.seq, j.image)).collect();
        let bits = match self.staged.take() {
            Some(staged) if staged.keys == keys => staged.bits,
            _ => stage_patches(jobs)?,
        };
        let grids: Vec<(usize, usize, usize)> = jobs
            .iter()
            .map(|j| (j.grid.t as usize, j.grid.h as usize, j.grid.w as usize))
            .collect();
        let mut outs: Vec<CudaSlice<u16>> = Vec::with_capacity(jobs.len());
        for job in jobs {
            outs.push(
                self.stream
                    .alloc_zeros::<u16>((job.patches / 4 * self.hidden).max(1))
                    .map_err(exec_err)?,
            );
        }
        let allocated = self.stream.record_event(None).map_err(exec_err)?;
        self.vision_stream.wait(&allocated).map_err(exec_err)?;
        let mut probe = None;
        if self.l1_vision {
            let start = Arc::new(AtomicU64::new(0));
            let ready = Arc::new(AtomicU64::new(0));
            if self.l1_callbacks
                && let Err(e) = launch_stamp(&self.vision_stream, &start)
            {
                eprintln!("scriptor: l1 host callbacks unavailable: {e}");
                self.l1_callbacks = false;
            }
            let copy_from = self
                .vision_stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(exec_err)?;
            probe = Some((copy_from, start, ready));
        }
        let input = self.vision_stream.clone_htod(&bits).map_err(exec_err)?;
        let probe = match probe {
            Some((copy_from, start, ready)) => {
                let copy_to = self
                    .vision_stream
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(exec_err)?;
                Some(VisionProbe {
                    copy_from,
                    copy_to,
                    start,
                    ready,
                })
            }
            None => None,
        };
        {
            let mut refs: Vec<&mut CudaSlice<u16>> = outs.iter_mut().collect();
            self.vision
                .run(
                    &self.vision_stream,
                    &mut self.vision_gemm,
                    &mut self.vision_ws,
                    &input,
                    &grids,
                    &mut refs,
                )
                .map_err(exec_err)?;
        }
        if let (Some(p), true) = (probe.as_ref(), self.l1_callbacks)
            && let Err(e) = launch_stamp(&self.vision_stream, &p.ready)
        {
            eprintln!("scriptor: l1 host callbacks unavailable: {e}");
            self.l1_callbacks = false;
        }
        let done = self
            .vision_stream
            .record_event(probe.is_some().then_some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(exec_err)?;
        let token = self.next_vision;
        self.next_vision += 1;
        self.vision_pending.insert(
            token,
            VisionPending {
                done,
                l1: probe,
                outs,
                grids,
                _input: input,
            },
        );
        Ok(token)
    }

    fn vision_ready(&mut self, token: VisionToken) -> Result<bool, EngineError> {
        match self.vision_pending.get(&token) {
            Some(p) => p.done.try_is_complete().map_err(exec_err),
            None => Err(EngineError::Executor("unknown vision batch".into())),
        }
    }

    fn vision_finish(&mut self, token: VisionToken) -> Result<Vec<GpuEmbedding>, EngineError> {
        let pending = self
            .vision_pending
            .remove(&token)
            .ok_or_else(|| EngineError::Executor("unknown vision batch".into()))?;
        pending.done.synchronize().map_err(exec_err)?;
        if let Some(p) = &pending.l1 {
            let ns = |ms: f32| (f64::from(ms) * 1e6).max(0.0) as u64;
            let copy_ns = p.copy_from.elapsed_ms(&p.copy_to).map_or(0, ns);
            let compute_ns = p.copy_to.elapsed_ms(&pending.done).map_or(0, ns);
            let stamp = |s: &AtomicU64| Some(s.load(Ordering::Acquire)).filter(|&v| v > 0);
            self.l1_timing.insert(
                token,
                scriptor_engine::l1_counters::VisionTiming {
                    copy_ns,
                    device_ns: copy_ns + compute_ns,
                    start_host_ns: stamp(&p.start),
                    ready_host_ns: stamp(&p.ready),
                },
            );
        }
        Ok(pending
            .outs
            .into_iter()
            .zip(&pending.grids)
            .map(|(buf, &(t, h, w))| GpuEmbedding {
                buf,
                rows: t * h * w / 4,
                hidden: self.hidden,
                grid: (t, h / 2, w / 2),
            })
            .collect())
    }

    fn prefill(&mut self, rows: &[PrefillRow<GpuEmbedding>]) -> Result<Vec<Vec<f32>>, EngineError> {
        self.prefill_images(rows, true)
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        let tokens: Vec<[u32; 1]> = rows.iter().map(|r| [r.token]).collect();
        let positions: Vec<[[i32; 3]; 1]> = rows
            .iter()
            .map(|r| [self.rope(r.seq, r.position)])
            .collect();
        let decode: Vec<DecodeSeq<'_>> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| DecodeSeq {
                tokens: &tokens[i],
                positions: &positions[i],
                block_table: &r.blocks,
                cached: r.position,
            })
            .collect();
        let sample: Vec<u32> = (0..rows.len() as u32).collect();
        let out = self
            .text
            .forward(
                &self.decoder,
                &mut self.kv,
                &StepInput {
                    prefill: &[],
                    decode: &decode,
                    image_rows: &[],
                    sample_rows: &sample,
                    keep_hidden: false,
                },
                None,
            )
            .map_err(exec_err)?;
        self.download_rows(&out.logits, rows.len())
    }

    fn release(&mut self, seq: RequestId) {
        self.seqs.remove(&seq);
        self.device_step.release(seq);
    }

    fn supports_device_step(&self) -> bool {
        true
    }

    fn lookahead_tokens(&self) -> usize {
        self.device_step.lookahead()
    }

    fn graph_stats(&self) -> GraphStats {
        self.device_step.stats()
    }

    fn decode_device(&mut self, rows: &[DeviceRow<'_>]) -> Result<Vec<StepRowResult>, EngineError> {
        self.decode_device_begin(rows)?;
        self.decode_device_finish()
    }

    fn decode_device_begin(&mut self, rows: &[DeviceRow<'_>]) -> Result<bool, EngineError> {
        let fresh: Vec<usize> = (0..rows.len())
            .filter(|&i| !self.device_step.tracks(rows[i].row.seq))
            .collect();
        if !fresh.is_empty() {
            let mut chained: Vec<usize> = Vec::new();
            for (n, &i) in fresh.iter().enumerate() {
                match self
                    .seqs
                    .get(&rows[i].row.seq)
                    .and_then(|s| s.feed.as_ref())
                {
                    Some(Feed::Boot { .. } | Feed::Verified { .. }) => chained.push(n),
                    Some(Feed::Prompt) => {}
                    None if self.depth == 0 => {}
                    None => {
                        return Err(EngineError::Executor(format!(
                            "sequence {} has no draft state",
                            rows[i].row.seq
                        )));
                    }
                }
            }
            let mut drafts: Vec<Vec<u32>> = vec![Vec::new(); fresh.len()];
            if !chained.is_empty() {
                let booting: Vec<DecodeRow> = chained
                    .iter()
                    .map(|&n| rows[fresh[n]].row.clone())
                    .collect();
                for (&n, d) in chained.iter().zip(self.draft_chain(&booting)?) {
                    drafts[n] = d;
                }
            }
            for (n, &i) in fresh.iter().enumerate() {
                let id = rows[i].row.seq;
                let delta = self.seqs.get(&id).map_or(0, |s| s.delta);
                self.device_step.adopt(id, &rows[i], delta, &drafts[n])?;
                if let Some(state) = self.seqs.get_mut(&id) {
                    state.feed = None;
                }
            }
        }
        let active: Vec<ActiveRow<'_>> = rows
            .iter()
            .map(|r| ActiveRow {
                slot: self.device_step.slot(r.row.seq).expect("adopted above"),
                source: r,
            })
            .collect();
        self.order_decode_after_main()?;
        self.device_step.launch(
            StepWeights {
                decoder: &self.decoder,
                mtp: self.mtp.as_ref(),
            },
            &mut self.kv,
            self.mtp_kv.as_mut(),
            &active,
        )?;
        Ok(true)
    }

    fn decode_device_finish(&mut self) -> Result<Vec<StepRowResult>, EngineError> {
        self.device_step.collect()
    }

    fn take_host_sample(&mut self) -> Option<scriptor_engine::l1_counters::StepHostSample> {
        self.device_step.take_host_sample()
    }

    fn take_vision_timing(
        &mut self,
        token: VisionToken,
    ) -> Option<scriptor_engine::l1_counters::VisionTiming> {
        self.l1_timing.remove(&token)
    }

    fn draft_depth(&self) -> usize {
        self.depth
    }

    fn decode_spec(&mut self, rows: &[DecodeRow]) -> Result<Vec<SpecRowOutput>, EngineError> {
        self.speculate(rows)
    }
}
