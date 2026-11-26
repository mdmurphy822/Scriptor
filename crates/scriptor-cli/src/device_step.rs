//! The device-resident decode step: main pass, token choice, draft acceptance and the MTP draft
//! chain all run on the device from persistent per-sequence state, and only the emitted tokens
//! and their margins return to the host. One CUDA graph per batch bucket replays the whole step.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream, PinnedHostSlice};
use scriptor_cache::BlockId;
use scriptor_cuda::{CapturedGraph, CudaError, DeviceArena};
use scriptor_engine::{DeviceRow, EngineError, GraphStats, RequestId, StepRowResult};
use scriptor_kernels::cublaslt::{GemmEngine, GemmOptions};
use scriptor_kernels::heuristics::TuningTable;
use scriptor_kernels::sampling::{
    RowParams, SamplingKernels, SplitScratch, SplitSelect, mask_words, rows_all_greedy,
};
use scriptor_kernels::step::{RowState, StepKernels, result_width};
use scriptor_kernels::verify::MtpAcceptKernel;
use scriptor_model::gpu::{
    GpuDecoderWeights, GpuKvPool, GpuMtpWeights, StepPlan, TextExecutor, TextGeometry,
};
use scriptor_spec::{StepShape, graph_buckets, step_shape};

use crate::gpu_executor::exec_err;
use crate::slot_book::SlotBook;
use crate::state_scratch::StateScratch;
use crate::table_mirror::TableMirror;

/// Key of a captured step graph: the bucket and whether its token choice is the split selector.
type GraphKey = (usize, bool);

/// Whether a step takes the split selector: the selector is enabled and serves the step's row
/// count, the logits are bf16, and every active row takes the argmax on the device. The
/// decision uses only host-side row parameters, so it needs no device readback.
fn step_uses_split<'a>(
    select: Option<&SplitSelect>,
    wide: bool,
    served: bool,
    active: impl IntoIterator<Item = &'a RowParams>,
) -> bool {
    select.is_some() && !wide && served && rows_all_greedy(active)
}

/// The captured graph to replay for a step of `bucket` rows that wants the split selector when
/// `split`: the split graph when it was captured, else the single-block graph.
fn replay_key(captured: impl Fn(GraphKey) -> bool, bucket: usize, split: bool) -> Option<GraphKey> {
    [(bucket, true), (bucket, false)]
        .into_iter()
        .filter(|&(_, s)| split || !s)
        .find(|&key| captured(key))
}

/// Bounds of the sequences the device step serves.
#[derive(Debug, Clone, Copy)]
pub struct StepLimits {
    /// Concurrent sequences, one device slot each.
    pub max_seqs: usize,
    /// Longest sequence in tokens.
    pub max_context: usize,
}

/// Weights one step reads.
#[derive(Clone, Copy)]
pub struct StepWeights<'a> {
    pub decoder: &'a GpuDecoderWeights,
    pub mtp: Option<&'a GpuMtpWeights>,
}

/// One sequence of a step with its device slot.
pub struct ActiveRow<'a> {
    pub slot: usize,
    pub source: &'a DeviceRow<'a>,
}

/// Which graph baseline a capture started from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureScope {
    /// No graphs were resident when the capture began.
    Initial,
    /// Graphs from an earlier capture were resident and were destroyed before the baseline
    /// free-memory sample.
    Recapture,
}

/// What a pair of free-memory samples around a capture says about its device memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryReading {
    /// The driver could not report free memory at one of the samples; nothing is claimed.
    Unavailable,
    /// Free memory did not decrease; the sample is too coarse or shared to attribute any
    /// bytes, and no zero-residency claim is made.
    NoObservedGrowth,
    /// Free device memory decreased by this many bytes across the sampled span. The value is an
    /// observed delta of a shared device, not an allocation receipt.
    Observed(u64),
}

impl MemoryReading {
    /// Reading from the free-memory samples taken before and after a span.
    pub fn from_samples(before: Option<usize>, after: Option<usize>) -> Self {
        match (before, after) {
            (Some(before), Some(after)) => match before.saturating_sub(after) {
                0 => Self::NoObservedGrowth,
                bytes => Self::Observed(bytes as u64),
            },
            _ => Self::Unavailable,
        }
    }

    /// Reading of two consecutive spans: observed bytes add, and an unavailable span makes the
    /// whole unavailable.
    pub fn then(self, next: Self) -> Self {
        match (self, next) {
            (Self::Unavailable, _) | (_, Self::Unavailable) => Self::Unavailable,
            (Self::Observed(a), Self::Observed(b)) => Self::Observed(a.saturating_add(b)),
            (Self::Observed(a), Self::NoObservedGrowth)
            | (Self::NoObservedGrowth, Self::Observed(a)) => Self::Observed(a),
            (Self::NoObservedGrowth, Self::NoObservedGrowth) => Self::NoObservedGrowth,
        }
    }

    /// Observed bytes, or zero when nothing was observed; zero is not a residency claim.
    #[cfg(test)]
    pub fn bytes(self) -> u64 {
        match self {
            Self::Observed(bytes) => bytes,
            _ => 0,
        }
    }
}

/// Rows of the split selector's scratch: the configured `max_rows` cap bounded by the most
/// verification rows one step holds, `max_seqs * q`.
pub fn split_scratch_rows(max_rows: usize, max_seqs: usize, q: usize) -> usize {
    max_rows.min(max_seqs.saturating_mul(q))
}

/// Bytes requested for the split selector's partial buffer holding `rows` rows.
pub fn split_scratch_bytes(rows: usize) -> u64 {
    (scriptor_kernels::sampling::split_scratch_words(rows) * std::mem::size_of::<u32>()) as u64
}

/// What capturing the step graphs produced.
///
/// The memory fields are observed free-memory deltas of the device around the capture, scoped
/// as described per field; they make no claim of budget completeness and are not charged to
/// any admission budget. Selector scratch is reported separately and is never part of them.
#[derive(Debug, Clone, Copy)]
pub struct CaptureReport {
    /// Graphs captured, split-selector variants included.
    pub graphs: usize,
    /// Wall-clock capture time of every graph, warm-ups included.
    pub seconds: f64,
    /// Observed free-memory change across warm-up and capture of every graph, sampled after
    /// any earlier graphs were destroyed. It excludes the selector scratch, which was allocated
    /// before the capture.
    pub memory: MemoryReading,
    /// Whether the capture replaced earlier graphs.
    pub scope: CaptureScope,
    /// Graphs whose token choice is the split selector.
    pub split_graphs: usize,
    /// The share of `seconds` spent on split-selector graphs.
    pub split_seconds: f64,
    /// Observed free-memory change summed over the split-selector variants' warm-up and
    /// capture only, including any lazy or unrelated allocation in those spans.
    pub split_memory: MemoryReading,
    /// Bytes requested for the selector's partial buffer, from its row capacity; zero when the
    /// selector is off. Not included in `bytes`.
    pub split_scratch_bytes: u64,
    /// Rows of that buffer. Rows are verification rows (`seqs * (k + 1)`), not active requests.
    pub split_scratch_rows: usize,
    /// Buckets that also have a selector graph: their verification rows are within the
    /// selector's row bound and scratch.
    pub split_served_buckets: usize,
    /// Buckets served only by the single-block kernel because the selector is off, the logits
    /// are wide, or their verification rows exceed the row bound.
    pub fallback_buckets: usize,
}

/// Raw buffer contents of the device-resident step, compared bit for bit by tests.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub struct StateBits {
    pub logits: Vec<u32>,
    pub hidden: Vec<u16>,
    pub targets: Vec<u32>,
    pub margins: Vec<u32>,
}

/// Device buffers, slot bookkeeping and graphs of the device-resident step.
pub struct DeviceStep {
    stream: Arc<CudaStream>,
    text: TextExecutor,
    plan: StepPlan,
    sampling: SamplingKernels,
    accept: MtpAcceptKernel,
    step: StepKernels,
    k: usize,
    q: usize,
    vocab: usize,
    words: usize,
    stride: usize,
    block_size: usize,
    max_seqs: usize,
    max_ctx: usize,
    cached: CudaSlice<i32>,
    limit: CudaSlice<i32>,
    delta: CudaSlice<i32>,
    last: CudaSlice<u32>,
    params: CudaSlice<RowParams>,
    neutral: CudaSlice<RowParams>,
    prompt_bits: CudaSlice<u32>,
    output_bits: CudaSlice<u32>,
    out_counts: CudaSlice<u32>,
    drafts: CudaSlice<u32>,
    draft_counts: CudaSlice<i32>,
    counts_want: Vec<i32>,
    counts_dev: Vec<i32>,
    logits: CudaSlice<f32>,
    wide_logits: bool,
    hidden: CudaSlice<u16>,
    carry: [CudaSlice<u16>; 2],
    targets: CudaSlice<u32>,
    margins: CudaSlice<f32>,
    emitted: CudaSlice<u32>,
    n_emitted: CudaSlice<u32>,
    mark_slots: CudaSlice<u32>,
    draft_tok: CudaSlice<u32>,
    draft_margin: CudaSlice<f32>,
    result: CudaSlice<u32>,
    pinned: PinnedHostSlice<u32>,
    inflight: Option<Vec<usize>>,
    slots: SlotBook,
    tables: TableMirror,
    host_delta: Vec<i32>,
    scratch: StateScratch,
    buckets: Vec<usize>,
    graphs: HashMap<GraphKey, CapturedGraph>,
    split: Option<SplitSelect>,
    split_scratch: Option<SplitScratch>,
    warned: HashSet<usize>,
    stats: GraphStats,
    timer: Option<[CudaEvent; 2]>,
    timed: Option<usize>,
    host_probe: bool,
    launch_cpu: u32,
    host_sample: Option<scriptor_engine::l1_counters::StepHostSample>,
    active_telemetry: bool,
    #[cfg(test)]
    tap: Option<VerifyTap>,
    #[cfg(test)]
    draft_patch: Vec<(RequestId, Vec<u32>)>,
}

/// Copy of the verification logits and drafts of an eagerly run step.
#[cfg(test)]
struct VerifyTap {
    logits: CudaSlice<u16>,
    drafts: CudaSlice<u32>,
}

/// The CPU the calling thread is running on; zero when the kernel does not report it.
fn current_cpu() -> u32 {
    // SAFETY: `sched_getcpu` takes no arguments and only reads scheduler state.
    let cpu = unsafe { libc::sched_getcpu() };
    u32::try_from(cpu).unwrap_or(0)
}

/// CPU time of the calling thread in nanoseconds; zero when the clock is unavailable.
fn thread_cpu_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable `timespec` for the duration of the call.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if rc != 0 {
        return 0;
    }
    (ts.tv_sec as u64) * 1_000_000_000 + ts.tv_nsec as u64
}

impl DeviceStep {
    /// Allocates every buffer for `limits` at draft depth `k`. The step owns its own executor
    /// so its scratch buffers keep fixed addresses.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        arena: &mut DeviceArena,
        geometry: TextGeometry,
        sm_count: usize,
        tuning: TuningTable,
        options: GemmOptions,
        kv: &GpuKvPool,
        k: usize,
        limits: StepLimits,
    ) -> Result<Self, EngineError> {
        let seqs = limits.max_seqs.max(1);
        let wide_logits = std::env::var("SCRIPTOR_SAMPLING_LOGITS").as_deref() == Ok("f32");
        let split = std::env::var("SCRIPTOR_SPLIT_SELECT")
            .ok()
            .and_then(|v| SplitSelect::parse(&v));
        let q = k + 1;
        let rows = seqs * q;
        let block_size = kv.block_size();
        let stride = (limits.max_context + 2 * k).div_ceil(block_size) + 1;
        let words = mask_words(geometry.vocab);
        let hidden = geometry.hidden;
        let mut text = TextExecutor::new(ctx, stream, arena, geometry, sm_count, tuning, options)
            .map_err(exec_err)?;
        let max_ctx = stride * block_size;
        let buckets = graph_buckets(seqs);
        let shapes: Vec<(usize, usize)> = buckets.iter().flat_map(|&b| [(b, q), (b, 1)]).collect();
        text.reserve_resident(&shapes, max_ctx).map_err(exec_err)?;
        let plan = StepPlan::new(stream, kv, rows, seqs, stride).map_err(exec_err)?;
        let z32 = |n: usize| stream.alloc_zeros::<i32>(n.max(1)).map_err(exec_err);
        let zu = |n: usize| stream.alloc_zeros::<u32>(n.max(1)).map_err(exec_err);
        let zf = |n: usize| stream.alloc_zeros::<f32>(n.max(1)).map_err(exec_err);
        let zh = |n: usize| stream.alloc_zeros::<u16>(n.max(1)).map_err(exec_err);
        let idle = vec![-1_i32; seqs];
        let mut cached = z32(seqs)?;
        stream.memcpy_htod(&idle, &mut cached).map_err(exec_err)?;
        let neutral_host: Vec<RowParams> = (0..seqs as u32).map(RowParams::greedy).collect();
        let neutral = stream.clone_htod(&neutral_host).map_err(exec_err)?;
        let params = stream.clone_htod(&neutral_host).map_err(exec_err)?;
        let width = result_width(q);
        // SAFETY: the buffer is fully written below before any read.
        let mut pinned =
            unsafe { ctx.alloc_pinned_with_flags::<u32>(seqs * width, 0) }.map_err(exec_err)?;
        pinned.as_mut_slice().map_err(exec_err)?.fill(0);
        let draft_counts = stream.clone_htod(&vec![k as i32; seqs]).map_err(exec_err)?;
        let timer = if std::env::var("SCRIPTOR_DECODE_STEP_TIMING").as_deref() == Ok("1") {
            let event = || {
                ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(exec_err)
            };
            Some([event()?, event()?])
        } else {
            None
        };
        let mut step = Self {
            stream: stream.clone(),
            text,
            plan,
            sampling: SamplingKernels::load(ctx).map_err(exec_err)?,
            accept: MtpAcceptKernel::load(ctx).map_err(exec_err)?,
            step: StepKernels::load(ctx).map_err(exec_err)?,
            k,
            q,
            vocab: geometry.vocab,
            words,
            stride,
            block_size,
            max_seqs: seqs,
            max_ctx,
            cached,
            limit: z32(seqs)?,
            delta: z32(seqs)?,
            last: zu(seqs)?,
            params,
            neutral,
            prompt_bits: zu(seqs * words)?,
            output_bits: zu(seqs * words)?,
            out_counts: zu(seqs * geometry.vocab)?,
            drafts: zu(seqs * k)?,
            draft_counts,
            counts_want: vec![k as i32; seqs],
            counts_dev: vec![k as i32; seqs],
            logits: zf(if wide_logits {
                rows * geometry.vocab
            } else {
                1
            })?,
            wide_logits,
            hidden: zh(rows * hidden)?,
            carry: [zh(seqs * hidden)?, zh(seqs * hidden)?],
            targets: zu(rows)?,
            margins: zf(rows)?,
            emitted: zu(rows)?,
            n_emitted: zu(seqs)?,
            mark_slots: zu(rows)?,
            draft_tok: zu(seqs)?,
            draft_margin: zf(seqs)?,
            result: zu(seqs * width)?,
            pinned,
            inflight: None,
            slots: SlotBook::new(seqs),
            tables: TableMirror::new(seqs, stride),
            host_delta: vec![0; seqs],
            scratch: StateScratch::new(seqs),
            buckets,
            graphs: HashMap::new(),
            split: None,
            split_scratch: None,
            warned: HashSet::new(),
            stats: GraphStats::default(),
            timer,
            timed: None,
            host_probe: scriptor_engine::l1_counters::switches_from_env()
                .unwrap_or_default()
                .counters,
            launch_cpu: 0,
            host_sample: None,
            active_telemetry: scriptor_engine::active_telemetry::enable_from_env(),
            #[cfg(test)]
            tap: None,
            #[cfg(test)]
            draft_patch: Vec::new(),
        };
        step.set_split_select(split)?;
        Ok(step)
    }

    /// Sets the two-stage greedy target selector used for batches of verified greedy rows; `None`
    /// keeps the single-block kernel everywhere. Allocates the selector's partial buffer, so it
    /// must run before [`Self::capture`]; captured graphs are dropped and must be recaptured.
    /// The selector is experimental and off by default; its memory is reported by
    /// [`CaptureReport`] and is not charged to any admission budget.
    pub fn set_split_select(&mut self, select: Option<SplitSelect>) -> Result<(), EngineError> {
        self.graphs.clear();
        self.stats.captured = 0;
        self.split = select.filter(SplitSelect::is_valid);
        self.split_scratch = match &self.split {
            Some(s) => Some(
                self.sampling
                    .alloc_split_scratch(
                        &self.stream,
                        split_scratch_rows(s.max_rows, self.max_seqs, self.q),
                    )
                    .map_err(exec_err)?,
            ),
            None => None,
        };
        Ok(())
    }

    /// Whether the split selector serves a step of `bucket` rows.
    fn split_serves(&self, bucket: usize) -> bool {
        match (&self.split, &self.split_scratch) {
            (Some(select), Some(scratch)) => {
                self.sampling
                    .split_serves(select, bucket * self.q, scratch.rows())
            }
            _ => false,
        }
    }

    /// Directs every later upload, launch and graph capture to `stream`. Graphs already
    /// captured keep the execution context they were captured with.
    pub fn set_stream(&mut self, stream: &Arc<CudaStream>) {
        self.stream = stream.clone();
        self.text.set_stream(stream);
    }

    /// The execution-context handle of every kernel node of every captured graph.
    #[cfg(test)]
    pub fn graph_node_contexts(&self) -> Result<Vec<usize>, EngineError> {
        let mut all = Vec::new();
        for graph in self.graphs.values() {
            all.extend(graph.kernel_node_contexts().map_err(exec_err)?);
        }
        Ok(all)
    }

    /// The raw bits of the fp32 logits buffer (empty unless wide logits are on), the hidden
    /// states and the emitted targets and margins of the last step.
    #[cfg(test)]
    pub fn state_bits(&self) -> Result<StateBits, EngineError> {
        self.stream.synchronize().map_err(exec_err)?;
        Ok(StateBits {
            logits: self
                .stream
                .clone_dtoh(&self.logits)
                .map_err(exec_err)?
                .iter()
                .map(|v| v.to_bits())
                .collect(),
            hidden: self.stream.clone_dtoh(&self.hidden).map_err(exec_err)?,
            targets: self.stream.clone_dtoh(&self.targets).map_err(exec_err)?,
            margins: self
                .stream
                .clone_dtoh(&self.margins)
                .map_err(exec_err)?
                .iter()
                .map(|v| v.to_bits())
                .collect(),
        })
    }

    /// Starts copying the bf16 verification logits and the drafts of every eagerly run step
    /// into buffers sized for `rows` rows, for [`Self::tapped`].
    #[cfg(test)]
    pub fn start_tap(&mut self, rows: usize) -> Result<(), EngineError> {
        self.tap = Some(VerifyTap {
            logits: self
                .stream
                .alloc_zeros::<u16>((rows * self.vocab).max(1))
                .map_err(exec_err)?,
            drafts: self
                .stream
                .alloc_zeros::<u32>((self.max_seqs * self.k).max(1))
                .map_err(exec_err)?,
        });
        Ok(())
    }

    /// Arms draft overwrites for the next launch: each entry replaces the `k` drafts of the
    /// sequence's slot after the step state is uploaded and before the step runs. The next launch
    /// fails if an entry names an untracked sequence, has a length other than `k`, or targets a
    /// slot whose proposed draft count is below `k`.
    #[cfg(test)]
    pub fn overwrite_drafts(&mut self, patches: Vec<(RequestId, Vec<u32>)>) {
        self.draft_patch = patches;
    }

    /// Writes the armed draft overwrites into the device draft buffer and clears them.
    #[cfg(test)]
    fn apply_draft_patch(&mut self) -> Result<(), EngineError> {
        for (id, drafts) in std::mem::take(&mut self.draft_patch) {
            let slot = self
                .slots
                .slot_of(id)
                .ok_or_else(|| EngineError::Executor(format!("draft patch for untracked {id}")))?;
            if drafts.len() != self.k || self.counts_want[slot] != self.k as i32 {
                return Err(EngineError::Executor(format!(
                    "draft patch of {} drafts for slot {slot} with {} proposed, depth {}",
                    drafts.len(),
                    self.counts_want[slot],
                    self.k
                )));
            }
            let mut view = self.drafts.slice_mut(slot * self.k..(slot + 1) * self.k);
            self.stream
                .memcpy_htod(&drafts, &mut view)
                .map_err(exec_err)?;
        }
        Ok(())
    }

    /// The tapped bf16 logits of the first `rows` rows and all drafts of the last eager step.
    #[cfg(test)]
    pub fn tapped(&self, rows: usize) -> Result<(Vec<u16>, Vec<u32>), EngineError> {
        let tap = self
            .tap
            .as_ref()
            .ok_or_else(|| EngineError::Executor("no verification tap".into()))?;
        self.stream.synchronize().map_err(exec_err)?;
        let logits = self.stream.clone_dtoh(&tap.logits).map_err(exec_err)?;
        let drafts = self.stream.clone_dtoh(&tap.drafts).map_err(exec_err)?;
        Ok((logits[..rows * self.vocab].to_vec(), drafts))
    }

    /// Copies the verification logits and drafts of an eager step into the tap.
    #[cfg(test)]
    fn tap_verification(&mut self, b: usize) -> Result<(), EngineError> {
        let Some(tap) = self.tap.as_mut() else {
            return Ok(());
        };
        let n = (b * self.q * self.vocab).min(tap.logits.len());
        let src = self.text.logits_bf16().slice(0..n);
        let mut dst = tap.logits.slice_mut(0..n);
        self.stream.memcpy_dtod(&src, &mut dst).map_err(exec_err)?;
        let m = (self.max_seqs * self.k).min(tap.drafts.len());
        let src = self.drafts.slice(0..m);
        let mut dst = tap.drafts.slice_mut(0..m);
        self.stream.memcpy_dtod(&src, &mut dst).map_err(exec_err)
    }

    /// Tokens past a sequence's length the step may write KV for.
    pub fn lookahead(&self) -> usize {
        if self.k >= 2 { 2 * self.k } else { self.k }
    }

    /// Whether the sequence already holds a slot.
    pub fn tracks(&self, id: RequestId) -> bool {
        self.slots.slot_of(id).is_some()
    }

    /// Slot of a tracked sequence.
    pub fn slot(&self, id: RequestId) -> Option<usize> {
        self.slots.slot_of(id)
    }

    /// The graph counters.
    pub fn stats(&self) -> GraphStats {
        self.stats
    }

    /// The step's own GEMM engine, for shape recording.
    pub fn gemm_engine(&mut self) -> &mut GemmEngine {
        self.text.gemm_engine()
    }

    /// Gives `id` a slot, uploads its sampling masks, rotary delta and initial `drafts`. Up to
    /// the draft depth may be given; the sequence's first step verifies exactly those drafts, so
    /// an empty list makes it verify its last token alone. Host checks run before any copy, and
    /// a failed adoption leaves no owner and returns the slot unless the error is fatal.
    pub fn adopt(
        &mut self,
        id: RequestId,
        row: &DeviceRow<'_>,
        delta: i64,
        drafts: &[u32],
    ) -> Result<usize, EngineError> {
        let masks = row.masks;
        if masks.prompt_words().len() < self.words || masks.count_words().len() < self.vocab {
            return Err(EngineError::Executor(
                "sampling masks do not match the vocabulary".into(),
            ));
        }
        if drafts.len() > self.k {
            return Err(EngineError::Executor(format!(
                "{} initial drafts for depth {}",
                drafts.len(),
                self.k
            )));
        }
        let upload = |stream: &Arc<CudaStream>,
                      src: &[u32],
                      dst: &mut CudaSlice<u32>,
                      at: usize|
         -> Result<(), EngineError> {
            let mut view = dst.slice_mut(at..at + src.len());
            stream.memcpy_htod(src, &mut view).map_err(exec_err)
        };
        let (stream, words, vocab, k) = (&self.stream, self.words, self.vocab, self.k);
        let (prompt_bits, output_bits, out_counts, draft_buf) = (
            &mut self.prompt_bits,
            &mut self.output_bits,
            &mut self.out_counts,
            &mut self.drafts,
        );
        let slot = self.slots.adopt(id, |slot| {
            upload(
                stream,
                &masks.prompt_words()[..words],
                prompt_bits,
                slot * words,
            )?;
            upload(
                stream,
                &masks.output_words()[..words],
                output_bits,
                slot * words,
            )?;
            upload(
                stream,
                &masks.count_words()[..vocab],
                out_counts,
                slot * vocab,
            )?;
            if !drafts.is_empty() {
                upload(stream, drafts, draft_buf, slot * k)?;
            }
            Ok(())
        });
        let slot = slot?;
        self.counts_want[slot] = drafts.len() as i32;
        self.host_delta[slot] = delta as i32;
        self.tables.invalidate(slot);
        Ok(slot)
    }

    /// Frees the slot of `id`.
    pub fn release(&mut self, id: RequestId) {
        if let Some(slot) = self.slots.release(id) {
            self.tables.invalidate(slot);
        }
    }

    /// The shape of a step over `rows`, whose bucket covers their highest slot.
    fn shape_of(&self, rows: &[ActiveRow<'_>]) -> Result<StepShape, EngineError> {
        step_shape(&self.buckets, rows.iter().map(|r| r.slot)).map_err(|top| {
            EngineError::Executor(format!(
                "{top} rows exceed the largest step bucket of {}",
                self.max_seqs
            ))
        })
    }

    /// Uploads the per-step state of `rows` into the first `bucket` entries of each buffer.
    fn upload_state(&mut self, bucket: usize, rows: &[ActiveRow<'_>]) -> Result<(), EngineError> {
        self.scratch.reset(bucket);
        for row in rows {
            let s = row.slot;
            let r = row.source.row;
            self.scratch.set(
                s,
                r.position as i32,
                (r.blocks.len() * self.block_size) as i32,
                r.token,
                row.source.params,
            );
            self.sync_table(s, &r.blocks)?;
        }
        let copy = |stream: &Arc<CudaStream>, src: &[i32], dst: &mut CudaSlice<i32>| {
            let mut view = dst.slice_mut(0..src.len());
            stream.memcpy_htod(src, &mut view).map_err(exec_err)
        };
        copy(&self.stream, self.scratch.cached(), &mut self.cached)?;
        copy(&self.stream, self.scratch.limit(), &mut self.limit)?;
        copy(&self.stream, &self.host_delta[..bucket], &mut self.delta)?;
        let mut view = self.last.slice_mut(0..bucket);
        self.stream
            .memcpy_htod(self.scratch.last(), &mut view)
            .map_err(exec_err)?;
        let mut view = self.params.slice_mut(0..bucket);
        self.stream
            .memcpy_htod(self.scratch.params(), &mut view)
            .map_err(exec_err)?;
        if self.counts_want[..bucket] != self.counts_dev[..bucket] {
            let mut view = self.draft_counts.slice_mut(0..bucket);
            self.stream
                .memcpy_htod(&self.counts_want[..bucket], &mut view)
                .map_err(exec_err)?;
            let (want, dev) = (&self.counts_want, &mut self.counts_dev);
            dev[..bucket].copy_from_slice(&want[..bucket]);
        }
        Ok(())
    }

    /// Writes the block table of `slot` to the device unless the mirror shows it already holds
    /// `blocks`.
    fn sync_table(&mut self, slot: usize, blocks: &[BlockId]) -> Result<(), EngineError> {
        let at = slot * self.stride;
        let (stream, plan) = (&self.stream, &mut self.plan);
        self.tables.sync(slot, blocks, |table| {
            let mut view = plan.block_tables_mut().slice_mut(at..at + table.len());
            stream.memcpy_htod(table, &mut view).map_err(exec_err)
        })
    }

    /// Enqueues one step for `rows` and its result download without waiting; [`Self::collect`]
    /// returns the tokens.
    pub fn launch(
        &mut self,
        weights: StepWeights<'_>,
        kv: &mut GpuKvPool,
        mtp_kv: Option<&mut GpuKvPool>,
        rows: &[ActiveRow<'_>],
    ) -> Result<(), EngineError> {
        self.timed = None;
        if self.host_probe {
            self.launch_cpu = current_cpu();
        }
        let shape = self.shape_of(rows)?;
        let bucket = shape.bucket;
        self.upload_state(bucket, rows)?;
        #[cfg(test)]
        self.apply_draft_patch()?;
        let split = step_uses_split(
            self.split.as_ref(),
            self.wide_logits,
            self.split_serves(bucket),
            rows.iter().map(|r| &r.source.params),
        );
        let key = replay_key(|k| self.graphs.contains_key(&k), bucket, split);
        if let Some(graph) = key.and_then(|k| self.graphs.get(&k)) {
            let started = self
                .timer
                .as_ref()
                .is_some_and(|[start, _]| start.record(&self.stream).is_ok());
            graph.launch().map_err(exec_err)?;
            self.stats.replays += 1;
            self.stats.split_replays += u64::from(key.is_some_and(|(_, split)| split));
            let ended = started
                && self
                    .timer
                    .as_ref()
                    .is_some_and(|[_, end]| end.record(&self.stream).is_ok());
            self.timed = ended.then_some(bucket);
        } else {
            self.note_fallback(bucket);
            self.body(weights, kv, mtp_kv, bucket, split)?;
        }
        self.stream
            .memcpy_dtoh(&self.result, &mut self.pinned)
            .map_err(exec_err)?;
        self.stats.occupancy.record(&shape);
        if self.active_telemetry {
            scriptor_engine::active_telemetry::record_step(shape.active, shape.bucket);
        }
        self.inflight = Some(rows.iter().map(|r| r.slot).collect());
        Ok(())
    }

    /// Waits for the step enqueued by [`Self::launch`] and returns each row's emitted tokens
    /// in launch order.
    pub fn collect(&mut self) -> Result<Vec<StepRowResult>, EngineError> {
        let slots = self
            .inflight
            .take()
            .ok_or_else(|| EngineError::Executor("no device-resident step is in flight".into()))?;
        if self.host_probe {
            let wall_from = std::time::Instant::now();
            let cpu_from = thread_cpu_ns();
            self.stream.synchronize().map_err(exec_err)?;
            let cpu_ns = thread_cpu_ns().saturating_sub(cpu_from);
            let wall_ns = wall_from.elapsed().as_nanos() as u64;
            self.host_sample = Some(scriptor_engine::l1_counters::StepHostSample {
                sync_wall_ns: wall_ns,
                sync_cpu_ns: cpu_ns,
                cpu: self.launch_cpu,
            });
        } else {
            self.stream.synchronize().map_err(exec_err)?;
        }
        self.note_device_time();
        for &slot in &slots {
            self.counts_want[slot] = self.k as i32;
        }
        let words = self.pinned.as_slice().map_err(exec_err)?;
        let width = result_width(self.q);
        slots
            .iter()
            .map(|&slot| {
                let w = &words[slot * width..(slot + 1) * width];
                let n = (w[0] as usize).min(self.q);
                Ok(StepRowResult {
                    tokens: w[1..1 + n].to_vec(),
                    margins: w[1 + self.q..1 + self.q + n]
                        .iter()
                        .map(|&b| f32::from_bits(b))
                        .collect(),
                })
            })
            .collect()
    }

    /// Turns the per-step host timing probe on or off.
    #[cfg(test)]
    pub fn set_host_probe(&mut self, on: bool) {
        self.host_probe = on;
    }

    /// The host timing of the step last collected; taken once.
    pub fn take_host_sample(&mut self) -> Option<scriptor_engine::l1_counters::StepHostSample> {
        self.host_sample.take()
    }

    /// Adds the device time of the collected graph replay to its bucket's counters when the
    /// replay was timed.
    fn note_device_time(&mut self) {
        let (Some(bucket), Some([start, end])) = (self.timed.take(), self.timer.as_ref()) else {
            return;
        };
        if let Ok(ms) = start.elapsed_ms(end) {
            let nanos = (f64::from(ms) * 1e6).max(0.0) as u64;
            self.stats.occupancy.record_time(bucket, nanos);
        }
    }

    fn note_fallback(&mut self, bucket: usize) {
        self.stats.fallbacks += 1;
        if self.warned.insert(bucket) {
            eprintln!(
                "scriptor: no captured graph for {bucket} rows at draft depth {}; \
                 running the step eagerly",
                self.k
            );
        }
    }

    /// Captures one graph per bucket after an eager warm-up of each.
    pub fn capture(
        &mut self,
        weights: StepWeights<'_>,
        kv: &mut GpuKvPool,
        mut mtp_kv: Option<&mut GpuKvPool>,
    ) -> Result<CaptureReport, EngineError> {
        let started = Instant::now();
        let scope = if self.graphs.is_empty() {
            CaptureScope::Initial
        } else {
            CaptureScope::Recapture
        };
        self.graphs.clear();
        self.stream.synchronize().map_err(exec_err)?;
        let free_before = mem_free(&self.stream);
        let (mut split_seconds, mut split_memory) = (0.0_f64, MemoryReading::NoObservedGrowth);
        for bucket in self.buckets.clone() {
            let mut variants = vec![false];
            if self.split_serves(bucket) && !self.wide_logits {
                variants.push(true);
            }
            for split in variants {
                let variant_started = Instant::now();
                let variant_free = mem_free(&self.stream);
                self.body(weights, kv, mtp_kv.as_deref_mut(), bucket, split)?;
                self.stream.synchronize().map_err(exec_err)?;
                let stream = self.stream.clone();
                let graph = CapturedGraph::capture(&stream, |_| {
                    self.body(weights, kv, mtp_kv.as_deref_mut(), bucket, split)
                        .map_err(|e| CudaError::Graph(e.to_string()))
                })
                .map_err(exec_err)?;
                self.graphs.insert((bucket, split), graph);
                self.stream.synchronize().map_err(exec_err)?;
                if split {
                    split_seconds += variant_started.elapsed().as_secs_f64();
                    split_memory = split_memory.then(MemoryReading::from_samples(
                        variant_free,
                        mem_free(&self.stream),
                    ));
                }
            }
        }
        self.stats.captured = self.graphs.len();
        let memory = MemoryReading::from_samples(free_before, mem_free(&self.stream));
        let split_graphs = self.graphs.keys().filter(|&&(_, split)| split).count();
        let (scratch_rows, scratch_bytes) = match &self.split_scratch {
            Some(scratch) => (scratch.rows(), split_scratch_bytes(scratch.rows())),
            None => (0, 0),
        };
        Ok(CaptureReport {
            graphs: self.graphs.len(),
            seconds: started.elapsed().as_secs_f64(),
            memory,
            scope,
            split_graphs,
            split_seconds,
            split_memory,
            split_scratch_bytes: scratch_bytes,
            split_scratch_rows: scratch_rows,
            split_served_buckets: split_graphs,
            fallback_buckets: self.buckets.len().saturating_sub(split_graphs),
        })
    }

    /// Enqueues the whole step for `b` rows: verification, choice, acceptance, token marking,
    /// the MTP draft chain and the result pack. It uploads, allocates and synchronises nothing.
    fn body(
        &mut self,
        w: StepWeights<'_>,
        kv: &mut GpuKvPool,
        mtp_kv: Option<&mut GpuKvPool>,
        b: usize,
        split: bool,
    ) -> Result<(), EngineError> {
        let (k, q) = (self.k, self.q);
        let wide = self.wide_logits;
        self.plan
            .shape(b, q, b * q, self.max_ctx)
            .map_err(exec_err)?;
        {
            let (tables, rows) = self.plan.tables_and_rows();
            let state = RowState {
                cached: &self.cached,
                limit: &self.limit,
                delta: &self.delta,
                tables,
                stride: self.stride,
                block_size: self.block_size,
            };
            self.step
                .verify_prepare(
                    &self.stream,
                    &state,
                    &self.last,
                    &self.drafts,
                    k,
                    b,
                    rows,
                    &mut self.mark_slots,
                )
                .map_err(exec_err)?;
        }
        self.text
            .forward_resident(
                w.decoder,
                kv,
                &self.plan,
                wide.then_some(&mut self.logits),
                Some(&mut self.hidden),
            )
            .map_err(exec_err)?;
        #[cfg(test)]
        self.tap_verification(b)?;
        if wide {
            self.sampling
                .choose_spec(
                    &self.stream,
                    &mut self.logits,
                    b,
                    k,
                    self.vocab,
                    &self.params,
                    &self.prompt_bits,
                    &self.output_bits,
                    &self.out_counts,
                    self.words,
                    &self.drafts,
                    &mut self.targets,
                    &mut self.margins,
                )
                .map_err(exec_err)?;
        } else {
            let ran_split = match (split, &self.split, self.split_scratch.as_mut()) {
                (true, Some(select), Some(scratch)) => self
                    .sampling
                    .choose_spec_bf16_split(
                        &self.stream,
                        select,
                        scratch,
                        self.text.logits_bf16(),
                        b,
                        k,
                        self.vocab,
                        &self.params,
                        &self.prompt_bits,
                        &self.output_bits,
                        &self.out_counts,
                        self.words,
                        &self.drafts,
                        &mut self.targets,
                        &mut self.margins,
                    )
                    .map_err(exec_err)?,
                _ => false,
            };
            if !ran_split {
                self.sampling
                    .choose_spec_bf16(
                        &self.stream,
                        self.text.logits_bf16(),
                        b,
                        k,
                        self.vocab,
                        &self.params,
                        &self.prompt_bits,
                        &self.output_bits,
                        &self.out_counts,
                        self.words,
                        &self.drafts,
                        &mut self.targets,
                        &mut self.margins,
                    )
                    .map_err(exec_err)?;
            }
        }
        self.accept
            .accept(
                &self.stream,
                &self.drafts,
                &self.targets,
                &self.draft_counts,
                None,
                b,
                k,
                &mut self.emitted,
                &mut self.n_emitted,
            )
            .map_err(exec_err)?;
        self.sampling
            .mark_tokens(
                &self.stream,
                &mut self.output_bits,
                &mut self.out_counts,
                &self.mark_slots,
                &self.emitted,
                b * q,
                self.words,
                self.vocab,
            )
            .map_err(exec_err)?;
        if k > 0 {
            let mtp = w
                .mtp
                .ok_or_else(|| EngineError::Executor("missing MTP weights".into()))?;
            let mtp_kv =
                mtp_kv.ok_or_else(|| EngineError::Executor("missing MTP KV pool".into()))?;
            self.draft_chain(mtp, mtp_kv, b)?;
        }
        self.step
            .pack(
                &self.stream,
                &self.emitted,
                &self.n_emitted,
                &self.margins,
                b,
                q,
                &mut self.result,
            )
            .map_err(exec_err)
    }

    /// Runs MTP pass 0 over the verification hidden states and the chain passes after it,
    /// storing each pass's greedy draft.
    fn draft_chain(
        &mut self,
        mtp: &GpuMtpWeights,
        mtp_kv: &mut GpuKvPool,
        b: usize,
    ) -> Result<(), EngineError> {
        let (k, q) = (self.k, self.q);
        let wide = self.wide_logits;
        self.plan.shape(b, q, b, self.max_ctx).map_err(exec_err)?;
        {
            let (tables, rows) = self.plan.tables_and_rows();
            let state = RowState {
                cached: &self.cached,
                limit: &self.limit,
                delta: &self.delta,
                tables,
                stride: self.stride,
                block_size: self.block_size,
            };
            self.step
                .mtp_prepare(
                    &self.stream,
                    &state,
                    &self.emitted,
                    &self.n_emitted,
                    k,
                    b,
                    rows,
                )
                .map_err(exec_err)?;
        }
        let keep = k > 1;
        let fused = !wide && self.text.fused_draft_head(b);
        {
            let [first, _] = &mut self.carry;
            self.text
                .mtp_forward_resident(
                    mtp,
                    mtp_kv,
                    &self.plan,
                    &self.hidden,
                    wide.then_some(&mut self.logits),
                    fused.then_some(&mut self.draft_tok),
                    keep.then_some(first),
                )
                .map_err(exec_err)?;
        }
        self.pick_draft(0, b, fused)?;
        for pass in 1..k {
            self.plan.shape(b, 1, b, self.max_ctx).map_err(exec_err)?;
            {
                let (tables, rows) = self.plan.tables_and_rows();
                let state = RowState {
                    cached: &self.cached,
                    limit: &self.limit,
                    delta: &self.delta,
                    tables,
                    stride: self.stride,
                    block_size: self.block_size,
                };
                self.step
                    .chain_prepare(
                        &self.stream,
                        &state,
                        &self.n_emitted,
                        &self.drafts,
                        k,
                        pass,
                        b,
                        rows,
                    )
                    .map_err(exec_err)?;
            }
            let keep = pass + 1 < k;
            let [a, c] = &mut self.carry;
            let (src, dst) = if (pass - 1) % 2 == 0 {
                (&*a, c)
            } else {
                (&*c, a)
            };
            self.text
                .mtp_forward_resident(
                    mtp,
                    mtp_kv,
                    &self.plan,
                    src,
                    wide.then_some(&mut self.logits),
                    fused.then_some(&mut self.draft_tok),
                    keep.then_some(dst),
                )
                .map_err(exec_err)?;
            self.pick_draft(pass, b, fused)?;
        }
        Ok(())
    }

    /// Stores the greedy draft of each of `b` rows into draft `index`: taken from the MTP logits,
    /// or, when `fused`, already in `draft_tok` from the head's on-device argmax.
    fn pick_draft(&mut self, index: usize, b: usize, fused: bool) -> Result<(), EngineError> {
        if fused {
            // The head's argmax already wrote the tokens.
        } else if self.wide_logits {
            self.sampling
                .choose(
                    &self.stream,
                    true,
                    &mut self.logits,
                    b,
                    self.vocab,
                    &self.neutral,
                    &self.prompt_bits,
                    &self.output_bits,
                    &self.out_counts,
                    self.words,
                    &mut self.draft_tok,
                    &mut self.draft_margin,
                )
                .map_err(exec_err)?;
        } else {
            self.sampling
                .choose_bf16(
                    &self.stream,
                    true,
                    self.text.logits_bf16(),
                    b,
                    self.vocab,
                    &self.neutral,
                    &self.prompt_bits,
                    &self.output_bits,
                    &self.out_counts,
                    self.words,
                    &mut self.draft_tok,
                    &mut self.draft_margin,
                )
                .map_err(exec_err)?;
        }
        self.step
            .store_drafts(
                &self.stream,
                &self.draft_tok,
                &mut self.drafts,
                self.k,
                index,
                b,
            )
            .map_err(exec_err)
    }
}

/// Free device bytes, or `None` when the driver cannot report them.
fn mem_free(stream: &Arc<CudaStream>) -> Option<usize> {
    stream.context().mem_get_info().ok().map(|(free, _)| free)
}

#[cfg(test)]
mod split_route_tests {
    use super::*;

    fn select() -> SplitSelect {
        SplitSelect {
            block_threads: 128,
            ctas_per_row: 8,
            max_rows: 8,
        }
    }

    fn sampled(slot: u32) -> RowParams {
        let mut p = RowParams::greedy(slot);
        p.temperature = 0.8;
        p
    }

    #[test]
    fn split_is_off_without_a_selector() {
        let rows = [RowParams::greedy(0)];
        assert!(!step_uses_split(None, false, true, &rows));
    }

    #[test]
    fn split_serves_only_all_greedy_bf16_steps_within_its_bound() {
        let s = select();
        let greedy = [RowParams::greedy(0), RowParams::greedy(1)];
        assert!(step_uses_split(Some(&s), false, true, &greedy));
        assert!(step_uses_split(Some(&s), false, true, &[]));
        assert!(!step_uses_split(Some(&s), true, true, &greedy));
        assert!(!step_uses_split(Some(&s), false, false, &greedy));
    }

    #[test]
    fn mixed_or_stochastic_batches_keep_the_single_block_kernel() {
        let s = select();
        let mixed = [RowParams::greedy(0), sampled(1)];
        assert!(!step_uses_split(Some(&s), false, true, &mixed));
        assert!(!step_uses_split(Some(&s), false, true, &[sampled(0)]));
        let mut nan = RowParams::greedy(0);
        nan.temperature = f32::NAN;
        assert!(!step_uses_split(Some(&s), false, true, &[nan]));
    }

    #[test]
    fn replay_picks_the_graph_captured_for_the_route() {
        let both = |k: GraphKey| k.0 == 4;
        assert_eq!(replay_key(both, 4, true), Some((4, true)));
        assert_eq!(replay_key(both, 4, false), Some((4, false)));
        let only_single = |k: GraphKey| k == (4, false);
        assert_eq!(replay_key(only_single, 4, true), Some((4, false)));
        let only_split = |k: GraphKey| k == (4, true);
        assert_eq!(replay_key(only_split, 4, false), None);
        assert_eq!(replay_key(|_| false, 4, true), None);
        assert_eq!(replay_key(both, 2, true), None);
    }
}

#[cfg(test)]
mod capture_memory_tests {
    use super::*;

    const MIB: usize = 1 << 20;

    #[test]
    fn first_capture_reports_the_observed_decrease() {
        let r = MemoryReading::from_samples(Some(10 * MIB), Some(7 * MIB));
        assert_eq!(r, MemoryReading::Observed(3 * MIB as u64));
        assert_eq!(r.bytes(), 3 * MIB as u64);
    }

    #[test]
    fn equal_size_recapture_with_a_post_clear_baseline_still_observes_the_graphs() {
        // The baseline is sampled after the old graphs were destroyed, so the new graphs'
        // allocation shows as a decrease instead of netting against the release.
        let r = MemoryReading::from_samples(Some(10 * MIB), Some(7 * MIB));
        assert_eq!(r, MemoryReading::Observed(3 * MIB as u64));
    }

    #[test]
    fn release_larger_than_allocation_makes_no_residency_claim() {
        let r = MemoryReading::from_samples(Some(7 * MIB), Some(10 * MIB));
        assert_eq!(r, MemoryReading::NoObservedGrowth);
        assert_eq!(r.bytes(), 0);
    }

    #[test]
    fn equal_samples_make_no_residency_claim() {
        let r = MemoryReading::from_samples(Some(5 * MIB), Some(5 * MIB));
        assert_eq!(r, MemoryReading::NoObservedGrowth);
    }

    #[test]
    fn unavailable_samples_are_unavailable_not_zero() {
        for (a, b) in [(None, Some(1)), (Some(1), None), (None, None)] {
            assert_eq!(
                MemoryReading::from_samples(a, b),
                MemoryReading::Unavailable
            );
        }
    }

    #[test]
    fn spans_combine_without_inventing_or_hiding_observations() {
        let o = MemoryReading::Observed(4);
        let n = MemoryReading::NoObservedGrowth;
        let u = MemoryReading::Unavailable;
        assert_eq!(o.then(o), MemoryReading::Observed(8));
        assert_eq!(n.then(o), o);
        assert_eq!(o.then(n), o);
        assert_eq!(n.then(n), n);
        assert_eq!(o.then(u), u);
        assert_eq!(u.then(n), u);
    }

    #[test]
    fn scratch_rows_follow_the_cap_and_the_step_row_count() {
        assert_eq!(split_scratch_rows(8, 16, 1), 8);
        assert_eq!(split_scratch_rows(8, 16, 4), 8);
        assert_eq!(split_scratch_rows(64, 2, 1), 2);
        assert_eq!(split_scratch_rows(64, 2, 4), 8);
        assert_eq!(split_scratch_rows(3, 16, 4), 3);
    }

    #[test]
    fn scratch_bytes_derive_from_row_capacity() {
        let per_row = (1024 * scriptor_kernels::sampling::TOP2_WORDS * 4) as u64;
        assert_eq!(split_scratch_bytes(1), per_row);
        assert_eq!(split_scratch_bytes(8), 8 * per_row);
        assert_eq!(
            split_scratch_bytes(split_scratch_rows(8, 2, 1)),
            2 * per_row
        );
        assert_eq!(
            split_scratch_bytes(split_scratch_rows(8, 16, 4)),
            8 * per_row
        );
        assert_eq!(
            split_scratch_bytes(split_scratch_rows(64, 16, 4)),
            64 * per_row
        );
        assert_eq!(split_scratch_bytes(0), 4);
    }
}
