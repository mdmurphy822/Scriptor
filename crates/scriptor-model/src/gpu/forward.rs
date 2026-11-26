//! One forward over mixed rows: prefill sequences (varlen causal attention) and decode
//! sequences (paged attention over `q_len` rows), sharing every projection GEMM.

use std::sync::{Arc, Mutex};

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{
    CudaContext, CudaEvent, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, PinnedHostSlice, result,
};
use scriptor_cuda::DeviceArena;
use scriptor_kernels::activation::ActivationKernels;
use scriptor_kernels::attention::{
    AttnResidency, DECODE_MAX_ROWS, DecodeCapacity, DecodeGeometry, DecodeRoute, DecodeWorkspace,
    PagedDecodeAttention, PairOrderKernel, PairOrderMode, SeqTables, VarlenAttention,
    VarlenGeometry, decode_route, pair_order_pairs, partial_rows,
};
use scriptor_kernels::cublaslt::{GemmEngine, GemmOperands, GemmOptions, GemmProblem};
use scriptor_kernels::fused::FusedKernels;
use scriptor_kernels::heuristics::{Epilogue, GemmDtype, TuningTable};
use scriptor_kernels::kv::{KvAppendKernels, append_slots};
use scriptor_kernels::norm::{MtpPrologue, NormKernels};
use scriptor_kernels::rope::RopeKernels;
use scriptor_kernels::rope_kv::RopeKvSelect;
use scriptor_kernels::scatter::ScatterKernels;
use scriptor_kernels::vision::VisionKernels;

use super::decoder::{
    GpuDecoderWeights, GpuKvPool, GpuLayerWeights, GpuTextError, LayerCache, TextGeometry,
};
use super::fused_forward::{Fusion, StageProfiler};
use super::logits::{RowOps, gather_rows, head_gemm_rows, lm_head_logits};

/// A sequence prefilled from an empty cache: every prompt row in one forward.
#[derive(Debug, Clone, Copy)]
pub struct PrefillSeq<'a> {
    /// Token ids of the prompt rows.
    pub tokens: &'a [u32],
    /// Rotary `[t, h, w]` position of each row.
    pub positions: &'a [[i32; 3]],
    /// Physical block of each logical block of the sequence.
    pub block_table: &'a [u32],
}

/// A sequence advanced by `tokens.len()` rows past `cached` cached tokens.
#[derive(Debug, Clone, Copy)]
pub struct DecodeSeq<'a> {
    /// Token ids of the new rows (the last accepted token and any drafts).
    pub tokens: &'a [u32],
    /// Rotary `[t, h, w]` position of each new row.
    pub positions: &'a [[i32; 3]],
    /// Physical block of each logical block of the sequence.
    pub block_table: &'a [u32],
    /// Tokens already in the cache before this step.
    pub cached: usize,
}

/// The rows of one forward. Batch rows are numbered prefill sequences first, then decode
/// sequences, each in order.
#[derive(Debug, Clone, Copy)]
pub struct StepInput<'a> {
    pub prefill: &'a [PrefillSeq<'a>],
    pub decode: &'a [DecodeSeq<'a>],
    /// Batch row receiving each row of the image embeddings, in embedding order.
    pub image_rows: &'a [u32],
    /// Batch rows whose logits are returned, in output order.
    pub sample_rows: &'a [u32],
    /// Returns the post-final-norm hidden state of every batch row.
    pub keep_hidden: bool,
}

/// Result of [`TextExecutor::forward`].
pub struct StepOutput {
    /// fp32 logits `[sampled, vocab]`.
    pub logits: CudaSlice<f32>,
    /// Number of logits rows.
    pub sampled: usize,
    /// Vocabulary width of a logits row.
    pub vocab: usize,
    /// bf16 hidden state `[rows, hidden]` of every batch row when requested.
    pub hidden: Option<CudaSlice<u16>>,
}

/// Row bookkeeping of one forward, uploaded to the device.
pub(crate) struct DevicePlan {
    pub rows: usize,
    pub prefill_rows: usize,
    pub prefill_seqs: usize,
    pub max_prefill_len: usize,
    pub decode_rows: usize,
    pub decode_seqs: usize,
    pub decode_q_len: usize,
    pub max_ctx: usize,
    pub max_blocks: usize,
    pub block_size: usize,
    pub image_count: usize,
    pub sample_count: usize,
    pub zero_rows: Vec<usize>,
    pub ids: CudaSlice<i32>,
    pub positions: CudaSlice<i32>,
    pub slots: CudaSlice<i32>,
    pub cu_seqlens: CudaSlice<i32>,
    pub block_tables: CudaSlice<i32>,
    pub seq_lens: CudaSlice<i32>,
    pub image_slots: CudaSlice<i32>,
    pub sample_rows: CudaSlice<i32>,
}

const TABLE_COUNT: usize = 8;

/// One page-locked staging region holding every row table of a pass, with the event that marks
/// when the device copies reading it have completed.
struct StagingSlot {
    host: PinnedHostSlice<i32>,
    done: CudaEvent,
}

/// Persistent page-locked staging for the row tables of one stream; successive passes alternate
/// between two regions so a region is only rewritten after its previous copies have finished.
struct PlanStaging {
    stream: Arc<CudaStream>,
    slots: [Option<StagingSlot>; 2],
    next: usize,
}

static PLAN_STAGING: Mutex<Vec<Arc<Mutex<PlanStaging>>>> = Mutex::new(Vec::new());

fn staging_for(stream: &Arc<CudaStream>) -> Arc<Mutex<PlanStaging>> {
    let mut all = PLAN_STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(found) = all.iter().find(|s| {
        s.lock()
            .map(|g| Arc::ptr_eq(&g.stream, stream))
            .unwrap_or(false)
    }) {
        return found.clone();
    }
    let created = Arc::new(Mutex::new(PlanStaging {
        stream: stream.clone(),
        slots: [None, None],
        next: 0,
    }));
    all.push(created.clone());
    created
}

/// Uploads each table to a fresh device slice through persistent page-locked staging with one
/// asynchronous copy per table; an empty table uploads a single zero.
fn upload_tables(
    stream: &Arc<CudaStream>,
    tables: [&[i32]; TABLE_COUNT],
) -> Result<[CudaSlice<i32>; TABLE_COUNT], GpuTextError> {
    let lens: Vec<usize> = tables.iter().map(|t| t.len().max(1)).collect();
    let total: usize = lens.iter().sum();
    let staging = staging_for(stream);
    let mut guard = staging.lock().unwrap_or_else(|e| e.into_inner());
    let index = guard.next;
    guard.next ^= 1;
    let ctx = stream.context();
    let slot = &mut guard.slots[index];
    match slot {
        Some(s) if s.host.len() >= total => s.done.synchronize()?,
        _ => {
            if let Some(old) = slot.take() {
                old.done.synchronize()?;
            }
            // SAFETY: every element read by a copy is written below first.
            let host = unsafe { ctx.alloc_pinned::<i32>(total.next_power_of_two())? };
            let done = ctx.new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))?;
            *slot = Some(StagingSlot { host, done });
        }
    }
    let slot = slot.as_mut().expect("slot filled above");
    let mut offsets = Vec::with_capacity(TABLE_COUNT);
    {
        let host = slot.host.as_mut_slice()?;
        let mut at = 0;
        for (table, &len) in tables.iter().zip(&lens) {
            host[at..at + len].fill(0);
            host[at..at + table.len()].copy_from_slice(table);
            offsets.push(at);
            at += len;
        }
    }
    let host = slot.host.as_slice()?;
    let mut out: Vec<CudaSlice<i32>> = Vec::with_capacity(TABLE_COUNT);
    for (&at, &len) in offsets.iter().zip(&lens) {
        // SAFETY: the whole slice is overwritten by the copy below.
        let mut dst = unsafe { stream.alloc::<i32>(len)? };
        {
            let (ptr, _guard) = dst.device_ptr_mut(stream);
            // SAFETY: `host[at..at + len]` is page-locked, stays alive and unmodified until
            // `done` completes, and the destination holds `len` elements.
            unsafe { result::memcpy_htod_async(ptr, &host[at..at + len], stream.cu_stream())? };
        }
        out.push(dst);
    }
    slot.done.record(stream)?;
    Ok(out
        .try_into()
        .unwrap_or_else(|_| unreachable!("one slice per table")))
}

fn input_err(msg: impl Into<String>) -> GpuTextError {
    GpuTextError::Input(msg.into())
}

impl DevicePlan {
    /// Validates `input` against the pool geometry and uploads the row tables.
    pub(crate) fn build(
        stream: &Arc<CudaStream>,
        g: &TextGeometry,
        kv: &GpuKvPool,
        input: &StepInput<'_>,
    ) -> Result<Self, GpuTextError> {
        let block_size = kv.block_size();
        let mut ids: Vec<i32> = Vec::new();
        let mut positions: Vec<i32> = Vec::new();
        let mut slots: Vec<i32> = Vec::new();
        let mut zero_rows = Vec::new();
        let mut cu_seqlens = vec![0i32];
        let mut max_prefill_len = 0;

        let mut push_rows = |tokens: &[u32],
                             pos: &[[i32; 3]],
                             table: &[u32],
                             start: usize|
         -> Result<(), GpuTextError> {
            if tokens.is_empty() || tokens.len() != pos.len() {
                return Err(input_err(
                    "each sequence needs one position per token and at least one token",
                ));
            }
            if table.len() * block_size < start + tokens.len() {
                return Err(input_err(format!(
                    "block table of {} blocks cannot hold {} tokens",
                    table.len(),
                    start + tokens.len()
                )));
            }
            if table.iter().any(|&b| b as usize >= kv.num_blocks()) {
                return Err(input_err("block table names a block outside the pool"));
            }
            for (t, p) in tokens.iter().zip(pos) {
                if *t as usize >= g.vocab {
                    return Err(input_err(format!("token id {t} is outside the vocabulary")));
                }
                if p[0] == 0 {
                    zero_rows.push(ids.len());
                }
                ids.push(*t as i32);
                positions.extend_from_slice(p);
            }
            slots.extend(append_slots(table, block_size, start, tokens.len()));
            Ok(())
        };

        for seq in input.prefill {
            push_rows(seq.tokens, seq.positions, seq.block_table, 0)?;
            max_prefill_len = max_prefill_len.max(seq.tokens.len());
            cu_seqlens.push(*cu_seqlens.last().expect("non-empty") + seq.tokens.len() as i32);
        }
        let prefill_rows: usize = input.prefill.iter().map(|s| s.tokens.len()).sum();

        let decode_q_len = input.decode.first().map_or(0, |s| s.tokens.len());
        let mut max_ctx = 0;
        let mut max_blocks = 0;
        for seq in input.decode {
            if seq.tokens.len() != decode_q_len {
                return Err(input_err(
                    "decode sequences must all have the same row count",
                ));
            }
            push_rows(seq.tokens, seq.positions, seq.block_table, seq.cached)?;
            let len = seq.cached + seq.tokens.len();
            max_ctx = max_ctx.max(len);
            max_blocks = max_blocks.max(len.div_ceil(block_size));
        }
        let rows = ids.len();
        if rows == 0 {
            return Err(input_err("a forward needs at least one row"));
        }
        let decode_seqs = input.decode.len();
        if decode_seqs > 0 {
            DecodeGeometry {
                q_heads: g.heads,
                kv_heads: g.kv_heads,
                block_size,
            }
            .validate(decode_q_len)
            .map_err(|e| input_err(format!("{e} (at most {DECODE_MAX_ROWS} rows per KV head)")))?;
        }

        let mut tables = vec![0i32; (decode_seqs * max_blocks).max(1)];
        let mut seq_lens = Vec::with_capacity(decode_seqs);
        for (s, seq) in input.decode.iter().enumerate() {
            let len = seq.cached + seq.tokens.len();
            seq_lens.push(len as i32);
            for (b, slot) in tables[s * max_blocks..(s + 1) * max_blocks]
                .iter_mut()
                .enumerate()
                .take(len.div_ceil(block_size))
            {
                *slot = seq.block_table[b] as i32;
            }
        }

        let to_rows = |name: &str, list: &[u32]| -> Result<Vec<i32>, GpuTextError> {
            list.iter()
                .map(|&r| {
                    if (r as usize) < rows {
                        Ok(r as i32)
                    } else {
                        Err(input_err(format!(
                            "{name} row {r} is outside the {rows} rows"
                        )))
                    }
                })
                .collect()
        };
        let image_slots = to_rows("image", input.image_rows)?;
        let sample_rows = to_rows("sample", input.sample_rows)?;

        let [
            ids,
            positions,
            slots,
            cu_seqlens,
            block_tables,
            seq_lens,
            image_slots_dev,
            sample_rows_dev,
        ] = upload_tables(
            stream,
            [
                &ids,
                &positions,
                &slots,
                &cu_seqlens,
                &tables,
                &seq_lens,
                &image_slots,
                &sample_rows,
            ],
        )?;
        Ok(Self {
            rows,
            prefill_rows,
            prefill_seqs: input.prefill.len(),
            max_prefill_len,
            decode_rows: rows - prefill_rows,
            decode_seqs,
            decode_q_len,
            max_ctx,
            max_blocks,
            block_size,
            image_count: image_slots.len(),
            sample_count: sample_rows.len(),
            zero_rows,
            ids,
            positions,
            slots,
            cu_seqlens,
            block_tables,
            seq_lens,
            image_slots: image_slots_dev,
            sample_rows: sample_rows_dev,
        })
    }
}

/// Activation scratch sized for a row count; grows on demand.
pub(super) struct Work {
    rows: usize,
    samples: usize,
    pub h: CudaSlice<u16>,
    pub hn: CudaSlice<u16>,
    pub h2: CudaSlice<u16>,
    pub hn2: CudaSlice<u16>,
    pub qkv: CudaSlice<u16>,
    pub q: CudaSlice<u16>,
    pub k: CudaSlice<u16>,
    pub v: CudaSlice<u16>,
    pub attn: CudaSlice<u16>,
    pub o: CudaSlice<u16>,
    pub o2: CudaSlice<u16>,
    pub gu: CudaSlice<u16>,
    pub act: CudaSlice<u16>,
    pub t1: CudaSlice<u16>,
    pub t2: CudaSlice<u16>,
    pub cat: CudaSlice<u16>,
    pub dq: CudaSlice<u16>,
    pub dout: CudaSlice<u16>,
    pub sh: CudaSlice<u16>,
    pub lg: CudaSlice<u16>,
}

impl Work {
    fn alloc(
        stream: &Arc<CudaStream>,
        g: &TextGeometry,
        rows: usize,
        samples: usize,
    ) -> Result<Self, GpuTextError> {
        let z = |per_row: usize, n: usize| stream.alloc_zeros::<u16>((n * per_row).max(1));
        Ok(Self {
            rows,
            samples,
            h: z(g.hidden, rows)?,
            hn: z(g.hidden, rows)?,
            h2: z(g.hidden, rows)?,
            hn2: z(g.hidden, rows)?,
            qkv: z(g.q_dim() + 2 * g.kv_dim(), rows)?,
            q: z(g.q_dim(), rows)?,
            k: z(g.kv_dim(), rows)?,
            v: z(g.kv_dim(), rows)?,
            attn: z(g.q_dim(), rows)?,
            o: z(g.hidden, rows)?,
            o2: z(g.hidden, rows)?,
            gu: z(2 * g.intermediate, rows)?,
            act: z(g.intermediate, rows)?,
            t1: z(g.hidden, rows)?,
            t2: z(g.hidden, rows)?,
            cat: z(2 * g.hidden, rows)?,
            dq: z(g.q_dim(), rows)?,
            dout: z(g.q_dim(), rows)?,
            sh: z(g.hidden, head_gemm_rows(samples))?,
            lg: z(g.vocab, head_gemm_rows(samples))?,
        })
    }

    pub(super) fn ensure(
        &mut self,
        stream: &Arc<CudaStream>,
        g: &TextGeometry,
        rows: usize,
        samples: usize,
    ) -> Result<(), GpuTextError> {
        if rows > self.rows || samples > self.samples {
            scriptor_cuda::fault::check_alloc("prefill_work")?;
            *self = Self::alloc(stream, g, rows.max(self.rows), samples.max(self.samples))?;
        }
        Ok(())
    }
}

/// GEMM shape-log phase of a pass over `plan`: `prefill`, `decode` or `mixed` by the rows it
/// carries, prefixed `mtp-` for an MTP-layer pass.
pub(super) fn gemm_phase(plan: &DevicePlan, mtp: bool) -> &'static str {
    match (mtp, plan.prefill_rows > 0, plan.decode_rows > 0) {
        (false, true, false) => "prefill",
        (false, false, _) => "decode",
        (false, true, true) => "mixed",
        (true, true, false) => "mtp-prefill",
        (true, false, _) => "mtp-decode",
        (true, true, true) => "mtp-mixed",
    }
}

/// `out = x * w^T` over `m` rows in bf16 with fp32 accumulation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_bf16(
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    w: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GpuTextError> {
    let problem = GemmProblem {
        m,
        n,
        k,
        dtype: GemmDtype::Bf16,
        epilogue: Epilogue::None,
    };
    let operands = GemmOperands {
        x: x.device_ptr(stream).0,
        w: w.device_ptr(stream).0,
        out: out.device_ptr(stream).0,
        ..Default::default()
    };
    gemm.gemm(&problem, &operands)?;
    Ok(())
}

/// Runs the decoder over mixed prefill and decode rows on one stream.
pub struct TextExecutor {
    pub(super) stream: Arc<CudaStream>,
    pub(super) geometry: TextGeometry,
    sm_count: usize,
    pub(super) norm: NormKernels,
    pub(super) mtp_prologue: MtpPrologue,
    pub(super) fused_mtp_prologue: bool,
    pub(super) act: ActivationKernels,
    pub(super) rope: RopeKernels,
    pub(super) scatter: ScatterKernels,
    pub(super) kv_append: KvAppendKernels,
    pub(super) fused: FusedKernels,
    pub(super) fusion: Fusion,
    pub(super) rope_kv: RopeKvSelect,
    pub(super) gate_up_kernels: VisionKernels,
    pub(super) gate_up_silu: bool,
    pub(super) profiler: Option<StageProfiler>,
    varlen: VarlenAttention,
    decode: PagedDecodeAttention,
    pair_mode: PairOrderMode,
    residency: AttnResidency,
    direct_decode: bool,
    ordered: Option<(PagedDecodeAttention, PairOrderKernel)>,
    pub(super) gemm: GemmEngine,
    pub(super) ops: RowOps,
    pub(super) work: Work,
    decode_ws: Option<DecodeWorkspace>,
}

impl TextExecutor {
    /// Loads the kernels into `ctx`. The cuBLASLt workspace is reserved from `arena`;
    /// `sm_count` sizes the split-K decode plan.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        arena: &mut DeviceArena,
        geometry: TextGeometry,
        sm_count: usize,
        tuning: TuningTable,
        options: GemmOptions,
    ) -> Result<Self, GpuTextError> {
        let residency = AttnResidency::from_env().map_err(GpuTextError::Unsupported)?;
        let mut executor = Self::build(
            ctx, stream, arena, geometry, sm_count, tuning, options, residency,
        )?;
        executor.set_pair_order_mode(PairOrderMode::from_env())?;
        Ok(executor)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        arena: &mut DeviceArena,
        geometry: TextGeometry,
        sm_count: usize,
        tuning: TuningTable,
        options: GemmOptions,
        residency: AttnResidency,
    ) -> Result<Self, GpuTextError> {
        Ok(Self {
            stream: stream.clone(),
            geometry,
            sm_count,
            norm: NormKernels::load(ctx)?,
            mtp_prologue: MtpPrologue::load(ctx)?,
            fused_mtp_prologue: false,
            act: ActivationKernels::load(ctx)?,
            rope: RopeKernels::load(ctx)?,
            scatter: ScatterKernels::load(ctx)?,
            kv_append: KvAppendKernels::load(ctx)?,
            fused: FusedKernels::load(ctx)?,
            fusion: Fusion::default(),
            rope_kv: RopeKvSelect::default(),
            gate_up_kernels: VisionKernels::load(ctx)?,
            gate_up_silu: true,
            profiler: None,
            varlen: VarlenAttention::load(ctx)?,
            decode: PagedDecodeAttention::load_with(ctx, residency, false)?,
            pair_mode: PairOrderMode::Off,
            residency,
            direct_decode: false,
            ordered: None,
            gemm: GemmEngine::new(ctx, stream, arena, tuning, options)?,
            ops: RowOps::load(ctx)?,
            work: Work::alloc(stream, &geometry, 1, 1)?,
            decode_ws: None,
        })
    }

    /// Selects whether decode attention ranks its work pairs once per pass (loading the ordered
    /// kernels on first use) or in every block. The default, from `SCRIPTOR_ATTN_PAIR_ORDER`,
    /// is every block. The ordered kernels use the resident-block count selected by
    /// `SCRIPTOR_ATTN_RESIDENT`.
    pub fn set_pair_order_mode(&mut self, mode: PairOrderMode) -> Result<(), GpuTextError> {
        if mode != PairOrderMode::Off && self.ordered.is_none() {
            let ctx = self.stream.context();
            self.ordered = Some((
                PagedDecodeAttention::load_with(ctx, self.residency, true)?,
                PairOrderKernel::load(ctx)?,
            ));
        }
        self.pair_mode = mode;
        Ok(())
    }

    /// Opts pure-decode attention in to reading and writing the activation buffers in place
    /// instead of staging the decode rows through separate buffers. Off by default; a batch
    /// with prefill rows is always staged.
    pub fn set_direct_decode(&mut self, direct: bool) {
        self.direct_decode = direct;
    }

    /// Whether pure-decode attention uses the in-place buffers.
    pub fn direct_decode(&self) -> bool {
        self.direct_decode
    }

    /// The pair-order mode the split-K workspace is sized for: the selected mode under the
    /// direct route, which is the only route that hoists, otherwise off.
    pub fn workspace_pair_mode(&self) -> PairOrderMode {
        if self.direct_decode {
            self.pair_mode
        } else {
            PairOrderMode::Off
        }
    }

    /// The selected pair-order mode.
    pub fn pair_order_mode(&self) -> PairOrderMode {
        self.pair_mode
    }

    /// Directs every later launch to `stream`. Launch geometry, split plans and algorithm
    /// choices are unchanged.
    pub fn set_stream(&mut self, stream: &Arc<CudaStream>) {
        self.stream = stream.clone();
        self.gemm.set_stream(stream);
    }

    /// Measures, in memory only, the cuBLASLt candidate choices for the decoder's projections at
    /// every row bucket from 256 to the bucket holding `max_rows`; only candidates whose output
    /// is bit-identical to the first candidate's are eligible.
    pub fn autotune(&mut self, max_rows: usize) -> Result<super::TuneReport, GpuTextError> {
        let g = self.geometry;
        let buckets = super::tune_buckets(256, max_rows);
        let qd = g.heads * g.head_dim;
        let kd = g.kv_heads * g.head_dim;
        let shapes = [
            (qd + 2 * kd, g.hidden),
            (g.hidden, qd),
            (2 * g.intermediate, g.hidden),
            (g.hidden, g.intermediate),
        ];
        let problems: Vec<GemmProblem> = shapes
            .iter()
            .flat_map(|&(n, k)| {
                buckets.iter().map(move |&m| GemmProblem {
                    m,
                    n,
                    k,
                    dtype: GemmDtype::Bf16,
                    epilogue: Epilogue::None,
                })
            })
            .collect();
        let gemm = self
            .gemm
            .autotune_in_memory(&problems, super::TUNE_ITERATIONS)?;
        Ok(super::TuneReport { gemm })
    }

    /// The bf16 logits `[rows, vocab]` the last resident pass left in the executor's scratch
    /// when it was given no fp32 destination.
    pub fn logits_bf16(&self) -> &CudaSlice<u16> {
        &self.work.lg
    }

    pub(super) fn sm_count(&self) -> usize {
        self.sm_count
    }

    pub(super) fn decode_ws_capacity(&self) -> DecodeCapacity {
        self.decode_ws
            .as_ref()
            .map_or_else(DecodeCapacity::default, DecodeWorkspace::capacity)
    }

    pub(super) fn set_decode_ws(&mut self, ws: DecodeWorkspace) {
        self.decode_ws = Some(ws);
    }

    /// The executor's GEMM engine, for shape recording (see [`GemmEngine::set_shape_log`]).
    pub fn gemm_engine(&mut self) -> &mut GemmEngine {
        &mut self.gemm
    }

    /// Geometry the executor was built for.
    pub fn geometry(&self) -> &TextGeometry {
        &self.geometry
    }

    /// Embeds the rows (with image embeddings scattered in), runs every layer, applies the
    /// final norm, and computes fp32 logits for the sampled rows only. `image_embeds` is
    /// `[image_rows, hidden]` bf16.
    pub fn forward(
        &mut self,
        weights: &GpuDecoderWeights,
        kv: &mut GpuKvPool,
        input: &StepInput<'_>,
        image_embeds: Option<&CudaSlice<u16>>,
    ) -> Result<StepOutput, GpuTextError> {
        let g = self.geometry;
        if weights.geometry != g {
            return Err(input_err("weights were loaded for a different geometry"));
        }
        if kv.layer_count() != weights.layers.len() {
            return Err(input_err(format!(
                "pool holds {} layers, model has {}",
                kv.layer_count(),
                weights.layers.len()
            )));
        }
        if weights.layers.is_empty() {
            return Err(GpuTextError::Unsupported("a model without layers".into()));
        }
        let plan = DevicePlan::build(&self.stream, &g, kv, input)?;
        self.gemm.set_shape_phase(gemm_phase(&plan, false));
        self.work
            .ensure(&self.stream, &g, plan.rows, plan.sample_count)?;

        self.scatter.embedding_gather(
            &self.stream,
            &weights.embed_tokens,
            &plan.ids,
            &mut self.work.h,
            plan.rows,
            g.hidden,
            g.vocab,
        )?;
        if plan.image_count > 0 {
            let embeds = image_embeds
                .ok_or_else(|| input_err("image rows were given without image embeddings"))?;
            if embeds.len() < plan.image_count * g.hidden {
                return Err(input_err(
                    "image embeddings are shorter than the image rows",
                ));
            }
            self.scatter.image_scatter(
                &self.stream,
                &mut self.work.h,
                embeds,
                &plan.image_slots,
                plan.image_count,
                g.hidden,
            )?;
        }
        self.norm.rms_norm(
            &self.stream,
            &self.work.h,
            &weights.layers[0].input_norm,
            &mut self.work.hn,
            plan.rows,
            g.hidden,
            g.eps,
        )?;
        self.run_stack(&weights.layers, kv.layers_mut(), &weights.final_norm, &plan)?;
        self.finish(
            &plan,
            &weights.lm_head,
            HiddenSource::Normed,
            input.keep_hidden,
        )
    }

    /// Gathers the sampled rows of the normed stack output, computes fp32 logits with `head`,
    /// and optionally copies out every row of `source`.
    pub(super) fn finish(
        &mut self,
        plan: &DevicePlan,
        head: &CudaSlice<u16>,
        source: HiddenSource,
        keep_hidden: bool,
    ) -> Result<StepOutput, GpuTextError> {
        let g = self.geometry;
        let s = plan.sample_count;
        gather_rows(
            &self.scatter,
            &self.stream,
            &self.work.hn,
            &plan.sample_rows,
            &mut self.work.sh,
            s,
            plan.rows,
            g.hidden,
        )?;
        let logits = lm_head_logits(
            &mut self.gemm,
            &self.ops,
            &self.stream,
            &self.work.sh,
            s,
            head,
            g.vocab,
            g.hidden,
            &mut self.work.lg,
        )?;
        let hidden = if keep_hidden {
            let buf = match source {
                HiddenSource::Normed => &self.work.hn,
                HiddenSource::Residual => &self.work.h,
            };
            let mut out = self
                .stream
                .alloc_zeros::<u16>((plan.rows * g.hidden).max(1))?;
            self.ops.copy_rows(
                &self.stream,
                buf,
                0,
                g.hidden,
                &mut out,
                0,
                g.hidden,
                plan.rows,
                g.hidden,
            )?;
            Some(out)
        } else {
            None
        };
        Ok(StepOutput {
            logits,
            sampled: s,
            vocab: g.vocab,
            hidden,
        })
    }

    /// Runs `layers` in order over the rows in `work.hn` / `work.h`; the last layer's fused
    /// residual add normalises with `tail_norm`, leaving the residual stream in `work.h` and its
    /// normalised form in `work.hn`.
    pub(super) fn run_stack(
        &mut self,
        layers: &[GpuLayerWeights],
        caches: &mut [LayerCache],
        tail_norm: &CudaSlice<u16>,
        plan: &DevicePlan,
    ) -> Result<(), GpuTextError> {
        self.prepare_pair_order(plan)?;
        let run = (|| -> Result<(), GpuTextError> {
            for (l, (layer, cache)) in layers.iter().zip(caches.iter_mut()).enumerate() {
                let next = layers.get(l + 1).map_or(tail_norm, |n| &n.input_norm);
                match self.fusion {
                    Fusion::Unfused if self.profiler.is_none() => {
                        self.layer(layer, cache, next, plan)?
                    }
                    fusion => self.layer_staged(fusion, layer, cache, next, plan)?,
                }
            }
            Ok(())
        })();
        if let Some(ws) = self.decode_ws.as_mut() {
            ws.invalidate_pair_order();
        }
        run
    }

    /// Split-K factor and chunk of the decode rows of `plan`, growing the split-K workspace to
    /// cover their partial rows. The pair order is sized in the same allocation, but only for
    /// a launch the workspace has to grow for anyway: a launch whose order the workspace does
    /// not hold runs the unordered kernel and never grows the workspace on its own, so a
    /// workspace reserved before graph capture is never reallocated during it.
    fn decode_split_plan(
        &mut self,
        plan: &DevicePlan,
        geo: &DecodeGeometry,
    ) -> Result<(usize, usize), GpuTextError> {
        let g = self.geometry;
        let (splits, chunk) =
            PagedDecodeAttention::plan_splits(geo, plan.decode_seqs, plan.max_ctx, self.sm_count);
        let need = partial_rows(plan.decode_seqs, splits, plan.decode_q_len, g.heads);
        let held = self.decode_ws_capacity();
        if self.decode_ws.is_none() || held.rows < need {
            let order = pair_order_pairs(plan.decode_seqs, splits, self.workspace_pair_mode());
            let ws = DecodeWorkspace::with_capacity(
                &self.stream,
                DecodeCapacity {
                    rows: need.max(held.rows),
                    order_pairs: order.max(held.order_pairs),
                },
            )?;
            self.decode_ws = Some(ws);
        }
        Ok((splits, chunk))
    }

    fn mixed(&self, plan: &DevicePlan) -> bool {
        plan.prefill_rows > 0
    }

    /// Route of the decode rows of `plan` at `splits`.
    fn route_for(&self, plan: &DevicePlan, splits: usize) -> DecodeRoute {
        decode_route(
            self.mixed(plan),
            self.direct_decode,
            self.pair_mode,
            plan.decode_seqs * splits,
            self.decode_ws_capacity().order_pairs,
        )
    }

    fn decode_geometry(&self, plan: &DevicePlan) -> DecodeGeometry {
        DecodeGeometry {
            q_heads: self.geometry.heads,
            kv_heads: self.geometry.kv_heads,
            block_size: plan.block_size,
        }
    }

    /// Ranks the decode work pairs of `plan` once for every layer of the pass when the pair
    /// order is selected for them. It runs after the pass's sequence lengths are staged and
    /// before the first layer, so each pass, and each replay of a captured pass, orders its own
    /// lengths.
    fn prepare_pair_order(&mut self, plan: &DevicePlan) -> Result<(), GpuTextError> {
        if self.pair_mode == PairOrderMode::Off
            || !self.direct_decode
            || self.mixed(plan)
            || plan.decode_rows == 0
            || self.ordered.is_none()
        {
            return Ok(());
        }
        let geo = self.decode_geometry(plan);
        let (splits, chunk) = self.decode_split_plan(plan, &geo)?;
        if self.route_for(plan, splits) != DecodeRoute::DirectOrdered {
            return Ok(());
        }
        let (_, builder) = self.ordered.as_ref().expect("checked above");
        let ws = self.decode_ws.as_mut().expect("workspace allocated above");
        builder.build(
            &self.stream,
            &plan.seq_lens,
            ws,
            plan.decode_seqs,
            splits,
            chunk,
        )?;
        Ok(())
    }

    /// One sandwich-norm decoder layer:
    /// `h += post_self_attn_norm(attn(input_norm(h)))`, `h += post_mlp_norm(mlp(post_attention_norm(h)))`.
    fn layer(
        &mut self,
        w: &GpuLayerWeights,
        cache: &mut LayerCache,
        next_norm: &CudaSlice<u16>,
        plan: &DevicePlan,
    ) -> Result<(), GpuTextError> {
        let g = self.geometry;
        let rows = plan.rows;
        let s = &self.stream;
        gemm_bf16(
            &mut self.gemm,
            s,
            &self.work.hn,
            &w.qkv,
            &mut self.work.qkv,
            rows,
            g.q_dim() + 2 * g.kv_dim(),
            g.hidden,
        )?;
        self.fused.split_qkv(
            s,
            &self.work.qkv,
            &mut self.work.q,
            &mut self.work.k,
            &mut self.work.v,
            rows,
            g.q_dim(),
            g.kv_dim(),
        )?;
        self.rope.mrope(
            s,
            &mut self.work.q,
            &mut self.work.k,
            &plan.positions,
            rows,
            g.heads,
            g.kv_heads,
            g.head_dim,
            g.rope_theta,
            g.mrope_section,
        )?;
        self.kv_append.append(
            s,
            &mut cache.key,
            &mut cache.value,
            &self.work.k,
            &self.work.v,
            &plan.slots,
            rows,
            g.kv_heads,
            g.head_dim,
        )?;
        self.attention(plan, cache)?;
        let s = &self.stream;
        gemm_bf16(
            &mut self.gemm,
            s,
            &self.work.attn,
            &w.o,
            &mut self.work.o,
            rows,
            g.hidden,
            g.q_dim(),
        )?;
        self.norm.rms_norm(
            s,
            &self.work.o,
            &w.post_self_attn_norm,
            &mut self.work.o2,
            rows,
            g.hidden,
            g.eps,
        )?;
        self.norm.add_rms_norm(
            s,
            &self.work.h,
            &self.work.o2,
            &w.post_attention_norm,
            &mut self.work.h2,
            &mut self.work.hn2,
            rows,
            g.hidden,
            g.eps,
        )?;
        gemm_bf16(
            &mut self.gemm,
            s,
            &self.work.hn2,
            &w.gate_up,
            &mut self.work.gu,
            rows,
            2 * g.intermediate,
            g.hidden,
        )?;
        self.act
            .silu_mul_fused(s, &self.work.gu, &mut self.work.act, rows, g.intermediate)?;
        gemm_bf16(
            &mut self.gemm,
            s,
            &self.work.act,
            &w.down,
            &mut self.work.o,
            rows,
            g.hidden,
            g.intermediate,
        )?;
        self.norm.rms_norm(
            s,
            &self.work.o,
            &w.post_mlp_norm,
            &mut self.work.o2,
            rows,
            g.hidden,
            g.eps,
        )?;
        self.norm.add_rms_norm(
            s,
            &self.work.h2,
            &self.work.o2,
            next_norm,
            &mut self.work.h,
            &mut self.work.hn,
            rows,
            g.hidden,
            g.eps,
        )?;
        Ok(())
    }

    /// Attention over the projected rows: varlen causal attention for the prefill rows and
    /// paged attention against `cache` for the decode rows; results land in `work.attn`.
    pub(super) fn attention(
        &mut self,
        plan: &DevicePlan,
        cache: &LayerCache,
    ) -> Result<(), GpuTextError> {
        let g = self.geometry;
        let scale = (g.head_dim as f32).powf(-0.5);
        let qd = g.q_dim();
        if plan.prefill_rows > 0 {
            self.varlen.run(
                &self.stream,
                &VarlenGeometry::prefill(g.heads, g.kv_heads),
                &self.work.q,
                &self.work.k,
                &self.work.v,
                &mut self.work.attn,
                &plan.cu_seqlens,
                plan.prefill_seqs,
                plan.max_prefill_len,
                scale,
            )?;
        }
        if plan.decode_rows > 0 {
            let geo = self.decode_geometry(plan);
            let (splits, chunk) = self.decode_split_plan(plan, &geo)?;
            let route = self.route_for(plan, splits);
            let ordered = route == DecodeRoute::DirectOrdered
                && self
                    .decode_ws
                    .as_ref()
                    .is_some_and(|ws| ws.pair_order_matches(plan.decode_seqs, splits, chunk));
            let ws = self.decode_ws.as_mut().expect("workspace allocated above");
            let tables = SeqTables {
                block_tables: &plan.block_tables,
                max_blocks: plan.max_blocks,
            };
            let kernel = match (&self.ordered, ordered) {
                (Some((kernel, _)), true) => kernel,
                _ => &self.decode,
            };
            if route == DecodeRoute::Staged {
                self.ops.copy_rows(
                    &self.stream,
                    &self.work.q,
                    plan.prefill_rows * qd,
                    qd,
                    &mut self.work.dq,
                    0,
                    qd,
                    plan.decode_rows,
                    qd,
                )?;
                kernel.run(
                    &self.stream,
                    &geo,
                    &cache.key,
                    &cache.value,
                    &tables,
                    &plan.seq_lens,
                    &self.work.dq,
                    &mut self.work.dout,
                    ws,
                    plan.decode_seqs,
                    plan.decode_q_len,
                    splits,
                    chunk,
                    scale,
                )?;
                self.ops.copy_rows(
                    &self.stream,
                    &self.work.dout,
                    0,
                    qd,
                    &mut self.work.attn,
                    plan.prefill_rows * qd,
                    qd,
                    plan.decode_rows,
                    qd,
                )?;
            } else {
                // Pure decode: the decode rows are rows `0 .. decode_rows` of `q` and `attn`,
                // laid out exactly as `dq` and `dout` would hold them, so the kernel reads and
                // writes the activation buffers directly.
                kernel.run(
                    &self.stream,
                    &geo,
                    &cache.key,
                    &cache.value,
                    &tables,
                    &plan.seq_lens,
                    &self.work.q,
                    &mut self.work.attn,
                    ws,
                    plan.decode_seqs,
                    plan.decode_q_len,
                    splits,
                    chunk,
                    scale,
                )?;
            }
        }
        Ok(())
    }
}

/// Which activation buffer an output's hidden rows are copied from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HiddenSource {
    /// The normalised stack output.
    Normed,
    /// The residual stream before the tail norm.
    Residual,
}

#[cfg(all(test, feature = "cuda"))]
mod staging_tests {
    use super::*;
    use scriptor_cuda::Device;

    const BLOCK: usize = 16;

    fn geometry() -> TextGeometry {
        TextGeometry {
            hidden: 256,
            intermediate: 768,
            layers: 1,
            heads: 2,
            kv_heads: 1,
            head_dim: 128,
            vocab: 512,
            eps: 1e-5,
            rope_theta: 10000.0,
            mrope_section: [8, 12, 12],
        }
    }

    fn positions(n: usize, from: usize) -> Vec<[i32; 3]> {
        (0..n).map(|i| [(from + i + 1) as i32, 1, 2]).collect()
    }

    fn fetch(stream: &Arc<CudaStream>, v: &CudaSlice<i32>) -> Vec<i32> {
        stream.clone_dtoh(v).expect("download")
    }

    /// Builds one plan from `prefill` prompt lengths and `decode` cached lengths and checks every
    /// uploaded table against the tables computed on the host.
    fn check(
        stream: &Arc<CudaStream>,
        g: &TextGeometry,
        kv: &GpuKvPool,
        prefill: &[usize],
        decode: &[usize],
    ) {
        let table: Vec<u32> = (0..kv.num_blocks() as u32).collect();
        let pre_pos: Vec<Vec<[i32; 3]>> = prefill.iter().map(|&n| positions(n, 0)).collect();
        let pre_tok: Vec<Vec<u32>> = prefill
            .iter()
            .map(|&n| (0..n as u32).map(|t| t % 500).collect())
            .collect();
        let dec_pos: Vec<Vec<[i32; 3]>> = decode.iter().map(|&c| positions(2, c)).collect();
        let dec_tok: Vec<Vec<u32>> = decode.iter().map(|&c| vec![(c % 500) as u32, 7]).collect();
        let pre: Vec<PrefillSeq> = prefill
            .iter()
            .enumerate()
            .map(|(i, _)| PrefillSeq {
                tokens: &pre_tok[i],
                positions: &pre_pos[i],
                block_table: &table,
            })
            .collect();
        let dec: Vec<DecodeSeq> = decode
            .iter()
            .enumerate()
            .map(|(i, &c)| DecodeSeq {
                tokens: &dec_tok[i],
                positions: &dec_pos[i],
                block_table: &table,
                cached: c,
            })
            .collect();
        let rows = prefill.iter().sum::<usize>() + 2 * decode.len();
        let sample: Vec<u32> = (0..rows as u32).step_by(2).collect();
        let input = StepInput {
            prefill: &pre,
            decode: &dec,
            image_rows: &[],
            sample_rows: &sample,
            keep_hidden: false,
        };
        let plan = DevicePlan::build(stream, g, kv, &input).expect("plan");

        let mut ids = Vec::new();
        let mut pos = Vec::new();
        let mut slots = Vec::new();
        let mut cu = vec![0i32];
        for (i, s) in pre.iter().enumerate() {
            ids.extend(s.tokens.iter().map(|&t| t as i32));
            pos.extend(s.positions.iter().flatten().copied());
            slots.extend(append_slots(s.block_table, BLOCK, 0, s.tokens.len()));
            cu.push(cu[i] + s.tokens.len() as i32);
        }
        for s in &dec {
            ids.extend(s.tokens.iter().map(|&t| t as i32));
            pos.extend(s.positions.iter().flatten().copied());
            slots.extend(append_slots(s.block_table, BLOCK, s.cached, s.tokens.len()));
        }
        let pad = |v: Vec<i32>| if v.is_empty() { vec![0] } else { v };
        assert_eq!(fetch(stream, &plan.ids), pad(ids));
        assert_eq!(fetch(stream, &plan.positions), pad(pos));
        assert_eq!(fetch(stream, &plan.slots), pad(slots));
        assert_eq!(fetch(stream, &plan.cu_seqlens), cu);
        assert_eq!(
            fetch(stream, &plan.seq_lens),
            pad(dec.iter().map(|s| (s.cached + 2) as i32).collect())
        );
        assert_eq!(
            fetch(stream, &plan.sample_rows),
            pad(sample.iter().map(|&r| r as i32).collect())
        );
        assert_eq!(fetch(stream, &plan.image_slots), vec![0]);
        let tables = fetch(stream, &plan.block_tables);
        for (s, seq) in dec.iter().enumerate() {
            let used = (seq.cached + 2).div_ceil(BLOCK);
            for b in 0..plan.max_blocks {
                let want = if b < used { b as i32 } else { 0 };
                assert_eq!(tables[s * plan.max_blocks + b], want);
            }
        }
    }

    /// Plans built repeatedly over varying row sets hold the exact tables, and repeated passes
    /// leave the device memory footprint unchanged.
    /// Run with `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda plan_staging -- --ignored`.
    #[test]
    #[ignore = "needs a CUDA device; set SCRIPTOR_GPU_TESTS=1"]
    fn plan_staging_matches_tables_and_does_not_grow() {
        if std::env::var_os("SCRIPTOR_GPU_TESTS").is_none() {
            return;
        }
        let dev = Device::open().expect("gpu");
        let stream = dev.new_stream().expect("stream");
        let g = geometry();
        let kv = GpuKvPool::new(&stream, 1, 64, BLOCK, &g).expect("pool");
        let shapes: [(&[usize], &[usize]); 6] = [
            (&[5, 9], &[]),
            (&[], &[3, 40, 17]),
            (&[40], &[100, 2]),
            (&[1], &[1]),
            (&[200, 3, 30], &[]),
            (&[], &[500]),
        ];
        for (pre, dec) in shapes.iter().chain(shapes.iter().rev()) {
            check(&stream, &g, &kv, pre, dec);
        }
        let free = |stream: &Arc<CudaStream>| {
            stream.synchronize().expect("sync");
            let free = cudarc::driver::result::mem_get_info().expect("meminfo").0;
            (free, dev.pool_reserved_bytes().expect("reserved"))
        };
        for (pre, dec) in &shapes {
            check(&stream, &g, &kv, pre, dec);
        }
        let after_one = free(&stream);
        for _ in 0..200 {
            for (pre, dec) in &shapes {
                check(&stream, &g, &kv, pre, dec);
            }
        }
        let after_many = free(&stream);
        println!("free/reserved after one pass {after_one:?}, after many {after_many:?}");
        assert_eq!(
            after_many.1, after_one.1,
            "allocator reservation grew from {} to {}",
            after_one.1, after_many.1
        );
        let slack = 64 << 20;
        assert!(
            after_many.0 + slack >= after_one.0,
            "free memory shrank from {} to {}",
            after_one.0,
            after_many.0
        );
        assert_eq!(
            dev.pool_release_threshold().expect("threshold"),
            dev.info().total_bytes
        );
    }
}
