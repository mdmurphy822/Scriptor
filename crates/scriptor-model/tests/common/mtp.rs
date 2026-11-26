//! Shared rig of the real-checkpoint MTP prologue tests: the decoder and MTP weights, real
//! per-row hidden states, the reference and fused prologue enqueue helpers over preallocated
//! scratch, paired per-route pass state, CUDA-event and wall timing helpers, and the per-binary
//! time budget.
#![cfg(feature = "cuda")]
#![allow(dead_code)]

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaSlice, CudaStream};
use scriptor_cuda::{CapturedGraph, CudaError, Device};
use scriptor_kernels::cublaslt::GemmOptions;
use scriptor_kernels::heuristics::{TuningTable, median};
use scriptor_kernels::norm::{MtpPrologue, NormKernels};
use scriptor_kernels::scatter::ScatterKernels;
use scriptor_model::gpu::{
    DecodeSeq, GpuDecoderWeights, GpuKvPool, GpuMtpWeights, RowOps, StepInput, StepPlan,
    TextExecutor, TextGeometry,
};
use std::sync::Arc;

pub const BATCH: usize = 128;
pub const CONTEXT: usize = 300;
pub const BLOCK: usize = 16;
pub const COUNTS: [usize; 6] = [1, 8, 16, 32, 64, 128];
/// Wall-clock allowance of one test binary, setup included.
pub const BUDGET_S: f64 = 30.0;
/// Consecutive MTP passes of a carried chain.
pub const CHAIN_STEPS: usize = 4;
/// Timed repetitions per measurement (the median is reported).
pub const REPS: usize = 12;

static START: OnceLock<Instant> = OnceLock::new();
static RIG: Mutex<Option<MtpRig>> = Mutex::new(None);

/// Prints the time elapsed since the binary's first rig use and fails when the budget is spent.
pub fn check_budget(label: &str) {
    let elapsed = START.get_or_init(Instant::now).elapsed().as_secs_f64();
    println!("[{label}] elapsed {elapsed:.1} s of the {BUDGET_S:.0} s binary budget");
    assert!(
        elapsed <= BUDGET_S,
        "{label}: {elapsed:.1} s exceeds the {BUDGET_S:.0} s binary budget"
    );
}

/// Runs `f` on the binary's single rig, building it (with the weights and hidden rows) on the
/// first call.
pub fn with_rig<R>(f: impl FnOnce(&mut MtpRig) -> R) -> R {
    START.get_or_init(Instant::now);
    let mut guard = RIG.lock().unwrap_or_else(|e| e.into_inner());
    let rig = guard.get_or_insert_with(MtpRig::new);
    let out = f(rig);
    drop(guard);
    out
}

pub struct MtpRig {
    pub dev: Device,
    pub stream: Arc<CudaStream>,
    pub geometry: TextGeometry,
    pub exec: TextExecutor,
    pub mtp: GpuMtpWeights,
    pub scatter: ScatterKernels,
    pub norm: NormKernels,
    pub ops: RowOps,
    pub fused: MtpPrologue,
    /// Real per-row hidden states of a B=128 decode step, `[BATCH, hidden]`.
    pub hidden: CudaSlice<u16>,
    pub tables: Vec<Vec<u32>>,
    pub blocks_per_seq: usize,
}

impl MtpRig {
    fn new() -> Self {
        assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
        let (spec, files) = super::open_checkpoint();
        let dev = Device::open().expect("gpu");
        // SAFETY: the rig allocates, uploads, runs and reads back on the one stream below, so
        // stream order alone serialises its buffers, as in the executor; without tracking no
        // event recorded before a capture becomes a dependency of the captured work.
        unsafe { dev.context().disable_event_tracking() };
        let stream = dev.new_stream().expect("stream");
        let mut arena = dev.arena(256 << 20, &stream).expect("arena");
        let geometry = TextGeometry::from_spec(&spec.text).expect("geometry");
        let sm_count = dev
            .context()
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .expect("sm count") as usize;
        let mut exec = TextExecutor::new(
            dev.context(),
            &stream,
            &mut arena,
            geometry,
            sm_count,
            TuningTable::default(),
            GemmOptions {
                fp8: false,
                workspace_bytes: 64 << 20,
            },
        )
        .expect("executor");
        let weights = GpuDecoderWeights::load(&stream, &files, &spec.text).expect("weights");
        let mtp = GpuMtpWeights::load_sharing(&stream, &files, &spec.text, &weights).expect("mtp");
        let blocks_per_seq = (CONTEXT + CHAIN_STEPS).div_ceil(BLOCK);
        let mut kv = GpuKvPool::new(
            &stream,
            geometry.layers,
            BATCH * blocks_per_seq,
            BLOCK,
            &geometry,
        )
        .expect("pool");
        let tables: Vec<Vec<u32>> = (0..BATCH)
            .map(|s| {
                (0..blocks_per_seq)
                    .map(|b| (s * blocks_per_seq + b) as u32)
                    .collect()
            })
            .collect();
        let tokens: Vec<[u32; 1]> = (0..BATCH).map(|s| [(s * 7 + 11) as u32 % 4000]).collect();
        let positions = [[CONTEXT as i32; 3]];
        let decode: Vec<DecodeSeq> = (0..BATCH)
            .map(|s| DecodeSeq {
                tokens: &tokens[s],
                positions: &positions,
                block_table: &tables[s],
                cached: CONTEXT,
            })
            .collect();
        let rows: Vec<u32> = (0..BATCH as u32).collect();
        let out = exec
            .forward(
                &weights,
                &mut kv,
                &StepInput {
                    prefill: &[],
                    decode: &decode,
                    image_rows: &[],
                    sample_rows: &rows,
                    keep_hidden: true,
                },
                None,
            )
            .expect("step");
        stream.synchronize().expect("sync");
        let hidden = out.hidden.expect("hidden rows");
        exec.reserve_resident(&[(BATCH, 1)], CONTEXT + CHAIN_STEPS)
            .expect("reserve");
        let ctx = dev.context();
        let (scatter, norm, ops, fused) = (
            ScatterKernels::load(ctx).expect("scatter"),
            NormKernels::load(ctx).expect("norm"),
            RowOps::load(ctx).expect("ops"),
            MtpPrologue::load(ctx).expect("fused"),
        );
        Self {
            dev,
            stream,
            geometry,
            exec,
            mtp,
            scatter,
            norm,
            ops,
            fused,
            hidden,
            tables,
            blocks_per_seq,
        }
    }

    /// Token ids of `count` rows: real ids plus zero, negative and out-of-vocabulary values.
    pub fn ids(&self, count: usize, salt: usize) -> Vec<i32> {
        let vocab = self.geometry.vocab as i32;
        let special = [0, -1, vocab, vocab + 5, i32::MIN, vocab - 1, 1, i32::MAX];
        (0..count)
            .map(|r| match r % 11 {
                1..=8 if r < 16 => special[(r + salt) % special.len()],
                _ => ((r * 131 + salt * 17 + 5) % self.geometry.vocab) as i32,
            })
            .collect()
    }

    /// Token ids of `count` rows that all lie inside the vocabulary.
    pub fn valid_ids(&self, count: usize, salt: usize) -> Vec<i32> {
        (0..count)
            .map(|r| ((r * 131 + salt * 17 + 5) % self.geometry.vocab) as i32)
            .collect()
    }

    /// Hidden rows `[count, hidden]`: the batch rows rotated by `shift` (real rows only).
    pub fn hidden_rows(&self, count: usize, shift: usize) -> CudaSlice<u16> {
        let h = self.geometry.hidden;
        let all = self.stream.clone_dtoh(&self.hidden).expect("dtoh");
        let mut rows = Vec::with_capacity(count * h);
        for r in 0..count {
            let src = (r + shift) % BATCH;
            rows.extend_from_slice(&all[src * h..(src + 1) * h]);
        }
        self.stream.clone_htod(&rows).expect("htod")
    }

    /// Greedy token of each of `rows` logits rows (lowest index on ties).
    pub fn greedy(&self, logits: &[f32], rows: usize) -> Vec<i32> {
        let vocab = self.geometry.vocab;
        (0..rows)
            .map(|r| {
                let row = &logits[r * vocab..(r + 1) * vocab];
                let mut best = 0;
                for (i, v) in row.iter().enumerate() {
                    if *v > row[best] {
                        best = i;
                    }
                }
                best as i32
            })
            .collect()
    }
}

/// Preallocated prologue scratch for up to `rows` rows: the reference intermediates and one
/// `eh_proj` input buffer per route.
pub struct Scratch {
    pub t1: CudaSlice<u16>,
    pub t2: CudaSlice<u16>,
    pub cat_ref: CudaSlice<u16>,
    pub cat_fused: CudaSlice<u16>,
    pub rows: usize,
}

impl Scratch {
    pub fn new(rig: &MtpRig, rows: usize) -> Self {
        let h = rig.geometry.hidden;
        let z = |n: usize| rig.stream.alloc_zeros::<u16>(n).expect("scratch");
        Self {
            t1: z(rows * h),
            t2: z(rows * h),
            cat_ref: z(rows * 2 * h),
            cat_fused: z(rows * 2 * h),
            rows,
        }
    }
}

/// Enqueues exactly the reference prologue (gather, norm, copy, norm, copy) into the scratch;
/// nothing is allocated or cleared.
pub fn enqueue_reference(
    rig: &MtpRig,
    s: &mut Scratch,
    ids: &CudaSlice<i32>,
    hidden: &CudaSlice<u16>,
) {
    let g = rig.geometry;
    let (st, rows) = (&rig.stream, s.rows);
    rig.scatter
        .embedding_gather(
            st,
            &rig.mtp.embed_tokens,
            ids,
            &mut s.t1,
            rows,
            g.hidden,
            g.vocab,
        )
        .expect("gather");
    rig.norm
        .rms_norm(st, &s.t1, &rig.mtp.enorm, &mut s.t2, rows, g.hidden, g.eps)
        .expect("enorm");
    rig.ops
        .copy_rows(
            st,
            &s.t2,
            0,
            g.hidden,
            &mut s.cat_ref,
            0,
            2 * g.hidden,
            rows,
            g.hidden,
        )
        .expect("copy");
    rig.norm
        .rms_norm(st, hidden, &rig.mtp.hnorm, &mut s.t1, rows, g.hidden, g.eps)
        .expect("hnorm");
    rig.ops
        .copy_rows(
            st,
            &s.t1,
            0,
            g.hidden,
            &mut s.cat_ref,
            g.hidden,
            2 * g.hidden,
            rows,
            g.hidden,
        )
        .expect("copy");
}

/// Enqueues the fused prologue kernel into the scratch.
pub fn enqueue_fused(rig: &MtpRig, s: &mut Scratch, ids: &CudaSlice<i32>, hidden: &CudaSlice<u16>) {
    let g = rig.geometry;
    rig.fused
        .launch(
            &rig.stream,
            &rig.mtp.embed_tokens,
            ids,
            hidden,
            &rig.mtp.enorm,
            &rig.mtp.hnorm,
            &mut s.cat_fused,
            s.rows,
            g.hidden,
            g.vocab,
            g.eps,
        )
        .expect("fused");
}

/// Index of the first differing element, if any.
pub fn first_diff<T: PartialEq>(a: &[T], b: &[T]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    (0..a.len()).find(|&i| a[i] != b[i])
}

/// One route's pass state: its own MTP key/value pool, plan, logits and the two carry buffers a
/// chain alternates between.
pub struct Route {
    pub fused: bool,
    pub rows: usize,
    pub plan: StepPlan,
    pub kv: GpuKvPool,
    pub logits: CudaSlice<f32>,
    pub carry: [CudaSlice<u16>; 2],
}

impl Route {
    pub fn new(rig: &MtpRig, rows: usize, fused: bool) -> Self {
        let g = rig.geometry;
        let s = &rig.stream;
        let kv = GpuKvPool::new(s, 1, rows * rig.blocks_per_seq, BLOCK, &g).expect("mtp pool");
        let mut plan = StepPlan::new(s, &kv, rows, rows, rig.blocks_per_seq).expect("plan");
        let mut tables = Vec::new();
        for t in rig.tables.iter().take(rows) {
            tables.extend(t.iter().map(|&b| b as i32));
        }
        s.memcpy_htod(&tables, plan.block_tables_mut())
            .expect("tables");
        let sample: Vec<i32> = (0..rows as i32).collect();
        s.memcpy_htod(&sample, plan.rows_mut().sample_rows)
            .expect("sample");
        let z = || s.alloc_zeros::<u16>(rows * g.hidden).expect("carry");
        Self {
            fused,
            rows,
            plan,
            kv,
            logits: s.alloc_zeros::<f32>(rows * g.vocab).expect("logits"),
            carry: [z(), z()],
        }
    }

    /// Writes the row tables of chain step `step`: every sequence's new token sits at position
    /// `CONTEXT + step` with `CONTEXT + step + 1` tokens of context, and `ids` are the inputs.
    pub fn prepare(&mut self, rig: &MtpRig, step: usize, ids: &[i32]) {
        let s = &rig.stream;
        let pos = CONTEXT + step;
        self.plan
            .shape(self.rows, 1, self.rows, pos + 1)
            .expect("shape");
        let slots: Vec<i32> = rig
            .tables
            .iter()
            .take(self.rows)
            .map(|t| (t[pos / BLOCK] as usize * BLOCK + pos % BLOCK) as i32)
            .collect();
        let rows = self.plan.rows_mut();
        s.memcpy_htod(&slots, rows.slots).expect("slots");
        s.memcpy_htod(&vec![(pos + 1) as i32; self.rows], rows.seq_lens)
            .expect("lens");
        s.memcpy_htod(&vec![pos as i32; self.rows * 3], rows.positions)
            .expect("positions");
        s.memcpy_htod(ids, rows.ids).expect("ids");
    }

    /// Key and value bits of the route's MTP layer cache.
    pub fn kv_bits(&self, rig: &MtpRig) -> (Vec<u16>, Vec<u16>) {
        let layer = self.kv.layer(0).expect("layer");
        (
            rig.stream.clone_dtoh(&layer.key).expect("key"),
            rig.stream.clone_dtoh(&layer.value).expect("value"),
        )
    }
}

/// One resident MTP pass over `plan`.
pub fn run_mtp(
    exec: &mut TextExecutor,
    mtp: &GpuMtpWeights,
    kv: &mut GpuKvPool,
    plan: &StepPlan,
    hidden: &CudaSlice<u16>,
    logits: &mut CudaSlice<f32>,
    carry: &mut CudaSlice<u16>,
) -> Result<(), scriptor_model::gpu::GpuTextError> {
    exec.mtp_forward_resident(mtp, kv, plan, hidden, Some(logits), None, Some(carry))
}

/// One chain step on `route`: selects its prologue, writes the tables for `step` with `ids`,
/// and runs the pass from `h0` (step 0) or from the previous step's carry.
pub fn chain_step(
    rig: &mut MtpRig,
    route: &mut Route,
    step: usize,
    ids: &[i32],
    h0: &CudaSlice<u16>,
) {
    rig.exec.set_fused_mtp_prologue(route.fused);
    route.prepare(rig, step, ids);
    let [c0, c1] = &mut route.carry;
    let (src, dst): (&CudaSlice<u16>, &mut CudaSlice<u16>) = if step == 0 {
        (h0, c0)
    } else if step % 2 == 1 {
        (&*c0, c1)
    } else {
        (&*c1, c0)
    };
    run_mtp(
        &mut rig.exec,
        &rig.mtp,
        &mut route.kv,
        &route.plan,
        src,
        &mut route.logits,
        dst,
    )
    .expect("mtp pass");
}

/// Captures `body` into an instantiated graph on the rig's stream.
pub fn capture(
    stream: &Arc<CudaStream>,
    body: impl FnOnce() -> Result<(), String>,
) -> CapturedGraph {
    CapturedGraph::capture(stream, |_| body().map_err(CudaError::Graph)).expect("capture")
}

/// GPU time in microseconds between two events around `f`'s enqueued work.
pub fn gpu_us(stream: &Arc<CudaStream>, f: &mut dyn FnMut()) -> f64 {
    stream.synchronize().expect("sync");
    let a = stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .expect("event");
    f();
    let b = stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .expect("event");
    stream.synchronize().expect("sync");
    f64::from(a.elapsed_ms(&b).expect("elapsed")) * 1e3
}

/// Synchronised wall latency in microseconds of `f` (host enqueue through completion).
pub fn wall_us(stream: &Arc<CudaStream>, f: &mut dyn FnMut()) -> f64 {
    stream.synchronize().expect("sync");
    let start = Instant::now();
    f();
    stream.synchronize().expect("sync");
    start.elapsed().as_secs_f64() * 1e6
}

/// Median of `REPS` samples of `measure` after three discarded warm-up runs.
pub fn median_of(measure: &mut dyn FnMut() -> f64) -> f64 {
    for _ in 0..3 {
        measure();
    }
    let samples: Vec<f64> = (0..REPS).map(|_| measure()).collect();
    median(&samples).expect("samples")
}
