//! GPU gate for the vision/decode SM split: decode graphs captured and replayed from a decode
//! green-context partition, with vision-shaped GEMM work running on the vision partition, must
//! produce the same sampled tokens, margins, logits and hidden states bit for bit as the
//! full-device run. Also prints serial, overlapped and partitioned timings.
//!
//! Run with
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-cli --release sm_split_gate -- --ignored --nocapture --test-threads=1`.

use std::time::{Duration, Instant};

use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use half::bf16;
use scriptor_cuda::{DeviceArena, SmSplit};
use scriptor_kernels::cublaslt::{GemmEngine, GemmOperands, GemmOptions, GemmProblem};
use scriptor_kernels::heuristics::{Epilogue, GemmDtype, TuningTable};
use std::sync::Arc;

use super::{
    Rng, TOKENS, device_step, executor, fixture, required_split_from_env, sequences, settle_grant,
    start,
};
use crate::device_step::StateBits;
use crate::gpu_executor::GpuExecutor;

/// The splits the gate always covers, as `(vision, decode)` SMs.
const SPLITS: [(u32, u32); 3] = [(32, 16), (40, 8), (24, 24)];

/// The splits a run covers: [`SPLITS`] followed by the required split when it is not among
/// them.
pub(super) fn splits(required: Option<(u32, u32)>) -> Vec<(u32, u32)> {
    let mut all = SPLITS.to_vec();
    if let Some(pair) = required
        && !all.contains(&pair)
    {
        all.push(pair);
    }
    all
}
/// Activation rows of one synthetic vision batch.
const VISION_ROWS: usize = 4096;
/// `(n, k)` of the GEMMs of one synthetic vision layer: qkv, output projection, gate/up, down.
const VISION_LAYER: [(usize, usize); 4] = [(3072, 1024), (1024, 1024), (8192, 1024), (1024, 4096)];
/// Layers per synthetic vision batch.
const VISION_LAYERS: usize = 24;
/// Synthetic vision batches in flight during the correctness pass.
const CHECK_BATCHES: usize = 4;
/// Synthetic vision batches per timing measurement.
const TIMED_BATCHES: usize = 8;
/// Generation rounds per timing measurement.
const TIMED_ROUNDS: usize = 40;

/// Vision-shaped cuBLASLt work on the executor's vision stream.
pub(super) struct VisionLoad {
    stream: Arc<CudaStream>,
    engine: GemmEngine,
    x: CudaSlice<u16>,
    weights: Vec<CudaSlice<u16>>,
    out: CudaSlice<u16>,
    _arena: DeviceArena,
}

fn random_bf16(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| bf16::from_f32(rng.next() * 0.1).to_bits())
        .collect()
}

impl VisionLoad {
    pub(super) fn new(exec: &GpuExecutor) -> Self {
        let stream = exec.vision_stream().clone();
        let options = GemmOptions::default();
        let mut arena =
            DeviceArena::new(&stream, options.workspace_bytes + (4 << 20)).expect("arena");
        let engine = GemmEngine::new(
            exec.context(),
            &stream,
            &mut arena,
            TuningTable::default(),
            options,
        )
        .expect("vision engine");
        let mut rng = Rng(0x0dd5_eed5_1234_9876);
        let max_k = VISION_LAYER.iter().map(|&(_, k)| k).max().unwrap_or(0);
        let max_n = VISION_LAYER.iter().map(|&(n, _)| n).max().unwrap_or(0);
        let x = stream
            .clone_htod(&random_bf16(&mut rng, VISION_ROWS * max_k))
            .expect("x");
        let weights = VISION_LAYER
            .iter()
            .map(|&(n, k)| {
                stream
                    .clone_htod(&random_bf16(&mut rng, n * k))
                    .expect("weights")
            })
            .collect();
        let out = stream.alloc_zeros::<u16>(VISION_ROWS * max_n).expect("out");
        stream.synchronize().expect("sync");
        Self {
            stream,
            engine,
            x,
            weights,
            out,
            _arena: arena,
        }
    }

    fn problem(&self, index: usize, rows: usize) -> (GemmProblem, GemmOperands) {
        let (n, k) = VISION_LAYER[index];
        let problem = GemmProblem {
            m: rows,
            n,
            k,
            dtype: GemmDtype::Bf16,
            epilogue: Epilogue::None,
        };
        let operands = GemmOperands {
            x: self.x.device_ptr(&self.stream).0,
            w: self.weights[index].device_ptr(&self.stream).0,
            out: self.out.device_ptr(&self.stream).0,
            ..Default::default()
        };
        (problem, operands)
    }

    /// Enqueues `batches` synthetic vision batches without waiting.
    pub(super) fn enqueue(&mut self, batches: usize) {
        for _ in 0..batches {
            for _ in 0..VISION_LAYERS {
                for index in 0..VISION_LAYER.len() {
                    let (problem, operands) = self.problem(index, VISION_ROWS);
                    self.engine.gemm(&problem, &operands).expect("vision gemm");
                }
            }
        }
    }

    /// The bits of one fixed GEMM's output: cuBLASLt on this stream against its algorithm
    /// choice must not depend on the stream's SM partition.
    pub(super) fn probe(&mut self) -> Vec<u16> {
        let (problem, operands) = self.problem(0, 1024);
        self.engine.gemm(&problem, &operands).expect("probe gemm");
        let host = self.stream.clone_dtoh(&self.out).expect("probe download");
        host[..1024 * VISION_LAYER[0].0].to_vec()
    }
}

/// Runs `rounds` generations of three sequences on the device step; with `record`, keeps the
/// raw step buffers after every step.
fn drive(exec: &mut GpuExecutor, rounds: usize, record: bool) -> (Vec<Vec<u32>>, Vec<StateBits>) {
    let mut tokens = Vec::new();
    let mut bits = Vec::new();
    for _ in 0..rounds {
        let mut seqs = sequences(3);
        start(exec, &mut seqs);
        while seqs.iter().any(|s| s.out.len() < TOKENS) {
            device_step(exec, &mut seqs);
            if record {
                bits.push(exec.step_bits().expect("step bits"));
            }
        }
        tokens.extend(seqs.into_iter().map(|s| s.out));
    }
    (tokens, bits)
}

struct Timing {
    vision_alone: Duration,
    decode_alone: Duration,
    decode_with_vision: Duration,
    total_concurrent: Duration,
}

struct Outcome {
    tokens: Vec<Vec<u32>>,
    bits: Vec<StateBits>,
    probe: Vec<u16>,
    timing: Option<Timing>,
}

fn sm_count(exec: &GpuExecutor) -> i32 {
    exec.context()
        .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .expect("sm count")
}

/// Builds an executor (optionally partitioned), captures its graphs, and runs the fixed decode
/// workload, with `load_batches` synthetic vision batches in flight when non-zero.
fn run_case(
    tag: &str,
    depth: usize,
    wide: bool,
    split: Option<SmSplit>,
    load_batches: usize,
    timed: bool,
) -> Result<Outcome, String> {
    let fx = fixture(tag);
    // SAFETY: the gate runs single-threaded (`--test-threads=1`); the variable is read once
    // while the executor is built and removed straight after.
    unsafe {
        if wide {
            std::env::set_var("SCRIPTOR_SAMPLING_LOGITS", "f32");
        } else {
            std::env::remove_var("SCRIPTOR_SAMPLING_LOGITS");
        }
    }
    let mut exec = executor(&fx, depth);
    // SAFETY: as above.
    unsafe { std::env::remove_var("SCRIPTOR_SAMPLING_LOGITS") };

    let before = sm_count(&exec);
    if let Some(split) = split
        && let Err(e) = exec.partition(split)
    {
        return Err(format!("{e:?}"));
    }
    assert_eq!(
        sm_count(&exec),
        before,
        "the device SM count that sizes launch geometry changed after partitioning"
    );
    let report = exec.capture_graphs().expect("capture");
    assert!(report.graphs >= 4, "buckets 1, 2, 4 and 8 are captured");
    let nodes = exec.graph_node_contexts().expect("node contexts");
    assert!(!nodes.is_empty(), "no kernel nodes were captured");
    match exec.partition_contexts() {
        Some((vision, decode)) => {
            assert_ne!(vision, decode, "the partitions share a context handle");
            assert!(
                nodes.iter().all(|&c| c == decode),
                "a decode graph kernel node was captured outside the decode partition"
            );
        }
        None => assert!(
            nodes.windows(2).all(|w| w[0] == w[1]),
            "unpartitioned graph nodes disagree on their context"
        ),
    }

    let mut load = VisionLoad::new(&exec);
    let probe = load.probe();
    load.enqueue(load_batches);
    let (tokens, bits) = drive(&mut exec, 1, true);
    exec.vision_stream().synchronize().expect("vision sync");

    let timing = timed.then(|| {
        let started = Instant::now();
        load.enqueue(TIMED_BATCHES);
        exec.vision_stream().synchronize().expect("vision sync");
        let vision_alone = started.elapsed();

        let started = Instant::now();
        drive(&mut exec, TIMED_ROUNDS, false);
        let decode_alone = started.elapsed();

        let started = Instant::now();
        load.enqueue(TIMED_BATCHES);
        drive(&mut exec, TIMED_ROUNDS, false);
        let decode_with_vision = started.elapsed();
        exec.vision_stream().synchronize().expect("vision sync");
        let total_concurrent = started.elapsed();
        Timing {
            vision_alone,
            decode_alone,
            decode_with_vision,
            total_concurrent,
        }
    });
    Ok(Outcome {
        tokens,
        bits,
        probe,
        timing,
    })
}

/// Splits the device cannot form are skipped, except the one the engine must support and the
/// one `SCRIPTOR_REQUIRE_SPLIT` names.
const REQUIRED_SPLIT: (u32, u32) = (32, 16);

/// The outcome of a partitioned case when its split was granted; a refused split is skipped
/// or, when required, fails the test.
fn granted(
    split: (u32, u32),
    required: Option<(u32, u32)>,
    got: Result<Outcome, String>,
) -> Option<Outcome> {
    let formed = got.as_ref().map(|_| ()).map_err(Clone::clone);
    settle_grant(split, REQUIRED_SPLIT, required, formed).then(|| got.expect("granted"))
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn report(label: &str, t: &Timing) {
    let serial = t.vision_alone + t.decode_alone;
    println!(
        "{label:<22} vision alone {:8.1} ms | decode alone {:8.1} ms | serial {:8.1} ms | \
         decode while vision {:8.1} ms | concurrent total {:8.1} ms | vs serial {:+6.1}%",
        ms(t.vision_alone),
        ms(t.decode_alone),
        ms(serial),
        ms(t.decode_with_vision),
        ms(t.total_concurrent),
        (ms(t.total_concurrent) / ms(serial) - 1.0) * 100.0,
    );
}

fn check_equal(label: &str, want: &Outcome, got: &Outcome) {
    assert_eq!(got.tokens, want.tokens, "{label}: sampled tokens diverged");
    assert_eq!(
        got.bits.len(),
        want.bits.len(),
        "{label}: step counts diverged"
    );
    for (i, (g, w)) in got.bits.iter().zip(&want.bits).enumerate() {
        assert_eq!(g.targets, w.targets, "{label}: step {i} targets diverged");
        assert_eq!(g.margins, w.margins, "{label}: step {i} margins diverged");
        assert_eq!(g.hidden, w.hidden, "{label}: step {i} hidden diverged");
        assert_eq!(g.logits, w.logits, "{label}: step {i} logits diverged");
    }
    assert_eq!(
        got.probe, want.probe,
        "{label}: a cuBLASLt GEMM on the vision stream diverged"
    );
}

#[test]
#[ignore = "needs GPU"]
fn sm_split_gate_is_bitwise_identical_to_the_full_device() {
    let required = required_split_from_env();
    for depth in [0usize, 3] {
        for wide in [false, true] {
            let tag = format!("gate-k{depth}-w{}", wide as u8);
            let reference = run_case(&format!("{tag}-ref"), depth, wide, None, 0, false)
                .expect("unpartitioned run");
            assert!(!reference.bits.is_empty());
            if wide {
                assert!(
                    reference.bits.iter().any(|b| !b.logits.is_empty()),
                    "wide logits were not recorded"
                );
            }
            let overlapped = run_case(
                &format!("{tag}-overlap"),
                depth,
                wide,
                None,
                CHECK_BATCHES,
                false,
            )
            .expect("unpartitioned run");
            check_equal(
                &format!("k={depth} wide={wide} full device with concurrent vision"),
                &reference,
                &overlapped,
            );
            for (vision, decode) in splits(required) {
                let split = SmSplit { vision, decode };
                let got = run_case(
                    &format!("{tag}-{vision}-{decode}"),
                    depth,
                    wide,
                    Some(split),
                    CHECK_BATCHES,
                    false,
                );
                let Some(got) = granted((vision, decode), required, got) else {
                    continue;
                };
                check_equal(
                    &format!("k={depth} wide={wide} split {split}"),
                    &reference,
                    &got,
                );
                println!("k={depth} wide={wide} split {split}: bitwise identical");
                if Some((vision, decode)) == required {
                    println!("REQUIRED split {vision}/{decode} granted and bitwise identical");
                }
            }
        }
    }
}

#[test]
#[ignore = "needs GPU"]
fn sm_split_gate_timing() {
    let required = required_split_from_env();
    let depth = 3;
    let none = run_case("gate-time-none", depth, false, None, 0, true).expect("unpartitioned run");
    report(
        "full device",
        none.timing.as_ref().expect("timing measured"),
    );
    for (vision, decode) in splits(required) {
        let split = SmSplit { vision, decode };
        let got = run_case(
            &format!("gate-time-{vision}-{decode}"),
            depth,
            false,
            Some(split),
            0,
            true,
        );
        let Some(got) = granted((vision, decode), required, got) else {
            continue;
        };
        report(
            &format!("split {vision}/{decode}"),
            got.timing.as_ref().expect("timing measured"),
        );
    }
}

#[test]
fn gate_splits_append_the_required_pair_once() {
    assert_eq!(splits(None), SPLITS.to_vec());
    assert_eq!(splits(Some((36, 12))).len(), 4);
    assert_eq!(splits(Some((36, 12))).last(), Some(&(36, 12)));
    assert_eq!(splits(Some((32, 16))), SPLITS.to_vec());
    assert_eq!(&splits(Some((36, 12)))[..3], &SPLITS[..]);
}
