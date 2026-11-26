//! Paired GPU timing of cuBLASLt candidates on captured GLM-OCR vision operands.
//!
//! Each test covers one vision GEMM shape. The fixture images run once through the real
//! vision tower on a baseline engine that copies the inputs of that shape's dispatches (the
//! first layer's, and a later layer's as a hold-out where the shape repeats per layer). The
//! same engine then compares candidate 0 against every other heuristic candidate of the shape:
//!
//! - bit equality of the full output against candidate 0 on every captured operand set and on
//!   synthetic operands of the same shape;
//! - paired timing on the captured operands and on the synthetic operands, each with the
//!   operands L2-resident (warm) and with L2 flushed before every sample (flushed). Each pair
//!   holds one sample of each candidate in a seeded random order; every sample is one
//!   dispatch between two device events, with the stream kept busy up to the start event.
//!   Both candidates run on the same engine, workspace and library.
//!
//! Per candidate, operand set and cache state the test prints the medians, the median paired
//! difference (candidate minus candidate 0) with its p10/p90 and a 95% bootstrap interval, and
//! the verdict: `faster`/`slower` when the interval excludes 0, `no difference` otherwise, and
//! whether the admission rule ([`PairedTiming::clear_gain`] at [`ADMIT_MARGIN`]) holds. A fresh
//! engine then runs the current admission on the shape and its choice must be bit-identical to
//! candidate 0 on every captured operand set. Run one shape with
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda --profile gpu-test --test gpu_admission_paired_real -- --ignored --nocapture --test-threads 1 <test name>`.
#![cfg(feature = "cuda")]

mod common;

use scriptor_kernels::cublaslt::{
    ADMIT_MARGIN, CapturedOperands, GemmEngine, GemmOperands, GemmProblem, L2Flush, PairOrder,
    PairedTiming, SyntheticOperands, XorShift, bootstrap_median_interval,
};
use scriptor_kernels::heuristics::{Epilogue, GemmDtype, ShapeKey, TuningTable};

use common::fixtures::Rig;

/// Pairs per candidate, operand set and cache state.
const PAIRS: usize = 100;
/// Untimed launches of both candidates before each paired run.
const WARMUP: usize = 3;
/// Bootstrap resamples of the median paired difference.
const RESAMPLES: usize = 2000;
/// Each tail of the bootstrap interval.
const TAIL: f64 = 0.025;
/// Occurrences of a per-layer shape captured: the first layer's dispatch and a later layer's.
const LAYER_OCCURRENCES: &[usize] = &[0, 12];
/// Occurrence of a shape dispatched once per batch.
const SINGLE_OCCURRENCE: &[usize] = &[0];
/// Paired samples per candidate requested of the admission run.
const ADMIT_ITERATIONS: usize = 7;

/// Device buffers behind one operand set.
enum Source<'a> {
    Real(&'a CapturedOperands),
    Synthetic(&'a SyntheticOperands, usize, usize),
}

/// One measured operand set.
struct OperandSet<'a> {
    label: String,
    source: Source<'a>,
    ops: GemmOperands,
    /// Full output of candidate 0 on this set.
    reference: Vec<u8>,
}

impl OperandSet<'_> {
    fn is_real(&self) -> bool {
        matches!(self.source, Source::Real(_))
    }

    /// Host copy of the set's full output buffer.
    fn read(&self) -> Vec<u8> {
        match self.source {
            Source::Real(c) => c.host_out().expect("download"),
            Source::Synthetic(s, m, n) => {
                let rows: Vec<usize> = (0..m).collect();
                s.host_out_rows(&rows, n).expect("download")
            }
        }
    }
}

fn problem(m: usize, n: usize, k: usize, epilogue: Epilogue) -> GemmProblem {
    GemmProblem {
        m,
        n,
        k,
        dtype: GemmDtype::Bf16,
        epilogue,
    }
}

/// Number of bf16 values that differ between two outputs.
fn differing(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(2)
        .zip(b.chunks_exact(2))
        .filter(|(x, y)| x != y)
        .count()
}

fn run_out(engine: &mut GemmEngine, p: &GemmProblem, set: &OperandSet, idx: usize) -> Vec<u8> {
    engine.gemm_with(p, &set.ops, idx).expect("gemm");
    set.read()
}

/// Paired timing of candidate `idx` against candidate 0 on `set`, printed as one table row.
fn measure(
    engine: &mut GemmEngine,
    p: &GemmProblem,
    set: &OperandSet,
    idx: usize,
    flush: Option<&mut L2Flush>,
    seed: u64,
) {
    for _ in 0..WARMUP {
        engine.gemm_with(p, &set.ops, 0).expect("warmup");
        engine.gemm_with(p, &set.ops, idx).expect("warmup");
    }
    let cache = if flush.is_some() { "flushed" } else { "warm" };
    let mut order = PairOrder::Random(XorShift(seed));
    let (first, cand) = engine
        .time_paired(p, &set.ops, [0, idx], PAIRS, flush, &mut order)
        .expect("paired timing");
    let t = PairedTiming::from_pairs(&first, &cand).expect("pairs");
    let diffs: Vec<f64> = cand.iter().zip(&first).map(|(c, f)| c - f).collect();
    let (lo, hi) = bootstrap_median_interval(&diffs, RESAMPLES, TAIL, seed).expect("interval");
    let verdict = if hi < 0.0 {
        "faster"
    } else if lo > 0.0 {
        "slower"
    } else {
        "no difference"
    };
    println!(
        "| {} | {cache} | {idx} | {:.1} | {:.1} | {:+.2} ({:+.2}%) | {:+.2} / {:+.2} | [{:+.2}, {:+.2}] | {verdict} | {} |",
        set.label,
        t.first_median,
        t.candidate_median,
        t.diff_median,
        100.0 * t.relative(t.diff_median),
        t.diff_p10,
        t.diff_p90,
        lo,
        hi,
        if t.clear_gain(ADMIT_MARGIN) {
            "yes"
        } else {
            "no"
        },
    );
}

/// Captures the shape's operands from the real tower and runs the comparison described in the
/// module documentation. `observed` is the candidate the earlier exact-shape admission run
/// chose for this shape; it is labelled in the output. `occurrences` are the dispatches captured,
/// and the capture must hold exactly those.
fn compare_shape(
    m: usize,
    n: usize,
    k: usize,
    epilogue: Epilogue,
    observed: usize,
    occurrences: &[usize],
) {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let mut rig = Rig::new();
    let mut engine = rig.engine(TuningTable::default());
    let p = problem(m, n, k, epilogue);
    let key = ShapeKey {
        m,
        n,
        k,
        epilogue,
        dtype: GemmDtype::Bf16,
    };
    engine.set_operand_capture(Some(vec![(key, occurrences.to_vec())]));
    rig.run_all(&mut engine);
    let captured = engine.take_operand_capture();
    engine.set_operand_capture(None);
    let got: Vec<usize> = captured.iter().map(|c| c.occurrence).collect();
    println!("m{m} n{n} k{k}: captured occurrences {got:?} of {occurrences:?}");
    assert_eq!(
        got, occurrences,
        "m{m} n{n} k{k}: the fixture batches must dispatch exactly the requested occurrences"
    );
    let synthetic = SyntheticOperands::new(&rig.stream, &p, &engine).expect("synthetic");

    let count = engine.candidate_count(&p).expect("candidates");
    println!(
        "m{m} n{n} k{k} {}: {count} candidates, workspace {} bytes, earlier admission chose candidate {observed}",
        epilogue.name(),
        engine.workspace().len
    );
    for idx in 0..count {
        println!(
            "candidate {idx}: [{}]",
            engine.candidate_identity(&p, idx).expect("id").summary()
        );
    }

    let mut sets: Vec<OperandSet> = captured
        .iter()
        .map(|c| {
            assert_eq!(c.problem(), p);
            OperandSet {
                label: format!("real occurrence {}", c.occurrence),
                source: Source::Real(c),
                ops: c.operands(),
                reference: Vec::new(),
            }
        })
        .collect();
    sets.push(OperandSet {
        label: "synthetic".to_owned(),
        source: Source::Synthetic(&synthetic, m, n),
        ops: synthetic.operands(),
        reference: Vec::new(),
    });
    for set in &mut sets {
        set.reference = run_out(&mut engine, &p, set, 0);
    }

    let mut eligible = Vec::new();
    for idx in 1..count {
        let mut equal_on_real = true;
        let mut notes = Vec::new();
        for set in &sets {
            let diff = differing(&set.reference, &run_out(&mut engine, &p, set, idx));
            equal_on_real &= !set.is_real() || diff == 0;
            notes.push(format!(
                "{}: {diff} of {} differ",
                set.label,
                set.reference.len() / 2
            ));
        }
        println!("candidate {idx} vs candidate 0: {}", notes.join(", "));
        if equal_on_real {
            eligible.push(idx);
        }
    }

    let mut flush = L2Flush::new(&rig.stream).expect("flush buffer");
    println!(
        "\npaired timing: {PAIRS} pairs per row, random order, L2 flush buffer {} bytes; diff = candidate minus candidate 0 (us); interval = 95% bootstrap of the median diff; admit = p75 diff below -{:.0}% of candidate 0's median",
        flush.len(),
        100.0 * ADMIT_MARGIN
    );
    println!(
        "| operands | cache | candidate | cand 0 median us | candidate median us | median diff | p10 / p90 | interval | verdict | admit |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");
    let mut seed = 0x5eed_0000_u64 ^ ((m as u64) << 32) ^ ((n as u64) << 16) ^ k as u64;
    for &idx in &eligible {
        for set in &sets {
            seed = seed.wrapping_add(0x9e37_79b9);
            measure(&mut engine, &p, set, idx, None, seed);
            seed = seed.wrapping_add(0x9e37_79b9);
            measure(&mut engine, &p, set, idx, Some(&mut flush), seed);
        }
    }
    if eligible.is_empty() {
        println!("no candidate is bit-identical to candidate 0 on the captured operands");
    }

    let mut fresh = rig.engine(TuningTable::default());
    let started = std::time::Instant::now();
    let admitted = fresh
        .autotune_in_memory(&[p], ADMIT_ITERATIONS)
        .expect("autotune");
    let secs = started.elapsed().as_secs_f64();
    let t = admitted.first().expect("admitted");
    let best = t.best.map_or("none eligible".to_owned(), |(i, b)| {
        format!(
            "best {i}: median diff {:+.2}us p75 {:+.2}us",
            b.diff_median, b.diff_p75
        )
    });
    println!(
        "\ncurrent admission ({secs:.3}s): candidate {} of {} ({} not bit-identical on synthetic rows), {:.1}us -> {:.1}us, {best}",
        t.algo_index, t.candidates, t.rejected, t.first_micros, t.micros
    );
    let chosen = fresh.tuned_index(&p).expect("selection");
    for set in &sets {
        let out = run_out(&mut fresh, &p, set, chosen);
        assert!(
            differing(&set.reference, &out) == 0,
            "{}: admitted candidate {chosen} differs from candidate 0",
            set.label
        );
    }
    let result = if t.algo_index == 0 {
        "keep baseline"
    } else {
        "candidate pinned"
    };
    println!("result for m{m} n{n} k{k} {}: {result}", epilogue.name());
}

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn paired_m2432_n3072_k1024_bias() {
    compare_shape(2432, 3072, 1024, Epilogue::Bias, 1, LAYER_OCCURRENCES);
}

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn paired_m48_n8192_k1024_bias() {
    compare_shape(48, 8192, 1024, Epilogue::Bias, 2, LAYER_OCCURRENCES);
}

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn paired_m80_n8192_k1024_bias() {
    compare_shape(80, 8192, 1024, Epilogue::Bias, 1, LAYER_OCCURRENCES);
}

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn paired_m608_n9216_k1536_none() {
    compare_shape(608, 9216, 1536, Epilogue::None, 2, SINGLE_OCCURRENCE);
}
