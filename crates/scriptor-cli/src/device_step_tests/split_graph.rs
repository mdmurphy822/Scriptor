//! GPU checks of the two-stage target selector inside the captured device-step graphs.
//!
//! * Replay: a split-selector executor and a single-block executor are driven in lockstep over
//!   changing logits, parameters, history and drafts; the same bucket goes greedy, then mixed
//!   sampled, then greedy again, with finished sequences leaving inactive rows inside the
//!   bucket. Targets, margin bits, accepted tokens and the device state must match every step.
//! * Accounting: the incremental graph memory and capture time of the split variants.
//! * End to end: wall-clock decode time of both executors, alternating order.
//!
//! The selector is enabled through the experimental `SCRIPTOR_SPLIT_SELECT` variable while the
//! executor is built. Run with
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-cli --features cuda --profile gpu-test split_graph -- --ignored --nocapture --test-threads=1`.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use scriptor_engine::{DecodeRow, DeviceRow, ModelExecutor, StepRowResult};
use scriptor_kernels::sampling::RowParams;

use super::{Fixture, MAX_SEQS, Seq, TOKENS, executor, fixture};
use crate::device_step::{CaptureReport, StateBits};
use crate::gpu_executor::GpuExecutor;

/// Selector choice under test: 128-thread blocks, four per row, every row count of the tests.
const SPLIT: &str = "128x4x32";
/// Longest a test may run.
const BUDGET: Duration = Duration::from_secs(30);

static ENV: Mutex<()> = Mutex::new(());

/// Builds an executor with the split selector set to `split` (`None` leaves it off).
fn executor_split(fx: &Fixture, depth: usize, split: Option<&str>) -> GpuExecutor {
    let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: the variable is read once while the executor is built, under the lock, and
    // removed straight after; the GPU tests run with `--test-threads=1`.
    unsafe {
        match split {
            Some(v) => std::env::set_var("SCRIPTOR_SPLIT_SELECT", v),
            None => std::env::remove_var("SCRIPTOR_SPLIT_SELECT"),
        }
    }
    let exec = executor(fx, depth);
    // SAFETY: as above.
    unsafe { std::env::remove_var("SCRIPTOR_SPLIT_SELECT") };
    exec
}

/// How a step's rows sample.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Mode {
    /// Every active row takes the argmax.
    Greedy,
    /// Even rows sample, odd rows take the argmax.
    Mixed,
    /// Every active row samples.
    Sampled,
}

pub(super) fn params_in(seq: &Seq, index: usize, mode: Mode) -> RowParams {
    let sampled = match mode {
        Mode::Greedy => false,
        Mode::Mixed => index.is_multiple_of(2),
        Mode::Sampled => true,
    };
    RowParams {
        seed: 7 + index as u64,
        repetition: [1.0, 1.1, 1.25][index % 3],
        presence: [0.0, 0.5][index % 2],
        frequency: [0.0, 0.25, 0.0][index % 3],
        temperature: if sampled { 0.9 } else { 0.0 },
        top_p: if sampled { 0.95 } else { 1.0 },
        top_k: if sampled { 40 } else { 1 },
        step: seq.out.len() as u32,
        slot: 0,
    }
}

/// One device step over the sequences at `active`; their emitted tokens are appended.
pub(super) fn step_subset(
    exec: &mut GpuExecutor,
    seqs: &mut [Seq],
    active: &[usize],
    mode: Mode,
) -> Vec<StepRowResult> {
    let rows: Vec<DecodeRow> = active.iter().map(|&i| seqs[i].row()).collect();
    let device_rows: Vec<DeviceRow<'_>> = active
        .iter()
        .zip(&rows)
        .map(|(&i, row)| DeviceRow {
            row,
            params: params_in(&seqs[i], i, mode),
            masks: &seqs[i].masks,
        })
        .collect();
    let results = exec.decode_device(&device_rows).expect("decode_device");
    drop(device_rows);
    for (&i, r) in active.iter().zip(&results) {
        assert!(!r.tokens.is_empty() && r.tokens.len() == r.margins.len());
        for &t in &r.tokens {
            seqs[i].push(t);
        }
    }
    results
}

fn margin_bits(r: &StepRowResult) -> Vec<u32> {
    r.margins.iter().map(|m| m.to_bits()).collect()
}

fn mode_of_step(step: usize) -> Mode {
    match step % 12 {
        0..=3 => Mode::Greedy,
        4..=7 => Mode::Mixed,
        8..=9 => Mode::Sampled,
        _ => Mode::Greedy,
    }
}

fn assert_states_equal(a: &StateBits, b: &StateBits, what: &str) {
    assert_eq!(a.targets, b.targets, "{what}: targets");
    assert_eq!(a.margins, b.margins, "{what}: margin bits");
    assert_eq!(a.hidden, b.hidden, "{what}: hidden states");
    assert_eq!(a.logits, b.logits, "{what}: fp32 logits");
}

fn check_graph_replay(depth: usize) {
    let fx = fixture(&format!("split-replay-k{depth}"));
    let mut off = executor_split(&fx, depth, None);
    let mut on = executor_split(&fx, depth, Some(SPLIT));
    let report_off = off.capture_graphs().expect("capture off");
    let report_on = on.capture_graphs().expect("capture on");
    assert_eq!(report_off.split_graphs, 0);
    assert!(report_on.split_graphs > 0, "the selector served no bucket");
    assert_eq!(report_on.graphs, report_off.graphs + report_on.split_graphs);

    let mut greedy_steps = 0u64;
    let mut steps = 0u64;
    for (round, count) in [3usize, 5, 2, 7].into_iter().enumerate() {
        let seed = 0xfeed_beef_1234_5678 ^ ((round as u64 + 1) * 0x9e37_79b9);
        let mut seqs_off = super::seeded_sequences(count, seed);
        let mut seqs_on = super::seeded_sequences(count, seed);
        super::start(&mut off, &mut seqs_off);
        super::start(&mut on, &mut seqs_on);
        let mut step = 0usize;
        loop {
            // Sequences that reached their length leave inactive rows inside the bucket; the
            // first and last stay longest.
            let active: Vec<usize> = (0..count)
                .filter(|&i| seqs_off[i].out.len() < TOKENS - (i % 3) * 4)
                .collect();
            if active.is_empty() {
                break;
            }
            let mode = mode_of_step(step);
            let want = step_subset(&mut off, &mut seqs_off, &active, mode);
            let got = step_subset(&mut on, &mut seqs_on, &active, mode);
            let what = format!("k {depth} round {round} step {step} {mode:?}");
            for (i, (w, g)) in want.iter().zip(&got).enumerate() {
                assert_eq!(g.tokens, w.tokens, "{what} row {i} accepted tokens");
                assert_eq!(margin_bits(g), margin_bits(w), "{what} row {i} margin bits");
            }
            assert_states_equal(
                &on.step_bits().expect("on bits"),
                &off.step_bits().expect("off bits"),
                &what,
            );
            steps += 1;
            greedy_steps += u64::from(mode == Mode::Greedy);
            step += 1;
        }
        for (a, b) in seqs_off.iter().zip(&seqs_on) {
            assert_eq!(
                a.out, b.out,
                "k {depth} round {round} output of sequence {}",
                a.id
            );
            off.release(a.id);
            on.release(b.id);
        }
    }
    let (s_off, s_on) = (off.graph_stats(), on.graph_stats());
    assert_eq!(
        (s_off.fallbacks, s_on.fallbacks),
        (0, 0),
        "a step ran eagerly"
    );
    assert_eq!(s_off.replays, steps);
    assert_eq!(s_on.replays, steps);
    assert_eq!(s_off.split_replays, 0);
    assert_eq!(
        s_on.split_replays, greedy_steps,
        "split graphs replay exactly on the all-greedy steps"
    );
    assert!(
        greedy_steps > 0 && greedy_steps < steps,
        "modes were not mixed"
    );
    eprintln!(
        "SPLIT_REPLAY k={depth} steps {steps} split_replays {} single_replays {}",
        s_on.split_replays,
        s_on.replays - s_on.split_replays
    );
}

#[test]
#[ignore = "needs GPU"]
fn split_graph_replay_matches_off_through_greedy_mixed_greedy_and_padding_without_drafts() {
    check_graph_replay(0);
}

#[test]
#[ignore = "needs GPU"]
fn split_graph_replay_matches_off_through_greedy_mixed_greedy_and_padding_with_three_drafts() {
    check_graph_replay(3);
}

fn print_capture(label: &str, depth: usize, r: &CaptureReport) {
    eprintln!(
        "SPLIT_GRAPH_ACCOUNTING {label} k={depth} graphs {} split_graphs {} \
         observed_delta_mib {:.1} capture_s {:.3} split_observed_delta_mib {:.1} split_capture_s {:.3} scratch_requested_mib {:.1} scope {:?} memory {:?}",
        r.graphs,
        r.split_graphs,
        r.memory.bytes() as f64 / f64::from(1 << 20),
        r.seconds,
        r.split_memory.bytes() as f64 / f64::from(1 << 20),
        r.split_seconds,
        r.split_scratch_bytes as f64 / f64::from(1 << 20),
        r.scope,
        r.memory
    );
}

fn check_accounting(depth: usize) {
    let started = Instant::now();
    let fx = fixture(&format!("split-accounting-k{depth}"));
    let mut seen: Vec<(bool, CaptureReport)> = Vec::new();
    // Alternate which configuration is built first so allocator and module-load warm-up fall on
    // each in turn.
    for on_first in [false, true, true, false] {
        let mut exec = executor_split(&fx, depth, on_first.then_some(SPLIT));
        let report = exec.capture_graphs().expect("capture");
        print_capture(if on_first { "on" } else { "off" }, depth, &report);
        seen.push((on_first, report));
        drop(exec);
    }
    for (on, r) in &seen {
        assert_eq!(r.split_graphs > 0, *on);
        assert_eq!(r.split_scratch_bytes > 0, *on);
        assert_eq!(r.split_memory.bytes() > 0 || r.split_seconds > 0.0, *on);
    }
    let mean = |on: bool, f: fn(&CaptureReport) -> f64| {
        let v: Vec<f64> = seen
            .iter()
            .filter(|(o, _)| *o == on)
            .map(|(_, r)| f(r))
            .collect();
        v.iter().sum::<f64>() / v.len() as f64
    };
    eprintln!(
        "SPLIT_GRAPH_ACCOUNTING_DELTA k={depth} incremental_observed_delta_mib {:.1} \
         incremental_capture_s {:.3} (mean of two builds each; observed bytes are the driver's \
         free-memory delta, granular and shared with other processes)",
        (mean(true, |r| r.memory.bytes() as f64) - mean(false, |r| r.memory.bytes() as f64))
            / f64::from(1 << 20),
        mean(true, |r| r.seconds) - mean(false, |r| r.seconds)
    );
    assert!(started.elapsed() < BUDGET, "test exceeded its time budget");
}

#[test]
#[ignore = "needs GPU"]
fn split_graph_accounting_prints_incremental_memory_and_capture_time_without_drafts() {
    check_accounting(0);
}

#[test]
#[ignore = "needs GPU"]
fn split_graph_accounting_prints_incremental_memory_and_capture_time_with_three_drafts() {
    check_accounting(3);
}

/// Decodes `count` sequences to their length and returns microseconds per device step.
fn timed_decode(exec: &mut GpuExecutor, count: usize, seed: u64) -> (f64, Vec<Vec<u32>>) {
    let mut seqs = super::seeded_sequences(count, seed);
    super::start(exec, &mut seqs);
    let all: Vec<usize> = (0..count).collect();
    let started = Instant::now();
    let mut steps = 0u32;
    while seqs.iter().any(|s| s.out.len() < TOKENS) {
        step_subset(exec, &mut seqs, &all, Mode::Greedy);
        steps += 1;
    }
    let us = started.elapsed().as_secs_f64() * 1e6 / f64::from(steps);
    let out = seqs.iter().map(|s| s.out.clone()).collect();
    for s in &seqs {
        exec.release(s.id);
    }
    (us, out)
}

#[test]
#[ignore = "needs GPU"]
fn split_end_to_end_decode_wall_clock_alternates_order_and_matches_tokens() {
    let started = Instant::now();
    let fx = fixture("split-e2e");
    let mut off = executor_split(&fx, 3, None);
    let mut on = executor_split(&fx, 3, Some(SPLIT));
    off.capture_graphs().expect("capture off");
    on.capture_graphs().expect("capture on");
    for count in [1usize, 4, MAX_SEQS] {
        let (mut offs, mut ons) = (Vec::new(), Vec::new());
        for repeat in 0..4 {
            let seed = 0xabcd_0000 + repeat;
            let (off_run, on_run);
            if repeat % 2 == 0 {
                off_run = timed_decode(&mut off, count, seed);
                on_run = timed_decode(&mut on, count, seed);
            } else {
                on_run = timed_decode(&mut on, count, seed);
                off_run = timed_decode(&mut off, count, seed);
            }
            let ((o, want), (n, got)) = (off_run, on_run);
            assert_eq!(
                got, want,
                "tokens diverged at {count} sequences repeat {repeat}"
            );
            eprintln!(
                "E2E_DECODE_US_PER_STEP (synthetic 2-layer checkpoint, host wall clock) \
                 seqs {count} repeat {repeat} order {} off {o:.1} on {n:.1}",
                if repeat % 2 == 0 {
                    "off-first"
                } else {
                    "on-first"
                }
            );
            offs.push(o);
            ons.push(n);
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (o, n) = (med(&mut offs), med(&mut ons));
        eprintln!(
            "E2E_DECODE_US_PER_STEP_MEDIAN seqs {count} off {o:.1} on {n:.1} ({:.3}x)",
            o / n
        );
    }
    assert!(
        on.graph_stats().split_replays > 0,
        "no step replayed a split graph"
    );
    assert!(started.elapsed() < BUDGET, "test exceeded its time budget");
}
