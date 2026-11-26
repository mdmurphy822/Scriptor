//! Exact-shape cuBLASLt admission on the real vision tower over real fixture images. A
//! baseline engine (no tuning, every GEMM on its exact shape's first candidate) runs each image
//! and one batch of all images while recording the dispatched shapes with GPU time; a second
//! engine admits exactly those shapes and runs the same inputs. Every output embedding must be
//! bit-identical to the baseline's. A second test records the admission cost of a first boot
//! and the reuse of its saved table by a second boot. Set `SCRIPTOR_SNAPSHOT_DIR` to the
//! checkpoint directory (or leave it unset to use the Hugging Face cache) and run
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda --profile gpu-test --test gpu_exact_shape_real -- --ignored --nocapture --test-threads 1`.
#![cfg(feature = "cuda")]

mod common;

use scriptor_kernels::cublaslt::{BootTuned, GemmEngine, GemmProblem};
use scriptor_kernels::heuristics::{Selection, ShapeInventory, TuningTable};

use common::fixtures::Rig;

/// Most observed shapes admitted.
const MAX_SHAPES: usize = 48;
/// Paired samples per candidate requested during admission (admission takes at least
/// `ADMIT_MIN_PAIRS`).
const TUNE_ITERATIONS: usize = 7;

fn total_micros(inv: &ShapeInventory) -> f64 {
    inv.ranked().iter().map(|(_, _, s)| s.gpu_micros).sum()
}

/// Runs the rig once untimed, then once on `baseline` recording timed shapes; returns the
/// recorded outputs and inventory.
fn observe(rig: &mut Rig, baseline: &mut GemmEngine) -> (Vec<Vec<Vec<u16>>>, ShapeInventory) {
    for b in &rig.batches {
        println!("{}: grids {:?}, {} patch rows", b.label, b.grids, b.rows());
    }
    rig.run_all(baseline);
    baseline.set_shape_phase("vision");
    baseline.set_shape_log(Some(true));
    let reference = rig.run_all(baseline);
    let observed = baseline.take_shape_log();
    baseline.set_shape_log(None);
    (reference, observed)
}

fn problems(observed: &ShapeInventory) -> Vec<GemmProblem> {
    observed
        .admission_shapes(MAX_SHAPES)
        .into_iter()
        .map(|k| GemmProblem {
            m: k.m,
            n: k.n,
            k: k.k,
            dtype: k.dtype,
            epilogue: k.epilogue,
        })
        .collect()
}

fn print_admitted(admitted: &[BootTuned]) {
    for t in admitted {
        let best = t.best.map_or("no eligible candidate".to_owned(), |(i, p)| {
            format!(
                "best candidate {i}: paired diff median {:+.1}us ({:+.1}%), p25 {:+.1} p75 {:+.1}, {} pairs",
                p.diff_median,
                100.0 * p.relative(p.diff_median),
                p.diff_p25,
                p.diff_p75,
                p.pairs
            )
        });
        println!(
            "m{} n{} k{} {}: candidate {} of {} ({} not bit-identical) {:.1}us -> {:.1}us; {best} [{}]",
            t.m,
            t.key.n,
            t.key.k,
            t.key.epilogue.name(),
            t.algo_index,
            t.candidates,
            t.rejected,
            t.first_micros,
            t.micros,
            t.chosen.summary()
        );
    }
}

fn assert_same_outputs(rig: &Rig, want: &[Vec<Vec<u16>>], have: &[Vec<Vec<u16>>]) {
    let mut differing = Vec::new();
    for (b, (w_all, h_all)) in rig.batches.iter().zip(want.iter().zip(have)) {
        for (i, (w, h)) in w_all.iter().zip(h_all).enumerate() {
            let diff = w.iter().zip(h).filter(|(x, y)| x != y).count();
            println!("{} image {i}: {diff} of {} values differ", b.label, w.len());
            if diff > 0 {
                differing.push(format!("{} image {i}: {diff}", b.label));
            }
        }
    }
    assert!(differing.is_empty(), "tuned output differs: {differing:?}");
}

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn exact_shape_admission_keeps_real_vision_output_bit_identical() {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let mut rig = Rig::new();
    let mut baseline = rig.engine(TuningTable::default());
    let mut tuned = rig.engine(TuningTable::default());
    let (reference, observed) = observe(&mut rig, &mut baseline);
    println!("\nbaseline dispatches\n{}", observed.report());

    let problems = problems(&observed);
    let started = std::time::Instant::now();
    let admitted = tuned
        .autotune_in_memory(&problems, TUNE_ITERATIONS)
        .expect("autotune");
    let tune_secs = started.elapsed().as_secs_f64();
    println!(
        "admitted {} of {} observed cuBLASLt shapes in {tune_secs:.2}s",
        admitted.len(),
        problems.len()
    );
    print_admitted(&admitted);

    rig.run_all(&mut tuned);
    tuned.set_shape_phase("vision");
    tuned.set_shape_log(Some(true));
    let got = rig.run_all(&mut tuned);
    let after = tuned.take_shape_log();
    println!("\nexact-shape tuned dispatches\n{}", after.report());
    println!(
        "timed cuBLASLt+skinny GEMM total (serial per-GEMM diagnostic, stream synchronised around each dispatch; not end-to-end time): baseline {:.0}us, tuned {:.0}us",
        total_micros(&observed),
        total_micros(&after)
    );
    let used = after.non_baseline();
    if used.is_empty() {
        println!("receipt: baseline fallback, no admitted speed gain");
    } else {
        for (phase, k, st) in &used {
            let base = observed
                .ranked()
                .into_iter()
                .find(|(p, key, _)| p == phase && key == k)
                .map_or(0.0, |(_, _, s)| s.gpu_micros);
            println!(
                "receipt: {phase} m{} n{} k{} {}: {} non-baseline dispatches via {}, serial gpu {:.0}us vs baseline {:.0}us",
                k.m,
                k.n,
                k.k,
                k.epilogue.name(),
                st.non_baseline_calls,
                st.route,
                st.gpu_micros,
                base
            );
        }
    }
    assert_same_outputs(&rig, &reference, &got);
}

/// First boot: admits every observed shape and saves the table. Second boot: a fresh engine
/// over the reloaded table admits nothing, reports how each shape binds, and reproduces the
/// baseline outputs bit for bit.
#[test]
#[ignore = "needs GPU and the checkpoint"]
fn admission_cost_and_table_reuse_across_two_boots() {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let mut rig = Rig::new();
    let mut baseline = rig.engine(TuningTable::default());
    let (reference, observed) = observe(&mut rig, &mut baseline);
    let problems = problems(&observed);

    let mut first = rig.engine(TuningTable::default());
    let started = std::time::Instant::now();
    let admitted = first
        .autotune_in_memory(&problems, TUNE_ITERATIONS)
        .expect("autotune");
    let first_secs = started.elapsed().as_secs_f64();
    let pinned = admitted.iter().filter(|t| t.algo_index != 0).count();
    println!(
        "boot 1: admitted {} of {} shapes in {first_secs:.2}s, {pinned} pinned off candidate 0",
        admitted.len(),
        problems.len()
    );
    print_admitted(&admitted);
    let path = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("scriptor-exact-shape-{}.toml", std::process::id()))
        .join("gemm-tune.toml");
    first.tuning().write_file(&path).expect("write table");
    drop(first);

    let loaded = TuningTable::load(&path).expect("load table");
    std::fs::remove_dir_all(path.parent().expect("dir")).ok();
    let mut second = rig.engine(loaded);
    let started = std::time::Instant::now();
    let again = second
        .autotune_in_memory(&problems, TUNE_ITERATIONS)
        .expect("autotune");
    let second_secs = started.elapsed().as_secs_f64();
    let mut reused = 0;
    let mut fallbacks = Vec::new();
    for t in &admitted {
        let p = GemmProblem {
            m: t.m,
            n: t.key.n,
            k: t.key.k,
            dtype: t.key.dtype,
            epilogue: t.key.epilogue,
        };
        match second.selection(&p).expect("selection") {
            Selection::Tuned(i) if second.candidate_identity(&p, i).expect("id") == t.chosen => {
                reused += 1
            }
            other => fallbacks.push(format!("m{} n{} k{}: {other:?}", p.m, p.n, p.k)),
        }
    }
    println!(
        "boot 2: admitted {} shapes in {second_secs:.3}s; {reused} of {} saved choices reused, fallbacks {fallbacks:?}",
        again.len(),
        admitted.len()
    );
    assert!(
        again.is_empty(),
        "boot 2 re-admitted {} shapes",
        again.len()
    );
    let got = rig.run_all(&mut second);
    assert_same_outputs(&rig, &reference, &got);
}
