//! Times every cuBLASLt candidate of the vision tower's GEMM shapes at several row counts.
//! Run with `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda --release --test vision_gemm_shapes_gpu -- --ignored --nocapture`.
#![cfg(feature = "cuda")]

use scriptor_cuda::{Device, DeviceArena};
use scriptor_kernels::cublaslt::{GemmEngine, GemmOptions, GemmProblem, SyntheticOperands};
use scriptor_kernels::heuristics::{Epilogue, GemmDtype, TuningTable, median};

#[test]
#[ignore = "needs a GPU"]
fn vision_gemm_candidates_by_rows() {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let mut arena = dev.arena(512 << 20, &stream).expect("arena");
    let mut engine = GemmEngine::new(
        dev.context(),
        &stream,
        &mut arena,
        TuningTable::default(),
        GemmOptions {
            fp8: false,
            workspace_bytes: 64 << 20,
        },
    )
    .expect("engine");
    let shapes = [
        ("gate_up", 8192usize, 1024usize),
        ("half", 4096, 1024),
        ("qkv", 3072, 1024),
        ("proj", 1024, 1024),
        ("down", 1024, 4096),
    ];
    for (name, n, k) in shapes {
        for m in [2540usize, 10080] {
            let p = GemmProblem {
                m,
                n,
                k,
                dtype: GemmDtype::Bf16,
                epilogue: Epilogue::None,
            };
            let ops = SyntheticOperands::new(&stream, &p, &engine).expect("ops");
            let ops = ops.operands();
            let count = engine.candidate_count(&p).expect("candidates");
            let mut line = String::new();
            let mut best = (0usize, f64::MAX);
            let mut first = 0.0;
            for idx in 0..count {
                let t = engine.time_gemm(&p, &ops, idx, 2, 5).expect("time");
                let us = median(&t).unwrap_or(f64::MAX);
                if idx == 0 {
                    first = us;
                }
                if us < best.1 {
                    best = (idx, us);
                }
                line.push_str(&format!(" {us:.0}"));
            }
            let tf = |us: f64| 2.0 * (m * n * k) as f64 / (us * 1e-6) / 1e12;
            println!(
                "{name} m={m}: first {first:.0}us ({:.0} TF) best idx {} {:.0}us ({:.0} TF) | all:{line}",
                tf(first),
                best.0,
                best.1,
                tf(best.1)
            );
        }
    }
}

/// A fresh engine and the arena holding its workspace, which must outlive it.
fn engine(
    dev: &Device,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> (GemmEngine, DeviceArena) {
    let mut arena = dev.arena(512 << 20, stream).expect("arena");
    let engine = GemmEngine::new(
        dev.context(),
        stream,
        &mut arena,
        TuningTable::default(),
        GemmOptions {
            fp8: false,
            workspace_bytes: 64 << 20,
        },
    )
    .expect("engine");
    (engine, arena)
}

fn tune_problems() -> Vec<GemmProblem> {
    let shapes = [
        (3072usize, 1024usize, Epilogue::Bias),
        (1024, 1024, Epilogue::Bias),
        (1024, 4096, Epilogue::None),
        (8192, 1024, Epilogue::Bias),
        (4096, 1536, Epilogue::None),
        (1536, 2048, Epilogue::None),
        (9216, 1536, Epilogue::None),
        (1536, 4608, Epilogue::None),
    ];
    let mut problems = Vec::new();
    for (n, k, epilogue) in shapes {
        for m in [1024usize, 2048, 4096, 8192, 16384] {
            problems.push(GemmProblem {
                m,
                n,
                k,
                dtype: GemmDtype::Bf16,
                epilogue,
            });
        }
    }
    problems
}

/// Runs the in-memory autotune over the vision and text GEMM shapes and prints, per key, the
/// pinned candidate with its time.
#[test]
#[ignore = "needs a GPU"]
fn autotune_in_memory_reports_pinned_candidates() {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let (mut engine, _arena) = engine(&dev, &stream);
    let started = std::time::Instant::now();
    let tuned = engine
        .autotune_in_memory(&tune_problems(), 5)
        .expect("autotune");
    println!("autotune took {:.1}s", started.elapsed().as_secs_f64());
    for t in tuned {
        println!(
            "m{} n{} k{} {}: cands {} not-bit-identical {} pinned {} | {:.0}us -> {:.0}us ({:+.1}%)",
            t.key.m_bucket,
            t.key.n,
            t.key.k,
            t.key.epilogue.name(),
            t.candidates,
            t.rejected,
            t.algo_index,
            t.first_micros,
            t.micros,
            (t.micros / t.first_micros - 1.0) * 100.0,
        );
    }
}

/// Tunes twice on fresh engines. Whatever candidates the two boots pin, the pinned candidate's
/// full output must equal candidate 0's bit for bit, so both boots compute identical results.
#[test]
#[ignore = "needs a GPU"]
fn two_boots_pin_candidates_with_identical_output() {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let problems = tune_problems();
    let mut boots = Vec::new();
    for _ in 0..2 {
        let (mut engine, arena) = engine(&dev, &stream);
        let tuned = engine.autotune_in_memory(&problems, 5).expect("autotune");
        boots.push((engine, tuned, arena));
    }
    let mut differing_choices = 0;
    for (a, b) in boots[0].1.iter().zip(&boots[1].1) {
        assert_eq!(a.key, b.key);
        differing_choices += usize::from(a.algo_index != b.algo_index);
    }
    println!("keys with differing pinned index across boots: {differing_choices}");
    for (engine, tuned, _) in &mut boots {
        for t in tuned.iter() {
            let p = GemmProblem {
                m: t.key.m_bucket,
                n: t.key.n,
                k: t.key.k,
                dtype: GemmDtype::Bf16,
                epilogue: t.key.epilogue,
            };
            let operands = SyntheticOperands::new(&stream, &p, engine).expect("operands");
            let ops = operands.operands();
            let all_rows: Vec<usize> = (0..p.m).collect();
            engine.gemm_with(&p, &ops, 0).expect("candidate 0");
            let reference = operands.host_out_rows(&all_rows, p.n).expect("rows");
            let pinned_index = engine.tuned_index(&p).expect("selection");
            engine.gemm_with(&p, &ops, pinned_index).expect("pinned");
            let pinned = operands.host_out_rows(&all_rows, p.n).expect("rows");
            assert!(
                reference == pinned,
                "m{} n{} k{}: pinned candidate {} differs from candidate 0",
                p.m,
                p.n,
                p.k,
                t.algo_index
            );
        }
    }
}

/// After the in-memory autotune, a problem at the admitted row count runs the candidate whose
/// identity equals the admitted choice, and a non-power-of-two row count inside the same bucket
/// runs its own first candidate, bit for bit equal to an explicit candidate-0 launch.
#[test]
#[ignore = "needs a GPU"]
fn tuned_choice_binds_to_the_admitted_row_count_only() {
    use scriptor_kernels::heuristics::{BaselineReason, Selection};
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let (mut engine, _arena) = engine(&dev, &stream);
    let tuned = engine
        .autotune_in_memory(&tune_problems(), 5)
        .expect("autotune");
    for t in &tuned {
        let shape = |m: usize| GemmProblem {
            m,
            n: t.key.n,
            k: t.key.k,
            dtype: GemmDtype::Bf16,
            epilogue: t.key.epilogue,
        };
        let at = shape(t.m);
        let sel = engine.selection(&at).expect("selection");
        let idx = sel.index();
        assert!(matches!(sel, Selection::Tuned(_)), "m{}: {sel:?}", t.m);
        assert_eq!(engine.candidate_identity(&at, idx).expect("id"), t.chosen);
        let odd = shape(t.m - 37);
        assert_eq!(
            engine.selection(&odd).expect("selection"),
            Selection::Baseline(BaselineReason::Untuned)
        );
        let operands = SyntheticOperands::new(&stream, &odd, &engine).expect("operands");
        let ops = operands.operands();
        let rows: Vec<usize> = (0..odd.m).collect();
        engine.gemm_with(&odd, &ops, 0).expect("candidate 0");
        let reference = operands.host_out_rows(&rows, odd.n).expect("rows");
        engine.gemm(&odd, &ops).expect("gemm");
        let got = operands.host_out_rows(&rows, odd.n).expect("rows");
        assert!(reference == got, "m{} n{} k{}", odd.m, odd.n, odd.k);
        println!(
            "m{} n{} k{} {}: pinned {} [{}] | m{} -> baseline [{}]",
            t.m,
            t.key.n,
            t.key.k,
            t.key.epilogue.name(),
            idx,
            t.chosen.summary(),
            odd.m,
            engine.candidate_identity(&odd, 0).expect("id").summary(),
        );
    }
}
