//! Exact-shape cuBLASLt admission on synthetic operands: row counts sharing one bucket are
//! admitted independently, each runs its own admitted algorithm with output bit-identical to
//! its own first candidate, unadmitted neighbours run their first candidate, and a fresh
//! engine over the reloaded table either runs the admitted identity or runs candidate 0 for
//! a reason its own candidate list confirms, printing both engines' candidate lists. Also reports what the former bucket-ordinal selection
//! would have run for the same row counts. Run with
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda --profile gpu-test --test exact_shape_admission_gpu -- --ignored --nocapture --test-threads 1`.
#![cfg(feature = "cuda")]

use scriptor_cuda::{Device, DeviceArena};
use scriptor_kernels::cublaslt::{GemmEngine, GemmOptions, GemmProblem, SyntheticOperands};
use scriptor_kernels::heuristics::{
    AlgoIdentity, BaselineReason, Epilogue, GemmDtype, Selection, TuningTable, m_bucket,
};

/// Weight shapes `(n, k, epilogue)` covered: vision qkv and decoder qkv.
const SHAPES: [(usize, usize, Epilogue); 2] =
    [(3072, 1024, Epilogue::Bias), (4096, 1536, Epilogue::None)];
/// Exact row counts covered; 1100 and 1900 share the 2048 bucket.
const ROWS: [usize; 3] = [1100, 1900, 2048];

/// A fresh engine over `tuning` and the arena holding its workspace, which must outlive it.
fn new_engine(
    dev: &Device,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    tuning: TuningTable,
) -> (GemmEngine, DeviceArena) {
    let mut arena = dev.arena(256 << 20, stream).expect("arena");
    let engine = GemmEngine::new(
        dev.context(),
        stream,
        &mut arena,
        tuning,
        GemmOptions {
            fp8: false,
            workspace_bytes: 64 << 20,
        },
    )
    .expect("engine");
    (engine, arena)
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

/// Every output row of `p` run through `gemm()` and through candidate 0, as bytes.
fn tuned_and_first(
    engine: &mut GemmEngine,
    p: &GemmProblem,
    ops: &SyntheticOperands,
) -> (Vec<u8>, Vec<u8>) {
    let rows: Vec<usize> = (0..p.m).collect();
    engine
        .gemm_with(p, &ops.operands(), 0)
        .expect("candidate 0");
    let first = ops.host_out_rows(&rows, p.n).expect("rows");
    engine.gemm(p, &ops.operands()).expect("gemm");
    let tuned = ops.host_out_rows(&rows, p.n).expect("rows");
    (tuned, first)
}

#[test]
#[ignore = "needs a GPU"]
fn exact_rows_in_one_bucket_admit_and_run_independently() {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let (mut engine, _arena) = new_engine(&dev, &stream, TuningTable::default());
    let problems: Vec<GemmProblem> = SHAPES
        .iter()
        .flat_map(|&(n, k, e)| ROWS.iter().map(move |&m| problem(m, n, k, e)))
        .collect();
    let started = std::time::Instant::now();
    let tuned = engine.autotune_in_memory(&problems, 5).expect("autotune");
    println!(
        "exact admission of {} shapes took {:.2}s",
        problems.len(),
        started.elapsed().as_secs_f64()
    );
    assert_eq!(tuned.len(), problems.len());
    for t in &tuned {
        let p = problem(t.m, t.key.n, t.key.k, t.key.epilogue);
        let sel = engine.selection(&p).expect("selection");
        let Selection::Tuned(idx) = sel else {
            panic!("m{} n{} k{}: {sel:?}", p.m, p.n, p.k);
        };
        assert_eq!(engine.candidate_identity(&p, idx).expect("id"), t.chosen);
        let ops = SyntheticOperands::new(&stream, &p, &engine).expect("operands");
        let (got, first) = tuned_and_first(&mut engine, &p, &ops);
        assert!(
            got == first,
            "m{} n{} k{}: tuned output differs from candidate 0",
            p.m,
            p.n,
            p.k
        );
        let neighbour = problem(p.m + 1, p.n, p.k, p.epilogue);
        assert_eq!(
            engine.selection(&neighbour).expect("selection"),
            Selection::Baseline(BaselineReason::Untuned)
        );
        println!(
            "m{} n{} k{} {}: candidate {} of {} ({} not bit-identical) {:.1}us -> {:.1}us [{}]",
            p.m,
            p.n,
            p.k,
            p.epilogue.name(),
            idx,
            t.candidates,
            t.rejected,
            t.first_micros,
            t.micros,
            t.chosen.summary()
        );
    }

    let reloaded = TuningTable::from_toml(&engine.tuning().to_toml()).expect("reload");
    assert_eq!(&reloaded, engine.tuning());
    let (mut again, _again_arena) = new_engine(&dev, &stream, reloaded);
    let mut reused = 0usize;
    for t in &tuned {
        let p = problem(t.m, t.key.n, t.key.k, t.key.epilogue);
        let key = p.shape_key();
        let binding = again
            .tuning()
            .get(&key)
            .and_then(|e| e.binding)
            .expect("reloaded entry carries its binding");
        let sel = again.selection(&p).expect("selection");
        let fresh = candidates(&mut again, &p);
        let original = candidates(&mut engine, &p);
        println!(
            "reload {key:?}: {sel:?}; bound m {} ws {} lib {}\n  bound baseline {}\n  bound chosen   {}\n  fresh list equals admitting engine's list: {}",
            binding.m,
            binding.workspace_len,
            binding.library_version,
            binding.baseline.encode(),
            binding.chosen.encode(),
            fresh == original
        );
        for (i, id) in fresh.iter().enumerate() {
            let same = original.get(i) == Some(id);
            println!(
                "  fresh[{i}] {}{}{}{}",
                id.encode(),
                if *id == binding.baseline {
                    " =baseline"
                } else {
                    ""
                },
                if *id == binding.chosen {
                    " =chosen"
                } else {
                    ""
                },
                if same {
                    ""
                } else {
                    " (differs from admitting engine)"
                }
            );
        }
        for (i, id) in original.iter().enumerate() {
            if fresh.get(i) != Some(id) {
                println!("  admitting[{i}] {}", id.encode());
            }
        }
        match sel {
            Selection::Tuned(i) => {
                assert_eq!(fresh[i], binding.chosen, "{key:?}: tuned identity");
                assert_eq!(fresh[0], binding.baseline, "{key:?}: tuned baseline");
                reused += 1;
            }
            Selection::Baseline(BaselineReason::BaselineMismatch) => {
                assert_ne!(
                    fresh[0], binding.baseline,
                    "{key:?}: baseline_mismatch reason"
                );
            }
            Selection::Baseline(BaselineReason::NotOffered) => {
                assert_eq!(fresh[0], binding.baseline, "{key:?}: not_offered reason");
                assert!(
                    !fresh.contains(&binding.chosen),
                    "{key:?}: not_offered while the chosen identity is listed"
                );
            }
            Selection::Baseline(r) => {
                panic!("{key:?}: reloaded bound entry fell back for {}", r.name())
            }
        }
        let ops = SyntheticOperands::new(&stream, &p, &again).expect("operands");
        let (got, first) = tuned_and_first(&mut again, &p, &ops);
        assert!(
            got == first,
            "{key:?}: reloaded output differs from candidate 0"
        );
    }
    println!(
        "reloaded table reused {reused} of {} admitted choices; the rest ran their baseline",
        tuned.len()
    );
}

/// Identities of every heuristic candidate of `p` on `engine`, in heuristic order.
fn candidates(engine: &mut GemmEngine, p: &GemmProblem) -> Vec<AlgoIdentity> {
    let count = engine.candidate_count(p).expect("count");
    (0..count)
        .map(|i| engine.candidate_identity(p, i).expect("id"))
        .collect()
}

/// Prints, per exact row count, what the former selection (the bucket's admitted list
/// position applied to the exact list, clamped) would have run, and whether its output
/// matched the exact shape's first candidate.
#[test]
#[ignore = "needs a GPU"]
fn report_former_bucket_ordinal_selection() {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let (mut engine, _arena) = new_engine(&dev, &stream, TuningTable::default());
    let buckets: Vec<GemmProblem> = SHAPES
        .iter()
        .map(|&(n, k, e)| problem(m_bucket(ROWS[0]), n, k, e))
        .collect();
    let tuned = engine.autotune_in_memory(&buckets, 5).expect("autotune");
    for t in &tuned {
        let bucket = problem(t.m, t.key.n, t.key.k, t.key.epilogue);
        for &m in &ROWS[..2] {
            let p = problem(m, bucket.n, bucket.k, bucket.epilogue);
            let count = engine.candidate_count(&p).expect("count");
            let former = t.algo_index.min(count - 1);
            let former_id = engine.candidate_identity(&p, former).expect("id");
            let ops = SyntheticOperands::new(&stream, &p, &engine).expect("operands");
            let rows: Vec<usize> = (0..p.m).collect();
            engine
                .gemm_with(&p, &ops.operands(), 0)
                .expect("candidate 0");
            let first = ops.host_out_rows(&rows, p.n).expect("rows");
            engine
                .gemm_with(&p, &ops.operands(), former)
                .expect("former");
            let got = ops.host_out_rows(&rows, p.n).expect("rows");
            let fixed = engine.selection(&p).expect("selection");
            println!(
                "n{} k{} m{m}: bucket pinned {} [{}]; former ran {} of {count} [{}], same identity as admitted: {}, output equals candidate 0: {}; fixed runs {fixed:?}",
                p.n,
                p.k,
                t.algo_index,
                t.chosen.summary(),
                former,
                former_id.summary(),
                former_id == t.chosen,
                got == first,
            );
        }
    }
}
