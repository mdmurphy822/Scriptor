//! Per-stage timing of the vision tower on the real checkpoint, fused against unfused, over
//! synthetic text images of the sizes Semantik sends. Set `SCRIPTOR_SNAPSHOT_DIR` to the checkpoint
//! directory (or leave it unset to use the Hugging Face cache) and run
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda --release --test gpu_vision_profile_real -- --ignored --nocapture`.
#![cfg(feature = "cuda")]

mod common;

use cudarc::driver::sys::CUevent_flags;
use scriptor_cuda::Device;
use scriptor_kernels::cublaslt::{GemmEngine, GemmOptions, SyntheticOperands};
use scriptor_kernels::heuristics::{Epilogue, GemmDtype, TuningTable};
use scriptor_model::gpu::{Fusion, GpuVision, VisionStageStat, VisionWorkspace};

use common::{open_checkpoint, shared_vision, synthetic_patches};

const WARMUP: usize = 2;
const ITERS: usize = 5;
/// Patches the workspace holds: the checkpoint's 12288-token ceiling.
const BUDGET: usize = 49152;

/// A `(t, h, w)` patch grid.
type Grid = (usize, usize, usize);

/// Named batches of patch grids.
fn batches() -> Vec<(&'static str, Vec<Grid>)> {
    let mixed: Vec<Grid> = vec![
        (1, 8, 60),
        (1, 6, 44),
        (1, 20, 36),
        (1, 12, 28),
        (1, 4, 52),
        (1, 30, 24),
        (1, 10, 64),
        (1, 6, 18),
        (1, 16, 40),
        (1, 8, 32),
        (1, 24, 20),
        (1, 4, 24),
    ];
    vec![
        ("full page 120x84 patches", vec![(1, 120, 84)]),
        ("half column 56x84", vec![(1, 56, 84)]),
        ("small line 4x100", vec![(1, 4, 100)]),
        ("mixed 12 crops", mixed),
    ]
}

fn class(name: &str) -> &'static str {
    match name {
        "patch_embed" => "patch embed",
        "norm" | "join_attn" | "join_mlp" | "join_attn_norm" | "join_mlp_norm" => {
            "norms (+residual)"
        }
        "qkv_gemm" => "QKV GEMM",
        "qk_norm_rope" | "qkv_prep" => "q/k norm + rope",
        "qkv_split" => "q/k/v split",
        "attention" => "attention",
        "proj_gemm" => "out proj GEMM",
        "gate_up_gemm" | "gate_up_silu_gemm" | "down_gemm" => "MLP GEMMs",
        "silu_mul" => "activation",
        "output_copy" => "output copy",
        _ => "merger / downsample",
    }
}

struct Ceilings {
    copy_gbps: f64,
    bf16_tflops: f64,
}

fn measure_ceilings(dev: &Device, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> Ceilings {
    let bytes = 256usize << 20;
    let src = stream.alloc_zeros::<u8>(bytes).expect("src");
    let mut dst = stream.alloc_zeros::<u8>(bytes).expect("dst");
    for _ in 0..3 {
        stream.memcpy_dtod(&src, &mut dst).expect("warm copy");
    }
    stream.synchronize().expect("sync");
    let (a, b) = (
        stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event"),
        {
            for _ in 0..20 {
                stream.memcpy_dtod(&src, &mut dst).expect("copy");
            }
            stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .expect("event")
        },
    );
    let ms = f64::from(a.elapsed_ms(&b).expect("elapsed")) / 20.0;
    let copy_gbps = 2.0 * bytes as f64 / (ms * 1e-3) / 1e9;

    let mut arena = dev.arena(256 << 20, stream).expect("arena");
    let mut gemm = GemmEngine::new(
        dev.context(),
        stream,
        &mut arena,
        TuningTable::default(),
        GemmOptions {
            fp8: false,
            workspace_bytes: 64 << 20,
        },
    )
    .expect("gemm");
    let p = scriptor_kernels::cublaslt::GemmProblem {
        m: 8192,
        n: 8192,
        k: 8192,
        dtype: GemmDtype::Bf16,
        epilogue: Epilogue::None,
    };
    let operands = SyntheticOperands::new(stream, &p, &gemm).expect("operands");
    let count = gemm.candidate_count(&p).expect("candidates");
    let mut best = f64::INFINITY;
    for idx in 0..count {
        let times = gemm
            .time_gemm(&p, &operands.operands(), idx, 2, 5)
            .expect("time");
        best = best.min(times.iter().copied().fold(f64::INFINITY, f64::min));
    }
    let bf16_tflops = 2.0 * 8192f64.powi(3) / (best * 1e-6) / 1e12;
    Ceilings {
        copy_gbps,
        bf16_tflops,
    }
}

fn print_stats(label: &str, wall_ms: f64, stats: &[VisionStageStat], c: &Ceilings) {
    println!("\n{label}: vision wall {wall_ms:.2} ms per batch");
    println!("| stage | calls | ms | TFLOP/s | % of BF16 peak | GB/s | % of copy BW |");
    println!("|---|---|---|---|---|---|---|");
    let iters = ITERS as f64;
    for s in stats {
        let tf = s.tflops();
        let gb = s.gbps();
        println!(
            "| {} | {} | {:.3} | {} | {} | {:.0} | {:.0}% |",
            s.name,
            s.calls / ITERS as u32,
            s.millis / iters,
            if s.flops > 0 {
                format!("{tf:.1}")
            } else {
                "-".into()
            },
            if s.flops > 0 {
                format!("{:.0}%", 100.0 * tf / c.bf16_tflops)
            } else {
                "-".into()
            },
            gb,
            100.0 * gb / c.copy_gbps
        );
    }
    println!(
        "| total | | {:.3} | | | | |",
        stats.iter().map(|s| s.millis).sum::<f64>() / iters
    );
    let mut classes: Vec<(&str, f64, u64, u64)> = Vec::new();
    for s in stats {
        let k = class(s.name);
        match classes.iter_mut().find(|c| c.0 == k) {
            Some(e) => {
                e.1 += s.millis;
                e.2 += s.flops;
                e.3 += s.bytes;
            }
            None => classes.push((k, s.millis, s.flops, s.bytes)),
        }
    }
    println!("\n| op class | ms | share | TFLOP/s | GB/s |");
    println!("|---|---|---|---|---|");
    let total: f64 = classes.iter().map(|c| c.1).sum();
    for (k, ms, fl, by) in classes {
        println!(
            "| {k} | {:.3} | {:.1}% | {} | {:.0} |",
            ms / iters,
            100.0 * ms / total,
            if fl > 0 {
                format!("{:.1}", fl as f64 / (ms * 1e-3) / 1e12)
            } else {
                "-".into()
            },
            by as f64 / (ms * 1e-3) / 1e9
        );
    }
}

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn vision_stage_profile_fused_vs_unfused() {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let (spec, files) = open_checkpoint();
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let ceilings = measure_ceilings(&dev, &stream);
    println!(
        "\nceilings: device copy {:.0} GB/s (read+write), BF16 dense {:.1} TFLOP/s (best cuBLASLt candidate, 8192^3)",
        ceilings.copy_gbps, ceilings.bf16_tflops
    );

    let mut arena = dev.arena(1 << 30, &stream).expect("arena");
    let mut gemm = GemmEngine::new(
        dev.context(),
        &stream,
        &mut arena,
        TuningTable::default(),
        GemmOptions {
            fp8: false,
            workspace_bytes: 64 << 20,
        },
    )
    .expect("gemm");
    let shared = shared_vision(&dev, &spec, &files);
    let out_dim = shared.merger.out_dim;
    let mut vision = GpuVision::with_params(dev.context(), shared.params()).expect("vision");
    let mut ws = VisionWorkspace::new(&stream, vision.params(), BUDGET).expect("workspace");

    for (name, grids) in batches() {
        let mut bits: Vec<u16> = Vec::new();
        for (i, &(t, h, w)) in grids.iter().enumerate() {
            bits.extend(synthetic_patches(t, h, w, 0x9e37_79b9 + i as u64));
        }
        let input = stream.clone_htod(&bits).expect("patches");
        let mut outs: Vec<_> = grids
            .iter()
            .map(|&(t, h, w)| {
                stream
                    .alloc_zeros::<u16>(t * h * w / 4 * out_dim)
                    .expect("out")
            })
            .collect();
        let patches: usize = grids.iter().map(|&(t, h, w)| t * h * w).sum();
        for (fusion, gate_up_silu) in [
            (Fusion::Unfused, false),
            (Fusion::Fused, false),
            (Fusion::Fused, true),
        ] {
            vision.set_fusion(fusion);
            vision.set_gate_up_silu(gate_up_silu);
            vision.set_profile(false);
            let mut run = |vision: &GpuVision, gemm: &mut GemmEngine, ws: &mut VisionWorkspace| {
                let mut refs: Vec<_> = outs.iter_mut().collect();
                vision
                    .run(&stream, gemm, ws, &input, &grids, &mut refs)
                    .expect("run");
            };
            for _ in 0..WARMUP {
                run(&vision, &mut gemm, &mut ws);
            }
            stream.synchronize().expect("sync");
            let a = stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .expect("event");
            for _ in 0..ITERS {
                run(&vision, &mut gemm, &mut ws);
            }
            let b = stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .expect("event");
            let wall_ms = f64::from(a.elapsed_ms(&b).expect("elapsed")) / ITERS as f64;
            vision.set_profile(true);
            for _ in 0..ITERS {
                run(&vision, &mut gemm, &mut ws);
            }
            let stats = vision.take_profile(&stream).expect("profile");
            print_stats(
                &format!("{name} ({patches} patches), {fusion:?}, gate/up silu {gate_up_silu}"),
                wall_ms,
                &stats,
                &ceilings,
            );
        }
    }
}

/// Workspace bytes of the sweep engine; `SCRIPTOR_SWEEP_WS_MB` overrides the default of 64.
fn ws_bytes() -> usize {
    std::env::var("SCRIPTOR_SWEEP_WS_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64)
        << 20
}

/// Epilogue of the swept problems; `SCRIPTOR_SWEEP_NOBIAS=1` selects none.
fn sweep_epilogue() -> Epilogue {
    if std::env::var("SCRIPTOR_SWEEP_NOBIAS").is_ok() {
        Epilogue::None
    } else {
        Epilogue::Bias
    }
}

/// Rows swept by [`vision_gemm_candidate_sweep`]; `SCRIPTOR_SWEEP_ROWS` (comma separated) overrides.
fn sweep_rows() -> Vec<usize> {
    std::env::var("SCRIPTOR_SWEEP_ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![10080, 4704, 400, 2520])
}

/// Heuristic candidates of each tower GEMM at the rows of `m`: first-candidate time against the
/// fastest candidate.
#[test]
#[ignore = "needs GPU"]
fn vision_gemm_candidate_sweep() {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let mut arena = dev.arena(ws_bytes() + (4 << 20), &stream).expect("arena");
    let mut gemm = GemmEngine::new(
        dev.context(),
        &stream,
        &mut arena,
        TuningTable::default(),
        GemmOptions {
            fp8: false,
            workspace_bytes: ws_bytes(),
        },
    )
    .expect("gemm");
    println!("| m | n | k | epilogue | candidates | first us | best us | best idx |");
    println!("|---|---|---|---|---|---|---|---|");
    for m in sweep_rows() {
        for (n, k) in [
            (3072usize, 1024usize),
            (1024, 1024),
            (8192, 1024),
            (1024, 4096),
        ] {
            let p = scriptor_kernels::cublaslt::GemmProblem {
                m,
                n,
                k,
                dtype: GemmDtype::Bf16,
                epilogue: sweep_epilogue(),
            };
            let operands = SyntheticOperands::new(&stream, &p, &gemm).expect("operands");
            let count = gemm.candidate_count(&p).expect("candidates");
            let mut times = Vec::new();
            for idx in 0..count {
                let t = gemm
                    .time_gemm(&p, &operands.operands(), idx, 2, 5)
                    .expect("time");
                times.push(t.iter().copied().fold(f64::INFINITY, f64::min));
            }
            let (bi, bt) =
                times.iter().enumerate().fold(
                    (0, f64::INFINITY),
                    |a, (i, &t)| if t < a.1 { (i, t) } else { a },
                );
            println!(
                "| {m} | {n} | {k} | {:?} | {count} | {:.0} | {bt:.0} | {bi} |",
                p.epilogue, times[0]
            );
        }
    }
}

/// Tower wall time and whole-tower TFLOP/s against total patches per batch, for batches of
/// equal 400-patch crops (which isolates batching from per-image attention growth) and for one
/// page-shaped image of the same patch count. `SCRIPTOR_SWEEP_BATCH_PATCHES` (comma separated)
/// overrides the swept sizes.
#[test]
#[ignore = "needs GPU and the checkpoint"]
fn vision_throughput_against_batch_patches() {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let (spec, files) = open_checkpoint();
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let mut arena = dev.arena(1 << 30, &stream).expect("arena");
    let mut gemm = GemmEngine::new(
        dev.context(),
        &stream,
        &mut arena,
        TuningTable::default(),
        GemmOptions {
            fp8: false,
            workspace_bytes: 64 << 20,
        },
    )
    .expect("gemm");
    let shared = shared_vision(&dev, &spec, &files);
    let out_dim = shared.merger.out_dim;
    let mut vision = GpuVision::with_params(dev.context(), shared.params()).expect("vision");
    let mut ws = VisionWorkspace::new(&stream, vision.params(), BUDGET).expect("workspace");
    let sizes: Vec<usize> = std::env::var("SCRIPTOR_SWEEP_BATCH_PATCHES")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| {
            vec![
                400, 800, 1200, 1600, 2000, 2400, 3200, 4000, 4800, 6400, 8000, 9600, 12000,
            ]
        });
    println!("| shape | patches | ms | TFLOP/s | us per patch |");
    println!("|---|---|---|---|---|");
    for &target in &sizes {
        let crops = target / 400;
        let page_rows = (target / 84).max(2) & !1;
        let shapes: [(&str, Vec<Grid>); 2] = [
            ("400-patch crops", vec![(1, 4, 100); crops.max(1)]),
            ("one page-shaped image", vec![(1, page_rows, 84)]),
        ];
        for (label, grids) in shapes {
            let mut bits: Vec<u16> = Vec::new();
            for (i, &(t, h, w)) in grids.iter().enumerate() {
                bits.extend(synthetic_patches(t, h, w, 0x9e37_79b9 + i as u64));
            }
            let input = stream.clone_htod(&bits).expect("patches");
            let mut outs: Vec<_> = grids
                .iter()
                .map(|&(t, h, w)| {
                    stream
                        .alloc_zeros::<u16>(t * h * w / 4 * out_dim)
                        .expect("out")
                })
                .collect();
            let patches: usize = grids.iter().map(|&(t, h, w)| t * h * w).sum();
            vision.set_fusion(Fusion::Fused);
            vision.set_gate_up_silu(true);
            vision.set_profile(false);
            let mut run = |vision: &GpuVision, gemm: &mut GemmEngine, ws: &mut VisionWorkspace| {
                let mut refs: Vec<_> = outs.iter_mut().collect();
                vision
                    .run(&stream, gemm, ws, &input, &grids, &mut refs)
                    .expect("run");
            };
            for _ in 0..WARMUP {
                run(&vision, &mut gemm, &mut ws);
            }
            stream.synchronize().expect("sync");
            let a = stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .expect("event");
            for _ in 0..ITERS {
                run(&vision, &mut gemm, &mut ws);
            }
            let b = stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .expect("event");
            let wall_ms = f64::from(a.elapsed_ms(&b).expect("elapsed")) / ITERS as f64;
            vision.set_profile(true);
            for _ in 0..ITERS {
                run(&vision, &mut gemm, &mut ws);
            }
            let stats = vision.take_profile(&stream).expect("profile");
            let flops: u64 = stats.iter().map(|s| s.flops).sum::<u64>() / ITERS as u64;
            println!(
                "| {label} | {patches} | {wall_ms:.2} | {:.1} | {:.2} |",
                flops as f64 / (wall_ms * 1e-3) / 1e12,
                wall_ms * 1e3 / patches as f64
            );
        }
    }
}
