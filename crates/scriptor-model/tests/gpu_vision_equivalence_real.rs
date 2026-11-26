//! Fused against unfused vision tower on the real checkpoint over synthetic text images. Set
//! `SCRIPTOR_SNAPSHOT_DIR` to the checkpoint directory (or leave it unset to use the Hugging Face
//! cache) and run
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda --release --test gpu_vision_equivalence_real -- --ignored --nocapture`.
#![cfg(feature = "cuda")]

mod common;

use scriptor_cuda::Device;
use scriptor_kernels::cublaslt::{GemmEngine, GemmOptions};
use scriptor_kernels::heuristics::TuningTable;
use scriptor_model::gpu::{Fusion, GpuVision, VisionWorkspace, bf16_to_f32};
use scriptor_model::reference::TensorSource;
use scriptor_model::{Merger, VisionTower};

use common::{open_checkpoint, shared_vision, synthetic_patches};

/// Smallest cosine similarity per image accepted between the fused and unfused embeddings.
const MIN_COSINE: f32 = 0.998;
/// Largest `|fused - unfused|` accepted, relative to the largest unfused magnitude of the image.
const MAX_RELATIVE_DELTA: f32 = 0.40;

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn fused_vision_matches_unfused_on_real_weights() {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let (spec, files) = open_checkpoint();
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let mut arena = dev.arena(512 << 20, &stream).expect("arena");
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
    let grids = [
        (1usize, 4usize, 100usize),
        (1, 56, 84),
        (1, 8, 60),
        (1, 20, 36),
        (1, 120, 84),
    ];
    let total: usize = grids.iter().map(|&(t, h, w)| t * h * w).sum();
    let mut ws = VisionWorkspace::new(&stream, vision.params(), total).expect("workspace");
    let mut bits = Vec::new();
    for (i, &(t, h, w)) in grids.iter().enumerate() {
        bits.extend(synthetic_patches(t, h, w, 77 + i as u64));
    }
    let input = stream.clone_htod(&bits).expect("patches");

    let mut results: Vec<Vec<Vec<f32>>> = Vec::new();
    for fusion in [Fusion::Unfused, Fusion::Fused] {
        vision.set_fusion(fusion);
        let mut outs: Vec<_> = grids
            .iter()
            .map(|&(t, h, w)| {
                stream
                    .alloc_zeros::<u16>(t * h * w / 4 * out_dim)
                    .expect("out")
            })
            .collect();
        let mut refs: Vec<_> = outs.iter_mut().collect();
        vision
            .run(&stream, &mut gemm, &mut ws, &input, &grids, &mut refs)
            .expect("run");
        results.push(
            outs.iter()
                .map(|o| bf16_to_f32(&stream.clone_dtoh(o).expect("download")))
                .collect(),
        );
    }
    for (i, grid) in grids.iter().enumerate() {
        let (a, b) = (&results[0][i], &results[1][i]);
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let cos = (dot / (na * nb)) as f32;
        let max = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        let scale = a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        println!(
            "grid {grid:?}: cosine {cos:.6}, max |delta| {max:.4} ({:.2}% of max |value| {scale:.2})",
            100.0 * max / scale
        );
        assert!(cos >= MIN_COSINE, "grid {grid:?}: cosine {cos}");
        assert!(
            max <= MAX_RELATIVE_DELTA * scale,
            "grid {grid:?}: max |delta| {max} of {scale}"
        );
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb)) as f32
}

/// Both kernel sets against the f32 reference tower on one small image: the fused set must be
/// as close to the reference as the unfused set, within a cosine margin of 5e-4.
#[test]
#[ignore = "needs GPU and the checkpoint"]
fn fused_and_unfused_vision_track_the_f32_reference() {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let (spec, files) = open_checkpoint();
    let source = TensorSource::new(&files);
    let tower = VisionTower::load(&source, &spec.vision).expect("tower");
    let merger = Merger::load(&source, &spec.vision).expect("merger");
    let grid = (1usize, 4usize, 24usize);
    let bits = synthetic_patches(grid.0, grid.1, grid.2, 5);
    let patches = bf16_to_f32(&bits);
    let want = merger.forward(&tower.forward(&patches, &[grid]).expect("reference"));

    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let mut arena = dev.arena(256 << 20, &stream).expect("arena");
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
    let mut ws = VisionWorkspace::new(&stream, vision.params(), 96).expect("workspace");
    let input = stream.clone_htod(&bits).expect("patches");
    let mut cosines = Vec::new();
    for fusion in [Fusion::Unfused, Fusion::Fused] {
        vision.set_fusion(fusion);
        let mut out = stream
            .alloc_zeros::<u16>(grid.0 * grid.1 * grid.2 / 4 * out_dim)
            .expect("out");
        vision
            .run(
                &stream,
                &mut gemm,
                &mut ws,
                &input,
                &[grid],
                &mut [&mut out],
            )
            .expect("run");
        let got = bf16_to_f32(&stream.clone_dtoh(&out).expect("download"));
        let cos = cosine(&got, &want);
        println!("{fusion:?} against the f32 reference: cosine {cos:.6}");
        cosines.push(cos);
    }
    assert!(cosines[1] >= cosines[0] - 5e-4, "{cosines:?}");
}

/// The SiLU-fused gate/up projection against the vendor GEMM plus `silu_mul` in the fused
/// kernel set: embedding differences on real weights.
#[test]
#[ignore = "needs GPU and the checkpoint"]
fn gate_up_silu_fusion_tracks_vendor_gemm_on_real_weights() {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let (spec, files) = open_checkpoint();
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let mut arena = dev.arena(512 << 20, &stream).expect("arena");
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
    let grids = [
        (1usize, 4usize, 100usize),
        (1, 56, 84),
        (1, 8, 60),
        (1, 120, 84),
    ];
    let total: usize = grids.iter().map(|&(t, h, w)| t * h * w).sum();
    let mut ws = VisionWorkspace::new(&stream, vision.params(), total).expect("workspace");
    let mut bits = Vec::new();
    for (i, &(t, h, w)) in grids.iter().enumerate() {
        bits.extend(synthetic_patches(t, h, w, 91 + i as u64));
    }
    let input = stream.clone_htod(&bits).expect("patches");
    vision.set_fusion(Fusion::Fused);
    let mut results: Vec<Vec<Vec<f32>>> = Vec::new();
    for on in [false, true] {
        vision.set_gate_up_silu(on);
        let mut outs: Vec<_> = grids
            .iter()
            .map(|&(t, h, w)| {
                stream
                    .alloc_zeros::<u16>(t * h * w / 4 * out_dim)
                    .expect("out")
            })
            .collect();
        let mut refs: Vec<_> = outs.iter_mut().collect();
        vision
            .run(&stream, &mut gemm, &mut ws, &input, &grids, &mut refs)
            .expect("run");
        results.push(
            outs.iter()
                .map(|o| bf16_to_f32(&stream.clone_dtoh(o).expect("download")))
                .collect(),
        );
    }
    for (i, grid) in grids.iter().enumerate() {
        let (a, b) = (&results[0][i], &results[1][i]);
        let max = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        let mean = a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from((x - y).abs()))
            .sum::<f64>()
            / a.len() as f64;
        let scale = a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        println!(
            "grid {grid:?}: max |delta| {max:.5}, mean |delta| {mean:.6}, max |value| {scale:.2}"
        );
        assert!(
            max <= 0.01 * scale,
            "grid {grid:?}: max |delta| {max} of {scale}"
        );
    }
}
