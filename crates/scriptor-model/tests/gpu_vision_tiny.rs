//! GPU vision executor against the CPU reference on a synthetic tower with 64-wide heads.
//! Run with `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda gpu_vision -- --ignored`.
#![cfg(feature = "cuda")]

use scriptor_cuda::Device;
use scriptor_kernels::cublaslt::{GemmEngine, GemmOptions};
use scriptor_kernels::heuristics::TuningTable;
use scriptor_kernels::reference::conv::PATCH_ELEMS;
use scriptor_model::gpu::{
    Fusion, GpuVision, GpuVisionParams, VisionWorkspace, bf16_bits, bf16_to_f32,
};
use scriptor_model::reference::vision::VisionBlock;
use scriptor_model::{Linear, Merger, VisionTower};

const HIDDEN: usize = 128;
const HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const INTER: usize = 192;
const OUT: usize = 96;
const MID: usize = 160;
const DEPTH: usize = 2;
const TOLERANCE: f32 = 1e-2;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// Values rounded to BF16 so the reference and the device read identical weights.
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        let raw: Vec<f32> = (0..n).map(|_| self.next() * scale).collect();
        bf16_to_f32(&bf16_bits(&raw))
    }

    fn norm(&mut self, n: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..n).map(|_| 1.0 + 0.1 * self.next()).collect();
        bf16_to_f32(&bf16_bits(&raw))
    }

    fn linear(&mut self, out_dim: usize, in_dim: usize, bias: bool) -> Linear {
        let scale = (in_dim as f32).powf(-0.5);
        Linear {
            weight: self.vec(out_dim * in_dim, scale),
            bias: bias.then(|| self.vec(out_dim, 0.1)),
            out_dim,
            in_dim,
        }
    }
}

fn tower(rng: &mut Rng) -> VisionTower {
    let blocks = (0..DEPTH)
        .map(|_| VisionBlock {
            norm1: rng.norm(HIDDEN),
            norm2: rng.norm(HIDDEN),
            qkv: rng.linear(3 * HIDDEN, HIDDEN, true),
            q_norm: rng.norm(HEAD_DIM),
            k_norm: rng.norm(HEAD_DIM),
            proj: rng.linear(HIDDEN, HIDDEN, true),
            gate: rng.linear(INTER, HIDDEN, true),
            up: rng.linear(INTER, HIDDEN, true),
            down: rng.linear(HIDDEN, INTER, true),
        })
        .collect();
    VisionTower {
        patch_embed: rng.linear(HIDDEN, PATCH_ELEMS, true),
        blocks,
        post_layernorm: rng.norm(HIDDEN),
        hidden: HIDDEN,
        heads: HEADS,
        head_dim: HEAD_DIM,
        eps: 1e-5,
        rope_theta: 10000.0,
    }
}

fn merger(rng: &mut Rng) -> Merger {
    Merger {
        downsample_weight: rng.vec(OUT * HIDDEN * 4, (HIDDEN as f32 * 4.0).powf(-0.5)),
        downsample_bias: rng.vec(OUT, 0.1),
        proj: rng.linear(OUT, OUT, false),
        norm_weight: rng.norm(OUT),
        norm_bias: rng.vec(OUT, 0.1),
        gate: rng.linear(MID, OUT, false),
        up: rng.linear(MID, OUT, false),
        down: rng.linear(OUT, MID, false),
        in_dim: HIDDEN,
        out_dim: OUT,
    }
}

#[test]
#[ignore = "needs GPU"]
fn gpu_vision_tiny_batch_matches_reference() {
    if std::env::var("SCRIPTOR_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("SCRIPTOR_GPU_TESTS is not 1; skipping");
        return;
    }
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let (tower, merger) = (tower(&mut rng), merger(&mut rng));
    let grids = [(1usize, 4usize, 6usize), (1, 2, 2), (2, 6, 4)];
    let total: usize = grids.iter().map(|&(t, h, w)| t * h * w).sum();
    let patches = rng.vec(total * PATCH_ELEMS, 1.0);

    let hidden_states = tower.forward(&patches, &grids).expect("reference tower");
    let want = merger.forward(&hidden_states);

    let device = Device::open().expect("gpu");
    let stream = device.new_stream().expect("stream");
    let mut arena = device.arena(128 << 20, &stream).expect("arena");
    let mut gemm = GemmEngine::new(
        device.context(),
        &stream,
        &mut arena,
        TuningTable::default(),
        GemmOptions::default(),
    )
    .expect("gemm engine");
    let params = GpuVisionParams::from_reference(&stream, &tower, &merger).expect("weights");
    let vision = GpuVision::new(device.context(), params).expect("vision");
    let mut ws = VisionWorkspace::new(&stream, vision.params(), total + 16).expect("workspace");
    let input = stream.clone_htod(&bf16_bits(&patches)).expect("patches");
    let mut outs: Vec<_> = grids
        .iter()
        .map(|&(t, h, w)| {
            stream
                .alloc_zeros::<u16>(t * h * w / 4 * OUT)
                .expect("output")
        })
        .collect();
    let mut refs: Vec<_> = outs.iter_mut().collect();
    let stats = vision
        .run(&stream, &mut gemm, &mut ws, &input, &grids, &mut refs)
        .expect("run");
    assert_eq!(stats.patches, total);
    assert_eq!(stats.frames, 4);
    assert_eq!(stats.tokens * 4, total);

    let mut got = Vec::new();
    for out in &outs {
        got.extend(bf16_to_f32(&stream.clone_dtoh(out).expect("download")));
    }
    assert_eq!(got.len(), want.len());
    let diff = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(diff < TOLERANCE, "max |delta| {diff} exceeds {TOLERANCE}");
}

/// Cosine similarity and largest absolute difference of two vectors.
fn compare(a: &[f32], b: &[f32]) -> (f32, f32) {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let max = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    ((dot / (na * nb)) as f32, max)
}

/// The fused kernel set (strided attention, bias folded into the residual join) against the
/// unfused set on the tiny tower: cosine at least 0.9999 and every value within 4e-3.
#[test]
#[ignore = "needs GPU"]
fn gpu_vision_tiny_fused_matches_unfused() {
    if std::env::var("SCRIPTOR_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("SCRIPTOR_GPU_TESTS is not 1; skipping");
        return;
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (tower, merger) = (tower(&mut rng), merger(&mut rng));
    let grids = [(1usize, 4usize, 6usize), (1, 2, 2), (2, 6, 4), (1, 8, 10)];
    let total: usize = grids.iter().map(|&(t, h, w)| t * h * w).sum();
    let patches = rng.vec(total * PATCH_ELEMS, 1.0);

    let device = Device::open().expect("gpu");
    let stream = device.new_stream().expect("stream");
    let mut arena = device.arena(128 << 20, &stream).expect("arena");
    let mut gemm = GemmEngine::new(
        device.context(),
        &stream,
        &mut arena,
        TuningTable::default(),
        GemmOptions::default(),
    )
    .expect("gemm engine");
    let params = GpuVisionParams::from_reference(&stream, &tower, &merger).expect("weights");
    let mut vision = GpuVision::new(device.context(), params).expect("vision");
    let mut ws = VisionWorkspace::new(&stream, vision.params(), total + 16).expect("workspace");
    let input = stream.clone_htod(&bf16_bits(&patches)).expect("patches");

    let mut results = Vec::new();
    for fusion in [Fusion::Unfused, Fusion::Fused] {
        vision.set_fusion(fusion);
        let mut outs: Vec<_> = grids
            .iter()
            .map(|&(t, h, w)| {
                stream
                    .alloc_zeros::<u16>(t * h * w / 4 * OUT)
                    .expect("output")
            })
            .collect();
        let mut refs: Vec<_> = outs.iter_mut().collect();
        vision
            .run(&stream, &mut gemm, &mut ws, &input, &grids, &mut refs)
            .expect("run");
        let mut got = Vec::new();
        for out in &outs {
            got.extend(bf16_to_f32(&stream.clone_dtoh(out).expect("download")));
        }
        results.push(got);
    }
    let (cos, max) = compare(&results[0], &results[1]);
    eprintln!("tiny fused vs unfused: cosine {cos}, max |delta| {max}");
    assert!(cos >= 0.9999, "cosine {cos}");
    assert!(max <= 4e-3, "max |delta| {max}");
}
