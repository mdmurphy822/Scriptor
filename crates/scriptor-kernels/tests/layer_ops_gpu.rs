//! GPU checks of the layer-op kernels against the CPU references.
//! Run with `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda layer_ops -- --ignored`.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use scriptor_cuda::Device;
use scriptor_kernels::activation::ActivationKernels;
use scriptor_kernels::kv::{KvAppendKernels, append_reference, append_slots};
use scriptor_kernels::norm::NormKernels;
use scriptor_kernels::reference::rope::{
    MROPE_SECTIONS, mrope_gptj, mrope_positions_one_image, rope_vision_2d, vision_positions,
};
use scriptor_kernels::reference::{activation, norm, scatter};
use scriptor_kernels::rope::RopeKernels;
use scriptor_kernels::scatter::ScatterKernels;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn to_bf16(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7fc0;
    }
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn from_bf16(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| from_bf16(to_bf16(*x))).collect()
}

fn bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| to_bf16(*x)).collect()
}

fn close(what: &str, got: &[f32], want: &[f32], rel: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = rel * (w.abs().max(scale * 0.05));
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: got {g}, want {w}, tol {tol}"
        );
    }
}

fn setup() -> Option<(Device, Arc<CudaStream>)> {
    if std::env::var("SCRIPTOR_GPU_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let device = Device::open().expect("gpu expected");
    let stream = device.new_stream().expect("stream");
    Some((device, stream))
}

fn up(s: &Arc<CudaStream>, v: &[f32]) -> CudaSlice<u16> {
    s.clone_htod(&bits(v)).expect("upload")
}

fn up_i32(s: &Arc<CudaStream>, v: &[i32]) -> CudaSlice<i32> {
    s.clone_htod(v).expect("upload")
}

fn zeros(s: &Arc<CudaStream>, n: usize) -> CudaSlice<u16> {
    s.clone_htod(&vec![0u16; n]).expect("upload")
}

fn down(s: &Arc<CudaStream>, d: &CudaSlice<u16>) -> Vec<f32> {
    let h: Vec<u16> = s.clone_dtoh(d).expect("download");
    h.into_iter().map(from_bf16).collect()
}

#[test]
#[ignore = "needs GPU"]
fn layer_ops_rms_norm() {
    let Some((dev, s)) = setup() else { return };
    let k = NormKernels::load(dev.context()).expect("load");
    for (rows, n) in [(5, 1536), (3, 1000), (2, 1024)] {
        let mut rng = Rng::new(n as u64);
        let x = round(&rng.vec(rows * n));
        let w = round(&rng.vec(n));
        let (dx, dw) = (up(&s, &x), up(&s, &w));
        let mut out = zeros(&s, rows * n);
        k.rms_norm(&s, &dx, &dw, &mut out, rows, n, 1e-5)
            .expect("launch");
        close(
            "rms",
            &down(&s, &out),
            &norm::rms_norm(&x, &w, n, 1e-5),
            1e-2,
        );
    }
}

#[test]
#[ignore = "needs GPU"]
fn layer_ops_add_rms_norm() {
    let Some((dev, s)) = setup() else { return };
    let k = NormKernels::load(dev.context()).expect("load");
    let (rows, n) = (4, 1536);
    let mut rng = Rng::new(11);
    let x = round(&rng.vec(rows * n));
    let r = round(&rng.vec(rows * n));
    let w = round(&rng.vec(n));
    let (dx, dr, dw) = (up(&s, &x), up(&s, &r), up(&s, &w));
    let mut sum = zeros(&s, rows * n);
    let mut out = zeros(&s, rows * n);
    k.add_rms_norm(&s, &dx, &dr, &dw, &mut sum, &mut out, rows, n, 1e-5)
        .expect("launch");
    let (want_sum, _) = norm::add_rms_norm(&x, &r, &w, n, 1e-5);
    let want_sum = round(&want_sum);
    let want_out = norm::rms_norm(&want_sum, &w, n, 1e-5);
    close("sum", &down(&s, &sum), &want_sum, 1e-2);
    close("out", &down(&s, &out), &want_out, 1e-2);
}

#[test]
#[ignore = "needs GPU"]
fn layer_ops_layer_norm_gelu() {
    let Some((dev, s)) = setup() else { return };
    let k = NormKernels::load(dev.context()).expect("load");
    let (rows, n) = (6, 1536);
    let mut rng = Rng::new(21);
    let x = round(&rng.vec(rows * n));
    let w = round(&rng.vec(n));
    let b = round(&rng.vec(n));
    let (dx, dw, db) = (up(&s, &x), up(&s, &w), up(&s, &b));
    let mut out = zeros(&s, rows * n);
    k.layer_norm_gelu(&s, &dx, &dw, &db, &mut out, rows, n, 1e-5)
        .expect("launch");
    close(
        "ln_gelu",
        &down(&s, &out),
        &norm::layer_norm_gelu(&x, &w, &b, n, 1e-5),
        1e-2,
    );
}

#[test]
#[ignore = "needs GPU"]
fn layer_ops_silu_mul() {
    let Some((dev, s)) = setup() else { return };
    let k = ActivationKernels::load(dev.context()).expect("load");
    for width in [4096usize, 4608] {
        let rows = 3;
        let mut rng = Rng::new(width as u64);
        let gate_up = round(&rng.vec(rows * 2 * width));
        let dg = up(&s, &gate_up);
        let mut out = zeros(&s, rows * width);
        k.silu_mul_fused(&s, &dg, &mut out, rows, width)
            .expect("launch");
        close(
            "fused",
            &down(&s, &out),
            &activation::silu_mul_fused(&gate_up, width),
            1e-2,
        );
    }
    let count = 5001;
    let mut rng = Rng::new(5);
    let (g, u) = (round(&rng.vec(count)), round(&rng.vec(count)));
    let (dg, du) = (up(&s, &g), up(&s, &u));
    let mut out = zeros(&s, count);
    k.silu_mul(&s, &dg, &du, &mut out, count).expect("launch");
    close(
        "split",
        &down(&s, &out),
        &activation::silu_mul(&g, &u),
        1e-2,
    );
}

#[test]
#[ignore = "needs GPU"]
fn layer_ops_mrope() {
    let Some((dev, s)) = setup() else { return };
    let k = RopeKernels::load(dev.context()).expect("load");
    let (qh, kh, hd) = (16usize, 8usize, 128usize);
    let positions = mrope_positions_one_image(5, 4, 6, 3);
    let tokens = positions.len();
    let mut rng = Rng::new(31);
    let q = round(&rng.vec(tokens * qh * hd));
    let kk = round(&rng.vec(tokens * kh * hd));
    let flat: Vec<i32> = positions.iter().flatten().copied().collect();
    let (mut dq, mut dk, dp) = (up(&s, &q), up(&s, &kk), up_i32(&s, &flat));
    k.mrope(
        &s,
        &mut dq,
        &mut dk,
        &dp,
        tokens,
        qh,
        kh,
        hd,
        10_000.0,
        MROPE_SECTIONS,
    )
    .expect("launch");
    let (mut wq, mut wk) = (q.clone(), kk.clone());
    mrope_gptj(&mut wq, &positions, qh, hd, 10_000.0, MROPE_SECTIONS);
    mrope_gptj(&mut wk, &positions, kh, hd, 10_000.0, MROPE_SECTIONS);
    close("q", &down(&s, &dq), &wq, 1e-2);
    close("k", &down(&s, &dk), &wk, 1e-2);
}

#[test]
#[ignore = "needs GPU"]
fn layer_ops_vision_qk_norm_rope() {
    let Some((dev, s)) = setup() else { return };
    let k = RopeKernels::load(dev.context()).expect("load");
    let (heads, hd) = (16usize, 64usize);
    let positions = vision_positions(1, 4, 6);
    let tokens = positions.len();
    let mut rng = Rng::new(41);
    let qkv = round(&rng.vec(tokens * 3 * heads * hd));
    let qw = round(&rng.vec(hd));
    let kw = round(&rng.vec(hd));
    let flat: Vec<i32> = positions.iter().flatten().copied().collect();
    let (mut dqkv, dqw, dkw, dp) = (up(&s, &qkv), up(&s, &qw), up(&s, &kw), up_i32(&s, &flat));
    k.vision_qk_norm_rope(
        &s, &mut dqkv, &dqw, &dkw, &dp, tokens, heads, 10_000.0, 1e-5,
    )
    .expect("launch");
    let got = down(&s, &dqkv);
    let section = heads * hd;
    let pick = |src: &[f32], which: usize| -> Vec<f32> {
        (0..tokens)
            .flat_map(|t| src[(t * 3 + which) * section..][..section].to_vec())
            .collect()
    };
    for (which, w, name) in [(0, &qw, "q"), (1, &kw, "k")] {
        let mut want = round(&norm::head_rms_norm(&pick(&qkv, which), w, 1e-5));
        rope_vision_2d(&mut want, &positions, heads, hd, 10_000.0);
        close(name, &pick(&got, which), &want, 1e-2);
    }
    assert_eq!(pick(&got, 2), pick(&qkv, 2), "v section untouched");
}

#[test]
#[ignore = "needs GPU"]
fn layer_ops_gather_scatter() {
    let Some((dev, s)) = setup() else { return };
    let k = ScatterKernels::load(dev.context()).expect("load");
    let (vocab, dim) = (50usize, 1536usize);
    let mut rng = Rng::new(51);
    let table = round(&rng.vec(vocab * dim));
    let ids = [3u32, 49, 0, 17, 17];
    let ids_i32: Vec<i32> = ids.iter().map(|&i| i as i32).chain([-1, 50]).collect();
    let (dt, di) = (up(&s, &table), up_i32(&s, &ids_i32));
    let mut out = zeros(&s, ids_i32.len() * dim);
    k.embedding_gather(&s, &dt, &di, &mut out, ids_i32.len(), dim, vocab)
        .expect("launch");
    let mut want = scatter::embedding_gather(&table, dim, &ids);
    want.extend(std::iter::repeat_n(0.0, 2 * dim));
    assert_eq!(down(&s, &out), want);

    let tokens = 7usize;
    let mut embeds = round(&rng.vec(tokens * dim));
    let img = round(&rng.vec(3 * dim));
    let (mut de, dimg, dpos) = (up(&s, &embeds), up(&s, &img), up_i32(&s, &[1, 4, 5]));
    k.image_scatter(&s, &mut de, &dimg, &dpos, 3, dim)
        .expect("launch");
    scatter::image_scatter(&mut embeds, dim, &img, &[1, 4, 5]);
    assert_eq!(down(&s, &de), embeds);
}

#[test]
#[ignore = "needs GPU"]
fn layer_ops_kv_append() {
    let Some((dev, s)) = setup() else { return };
    let k = KvAppendKernels::load(dev.context()).expect("load");
    let (kv_heads, hd) = (8usize, 128usize);
    let row = kv_heads * hd;
    for block in [16usize, 32] {
        let table = [4u32, 1, 6, 0];
        // Prefill chunk crossing a block boundary, then two accepted tokens.
        let mut slots = append_slots(&table, block, 0, block + 5);
        slots.extend(append_slots(&table, block, block + 5, 2));
        slots.push(-1);
        let tokens = slots.len();
        let mut rng = Rng::new(block as u64);
        let kn = round(&rng.vec(tokens * row));
        let vn = round(&rng.vec(tokens * row));
        let cache_slots = 7 * block;
        let (mut dk, mut dv) = (zeros(&s, cache_slots * row), zeros(&s, cache_slots * row));
        let (dkn, dvn, ds) = (up(&s, &kn), up(&s, &vn), up_i32(&s, &slots));
        k.append(&s, &mut dk, &mut dv, &dkn, &dvn, &ds, tokens, kv_heads, hd)
            .expect("launch");
        let mut wk = vec![0.0f32; cache_slots * row];
        let mut wv = wk.clone();
        append_reference(&mut wk, &kn, &slots, row);
        append_reference(&mut wv, &vn, &slots, row);
        assert_eq!(down(&s, &dk), wk, "k block {block}");
        assert_eq!(down(&s, &dv), wv, "v block {block}");
    }
}
