//! GPU checks of the vision attention over packed `qkv` and the bias + residual + RMSNorm join
//! against the existing varlen attention and norm kernels.
//! Run with `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda vision -- --ignored`.
#![cfg(feature = "cuda")]

use scriptor_cuda::Device;
use scriptor_kernels::attention::{VarlenAttention, VarlenGeometry};
use scriptor_kernels::norm::NormKernels;
use scriptor_kernels::vision::VisionKernels;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn bits(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next() * scale).to_bits())
            .collect()
    }
}

fn f(bits: u16) -> f32 {
    half::bf16::from_bits(bits).to_f32()
}

fn gpu() -> Option<Device> {
    (std::env::var("SCRIPTOR_GPU_TESTS").as_deref() == Ok("1"))
        .then(|| Device::open().expect("gpu"))
}

/// Runs both attention kernels on the same packed `qkv` and returns the largest absolute
/// difference and the number of differing outputs.
fn compare_attention(lengths: &[usize], heads: usize, scale_in: f32) -> (f32, usize) {
    let dev = gpu().expect("SCRIPTOR_GPU_TESTS");
    let stream = dev.new_stream().expect("stream");
    let old = VarlenAttention::load(dev.context()).expect("varlen");
    let new = VisionKernels::load(dev.context()).expect("vision");
    let mut cu = vec![0i32];
    for l in lengths {
        cu.push(cu.last().unwrap() + *l as i32);
    }
    let tokens = *cu.last().unwrap() as usize;
    let width = heads * 64;
    let mut rng = Rng(0x2545_F491_4F6C_DD1D ^ tokens as u64);
    let qkv = rng.bits(tokens * 3 * width, scale_in);
    let (mut q, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for row in qkv.chunks_exact(3 * width) {
        q.extend_from_slice(&row[..width]);
        k.extend_from_slice(&row[width..2 * width]);
        v.extend_from_slice(&row[2 * width..]);
    }
    let d_cu = stream.clone_htod(&cu).expect("cu");
    let d_qkv = stream.clone_htod(&qkv).expect("qkv");
    let (dq, dk, dv) = (
        stream.clone_htod(&q).unwrap(),
        stream.clone_htod(&k).unwrap(),
        stream.clone_htod(&v).unwrap(),
    );
    let mut want = stream.alloc_zeros::<u16>(tokens * width).unwrap();
    let mut got = stream.alloc_zeros::<u16>(tokens * width).unwrap();
    let max_len = *lengths.iter().max().unwrap();
    let scale = 0.125f32;
    old.run(
        &stream,
        &VarlenGeometry::vision(heads),
        &dq,
        &dk,
        &dv,
        &mut want,
        &d_cu,
        lengths.len(),
        max_len,
        scale,
    )
    .expect("old");
    new.attention(
        &stream,
        &d_qkv,
        &mut got,
        &d_cu,
        lengths.len(),
        max_len,
        heads,
        scale,
    )
    .expect("new");
    stream.synchronize().unwrap();
    let want = stream.clone_dtoh(&want).unwrap();
    let got = stream.clone_dtoh(&got).unwrap();
    let mut max = 0.0f32;
    let mut differing = 0usize;
    for (a, b) in want.iter().zip(&got) {
        if a != b {
            differing += 1;
        }
        max = max.max((f(*a) - f(*b)).abs());
    }
    (max, differing)
}

#[test]
#[ignore = "needs GPU"]
fn vision_attention_equals_varlen_on_split_inputs() {
    if gpu().is_none() {
        return;
    }
    for (lens, heads) in [
        (vec![1usize], 2usize),
        (vec![17], 16),
        (vec![63, 64, 65], 16),
        (vec![127, 128, 129, 200], 16),
        (vec![920, 5, 3000], 16),
        (vec![4096], 16),
    ] {
        let (max, differing) = compare_attention(&lens, heads, 2.0);
        eprintln!("lengths {lens:?} heads {heads}: max |delta| {max}, {differing} differing");
        assert!(max <= 1.0 / 128.0, "{lens:?}: max |delta| {max}");
    }
}

#[test]
#[ignore = "needs GPU"]
fn vision_join_matches_add_rms_norm() {
    let Some(dev) = gpu() else { return };
    let stream = dev.new_stream().expect("stream");
    let norm = NormKernels::load(dev.context()).expect("norm");
    let vision = VisionKernels::load(dev.context()).expect("vision");
    for (rows, n) in [
        (1usize, 128usize),
        (7, 1024),
        (300, 1024),
        (5, 4096),
        (3, 8),
    ] {
        let mut rng = Rng(0x9e37_79b9 ^ (rows * n) as u64);
        let x = rng.bits(rows * n, 3.0);
        let bias = rng.bits(n, 0.5);
        let resid = rng.bits(rows * n, 2.0);
        let w = rng.bits(n, 1.0);
        let (dx, db, dr, dw) = (
            stream.clone_htod(&x).unwrap(),
            stream.clone_htod(&bias).unwrap(),
            stream.clone_htod(&resid).unwrap(),
            stream.clone_htod(&w).unwrap(),
        );
        // Reference: bias added with its own bf16 rounding, then the existing join.
        let biased: Vec<u16> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| half::bf16::from_f32(f(v) + f(bias[i % n])).to_bits())
            .collect();
        let dbiased = stream.clone_htod(&biased).unwrap();
        let mut want_sum = stream.alloc_zeros::<u16>(rows * n).unwrap();
        let mut want_out = stream.alloc_zeros::<u16>(rows * n).unwrap();
        norm.add_rms_norm(
            &stream,
            &dbiased,
            &dr,
            &dw,
            &mut want_sum,
            &mut want_out,
            rows,
            n,
            1e-5,
        )
        .unwrap();
        let mut got_sum = stream.alloc_zeros::<u16>(rows * n).unwrap();
        let mut got_out = stream.alloc_zeros::<u16>(rows * n).unwrap();
        vision
            .bias_add_rms_norm(
                &stream,
                &dx,
                Some(&db),
                &dr,
                &dw,
                &mut got_sum,
                &mut got_out,
                rows,
                n,
                1e-5,
            )
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            stream.clone_dtoh(&want_sum).unwrap(),
            stream.clone_dtoh(&got_sum).unwrap(),
            "sum rows {rows} n {n}"
        );
        let (a, b) = (
            stream.clone_dtoh(&want_out).unwrap(),
            stream.clone_dtoh(&got_out).unwrap(),
        );
        let max = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (f(*p) - f(*q)).abs() / (f(*p).abs() + 1e-3))
            .fold(0.0f32, f32::max);
        eprintln!("join rows {rows} n {n}: max relative delta {max}");
        assert!(max <= 1.0 / 64.0, "rows {rows} n {n}: {max}");
    }
}

/// Attention over a raw-q `qkv` with the q norm + RoPE folded into the q load must match the
/// standalone q/k norm + RoPE kernel followed by the stored-q attention.
#[test]
#[ignore = "needs GPU"]
fn vision_attention_fold_q_matches_qk_norm_rope_then_attention() {
    let Some(dev) = gpu() else { return };
    let stream = dev.new_stream().expect("stream");
    let rope = scriptor_kernels::rope::RopeKernels::load(dev.context()).expect("rope");
    let vision = VisionKernels::load(dev.context()).expect("vision");
    let heads = 16usize;
    for grids in [
        vec![(1usize, 2usize, 2usize)],
        vec![(1, 6, 10), (1, 4, 4)],
        vec![(1, 12, 14), (2, 8, 10), (1, 2, 2)],
        vec![(1, 40, 50)],
    ] {
        let mut positions: Vec<i32> = Vec::new();
        let mut cu = vec![0i32];
        for &(t, h, w) in &grids {
            for _ in 0..t {
                for r in 0..h {
                    for c in 0..w {
                        positions.extend([r as i32, c as i32]);
                    }
                }
                cu.push(cu.last().unwrap() + (h * w) as i32);
            }
        }
        let tokens = positions.len() / 2;
        let lengths: Vec<i32> = cu.windows(2).map(|w| w[1] - w[0]).collect();
        let max_len = *lengths.iter().max().unwrap() as usize;
        let mut rng = Rng(0xabcd_ef01 ^ tokens as u64);
        let qkv = rng.bits(tokens * 3 * heads * 64, 2.0);
        let q_w = rng.bits(64, 1.0);
        let k_w = rng.bits(64, 1.0);
        let (d_cu, d_pos, d_qw, d_kw) = (
            stream.clone_htod(&cu).unwrap(),
            stream.clone_htod(&positions).unwrap(),
            stream.clone_htod(&q_w).unwrap(),
            stream.clone_htod(&k_w).unwrap(),
        );
        let mut d_old = stream.clone_htod(&qkv).unwrap();
        let mut d_new = stream.clone_htod(&qkv).unwrap();
        let (theta, eps, scale) = (10_000.0f32, 1e-5f32, 0.125f32);
        rope.vision_qk_norm_rope(
            &stream, &mut d_old, &d_qw, &d_kw, &d_pos, tokens, heads, theta, eps,
        )
        .unwrap();
        rope.vision_k_norm_rope(
            &stream, &mut d_new, &d_qw, &d_kw, &d_pos, tokens, heads, theta, eps,
        )
        .unwrap();
        let mut want = stream.alloc_zeros::<u16>(tokens * heads * 64).unwrap();
        let mut got = stream.alloc_zeros::<u16>(tokens * heads * 64).unwrap();
        vision
            .attention(
                &stream,
                &d_old,
                &mut want,
                &d_cu,
                lengths.len(),
                max_len,
                heads,
                scale,
            )
            .unwrap();
        vision
            .attention_fold_q(
                &stream,
                &d_new,
                &mut got,
                &d_cu,
                lengths.len(),
                max_len,
                heads,
                scale,
                &d_qw,
                &d_pos,
                theta,
                eps,
            )
            .unwrap();
        stream.synchronize().unwrap();
        let (a, b) = (
            stream.clone_dtoh(&want).unwrap(),
            stream.clone_dtoh(&got).unwrap(),
        );
        let differing = a.iter().zip(&b).filter(|(p, q)| p != q).count();
        let max = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (f(*p) - f(*q)).abs())
            .fold(0.0f32, f32::max);
        eprintln!("fold_q grids {grids:?}: max |delta| {max}, {differing} differing");
        assert_eq!(differing, 0, "{grids:?}: max |delta| {max}");
    }
}
