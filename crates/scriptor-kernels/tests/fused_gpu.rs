//! GPU checks that each fused kernel equals the unfused kernel sequence bit for bit.
//! Run with `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda fused -- --ignored`.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use scriptor_cuda::Device;
use scriptor_kernels::fused::FusedKernels;
use scriptor_kernels::kv::{KvAppendKernels, append_slots};
use scriptor_kernels::norm::NormKernels;
use scriptor_kernels::rope::RopeKernels;

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
    fn bits(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next() * scale).to_bits())
            .collect()
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

fn up(s: &Arc<CudaStream>, v: &[u16]) -> CudaSlice<u16> {
    s.clone_htod(v).expect("upload")
}

fn down(s: &Arc<CudaStream>, d: &CudaSlice<u16>) -> Vec<u16> {
    s.clone_dtoh(d).expect("download")
}

#[test]
#[ignore = "needs GPU"]
fn fused_norm_add_norm_equals_rms_then_add_rms() {
    let Some((dev, s)) = setup() else { return };
    let norm = NormKernels::load(dev.context()).expect("load");
    let fused = FusedKernels::load(dev.context()).expect("load");
    for (rows, n) in [(1, 1536), (3, 1536), (7, 1000), (5, 256), (129, 1536)] {
        let mut rng = Rng::new((rows * n) as u64);
        let x = up(&s, &rng.bits(rows * n, 4.0));
        let resid = rng.bits(rows * n, 2.0);
        let w_post = up(&s, &rng.bits(n, 1.0));
        let w_next = up(&s, &rng.bits(n, 1.0));

        let d_resid = up(&s, &resid);
        let mut y = up(&s, &vec![0; rows * n]);
        let mut sum = up(&s, &vec![0; rows * n]);
        let mut out = up(&s, &vec![0; rows * n]);
        norm.rms_norm(&s, &x, &w_post, &mut y, rows, n, 1e-5)
            .expect("rms");
        norm.add_rms_norm(&s, &d_resid, &y, &w_next, &mut sum, &mut out, rows, n, 1e-5)
            .expect("add rms");

        let mut f_sum = up(&s, &resid);
        let mut f_out = up(&s, &vec![0; rows * n]);
        let f_resid_in = up(&s, &resid);
        fused
            .norm_add_norm(
                &s,
                &x,
                &f_resid_in,
                &w_post,
                &w_next,
                &mut f_sum,
                &mut f_out,
                rows,
                n,
                1e-5,
            )
            .expect("fused");
        assert_eq!(down(&s, &f_sum), down(&s, &sum), "sum rows={rows} n={n}");
        assert_eq!(down(&s, &f_out), down(&s, &out), "out rows={rows} n={n}");
    }
}

struct RopeCase {
    tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    block_size: usize,
    start: usize,
}

#[test]
#[ignore = "needs GPU"]
fn fused_rope_kv_equals_mrope_then_append() {
    let Some((dev, s)) = setup() else { return };
    let rope = RopeKernels::load(dev.context()).expect("load");
    let append = KvAppendKernels::load(dev.context()).expect("load");
    let fused = FusedKernels::load(dev.context()).expect("load");
    let head_dim = 128;
    let sections = [16, 24, 24];
    let cases = [
        RopeCase {
            tokens: 1,
            q_heads: 16,
            kv_heads: 8,
            block_size: 16,
            start: 15,
        },
        RopeCase {
            tokens: 7,
            q_heads: 2,
            kv_heads: 1,
            block_size: 16,
            start: 13,
        },
        RopeCase {
            tokens: 33,
            q_heads: 16,
            kv_heads: 8,
            block_size: 32,
            start: 30,
        },
        RopeCase {
            tokens: 5,
            q_heads: 4,
            kv_heads: 2,
            block_size: 16,
            start: 0,
        },
    ];
    for (c, case) in cases.iter().enumerate() {
        let mut rng = Rng::new(100 + c as u64);
        let (qe, ke) = (case.q_heads * head_dim, case.kv_heads * head_dim);
        let q = rng.bits(case.tokens * qe, 3.0);
        let k = rng.bits(case.tokens * ke, 3.0);
        let v = up(&s, &rng.bits(case.tokens * ke, 3.0));
        let positions: Vec<i32> = (0..case.tokens * 3)
            .map(|i| (i as i32 * 37 + 5) % 4001)
            .collect();
        let d_pos = s.clone_htod(&positions).expect("positions");
        let table = [9u32, 2, 5, 0, 7, 3];
        let mut slots = append_slots(&table, case.block_size, case.start, case.tokens);
        if case.tokens > 4 {
            slots[2] = -1;
        }
        let d_slots = s.clone_htod(&slots).expect("slots");
        let cache_elems = table.len() * case.block_size * ke;
        let seed = rng.bits(cache_elems, 9.0);

        let (mut uq, mut uk) = (up(&s, &q), up(&s, &k));
        let (mut ukc, mut uvc) = (up(&s, &seed), up(&s, &seed));
        rope.mrope(
            &s,
            &mut uq,
            &mut uk,
            &d_pos,
            case.tokens,
            case.q_heads,
            case.kv_heads,
            head_dim,
            10000.0,
            sections,
        )
        .expect("mrope");
        append
            .append(
                &s,
                &mut ukc,
                &mut uvc,
                &uk,
                &v,
                &d_slots,
                case.tokens,
                case.kv_heads,
                head_dim,
            )
            .expect("append");

        let (mut fq, mut fk) = (up(&s, &q), up(&s, &k));
        let (mut fkc, mut fvc) = (up(&s, &seed), up(&s, &seed));
        fused
            .rope_kv_append(
                &s,
                &mut fq,
                &mut fk,
                &v,
                &mut fkc,
                &mut fvc,
                &d_pos,
                &d_slots,
                case.tokens,
                case.q_heads,
                case.kv_heads,
                head_dim,
                10000.0,
                sections,
            )
            .expect("fused");

        assert_eq!(down(&s, &fq), down(&s, &uq), "q case {c}");
        assert_eq!(down(&s, &fk), down(&s, &uk), "k case {c}");
        assert_eq!(down(&s, &fkc), down(&s, &ukc), "key cache case {c}");
        assert_eq!(down(&s, &fvc), down(&s, &uvc), "value cache case {c}");
    }
}

#[test]
#[ignore = "needs GPU"]
fn split_qkv_and_split_rope_kv_equal_separate_projections() {
    let Some((dev, s)) = setup() else { return };
    let rope = RopeKernels::load(dev.context()).expect("load");
    let append = KvAppendKernels::load(dev.context()).expect("load");
    let fused = FusedKernels::load(dev.context()).expect("load");
    let head_dim = 128;
    let sections = [16, 24, 24];
    for (c, &(tokens, q_heads, kv_heads, start)) in [
        (1usize, 16usize, 8usize, 15usize),
        (7, 2, 1, 13),
        (33, 16, 8, 30),
    ]
    .iter()
    .enumerate()
    {
        let mut rng = Rng::new(300 + c as u64);
        let (qe, ke) = (q_heads * head_dim, kv_heads * head_dim);
        let q = rng.bits(tokens * qe, 3.0);
        let k = rng.bits(tokens * ke, 3.0);
        let v = rng.bits(tokens * ke, 3.0);
        let mut qkv = Vec::new();
        for t in 0..tokens {
            qkv.extend_from_slice(&q[t * qe..(t + 1) * qe]);
            qkv.extend_from_slice(&k[t * ke..(t + 1) * ke]);
            qkv.extend_from_slice(&v[t * ke..(t + 1) * ke]);
        }
        let d_qkv = up(&s, &qkv);
        let zero = |n: usize| up(&s, &vec![0u16; n]);

        let (mut sq, mut sk, mut sv) = (zero(q.len()), zero(k.len()), zero(v.len()));
        fused
            .split_qkv(&s, &d_qkv, &mut sq, &mut sk, &mut sv, tokens, qe, ke)
            .expect("split");
        assert_eq!(down(&s, &sq), q, "split q case {c}");
        assert_eq!(down(&s, &sk), k, "split k case {c}");
        assert_eq!(down(&s, &sv), v, "split v case {c}");

        let positions: Vec<i32> = (0..tokens * 3)
            .map(|i| (i as i32 * 37 + 5) % 4001)
            .collect();
        let d_pos = s.clone_htod(&positions).expect("positions");
        let table = [9u32, 2, 5, 0, 7, 3];
        let mut slots = append_slots(&table, 16, start, tokens);
        if tokens > 4 {
            slots[2] = -1;
        }
        let d_slots = s.clone_htod(&slots).expect("slots");
        let seed = rng.bits(table.len() * 16 * ke, 9.0);

        let (mut uq, mut uk, uv) = (up(&s, &q), up(&s, &k), up(&s, &v));
        let (mut ukc, mut uvc) = (up(&s, &seed), up(&s, &seed));
        rope.mrope(
            &s, &mut uq, &mut uk, &d_pos, tokens, q_heads, kv_heads, head_dim, 10000.0, sections,
        )
        .expect("mrope");
        append
            .append(
                &s, &mut ukc, &mut uvc, &uk, &uv, &d_slots, tokens, kv_heads, head_dim,
            )
            .expect("append");

        let (mut fq, mut fk, mut fv) = (zero(q.len()), zero(k.len()), zero(v.len()));
        let (mut fkc, mut fvc) = (up(&s, &seed), up(&s, &seed));
        fused
            .split_rope_kv_append(
                &s, &d_qkv, &mut fq, &mut fk, &mut fv, &mut fkc, &mut fvc, &d_pos, &d_slots,
                tokens, q_heads, kv_heads, head_dim, 10000.0, sections,
            )
            .expect("fused");
        assert_eq!(down(&s, &fq), down(&s, &uq), "q case {c}");
        assert_eq!(down(&s, &fk), down(&s, &uk), "k case {c}");
        assert_eq!(down(&s, &fv), v, "v case {c}");
        assert_eq!(down(&s, &fkc), down(&s, &ukc), "key cache case {c}");
        assert_eq!(down(&s, &fvc), down(&s, &uvc), "value cache case {c}");
    }
}
