//! GPU checks that the shared-sin/cos (A) and pure-decode no-scratch (B) variants of the fused
//! RoPE + KV-append kernel equal the reference kernel byte for byte, plus a per-variant kernel
//! cost report. Run with
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda --release --test rope_kv_variants_gpu -- --ignored --nocapture --test-threads 1`.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaSlice, CudaStream};
use scriptor_cuda::Device;
use scriptor_kernels::fused::FusedKernels;
use scriptor_kernels::kv::append_slots;
use scriptor_kernels::rope_kv::RopeKvPlan;

const VARIANTS: [(&str, RopeKvPlan); 3] = [
    (
        "A shared",
        RopeKvPlan {
            shared_trig: true,
            skip_kv_scratch: false,
        },
    ),
    (
        "B decode",
        RopeKvPlan {
            shared_trig: false,
            skip_kv_scratch: true,
        },
    ),
    (
        "A+B",
        RopeKvPlan {
            shared_trig: true,
            skip_kv_scratch: true,
        },
    ),
];

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

struct Case {
    tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    sections: [usize; 3],
    block_size: usize,
    start: usize,
    /// Token indices whose slot is negative (padding).
    padding: &'static [usize],
}

const G128: [usize; 3] = [16, 24, 24];
const G64: [usize; 3] = [8, 12, 12];

/// Positions covering ordinary decode, the first and last position of a block, large values,
/// negative values and image-axis rows whose three components differ.
fn positions(tokens: usize, block_size: usize, start: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(tokens * 3);
    for t in 0..tokens {
        let base = (start + t) as i32;
        let row = match t % 6 {
            0 => [base; 3],
            1 => [base, base / 7 + 3, base / 5 + 1],
            2 => [-(base + 1), -3, -(base % 11)],
            3 => [(block_size as i32) * 7 - 1, (block_size as i32) * 7, 0],
            4 => [40_000 + base, 17, 3 * base],
            _ => [0, 0, 0],
        };
        out.extend_from_slice(&row);
    }
    out
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            tokens: 1,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            sections: G128,
            block_size: 16,
            start: 15,
            padding: &[],
        },
        Case {
            tokens: 1,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            sections: G128,
            block_size: 16,
            start: 16,
            padding: &[0],
        },
        Case {
            tokens: 7,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 128,
            sections: G128,
            block_size: 16,
            start: 13,
            padding: &[2, 3],
        },
        Case {
            tokens: 33,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            sections: G128,
            block_size: 32,
            start: 30,
            padding: &[0, 5, 32],
        },
        Case {
            tokens: 64,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            sections: G128,
            block_size: 16,
            start: 0,
            padding: &[63],
        },
        Case {
            tokens: 5,
            q_heads: 4,
            kv_heads: 2,
            head_dim: 64,
            sections: G64,
            block_size: 16,
            start: 31,
            padding: &[1],
        },
        Case {
            tokens: 130,
            q_heads: 3,
            kv_heads: 3,
            head_dim: 128,
            sections: G128,
            block_size: 16,
            start: 1,
            padding: &[],
        },
    ]
}

/// One case run through the reference and every variant; compares the query rows, both caches
/// and, where a variant writes them, the standalone key and value rows.
fn check_case(dev: &Device, s: &Arc<CudaStream>, c: usize, case: &Case, label: &str) {
    let fused = FusedKernels::load(dev.context()).expect("load");
    let mut rng = Rng::new(900 + c as u64);
    let (qe, ke) = (case.q_heads * case.head_dim, case.kv_heads * case.head_dim);
    let qkv = rng.bits(case.tokens * (qe + 2 * ke), 3.0);
    let d_qkv = up(s, &qkv);
    let d_pos = s
        .clone_htod(&positions(case.tokens, case.block_size, case.start))
        .expect("positions");
    let table = [9u32, 2, 5, 0, 7, 3, 11, 4, 6, 1, 8, 10];
    let mut slots = append_slots(&table, case.block_size, case.start, case.tokens);
    for &p in case.padding {
        slots[p] = -1;
    }
    let d_slots = s.clone_htod(&slots).expect("slots");
    let seed = rng.bits(table.len() * case.block_size * ke, 9.0);
    let zero = |n: usize| up(s, &vec![0xFFFFu16; n]);

    let (mut rq, mut rk, mut rv) = (
        zero(case.tokens * qe),
        zero(case.tokens * ke),
        zero(case.tokens * ke),
    );
    let (mut rkc, mut rvc) = (up(s, &seed), up(s, &seed));
    fused
        .split_rope_kv_append(
            s,
            &d_qkv,
            &mut rq,
            &mut rk,
            &mut rv,
            &mut rkc,
            &mut rvc,
            &d_pos,
            &d_slots,
            case.tokens,
            case.q_heads,
            case.kv_heads,
            case.head_dim,
            10000.0,
            case.sections,
        )
        .expect("reference");
    let (wq, wk, wv, wkc, wvc) = (
        down(s, &rq),
        down(s, &rk),
        down(s, &rv),
        down(s, &rkc),
        down(s, &rvc),
    );

    for (name, plan) in VARIANTS {
        let (mut q, mut k, mut v) = (
            zero(case.tokens * qe),
            zero(case.tokens * ke),
            zero(case.tokens * ke),
        );
        let (mut kc, mut vc) = (up(s, &seed), up(s, &seed));
        let scratch = (!plan.skip_kv_scratch).then_some((&mut k, &mut v));
        fused
            .split_rope_kv_append_variant(
                s,
                plan,
                &d_qkv,
                &mut q,
                scratch,
                &mut kc,
                &mut vc,
                &d_pos,
                &d_slots,
                case.tokens,
                case.q_heads,
                case.kv_heads,
                case.head_dim,
                10000.0,
                case.sections,
            )
            .expect("variant");
        let tag = format!("{label} case {c} {name}");
        assert_eq!(down(s, &q), wq, "q {tag}");
        assert_eq!(down(s, &kc), wkc, "key cache {tag}");
        assert_eq!(down(s, &vc), wvc, "value cache {tag}");
        if !plan.skip_kv_scratch {
            assert_eq!(down(s, &k), wk, "k rows {tag}");
            assert_eq!(down(s, &v), wv, "v rows {tag}");
        } else {
            assert_eq!(
                down(s, &k),
                vec![0xFFFF; case.tokens * ke],
                "k untouched {tag}"
            );
            assert_eq!(
                down(s, &v),
                vec![0xFFFF; case.tokens * ke],
                "v untouched {tag}"
            );
        }
    }
}

#[test]
#[ignore = "needs GPU"]
fn variants_equal_reference_q_and_cache_bytes() {
    let Some((dev, s)) = setup() else { return };
    for (c, case) in cases().iter().enumerate() {
        check_case(&dev, &s, c, case, "bytes");
    }
}

#[test]
#[ignore = "needs GPU"]
fn shared_variant_matches_the_standalone_rows_with_padding_everywhere() {
    let Some((dev, s)) = setup() else { return };
    let case = Case {
        tokens: 40,
        q_heads: 16,
        kv_heads: 8,
        head_dim: 128,
        sections: G128,
        block_size: 16,
        start: 0,
        padding: &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        ],
    };
    check_case(&dev, &s, 100, &case, "padding");
}

/// Mean kernel time over `iters` launches with CUDA events.
fn time_variant(
    dev: &Device,
    s: &Arc<CudaStream>,
    fused: &FusedKernels,
    plan: Option<RopeKvPlan>,
    tokens: usize,
) -> f64 {
    let (q_heads, kv_heads, head_dim) = (16usize, 8usize, 128usize);
    let (qe, ke) = (q_heads * head_dim, kv_heads * head_dim);
    let mut rng = Rng::new(7);
    let d_qkv = up(s, &rng.bits(tokens * (qe + 2 * ke), 3.0));
    let d_pos = s
        .clone_htod(&positions(tokens, 16, 300))
        .expect("positions");
    let blocks = tokens.div_ceil(16) + 2;
    let table: Vec<u32> = (0..blocks as u32).collect();
    let d_slots = s
        .clone_htod(&append_slots(&table, 16, 0, tokens))
        .expect("slots");
    let mut q = up(s, &vec![0; tokens * qe]);
    let (mut k, mut v) = (up(s, &vec![0; tokens * ke]), up(s, &vec![0; tokens * ke]));
    let (mut kc, mut vc) = (
        up(s, &vec![0; blocks * 16 * ke]),
        up(s, &vec![0; blocks * 16 * ke]),
    );
    let mut launch = |s: &Arc<CudaStream>| match plan {
        None => fused
            .split_rope_kv_append(
                s, &d_qkv, &mut q, &mut k, &mut v, &mut kc, &mut vc, &d_pos, &d_slots, tokens,
                q_heads, kv_heads, head_dim, 10000.0, G128,
            )
            .expect("launch"),
        Some(p) => {
            let scratch = (!p.skip_kv_scratch).then_some((&mut k, &mut v));
            fused
                .split_rope_kv_append_variant(
                    s, p, &d_qkv, &mut q, scratch, &mut kc, &mut vc, &d_pos, &d_slots, tokens,
                    q_heads, kv_heads, head_dim, 10000.0, G128,
                )
                .expect("launch")
        }
    };
    for _ in 0..20 {
        launch(s);
    }
    s.synchronize().expect("sync");
    let flags = Some(CUevent_flags::CU_EVENT_DEFAULT);
    let (a, b) = (
        dev.context().new_event(flags).expect("event"),
        dev.context().new_event(flags).expect("event"),
    );
    const ITERS: usize = 200;
    a.record(s).expect("record");
    for _ in 0..ITERS {
        launch(s);
    }
    b.record(s).expect("record");
    s.synchronize().expect("sync");
    f64::from(a.elapsed_ms(&b).expect("elapsed")) * 1000.0 / ITERS as f64
}

#[test]
#[ignore = "needs GPU"]
fn kernel_cost_by_variant() {
    let Some((dev, s)) = setup() else { return };
    let fused = FusedKernels::load(dev.context()).expect("load");
    println!("| tokens | reference us | A us | B us | A+B us |");
    println!("|---|---|---|---|---|");
    for tokens in [1usize, 8, 32, 64, 128, 512, 2048] {
        let r = time_variant(&dev, &s, &fused, None, tokens);
        let t: Vec<f64> = VARIANTS
            .iter()
            .map(|(_, p)| time_variant(&dev, &s, &fused, Some(*p), tokens))
            .collect();
        println!(
            "| {tokens} | {r:.2} | {:.2} | {:.2} | {:.2} |",
            t[0], t[1], t[2]
        );
    }
}
