//! Device attention against f64 references.
//! Run with `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda attention -- --ignored`.
#![cfg(feature = "cuda")]

use scriptor_cuda::Device;
use scriptor_kernels::attention::{
    DecodeGeometry, DecodeWorkspace, PagedDecodeAttention, SeqTables, VarlenAttention,
    VarlenGeometry,
};
use scriptor_kernels::reference::attention::{PagedKv, paged_decode_attention};

const SM_COUNT: usize = 48;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_value(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn gpu_enabled() -> bool {
    std::env::var("SCRIPTOR_GPU_TESTS").as_deref() == Ok("1")
}

fn random_bf16(rng: &mut Rng, n: usize, amp: f32) -> Vec<u16> {
    (0..n).map(|_| bf16_bits(rng.next() * amp)).collect()
}

/// Attention output of one query row, computed in f64 over bf16-rounded inputs.
#[allow(clippy::too_many_arguments)]
fn reference_row(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    span: (usize, usize),
    row: usize,
    head: usize,
    geo: &VarlenGeometry,
    scale: f32,
) -> Vec<f32> {
    let (start, end) = span;
    let (qh, kvh, hd) = (geo.q_heads, geo.kv_heads, geo.head_dim);
    let kv_head = head / (qh / kvh);
    let visible = if geo.causal {
        row - start + 1
    } else {
        end - start
    };
    let qrow = &q[(row * qh + head) * hd..][..hd];
    let scores: Vec<f64> = (0..visible)
        .map(|j| {
            let krow = &k[((start + j) * kvh + kv_head) * hd..][..hd];
            qrow.iter()
                .zip(krow)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>()
                * f64::from(scale)
        })
        .collect();
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let denom: f64 = exps.iter().sum();
    let mut out = vec![0.0f64; hd];
    for (j, e) in exps.iter().enumerate() {
        let vrow = &v[((start + j) * kvh + kv_head) * hd..][..hd];
        for (o, x) in out.iter_mut().zip(vrow) {
            *o += e / denom * f64::from(*x);
        }
    }
    out.into_iter().map(|x| x as f32).collect()
}

fn run_varlen(lengths: &[usize], geo: VarlenGeometry) {
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let kernels = VarlenAttention::load(dev.context()).expect("kernels");
    let mut cu = vec![0usize];
    for l in lengths {
        cu.push(cu.last().unwrap() + l);
    }
    let tokens = *cu.last().unwrap();
    let hd = geo.head_dim;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ tokens as u64);
    let qh = random_bf16(&mut rng, tokens * geo.q_heads * hd, 1.0);
    let kh = random_bf16(&mut rng, tokens * geo.kv_heads * hd, 1.0);
    let vh = random_bf16(&mut rng, tokens * geo.kv_heads * hd, 2.0);
    let d_cu = stream
        .clone_htod(&cu.iter().map(|&c| c as i32).collect::<Vec<_>>())
        .unwrap();
    let (dq, dk, dv) = (
        stream.clone_htod(&qh).unwrap(),
        stream.clone_htod(&kh).unwrap(),
        stream.clone_htod(&vh).unwrap(),
    );
    let mut d_out = stream.alloc_zeros::<u16>(qh.len()).unwrap();
    let scale = (hd as f32).powf(-0.5);
    let max_len = *lengths.iter().max().unwrap();
    kernels
        .run(
            &stream,
            &geo,
            &dq,
            &dk,
            &dv,
            &mut d_out,
            &d_cu,
            lengths.len(),
            max_len,
            scale,
        )
        .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<u16> = stream.clone_dtoh(&d_out).unwrap();

    let to_f32 = |x: &[u16]| x.iter().map(|&b| bf16_value(b)).collect::<Vec<f32>>();
    let (qf, kf, vf) = (to_f32(&qh), to_f32(&kh), to_f32(&vh));
    for (s, span) in cu.windows(2).enumerate() {
        let (start, end) = (span[0], span[1]);
        let len = end - start;
        // First, last and a spread of interior rows of every sequence, every head.
        let mut rows: Vec<usize> = vec![start, end - 1, start + len / 2, start + len / 3];
        rows.extend((0..8).map(|i| start + (i * 7919 + 13) % len));
        for &row in &rows {
            for head in 0..geo.q_heads {
                let want = reference_row(&qf, &kf, &vf, (start, end), row, head, &geo, scale);
                for (d, w) in want.iter().enumerate() {
                    let g = bf16_value(got[(row * geo.q_heads + head) * hd + d]);
                    assert!(
                        (g - w).abs() <= 1e-2 + 2e-2 * w.abs(),
                        "seq {s} len {len} row {row} head {head} dim {d} {geo:?}: gpu {g} ref {w}"
                    );
                }
            }
        }
    }
}

fn run_decode(ctx_len: usize, q_len: usize, block_size: usize) {
    let geo = DecodeGeometry::text(block_size);
    let (qh, kvh, hd) = (geo.q_heads, geo.kv_heads, 128usize);
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let kernels = PagedDecodeAttention::load(dev.context()).expect("kernels");
    let lens = [ctx_len, (ctx_len / 3).max(q_len)];
    let batch = lens.len();
    let max_len = *lens.iter().max().unwrap();
    let max_blocks = max_len.div_ceil(block_size);
    let num_blocks = lens.iter().map(|l| l.div_ceil(block_size)).sum::<usize>() + 3;

    let mut free: Vec<usize> = (0..num_blocks).rev().collect();
    let mut tables = vec![vec![0u32; max_blocks]; batch];
    #[allow(clippy::needless_range_loop)]
    for i in 0..max_blocks {
        for (s, l) in lens.iter().enumerate() {
            if i < l.div_ceil(block_size) {
                tables[s][i] = free.pop().unwrap() as u32;
            }
        }
    }
    let flat: Vec<i32> = tables.iter().flatten().map(|&b| b as i32).collect();

    let mut rng = Rng(0x1234_5678_9abc_def1 ^ ctx_len as u64);
    let cache_len = num_blocks * block_size * kvh * hd;
    let kc = random_bf16(&mut rng, cache_len, 1.0);
    let vc = random_bf16(&mut rng, cache_len, 2.0);
    let q = random_bf16(&mut rng, batch * q_len * qh * hd, 1.0);

    let d_kc = stream.clone_htod(&kc).unwrap();
    let d_vc = stream.clone_htod(&vc).unwrap();
    let d_q = stream.clone_htod(&q).unwrap();
    let d_tables = stream.clone_htod(&flat).unwrap();
    let d_lens = stream
        .clone_htod(&lens.iter().map(|&l| l as i32).collect::<Vec<_>>())
        .unwrap();
    let mut d_out = stream.alloc_zeros::<u16>(q.len()).unwrap();
    let (splits, chunk) = PagedDecodeAttention::plan_splits(&geo, batch, max_len, SM_COUNT);
    let mut ws = DecodeWorkspace::alloc(&stream, batch, splits, q_len, qh).unwrap();
    let seq_tables = SeqTables {
        block_tables: &d_tables,
        max_blocks,
    };
    let scale = (hd as f32).powf(-0.5);
    kernels
        .run(
            &stream,
            &geo,
            &d_kc,
            &d_vc,
            &seq_tables,
            &d_lens,
            &d_q,
            &mut d_out,
            &mut ws,
            batch,
            q_len,
            splits,
            chunk,
            scale,
        )
        .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<u16> = stream.clone_dtoh(&d_out).unwrap();

    let to_f32 = |x: &[u16]| x.iter().map(|&b| bf16_value(b)).collect::<Vec<f32>>();
    let (kf, vf, qf) = (to_f32(&kc), to_f32(&vc), to_f32(&q));
    let flat_tables: Vec<u32> = tables.iter().flatten().copied().collect();
    let kv = PagedKv {
        key_cache: &kf,
        value_cache: &vf,
        block_tables: &flat_tables,
        max_blocks,
        seq_lens: &lens,
        block_size,
    };
    let want = paged_decode_attention(&qf, &kv, q_len, qh, kvh, hd, scale);
    for (i, w) in want.iter().enumerate() {
        let g = bf16_value(got[i]);
        assert!(
            (g - w).abs() <= 1e-2 + 2e-2 * w.abs(),
            "idx {i} ctx {ctx_len} q_len {q_len} bs {block_size}: gpu {g} ref {w}"
        );
    }
}

macro_rules! varlen_cases {
    ($($name:ident: $lens:expr, $geo:expr;)*) => {$(
        #[test]
        #[ignore = "needs GPU"]
        fn $name() {
            if !gpu_enabled() {
                return;
            }
            run_varlen(&$lens, $geo);
        }
    )*};
}

macro_rules! decode_cases {
    ($($name:ident: $len:expr, $q:expr, $bs:expr;)*) => {$(
        #[test]
        #[ignore = "needs GPU"]
        fn $name() {
            if !gpu_enabled() {
                return;
            }
            run_decode($len, $q, $bs);
        }
    )*};
}

varlen_cases! {
    attention_vit_len17: [17], VarlenGeometry::vision(16);
    attention_vit_len920: [920], VarlenGeometry::vision(16);
    attention_vit_len6000: [6000], VarlenGeometry::vision(16);
    attention_vit_len24000: [24000], VarlenGeometry::vision(16);
    attention_vit_batch: [17, 920, 64, 65], VarlenGeometry::vision(16);
    attention_prefill_len300: [300], VarlenGeometry::prefill(16, 8);
    attention_prefill_len2048: [2048], VarlenGeometry::prefill(16, 8);
    attention_prefill_len16384: [16384], VarlenGeometry::prefill(16, 8);
    attention_prefill_batch: [300, 1, 129, 2048], VarlenGeometry::prefill(16, 8);
}

decode_cases! {
    attention_decode_len300: 300, 1, 16;
    attention_decode_len2048_q4: 2048, 4, 16;
    attention_decode_len16384_q3: 16384, 3, 16;
    attention_decode_len16384_q8: 16384, 8, 32;
    attention_decode_len17: 17, 1, 16;
}

struct DecodeCase {
    batch: usize,
    ctx: usize,
    q_len: usize,
    block_size: usize,
}

/// Host copies of key cache, value cache, queries and block tables.
type HostDecode = (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u32>);

struct DecodeData {
    d_kc: cudarc::driver::CudaSlice<u16>,
    d_vc: cudarc::driver::CudaSlice<u16>,
    d_q: cudarc::driver::CudaSlice<u16>,
    d_tables: cudarc::driver::CudaSlice<i32>,
    d_lens: cudarc::driver::CudaSlice<i32>,
    lens: Vec<usize>,
    max_blocks: usize,
    host: Option<HostDecode>,
}

/// Builds shuffled block tables and random caches; sequence lengths spread over `1..=ctx`
/// with the first sequence exactly `ctx`.
fn build_decode(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    c: &DecodeCase,
    keep_host: bool,
) -> DecodeData {
    let (qh, kvh, hd) = (16usize, 8usize, 128usize);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (c.ctx * 131 + c.batch * 7 + c.q_len) as u64);
    let lens: Vec<usize> = (0..c.batch)
        .map(|s| {
            if s == 0 {
                c.ctx
            } else {
                let f = 0.25 + 0.75 * ((rng.next() + 1.0) * 0.5).clamp(0.0, 1.0);
                ((c.ctx as f32 * f) as usize).clamp(c.q_len, c.ctx.max(c.q_len))
            }
        })
        .collect();
    let max_len = *lens.iter().max().unwrap();
    let max_blocks = max_len.div_ceil(c.block_size);
    let num_blocks: usize = lens.iter().map(|l| l.div_ceil(c.block_size)).sum::<usize>() + 3;
    let mut order: Vec<usize> = (0..num_blocks).collect();
    for i in (1..order.len()).rev() {
        let j = ((rng.next() + 1.0) * 0.5 * (i + 1) as f32) as usize % (i + 1);
        order.swap(i, j);
    }
    let mut tables = vec![0i32; c.batch * max_blocks];
    let mut at = 0;
    for (s, l) in lens.iter().enumerate() {
        for b in 0..l.div_ceil(c.block_size) {
            tables[s * max_blocks + b] = order[at] as i32;
            at += 1;
        }
    }
    let cache_len = num_blocks * c.block_size * kvh * hd;
    let kc = random_bf16(&mut rng, cache_len, 1.0);
    let vc = random_bf16(&mut rng, cache_len, 2.0);
    let q = random_bf16(&mut rng, c.batch * c.q_len * qh * hd, 1.0);
    DecodeData {
        d_kc: stream.clone_htod(&kc).unwrap(),
        d_vc: stream.clone_htod(&vc).unwrap(),
        d_q: stream.clone_htod(&q).unwrap(),
        d_tables: stream.clone_htod(&tables).unwrap(),
        d_lens: stream
            .clone_htod(&lens.iter().map(|&l| l as i32).collect::<Vec<_>>())
            .unwrap(),
        lens,
        max_blocks,
        host: keep_host.then(|| (kc, vc, q, tables.iter().map(|&b| b as u32).collect())),
    }
}

fn run_variant(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    kernels: &PagedDecodeAttention,
    partials: usize,
    c: &DecodeCase,
    data: &DecodeData,
    repeats: usize,
) -> (Vec<u16>, f64) {
    run_variant_poisoned(stream, kernels, partials, c, data, repeats, false)
}

/// As `run_variant`; with `poison` every launch follows a NaN fill of the SMs' shared memory.
fn run_variant_poisoned(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    kernels: &PagedDecodeAttention,
    partials: usize,
    c: &DecodeCase,
    data: &DecodeData,
    repeats: usize,
    poison: bool,
) -> (Vec<u16>, f64) {
    let geo = DecodeGeometry::text(c.block_size);
    let max_len = *data.lens.iter().max().unwrap();
    let (splits, chunk) = PagedDecodeAttention::plan_splits(&geo, c.batch, max_len, SM_COUNT);
    let mut ws = DecodeWorkspace::alloc(stream, c.batch, splits * partials, c.q_len, 16).unwrap();
    let mut out = stream
        .alloc_zeros::<u16>(c.batch * c.q_len * 16 * 128)
        .unwrap();
    let seq_tables = SeqTables {
        block_tables: &data.d_tables,
        max_blocks: data.max_blocks,
    };
    let scale = 128f32.powf(-0.5);
    let ctx = stream.context().clone();
    let mut launch = |out: &mut cudarc::driver::CudaSlice<u16>| {
        if poison {
            PagedDecodeAttention::poison_shared_memory(
                &ctx,
                stream,
                4 * SM_COUNT as u32 * 8,
                40_000,
            )
            .unwrap();
        }
        kernels
            .run(
                stream,
                &geo,
                &data.d_kc,
                &data.d_vc,
                &seq_tables,
                &data.d_lens,
                &data.d_q,
                out,
                &mut ws,
                c.batch,
                c.q_len,
                splits,
                chunk,
                scale,
            )
            .unwrap();
    };
    launch(&mut out);
    stream.synchronize().unwrap();
    let got: Vec<u16> = stream.clone_dtoh(&out).unwrap();
    let mut best = f64::MAX;
    for _ in 0..repeats {
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            launch(&mut out);
        }
        stream.synchronize().unwrap();
        best = best.min(t0.elapsed().as_secs_f64() / 10.0);
    }
    (got, best)
}

fn close(a: u16, b: f32) -> bool {
    let g = bf16_value(a);
    (g - b).abs() <= 1e-2 + 2e-2 * b.abs()
}

#[test]
#[ignore = "needs GPU"]
fn attention_decode_equals_baseline_and_reference() {
    if !gpu_enabled() {
        return;
    }
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let new = PagedDecodeAttention::load(dev.context()).expect("kernels");
    let base = PagedDecodeAttention::load_baseline(dev.context()).expect("baseline");
    for &batch in &[1usize, 7, 128] {
        for &ctx in &[1usize, 15, 16, 17, 300, 4000, 16384] {
            if batch == 128 && ctx > 4000 {
                continue;
            }
            for &q_len in &[1usize, 4] {
                let block_size = if ctx % 2 == 0 { 16 } else { 32 };
                let c = DecodeCase {
                    batch,
                    ctx: ctx.max(q_len),
                    q_len,
                    block_size,
                };
                let check_ref = batch * c.ctx <= 40_000;
                let data = build_decode(&stream, &c, check_ref);
                let (got, _) = run_variant(&stream, &new, 1, &c, &data, 0);
                let (old, _) = run_variant(&stream, &base, 4, &c, &data, 0);
                for (i, (&g, &o)) in got.iter().zip(&old).enumerate() {
                    assert!(
                        close(g, bf16_value(o)),
                        "idx {i} batch {batch} ctx {ctx} q_len {q_len}: new {} baseline {}",
                        bf16_value(g),
                        bf16_value(o)
                    );
                }
                if let Some((kc, vc, q, tables)) = &data.host {
                    let to_f32 = |x: &[u16]| x.iter().map(|&b| bf16_value(b)).collect::<Vec<f32>>();
                    let (kf, vf, qf) = (to_f32(kc), to_f32(vc), to_f32(q));
                    let kv = PagedKv {
                        key_cache: &kf,
                        value_cache: &vf,
                        block_tables: tables,
                        max_blocks: data.max_blocks,
                        seq_lens: &data.lens,
                        block_size,
                    };
                    let want =
                        paged_decode_attention(&qf, &kv, c.q_len, 16, 8, 128, 128f32.powf(-0.5));
                    for (i, w) in want.iter().enumerate() {
                        assert!(
                            close(got[i], *w),
                            "idx {i} batch {batch} ctx {ctx} q_len {q_len}: gpu {} ref {w}",
                            bf16_value(got[i])
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "needs GPU"]
fn attention_decode_bandwidth_vs_baseline() {
    if !gpu_enabled() {
        return;
    }
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let new = PagedDecodeAttention::load(dev.context()).expect("kernels");
    let base = PagedDecodeAttention::load_baseline(dev.context()).expect("baseline");
    {
        let src = stream.alloc_zeros::<u8>(1 << 29).unwrap();
        let mut dst = stream.alloc_zeros::<u8>(1 << 29).unwrap();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            for _ in 0..5 {
                stream.memcpy_dtod(&src, &mut dst).unwrap();
            }
            stream.synchronize().unwrap();
            best = best.min(t0.elapsed().as_secs_f64() / 5.0);
        }
        println!(
            "device copy (read + write) {:.0} GB/s",
            2.0 * (1u64 << 29) as f64 / best / 1e9
        );
    }
    println!("| batch | ctx | q_len | baseline us | baseline GB/s | new us | new GB/s |");
    for &(batch, ctx, q_len) in &[
        (128usize, 300usize, 1usize),
        (128, 300, 4),
        (128, 2000, 1),
        (32, 2000, 1),
        (8, 4000, 1),
        (1, 16384, 1),
        (1, 16384, 4),
    ] {
        let c = DecodeCase {
            batch,
            ctx,
            q_len,
            block_size: 16,
        };
        let data = build_decode(&stream, &c, false);
        let bytes: f64 = data
            .lens
            .iter()
            .map(|&l| (l * 8 * 128 * 2 * 2) as f64)
            .sum();
        let (_, tn) = run_variant(&stream, &new, 1, &c, &data, 5);
        let (_, tb) = run_variant(&stream, &base, 4, &c, &data, 5);
        println!(
            "| {batch} | {ctx} | {q_len} | {:.1} | {:.0} | {:.1} | {:.0} |",
            tb * 1e6,
            bytes / tb / 1e9,
            tn * 1e6,
            bytes / tn / 1e9
        );
    }
}

#[test]
#[ignore = "needs GPU"]
fn attention_decode_bitwise_equals_reference_with_poisoned_shared_memory() {
    if !gpu_enabled() {
        return;
    }
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let new = PagedDecodeAttention::load(dev.context()).expect("kernels");
    let reference = PagedDecodeAttention::load_reference(dev.context()).expect("reference");
    for &batch in &[1usize, 3, 22, 32, 48, 130] {
        for &ctx in &[1usize, 5, 16, 17, 31, 33, 100, 350, 1001, 4003, 9000] {
            if batch * ctx > 400_000 {
                continue;
            }
            for &q_len in &[1usize, 4, 8] {
                for &block_size in &[16usize, 32] {
                    let c = DecodeCase {
                        batch,
                        ctx: ctx.max(q_len),
                        q_len,
                        block_size,
                    };
                    let data = build_decode(&stream, &c, false);
                    let (want, _) =
                        run_variant_poisoned(&stream, &reference, 1, &c, &data, 0, true);
                    let (got, _) = run_variant_poisoned(&stream, &new, 1, &c, &data, 0, true);
                    let (clean, _) = run_variant(&stream, &new, 1, &c, &data, 0);
                    assert!(
                        want.iter().all(|&b| bf16_value(b).is_finite()),
                        "reference non-finite batch {batch} ctx {ctx} q_len {q_len}"
                    );
                    assert!(
                        got == want,
                        "batch {batch} ctx {ctx} q_len {q_len} bs {block_size}"
                    );
                    assert!(
                        clean == want,
                        "unpoisoned batch {batch} ctx {ctx} q_len {q_len}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "needs GPU"]
fn attention_decode_live_shapes_bandwidth() {
    if !gpu_enabled() {
        return;
    }
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let new = PagedDecodeAttention::load(dev.context()).expect("kernels");
    let base = PagedDecodeAttention::load_baseline(dev.context()).expect("baseline");
    println!("| batch | ctx | q_len | baseline us | baseline GB/s | new us | new GB/s |");
    for &(batch, ctx, q_len) in &[
        (32usize, 1000usize, 5usize),
        (32, 2000, 5),
        (32, 4000, 5),
        (32, 2000, 1),
        (16, 2000, 5),
        (8, 2000, 5),
        (1, 16384, 5),
    ] {
        let c = DecodeCase {
            batch,
            ctx,
            q_len,
            block_size: 16,
        };
        let data = build_decode(&stream, &c, false);
        let bytes: f64 = data
            .lens
            .iter()
            .map(|&l| (l * 8 * 128 * 2 * 2) as f64)
            .sum();
        let (_, tn) = run_variant(&stream, &new, 1, &c, &data, 20);
        let (_, tb) = run_variant(&stream, &base, 4, &c, &data, 20);
        println!(
            "| {batch} | {ctx} | {q_len} | {:.1} | {:.0} | {:.1} | {:.0} |",
            tb * 1e6,
            bytes / tb / 1e9,
            tn * 1e6,
            bytes / tn / 1e9
        );
    }
}

#[test]
#[ignore = "needs GPU"]
fn attention_decode_error_vs_reference_not_worse_than_baseline() {
    if !gpu_enabled() {
        return;
    }
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let new = PagedDecodeAttention::load(dev.context()).expect("kernels");
    let base = PagedDecodeAttention::load_baseline(dev.context()).expect("baseline");
    let (mut new_max, mut new_sum, mut base_max, mut base_sum, mut count) =
        (0f64, 0f64, 0f64, 0f64, 0usize);
    for &batch in &[1usize, 2, 5, 16, 32] {
        for &ctx in &[1usize, 7, 16, 63, 64, 65, 200, 513, 1000, 2048] {
            for &q_len in &[1usize, 5] {
                let c = DecodeCase {
                    batch,
                    ctx: ctx.max(q_len),
                    q_len,
                    block_size: if ctx % 2 == 0 { 16 } else { 32 },
                };
                let data = build_decode(&stream, &c, true);
                let (got_new, _) = run_variant(&stream, &new, 1, &c, &data, 0);
                let (got_base, _) = run_variant(&stream, &base, 4, &c, &data, 0);
                let (kc, vc, q, tables) = data.host.as_ref().unwrap();
                let to_f32 = |x: &[u16]| x.iter().map(|&b| bf16_value(b)).collect::<Vec<f32>>();
                let (kf, vf, qf) = (to_f32(kc), to_f32(vc), to_f32(q));
                let kv = PagedKv {
                    key_cache: &kf,
                    value_cache: &vf,
                    block_tables: tables,
                    max_blocks: data.max_blocks,
                    seq_lens: &data.lens,
                    block_size: c.block_size,
                };
                let want = paged_decode_attention(&qf, &kv, c.q_len, 16, 8, 128, 128f32.powf(-0.5));
                for (i, w) in want.iter().enumerate() {
                    let en = (bf16_value(got_new[i]) - w).abs() as f64;
                    let eb = (bf16_value(got_base[i]) - w).abs() as f64;
                    new_max = new_max.max(en);
                    base_max = base_max.max(eb);
                    new_sum += en;
                    base_sum += eb;
                    count += 1;
                }
            }
        }
    }
    let (new_mean, base_mean) = (new_sum / count as f64, base_sum / count as f64);
    println!("elements {count}");
    println!("new      max abs err {new_max:.6e} mean abs err {new_mean:.6e}");
    println!("baseline max abs err {base_max:.6e} mean abs err {base_mean:.6e}");
    assert!(new_max <= base_max * 1.0001 + 1e-6, "max error regressed");
    assert!(
        new_mean <= base_mean * 1.0001 + 1e-9,
        "mean error regressed"
    );
}

/// Keys per KV block of the row-invariance probe.
const PROBE_BLOCK: usize = 16;
/// Longest context the row-invariance probe configures the split chunks for.
const PROBE_MAX_CTX: usize = 1056;
/// Filler sequences available beside the probe sequence.
const PROBE_FILLERS: usize = 31;

/// A fixed key/value history for a probe sequence and filler sequences, with shuffled block
/// tables, and the probe query.
struct ProbeCache {
    kc: cudarc::driver::CudaSlice<u16>,
    vc: cudarc::driver::CudaSlice<u16>,
    probe_blocks: Vec<i32>,
    filler_blocks: Vec<Vec<i32>>,
    filler_lens: Vec<usize>,
    q_probe: Vec<u16>,
    position: usize,
}

fn filler_len(f: usize) -> usize {
    100 + 13 * f
}

fn probe_cache(stream: &std::sync::Arc<cudarc::driver::CudaStream>, position: usize) -> ProbeCache {
    let (kvh, hd, qh) = (8usize, 128usize, 16usize);
    let mut rng = Rng(0x7fb5_d329_728e_a185 ^ position as u64);
    let probe_len = position + 8;
    let filler_lens: Vec<usize> = (0..PROBE_FILLERS).map(filler_len).collect();
    let counts: Vec<usize> = std::iter::once(probe_len)
        .chain(filler_lens.iter().copied())
        .map(|l| l.div_ceil(PROBE_BLOCK))
        .collect();
    let total: usize = counts.iter().sum::<usize>() + 3;
    let mut order: Vec<i32> = (0..total as i32).collect();
    for i in (1..order.len()).rev() {
        let j = ((rng.next() + 1.0) * 0.5 * (i + 1) as f32) as usize % (i + 1);
        order.swap(i, j);
    }
    let mut at = 0;
    let mut take = |n: usize| {
        let blocks = order[at..at + n].to_vec();
        at += n;
        blocks
    };
    let probe_blocks = take(counts[0]);
    let filler_blocks = counts[1..].iter().map(|&n| take(n)).collect();
    let cache_len = total * PROBE_BLOCK * kvh * hd;
    let kc = random_bf16(&mut rng, cache_len, 1.0);
    let vc = random_bf16(&mut rng, cache_len, 2.0);
    let q_probe = random_bf16(&mut rng, qh * hd, 1.0);
    ProbeCache {
        kc: stream.clone_htod(&kc).unwrap(),
        vc: stream.clone_htod(&vc).unwrap(),
        probe_blocks,
        filler_blocks,
        filler_lens,
        q_probe,
        position,
    }
}

/// Attention output of the probe query at `cache.position`, as row `row` of a `q_len` window of
/// the sequence in batch slot `slot` of a `batch`-sequence launch split `splits` ways; the other
/// window rows and the filler sequences' queries are drawn from `seed`.
#[allow(clippy::too_many_arguments)]
fn probe_variant(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    kernels: &PagedDecodeAttention,
    cache: &ProbeCache,
    batch: usize,
    slot: usize,
    q_len: usize,
    row: usize,
    splits: usize,
    seed: u64,
) -> Vec<u16> {
    let (qh, hd) = (16usize, 128usize);
    let geo = DecodeGeometry::text(PROBE_BLOCK);
    let probe_len = cache.position + 1 + (q_len - 1 - row);
    let filler = |s: usize| if s < slot { s } else { s - 1 };
    let lens: Vec<usize> = (0..batch)
        .map(|s| {
            if s == slot {
                probe_len
            } else {
                cache.filler_lens[filler(s)]
            }
        })
        .collect();
    let max_blocks = lens.iter().map(|l| l.div_ceil(PROBE_BLOCK)).max().unwrap();
    let mut tables = vec![0i32; batch * max_blocks];
    for s in 0..batch {
        let blocks = if s == slot {
            &cache.probe_blocks
        } else {
            &cache.filler_blocks[filler(s)]
        };
        for (b, &id) in blocks.iter().enumerate().take(max_blocks) {
            tables[s * max_blocks + b] = id;
        }
    }
    let mut rng = Rng(seed ^ 0x2545_f491_4f6c_dd1d);
    let mut q = random_bf16(&mut rng, batch * q_len * qh * hd, 1.0);
    let at = (slot * q_len + row) * qh * hd;
    q[at..at + qh * hd].copy_from_slice(&cache.q_probe);
    let d_q = stream.clone_htod(&q).unwrap();
    let d_tables = stream.clone_htod(&tables).unwrap();
    let d_lens = stream
        .clone_htod(&lens.iter().map(|&l| l as i32).collect::<Vec<_>>())
        .unwrap();
    let chunk = scriptor_kernels::attention::split_chunk(PROBE_MAX_CTX, splits);
    let mut ws = DecodeWorkspace::alloc(stream, batch, splits, q_len, qh).unwrap();
    let mut out = stream.alloc_zeros::<u16>(batch * q_len * qh * hd).unwrap();
    let seq_tables = SeqTables {
        block_tables: &d_tables,
        max_blocks,
    };
    kernels
        .run(
            stream,
            &geo,
            &cache.kc,
            &cache.vc,
            &seq_tables,
            &d_lens,
            &d_q,
            &mut out,
            &mut ws,
            batch,
            q_len,
            splits,
            chunk,
            128f32.powf(-0.5),
        )
        .unwrap();
    stream.synchronize().unwrap();
    let all: Vec<u16> = stream.clone_dtoh(&out).unwrap();
    all[at..at + qh * hd].to_vec()
}

/// The attention output of one query row is bit-identical across its slot in the query window,
/// the batch slot and size of the launch, and the filler sequences' contents, at a fixed split
/// count. When the whole context of the probe row fits in the first split chunk, the output is
/// also bit-identical to the single-split output. Other differences across split counts and
/// query-window lengths are printed, not asserted.
#[test]
#[ignore = "needs GPU"]
fn paged_attention_row_invariant_to_window_and_splits() {
    if !gpu_enabled() {
        return;
    }
    let started = std::time::Instant::now();
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let kernels = PagedDecodeAttention::load(dev.context()).expect("kernels");
    let mut single_split: std::collections::HashMap<usize, Vec<Vec<u16>>> = Default::default();
    for (position, splits) in [
        (700usize, 1usize),
        (700, 2),
        (700, 3),
        (300, 1),
        (300, 2),
        (300, 3),
        (300, 12),
        (100, 1),
        (100, 12),
    ] {
        let cache = probe_cache(&stream, position);
        let mut firsts: Vec<Vec<u16>> = Vec::new();
        for q_len in [4usize, 5] {
            let last = q_len - 1;
            let variants = [
                (4usize, 0usize, 0usize),
                (4, 3, last),
                (8, 5, 2),
                (16, 15, 1),
                (32, 31, 0),
                (32, 17, last),
            ];
            let mut first: Option<Vec<u16>> = None;
            for (n, &(batch, slot, row)) in variants.iter().enumerate() {
                let got = probe_variant(
                    &stream, &kernels, &cache, batch, slot, q_len, row, splits, n as u64,
                );
                match &first {
                    None => first = Some(got),
                    Some(want) => assert!(
                        &got == want,
                        "position {position} splits {splits} q_len {q_len} batch {batch} slot {slot} row {row}: the probe row differs from the first variant"
                    ),
                }
            }
            firsts.push(first.unwrap());
        }
        let differing = |a: &[u16], b: &[u16]| a.iter().zip(b).filter(|(x, y)| x != y).count();
        eprintln!(
            "ATTN_ROW_DIAGNOSTIC position {position} splits {splits}: q_len 4 vs 5 differ in {} of {} words",
            differing(&firsts[0], &firsts[1]),
            firsts[0].len()
        );
        if splits == 1 {
            single_split.insert(position, firsts.clone());
            let one = probe_variant(&stream, &kernels, &cache, 16, 0, 1, 0, 1, 99);
            eprintln!(
                "ATTN_ROW_DIAGNOSTIC position {position} splits 1: q_len 1 vs 4 differ in {} words",
                differing(&one, &firsts[0])
            );
        } else if let Some(base) = single_split.get(&position) {
            eprintln!(
                "ATTN_ROW_DIAGNOSTIC position {position} splits {splits}: differ from the single-split row in {} of {} words",
                differing(&firsts[0], &base[0]),
                base[0].len()
            );
            let chunk = scriptor_kernels::attention::split_chunk(PROBE_MAX_CTX, splits);
            let longest_context = position + 5;
            if longest_context <= chunk {
                assert!(
                    firsts == *base,
                    "position {position} splits {splits}: the context of at most {longest_context} keys fits the first chunk of {chunk} keys but the row differs from the single-split row"
                );
            }
        }
        eprintln!(
            "phase attention_row_invariance_p{position}_s{splits} elapsed_s={:.3}",
            started.elapsed().as_secs_f64()
        );
    }
}
