//! GPU checks of the skinny-M bf16 GEMM: agreement with an f64 host reference and with
//! cuBLASLt on synthetic operands for every served weight shape, plus an L2-flushed
//! bandwidth comparison against cuBLASLt.
//! Run with `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda --release skinny -- --ignored --nocapture`.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use half::bf16;
use scriptor_cuda::{Device, DeviceArena};
use scriptor_kernels::cublaslt::{GemmEngine, GemmOperands, GemmOptions, GemmProblem, XorShift};
use scriptor_kernels::heuristics::{Epilogue, GemmDtype, TuningTable};
use scriptor_kernels::skinny_gemm::SHAPES;

const MS: [usize; 7] = [1, 2, 3, 32, 64, 127, 128];

fn gpu_enabled() -> bool {
    std::env::var("SCRIPTOR_GPU_TESTS").as_deref() == Ok("1")
}

struct Ctx {
    dev: Device,
    stream: Arc<CudaStream>,
    arena: DeviceArena,
}

impl Ctx {
    fn new() -> Self {
        assert!(gpu_enabled(), "set SCRIPTOR_GPU_TESTS=1 to run GPU tests");
        let dev = Device::open().expect("gpu");
        let stream = dev.new_stream().expect("stream");
        let arena = dev.arena(128 << 20, &stream).expect("arena");
        Self { dev, stream, arena }
    }

    fn engine(&mut self) -> GemmEngine {
        GemmEngine::new(
            self.dev.context(),
            &self.stream,
            &mut self.arena,
            TuningTable::default(),
            GemmOptions {
                fp8: false,
                workspace_bytes: 32 << 20,
            },
        )
        .expect("gemm engine")
    }

    fn ptr<T>(&self, s: &CudaSlice<T>) -> u64 {
        s.device_ptr(&self.stream).0
    }

    fn upload(&self, v: &[bf16]) -> CudaSlice<u16> {
        let bits: Vec<u16> = v.iter().map(|b| b.to_bits()).collect();
        self.stream.clone_htod(&bits).expect("upload")
    }

    fn download(&self, s: &CudaSlice<u16>) -> Vec<bf16> {
        self.stream.synchronize().expect("sync");
        self.stream
            .clone_dtoh(s)
            .expect("download")
            .into_iter()
            .map(bf16::from_bits)
            .collect()
    }
}

fn random_bf16(rng: &mut XorShift, n: usize, scale: f32) -> Vec<bf16> {
    (0..n)
        .map(|_| bf16::from_f32(rng.next_f32() * scale))
        .collect()
}

fn problem(m: usize, n: usize, k: usize) -> GemmProblem {
    GemmProblem {
        m,
        n,
        k,
        dtype: GemmDtype::Bf16,
        epilogue: Epilogue::None,
    }
}

/// Distance in bf16 steps between two finite values of the same sign class.
fn ulp_distance(a: bf16, b: bf16) -> u32 {
    let key = |v: bf16| {
        let bits = v.to_bits() as i32;
        if bits & 0x8000 != 0 {
            -(bits & 0x7fff)
        } else {
            bits
        }
    };
    key(a).abs_diff(key(b))
}

/// f64 products of every token row with weight row `col`, with the sum of absolute terms.
fn host_reference(x: &[bf16], w: &[bf16], m: usize, k: usize, col: usize) -> Vec<(bf16, f64)> {
    let wr: Vec<f64> = w[col * k..(col + 1) * k]
        .iter()
        .map(|v| f64::from(v.to_f32()))
        .collect();
    (0..m)
        .map(|r| {
            let terms = x[r * k..(r + 1) * k]
                .iter()
                .zip(&wr)
                .map(|(a, b)| f64::from(a.to_f32()) * b);
            let (acc, mag) = terms.fold((0.0, 0.0), |(s, a), t| (s + t, a + t.abs()));
            (bf16::from_f64(acc), mag)
        })
        .collect()
}

#[test]
#[ignore = "needs GPU"]
fn skinny_gemm_matches_f64_reference_and_cublaslt() {
    let mut ctx = Ctx::new();
    let mut engine = ctx.engine();
    for shape in SHAPES {
        let (n, k) = (shape.n, shape.k);
        let mut rng = XorShift(0x5eed_0000 ^ (n as u64) << 20 ^ k as u64);
        let w = random_bf16(&mut rng, n * k, 0.5);
        let dw = ctx.upload(&w);
        for m in MS {
            let x = random_bf16(&mut rng, m * k, 1.0);
            let dx = ctx.upload(&x);
            let out_s = ctx.stream.alloc_zeros::<u16>(m * n).expect("out");
            let out_c = ctx.stream.alloc_zeros::<u16>(m * n).expect("out");
            let p = problem(m, n, k);
            let ops = |out: &CudaSlice<u16>| GemmOperands {
                x: ctx.ptr(&dx),
                w: ctx.ptr(&dw),
                out: ctx.ptr(out),
                ..Default::default()
            };
            engine.set_skinny_enabled(true);
            engine.gemm(&p, &ops(&out_s)).expect("skinny");
            engine.gemm_with(&p, &ops(&out_c), 0).expect("cublaslt");
            let got = ctx.download(&out_s);
            let lt = ctx.download(&out_c);

            let step = if m * n * k > 100_000_000 { n / 509 } else { 1 };
            let mut cols: Vec<usize> = (0..n).step_by(step.max(1)).collect();
            cols.extend(n.saturating_sub(16)..n);
            let (mut worst, mut checked, mut beyond) = (0u32, 0usize, 0usize);
            for &c in &cols {
                let want = host_reference(&x, &w, m, k, c);
                for (r, (wv, mag)) in want.iter().enumerate() {
                    let g = got[r * n + c];
                    let d = ulp_distance(g, *wv);
                    checked += 1;
                    // Beyond one bf16 step the fp32 accumulation error (bounded by the sum
                    // of absolute terms) must account for the difference.
                    let slack =
                        (f64::from(g.to_f32()) - f64::from(wv.to_f32())).abs() <= 2e-6 * mag;
                    if d > 1 {
                        beyond += 1;
                        assert!(slack, "n={n} k={k} m={m} row={r} col={c}: {d} ulp");
                    }
                    worst = worst.max(d);
                }
            }
            let lt_worst = cols
                .iter()
                .map(|&c| {
                    let want = host_reference(&x, &w, m, k, c);
                    want.iter()
                        .enumerate()
                        .map(|(r, (wv, _))| ulp_distance(lt[r * n + c], *wv))
                        .max()
                        .unwrap_or(0)
                })
                .max()
                .unwrap_or(0);
            let equal = got.iter().zip(&lt).filter(|(a, b)| a == b).count();
            println!(
                "n={n} k={k} m={m}: max {worst} ulp vs f64 over {checked} elements ({beyond} beyond 1 ulp, all within fp32 accumulation error; cuBLASLt itself {lt_worst}), {:.4} exactly equal to cuBLASLt",
                equal as f64 / got.len() as f64
            );
        }
    }
}

/// The split-K counters return to zero, so repeated launches reproduce the output exactly.
#[test]
#[ignore = "needs GPU"]
fn skinny_gemm_repeats_bit_identically() {
    let mut ctx = Ctx::new();
    let mut engine = ctx.engine();
    for shape in SHAPES {
        let (n, k, m) = (shape.n, shape.k, 96);
        let mut rng = XorShift(0xabcdef ^ n as u64);
        let dw = ctx.upload(&random_bf16(&mut rng, n * k, 0.5));
        let dx = ctx.upload(&random_bf16(&mut rng, m * k, 1.0));
        let first = ctx.stream.alloc_zeros::<u16>(m * n).expect("out");
        let p = problem(m, n, k);
        let ops = |out: &CudaSlice<u16>| GemmOperands {
            x: ctx.ptr(&dx),
            w: ctx.ptr(&dw),
            out: ctx.ptr(out),
            ..Default::default()
        };
        engine.gemm(&p, &ops(&first)).expect("first");
        let want = ctx.download(&first);
        for _ in 0..20 {
            let again = ctx.stream.alloc_zeros::<u16>(m * n).expect("out");
            engine.gemm(&p, &ops(&again)).expect("again");
            assert!(ctx.download(&again) == want, "n={n} k={k} not repeatable");
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    v[v.len() / 2]
}

/// Times `run` after flushing the L2 with a write sweep of `flush`; microseconds, median.
fn time_flushed(
    ctx: &Ctx,
    flush: &mut CudaSlice<u8>,
    iterations: usize,
    mut run: impl FnMut(),
) -> f64 {
    use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
    let mut times = Vec::new();
    for i in 0..iterations + 3 {
        ctx.stream.memset_zeros(flush).expect("flush");
        let start = ctx.stream.record_event(Some(CU_EVENT_DEFAULT)).expect("ev");
        run();
        let end = ctx.stream.record_event(Some(CU_EVENT_DEFAULT)).expect("ev");
        end.synchronize().expect("sync");
        if i >= 3 {
            times.push(f64::from(start.elapsed_ms(&end).expect("elapsed")) * 1e3);
        }
    }
    median(&mut times)
}

#[test]
#[ignore = "needs GPU"]
fn skinny_gemm_bandwidth_against_cublaslt() {
    let mut ctx = Ctx::new();
    let mut engine = ctx.engine();
    let mut flush = ctx.stream.alloc_zeros::<u8>(256 << 20).expect("flush");
    for shape in SHAPES {
        let (n, k) = (shape.n, shape.k);
        let mut rng = XorShift(0x77 ^ n as u64);
        let dw = ctx.upload(&random_bf16(&mut rng, n * k, 0.5));
        let weight_bytes = (n * k * 2) as f64;
        for m in [128, 64, 32, 8, 2, 1] {
            let dx = ctx.upload(&random_bf16(&mut rng, m * k, 1.0));
            let out = ctx.stream.alloc_zeros::<u16>(m * n).expect("out");
            let p = problem(m, n, k);
            let ops = GemmOperands {
                x: ctx.ptr(&dx),
                w: ctx.ptr(&dw),
                out: ctx.ptr(&out),
                ..Default::default()
            };
            let skinny = time_flushed(&ctx, &mut flush, 30, || {
                engine.gemm(&p, &ops).expect("skinny");
            });
            let lt = time_flushed(&ctx, &mut flush, 30, || {
                engine.gemm_with(&p, &ops, 0).expect("cublaslt");
            });
            println!(
                "n={n} k={k} m={m}: skinny {skinny:.1} us = {:.1} GB/s | cuBLASLt {lt:.1} us = {:.1} GB/s | {:.2}x",
                weight_bytes / skinny / 1e3,
                weight_bytes / lt / 1e3,
                lt / skinny
            );
        }
    }
}

/// Back-to-back launches over distinct weight copies (no L2 reuse, no flush between launches),
/// the way a decode graph issues them: mean microseconds per launch and effective weight GB/s.
#[test]
#[ignore = "needs GPU"]
fn skinny_gemm_chained_bandwidth() {
    use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
    let mut ctx = Ctx::new();
    let mut engine = ctx.engine();
    for shape in SHAPES {
        let (n, k) = (shape.n, shape.k);
        let weight_bytes = (n * k * 2) as f64;
        let copies = (((768u64 << 20) as f64 / weight_bytes).ceil() as usize).max(4);
        let mut rng = XorShift(0x55 ^ n as u64);
        let base = random_bf16(&mut rng, n * k, 0.5);
        let ws: Vec<CudaSlice<u16>> = (0..copies).map(|_| ctx.upload(&base)).collect();
        for m in [128, 32] {
            let dx = ctx.upload(&random_bf16(&mut rng, m * k, 1.0));
            let out = ctx.stream.alloc_zeros::<u16>(m * n).expect("out");
            let p = problem(m, n, k);
            let mut times = Vec::new();
            for round in 0..8 {
                let start = ctx.stream.record_event(Some(CU_EVENT_DEFAULT)).expect("ev");
                for w in &ws {
                    let ops = GemmOperands {
                        x: ctx.ptr(&dx),
                        w: ctx.ptr(w),
                        out: ctx.ptr(&out),
                        ..Default::default()
                    };
                    engine.gemm(&p, &ops).expect("skinny");
                }
                let end = ctx.stream.record_event(Some(CU_EVENT_DEFAULT)).expect("ev");
                end.synchronize().expect("sync");
                if round >= 2 {
                    times.push(
                        f64::from(start.elapsed_ms(&end).expect("elapsed")) * 1e3 / copies as f64,
                    );
                }
            }
            let us = median(&mut times);
            println!(
                "chain n={n} k={k} m={m}: {us:.1} us = {:.1} GB/s",
                weight_bytes / us / 1e3
            );
        }
    }
}

/// Greedy tokens of the existing path: the plain skinny product into bf16 logits followed by the
/// sampler's greedy argmax, with the highest and second-highest values and their multiplicity.
#[allow(clippy::too_many_arguments)]
fn reference_tokens(
    ctx: &Ctx,
    engine: &mut GemmEngine,
    sampler: &scriptor_kernels::sampling::SamplingKernels,
    dx: &CudaSlice<u16>,
    dw: &CudaSlice<u16>,
    m: usize,
    n: usize,
    k: usize,
) -> (Vec<u32>, Vec<bf16>) {
    use scriptor_kernels::sampling::{RowParams, TokenMasks};
    let out = ctx.stream.alloc_zeros::<u16>(m.max(2) * n).expect("out");
    let ops = GemmOperands {
        x: ctx.ptr(dx),
        w: ctx.ptr(dw),
        out: ctx.ptr(&out),
        ..Default::default()
    };
    engine.gemm(&problem(m, n, k), &ops).expect("skinny");
    let masks = TokenMasks::new(m, n);
    let params: Vec<RowParams> = (0..m as u32).map(RowParams::greedy).collect();
    let d_params = ctx.stream.clone_htod(&params).expect("params");
    let prompt = ctx.stream.clone_htod(masks.prompt_words()).expect("p");
    let output = ctx.stream.clone_htod(masks.output_words()).expect("o");
    let counts = ctx.stream.clone_htod(masks.count_words()).expect("c");
    let mut tokens = ctx.stream.alloc_zeros::<u32>(m).expect("tokens");
    let mut margins = ctx.stream.alloc_zeros::<f32>(m).expect("margins");
    sampler
        .choose_bf16(
            &ctx.stream,
            true,
            &out,
            m,
            n,
            &d_params,
            &prompt,
            &output,
            &counts,
            masks.words(),
            &mut tokens,
            &mut margins,
        )
        .expect("choose");
    ctx.stream.synchronize().expect("sync");
    let toks = ctx.stream.clone_dtoh(&tokens).expect("dtoh");
    (toks, ctx.download(&out))
}

/// The fused head argmax picks exactly the tokens of the logits-plus-sampler path, including
/// rows whose maximum is shared by several columns after bf16 rounding (exact duplicates and
/// columns that differ only below the bf16 step) spread across weight-row tiles and CTAs.
#[test]
#[ignore = "needs GPU"]
fn fused_head_argmax_matches_logits_and_sampler() {
    argmax_parity(&[1usize, 2, 3, 8, 9, 16, 31, 32, 33, 48, 63, 64], false);
}

/// The wide fused argmax (65..=128 rows) picks exactly the tokens of the logits-plus-sampler
/// path on the same adversarial rows, and the narrow route stays selected at 64 rows.
#[test]
#[ignore = "needs GPU"]
fn wide_head_argmax_matches_logits_and_sampler() {
    argmax_parity(&[65usize, 72, 96, 120, 127, 128], true);
}

fn argmax_parity(ms: &[usize], wide: bool) {
    use scriptor_kernels::sampling::SamplingKernels;
    let mut ctx = Ctx::new();
    let mut engine = ctx.engine();
    if wide {
        assert!(
            !engine.supports_argmax(65, 59392, 1536),
            "wide is off by default"
        );
        engine.set_wide_argmax_enabled(true);
        assert!(engine.supports_argmax(128, 59392, 1536));
        assert!(!engine.supports_argmax(129, 59392, 1536));
    }
    let sampler = SamplingKernels::load(ctx.dev.context()).expect("sampler");
    let shapes = [(59392usize, 1536usize), (9216, 1536)];
    let (mut cases, mut tied_rows) = (0usize, 0usize);
    for (n, k) in shapes {
        for &m in ms {
            for seed in 0..3u64 {
                let mut rng = XorShift(0xfeed_0000 ^ (n as u64) << 8 ^ (m as u64) << 2 ^ seed);
                let mut w = random_bf16(&mut rng, n * k, 0.5);
                // Positive activations and a dominant all-ones direction make the planted
                // columns the row maxima.
                let x: Vec<bf16> = (0..m * k)
                    .map(|_| bf16::from_f32(0.5 + 0.5 * rng.next_f32().abs()))
                    .collect();
                if seed > 0 {
                    let planted = [n - 1, 7, n / 2, n / 2 + 1, 64, 63, 1000 + seed as usize];
                    for (i, &c) in planted.iter().enumerate() {
                        for j in 0..k {
                            let mut v = 1.0f32;
                            // Odd copies differ by far less than one bf16 step of the product.
                            if seed == 2 && i % 2 == 1 && j == 5 {
                                v += 1.0 / 4096.0;
                            }
                            w[c * k + j] = bf16::from_f32(v);
                        }
                    }
                }
                let dw = ctx.upload(&w);
                let dx = ctx.upload(&x);
                let (want, logits) =
                    reference_tokens(&ctx, &mut engine, &sampler, &dx, &dw, m, n, k);
                assert!(engine.supports_argmax(m, n, k));
                let mut got = ctx.stream.alloc_zeros::<u32>(m).expect("tokens");
                let launch = |engine: &mut GemmEngine, got: &mut CudaSlice<u32>| {
                    engine
                        .argmax_rows(m, n, k, ctx.ptr(&dx), ctx.ptr(&dw), ctx.ptr(got))
                        .expect("fused");
                };
                launch(&mut engine, &mut got);
                ctx.stream.synchronize().expect("sync");
                let first = ctx.stream.clone_dtoh(&got).expect("dtoh");
                // Repeated launches reproduce the tokens (the counter resets).
                launch(&mut engine, &mut got);
                ctx.stream.synchronize().expect("sync");
                let second = ctx.stream.clone_dtoh(&got).expect("dtoh");
                assert_eq!(first, second, "n={n} m={m} seed={seed}: repeat");
                assert_eq!(first, want, "n={n} k={k} m={m} seed={seed}");
                for r in 0..m {
                    let row = &logits[r * n..(r + 1) * n];
                    let best = row.iter().map(|v| v.to_f32()).fold(f32::MIN, f32::max);
                    let shared = row.iter().filter(|v| v.to_f32() == best).count();
                    if shared > 1 {
                        tied_rows += 1;
                        let lowest = row.iter().position(|v| v.to_f32() == best).unwrap();
                        assert_eq!(first[r] as usize, lowest, "tie must take the lowest index");
                    }
                }
                cases += 1;
            }
        }
    }
    println!(
        "fused argmax (wide={wide}) identical in {cases} cases ({tied_rows} rows with tied maxima)"
    );
    assert!(tied_rows > 0, "the test must exercise tied maxima");
}

/// Microbenchmark of one head product at the draft batch sizes: the logits-plus-sampler path
/// against the fused argmax, L2 flushed before each launch.
#[test]
#[ignore = "needs GPU"]
fn fused_head_argmax_timing() {
    use scriptor_kernels::sampling::SamplingKernels;
    let mut ctx = Ctx::new();
    let mut engine = ctx.engine();
    let sampler = SamplingKernels::load(ctx.dev.context()).expect("sampler");
    let (n, k) = (59392usize, 1536usize);
    let mut rng = XorShift(0x7157);
    let dw = ctx.upload(&random_bf16(&mut rng, n * k, 0.5));
    let flush = ctx.stream.alloc_zeros::<u8>(256 << 20).expect("flush");
    let mut flush_mut = flush;
    for m in [8usize, 16, 32, 64] {
        use scriptor_kernels::sampling::{RowParams, TokenMasks};
        let dx = ctx.upload(&random_bf16(&mut rng, m * k, 1.0));
        let out = ctx.stream.alloc_zeros::<u16>(m * n).expect("out");
        let masks = TokenMasks::new(m, n);
        let params: Vec<RowParams> = (0..m as u32).map(RowParams::greedy).collect();
        let d_params = ctx.stream.clone_htod(&params).expect("params");
        let prompt = ctx.stream.clone_htod(masks.prompt_words()).expect("p");
        let output = ctx.stream.clone_htod(masks.output_words()).expect("o");
        let counts = ctx.stream.clone_htod(masks.count_words()).expect("c");
        let mut tokens = ctx.stream.alloc_zeros::<u32>(m).expect("tokens");
        let mut margins = ctx.stream.alloc_zeros::<f32>(m).expect("margins");
        let ops = GemmOperands {
            x: ctx.ptr(&dx),
            w: ctx.ptr(&dw),
            out: ctx.ptr(&out),
            ..Default::default()
        };
        let mut time = |f: &mut dyn FnMut()| -> f64 {
            let reps = 40;
            let mut total = 0.0;
            for _ in 0..reps + 5 {
                ctx.stream.memset_zeros(&mut flush_mut).expect("flush");
                ctx.stream.synchronize().expect("sync");
                let t = std::time::Instant::now();
                f();
                ctx.stream.synchronize().expect("sync");
                total += t.elapsed().as_secs_f64();
            }
            // Includes five warm-up repetitions; report the mean of all as microseconds.
            total / f64::from(reps + 5) * 1e6
        };
        let mut old = || {
            engine.gemm(&problem(m, n, k), &ops).expect("gemm");
            sampler
                .choose_bf16(
                    &ctx.stream,
                    true,
                    &out,
                    m,
                    n,
                    &d_params,
                    &prompt,
                    &output,
                    &counts,
                    masks.words(),
                    &mut tokens,
                    &mut margins,
                )
                .expect("choose");
        };
        let t_old = time(&mut old);
        let mut fused_tokens = ctx.stream.alloc_zeros::<u32>(m).expect("tokens");
        let mut new = || {
            engine
                .argmax_rows(m, n, k, ctx.ptr(&dx), ctx.ptr(&dw), ctx.ptr(&fused_tokens))
                .expect("fused");
        };
        let t_new = time(&mut new);
        let _ = &mut fused_tokens;
        println!(
            "head m={m}: logits+sampler {t_old:.1} us, fused argmax {t_new:.1} us (wall, incl. launch and sync)"
        );
    }
}

/// The wide kernel agrees with the narrow kernels on the first 64 rows of a larger batch for
/// operands carrying special values: NaN and infinite weights, a column of exact zeros, rows
/// that are entirely -inf or NaN, and ties planted across tiles.
#[test]
#[ignore = "needs GPU"]
fn wide_head_argmax_matches_narrow_kernel_on_special_values() {
    let mut ctx = Ctx::new();
    let mut engine = ctx.engine();
    engine.set_wide_argmax_enabled(true);
    let (n, k) = (59392usize, 1536usize);
    let mut rng = XorShift(0xa11_5eed);
    let mut w = random_bf16(&mut rng, n * k, 0.5);
    for j in 0..k {
        w[3 * k + j] = bf16::from_f32(f32::INFINITY);
        w[40_000 * k + j] = bf16::from_f32(f32::INFINITY);
        w[777 * k + j] = bf16::from_f32(0.0);
    }
    w[5 * k + 17] = bf16::from_f32(f32::NAN);
    w[59_000 * k + 4] = bf16::from_f32(f32::NEG_INFINITY);
    let mut x = random_bf16(&mut rng, 128 * k, 1.0);
    // Row 10 is all -inf (its argmax resolves to column 0 or 3 by the tie rule), row 11 NaN.
    for j in 0..k {
        x[10 * k + j] = bf16::from_f32(f32::NEG_INFINITY);
        x[11 * k + j] = bf16::from_f32(f32::NAN);
    }
    let dw = ctx.upload(&w);
    let dx = ctx.upload(&x);
    let mut narrow = ctx.stream.alloc_zeros::<u32>(64).expect("tokens");
    engine
        .argmax_rows(64, n, k, ctx.ptr(&dx), ctx.ptr(&dw), ctx.ptr(&narrow))
        .expect("narrow");
    ctx.stream.synchronize().expect("sync");
    let want = ctx.stream.clone_dtoh(&narrow).expect("dtoh");
    let _ = &mut narrow;
    for m in [65usize, 96, 128] {
        let got_dev = ctx.stream.alloc_zeros::<u32>(m).expect("tokens");
        for _ in 0..2 {
            engine
                .argmax_rows(m, n, k, ctx.ptr(&dx), ctx.ptr(&dw), ctx.ptr(&got_dev))
                .expect("wide");
            ctx.stream.synchronize().expect("sync");
            let got = ctx.stream.clone_dtoh(&got_dev).expect("dtoh");
            assert_eq!(&got[..64], &want[..], "m={m}: first 64 rows");
        }
    }
}

/// Runs the wide kernel on `m` rows built from the 64 base rows of `x64` (rows past 64 repeat
/// the base rows in a permuted order) and asserts every row's token equals the narrow kernel's
/// token for the matching base row, which also checks the cross-tile maximum merge on rows the
/// narrow kernel cannot hold. Returns the narrow tokens of the base rows.
fn wide_rows_match_narrow(
    ctx: &Ctx,
    engine: &mut GemmEngine,
    w: &[bf16],
    x64: &[bf16],
    n: usize,
    k: usize,
    label: &str,
) -> Vec<u32> {
    let dw = ctx.upload(w);
    let narrow = ctx.stream.alloc_zeros::<u32>(64).expect("tokens");
    let dx64 = ctx.upload(x64);
    engine
        .argmax_rows(64, n, k, ctx.ptr(&dx64), ctx.ptr(&dw), ctx.ptr(&narrow))
        .expect("narrow");
    ctx.stream.synchronize().expect("sync");
    let want = ctx.stream.clone_dtoh(&narrow).expect("dtoh");
    for m in [65usize, 96, 128] {
        let source = |r: usize| if r < 64 { r } else { (r * 7 + 3) % 64 };
        let mut x = Vec::with_capacity(m * k);
        for r in 0..m {
            x.extend_from_slice(&x64[source(r) * k..(source(r) + 1) * k]);
        }
        let dx = ctx.upload(&x);
        let got_dev = ctx.stream.alloc_zeros::<u32>(m).expect("tokens");
        for _ in 0..2 {
            engine
                .argmax_rows(m, n, k, ctx.ptr(&dx), ctx.ptr(&dw), ctx.ptr(&got_dev))
                .expect("wide");
            ctx.stream.synchronize().expect("sync");
            let got = ctx.stream.clone_dtoh(&got_dev).expect("dtoh");
            for r in 0..m {
                assert_eq!(got[r], want[source(r)], "{label}: m={m} row {r}");
            }
        }
    }
    want
}

/// The wide kernel's output merge on every row up to 128: ties across tiles resolve to the
/// lowest index, all-zero and negative-zero rows tie across the whole vocabulary, and NaN and
/// infinite operands agree with the narrow kernel row for row.
#[test]
#[ignore = "needs GPU"]
fn wide_head_argmax_merge_ties_nan_inf_and_negative_zero_all_rows() {
    let mut ctx = Ctx::new();
    let mut engine = ctx.engine();
    engine.set_wide_argmax_enabled(true);
    let (n, k) = (59392usize, 1536usize);
    let bf = bf16::from_f32;

    // Finite weights: equal dominant columns in different tiles, a negative-zero column and a
    // zero column.
    let mut rng = XorShift(0x7_1e5);
    let mut w = random_bf16(&mut rng, n * k, 0.5);
    let tied = [100usize, 4097, 30_000, 59_391];
    for &c in &tied {
        for j in 0..k {
            w[c * k + j] = bf(1.0);
        }
    }
    for j in 0..k {
        w[200 * k + j] = bf(-0.0);
        w[201 * k + j] = bf(0.0);
    }
    let mut x = random_bf16(&mut rng, 64 * k, 1.0);
    for r in 0..64 {
        for j in 0..k {
            let v = x[r * k + j].to_f32();
            match r % 4 {
                // Positive activations: the tied columns are the maxima.
                0 => x[r * k + j] = bf(0.5 + v.abs()),
                // Zero and negative-zero activations: every logit is +0 or -0.
                1 => x[r * k + j] = bf(0.0),
                2 => x[r * k + j] = bf(-0.0),
                // Negative activations: the tied columns are the minima.
                _ => x[r * k + j] = bf(-0.5 - v.abs()),
            }
        }
    }
    let want = wide_rows_match_narrow(&ctx, &mut engine, &w, &x, n, k, "finite");
    for (r, &token) in want.iter().enumerate() {
        match r % 4 {
            0 => assert_eq!(token, tied[0] as u32, "row {r}: tie takes the lowest index"),
            1 | 2 => assert_eq!(token, 0, "row {r}: all-zero logits take index 0"),
            _ => {}
        }
    }

    // Non-finite operands: infinite ties across tiles, NaN weights and activations, rows that
    // are all -inf or all NaN.
    let mut w = random_bf16(&mut rng, n * k, 0.5);
    for j in 0..k {
        w[3 * k + j] = bf(f32::INFINITY);
        w[40_000 * k + j] = bf(f32::INFINITY);
        w[777 * k + j] = bf(-0.0);
    }
    w[5 * k + 17] = bf(f32::NAN);
    w[59_000 * k + 4] = bf(f32::NEG_INFINITY);
    let mut x = random_bf16(&mut rng, 64 * k, 1.0);
    for j in 0..k {
        x[10 * k + j] = bf(f32::NEG_INFINITY);
        x[11 * k + j] = bf(f32::NAN);
        x[12 * k + j] = bf(1.0);
        x[13 * k + j] = bf(0.0);
        x[14 * k + j] = bf(-0.0);
    }
    x[20 * k + 99] = bf(f32::NAN);
    wide_rows_match_narrow(&ctx, &mut engine, &w, &x, n, k, "non-finite");
    println!("wide merge verified on rows 0..128 with ties, zeros, negative zeros, NaN and inf");
}

/// Resource receipt of the wide kernel: registers, spills and shared memory as loaded.
#[test]
#[ignore = "needs GPU"]
fn wide_head_argmax_resource_receipt() {
    let mut ctx = Ctx::new();
    let skinny = scriptor_kernels::skinny_gemm::SkinnyGemm::new(ctx.dev.context(), &ctx.stream)
        .expect("skinny");
    let receipt = skinny.wide_argmax_receipt().expect("receipt");
    println!("wide argmax receipt: {receipt:?}");
    assert_eq!(receipt.local_bytes, 0, "no spills or stack frame");
    assert!(
        receipt.dynamic_shared_bytes as usize + receipt.static_shared_bytes as usize <= 101_376
    );
    let _ = &mut ctx;
}

/// Total head-plus-selection time at 65..=128 rows with L2 flushed before each launch:
/// logits plus sampler, the wide fused kernel, and the same product split into a 64-row narrow
/// launch plus a remainder launch (which reads the head weights twice).
#[test]
#[ignore = "needs GPU"]
fn wide_head_argmax_timing() {
    use scriptor_kernels::sampling::{RowParams, SamplingKernels, TokenMasks};
    let mut ctx = Ctx::new();
    let mut engine = ctx.engine();
    engine.set_wide_argmax_enabled(true);
    let sampler = SamplingKernels::load(ctx.dev.context()).expect("sampler");
    let (n, k) = (59392usize, 1536usize);
    let mut rng = XorShift(0x71_5e);
    let dw = ctx.upload(&random_bf16(&mut rng, n * k, 0.5));
    let mut flush = ctx.stream.alloc_zeros::<u8>(256 << 20).expect("flush");
    for m in [65usize, 80, 96, 112, 128] {
        let dx = ctx.upload(&random_bf16(&mut rng, m * k, 1.0));
        let out = ctx.stream.alloc_zeros::<u16>(m * n).expect("out");
        let masks = TokenMasks::new(m, n);
        let params: Vec<RowParams> = (0..m as u32).map(RowParams::greedy).collect();
        let d_params = ctx.stream.clone_htod(&params).expect("params");
        let prompt = ctx.stream.clone_htod(masks.prompt_words()).expect("p");
        let output = ctx.stream.clone_htod(masks.output_words()).expect("o");
        let counts = ctx.stream.clone_htod(masks.count_words()).expect("c");
        let mut tokens = ctx.stream.alloc_zeros::<u32>(m).expect("tokens");
        let mut margins = ctx.stream.alloc_zeros::<f32>(m).expect("margins");
        let ops = GemmOperands {
            x: ctx.ptr(&dx),
            w: ctx.ptr(&dw),
            out: ctx.ptr(&out),
            ..Default::default()
        };
        let fused = ctx.stream.alloc_zeros::<u32>(m).expect("tokens");
        let x_tail = ctx.ptr(&dx) + (64 * k * 2) as u64;
        let tail_tokens = ctx.stream.alloc_zeros::<u32>(m - 64).expect("tokens");
        let mut time = |f: &mut dyn FnMut()| -> f64 {
            let mut samples = Vec::new();
            for i in 0..45 {
                ctx.stream.memset_zeros(&mut flush).expect("flush");
                ctx.stream.synchronize().expect("sync");
                let t = std::time::Instant::now();
                f();
                ctx.stream.synchronize().expect("sync");
                if i >= 5 {
                    samples.push(t.elapsed().as_secs_f64() * 1e6);
                }
            }
            median(&mut samples)
        };
        let mut old = || {
            engine.gemm(&problem(m, n, k), &ops).expect("gemm");
            sampler
                .choose_bf16(
                    &ctx.stream,
                    true,
                    &out,
                    m,
                    n,
                    &d_params,
                    &prompt,
                    &output,
                    &counts,
                    masks.words(),
                    &mut tokens,
                    &mut margins,
                )
                .expect("choose");
        };
        let t_old = time(&mut old);
        let mut wide_launch = || {
            engine
                .argmax_rows(m, n, k, ctx.ptr(&dx), ctx.ptr(&dw), ctx.ptr(&fused))
                .expect("wide");
        };
        let t_wide = time(&mut wide_launch);
        let mut split = || {
            engine
                .argmax_rows(64, n, k, ctx.ptr(&dx), ctx.ptr(&dw), ctx.ptr(&fused))
                .expect("head");
            engine
                .argmax_rows(m - 64, n, k, x_tail, ctx.ptr(&dw), ctx.ptr(&tail_tokens))
                .expect("tail");
        };
        let t_split = time(&mut split);
        println!(
            "head m={m}: logits+sampler {t_old:.1} us, wide fused {t_wide:.1} us, two-launch split {t_split:.1} us (wall, flushed L2, median)"
        );
    }
}

/// Token counts at which a probe row is placed in the row-invariance checks.
const PROBE_MS: [usize; 16] = [
    1, 4, 16, 20, 32, 33, 40, 48, 64, 65, 80, 96, 100, 112, 120, 128,
];

/// Output bits of row `at` of `[m, k] x [n, k]^T`, where row `at` of the activations is `probe`
/// and every other row is drawn from `rng`.
#[allow(clippy::too_many_arguments)]
fn probe_output(
    ctx: &Ctx,
    engine: &mut GemmEngine,
    weights: &CudaSlice<u16>,
    shape: (usize, usize),
    m: usize,
    probe: &[bf16],
    at: usize,
    rng: &mut XorShift,
) -> Vec<u16> {
    let (_, k) = shape;
    let mut x = random_bf16(rng, m * k, 1.0);
    x[at * k..(at + 1) * k].copy_from_slice(probe);
    gemm_row(ctx, engine, weights, shape, m, &x, at)
}

/// How the rows other than the probe row are filled in [`probe_output_fixed`].
#[derive(Clone, Copy, Debug)]
enum Others {
    /// Every other row is zero.
    Zero,
    /// Every other row equals the probe row.
    Probe,
}

/// Output bits of row `at` of `[m, k] x [n, k]^T`, where row `at` of the activations is `probe`
/// and every other row is filled as `others` says.
#[allow(clippy::too_many_arguments)]
fn probe_output_fixed(
    ctx: &Ctx,
    engine: &mut GemmEngine,
    weights: &CudaSlice<u16>,
    shape: (usize, usize),
    m: usize,
    probe: &[bf16],
    at: usize,
    others: Others,
) -> Vec<u16> {
    let (_, k) = shape;
    let mut x = vec![bf16::ZERO; m * k];
    if let Others::Probe = others {
        for row in x.chunks_mut(k) {
            row.copy_from_slice(probe);
        }
    }
    x[at * k..(at + 1) * k].copy_from_slice(probe);
    gemm_row(ctx, engine, weights, shape, m, &x, at)
}

/// Output bits of row `at` of `[m, k] x [n, k]^T` for the activations `x`.
fn gemm_row(
    ctx: &Ctx,
    engine: &mut GemmEngine,
    weights: &CudaSlice<u16>,
    shape: (usize, usize),
    m: usize,
    x: &[bf16],
    at: usize,
) -> Vec<u16> {
    let (n, k) = shape;
    let dx = ctx.upload(x);
    let out = ctx.stream.alloc_zeros::<u16>(m * n).expect("out");
    let ops = GemmOperands {
        x: ctx.ptr(&dx),
        w: ctx.ptr(weights),
        out: ctx.ptr(&out),
        ..Default::default()
    };
    engine.gemm(&problem(m, n, k), &ops).expect("gemm");
    ctx.download(&out)[at * n..(at + 1) * n]
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// The output row of one activation row is bit-identical whatever the token count of the call
/// and the row's position in it, across the skinny kernel's token-group instantiations and
/// every served weight shape.
#[test]
#[ignore = "needs GPU"]
fn skinny_gemm_row_invariant_to_m_and_slot() {
    let started = std::time::Instant::now();
    let mut ctx = Ctx::new();
    let mut engine = ctx.engine();
    engine.set_skinny_enabled(true);
    for shape in SHAPES {
        let (n, k) = (shape.n, shape.k);
        let mut rng = XorShift(0x0a11_0000 ^ ((n as u64) << 20) ^ k as u64);
        let dw = ctx.upload(&random_bf16(&mut rng, n * k, 0.5));
        let probe = random_bf16(&mut rng, k, 1.0);
        let want = probe_output(&ctx, &mut engine, &dw, (n, k), 128, &probe, 0, &mut rng);
        for m in PROBE_MS {
            for at in [0, m / 2, m - 1] {
                let got = probe_output(&ctx, &mut engine, &dw, (n, k), m, &probe, at, &mut rng);
                assert!(
                    got == want,
                    "n={n} k={k} m={m} row={at}: the probe row differs from row 0 of the m=128 call"
                );
            }
        }
        eprintln!(
            "phase skinny_row_invariance_n{n}_k{k} elapsed_s={:.3}",
            started.elapsed().as_secs_f64()
        );
    }
}

/// Row positions at which a probe row is placed in the m=160 check.
const M160_ROWS: [usize; 10] = [0, 1, 31, 32, 63, 64, 80, 127, 128, 159];

/// Random contents of the other rows tried at each shape of the m=160 check.
const M160_RANDOM_CASES: usize = 100;

/// Token counts whose cuBLASLt candidate selection the m=160 check prints.
const M160_SELECTION_MS: [usize; 4] = [159, 160, 161, 192];

/// At m=160 (a draft depth of 4 over a 32-slot bucket), where the cuBLASLt route serves every
/// shape, the output row of one activation row is bit-identical across 100 random contents of
/// the other rows, all-zero and all-equal-to-the-probe other rows, and positions of the row.
/// The tuned candidate index and identity at neighbouring token counts are printed, not
/// asserted.
#[test]
#[ignore = "needs GPU"]
fn nvjet_row_invariant_to_acceptance_pattern_m160() {
    let started = std::time::Instant::now();
    let mut ctx = Ctx::new();
    let mut engine = ctx.engine();
    engine.set_skinny_enabled(true);
    for shape in SHAPES {
        let (n, k, m) = (shape.n, shape.k, 160);
        assert!(
            !scriptor_kernels::skinny_gemm::supports(m, n, k),
            "m={m} n={n} k={k} must route to cuBLASLt"
        );
        let mut rng = XorShift(0x0160_0000 ^ ((n as u64) << 20) ^ k as u64);
        let dw = ctx.upload(&random_bf16(&mut rng, n * k, 0.5));
        let probe = random_bf16(&mut rng, k, 1.0);
        let want = probe_output(&ctx, &mut engine, &dw, (n, k), m, &probe, 0, &mut rng);
        for case in 0..M160_RANDOM_CASES {
            let at = M160_ROWS[case % M160_ROWS.len()];
            let got = probe_output(&ctx, &mut engine, &dw, (n, k), m, &probe, at, &mut rng);
            assert!(
                got == want,
                "n={n} k={k} case={case} row={at}: the probe row differs from its first output"
            );
        }
        for others in [Others::Zero, Others::Probe] {
            for at in M160_ROWS {
                let got = probe_output_fixed(&ctx, &mut engine, &dw, (n, k), m, &probe, at, others);
                assert!(
                    got == want,
                    "n={n} k={k} others={others:?} row={at}: the probe row differs from its first output"
                );
            }
        }
        for sel_m in M160_SELECTION_MS {
            let p = problem(sel_m, n, k);
            match engine.tuned_index(&p) {
                Ok(index) => match engine.candidate_identity(&p, index) {
                    Ok(id) => eprintln!(
                        "selection n={n} k={k} m={sel_m} tuned_index={index} {}",
                        id.summary()
                    ),
                    Err(e) => eprintln!(
                        "selection n={n} k={k} m={sel_m} tuned_index={index} identity unavailable: {e:?}"
                    ),
                },
                Err(e) => eprintln!("selection n={n} k={k} m={sel_m} unavailable: {e:?}"),
            }
        }
        eprintln!(
            "phase nvjet_row_invariance_n{n}_k{k} elapsed_s={:.3}",
            started.elapsed().as_secs_f64()
        );
    }
}
