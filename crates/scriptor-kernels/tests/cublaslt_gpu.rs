//! GPU checks of the cuBLASLt GEMM wrapper against the CPU reference: bf16
//! products with and without epilogues, the fp8 capability probe and fp8
//! product, and tuning-file discipline.
//! Run with `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda cublaslt -- --ignored --nocapture`.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use half::bf16;
use scriptor_cuda::{Device, DeviceArena};
use scriptor_kernels::cublaslt::{
    GemmEngine, GemmOperands, GemmOptions, GemmProblem, SyntheticOperands, XorShift, e4m3_decode,
    e4m3_encode,
};
use scriptor_kernels::heuristics::{Epilogue, GemmDtype, TuningTable, cache_file};
use scriptor_kernels::reference::gemm::gemm_bias;

fn gpu_enabled() -> bool {
    std::env::var("SCRIPTOR_GPU_TESTS").as_deref() == Ok("1")
        || std::env::var("QE_GPU_TESTS").as_deref() == Ok("1")
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
        let arena = dev.arena(256 << 20, &stream).expect("arena");
        Self { dev, stream, arena }
    }

    fn engine(&mut self, fp8: bool, tuning: TuningTable) -> GemmEngine {
        GemmEngine::new(
            self.dev.context(),
            &self.stream,
            &mut self.arena,
            tuning,
            GemmOptions {
                fp8,
                workspace_bytes: 32 << 20,
            },
        )
        .expect("gemm engine")
    }

    fn ptr<T>(&self, s: &CudaSlice<T>) -> u64 {
        s.device_ptr(&self.stream).0
    }

    fn read_bf16(&self, s: &CudaSlice<u16>) -> Vec<f32> {
        self.stream.synchronize().expect("sync");
        self.stream
            .clone_dtoh(s)
            .expect("dtoh")
            .into_iter()
            .map(|b| bf16::from_bits(b).to_f32())
            .collect()
    }
}

fn bf16_round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| bf16::from_f32(x).to_f32()).collect()
}

fn bf16_bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| bf16::from_f32(x).to_bits()).collect()
}

fn rel_rms(got: &[f32], want: &[f32]) -> f32 {
    let num: f32 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
    let den: f32 = want.iter().map(|w| w * w).sum();
    (num / den.max(1e-20)).sqrt()
}

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

fn random(rng: &mut XorShift, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.next_f32() * scale).collect()
}

const N: usize = 384;
const K: usize = 512;

fn run_bf16(ctx: &mut Ctx, m: usize, epilogue: Epilogue) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = XorShift(0x1234_5678_9abc_def1);
    let x = bf16_round(&random(&mut rng, m * K, 1.0));
    let w = bf16_round(&random(&mut rng, N * K, 0.5));
    let bias = bf16_round(&random(&mut rng, N, 1.0));
    let dx = ctx.stream.clone_htod(&bf16_bits(&x)).expect("x");
    let dw = ctx.stream.clone_htod(&bf16_bits(&w)).expect("w");
    let db = ctx.stream.clone_htod(&bf16_bits(&bias)).expect("bias");
    let out = ctx.stream.alloc_zeros::<u16>(m * N).expect("out");
    let mut engine = ctx.engine(false, TuningTable::default());
    let p = GemmProblem {
        m,
        n: N,
        k: K,
        dtype: GemmDtype::Bf16,
        epilogue,
    };
    let ops = GemmOperands {
        x: ctx.ptr(&dx),
        w: ctx.ptr(&dw),
        bias: ctx.ptr(&db),
        out: ctx.ptr(&out),
        ..Default::default()
    };
    engine.gemm(&p, &ops).expect("gemm");
    let got = ctx.read_bf16(&out);
    let want_bias = epilogue.has_bias().then_some(bias.as_slice());
    let want = gemm_bias(&x, &w, want_bias, m, N, K);
    (got, want, bias)
}

#[test]
#[ignore = "needs GPU"]
fn cublaslt_bf16_product_matches_reference() {
    let mut ctx = Ctx::new();
    for m in [32, 100, 1024] {
        let (got, want, _) = run_bf16(&mut ctx, m, Epilogue::None);
        let e = rel_rms(&got, &want);
        assert!(e < 1e-2, "m={m} rel_rms {e}");
    }
}

#[test]
#[ignore = "needs GPU"]
fn cublaslt_bf16_bias_epilogue_matches_reference() {
    let mut ctx = Ctx::new();
    for m in [32, 257] {
        let (got, want, _) = run_bf16(&mut ctx, m, Epilogue::Bias);
        let e = rel_rms(&got, &want);
        assert!(e < 1e-2, "m={m} rel_rms {e}");
    }
}

#[test]
#[ignore = "needs GPU"]
fn cublaslt_bf16_gelu_epilogues_match_tanh_reference() {
    let mut ctx = Ctx::new();
    for epilogue in [Epilogue::Gelu, Epilogue::BiasGelu] {
        let m = 128;
        let (got, plain, _) = run_bf16(&mut ctx, m, epilogue);
        let want: Vec<f32> = plain.iter().map(|&v| gelu_tanh(v)).collect();
        let e = rel_rms(&got, &want);
        assert!(e < 2e-2, "{epilogue:?} rel_rms {e}");
    }
}

#[test]
#[ignore = "needs GPU"]
fn cublaslt_fp8_probe_and_product() {
    let mut ctx = Ctx::new();
    let mut engine = ctx.engine(true, TuningTable::default());
    let cap = engine.fp8_capability().clone();
    println!("fp8 capability: {cap:?}");
    if !cap.supported {
        return;
    }
    let m = 256;
    let mut rng = XorShift(0xfeed_beef_0123_4567);
    let x = bf16_round(&random(&mut rng, m * K, 2.0));
    let w_scale = 0.02f32;
    let w_codes: Vec<u8> = random(&mut rng, N * K, 1.0)
        .iter()
        .map(|&v| e4m3_encode(v * 100.0))
        .collect();
    let w_deq: Vec<f32> = w_codes.iter().map(|&c| e4m3_decode(c) * w_scale).collect();

    let dx = ctx.stream.clone_htod(&bf16_bits(&x)).expect("x");
    let dxq = ctx.stream.alloc_zeros::<u8>(m * K).expect("xq");
    let dxs = ctx.stream.alloc_zeros::<f32>(m).expect("x scales");
    let dw = ctx.stream.clone_htod(&w_codes).expect("w");
    let dws = ctx.stream.alloc_zeros::<f32>(N).expect("w scales");
    let out = ctx.stream.alloc_zeros::<u16>(m * N).expect("out");
    engine
        .quantize_activations(ctx.ptr(&dx), ctx.ptr(&dxq), ctx.ptr(&dxs), m, K)
        .expect("quantize");
    engine.fill_f32(ctx.ptr(&dws), w_scale, N).expect("fill");

    ctx.stream.synchronize().expect("sync");
    let xq: Vec<u8> = ctx.stream.clone_dtoh(&dxq).expect("xq down");
    let xs: Vec<f32> = ctx.stream.clone_dtoh(&dxs).expect("xs down");
    let x_deq: Vec<f32> = (0..m * K).map(|i| e4m3_decode(xq[i]) * xs[i / K]).collect();
    assert!(
        rel_rms(&x_deq, &x) < 0.07,
        "activation quantisation error {}",
        rel_rms(&x_deq, &x)
    );

    let p = GemmProblem {
        m,
        n: N,
        k: K,
        dtype: GemmDtype::Fp8E4m3,
        epilogue: Epilogue::None,
    };
    let ops = GemmOperands {
        x: ctx.ptr(&dxq),
        w: ctx.ptr(&dw),
        out: ctx.ptr(&out),
        x_scale: ctx.ptr(&dxs),
        w_scale: ctx.ptr(&dws),
        ..Default::default()
    };
    engine.gemm(&p, &ops).expect("fp8 gemm");
    let got = ctx.read_bf16(&out);
    let want = gemm_bias(&x_deq, &w_deq, None, m, N, K);
    let e = rel_rms(&got, &want);
    assert!(e < 2e-2, "fp8 rel_rms {e}");
}

#[test]
#[ignore = "needs GPU"]
fn cublaslt_fp8_refused_unless_enabled() {
    let mut ctx = Ctx::new();
    let mut engine = ctx.engine(false, TuningTable::default());
    let p = GemmProblem {
        m: 64,
        n: 256,
        k: 256,
        dtype: GemmDtype::Fp8E4m3,
        epilogue: Epilogue::None,
    };
    assert!(engine.candidate_count(&p).is_err());
}

#[test]
#[ignore = "needs GPU"]
fn cublaslt_tuning_file_is_written_only_by_tune() {
    let mut ctx = Ctx::new();
    let dir = std::env::temp_dir().join(format!("scriptor-gemm-tune-{}", std::process::id()));
    let path = cache_file(&dir, &ctx.dev.info().name, "test");

    let mut engine = ctx.engine(false, TuningTable::default());
    let p = GemmProblem {
        m: 64,
        n: 256,
        k: 256,
        dtype: GemmDtype::Bf16,
        epilogue: Epilogue::None,
    };
    let operands = SyntheticOperands::new(&ctx.stream, &p, &engine).expect("operands");
    engine.gemm(&p, &operands.operands()).expect("gemm");
    assert!(!dir.exists(), "running a GEMM must not create a cache");

    let table = engine.tune_and_save(&[p], 3, &path).expect("tune");
    assert_eq!(table.len(), 1);
    let loaded = TuningTable::load(&path).expect("load");
    assert_eq!(loaded, table);
    assert!(loaded.get(&p.shape_key()).is_some());
    std::fs::remove_dir_all(&dir).ok();
}
