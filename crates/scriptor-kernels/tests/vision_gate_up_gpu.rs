//! GPU checks of the SiLU-fused gate/up projection against the vendor GEMM followed by
//! `silu_mul_fused`, plus its timing at the vision tower's shapes.
//! Run with `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda --release --test vision_gate_up_gpu -- --ignored --nocapture --test-threads=1`.
#![cfg(feature = "cuda")]

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use scriptor_cuda::Device;
use scriptor_kernels::activation::ActivationKernels;
use scriptor_kernels::cublaslt::{GemmEngine, GemmOperands, GemmOptions, GemmProblem};
use scriptor_kernels::heuristics::{Epilogue, GemmDtype, TuningTable};
use scriptor_kernels::vision::VisionKernels;
use std::sync::Arc;

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

struct Rig {
    stream: Arc<CudaStream>,
    gemm: GemmEngine,
    act: ActivationKernels,
    vision: VisionKernels,
}

fn rig() -> Rig {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let mut arena = dev.arena(128 << 20, &stream).expect("arena");
    let gemm = GemmEngine::new(
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
    Rig {
        act: ActivationKernels::load(dev.context()).expect("act"),
        vision: VisionKernels::load(dev.context()).expect("vision"),
        stream,
        gemm,
    }
}

impl Rig {
    fn ptr(&self, s: &CudaSlice<u16>) -> u64 {
        s.device_ptr(&self.stream).0
    }

    #[allow(clippy::too_many_arguments)]
    fn unfused(
        &mut self,
        x: &CudaSlice<u16>,
        w: &CudaSlice<u16>,
        bias: Option<&CudaSlice<u16>>,
        gu: &mut CudaSlice<u16>,
        out: &mut CudaSlice<u16>,
        rows: usize,
        inter: usize,
        k: usize,
    ) {
        let problem = GemmProblem {
            m: rows,
            n: 2 * inter,
            k,
            dtype: GemmDtype::Bf16,
            epilogue: if bias.is_some() {
                Epilogue::Bias
            } else {
                Epilogue::None
            },
        };
        let ops = GemmOperands {
            x: self.ptr(x),
            w: self.ptr(w),
            bias: bias.map_or(0, |b| self.ptr(b)),
            out: self.ptr(gu),
            ..GemmOperands::default()
        };
        self.gemm.gemm(&problem, &ops).expect("gemm");
        self.act
            .silu_mul_fused(&self.stream, gu, out, rows, inter)
            .expect("silu");
    }

    #[allow(clippy::too_many_arguments)]
    fn fused(
        &self,
        x: &CudaSlice<u16>,
        w: &CudaSlice<u16>,
        bias: Option<&CudaSlice<u16>>,
        out: &mut CudaSlice<u16>,
        rows: usize,
        inter: usize,
        k: usize,
    ) {
        self.vision
            .gate_up_silu(&self.stream, x, w, bias, out, rows, inter, k)
            .expect("fused");
    }
}

#[test]
#[ignore = "needs GPU"]
fn gate_up_silu_matches_vendor_pair() {
    check_matches(4096, 1024);
}

#[test]
#[ignore = "needs GPU"]
fn gate_up_silu_matches_vendor_pair_text_shapes() {
    check_matches(4608, 1536);
}

fn check_matches(inter: usize, k: usize) {
    let mut r = rig();
    for (rows, with_bias) in [
        (1usize, false),
        (127, true),
        (130, true),
        (515, true),
        (2540, true),
        (2540, false),
    ] {
        let mut rng = Rng(0x9a7e ^ rows as u64);
        let x = rng.bits(rows * k, 1.0);
        let w = rng.bits(2 * inter * k, 0.2);
        let bias = rng.bits(2 * inter, 0.5);
        let db = with_bias.then(|| r.stream.clone_htod(&bias).unwrap());
        let (dx, dw) = (
            r.stream.clone_htod(&x).unwrap(),
            r.stream.clone_htod(&w).unwrap(),
        );
        let mut gu = r.stream.alloc_zeros::<u16>(rows * 2 * inter).unwrap();
        let mut want = r.stream.alloc_zeros::<u16>(rows * inter).unwrap();
        let mut got = r.stream.alloc_zeros::<u16>(rows * inter).unwrap();
        r.unfused(&dx, &dw, db.as_ref(), &mut gu, &mut want, rows, inter, k);
        r.fused(&dx, &dw, db.as_ref(), &mut got, rows, inter, k);
        let (want, got) = (
            r.stream.clone_dtoh(&want).unwrap(),
            r.stream.clone_dtoh(&got).unwrap(),
        );
        let (mut max, mut sum, mut rms_num, mut rms_den) = (0f32, 0f64, 0f64, 0f64);
        for (g, w) in got.iter().zip(&want) {
            let d = (f(*g) - f(*w)).abs();
            max = max.max(d);
            sum += d as f64;
            rms_num += (d * d) as f64;
            rms_den += (f(*w) * f(*w)) as f64;
        }
        let rel = (rms_num / rms_den.max(1e-20)).sqrt();
        println!(
            "rows {rows} bias {with_bias}: max abs {max:.5} mean abs {:.6} rel rms {rel:.5}",
            sum / got.len() as f64
        );
        assert!(rel < 0.02, "rows {rows}: rel rms {rel}");
    }
}

#[test]
#[ignore = "needs GPU"]
fn gate_up_small_tile_matches_large_tile_bitwise() {
    let r = rig();
    for (inter, k, rows) in [
        (4096usize, 1024usize, 2540usize),
        (4608, 1536, 515),
        (4096, 1024, 130),
    ] {
        let mut rng = Rng(11 ^ rows as u64);
        let x = r.stream.clone_htod(&rng.bits(rows * k, 1.0)).unwrap();
        let w = r.stream.clone_htod(&rng.bits(2 * inter * k, 0.2)).unwrap();
        let bias = r.stream.clone_htod(&rng.bits(2 * inter, 0.5)).unwrap();
        let mut a = r.stream.alloc_zeros::<u16>(rows * inter).unwrap();
        let mut b = r.stream.alloc_zeros::<u16>(rows * inter).unwrap();
        for (small, out) in [(false, &mut a), (true, &mut b)] {
            r.vision
                .gate_up_silu_with(&r.stream, small, &x, &w, Some(&bias), out, rows, inter, k)
                .unwrap();
        }
        let (a, b) = (
            r.stream.clone_dtoh(&a).unwrap(),
            r.stream.clone_dtoh(&b).unwrap(),
        );
        assert_eq!(a, b, "rows {rows}: small tile differs from large tile");
    }
}

#[test]
#[ignore = "needs GPU"]
fn gate_up_rule_path_output_equals_both_tiles_bitwise() {
    let r = rig();
    for (inter, k, rows) in [(4096usize, 1024usize, 256usize), (4608, 1536, 2540)] {
        let mut rng = Rng(23 ^ rows as u64);
        let x = r.stream.clone_htod(&rng.bits(rows * k, 1.0)).unwrap();
        let w = r.stream.clone_htod(&rng.bits(2 * inter * k, 0.2)).unwrap();
        let bias = r.stream.clone_htod(&rng.bits(2 * inter, 0.5)).unwrap();
        let mut outs = Vec::new();
        for small in [None, Some(false), Some(true)] {
            let mut out = r.stream.alloc_zeros::<u16>(rows * inter).unwrap();
            match small {
                None => {
                    r.vision
                        .gate_up_silu(&r.stream, &x, &w, Some(&bias), &mut out, rows, inter, k)
                }
                Some(small) => r.vision.gate_up_silu_with(
                    &r.stream,
                    small,
                    &x,
                    &w,
                    Some(&bias),
                    &mut out,
                    rows,
                    inter,
                    k,
                ),
            }
            .unwrap();
            outs.push(r.stream.clone_dtoh(&out).unwrap());
        }
        assert!(outs[0] == outs[1] && outs[0] == outs[2], "rows {rows}");
    }
}

#[test]
#[ignore = "needs GPU"]
fn gate_up_silu_timing() {
    time_pair(
        4096,
        1024,
        &[1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384],
    );
}

#[test]
#[ignore = "needs GPU"]
fn gate_up_silu_timing_text_shapes() {
    time_pair(4608, 1536, &[1280, 2540, 4096, 8192]);
}

fn time_pair(inter: usize, k: usize, row_counts: &[usize]) {
    let mut r = rig();
    for &rows in row_counts {
        let mut rng = Rng(7);
        let x = r.stream.clone_htod(&rng.bits(rows * k, 1.0)).unwrap();
        let w = r.stream.clone_htod(&rng.bits(2 * inter * k, 0.2)).unwrap();
        let mut gu = r.stream.alloc_zeros::<u16>(rows * 2 * inter).unwrap();
        let mut a = r.stream.alloc_zeros::<u16>(rows * inter).unwrap();
        let mut b = r.stream.alloc_zeros::<u16>(rows * inter).unwrap();
        let mut time = |label: &str, run: &mut dyn FnMut(&mut Rig)| {
            for _ in 0..3 {
                run(&mut r);
            }
            let mut ts = Vec::new();
            for _ in 0..8 {
                let s = r
                    .stream
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap();
                run(&mut r);
                let e = r
                    .stream
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap();
                e.synchronize().unwrap();
                ts.push(f64::from(s.elapsed_ms(&e).unwrap()) * 1e3);
            }
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let us = ts[ts.len() / 2];
            let tf = 2.0 * (rows * 2 * inter * k) as f64 / (us * 1e-6) / 1e12;
            println!("rows {rows} {label}: {us:.0} us ({tf:.0} TFLOP/s)");
        };
        time("vendor gemm + silu_mul", &mut |r| {
            r.unfused(&x, &w, None, &mut gu, &mut a, rows, inter, k)
        });
        time("fused gate/up", &mut |r| {
            r.fused(&x, &w, None, &mut b, rows, inter, k)
        });
        time("fused gate/up small tile", &mut |r| {
            r.vision
                .gate_up_silu_with(&r.stream, true, &x, &w, None, &mut b, rows, inter, k)
                .expect("small");
        });
    }
}
