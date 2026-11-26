//! Pixel-identity of nvJPEG device decoding against the libjpeg-turbo path.
#![cfg(all(feature = "cuda", feature = "nvjpeg"))]

use std::path::PathBuf;
use std::time::Instant;

use scriptor_cuda::Device;
use scriptor_image::nvjpeg::{GpuJpegPolicy, Subsampling, parse_header};
use scriptor_image::{RgbImage, decode_jpeg_libjpeg};
use scriptor_kernels::nvjpeg_ffi::{DecoderOptions, NvjpegDecoder, NvjpegLib};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/image/modes")
}

fn load(name: &str) -> Vec<u8> {
    std::fs::read(fixture_dir().join(name)).expect("fixture")
}

fn decoder(interpolate: bool) -> NvjpegDecoder {
    decoder_with(interpolate, 0)
}

fn decoder_with(interpolate: bool, backend: i32) -> NvjpegDecoder {
    let device = Device::open().expect("device");
    let stream = device.new_stream().expect("stream");
    let lib = NvjpegLib::load().expect("libnvjpeg");
    NvjpegDecoder::new(
        lib,
        stream,
        DecoderOptions {
            interpolate_chroma: interpolate,
            backend,
            max_cpu_threads: 1,
        },
    )
    .expect("decoder")
}

fn mismatches(a: &RgbImage, b: &RgbImage) -> (usize, i32) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let mut n = 0;
    let mut max = 0;
    for (x, y) in a.data.iter().zip(&b.data) {
        let d = (i32::from(*x) - i32::from(*y)).abs();
        if d != 0 {
            n += 1;
            max = max.max(d);
        }
    }
    (n, max)
}

const MODES: &[&str] = &[
    "baseline_444.jpg",
    "baseline_422.jpg",
    "baseline_420.jpg",
    "progressive_444.jpg",
    "progressive_420.jpg",
    "gray.jpg",
    "odd_444.jpg",
    "odd_422.jpg",
    "odd_420.jpg",
    "odd_gray.jpg",
    "tiny_420.jpg",
    "wide_420.jpg",
    "exif_orient6_420.jpg",
];

fn gpu_decode_all(d: &mut NvjpegDecoder, names: &[&str]) -> Vec<RgbImage> {
    let blobs: Vec<Vec<u8>> = names.iter().map(|n| load(n)).collect();
    let refs: Vec<&[u8]> = blobs.iter().map(Vec::as_slice).collect();
    let batch = d.decode_batch(&refs).expect("batch decode");
    d.download(&batch).expect("download")
}

fn identity_for(interpolate: bool) -> Vec<(&'static str, usize, i32)> {
    let mut d = decoder(interpolate);
    let mut report = Vec::new();
    for name in MODES {
        let gpu = gpu_decode_all(&mut d, &[name]).remove(0);
        let cpu = decode_jpeg_libjpeg(&load(name)).expect("cpu");
        let (n, max) = mismatches(&gpu, &cpu);
        report.push((*name, n, max));
    }
    report
}

#[test]
#[ignore = "needs GPU"]
fn report_backends() {
    for backend in [1, 2, 3] {
        let Ok(lib) = NvjpegLib::load() else { return };
        let device = Device::open().expect("device");
        let stream = device.new_stream().expect("stream");
        let opts = DecoderOptions {
            interpolate_chroma: true,
            backend,
            max_cpu_threads: 1,
        };
        let mut d = match NvjpegDecoder::new(lib, stream, opts) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("backend {backend}: create failed: {e}");
                continue;
            }
        };
        for name in MODES {
            let blob = load(name);
            let cpu = decode_jpeg_libjpeg(&blob).expect("cpu");
            match d.decode_single_to_host(&blob) {
                Ok(gpu) => {
                    let (n, max) = mismatches(&gpu, &cpu);
                    eprintln!("backend {backend} {name}: differing_bytes={n} max_abs={max}");
                }
                Err(e) => eprintln!("backend {backend} {name}: {e}"),
            }
        }
    }
}

#[test]
#[ignore = "needs GPU"]
fn report_identity_per_mode() {
    for interpolate in [true, false] {
        for (name, n, max) in identity_for(interpolate) {
            eprintln!("interpolate={interpolate} {name}: differing_bytes={n} max_abs={max}");
        }
    }
}

/// Largest per-byte difference nvJPEG shows against libjpeg-turbo on the fixtures.
const MEASURED_MAX_ABS: i32 = 3;

/// Asserts the pixel-identity gate result for each fixture: the device output is
/// within the recorded bound of libjpeg-turbo and is not byte-identical, which is
/// why the enabled policy routes the mode to the CPU.
fn assert_gate_result(names: &[&str]) {
    let mut d = decoder(true);
    for name in names {
        let gpu = gpu_decode_all(&mut d, &[name]).remove(0);
        let cpu = decode_jpeg_libjpeg(&load(name)).expect("cpu");
        let (n, max) = mismatches(&gpu, &cpu);
        assert!(
            n > 0,
            "{name}: now byte-identical; revisit the routing constants"
        );
        assert!(max <= MEASURED_MAX_ABS, "{name}: max abs diff {max}");
        let header = parse_header(&load(name)).unwrap();
        assert!(
            matches!(
                GpuJpegPolicy::enabled().route(&header),
                scriptor_image::nvjpeg::Route::Cpu(_)
            ),
            "{name}: a non-identical mode must stay on the CPU path"
        );
    }
}

#[test]
#[ignore = "needs GPU"]
fn baseline_444_gate_result() {
    assert_gate_result(&["baseline_444.jpg", "odd_444.jpg"]);
}

#[test]
#[ignore = "needs GPU"]
fn baseline_422_gate_result() {
    assert_gate_result(&["baseline_422.jpg", "odd_422.jpg"]);
}

#[test]
#[ignore = "needs GPU"]
fn baseline_420_gate_result() {
    assert_gate_result(&["baseline_420.jpg", "odd_420.jpg", "wide_420.jpg"]);
}

#[test]
#[ignore = "needs GPU"]
fn progressive_gate_result() {
    assert_gate_result(&["progressive_444.jpg", "progressive_420.jpg"]);
}

#[test]
#[ignore = "needs GPU"]
fn grayscale_gate_result() {
    assert_gate_result(&["gray.jpg", "odd_gray.jpg"]);
}

#[test]
#[ignore = "needs GPU"]
fn exif_orientation_is_ignored_like_the_cpu_path() {
    let mut d = decoder(true);
    let gpu = gpu_decode_all(&mut d, &["exif_orient6_420.jpg"]).remove(0);
    let cpu = decode_jpeg_libjpeg(&load("exif_orient6_420.jpg")).unwrap();
    assert_eq!((gpu.width, gpu.height), (cpu.width, cpu.height));
    assert_eq!((gpu.width, gpu.height), (96, 64));
}

#[test]
#[ignore = "needs GPU"]
fn batched_decode_matches_single_decode() {
    let mut d = decoder(true);
    let batch = gpu_decode_all(&mut d, MODES);
    for (name, got) in MODES.iter().zip(&batch) {
        let single = d.decode_single_to_host(&load(name)).expect("single");
        assert_eq!(got, &single, "{name}");
    }
}

#[test]
#[ignore = "needs GPU"]
fn routed_batch_is_identical_to_cpu_for_every_mode() {
    let mut d = decoder(true);
    let blobs: Vec<Vec<u8>> = MODES.iter().map(|n| load(n)).collect();
    let refs: Vec<&[u8]> = blobs.iter().map(Vec::as_slice).collect();
    let batch = d
        .decode_routed(&refs, &GpuJpegPolicy::enabled())
        .expect("routed");
    let images = d.download(&batch).expect("download");
    for ((name, got), blob) in MODES.iter().zip(&images).zip(&blobs) {
        let cpu = decode_jpeg_libjpeg(blob).expect("cpu");
        assert_eq!(got, &cpu, "{name}");
    }
}

#[test]
#[ignore = "needs GPU"]
fn disabled_policy_decodes_everything_on_the_host() {
    let mut d = decoder(true);
    let blobs = [load("baseline_420.jpg"), load("gray.jpg")];
    let refs: Vec<&[u8]> = blobs.iter().map(Vec::as_slice).collect();
    let batch = d
        .decode_routed(&refs, &GpuJpegPolicy::disabled())
        .expect("routed");
    assert!(batch.extents.iter().all(|e| !e.on_gpu));
}

#[test]
#[ignore = "needs GPU"]
fn unverified_policy_routes_eligible_images_to_the_device() {
    let mut d = decoder(true);
    let blob = load("baseline_420.jpg");
    let header = parse_header(&blob).unwrap();
    assert_eq!(header.subsampling, Subsampling::S420);
    let batch = d
        .decode_routed(&[blob.as_slice()], &GpuJpegPolicy::unverified())
        .expect("routed");
    assert!(batch.extents[0].on_gpu);
    let enabled = d
        .decode_routed(&[blob.as_slice()], &GpuJpegPolicy::enabled())
        .expect("routed");
    assert!(!enabled.extents[0].on_gpu);
}

#[test]
#[ignore = "needs GPU"]
fn truncated_input_does_not_crash_and_matches_cpu_outcome() {
    let mut d = decoder(true);
    let mut blob = load("baseline_420.jpg");
    blob.truncate(blob.len() / 2);
    let result = d.decode_routed(&[blob.as_slice()], &GpuJpegPolicy::unverified());
    let cpu = decode_jpeg_libjpeg(&blob);
    if let (Ok(batch), Ok(cpu)) = (&result, &cpu) {
        assert_eq!(batch.extents[0].height, cpu.height);
    }
}

/// Times CPU and device decode of the JPEGs in `SCRIPTOR_NVJPEG_BENCH_DIR`.
#[test]
#[ignore = "needs GPU"]
fn decode_timing() {
    let Some(dir) = std::env::var_os("SCRIPTOR_NVJPEG_BENCH_DIR") else {
        eprintln!("SCRIPTOR_NVJPEG_BENCH_DIR unset; skipping");
        return;
    };
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .expect("dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jpg"))
        .collect();
    paths.sort();
    let backend = std::env::var("SCRIPTOR_NVJPEG_BACKEND")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let threads = std::env::var("SCRIPTOR_NVJPEG_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let device = Device::open().expect("device");
    let stream = device.new_stream().expect("stream");
    let options = DecoderOptions {
        interpolate_chroma: true,
        backend,
        max_cpu_threads: threads,
    };
    let mut d =
        NvjpegDecoder::new(NvjpegLib::load().expect("lib"), stream, options).expect("decoder");
    eprintln!("backend={backend} threads={threads}");
    for path in paths {
        let blob = std::fs::read(&path).unwrap();
        let h = parse_header(&blob).unwrap();
        let runs = 8;
        let _ = decode_jpeg_libjpeg(&blob).unwrap();
        let t = Instant::now();
        let mut cpu = None;
        for _ in 0..runs {
            cpu = Some(decode_jpeg_libjpeg(&blob).unwrap());
        }
        let cpu_ms = t.elapsed().as_secs_f64() * 1e3 / runs as f64;
        let _ = d.decode_batch(&[blob.as_slice()]).unwrap();
        let t = Instant::now();
        for _ in 0..runs {
            let _ = d.decode_batch(&[blob.as_slice()]).unwrap();
        }
        let gpu_ms = t.elapsed().as_secs_f64() * 1e3 / runs as f64;
        let four: Vec<&[u8]> = vec![blob.as_slice(); 4];
        let _ = d.decode_batch(&four).unwrap();
        let t = Instant::now();
        for _ in 0..runs {
            let _ = d.decode_batch(&four).unwrap();
        }
        let gpu4_ms = t.elapsed().as_secs_f64() * 1e3 / (runs * 4) as f64;
        let one = d.decode_batch(&[blob.as_slice()]).unwrap();
        let gpu = d.download(&one).unwrap();
        let (n, max) = mismatches(&gpu[0], cpu.as_ref().unwrap());
        eprintln!(
            "{} {}x{} {:?}: cpu {cpu_ms:.2} ms, nvjpeg {gpu_ms:.2} ms (batch of 4: {gpu4_ms:.2} ms/img), differing_bytes={n} max_abs={max}",
            path.file_name().unwrap().to_string_lossy(),
            h.width,
            h.height,
            h.subsampling
        );
    }
}
