//! GPU checks of the preprocessing kernels against the CPU references.
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaStream;
use half::bf16;
use scriptor_cuda::Device;
use scriptor_image::{
    RgbImage, SmartResizeConfig, decode_jpeg, resize_bicubic_aa, smart_resize_with,
};
use scriptor_kernels::preprocess::{PatchifyBatch, PreprocessKernels, ResizeBatch};
use scriptor_kernels::reference::preprocess::{fused_affine, normalise_patchify, to_bf16};

fn fixture(name: &str) -> RgbImage {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/image")
        .join(name);
    let bytes = std::fs::read(path).expect("fixture");
    decode_jpeg(&bytes).expect("decode")
}

fn synthetic(height: u32, width: u32, seed: u32) -> RgbImage {
    let data = (0..height * width * 3)
        .map(|i| (i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 13) as u8)
        .collect();
    RgbImage {
        width,
        height,
        data,
    }
}

fn setup() -> (Device, Arc<CudaStream>, PreprocessKernels) {
    let device = Device::open().expect("device");
    let stream = device.new_stream().expect("stream");
    let kernels = PreprocessKernels::load(device.context()).expect("kernels");
    (device, stream, kernels)
}

fn run_patchify(images: &[&RgbImage]) -> Vec<u16> {
    let (_device, stream, kernels) = setup();
    let sizes: Vec<_> = images
        .iter()
        .map(|i| (i.height as usize, i.width as usize))
        .collect();
    let batch = PatchifyBatch::new(&sizes).unwrap();
    let rgb: Vec<u8> = images.iter().flat_map(|i| i.data.iter().copied()).collect();
    let d_rgb = stream.clone_htod(&rgb).unwrap();
    let d_desc = stream.clone_htod(&batch.words).unwrap();
    let mut d_out = stream.alloc_zeros::<u16>(batch.output_elems()).unwrap();
    kernels
        .normalise_patchify(&stream, &batch, &d_rgb, &d_desc, &mut d_out)
        .unwrap();
    stream.clone_dtoh(&d_out).unwrap()
}

fn reference_bits(image: &RgbImage) -> Vec<u16> {
    let (patches, _) = normalise_patchify(
        &image.data,
        image.height as usize,
        image.width as usize,
        &fused_affine(),
    );
    to_bf16(&patches).iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "needs GPU"]
fn patchify_is_bit_identical_on_fixture_jpegs() {
    let images = [
        fixture("w28_h28.jpg"),
        fixture("w140_h112.jpg"),
        fixture("w448_h420.jpg"),
    ];
    let refs: Vec<&RgbImage> = images.iter().collect();
    let got = run_patchify(&refs);
    let want: Vec<u16> = images.iter().flat_map(reference_bits).collect();
    assert_eq!(got.len(), want.len());
    let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert_eq!(bad, 0, "{bad} of {} bf16 values differ", want.len());
}

#[test]
#[ignore = "needs GPU"]
fn patchify_matches_image_crate_patchify_within_one_ulp() {
    let image = fixture("w448_h420.jpg");
    let got = run_patchify(&[&image]);
    let (_, want) = scriptor_image::patchify_bf16(&image);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let (g, w): (bf16, bf16) = (bf16::from_bits(*g), *w);
        let ulp = (i32::from(g.to_bits()) - i32::from(w.to_bits())).abs();
        assert!(ulp <= 1, "element {i}: {g} vs {w}");
    }
}

#[test]
#[ignore = "needs GPU"]
fn patchify_batches_images_of_different_sizes() {
    let a = synthetic(56, 112, 1);
    let b = synthetic(28, 28, 2);
    let c = synthetic(84, 56, 3);
    let got = run_patchify(&[&a, &b, &c]);
    let want: Vec<u16> = [&a, &b, &c].into_iter().flat_map(reference_bits).collect();
    assert_eq!(got, want);
}

fn run_resize(items: &[(&RgbImage, u32, u32)]) -> Vec<RgbImage> {
    let (_device, stream, kernels) = setup();
    let plan: Vec<_> = items
        .iter()
        .map(|(i, h, w)| {
            (
                (i.height as usize, i.width as usize),
                (*h as usize, *w as usize),
            )
        })
        .collect();
    let batch = ResizeBatch::new(&plan).unwrap();
    let src: Vec<u8> = items
        .iter()
        .flat_map(|(i, _, _)| i.data.iter().copied())
        .collect();
    let d_src = stream.clone_htod(&src).unwrap();
    let d_desc = stream.clone_htod(&batch.words).unwrap();
    let mut d_mid = stream.alloc_zeros::<u8>(batch.mid_bytes).unwrap();
    let mut d_dst = stream.alloc_zeros::<u8>(batch.dst_bytes).unwrap();
    kernels
        .resize_bicubic(&stream, &batch, &d_src, &d_desc, &mut d_mid, &mut d_dst)
        .unwrap();
    let out = stream.clone_dtoh(&d_dst).unwrap();
    items
        .iter()
        .zip(&batch.dst_offsets)
        .map(|((_, h, w), &off)| RgbImage {
            width: *w,
            height: *h,
            data: out[off..off + (*h as usize * *w as usize * 3)].to_vec(),
        })
        .collect()
}

fn assert_within_one(got: &RgbImage, want: &RgbImage) {
    assert_eq!((got.height, got.width), (want.height, want.width));
    let worst = got
        .data
        .iter()
        .zip(&want.data)
        .map(|(a, b)| i32::from(*a).abs_diff(i32::from(*b)))
        .max()
        .unwrap_or(0);
    assert!(worst <= 1, "max difference {worst}");
}

#[test]
#[ignore = "needs GPU"]
fn resize_matches_reference_on_the_non_conforming_fixture() {
    let image = fixture("w777_h333.jpg");
    let (h, w) = smart_resize_with(image.height, image.width, &SmartResizeConfig::default())
        .expect("target");
    let got = run_resize(&[(&image, h, w)]);
    let want = resize_bicubic_aa(&image, h, w).unwrap();
    assert_within_one(&got[0], &want);
}

#[test]
#[ignore = "needs GPU"]
fn resize_batches_shrink_grow_and_identity() {
    let big = synthetic(200, 310, 7);
    let small = synthetic(30, 45, 8);
    let same = synthetic(40, 40, 9);
    let items = [(&big, 56u32, 84u32), (&small, 112, 140), (&same, 40, 40)];
    let got = run_resize(&items);
    for ((image, h, w), out) in items.iter().zip(&got) {
        assert_within_one(out, &resize_bicubic_aa(image, *h, *w).unwrap());
    }
}
