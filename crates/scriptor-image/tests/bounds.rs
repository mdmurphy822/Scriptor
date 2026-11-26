use scriptor_image::{
    Format, ImageError, RASTER_PIXEL_FACTOR, SmartResizeConfig, image_header, plan_decode,
    preprocess,
};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/image")
            .join(name),
    )
    .unwrap()
}

fn png_header(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13];
    bytes.extend_from_slice(b"IHDR");
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&[8, 2, 0, 0, 0]);
    bytes
}

fn jpeg_header(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = vec![0xFF, 0xD8, 0xFF, 0xE0, 0, 4, 0, 0, 0xFF, 0xC0, 0, 11, 8];
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&[1, 1, 0x11, 0]);
    bytes
}

fn small_config(max_pixels: u64) -> SmartResizeConfig {
    SmartResizeConfig {
        min_pixels: 784,
        max_pixels,
        ..SmartResizeConfig::default()
    }
}

#[test]
fn header_dimensions_are_read_without_decoding() {
    assert_eq!(
        image_header(&png_header(40_000, 30_000)).unwrap(),
        (Format::Png, 40_000, 30_000)
    );
    assert_eq!(
        image_header(&jpeg_header(5000, 4000)).unwrap(),
        (Format::Jpeg, 5000, 4000)
    );
    assert_eq!(
        image_header(&fixture("w448_h420.jpg")).unwrap(),
        (Format::Jpeg, 448, 420)
    );
}

#[test]
fn oversized_png_is_refused_naming_max_pixels() {
    let err = plan_decode(&png_header(40_000, 30_000), 9_633_792, true).unwrap_err();
    assert!(matches!(
        err,
        ImageError::TooLarge {
            max_pixels: 9_633_792,
            ..
        }
    ));
    assert!(err.to_string().contains("max_pixels"));
}

#[test]
fn png_at_the_bound_is_planned_unscaled() {
    let max = 10_000u64;
    let side = ((max * RASTER_PIXEL_FACTOR) as f64).sqrt() as u32;
    let plan = plan_decode(&png_header(side, side), max, true).unwrap();
    assert_eq!(plan.scale_denom, 1);
    assert!(plan_decode(&png_header(side + 1, side + 1), max, true).is_err());
}

#[test]
fn jpeg_over_the_bound_takes_the_smallest_fitting_scale() {
    let max = 10_000u64;
    let bound = max * RASTER_PIXEL_FACTOR;
    for (width, height, denom) in [
        (150u16, 150u16, 1u32),
        (300, 300, 2),
        (600, 600, 4),
        (1200, 1200, 8),
    ] {
        let plan = plan_decode(&jpeg_header(width, height), max, true).unwrap();
        assert_eq!(plan.scale_denom, denom, "{width}x{height}");
        let scaled = u64::from(u32::from(width).div_ceil(denom))
            * u64::from(u32::from(height).div_ceil(denom));
        assert!(scaled <= bound);
    }
}

#[test]
fn jpeg_beyond_one_eighth_is_refused() {
    let err = plan_decode(&jpeg_header(65_535, 65_535), 9_633_792, true).unwrap_err();
    assert!(matches!(err, ImageError::TooLarge { .. }));
    assert!(plan_decode(&jpeg_header(600, 600), 10_000, false).is_err());
}

#[cfg(feature = "libjpeg")]
#[test]
fn oversized_jpeg_decodes_scaled_and_stays_within_max_pixels() {
    let bytes = fixture("w448_h420.jpg");
    let plan = plan_decode(&bytes, 10_000, true).unwrap();
    assert_eq!(plan.scale_denom, 4);
    let pre = preprocess(&bytes, &small_config(10_000)).unwrap();
    assert!(pre.resized);
    let pixels = pre.grid_thw.h as u64 * 14 * pre.grid_thw.w as u64 * 14;
    assert!(pixels <= 10_000, "{pixels}");
}

#[test]
fn conforming_jpeg_decodes_at_full_size() {
    let bytes = fixture("w448_h420.jpg");
    assert_eq!(plan_decode(&bytes, 9_633_792, true).unwrap().scale_denom, 1);
    assert!(preprocess(&bytes, &SmartResizeConfig::default()).is_ok());
}

#[test]
fn extreme_aspect_ratio_is_refused_before_decode() {
    let err = preprocess(&png_header(20_000, 20), &SmartResizeConfig::default()).unwrap_err();
    assert!(matches!(err, ImageError::AspectRatio { .. }));
}

#[test]
fn truncated_headers_are_decode_errors() {
    assert!(image_header(&[0xFF, 0xD8, 0xFF]).is_err());
    assert!(image_header(&[0x89, b'P', b'N', b'G']).is_err());
}
