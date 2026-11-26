//! Region crops for the recognition seat: the bbox crop of a page raster, the
//! resize to the vision encoder's patch grid, and the PNG data URL.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::codecs::png::FilterType as PngFilter;
use image::codecs::png::{CompressionType, PngEncoder};
use image::imageops::{self, FilterType};
use image::{ExtendedColorType, ImageEncoder, RgbImage};
use std::fmt;

/// The edge of the normalized box space a bbox is expressed in.
pub const BBOX_EXTENT: f64 = 1000.0;

/// The pixel edge of one vision patch.
pub const PATCH_EDGE: u32 = 14;

/// The patches merged along each edge.
pub const PATCH_MERGE: u32 = 2;

/// The patch expansion factor.
pub const PATCH_EXPAND_FACTOR: u32 = 1;

/// The multiple both resized edges are rounded to.
pub const IMAGE_FACTOR: u32 = PATCH_EDGE * PATCH_MERGE * PATCH_EXPAND_FACTOR;

/// The temporal patch size, used as both the temporal extent and factor of
/// the resize.
pub const T_PATCH_SIZE: u32 = 2;

/// The fewest pixels a resized image may have.
pub const MIN_PIXELS: u64 = 112 * 112;

/// The image tokens an image may occupy.
pub const IMAGE_TOKEN_BUDGET: u64 = 6_000;

/// The most pixels a resized image may have.
pub const MAX_PIXELS: u64 = 28 * 28 * 2 * IMAGE_TOKEN_BUDGET;

/// The media type and encoding prefix of the data URL.
const DATA_URL_PREFIX: &str = "data:image/png;base64,";

/// Why a region could not be cropped or encoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CropError {
    /// The bbox has a non-finite coordinate.
    InvalidBox,
    /// The box has no area, or its right or bottom edge precedes its left or
    /// top edge.
    EmptyRegion,
    /// The page raster has no pixels.
    EmptyPage,
    /// The resize target has no pixels.
    EmptyTarget,
    /// The PNG encoder refused the image.
    Encode(String),
}

impl fmt::Display for CropError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBox => f.write_str("the region box has a non-finite coordinate"),
            Self::EmptyRegion => f.write_str("the region box has no area"),
            Self::EmptyPage => f.write_str("the page raster has no pixels"),
            Self::EmptyTarget => f.write_str("the resize target has no pixels"),
            Self::Encode(detail) => write!(f, "the region could not be PNG encoded: {detail}"),
        }
    }
}

impl std::error::Error for CropError {}

/// The `(height, width)` an image of `h` by `w` pixels and `t` temporal
/// extent is resized to: both rounded half-to-even to a multiple of the
/// factor, shrunk by `floor` when the temporal-scaled area exceeds
/// `max_pixels`, grown by `ceil` when it is below `min_pixels`. `None` when
/// `t` is below `t_factor` or a factor is zero.
#[allow(clippy::too_many_arguments)]
pub fn smart_resize(
    t: u32,
    h: u32,
    w: u32,
    t_factor: u32,
    h_factor: u32,
    w_factor: u32,
    min_pixels: u64,
    max_pixels: u64,
) -> Option<(u32, u32)> {
    if t < t_factor || t_factor == 0 || h_factor == 0 || w_factor == 0 {
        return None;
    }
    let (t, h, w) = (f64::from(t), f64::from(h), f64::from(w));
    let (t_factor, h_factor, w_factor) = (
        f64::from(t_factor),
        f64::from(h_factor),
        f64::from(w_factor),
    );
    let mut h_bar = (h / h_factor).round_ties_even() * h_factor;
    let mut w_bar = (w / w_factor).round_ties_even() * w_factor;
    let t_bar = (t / t_factor).round_ties_even() * t_factor;
    let volume = t_bar * h_bar * w_bar;
    if volume > max_pixels as f64 {
        let beta = (t * h * w / max_pixels as f64).sqrt();
        h_bar = (h / beta / h_factor).floor() * h_factor;
        w_bar = (w / beta / w_factor).floor() * w_factor;
    } else if volume < min_pixels as f64 {
        let beta = (min_pixels as f64 / (t * h * w)).sqrt();
        h_bar = (h * beta / h_factor).ceil() * h_factor;
        w_bar = (w * beta / w_factor).ceil() * w_factor;
    }
    Some((h_bar as u32, w_bar as u32))
}

/// The `(height, width)` a region of `h` by `w` pixels is resized to before
/// encoding.
pub fn region_target(h: u32, w: u32) -> Option<(u32, u32)> {
    smart_resize(
        T_PATCH_SIZE,
        h,
        w,
        T_PATCH_SIZE,
        IMAGE_FACTOR,
        IMAGE_FACTOR,
        MIN_PIXELS,
        MAX_PIXELS,
    )
}

fn pixel_edge(normalized: f64, extent: u32) -> Option<i64> {
    let pixel = normalized * f64::from(extent) / BBOX_EXTENT;
    pixel.is_finite().then_some(pixel as i64)
}

/// The region of `page` that `bbox` (`[x0, y0, x1, y1]` in the 0..1000 box
/// space) names: each edge is `norm * extent / 1000` truncated toward zero;
/// pixels of the box outside the page are black.
pub fn crop_region(page: &RgbImage, bbox: [f64; 4]) -> Result<RgbImage, CropError> {
    let (width, height) = page.dimensions();
    if width == 0 || height == 0 {
        return Err(CropError::EmptyPage);
    }
    let edges = [
        pixel_edge(bbox[0], width),
        pixel_edge(bbox[1], height),
        pixel_edge(bbox[2], width),
        pixel_edge(bbox[3], height),
    ];
    let [Some(x0), Some(y0), Some(x1), Some(y1)] = edges else {
        return Err(CropError::InvalidBox);
    };
    if x1 <= x0 || y1 <= y0 {
        return Err(CropError::EmptyRegion);
    }
    let (crop_width, crop_height) = match (u32::try_from(x1 - x0), u32::try_from(y1 - y0)) {
        (Ok(crop_width), Ok(crop_height)) => (crop_width, crop_height),
        _ => return Err(CropError::EmptyRegion),
    };
    let overlap_x0 = x0.max(0);
    let overlap_y0 = y0.max(0);
    let overlap_x1 = x1.min(i64::from(width));
    let overlap_y1 = y1.min(i64::from(height));
    let mut crop = RgbImage::new(crop_width, crop_height);
    if overlap_x1 > overlap_x0 && overlap_y1 > overlap_y0 {
        let view = imageops::crop_imm(
            page,
            overlap_x0 as u32,
            overlap_y0 as u32,
            (overlap_x1 - overlap_x0) as u32,
            (overlap_y1 - overlap_y0) as u32,
        );
        imageops::replace(
            &mut crop,
            &view.to_image(),
            overlap_x0 - x0,
            overlap_y0 - y0,
        );
    }
    Ok(crop)
}

/// `region` resized (bicubic) to its [`region_target`] and encoded as a lossless
/// PNG.
pub fn encode_region(region: &RgbImage) -> Result<Vec<u8>, CropError> {
    let (width, height) = region.dimensions();
    if width == 0 || height == 0 {
        return Err(CropError::EmptyRegion);
    }
    let (target_height, target_width) =
        region_target(height, width).ok_or(CropError::EmptyTarget)?;
    if target_height == 0 || target_width == 0 {
        return Err(CropError::EmptyTarget);
    }
    let resized;
    let pixels = if (target_width, target_height) == (width, height) {
        region
    } else {
        resized = imageops::resize(region, target_width, target_height, FilterType::CatmullRom);
        &resized
    };
    let mut bytes = Vec::new();
    PngEncoder::new_with_quality(&mut bytes, CompressionType::Fast, PngFilter::Adaptive)
        .write_image(
            pixels.as_raw(),
            target_width,
            target_height,
            ExtendedColorType::Rgb8,
        )
        .map_err(|error| CropError::Encode(error.to_string()))?;
    Ok(bytes)
}

/// The `data:image/png;base64,` URL of `region` after [`encode_region`].
pub fn region_data_url(region: &RgbImage) -> Result<String, CropError> {
    let bytes = encode_region(region)?;
    Ok(format!("{DATA_URL_PREFIX}{}", STANDARD.encode(bytes)))
}

/// The data URL of the region of `page` that `bbox` names.
pub fn crop_data_url(page: &RgbImage, bbox: [f64; 4]) -> Result<String, CropError> {
    region_data_url(&crop_region(page, bbox)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn resize_of(h: u32, w: u32) -> (u32, u32) {
        region_target(h, w).unwrap()
    }

    #[test]
    fn constants_match_the_sdk_configuration() {
        assert_eq!(IMAGE_FACTOR, 28);
        assert_eq!(MAX_PIXELS, 9_408_000);
        assert_eq!(MIN_PIXELS, 12_544);
    }

    #[test]
    fn smart_resize_matches_the_python_function() {
        assert_eq!(resize_of(100, 100), (112, 112));
        assert_eq!(resize_of(28, 28), (84, 84));
        assert_eq!(resize_of(4000, 4000), (2156, 2156));
        assert_eq!(resize_of(14, 14), (84, 84));
        assert_eq!(resize_of(1000, 3), (1456, 28));
        assert_eq!(resize_of(3000, 2000), (2632, 1764));
        assert_eq!(resize_of(5, 2000), (28, 1596));
    }

    #[test]
    fn smart_resize_rounds_half_to_even() {
        assert_eq!(resize_of(70, 700), (56, 700));
        assert_eq!(resize_of(42, 700), (56, 700));
    }

    #[test]
    fn smart_resize_stays_inside_the_vision_token_budget() {
        for (width, height) in [(867, 6771), (6771, 867), (4096, 4096)] {
            let (h, w) = resize_of(height, width);
            assert!(u64::from(h) * u64::from(w) / (28 * 28) <= IMAGE_TOKEN_BUDGET);
            assert!(u64::from(h) * u64::from(w) / (28 * 28) < 6_084);
        }
    }

    #[test]
    fn smart_resize_refuses_a_temporal_extent_below_its_factor() {
        assert_eq!(
            smart_resize(1, 100, 100, 2, 28, 28, MIN_PIXELS, MAX_PIXELS),
            None
        );
    }

    fn gradient(width: u32, height: u32) -> RgbImage {
        RgbImage::from_fn(width, height, |x, y| Rgb([x as u8, y as u8, 7]))
    }

    #[test]
    fn crop_truncates_the_normalized_edges() {
        let page = gradient(300, 200);
        let crop = crop_region(&page, [100.5, 250.0, 600.9, 750.0]).unwrap();
        assert_eq!(crop.dimensions(), (150, 100));
        assert_eq!(*crop.get_pixel(0, 0), *page.get_pixel(30, 50));
    }

    #[test]
    fn crop_pads_outside_the_page_with_black() {
        let page = RgbImage::from_pixel(100, 100, Rgb([200, 200, 200]));
        let crop = crop_region(&page, [-100.0, 0.0, 500.0, 500.0]).unwrap();
        assert_eq!(crop.dimensions(), (60, 50));
        assert_eq!(*crop.get_pixel(0, 0), Rgb([0, 0, 0]));
        assert_eq!(*crop.get_pixel(59, 49), Rgb([200, 200, 200]));
    }

    #[test]
    fn crop_refuses_degenerate_boxes() {
        let page = gradient(100, 100);
        assert_eq!(
            crop_region(&page, [500.0, 0.0, 500.0, 900.0]),
            Err(CropError::EmptyRegion)
        );
        assert_eq!(
            crop_region(&page, [600.0, 0.0, 500.0, 900.0]),
            Err(CropError::EmptyRegion)
        );
        assert_eq!(
            crop_region(&page, [f64::NAN, 0.0, 5.0, 9.0]),
            Err(CropError::InvalidBox)
        );
    }

    #[test]
    fn data_url_is_a_decodable_png_on_the_patch_grid() {
        let url = crop_data_url(&gradient(400, 400), [0.0, 0.0, 1000.0, 1000.0]).unwrap();
        let payload = url.strip_prefix("data:image/png;base64,").unwrap();
        let bytes = STANDARD.decode(payload).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let decoded = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png).unwrap();
        assert_eq!((decoded.height(), decoded.width()), resize_of(400, 400));
        assert_eq!(decoded.width() % IMAGE_FACTOR, 0);
    }

    #[test]
    fn small_regions_are_grown_to_the_minimum() {
        let url = region_data_url(&gradient(28, 28)).unwrap();
        let bytes = STANDARD
            .decode(url.strip_prefix("data:image/png;base64,").unwrap())
            .unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert_eq!((decoded.height(), decoded.width()), (84, 84));
    }
}
