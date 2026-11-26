//! Image decoding and preprocessing reference.
//!
//! Turns a media reference into the `[N, 1176]` bf16 patch matrix, patch grid and
//! token count the vision tower consumes.

mod data_url;
mod error;
mod grid;
mod header;
mod jpeg;
mod normalize;
pub mod nvjpeg;
mod patchify;
mod resize;
mod smart_resize;

pub use data_url::{MediaPolicy, data_url_prefix, load_media, parse_data_url};
pub use error::ImageError;
pub use grid::{CHANNELS, GridThw, MERGE_SIZE, PATCH_ELEMS, PATCH_SIZE, TEMPORAL_PATCH_SIZE};
pub use header::{DecodePlan, Format, RASTER_PIXEL_FACTOR, image_header, plan_decode};
#[cfg(feature = "zune")]
pub use jpeg::decode_jpeg_zune;
pub use jpeg::{RgbImage, decode_image, decode_jpeg, decode_png, decode_webp};
#[cfg(feature = "libjpeg")]
pub use jpeg::{decode_jpeg_libjpeg, decode_jpeg_scaled};
pub use normalize::{MEAN, STD, bf16_tables, normalize_bf16, normalize_f32};
pub use patchify::{inner_offset, patch_origin, patchify_bf16};
pub use resize::resize_bicubic_aa;
pub use smart_resize::{
    IMAGE_FACTOR, MAX_ASPECT_RATIO, MAX_PIXELS, MIN_PIXELS, SmartResizeConfig, TEMPORAL_FACTOR,
    smart_resize, smart_resize_with,
};

/// The preprocessed form of one image.
#[derive(Clone, Debug)]
pub struct Preprocessed {
    /// Row-major `[patches, 1176]` bf16 matrix.
    pub pixels: Vec<half::bf16>,
    /// The patch grid.
    pub grid_thw: GridThw,
    /// Language-model tokens the image expands to.
    pub tokens: usize,
    /// Whether the decoded image had to be resized.
    pub resized: bool,
}

/// Decodes `bytes` within the raster bound the header allows, resizes to the
/// `smart_resize` target when needed, normalises and patchifies.
///
/// The header is read first: an aspect ratio beyond `config.max_aspect_ratio` or a raster
/// beyond [`RASTER_PIXEL_FACTOR`] times `config.max_pixels` is refused before any pixel
/// buffer exists, except JPEG, which decodes at a reduced DCT scale.
pub fn preprocess(bytes: &[u8], config: &SmartResizeConfig) -> Result<Preprocessed, ImageError> {
    let (_, width, height) = image_header(bytes)?;
    if let Some(limit) = config.max_aspect_ratio
        && f64::from(width.max(height)) / f64::from(width.min(height)) > limit
    {
        return Err(ImageError::AspectRatio {
            height,
            width,
            limit,
        });
    }
    let plan = plan_decode(bytes, config.max_pixels, cfg!(feature = "libjpeg"))?;
    let decoded = decode_planned(bytes, plan)?;
    let (height, width) = smart_resize_with(decoded.height, decoded.width, config)?;
    let resized = (height, width) != (decoded.height, decoded.width);
    let image = if resized {
        resize_bicubic_aa(&decoded, height, width)?
    } else {
        decoded
    };
    let (grid_thw, pixels) = patchify_bf16(&image);
    Ok(Preprocessed {
        tokens: grid_thw.tokens(),
        pixels,
        grid_thw,
        resized,
    })
}

fn decode_planned(bytes: &[u8], plan: DecodePlan) -> Result<RgbImage, ImageError> {
    #[cfg(feature = "libjpeg")]
    if plan.format == Format::Jpeg {
        return decode_jpeg_scaled(bytes, plan.scale_denom);
    }
    let _ = plan;
    decode_image(bytes)
}
