use std::io::Cursor;

use crate::error::ImageError;

/// The container format of an encoded image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// JPEG.
    Jpeg,
    /// PNG.
    Png,
    /// WebP.
    Webp,
}

fn header_err(format: &'static str, detail: &str) -> ImageError {
    ImageError::Decode {
        format,
        detail: detail.to_owned(),
    }
}

fn be16(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from(u16::from_be_bytes([
        *bytes.get(at)?,
        *bytes.get(at + 1)?,
    ])))
}

fn jpeg_dimensions(bytes: &[u8]) -> Result<(u32, u32), ImageError> {
    let truncated = || header_err("jpeg", "truncated header");
    let mut at = 2usize;
    loop {
        while bytes.get(at).ok_or_else(truncated)? != &0xFF {
            at += 1;
        }
        while bytes.get(at).ok_or_else(truncated)? == &0xFF {
            at += 1;
        }
        let marker = *bytes.get(at).ok_or_else(truncated)?;
        at += 1;
        match marker {
            0x00 | 0x01 | 0xD0..=0xD8 => continue,
            0xD9 | 0xDA => return Err(header_err("jpeg", "no frame header before scan data")),
            _ => {}
        }
        let len = be16(bytes, at).ok_or_else(truncated)? as usize;
        if matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xC8 | 0xCC) {
            let height = be16(bytes, at + 3).ok_or_else(truncated)?;
            let width = be16(bytes, at + 5).ok_or_else(truncated)?;
            return Ok((width, height));
        }
        at += len.max(2);
    }
}

fn png_dimensions(bytes: &[u8]) -> Result<(u32, u32), ImageError> {
    if bytes.len() < 24 || &bytes[12..16] != b"IHDR" {
        return Err(header_err("png", "missing IHDR chunk"));
    }
    let word =
        |at: usize| u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    Ok((word(16), word(20)))
}

fn webp_dimensions(bytes: &[u8]) -> Result<(u32, u32), ImageError> {
    let decoder = image_webp::WebPDecoder::new(Cursor::new(bytes))
        .map_err(|e| header_err("webp", &e.to_string()))?;
    Ok(decoder.dimensions())
}

/// Reads the format and `(width, height)` from the image header without decoding pixels.
pub fn image_header(bytes: &[u8]) -> Result<(Format, u32, u32), ImageError> {
    let (format, dims) = if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        (Format::Jpeg, jpeg_dimensions(bytes)?)
    } else if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        (Format::Png, png_dimensions(bytes)?)
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        (Format::Webp, webp_dimensions(bytes)?)
    } else {
        return Err(ImageError::UnknownFormat);
    };
    if dims.0 == 0 || dims.1 == 0 {
        return Err(ImageError::InvalidDimensions {
            width: dims.0,
            height: dims.1,
        });
    }
    Ok((format, dims.0, dims.1))
}

/// Host-side pixel multiple of `max_pixels` a decoded raster may occupy.
pub const RASTER_PIXEL_FACTOR: u64 = 4;

/// The JPEG DCT scale denominators libjpeg supports, smallest scaling first.
const JPEG_DENOMS: [u32; 4] = [1, 2, 4, 8];

/// Decoding plan chosen from the header against a pixel bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodePlan {
    /// Container format.
    pub format: Format,
    /// JPEG DCT scale denominator (`1` for every other format).
    pub scale_denom: u32,
}

fn too_large(width: u32, height: u32, max_pixels: u64, detail: String) -> ImageError {
    ImageError::TooLarge {
        width,
        height,
        max_pixels,
        detail,
    }
}

/// Chooses how to decode `bytes` so the host raster holds at most
/// [`RASTER_PIXEL_FACTOR`] times `max_pixels` pixels, reading only the header.
///
/// PNG and WebP above the bound are refused; a JPEG above it decodes at the smallest
/// libjpeg DCT scale that fits and is refused when even 1/8 does not.
pub fn plan_decode(
    bytes: &[u8],
    max_pixels: u64,
    jpeg_scaling: bool,
) -> Result<DecodePlan, ImageError> {
    let (format, width, height) = image_header(bytes)?;
    let bound = max_pixels.saturating_mul(RASTER_PIXEL_FACTOR);
    let pixels_at = |d: u32| u64::from(width.div_ceil(d)) * u64::from(height.div_ceil(d));
    if format != Format::Jpeg {
        let pixels = pixels_at(1);
        if pixels > bound {
            return Err(too_large(
                width,
                height,
                max_pixels,
                format!("{pixels} pixels against {RASTER_PIXEL_FACTOR} x max_pixels = {bound}"),
            ));
        }
        return Ok(DecodePlan {
            format,
            scale_denom: 1,
        });
    }
    let denoms: &[u32] = if jpeg_scaling {
        &JPEG_DENOMS
    } else {
        &JPEG_DENOMS[..1]
    };
    match denoms.iter().copied().find(|d| pixels_at(*d) <= bound) {
        Some(scale_denom) => Ok(DecodePlan {
            format,
            scale_denom,
        }),
        None => Err(too_large(
            width,
            height,
            max_pixels,
            format!(
                "{} pixels against {RASTER_PIXEL_FACTOR} x max_pixels = {bound} even at the smallest decode scale",
                pixels_at(1)
            ),
        )),
    }
}
