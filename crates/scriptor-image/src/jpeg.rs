use std::io::Cursor;

use crate::error::ImageError;

/// An 8-bit interleaved RGB raster.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `height * width * 3` bytes in row-major `R, G, B` order.
    pub data: Vec<u8>,
}

impl RgbImage {
    /// Returns the `[r, g, b]` triple at `(row, col)`.
    pub fn pixel(&self, row: u32, col: u32) -> [u8; 3] {
        let i = (row as usize * self.width as usize + col as usize) * 3;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }
}

fn decode_err(format: &'static str, detail: impl ToString) -> ImageError {
    ImageError::Decode {
        format,
        detail: detail.to_string(),
    }
}

fn checked(width: usize, height: usize) -> Result<(u32, u32), ImageError> {
    match (u32::try_from(width), u32::try_from(height)) {
        (Ok(w), Ok(h)) if w > 0 && h > 0 => Ok((w, h)),
        _ => Err(ImageError::InvalidDimensions {
            width: width as u32,
            height: height as u32,
        }),
    }
}

/// Decodes a JPEG into RGB through libjpeg-turbo when the `libjpeg` feature is on,
/// otherwise through zune-jpeg.
pub fn decode_jpeg(bytes: &[u8]) -> Result<RgbImage, ImageError> {
    #[cfg(feature = "libjpeg")]
    {
        decode_jpeg_libjpeg(bytes)
    }
    #[cfg(all(not(feature = "libjpeg"), feature = "zune"))]
    {
        decode_jpeg_zune(bytes)
    }
    #[cfg(not(any(feature = "libjpeg", feature = "zune")))]
    {
        let _ = bytes;
        Err(decode_err("jpeg", "no jpeg decoder feature enabled"))
    }
}

#[cfg(feature = "libjpeg")]
unsafe extern "C" {
    fn scriptor_jpeg_decode_rgb(
        data: *const u8,
        len: usize,
        out: *mut *mut u8,
        width: *mut u32,
        height: *mut u32,
        scale_denom: u32,
        message: *mut std::ffi::c_char,
        message_len: usize,
    ) -> std::ffi::c_int;
    fn scriptor_jpeg_free(buffer: *mut u8);
}

/// Decodes a JPEG into RGB with libjpeg-turbo (islow IDCT, fancy upsampling).
#[cfg(feature = "libjpeg")]
pub fn decode_jpeg_libjpeg(bytes: &[u8]) -> Result<RgbImage, ImageError> {
    decode_jpeg_scaled(bytes, 1)
}

/// Decodes a JPEG into RGB at `1/scale_denom` of its size (`1`, `2`, `4` or `8`) with
/// libjpeg-turbo's DCT scaling.
#[cfg(feature = "libjpeg")]
pub fn decode_jpeg_scaled(bytes: &[u8], scale_denom: u32) -> Result<RgbImage, ImageError> {
    let mut out: *mut u8 = std::ptr::null_mut();
    let (mut width, mut height) = (0u32, 0u32);
    let mut message = [0 as std::ffi::c_char; 256];
    // SAFETY: the shim reads `bytes.len()` bytes from `bytes`, writes only the out
    // parameters passed here, and `message` has the advertised length.
    let status = unsafe {
        scriptor_jpeg_decode_rgb(
            bytes.as_ptr(),
            bytes.len(),
            &mut out,
            &mut width,
            &mut height,
            scale_denom,
            message.as_mut_ptr(),
            message.len(),
        )
    };
    if status != 0 {
        // SAFETY: on failure the shim wrote a NUL-terminated string into `message`.
        let detail = unsafe { std::ffi::CStr::from_ptr(message.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        return Err(decode_err("jpeg", detail));
    }
    let (width, height) = checked(width as usize, height as usize)?;
    let len = width as usize * height as usize * 3;
    // SAFETY: on success `out` points to a malloc'd buffer of `len` initialised bytes.
    let data = unsafe { std::slice::from_raw_parts(out, len) }.to_vec();
    // SAFETY: `out` came from the shim's allocator and is not used afterwards.
    unsafe { scriptor_jpeg_free(out) };
    Ok(RgbImage {
        width,
        height,
        data,
    })
}

/// Decodes a JPEG into RGB with zune-jpeg.
#[cfg(feature = "zune")]
pub fn decode_jpeg_zune(bytes: &[u8]) -> Result<RgbImage, ImageError> {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    let options = DecoderOptions::default()
        .set_max_width(1 << 20)
        .set_max_height(1 << 20)
        .jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    let data = decoder
        .decode()
        .map_err(|e| decode_err("jpeg", format!("{e:?}")))?;
    let info = decoder
        .info()
        .ok_or_else(|| decode_err("jpeg", "missing image info"))?;
    let (width, height) = checked(info.width as usize, info.height as usize)?;
    if data.len() != width as usize * height as usize * 3 {
        return Err(decode_err("jpeg", "unexpected output size"));
    }
    Ok(RgbImage {
        width,
        height,
        data,
    })
}

/// Decodes a PNG into RGB, dropping any alpha channel.
pub fn decode_png(bytes: &[u8]) -> Result<RgbImage, ImageError> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|e| decode_err("png", e))?;
    let mut buf = vec![
        0u8;
        reader
            .output_buffer_size()
            .ok_or_else(|| decode_err("png", "image too large"))?
    ];
    let frame = reader
        .next_frame(&mut buf)
        .map_err(|e| decode_err("png", e))?;
    let (width, height) = checked(frame.width as usize, frame.height as usize)?;
    let channels = frame.color_type.samples();
    let pixels = width as usize * height as usize;
    let mut data = Vec::with_capacity(pixels * 3);
    for px in buf[..frame.buffer_size()].chunks_exact(channels) {
        match channels {
            1 | 2 => data.extend_from_slice(&[px[0], px[0], px[0]]),
            _ => data.extend_from_slice(&px[..3]),
        }
    }
    Ok(RgbImage {
        width,
        height,
        data,
    })
}

/// Decodes a WebP into RGB, dropping any alpha channel.
pub fn decode_webp(bytes: &[u8]) -> Result<RgbImage, ImageError> {
    let mut decoder =
        image_webp::WebPDecoder::new(Cursor::new(bytes)).map_err(|e| decode_err("webp", e))?;
    let (width, height) = decoder.dimensions();
    let (width, height) = checked(width as usize, height as usize)?;
    let alpha = decoder.has_alpha();
    let size = decoder
        .output_buffer_size()
        .ok_or_else(|| decode_err("webp", "image too large"))?;
    let mut buf = vec![0u8; size];
    decoder
        .read_image(&mut buf)
        .map_err(|e| decode_err("webp", e))?;
    let data = if alpha {
        buf.chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect()
    } else {
        buf
    };
    Ok(RgbImage {
        width,
        height,
        data,
    })
}

/// Decodes jpeg, png or webp bytes, chosen by signature, into RGB.
pub fn decode_image(bytes: &[u8]) -> Result<RgbImage, ImageError> {
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        decode_jpeg(bytes)
    } else if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        decode_png(bytes)
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        decode_webp(bytes)
    } else {
        Err(ImageError::UnknownFormat)
    }
}
