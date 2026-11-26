use crate::error::ImageError;
use crate::jpeg::RgbImage;

const CUBIC_A: f64 = -0.5;

fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((CUBIC_A + 2.0) * x - (CUBIC_A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * CUBIC_A
    } else {
        0.0
    }
}

struct Taps {
    start: usize,
    weights: Vec<f64>,
}

/// Antialiased bicubic taps for each output index: support `2 * scale` when shrinking,
/// `2` when growing, weights normalised to sum to one, half-pixel centres.
fn taps(input: usize, output: usize) -> Vec<Taps> {
    let scale = input as f64 / output as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let inv = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    (0..output)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let lo = ((center - support + 0.5) as i64).max(0) as usize;
            let hi = (((center + support + 0.5) as i64).max(0) as usize).min(input);
            let mut weights: Vec<f64> = (lo..hi)
                .map(|j| cubic((j as f64 - center + 0.5) * inv))
                .collect();
            let sum: f64 = weights.iter().sum();
            if sum != 0.0 {
                weights.iter_mut().for_each(|w| *w /= sum);
            }
            Taps { start: lo, weights }
        })
        .collect()
}

fn to_u8(v: f64) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// Resizes `image` to `out_height` x `out_width` with the bicubic antialias semantics of
/// torchvision's tensor path: separable, horizontal pass then vertical pass, each pass
/// rounded to uint8. Equal dimensions return the input unchanged. This is a reference
/// whose authority is the end-to-end gate, not bit identity with ATen's fixed-point path.
pub fn resize_bicubic_aa(
    image: &RgbImage,
    out_height: u32,
    out_width: u32,
) -> Result<RgbImage, ImageError> {
    if out_height == 0 || out_width == 0 {
        return Err(ImageError::InvalidDimensions {
            width: out_width,
            height: out_height,
        });
    }
    if image.height == out_height && image.width == out_width {
        return Ok(image.clone());
    }
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (ow, oh) = (out_width as usize, out_height as usize);
    let horizontal = taps(iw, ow);
    let vertical = taps(ih, oh);

    let mut mid = vec![0u8; ih * ow * 3];
    for row in 0..ih {
        for (col, tap) in horizontal.iter().enumerate() {
            for c in 0..3 {
                let acc: f64 = tap
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(k, w)| w * f64::from(image.data[(row * iw + tap.start + k) * 3 + c]))
                    .sum();
                mid[(row * ow + col) * 3 + c] = to_u8(acc);
            }
        }
    }
    let mut data = vec![0u8; oh * ow * 3];
    for (row, tap) in vertical.iter().enumerate() {
        for col in 0..ow {
            for c in 0..3 {
                let acc: f64 = tap
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(k, w)| w * f64::from(mid[((tap.start + k) * ow + col) * 3 + c]))
                    .sum();
                data[(row * ow + col) * 3 + c] = to_u8(acc);
            }
        }
    }
    Ok(RgbImage {
        width: out_width,
        height: out_height,
        data,
    })
}
