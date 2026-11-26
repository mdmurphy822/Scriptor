//! Page image to model input tensor.

use image::RgbImage;

/// Side length of the square model input.
pub const INPUT_SIZE: usize = 800;

const CUBIC_A: f32 = -0.75;

/// A page converted to the model's input tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessedPage {
    /// Channel-first `3 x 800 x 800` values in `0.0..=1.0`.
    pub data: Vec<f32>,
    /// Tensor shape, `[3, 800, 800]`.
    pub shape: [usize; 3],
    /// Width of the source page in pixels.
    pub page_width: u32,
    /// Height of the source page in pixels.
    pub page_height: u32,
    /// Source pixels per tensor pixel horizontally.
    pub scale_x: f32,
    /// Source pixels per tensor pixel vertically.
    pub scale_y: f32,
}

/// Resizes `page` to 800x800 without preserving aspect ratio, using bicubic
/// interpolation (a = -0.75, half-pixel centres, edge clamping, no
/// antialiasing) rounded to 8-bit values, then scales by 1/255 into a CHW
/// tensor.
#[must_use]
pub fn preprocess(page: &RgbImage) -> PreprocessedPage {
    let (width, height) = page.dimensions();
    let (w, h) = (width as usize, height as usize);
    let xs = taps(w, INPUT_SIZE);
    let ys = taps(h, INPUT_SIZE);
    let src = page.as_raw();
    let plane = INPUT_SIZE * INPUT_SIZE;
    let mut data = vec![0.0f32; 3 * plane];
    let mut horizontal = vec![0.0f32; h * INPUT_SIZE];
    for channel in 0..3 {
        for row in 0..h {
            for (x, (idx, wts)) in xs.iter().enumerate() {
                let mut acc = 0.0f32;
                for k in 0..4 {
                    acc += wts[k] * f32::from(src[(row * w + idx[k]) * 3 + channel]);
                }
                horizontal[row * INPUT_SIZE + x] = acc;
            }
        }
        for (y, (idx, wts)) in ys.iter().enumerate() {
            for x in 0..INPUT_SIZE {
                let mut acc = 0.0f32;
                for k in 0..4 {
                    acc += wts[k] * horizontal[idx[k] * INPUT_SIZE + x];
                }
                data[channel * plane + y * INPUT_SIZE + x] =
                    acc.clamp(0.0, 255.0).round_ties_even() / 255.0;
            }
        }
    }
    PreprocessedPage {
        data,
        shape: [3, INPUT_SIZE, INPUT_SIZE],
        page_width: width,
        page_height: height,
        scale_x: width as f32 / INPUT_SIZE as f32,
        scale_y: height as f32 / INPUT_SIZE as f32,
    }
}

fn cubic_near(x: f32) -> f32 {
    ((CUBIC_A + 2.0) * x - (CUBIC_A + 3.0)) * x * x + 1.0
}

fn cubic_far(x: f32) -> f32 {
    ((CUBIC_A * x - 5.0 * CUBIC_A) * x + 8.0 * CUBIC_A) * x - 4.0 * CUBIC_A
}

/// Per output index: the four clamped source indices and their weights.
fn taps(input: usize, output: usize) -> Vec<([usize; 4], [f32; 4])> {
    let scale = input as f32 / output as f32;
    (0..output)
        .map(|dst| {
            let src = scale * (dst as f32 + 0.5) - 0.5;
            let base = src.floor();
            let t = src - base;
            let weights = [
                cubic_far(t + 1.0),
                cubic_near(t),
                cubic_near(1.0 - t),
                cubic_far(2.0 - t),
            ];
            let last = input as isize - 1;
            let mut idx = [0usize; 4];
            for (k, slot) in idx.iter_mut().enumerate() {
                *slot = (base as isize - 1 + k as isize).clamp(0, last) as usize;
            }
            (idx, weights)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn constant_page_maps_to_constant_scaled_planes() {
        let page = RgbImage::from_pixel(37, 91, Rgb([255, 51, 0]));
        let out = preprocess(&page);
        assert_eq!(out.shape, [3, 800, 800]);
        assert_eq!(out.data.len(), 3 * 800 * 800);
        let plane = 800 * 800;
        assert!(out.data[..plane].iter().all(|v| (*v - 1.0).abs() < 1e-6));
        assert!(
            out.data[plane..2 * plane]
                .iter()
                .all(|v| (*v - 0.2).abs() < 1e-6)
        );
        assert!(out.data[2 * plane..].iter().all(|v| *v == 0.0));
        assert_eq!((out.page_width, out.page_height), (37, 91));
        assert!((out.scale_x - 37.0 / 800.0).abs() < 1e-9);
        assert!((out.scale_y - 91.0 / 800.0).abs() < 1e-9);
    }

    #[test]
    fn weights_sum_to_one_and_identity_size_is_exact() {
        for (_, wts) in taps(13, 800) {
            assert!((wts.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
        let mut page = RgbImage::new(800, 800);
        for (x, y, p) in page.enumerate_pixels_mut() {
            *p = Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]);
        }
        let out = preprocess(&page);
        assert!((out.data[801] - 1.0 / 255.0).abs() < 1e-6);
        assert!((out.data[800 * 800 + 800 * 2] - 2.0 / 255.0).abs() < 1e-6);
    }
}
