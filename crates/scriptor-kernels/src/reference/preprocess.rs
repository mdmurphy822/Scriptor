//! Pixel normalisation and patchify reference.

use half::bf16;

/// CLIP per-channel mean.
pub const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
/// CLIP per-channel standard deviation.
pub const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];
/// Patch edge in pixels.
pub const PATCH: usize = 14;
/// Frames per patch along the temporal axis.
pub const TEMPORAL: usize = 2;
/// Elements per flattened patch.
pub const PATCH_ELEMS: usize = 3 * TEMPORAL * PATCH * PATCH;

/// Per-channel `(scale, offset)` of the fused affine `y = x * scale + offset`
/// with `scale = (1/255)/std` and `offset = -mean/std`, evaluated as one fused
/// multiply-add.
pub fn fused_affine() -> [(f32, f32); 3] {
    let mut out = [(0.0, 0.0); 3];
    for c in 0..3 {
        out[c] = ((1.0 / 255.0) / STD[c], -MEAN[c] / STD[c]);
    }
    out
}

/// Normalises one 8-bit sample of `channel` with the fused affine.
pub fn normalize(value: u8, channel: usize, affine: &[(f32, f32); 3]) -> f32 {
    let (scale, offset) = affine[channel];
    f32::from(value).mul_add(scale, offset)
}

/// Normalises an interleaved RGB `height x width` image (`[h, w, 3]` u8) and
/// lays it out as the `[n, 1176]` patch matrix in f32.
///
/// Patches are ordered `(h/28, w/28, 2, 2)` (merge window row, merge window
/// column, row in window, column in window); each patch is flattened as
/// `(c, t, 14, 14)` with the single frame duplicated along `t`. Both sides must
/// be multiples of 28. Returns the patch matrix and `(grid_h, grid_w)` in
/// patches.
pub fn normalise_patchify(
    rgb: &[u8],
    height: usize,
    width: usize,
    affine: &[(f32, f32); 3],
) -> (Vec<f32>, (usize, usize)) {
    assert!(
        height.is_multiple_of(28) && width.is_multiple_of(28),
        "sides must be multiples of 28"
    );
    assert_eq!(rgb.len(), height * width * 3);
    let (gh, gw) = (height / PATCH, width / PATCH);
    let mut out = vec![0.0; gh * gw * PATCH_ELEMS];
    for window_row in 0..gh / 2 {
        for window_col in 0..gw / 2 {
            for dr in 0..2 {
                for dc in 0..2 {
                    let index = ((window_row * (gw / 2) + window_col) * 2 + dr) * 2 + dc;
                    let (row0, col0) =
                        ((window_row * 2 + dr) * PATCH, (window_col * 2 + dc) * PATCH);
                    let dst = &mut out[index * PATCH_ELEMS..(index + 1) * PATCH_ELEMS];
                    for c in 0..3 {
                        for t in 0..TEMPORAL {
                            for y in 0..PATCH {
                                for x in 0..PATCH {
                                    let src = ((row0 + y) * width + col0 + x) * 3 + c;
                                    dst[((c * TEMPORAL + t) * PATCH + y) * PATCH + x] =
                                        normalize(rgb[src], c, affine);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (out, (gh, gw))
}

/// Rounds an f32 patch matrix to bf16 (round-to-nearest-even).
pub fn to_bf16(values: &[f32]) -> Vec<bf16> {
    values.iter().map(|&v| bf16::from_f32(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(h: usize, w: usize) -> Vec<u8> {
        (0..h * w * 3).map(|i| (i * 7 % 251) as u8).collect()
    }

    #[test]
    fn shape_and_grid() {
        let (out, grid) = normalise_patchify(&image(56, 84), 56, 84, &fused_affine());
        assert_eq!(grid, (4, 6));
        assert_eq!(out.len(), 24 * PATCH_ELEMS);
    }

    #[test]
    fn frame_is_duplicated_along_t() {
        let (out, _) = normalise_patchify(&image(28, 28), 28, 28, &fused_affine());
        for patch in out.chunks(PATCH_ELEMS) {
            for c in 0..3 {
                let base = c * 2 * 196;
                assert_eq!(&patch[base..base + 196], &patch[base + 196..base + 392]);
            }
        }
    }

    #[test]
    fn patch_order_is_merge_window_major() {
        let (h, w) = (28, 56);
        let img = image(h, w);
        let a = fused_affine();
        let (out, _) = normalise_patchify(&img, h, w, &a);
        // patch index 1 = window (0,0), row 0, col 1 -> pixel origin (0, 14)
        let want = normalize(img[14 * 3 + 1], 1, &a);
        assert_eq!(out[PATCH_ELEMS + 2 * 196], want);
        // patch index 4 = window (0,1), row 0, col 0 -> pixel origin (0, 28)
        let want = normalize(img[28 * 3], 0, &a);
        assert_eq!(out[4 * PATCH_ELEMS], want);
        // patch index 2 = window (0,0), row 1, col 0 -> pixel origin (14, 0)
        let want = normalize(img[14 * w * 3 + 2], 2, &a);
        assert_eq!(out[2 * PATCH_ELEMS + 4 * 196], want);
    }

    #[test]
    fn normalisation_endpoints() {
        let a = fused_affine();
        assert!((normalize(0, 0, &a) + MEAN[0] / STD[0]).abs() < 1e-6);
        assert!((normalize(255, 2, &a) - (1.0 - MEAN[2]) / STD[2]).abs() < 1e-5);
    }

    #[test]
    fn bf16_rounds_nearest() {
        let v = to_bf16(&[1.0, 1.003, 1.006]);
        assert_eq!(v[0], bf16::from_f32(1.0));
        assert_eq!(v[1].to_f32(), 1.0);
        assert_eq!(v[2].to_f32(), 1.007_812_5);
    }
}
