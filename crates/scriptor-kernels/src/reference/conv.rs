//! Patch-embed and 2x2 downsample as GEMMs over the patchify order.

use super::gemm::gemm_bias;

/// Elements per flattened patch: `3 channels x 2 frames x 14 x 14`.
pub const PATCH_ELEMS: usize = 3 * 2 * 14 * 14;

/// Patch embedding: `Conv3d(kernel = stride = (2, 14, 14))` over patches
/// flattened as `(c, t, 14, 14)` is a GEMM. `patches` is `[n, 1176]`,
/// `weight` is `[hidden, 3, 2, 14, 14]` flattened to `[hidden, 1176]`, bias is
/// `[hidden]`. Returns `[n, hidden]`.
pub fn patch_embed(patches: &[f32], weight: &[f32], bias: &[f32], hidden: usize) -> Vec<f32> {
    assert_eq!(patches.len() % PATCH_ELEMS, 0);
    gemm_bias(
        patches,
        weight,
        Some(bias),
        patches.len() / PATCH_ELEMS,
        hidden,
        PATCH_ELEMS,
    )
}

/// 2x2 stride-2 downsample `Conv2d(in -> out, k = 2, s = 2)` over merge-window
/// ordered patch rows. `x` is `[n, in_dim]` where each run of 4 rows is one
/// window in `(row, col)` order; `weight` is `[out, in_dim, 2, 2]`; bias is
/// `[out]`. Returns `[n / 4, out]`.
pub fn downsample_2x2(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    assert_eq!(x.len() % (4 * in_dim), 0);
    assert_eq!(weight.len(), out_dim * in_dim * 4);
    let windows = x.len() / (4 * in_dim);
    // A[g, c * 4 + dy * 2 + dx] = x[4g + dy * 2 + dx, c], matching the weight's (c, dy, dx) flattening.
    let mut a = vec![0.0; windows * in_dim * 4];
    for g in 0..windows {
        for slot in 0..4 {
            for c in 0..in_dim {
                a[g * in_dim * 4 + c * 4 + slot] = x[(g * 4 + slot) * in_dim + c];
            }
        }
    }
    gemm_bias(&a, weight, Some(bias), windows, out_dim, in_dim * 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_embed_matches_direct_conv() {
        let hidden = 3;
        let patches: Vec<f32> = (0..2 * PATCH_ELEMS)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
            .collect();
        let weight: Vec<f32> = (0..hidden * PATCH_ELEMS)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
            .collect();
        let bias = [0.5, -0.5, 0.0];
        let got = patch_embed(&patches, &weight, &bias, hidden);
        for p in 0..2 {
            for o in 0..hidden {
                let mut acc = f64::from(bias[o]);
                for c in 0..3 {
                    for t in 0..2 {
                        for y in 0..14 {
                            for x in 0..14 {
                                let idx = ((c * 2 + t) * 14 + y) * 14 + x;
                                acc += f64::from(patches[p * PATCH_ELEMS + idx])
                                    * f64::from(weight[o * PATCH_ELEMS + idx]);
                            }
                        }
                    }
                }
                assert!((f64::from(got[p * hidden + o]) - acc).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn downsample_matches_direct_conv() {
        let (in_dim, out_dim, windows) = (3, 2, 2);
        let x: Vec<f32> = (0..windows * 4 * in_dim).map(|i| i as f32 * 0.1).collect();
        let w: Vec<f32> = (0..out_dim * in_dim * 4)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.2)
            .collect();
        let b = [1.0, -1.0];
        let got = downsample_2x2(&x, &w, &b, in_dim, out_dim);
        for g in 0..windows {
            for o in 0..out_dim {
                let mut acc = f64::from(b[o]);
                for c in 0..in_dim {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            acc += f64::from(x[(g * 4 + dy * 2 + dx) * in_dim + c])
                                * f64::from(w[((o * in_dim + c) * 2 + dy) * 2 + dx]);
                        }
                    }
                }
                assert!((f64::from(got[g * out_dim + o]) - acc).abs() < 1e-5);
            }
        }
    }
}
