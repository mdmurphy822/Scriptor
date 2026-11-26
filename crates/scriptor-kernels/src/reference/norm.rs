//! Normalisation references.

use super::activation::gelu_erf;

/// Plain RMSNorm over rows of `dim`: `y = x / sqrt(mean(x^2) + eps) * weight`.
/// `x` is `[rows, dim]`; the mean and root are computed in f64 then rounded.
pub fn rms_norm(x: &[f32], weight: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    assert_eq!(weight.len(), dim);
    assert_eq!(x.len() % dim, 0);
    let mut out = vec![0.0; x.len()];
    for (row, dst) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let mean_sq = row
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / dim as f64;
        let inv = 1.0 / (mean_sq + f64::from(eps)).sqrt();
        for ((d, &v), &w) in dst.iter_mut().zip(row).zip(weight) {
            *d = (f64::from(v) * inv) as f32 * w;
        }
    }
    out
}

/// Fused residual add and RMSNorm. Returns `(sum, normed)` where
/// `sum = x + residual` and `normed = rms_norm(sum)`; both are `[rows, dim]`.
pub fn add_rms_norm(
    x: &[f32],
    residual: &[f32],
    weight: &[f32],
    dim: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(x.len(), residual.len());
    let sum: Vec<f32> = x.iter().zip(residual).map(|(a, b)| a + b).collect();
    let normed = rms_norm(&sum, weight, dim, eps);
    (sum, normed)
}

/// Per-head RMSNorm: `x` is `[tokens, heads, head_dim]`, `weight` is `[head_dim]`
/// shared by every head.
pub fn head_rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    rms_norm(x, weight, weight.len(), eps)
}

/// LayerNorm over rows of `dim` with weight and bias, followed by erf-GELU.
pub fn layer_norm_gelu(x: &[f32], weight: &[f32], bias: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    assert_eq!(weight.len(), dim);
    assert_eq!(bias.len(), dim);
    assert_eq!(x.len() % dim, 0);
    let mut out = vec![0.0; x.len()];
    for (row, dst) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let n = dim as f64;
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = row
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        let inv = 1.0 / (var + f64::from(eps)).sqrt();
        for (i, d) in dst.iter_mut().enumerate() {
            let normed = ((f64::from(row[i]) - mean) * inv) as f32 * weight[i] + bias[i];
            *d = gelu_erf(normed);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_unit_weight_has_unit_rms() {
        let x = [1.0, -2.0, 3.0, 4.0, 0.5, 0.5, -0.5, -0.5];
        let y = rms_norm(&x, &[1.0; 4], 4, 0.0);
        for row in y.chunks(4) {
            let rms = (row.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
            assert!((rms - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn rms_norm_applies_weight_and_eps() {
        let y = rms_norm(&[2.0, 2.0], &[3.0, 0.5], 2, 0.0);
        assert!((y[0] - 3.0).abs() < 1e-6 && (y[1] - 0.5).abs() < 1e-6);
        let z = rms_norm(&[0.0, 0.0], &[1.0, 1.0], 2, 1e-5);
        assert_eq!(z, vec![0.0, 0.0]);
    }

    #[test]
    fn fused_matches_separate() {
        let x = [0.3, -1.0, 2.0, 0.1, 0.7, 0.2];
        let r = [1.0, 1.0, -1.0, 0.5, -0.5, 0.0];
        let w = [1.0, 2.0, 0.5];
        let (sum, normed) = add_rms_norm(&x, &r, &w, 3, 1e-5);
        let expect_sum: Vec<f32> = x.iter().zip(&r).map(|(a, b)| a + b).collect();
        assert_eq!(sum, expect_sum);
        assert_eq!(normed, rms_norm(&expect_sum, &w, 3, 1e-5));
    }

    #[test]
    fn head_norm_is_per_head() {
        let x = [1.0, 1.0, 10.0, 10.0];
        let y = head_rms_norm(&x, &[1.0, 1.0], 0.0);
        for v in y {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn layer_norm_gelu_zero_mean_input() {
        let x = [-1.0, 1.0];
        let y = layer_norm_gelu(&x, &[1.0, 1.0], &[0.0, 0.0], 2, 0.0);
        // normed = [-1, 1]; gelu(1) = 0.8413447, gelu(-1) = -0.1586553
        assert!((y[0] + 0.158_655_3).abs() < 1e-5);
        assert!((y[1] - 0.841_344_7).abs() < 1e-5);
    }
}
