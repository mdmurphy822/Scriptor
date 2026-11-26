//! Activation references.

/// Error function in f64 (power series below 3, continued fraction above).
pub fn erf_f64(x: f64) -> f64 {
    let a = x.abs();
    let value = if a < 3.0 {
        let x2 = a * a;
        let mut term = a;
        let mut sum = a;
        let mut n = 0.0;
        while term.abs() > 1e-17 * sum.abs() {
            n += 1.0;
            term *= -x2 / n;
            sum += term / (2.0 * n + 1.0);
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        // erfc(a) = exp(-a^2)/sqrt(pi) / (a + (1/2)/(a + 1/(a + (3/2)/(a + ...))))
        let mut frac = a;
        for k in (1..=80).rev() {
            frac = a + (f64::from(k) / 2.0) / frac;
        }
        1.0 - (-a * a).exp() / std::f64::consts::PI.sqrt() / frac
    };
    value.copysign(x)
}

/// Exact (erf) GELU: `x * 0.5 * (1 + erf(x / sqrt(2)))`.
pub fn gelu_erf(x: f32) -> f32 {
    let x = f64::from(x);
    (x * 0.5 * (1.0 + erf_f64(x / std::f64::consts::SQRT_2))) as f32
}

/// SiLU: `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    let x = f64::from(x);
    (x / (1.0 + (-x).exp())) as f32
}

/// SiLU-and-mul on separate operands: `silu(gate) * up`, elementwise.
pub fn silu_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    gate.iter().zip(up).map(|(&g, &u)| silu(g) * u).collect()
}

/// SiLU-and-mul on a fused projection output `[rows, 2 * width]` whose first
/// `width` columns are the gate and last `width` columns are the up branch.
/// Returns `[rows, width]`.
pub fn silu_mul_fused(gate_up: &[f32], width: usize) -> Vec<f32> {
    assert_eq!(gate_up.len() % (2 * width), 0);
    let mut out = Vec::with_capacity(gate_up.len() / 2);
    for row in gate_up.chunks_exact(2 * width) {
        let (gate, up) = row.split_at(width);
        out.extend(gate.iter().zip(up).map(|(&g, &u)| silu(g) * u));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_known_values() {
        let cases = [
            (0.0, 0.0),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (3.0, 0.999_977_909_503_001_4),
            (4.0, 0.999_999_984_582_742_1),
        ];
        for (x, e) in cases {
            assert!((erf_f64(x) - e).abs() < 1e-12, "erf({x})");
            assert!((erf_f64(-x) + e).abs() < 1e-12, "erf(-{x})");
        }
    }

    #[test]
    fn gelu_values() {
        assert_eq!(gelu_erf(0.0), 0.0);
        assert!((gelu_erf(1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((gelu_erf(-1.0) + 0.158_655_25).abs() < 1e-6);
        assert!((gelu_erf(10.0) - 10.0).abs() < 1e-6);
        assert!(gelu_erf(-10.0).abs() < 1e-6);
    }

    #[test]
    fn silu_values() {
        assert_eq!(silu(0.0), 0.0);
        assert!((silu(1.0) - 0.731_058_6).abs() < 1e-6);
        assert!((silu(-1.0) + 0.268_941_4).abs() < 1e-6);
    }

    #[test]
    fn fused_split_matches_separate() {
        let gate_up = [1.0, -1.0, 2.0, 3.0, 0.5, 0.25, -2.0, 4.0];
        let fused = silu_mul_fused(&gate_up, 2);
        let a = silu_mul(&[1.0, -1.0], &[2.0, 3.0]);
        let b = silu_mul(&[0.5, 0.25], &[-2.0, 4.0]);
        assert_eq!(fused, [a, b].concat());
    }
}
