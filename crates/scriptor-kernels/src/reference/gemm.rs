//! GEMM reference.

/// Row-major `C = A * W^T + bias`: `a` is `[m, k]`, `w` is `[n, k]`, optional
/// `bias` is `[n]`; result `[m, n]`. Accumulates in f64.
pub fn gemm_bias(
    a: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(w.len(), n * k);
    if let Some(b) = bias {
        assert_eq!(b.len(), n);
    }
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        let arow = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let wrow = &w[j * k..(j + 1) * k];
            let acc: f64 = arow
                .iter()
                .zip(wrow)
                .map(|(x, y)| f64::from(*x) * f64::from(*y))
                .sum();
            c[i * n + j] = acc as f32 + bias.map_or(0.0, |b| b[j]);
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_product_with_bias() {
        // A = [[1,2],[3,4]], W = [[1,0],[0,1],[1,1]]
        let c = gemm_bias(
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            Some(&[10.0, 20.0, 30.0]),
            2,
            3,
            2,
        );
        assert_eq!(c, vec![11.0, 22.0, 33.0, 13.0, 24.0, 37.0]);
    }

    #[test]
    fn no_bias() {
        let c = gemm_bias(&[2.0], &[3.0], None, 1, 1, 1);
        assert_eq!(c, vec![6.0]);
    }
}
