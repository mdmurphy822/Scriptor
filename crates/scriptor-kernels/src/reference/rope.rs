//! Rotary embedding references.

/// Multimodal section widths over the 64 frequency pairs of a 128-wide head:
/// temporal, height, width.
pub const MROPE_SECTIONS: [usize; 3] = [16, 24, 24];

/// GPT-J (interleaved-pair) M-RoPE applied in place to `x` shaped
/// `[tokens, heads, head_dim]`.
///
/// Pair `i` rotates elements `(2i, 2i+1)` by `position * theta^(-2i/head_dim)`.
/// Pairs `0..s0` take their position from component 0 (t), the next `s1` from
/// component 1 (h), the last `s2` from component 2 (w); `sections` must sum to
/// `head_dim / 2`. `positions` holds `[t, h, w]` per token.
pub fn mrope_gptj(
    x: &mut [f32],
    positions: &[[i32; 3]],
    heads: usize,
    head_dim: usize,
    theta: f64,
    sections: [usize; 3],
) {
    assert_eq!(sections.iter().sum::<usize>() * 2, head_dim);
    assert_eq!(x.len(), positions.len() * heads * head_dim);
    for (token, pos) in positions.iter().enumerate() {
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            for pair in 0..head_dim / 2 {
                let component = if pair < sections[0] {
                    0
                } else if pair < sections[0] + sections[1] {
                    1
                } else {
                    2
                };
                let freq = theta.powf(-2.0 * pair as f64 / head_dim as f64);
                let angle = f64::from(pos[component]) * freq;
                let (sin, cos) = angle.sin_cos();
                let a = f64::from(x[base + 2 * pair]);
                let b = f64::from(x[base + 2 * pair + 1]);
                x[base + 2 * pair] = (a * cos - b * sin) as f32;
                x[base + 2 * pair + 1] = (a * sin + b * cos) as f32;
            }
        }
    }
}

/// Vision 2-D axial NeoX RoPE applied in place to `x` shaped
/// `[tokens, heads, head_dim]`.
///
/// Pair `j` in `0..head_dim/2` rotates elements `(j, j + head_dim/2)`. The
/// first half of the pairs use the patch row index `h`, the second half the
/// column index `w`; within each half pair `k` has frequency
/// `theta^(-2k/(head_dim/2))`. `positions` holds `[h, w]` per token.
pub fn rope_vision_2d(
    x: &mut [f32],
    positions: &[[i32; 2]],
    heads: usize,
    head_dim: usize,
    theta: f64,
) {
    let half = head_dim / 2;
    let quarter = half / 2;
    assert_eq!(x.len(), positions.len() * heads * head_dim);
    for (token, pos) in positions.iter().enumerate() {
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            for pair in 0..half {
                let (axis, k) = if pair < quarter {
                    (0, pair)
                } else {
                    (1, pair - quarter)
                };
                let freq = theta.powf(-2.0 * k as f64 / half as f64);
                let angle = f64::from(pos[axis]) * freq;
                let (sin, cos) = angle.sin_cos();
                let a = f64::from(x[base + pair]);
                let b = f64::from(x[base + pair + half]);
                x[base + pair] = (a * cos - b * sin) as f32;
                x[base + pair + half] = (a * sin + b * cos) as f32;
            }
        }
    }
}

/// Positions of the merge-window patch order for one `h x w` patch grid:
/// windows row-major, then the 2x2 inside each window row-major. Each entry is
/// `[row, col]`. Repeated `t` times for multi-frame grids.
pub fn vision_positions(t: usize, h: usize, w: usize) -> Vec<[i32; 2]> {
    assert!(h.is_multiple_of(2) && w.is_multiple_of(2));
    let mut one = Vec::with_capacity(h * w);
    for wr in 0..h / 2 {
        for wc in 0..w / 2 {
            for dr in 0..2 {
                for dc in 0..2 {
                    one.push([(wr * 2 + dr) as i32, (wc * 2 + dc) as i32]);
                }
            }
        }
    }
    (0..t).flat_map(|_| one.iter().copied()).collect()
}

/// M-RoPE positions for a prompt with one image: text before the image gets
/// `t = h = w = i`; the image's `gh x gw` merged-token grid gets
/// `(start, start + row, start + col)` where `start` is the text length so far;
/// text after resumes at `start + max(gh, gw)`.
pub fn mrope_positions_one_image(
    before: usize,
    grid_h: usize,
    grid_w: usize,
    after: usize,
) -> Vec<[i32; 3]> {
    let mut out = Vec::with_capacity(before + grid_h * grid_w + after);
    for i in 0..before {
        out.push([i as i32; 3]);
    }
    let start = before as i32;
    for r in 0..grid_h {
        for c in 0..grid_w {
            out.push([start, start + r as i32, start + c as i32]);
        }
    }
    let resume = start + grid_h.max(grid_w) as i32;
    for i in 0..after {
        out.push([resume + i as i32; 3]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn mrope_position_zero_is_identity() {
        let mut x: Vec<f32> = (0..2 * 128).map(|i| i as f32 * 0.01).collect();
        let before = x.clone();
        mrope_gptj(&mut x, &[[0, 0, 0]], 2, 128, 10_000.0, MROPE_SECTIONS);
        assert_eq!(x, before);
    }

    #[test]
    fn mrope_matches_hand_rotation_and_preserves_norm() {
        let mut x = vec![0.0f32; 128];
        x[0] = 1.0;
        x[40] = 1.0; // pair 20, in the height section
        x[100] = 1.0; // pair 50, in the width section
        let before = norm(&x);
        mrope_gptj(&mut x, &[[3, 5, 7]], 1, 128, 10_000.0, MROPE_SECTIONS);
        assert!((x[0] - 3f32.cos()).abs() < 1e-6 && (x[1] - 3f32.sin()).abs() < 1e-6);
        let f20 = 10_000f64.powf(-40.0 / 128.0);
        assert!((f64::from(x[40]) - (5.0 * f20).cos()).abs() < 1e-6);
        assert!((f64::from(x[41]) - (5.0 * f20).sin()).abs() < 1e-6);
        let f50 = 10_000f64.powf(-100.0 / 128.0);
        assert!((f64::from(x[100]) - (7.0 * f50).cos()).abs() < 1e-6);
        assert!((norm(&x) - before).abs() < 1e-5);
    }

    #[test]
    fn mrope_text_positions_equal_single_axis() {
        let mut a: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        let mut b = a.clone();
        mrope_gptj(&mut a, &[[9, 9, 9]], 1, 128, 10_000.0, MROPE_SECTIONS);
        mrope_gptj(&mut b, &[[9, 9, 9]], 1, 128, 10_000.0, [64, 0, 0]);
        for (p, q) in a.iter().zip(&b) {
            assert!((p - q).abs() < 1e-6);
        }
    }

    #[test]
    fn vision_rope_axes_and_norm() {
        let mut x = vec![0.0f32; 64];
        x[0] = 1.0; // pair 0 (h axis, freq 1): partner x[32]
        x[16] = 1.0; // pair 16 (w axis, freq 1): partner x[48]
        let before = norm(&x);
        rope_vision_2d(&mut x, &[[2, 5]], 1, 64, 10_000.0);
        assert!((x[0] - 2f32.cos()).abs() < 1e-6 && (x[32] - 2f32.sin()).abs() < 1e-6);
        assert!((x[16] - 5f32.cos()).abs() < 1e-6 && (x[48] - 5f32.sin()).abs() < 1e-6);
        assert!((norm(&x) - before).abs() < 1e-5);
    }

    #[test]
    fn vision_positions_follow_merge_windows() {
        let p = vision_positions(1, 4, 4);
        assert_eq!(&p[..4], &[[0, 0], [0, 1], [1, 0], [1, 1]]);
        assert_eq!(&p[4..8], &[[0, 2], [0, 3], [1, 2], [1, 3]]);
        assert_eq!(&p[8..12], &[[2, 0], [2, 1], [3, 0], [3, 1]]);
        assert_eq!(vision_positions(2, 2, 2).len(), 8);
    }

    #[test]
    fn mrope_prompt_positions() {
        let p = mrope_positions_one_image(3, 2, 3, 2);
        assert_eq!(p[2], [2, 2, 2]);
        assert_eq!(p[3], [3, 3, 3]);
        assert_eq!(p[5], [3, 3, 5]);
        assert_eq!(p[8], [3, 4, 5]);
        assert_eq!(p[9], [6, 6, 6]);
        assert_eq!(p[10], [7, 7, 7]);
    }
}
