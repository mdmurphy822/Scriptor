//! Attention references.

/// Softmax-weighted sum for one query against `keys` positions, in f64.
/// `score(j)` and `value(j)` give the logit and value row for key `j < n`.
fn attend(
    n: usize,
    score: impl Fn(usize) -> f64,
    value: impl Fn(usize) -> Vec<f32>,
    out: &mut [f32],
) {
    let scores: Vec<f64> = (0..n).map(&score).collect();
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let denom: f64 = exps.iter().sum();
    let mut acc = vec![0.0f64; out.len()];
    for (j, e) in exps.iter().enumerate() {
        for (a, v) in acc.iter_mut().zip(value(j)) {
            *a += e / denom * f64::from(v);
        }
    }
    for (o, a) in out.iter_mut().zip(acc) {
        *o = a as f32;
    }
}

/// Variable-length attention with grouped KV heads.
///
/// `q` is `[tokens, q_heads, head_dim]`, `k` and `v` are
/// `[tokens, kv_heads, head_dim]`. `cu_seqlens` lists sequence boundaries
/// (`[0, l0, l0+l1, ...]`); attention never crosses a boundary. With `causal`
/// query `i` sees keys `<= i` of its own sequence. Query head `h` reads KV head
/// `h / (q_heads / kv_heads)`. Logits are scaled by `scale`. Returns
/// `[tokens, q_heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn varlen_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cu_seqlens: &[usize],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    causal: bool,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(q_heads % kv_heads, 0);
    let tokens = *cu_seqlens.last().expect("cu_seqlens is non-empty");
    assert_eq!(q.len(), tokens * q_heads * head_dim);
    assert_eq!(k.len(), tokens * kv_heads * head_dim);
    assert_eq!(v.len(), k.len());
    let group = q_heads / kv_heads;
    let mut out = vec![0.0; q.len()];
    for span in cu_seqlens.windows(2) {
        let (start, end) = (span[0], span[1]);
        for i in start..end {
            let visible = if causal { i - start + 1 } else { end - start };
            for h in 0..q_heads {
                let kvh = h / group;
                let qrow = &q[(i * q_heads + h) * head_dim..][..head_dim];
                let dst = &mut out[(i * q_heads + h) * head_dim..][..head_dim];
                attend(
                    visible,
                    |j| {
                        let krow = &k[((start + j) * kv_heads + kvh) * head_dim..][..head_dim];
                        qrow.iter()
                            .zip(krow)
                            .map(|(a, b)| f64::from(*a) * f64::from(*b))
                            .sum::<f64>()
                            * f64::from(scale)
                    },
                    |j| v[((start + j) * kv_heads + kvh) * head_dim..][..head_dim].to_vec(),
                    dst,
                );
            }
        }
    }
    out
}

/// Paged KV cache view: `key_cache` and `value_cache` are
/// `[num_blocks, block_size, kv_heads, head_dim]`; `block_tables` is
/// `[num_seqs, max_blocks]` of block indices; `seq_lens[s]` is the number of
/// cached tokens of sequence `s` including the `q_len` tokens being decoded.
pub struct PagedKv<'a> {
    pub key_cache: &'a [f32],
    pub value_cache: &'a [f32],
    pub block_tables: &'a [u32],
    pub max_blocks: usize,
    pub seq_lens: &'a [usize],
    pub block_size: usize,
}

/// Paged decode attention for `q_len` query tokens per sequence (`q_len = 1`
/// plain decode, up to `k + 1` verifying drafts). `q` is
/// `[num_seqs, q_len, q_heads, head_dim]`. Query `j` of sequence `s` attends
/// positions `0 ..= seq_lens[s] - q_len + j` (in-step causal mask). Returns the
/// same shape as `q`.
pub fn paged_decode_attention(
    q: &[f32],
    kv: &PagedKv<'_>,
    q_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let num_seqs = kv.seq_lens.len();
    assert_eq!(q.len(), num_seqs * q_len * q_heads * head_dim);
    let group = q_heads / kv_heads;
    let mut out = vec![0.0; q.len()];
    for s in 0..num_seqs {
        assert!(kv.seq_lens[s] >= q_len);
        let table = &kv.block_tables[s * kv.max_blocks..][..kv.max_blocks];
        let row = |pos: usize, kvh: usize| -> usize {
            let block = table[pos / kv.block_size] as usize;
            ((block * kv.block_size + pos % kv.block_size) * kv_heads + kvh) * head_dim
        };
        for j in 0..q_len {
            let visible = kv.seq_lens[s] - q_len + j + 1;
            for h in 0..q_heads {
                let kvh = h / group;
                let at = ((s * q_len + j) * q_heads + h) * head_dim;
                let qrow = &q[at..at + head_dim];
                attend(
                    visible,
                    |p| {
                        let o = row(p, kvh);
                        qrow.iter()
                            .zip(&kv.key_cache[o..o + head_dim])
                            .map(|(a, b)| f64::from(*a) * f64::from(*b))
                            .sum::<f64>()
                            * f64::from(scale)
                    },
                    |p| {
                        let o = row(p, kvh);
                        kv.value_cache[o..o + head_dim].to_vec()
                    },
                    &mut out[at..at + head_dim],
                );
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed) * 0.37).sin()).collect()
    }

    #[test]
    fn single_key_returns_value() {
        let out = varlen_attention(
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[0, 1],
            1,
            1,
            2,
            true,
            1.0,
        );
        assert!((out[0] - 5.0).abs() < 1e-6 && (out[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn causal_first_token_ignores_future() {
        let (q, k, v) = (data(12, 1.0), data(12, 2.0), data(12, 3.0));
        let causal = varlen_attention(&q, &k, &v, &[0, 6], 1, 1, 2, true, 0.5);
        let first = varlen_attention(&q[..2], &k[..2], &v[..2], &[0, 1], 1, 1, 2, true, 0.5);
        assert_eq!(&causal[..2], &first[..]);
        let full = varlen_attention(&q, &k, &v, &[0, 6], 1, 1, 2, false, 0.5);
        assert!((full[0] - causal[0]).abs() > 1e-4);
        // last token sees everything either way
        assert!((full[10] - causal[10]).abs() < 1e-6);
    }

    #[test]
    fn sequences_are_independent() {
        let (q, k, v) = (data(16, 1.0), data(8, 2.0), data(8, 3.0));
        let joint = varlen_attention(&q, &k, &v, &[0, 3, 8], 2, 1, 1, false, 1.0);
        let a = varlen_attention(&q[..6], &k[..3], &v[..3], &[0, 3], 2, 1, 1, false, 1.0);
        let b = varlen_attention(&q[6..], &k[3..], &v[3..], &[0, 5], 2, 1, 1, false, 1.0);
        assert_eq!(&joint[..6], &a[..]);
        assert_eq!(&joint[6..], &b[..]);
    }

    #[test]
    fn gqa_heads_share_kv() {
        let k = data(4, 1.0);
        let v = data(4, 2.0);
        let (a, b) = (data(2, 3.0), data(2, 9.0));
        let q = [a.clone(), a, b.clone(), b].concat();
        let out = varlen_attention(&q, &k, &v, &[0, 2], 2, 1, 2, false, 1.0);
        // q heads at token 0 are identical, so are their outputs
        assert_eq!(&out[0..2], &out[2..4]);
        assert_eq!(&out[4..6], &out[6..8]);
    }

    #[test]
    fn paged_matches_contiguous_causal() {
        let (heads, kvh, d, block) = (2, 1, 4, 4);
        let len = 10usize;
        let k = data(len * kvh * d, 5.0);
        let v = data(len * kvh * d, 6.0);
        let q = data(len * heads * d, 7.0);
        let want = varlen_attention(&q, &k, &v, &[0, len], heads, kvh, d, true, 0.5);

        // scatter K/V into shuffled blocks
        let blocks = len.div_ceil(block);
        let table: Vec<u32> = (0..blocks as u32).rev().collect();
        let mut kc = vec![0.0; blocks * block * kvh * d];
        let mut vc = kc.clone();
        for p in 0..len {
            let b = table[p / block] as usize;
            let dst = ((b * block + p % block) * kvh) * d;
            kc[dst..dst + kvh * d].copy_from_slice(&k[p * kvh * d..(p + 1) * kvh * d]);
            vc[dst..dst + kvh * d].copy_from_slice(&v[p * kvh * d..(p + 1) * kvh * d]);
        }
        let kv = PagedKv {
            key_cache: &kc,
            value_cache: &vc,
            block_tables: &table,
            max_blocks: blocks,
            seq_lens: &[len],
            block_size: block,
        };
        // verify the last 3 tokens as one step
        let q_len = 3;
        let got = paged_decode_attention(
            &q[(len - q_len) * heads * d..],
            &kv,
            q_len,
            heads,
            kvh,
            d,
            0.5,
        );
        for (a, b) in got.iter().zip(&want[(len - q_len) * heads * d..]) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
