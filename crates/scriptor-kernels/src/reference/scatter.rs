//! Embedding gather, image scatter and speculative accept references.

/// Gathers rows of `table` (`[vocab, dim]`) for `ids`; returns `[ids.len(), dim]`.
/// Panics if an id is outside the table.
pub fn embedding_gather(table: &[f32], dim: usize, ids: &[u32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(ids.len() * dim);
    for &id in ids {
        out.extend_from_slice(&table[id as usize * dim..(id as usize + 1) * dim]);
    }
    out
}

/// Overwrites rows of `embeds` (`[tokens, dim]`) at `positions[i]` with row `i`
/// of `image_embeds` (`[positions.len(), dim]`).
pub fn image_scatter(embeds: &mut [f32], dim: usize, image_embeds: &[f32], positions: &[usize]) {
    assert_eq!(image_embeds.len(), positions.len() * dim);
    for (i, &pos) in positions.iter().enumerate() {
        embeds[pos * dim..(pos + 1) * dim].copy_from_slice(&image_embeds[i * dim..(i + 1) * dim]);
    }
}

/// Positions in `ids` equal to `placeholder`, in order.
pub fn placeholder_positions(ids: &[u32], placeholder: u32) -> Vec<usize> {
    ids.iter()
        .enumerate()
        .filter(|&(_, &t)| t == placeholder)
        .map(|(i, _)| i)
        .collect()
}

/// Speculative accept. `draft` holds `k` proposed tokens, `target` the
/// verifier's greedy token at each of the `k + 1` positions (position `i`
/// conditioned on the first `i` drafts). Returns the emitted tokens: the
/// longest prefix of `draft` matching `target`, followed by the verifier's
/// token at the first mismatch (or the bonus token when all match).
pub fn mtp_accept(draft: &[u32], target: &[u32]) -> Vec<u32> {
    assert_eq!(target.len(), draft.len() + 1);
    let accepted = draft.iter().zip(target).take_while(|(d, t)| d == t).count();
    target[..=accepted].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_rows() {
        let table = [0.0, 0.1, 1.0, 1.1, 2.0, 2.1];
        assert_eq!(
            embedding_gather(&table, 2, &[2, 0, 2]),
            vec![2.0, 2.1, 0.0, 0.1, 2.0, 2.1]
        );
    }

    #[test]
    fn scatter_overwrites_only_positions() {
        let mut e = vec![0.0; 8];
        image_scatter(&mut e, 2, &[1.0, 1.0, 2.0, 2.0], &[1, 3]);
        assert_eq!(e, vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0]);
    }

    #[test]
    fn placeholders_found() {
        assert_eq!(placeholder_positions(&[1, 9, 9, 2, 9], 9), vec![1, 2, 4]);
    }

    #[test]
    fn accept_all_gives_bonus() {
        assert_eq!(mtp_accept(&[5, 6], &[5, 6, 7]), vec![5, 6, 7]);
    }

    #[test]
    fn accept_stops_at_first_mismatch() {
        assert_eq!(mtp_accept(&[5, 9, 7], &[5, 6, 7, 8]), vec![5, 6]);
        assert_eq!(mtp_accept(&[1], &[2, 3]), vec![2]);
        assert_eq!(mtp_accept(&[], &[4]), vec![4]);
    }
}
