//! Multi-token-prediction drafting: the MTP layer's input rows, the shifted
//! prompt ids that fill its KV, and the greedy draft chain.

use super::{SpecModel, SpecSequence};
use crate::error::EngineError;
use crate::sequence::RequestId;
use scriptor_kernels::reference::sampling::argmax_with_margin;
use std::collections::BTreeMap;

/// One MTP input position: the token that follows it and the main model's
/// hidden state at it.
#[derive(Debug, Clone)]
pub struct MtpFeed<H> {
    /// Token at the following position (the MTP input id).
    pub token: u32,
    /// Hidden state of the main model, or of the MTP layer inside a chain.
    pub hidden: H,
}

/// MTP forward over consecutive positions of one sequence.
#[derive(Debug, Clone)]
pub struct MtpRows<H> {
    pub seq: RequestId,
    /// Sequence position of the first row. Rows overwrite the MTP KV from
    /// here on and discard anything after the last row. The embedding of the
    /// row at position zero is zeroed.
    pub start_position: usize,
    pub rows: Vec<MtpFeed<H>>,
}

/// Result of an MTP forward for the last row of one sequence.
#[derive(Debug, Clone)]
pub struct MtpOutput<H> {
    /// Draft-head logits at the last row.
    pub logits: Vec<f32>,
    /// MTP layer output at the last row, fed to the next chain step.
    pub hidden: H,
}

/// MTP outputs of one forward in input order, with its measured duration.
#[derive(Debug, Clone)]
pub struct MtpBatch<H> {
    pub outputs: Vec<MtpOutput<H>>,
    /// Device time of the forward measured with CUDA events.
    pub seconds: f64,
}

/// Ids fed to the MTP layer when it prefills a prompt: the prompt shifted left
/// by one with the first generated token appended. Image placeholder ids stay
/// in place, so the layer sees the placeholder embedding at image positions.
pub fn shifted_ids(prompt: &[u32], first_generated: u32) -> Vec<u32> {
    prompt
        .iter()
        .skip(1)
        .copied()
        .chain(std::iter::once(first_generated))
        .collect()
}

/// Drafts per sequence and the `(sequences, seconds)` of each MTP pass.
pub(crate) struct DraftChain {
    pub(crate) drafts: Vec<Vec<u32>>,
    pub(crate) passes: Vec<(usize, f64)>,
}

/// Runs the greedy draft chain for the sequences in `ids`, `depth[i]` drafts
/// for `ids[i]`. The first pass consumes each sequence's pending feeds (all
/// positions the MTP KV lacks); later passes feed the previous draft with the
/// layer's own hidden state. Returns the drafts per sequence and the
/// `(sequences, seconds)` of each pass.
pub(crate) fn draft_chain<M: SpecModel>(
    model: &mut M,
    seqs: &mut BTreeMap<RequestId, SpecSequence<M::Hidden>>,
    ids: &[RequestId],
    depth: &[usize],
) -> Result<DraftChain, EngineError> {
    let mut drafts: Vec<Vec<u32>> = vec![Vec::new(); ids.len()];
    let mut passes = Vec::new();
    let max_depth = depth.iter().copied().max().unwrap_or(0);
    let mut next_position: Vec<usize> = vec![0; ids.len()];
    let mut carry: Vec<Option<M::Hidden>> = vec![None; ids.len()];
    for pass in 0..max_depth {
        let active: Vec<usize> = (0..ids.len()).filter(|&i| depth[i] > pass).collect();
        let mut groups = Vec::with_capacity(active.len());
        for &i in &active {
            if pass == 0 {
                let seq = seqs.get_mut(&ids[i]).expect("live sequence");
                let rows = std::mem::take(&mut seq.pending);
                let start = seq.mtp_len;
                seq.mtp_len += rows.len();
                next_position[i] = seq.mtp_len;
                groups.push(MtpRows {
                    seq: ids[i],
                    start_position: start,
                    rows,
                });
            } else {
                groups.push(MtpRows {
                    seq: ids[i],
                    start_position: next_position[i],
                    rows: vec![MtpFeed {
                        token: *drafts[i].last().expect("previous pass drafted"),
                        hidden: carry[i].take().expect("previous pass output"),
                    }],
                });
                next_position[i] += 1;
            }
        }
        let batch = model.mtp_forward(&groups)?;
        if batch.outputs.len() != active.len() {
            return Err(EngineError::Executor(format!(
                "mtp_forward returned {} outputs for {} sequences",
                batch.outputs.len(),
                active.len()
            )));
        }
        passes.push((active.len(), batch.seconds));
        for (&i, out) in active.iter().zip(batch.outputs) {
            drafts[i].push(argmax_with_margin(&out.logits).token);
            carry[i] = Some(out.hidden);
        }
    }
    Ok(DraftChain { drafts, passes })
}
