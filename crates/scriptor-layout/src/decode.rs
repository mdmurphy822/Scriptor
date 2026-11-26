//! Raw model outputs to ranked page-pixel detections.

use std::collections::BTreeMap;

use crate::model::{LayoutError, RawOutputs};

/// Score thresholds applied while decoding.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeConfig {
    /// Global score threshold.
    pub threshold: f32,
    /// Per-class thresholds by class id; classes not listed use
    /// [`DecodeConfig::threshold`].
    pub threshold_by_class: BTreeMap<usize, f32>,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            threshold_by_class: BTreeMap::new(),
        }
    }
}

impl DecodeConfig {
    /// The threshold used for the first-stage cut: the lowest of the global
    /// and every per-class threshold.
    #[must_use]
    pub fn pre_threshold(&self) -> f32 {
        self.threshold_by_class
            .values()
            .copied()
            .fold(self.threshold, f32::min)
    }

    fn threshold_for(&self, class: usize) -> f32 {
        self.threshold_by_class
            .get(&class)
            .copied()
            .unwrap_or(self.threshold)
    }
}

/// One detected region before postprocessing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// Class probability.
    pub score: f32,
    /// Class id.
    pub label: usize,
    /// `(x1, y1, x2, y2)` in page pixels.
    pub bbox: [f32; 4],
    /// Reading-order rank among all queries (0 is first).
    pub order: usize,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Ranks every query in reading order from the pairwise order logits.
///
/// Query `b` collects the votes `sum_{a<b} s[a][b] + sum_{a>b} (1 - s[b][a])`
/// with `s = sigmoid(order_logits)`; queries are ranked by ascending votes and
/// the returned vector holds each query's rank.
///
/// # Panics
///
/// Panics when `order_logits` holds fewer than `queries * queries` values.
#[must_use]
pub fn order_ranks(order_logits: &[f32], queries: usize) -> Vec<usize> {
    let s = |row: usize, col: usize| sigmoid(order_logits[row * queries + col]);
    let votes: Vec<f32> = (0..queries)
        .map(|b| {
            let upper: f32 = (0..b).map(|a| s(a, b)).sum();
            let lower: f32 = (b + 1..queries).map(|a| 1.0 - s(b, a)).sum();
            upper + lower
        })
        .collect();
    let mut pointers: Vec<usize> = (0..queries).collect();
    pointers.sort_by(|&x, &y| votes[x].total_cmp(&votes[y]));
    let mut ranks = vec![0usize; queries];
    for (rank, &query) in pointers.iter().enumerate() {
        ranks[query] = rank;
    }
    ranks
}

/// Decodes one page.
///
/// Class scores are `sigmoid(logits)`; the top `queries` entries of the
/// flattened `queries x classes` score matrix are taken (a query may appear
/// with several classes), boxes are converted from `(cx, cy, w, h)` to
/// `(x1, y1, x2, y2)` scaled by the page size, entries scoring below
/// [`DecodeConfig::pre_threshold`] are dropped, and the result is ordered by
/// reading-order rank. When per-class thresholds are set, each entry is then
/// held to its own class threshold.
///
/// # Errors
///
/// Returns [`LayoutError::Shape`] when the tensors do not match their
/// declared dimensions.
pub fn decode_page(
    raw: &RawOutputs,
    page_width: u32,
    page_height: u32,
    config: &DecodeConfig,
) -> Result<Vec<Detection>, LayoutError> {
    raw.validate()?;
    let (queries, classes) = (raw.queries, raw.classes);
    let ranks = order_ranks(&raw.order_logits, queries);
    let mut scored: Vec<(f32, usize)> = raw
        .logits
        .iter()
        .enumerate()
        .map(|(i, &logit)| (sigmoid(logit), i))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.truncate(queries);
    let (w, h) = (page_width as f32, page_height as f32);
    let pre = config.pre_threshold();
    let mut kept: Vec<Detection> = scored
        .into_iter()
        .filter(|&(score, _)| score >= pre)
        .map(|(score, flat)| {
            let (label, query) = (flat % classes, flat / classes);
            let b = &raw.pred_boxes[query * 4..query * 4 + 4];
            Detection {
                score,
                label,
                bbox: [
                    (b[0] - 0.5 * b[2]) * w,
                    (b[1] - 0.5 * b[3]) * h,
                    (b[0] + 0.5 * b[2]) * w,
                    (b[1] + 0.5 * b[3]) * h,
                ],
                order: ranks[query],
            }
        })
        .collect();
    kept.sort_by_key(|d| d.order);
    if !config.threshold_by_class.is_empty() {
        kept.retain(|d| d.score >= config.threshold_for(d.label));
    }
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_follow_pairwise_order_votes() {
        // s[a][b] high means a precedes b. Order: query 2, then 0, then 1.
        let n = 3;
        let mut logits = vec![0.0f32; n * n];
        let mut set = |a: usize, b: usize, before: bool| {
            let (lo, hi) = (a.min(b), a.max(b));
            logits[lo * n + hi] = if (a < b) == before { 8.0 } else { -8.0 };
        };
        set(2, 0, true);
        set(2, 1, true);
        set(0, 1, true);
        assert_eq!(order_ranks(&logits, n), vec![1, 2, 0]);
    }

    #[test]
    fn decode_scales_boxes_thresholds_and_orders() {
        let mut raw = RawOutputs::blank(4, 3);
        raw.set_query(0, 1, 3.0, [0.5, 0.5, 0.2, 0.4]);
        raw.set_query(2, 2, 3.0, [0.25, 0.25, 0.1, 0.1]);
        raw.set_query(3, 0, -2.0, [0.5; 4]); // sigmoid(-2) = 0.12 < 0.3
        raw.set_order(0, 2, -8.0); // query 2 precedes query 0
        let out = decode_page(&raw, 200, 100, &DecodeConfig::default()).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].label, out[0].order), (2, 0));
        assert_eq!(out[1].label, 1);
        for (got, want) in out[1].bbox.iter().zip([80.0, 30.0, 120.0, 70.0]) {
            assert!((got - want).abs() < 1e-3);
        }
        assert!(out[0].score > 0.95);
    }

    #[test]
    fn same_query_can_fire_for_two_classes() {
        let mut raw = RawOutputs::blank(2, 3);
        raw.set_query(0, 0, 2.0, [0.5; 4]);
        raw.logits[2] = 1.0;
        let out = decode_page(&raw, 10, 10, &DecodeConfig::default()).unwrap();
        let labels: Vec<_> = out.iter().map(|d| d.label).collect();
        assert_eq!(labels, [0, 2]);
    }

    #[test]
    fn per_class_thresholds_override_the_global_one() {
        let mut raw = RawOutputs::blank(3, 2);
        raw.set_query(0, 0, 0.0, [0.5; 4]); // score 0.5
        raw.set_query(1, 1, 0.0, [0.5; 4]); // score 0.5
        let mut cfg = DecodeConfig::default();
        cfg.threshold_by_class.insert(0, 0.6);
        cfg.threshold_by_class.insert(1, 0.1);
        let out = decode_page(&raw, 10, 10, &cfg).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].label, 1);
        assert!((cfg.pre_threshold() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn malformed_tensors_are_rejected() {
        let mut raw = RawOutputs::blank(2, 2);
        raw.logits.pop();
        assert!(decode_page(&raw, 1, 1, &DecodeConfig::default()).is_err());
    }
}
