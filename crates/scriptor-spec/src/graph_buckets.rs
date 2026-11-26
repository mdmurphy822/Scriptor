//! Batch buckets for captured decode graphs.

/// Batch sizes graphs are captured for: every power of two below `max_seqs`, then `max_seqs`
/// itself. A batch is padded with inactive rows up to its bucket.
pub fn graph_buckets(max_seqs: usize) -> Vec<usize> {
    let max_seqs = max_seqs.max(1);
    let mut buckets: Vec<usize> = std::iter::successors(Some(1usize), |b| b.checked_mul(2))
        .take_while(|&b| b < max_seqs)
        .collect();
    buckets.push(max_seqs);
    buckets
}

/// Smallest bucket holding `rows` sequences, or `None` when `rows` is zero or above every bucket.
pub fn bucket_for(buckets: &[usize], rows: usize) -> Option<usize> {
    if rows == 0 {
        return None;
    }
    buckets.iter().copied().find(|&b| b >= rows)
}

/// Row shape of one decode step over sequences held in persistent device slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepShape {
    /// Sequences the step serves.
    pub active: usize,
    /// Highest occupied slot plus one; one for a step with no sequences.
    pub span: usize,
    /// Bucket covering `span`: the graph the step replays.
    pub bucket: usize,
    /// Bucket covering `active` alone: the graph a gap-free slot layout would replay.
    pub compact: usize,
}

impl StepShape {
    /// Rows the step runs beyond the sequences it serves.
    pub fn padded(&self) -> usize {
        self.bucket - self.active
    }

    /// Rows the step runs beyond the bucket a gap-free slot layout would select.
    pub fn sparse(&self) -> usize {
        self.bucket - self.compact
    }
}

/// Shape of a step over sequences in `slots`, whose bucket covers the highest slot; `Err`
/// carries the span when it exceeds every bucket.
pub fn step_shape(
    buckets: &[usize],
    slots: impl IntoIterator<Item = usize>,
) -> Result<StepShape, usize> {
    let (mut active, mut top) = (0usize, None::<usize>);
    for slot in slots {
        active += 1;
        top = Some(top.map_or(slot, |t| t.max(slot)));
    }
    let span = top.map_or(1, |s| s + 1);
    let bucket = bucket_for(buckets, span).ok_or(span)?;
    let compact = bucket_for(buckets, active.max(1)).unwrap_or(bucket);
    Ok(StepShape {
        active,
        span,
        bucket,
        compact,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_are_powers_of_two_then_the_maximum() {
        assert_eq!(graph_buckets(32), vec![1, 2, 4, 8, 16, 32]);
        assert_eq!(graph_buckets(24), vec![1, 2, 4, 8, 16, 24]);
        assert_eq!(graph_buckets(1), vec![1]);
        assert_eq!(graph_buckets(0), vec![1]);
    }

    #[test]
    fn rows_pad_up_to_the_next_bucket() {
        let b = graph_buckets(24);
        assert_eq!(bucket_for(&b, 1), Some(1));
        assert_eq!(bucket_for(&b, 3), Some(4));
        assert_eq!(bucket_for(&b, 16), Some(16));
        assert_eq!(bucket_for(&b, 17), Some(24));
        assert_eq!(bucket_for(&b, 24), Some(24));
    }

    #[test]
    fn shapes_outside_the_captured_set_have_no_bucket() {
        let b = graph_buckets(8);
        assert_eq!(bucket_for(&b, 0), None);
        assert_eq!(bucket_for(&b, 9), None);
    }

    #[test]
    fn step_shape_follows_the_highest_slot() {
        let b = graph_buckets(32);
        let lone_top = step_shape(&b, [31]).unwrap();
        assert_eq!(
            lone_top,
            StepShape {
                active: 1,
                span: 32,
                bucket: 32,
                compact: 1
            }
        );
        assert_eq!(lone_top.padded(), 31);
        assert_eq!(lone_top.sparse(), 31);
        let dense = step_shape(&b, 0..5).unwrap();
        assert_eq!((dense.span, dense.bucket, dense.compact), (5, 8, 8));
        assert_eq!((dense.padded(), dense.sparse()), (3, 0));
        let gapped = step_shape(&b, [0, 2, 17]).unwrap();
        assert_eq!((gapped.active, gapped.span), (3, 18));
        assert_eq!((gapped.bucket, gapped.compact), (32, 4));
    }

    #[test]
    fn step_shape_matches_the_span_rule_for_every_slot_set_of_eight() {
        let b = graph_buckets(8);
        for mask in 0u32..256 {
            let slots: Vec<usize> = (0..8).filter(|&s| mask & (1 << s) != 0).collect();
            let shape = step_shape(&b, slots.iter().copied()).unwrap();
            let top = slots.iter().copied().max().map_or(1, |s| s + 1);
            assert_eq!(shape.span, top);
            assert_eq!(Some(shape.bucket), bucket_for(&b, top));
            assert_eq!(shape.active, slots.len());
            assert!(shape.compact <= shape.bucket);
            assert!(shape.compact >= shape.active);
        }
    }

    #[test]
    fn empty_step_takes_the_smallest_bucket_and_overflow_reports_the_span() {
        let b = graph_buckets(8);
        let empty = step_shape(&b, std::iter::empty()).unwrap();
        assert_eq!(
            (empty.active, empty.span, empty.bucket, empty.compact),
            (0, 1, 1, 1)
        );
        assert_eq!(step_shape(&b, [3, 8]), Err(9));
    }
}
