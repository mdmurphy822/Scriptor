//! Occupancy counters of device-resident decode steps: the rows each step's graph bucket ran
//! against the sequences it served and against the bucket a gap-free slot layout would select.

use scriptor_spec::StepShape;

/// Distinct graph buckets the counters keep separately; steps of further buckets count only in
/// the totals.
pub const OCCUPANCY_BUCKETS: usize = 16;

/// Entries of a step-count histogram: counts 0 through 32, where the last entry also holds every
/// larger count.
pub const STEP_HISTOGRAM_LEN: usize = 33;

/// Steps counted by an active-row or span value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepHistogram(pub [u64; STEP_HISTOGRAM_LEN]);

impl Default for StepHistogram {
    fn default() -> Self {
        Self([0; STEP_HISTOGRAM_LEN])
    }
}

impl StepHistogram {
    /// Counts one step of `value` rows.
    pub fn record(&mut self, value: usize) {
        self.0[value.min(STEP_HISTOGRAM_LEN - 1)] += 1;
    }

    /// Steps counted, summed over every entry.
    pub fn total(&self) -> u64 {
        self.0.iter().sum()
    }
}

/// Counters of the steps that ran one graph bucket.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BucketOccupancy {
    /// Bucket size in sequences; zero marks an unused entry.
    pub rows: u64,
    /// Steps that ran this bucket.
    pub steps: u64,
    /// Steps whose active count alone selects this bucket.
    pub compact_steps: u64,
    /// Steps of this bucket with a measured device time.
    pub timed_steps: u64,
    /// Summed device nanoseconds of the timed steps.
    pub timed_nanos: u64,
}

/// Lifetime occupancy counters of the device-resident decode step.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlotOccupancy {
    /// Steps recorded.
    pub steps: u64,
    /// Sequences served, summed over steps.
    pub active_rows: u64,
    /// Highest occupied slot plus one, summed over steps.
    pub span_rows: u64,
    /// Bucket rows run, summed over steps.
    pub bucket_rows: u64,
    /// Rows of the bucket the active count alone selects, summed over steps.
    pub compact_rows: u64,
    /// Steps whose bucket exceeds their active count.
    pub padded_steps: u64,
    /// Steps whose bucket exceeds the bucket of their active count.
    pub sparse_steps: u64,
    /// Steps by active sequence count.
    pub active_steps: StepHistogram,
    /// Steps by span (highest occupied slot plus one).
    pub span_steps: StepHistogram,
    /// Per-bucket counters in first-seen order.
    pub buckets: [BucketOccupancy; OCCUPANCY_BUCKETS],
}

impl SlotOccupancy {
    /// Counts one step of `shape`.
    pub fn record(&mut self, shape: &StepShape) {
        self.steps += 1;
        self.active_rows += shape.active as u64;
        self.span_rows += shape.span as u64;
        self.bucket_rows += shape.bucket as u64;
        self.compact_rows += shape.compact as u64;
        self.padded_steps += u64::from(shape.padded() > 0);
        self.sparse_steps += u64::from(shape.sparse() > 0);
        self.active_steps.record(shape.active);
        self.span_steps.record(shape.span);
        if let Some(entry) = self.entry(shape.bucket) {
            entry.steps += 1;
        }
        if let Some(entry) = self.entry(shape.compact) {
            entry.compact_steps += 1;
        }
    }

    /// Adds the device time of one step that ran `bucket`.
    pub fn record_time(&mut self, bucket: usize, nanos: u64) {
        if let Some(entry) = self.entry(bucket) {
            entry.timed_steps += 1;
            entry.timed_nanos += nanos;
        }
    }

    /// Bucket rows run beyond the sequences served, summed over steps.
    pub fn padded_rows(&self) -> u64 {
        self.bucket_rows - self.active_rows
    }

    /// Bucket rows run beyond the bucket of each step's active count, summed over steps.
    pub fn sparse_rows(&self) -> u64 {
        self.bucket_rows - self.compact_rows
    }

    /// The used per-bucket entries, ascending by bucket size.
    pub fn bucket_entries(&self) -> Vec<BucketOccupancy> {
        let mut used: Vec<BucketOccupancy> = self
            .buckets
            .iter()
            .copied()
            .filter(|b| b.rows > 0)
            .collect();
        used.sort_by_key(|b| b.rows);
        used
    }

    /// The entry of `bucket`, claiming an unused one on first sight; `None` once every entry
    /// holds another bucket.
    fn entry(&mut self, bucket: usize) -> Option<&mut BucketOccupancy> {
        let rows = bucket as u64;
        let at = self
            .buckets
            .iter()
            .position(|b| b.rows == rows || b.rows == 0)?;
        let entry = &mut self.buckets[at];
        entry.rows = rows;
        Some(entry)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use scriptor_spec::{graph_buckets, step_shape};

    use super::*;

    /// Closed-loop decode of `lengths` (decode steps per request) at `concurrency`, with slots
    /// taken lowest-free-first at admission and returned when a request finishes, as the
    /// device step's slot table does. A request becomes decodable `admit_delay` steps after
    /// its predecessor's slot frees, standing in for its vision and prefill.
    fn simulate(
        max_seqs: usize,
        concurrency: usize,
        lengths: &[usize],
        admit_delay: usize,
    ) -> SlotOccupancy {
        let buckets = graph_buckets(max_seqs);
        let mut free: BTreeSet<usize> = (0..max_seqs).collect();
        let mut active: BTreeMap<usize, usize> = BTreeMap::new();
        let first = concurrency.min(lengths.len());
        let mut waiting: Vec<(usize, usize)> = (0..first).map(|req| (req, 0)).collect();
        let mut next = first;
        let mut occ = SlotOccupancy::default();
        while !active.is_empty() || !waiting.is_empty() {
            let mut still = Vec::new();
            for (req, delay) in waiting.drain(..) {
                if delay == 0 {
                    let slot = free.pop_first().expect("a free slot");
                    active.insert(slot, lengths[req]);
                } else {
                    still.push((req, delay - 1));
                }
            }
            waiting = still;
            if active.is_empty() {
                continue;
            }
            let shape = step_shape(&buckets, active.keys().copied()).expect("covered");
            occ.record(&shape);
            let done: Vec<usize> = active
                .iter_mut()
                .filter_map(|(&slot, left)| {
                    *left -= 1;
                    (*left == 0).then_some(slot)
                })
                .collect();
            for slot in done {
                active.remove(&slot);
                free.insert(slot);
                if next < lengths.len() {
                    waiting.push((next, admit_delay));
                    next += 1;
                }
            }
        }
        occ
    }

    #[test]
    fn counters_add_up_per_step() {
        let buckets = graph_buckets(32);
        let mut occ = SlotOccupancy::default();
        occ.record(&step_shape(&buckets, [31]).unwrap());
        occ.record(&step_shape(&buckets, 0..32).unwrap());
        occ.record(&step_shape(&buckets, [0, 1, 2]).unwrap());
        assert_eq!(occ.steps, 3);
        assert_eq!(occ.active_rows, 1 + 32 + 3);
        assert_eq!(occ.span_rows, 32 + 32 + 3);
        assert_eq!(occ.bucket_rows, 32 + 32 + 4);
        assert_eq!(occ.compact_rows, 1 + 32 + 4);
        assert_eq!(occ.padded_rows(), 31 + 1);
        assert_eq!(occ.sparse_rows(), 31);
        assert_eq!((occ.padded_steps, occ.sparse_steps), (2, 1));
        let entries = occ.bucket_entries();
        let rows: Vec<u64> = entries.iter().map(|b| b.rows).collect();
        assert_eq!(rows, vec![1, 4, 32]);
        let by = |r: u64| entries.iter().find(|b| b.rows == r).copied().unwrap();
        assert_eq!((by(32).steps, by(32).compact_steps), (2, 1));
        assert_eq!((by(1).steps, by(1).compact_steps), (0, 1));
        assert_eq!((by(4).steps, by(4).compact_steps), (1, 1));
    }

    #[test]
    fn active_and_span_histograms_count_each_step_once() {
        let buckets = graph_buckets(32);
        let mut occ = SlotOccupancy::default();
        occ.record(&step_shape(&buckets, [31]).unwrap());
        occ.record(&step_shape(&buckets, 0..32).unwrap());
        occ.record(&step_shape(&buckets, [0, 1, 2]).unwrap());
        occ.record(&step_shape(&buckets, [0, 5]).unwrap());
        assert_eq!(occ.active_steps.total(), occ.steps);
        assert_eq!(occ.span_steps.total(), occ.steps);
        assert_eq!(occ.active_steps.0[1], 1);
        assert_eq!(occ.active_steps.0[2], 1);
        assert_eq!(occ.active_steps.0[3], 1);
        assert_eq!(occ.active_steps.0[32], 1);
        assert_eq!(occ.span_steps.0[32], 2);
        assert_eq!(occ.span_steps.0[3], 1);
        assert_eq!(occ.span_steps.0[6], 1);
        let by_value =
            |h: &StepHistogram| -> u64 { h.0.iter().enumerate().map(|(v, n)| v as u64 * n).sum() };
        assert_eq!(by_value(&occ.active_steps), occ.active_rows);
        assert_eq!(by_value(&occ.span_steps), occ.span_rows);
    }

    #[test]
    fn histogram_values_past_the_last_entry_share_it() {
        let mut h = StepHistogram::default();
        h.record(32);
        h.record(33);
        h.record(128);
        assert_eq!(h.0[STEP_HISTOGRAM_LEN - 1], 3);
        assert_eq!(h.total(), 3);
    }

    #[test]
    fn device_time_accumulates_per_bucket_and_entries_saturate() {
        let mut occ = SlotOccupancy::default();
        occ.record_time(8, 1_000);
        occ.record_time(8, 3_000);
        let e = occ.bucket_entries();
        assert_eq!(
            (e[0].rows, e[0].timed_steps, e[0].timed_nanos),
            (8, 2, 4_000)
        );
        for b in 100..100 + OCCUPANCY_BUCKETS {
            occ.record_time(b, 1);
        }
        assert_eq!(occ.bucket_entries().len(), OCCUPANCY_BUCKETS);
        assert_eq!(occ.bucket_entries()[0].timed_nanos, 4_000);
    }

    #[test]
    fn equal_lengths_with_instant_refill_run_no_sparse_rows() {
        let occ = simulate(32, 32, &[40; 128], 0);
        assert_eq!(occ.steps, 4 * 40);
        assert_eq!(occ.sparse_rows(), 0);
        assert_eq!(occ.padded_rows(), 0);
    }

    #[test]
    fn a_lone_survivor_in_the_top_slot_runs_the_full_bucket() {
        let mut lengths = vec![1; 32];
        lengths[31] = 10;
        let occ = simulate(32, 32, &lengths, 0);
        assert_eq!(occ.steps, 10);
        assert_eq!(occ.bucket_rows, 10 * 32);
        assert_eq!(occ.compact_rows, 32 + 9);
        assert_eq!(occ.sparse_rows(), 9 * 31);
        assert_eq!(occ.sparse_steps, 9);
    }

    #[test]
    fn mixed_length_drain_wastes_rows_only_where_slots_are_sparse() {
        let lengths: Vec<usize> = (0..256).map(|i| 5 + (i * 37 + 11) % 60).collect();
        let occ = simulate(32, 32, &lengths, 3);
        assert!(
            occ.sparse_rows() > 0,
            "the drained tail leaves gaps below a live slot"
        );
        assert!(occ.sparse_rows() <= occ.padded_rows());
        assert!(occ.compact_rows >= occ.active_rows);
        assert!(occ.span_rows <= occ.bucket_rows);
        let total: u64 = occ.bucket_entries().iter().map(|b| b.steps).sum();
        let compact: u64 = occ.bucket_entries().iter().map(|b| b.compact_steps).sum();
        assert_eq!((total, compact), (occ.steps, occ.steps));
    }
}
