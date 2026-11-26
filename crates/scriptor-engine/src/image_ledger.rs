//! Observation-only ledger of the images submitted to the scheduler: how many patches each
//! image has and how often an exact content key recurs.
//!
//! The ledger never influences admission, batching or output. Keys are remembered for the life
//! of the scheduler up to a fixed capacity; a key seen after the capacity is reached is
//! counted as untracked instead of being compared.

use scriptor_cache::EmbeddingKey;
use std::collections::{BTreeMap, HashSet};

/// Distinct keys the ledger remembers.
pub const IMAGE_KEY_CAPACITY: usize = 262_144;

/// Snapshot of the ledger's counters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageLedgerStats {
    /// Images submitted.
    pub images: u64,
    /// Patches of the submitted images.
    pub patches: u64,
    /// Submitted images by patch count rounded up to a power of two: `(upper bound, images)`
    /// ascending.
    pub patch_sizes: Vec<(u64, u64)>,
    /// Images whose key had not been seen before.
    pub unique: u64,
    /// Images whose key had been seen before.
    pub repeat: u64,
    /// Patches of the images counted in `repeat`.
    pub repeat_patches: u64,
    /// Images whose key was not compared because the ledger was full.
    pub untracked: u64,
}

/// Per-image patch counts and duplicate-key counters.
#[derive(Debug)]
pub struct ImageLedger {
    keys: HashSet<EmbeddingKey>,
    capacity: usize,
    sizes: BTreeMap<u64, u64>,
    stats: ImageLedgerStats,
}

impl Default for ImageLedger {
    fn default() -> Self {
        Self::with_capacity(IMAGE_KEY_CAPACITY)
    }
}

impl ImageLedger {
    /// A ledger that remembers at most `capacity` distinct keys.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            keys: HashSet::new(),
            capacity,
            sizes: BTreeMap::new(),
            stats: ImageLedgerStats::default(),
        }
    }

    /// Records one submitted image of `patches` patches with content key `key`.
    pub fn observe(&mut self, key: EmbeddingKey, patches: usize) {
        self.stats.images += 1;
        self.stats.patches += patches as u64;
        *self
            .sizes
            .entry(patches.max(1).next_power_of_two() as u64)
            .or_insert(0) += 1;
        if self.keys.contains(&key) {
            self.stats.repeat += 1;
            self.stats.repeat_patches += patches as u64;
        } else if self.keys.len() < self.capacity {
            self.keys.insert(key);
            self.stats.unique += 1;
        } else {
            self.stats.untracked += 1;
        }
    }

    /// The counters so far.
    pub fn snapshot(&self) -> ImageLedgerStats {
        ImageLedgerStats {
            patch_sizes: self.sizes.iter().map(|(&k, &v)| (k, v)).collect(),
            ..self.stats.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u8) -> EmbeddingKey {
        EmbeddingKey([n; 32])
    }

    #[test]
    fn a_repeated_key_counts_once_as_unique_and_after_that_as_repeat() {
        let mut l = ImageLedger::default();
        for (k, p) in [(1, 16), (2, 64), (1, 16), (3, 16), (1, 16)] {
            l.observe(key(k), p);
        }
        let s = l.snapshot();
        assert_eq!((s.images, s.unique, s.repeat), (5, 3, 2));
        assert_eq!(s.repeat_patches, 32);
        assert_eq!(s.patches, 16 + 64 + 16 + 16 + 16);
        assert_eq!(s.untracked, 0);
    }

    #[test]
    fn patch_sizes_round_up_to_powers_of_two() {
        let mut l = ImageLedger::default();
        for p in [4, 16, 17, 64] {
            l.observe(key(p as u8), p);
        }
        assert_eq!(
            l.snapshot().patch_sizes,
            vec![(4, 1), (16, 1), (32, 1), (64, 1)]
        );
    }

    #[test]
    fn keys_beyond_the_capacity_are_untracked_and_known_keys_still_match() {
        let mut l = ImageLedger::with_capacity(2);
        for k in [1, 2, 3, 1, 3] {
            l.observe(key(k), 8);
        }
        let s = l.snapshot();
        assert_eq!((s.unique, s.repeat, s.untracked), (2, 1, 2));
        assert_eq!(s.images, s.unique + s.repeat + s.untracked);
    }
}
