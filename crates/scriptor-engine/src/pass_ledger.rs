//! Prefill pass size accounting: tokens per executor pass, as a power-of-two histogram and exact
//! totals.

use std::collections::BTreeMap;

/// Pass size, in tokens, below which a prefill pass sits under the tensor-pipe ridge point.
pub const PREFILL_RIDGE_TOKENS: usize = 446;

/// Prefill pass sizes recorded by the scheduler.
#[derive(Debug, Default, Clone)]
pub struct PassLedger {
    sizes: BTreeMap<u64, u64>,
    tokens: u64,
    below_ridge_passes: u64,
    below_ridge_tokens: u64,
}

/// Snapshot of a [`PassLedger`] for the metrics endpoint.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PassStats {
    /// Passes by token count rounded up to a power of two: `(upper bound, passes)` ascending.
    pub sizes: Vec<(u64, u64)>,
    /// Tokens summed over all passes.
    pub tokens: u64,
    /// Passes of fewer than [`PREFILL_RIDGE_TOKENS`] tokens.
    pub below_ridge_passes: u64,
    /// Tokens summed over the passes counted in `below_ridge_passes`.
    pub below_ridge_tokens: u64,
}

impl PassLedger {
    /// Records one prefill pass of `tokens` tokens.
    pub fn record(&mut self, tokens: usize) {
        *self
            .sizes
            .entry(tokens.max(1).next_power_of_two() as u64)
            .or_insert(0) += 1;
        self.tokens += tokens as u64;
        if tokens < PREFILL_RIDGE_TOKENS {
            self.below_ridge_passes += 1;
            self.below_ridge_tokens += tokens as u64;
        }
    }

    /// The recorded totals.
    pub fn stats(&self) -> PassStats {
        PassStats {
            sizes: self.sizes.iter().map(|(&k, &v)| (k, v)).collect(),
            tokens: self.tokens,
            below_ridge_passes: self.below_ridge_passes,
            below_ridge_tokens: self.below_ridge_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_bucket_by_power_of_two_and_split_at_the_ridge() {
        let mut l = PassLedger::default();
        for t in [1, 445, 446, 1024, 1025] {
            l.record(t);
        }
        let s = l.stats();
        assert_eq!(s.sizes, vec![(1, 1), (512, 2), (1024, 1), (2048, 1)]);
        assert_eq!(s.tokens, 1 + 445 + 446 + 1024 + 1025);
        assert_eq!(s.below_ridge_passes, 2);
        assert_eq!(s.below_ridge_tokens, 446);
    }

    #[test]
    fn an_empty_ledger_reports_zeroes() {
        assert_eq!(PassLedger::default().stats(), PassStats::default());
    }
}
