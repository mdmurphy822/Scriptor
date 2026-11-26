//! Host staging of the per-slot step inputs that are rewritten and uploaded on every device
//! step. The vectors are sized once for the slot capacity, so filling them allocates nothing.
//! The vectors are uploaded from pageable memory: the driver stages a pageable source before
//! the copy call returns, and the device step launches and collects each step in order, so the
//! next step's refill never overlaps a read of the previous contents. The cudarc wrapper itself
//! does not synchronise the stream for an ordinary slice source.

use scriptor_kernels::sampling::RowParams;

/// Per-slot input arrays of one step, indexed by device slot.
pub struct StateScratch {
    cached: Vec<i32>,
    limit: Vec<i32>,
    last: Vec<u32>,
    params: Vec<RowParams>,
}

impl StateScratch {
    /// Scratch with room for `slots` slots.
    pub fn new(slots: usize) -> Self {
        Self {
            cached: Vec::with_capacity(slots),
            limit: Vec::with_capacity(slots),
            last: Vec::with_capacity(slots),
            params: Vec::with_capacity(slots),
        }
    }

    /// Resets the first `bucket` slots to idle: no cached position, no limit, token zero and
    /// greedy parameters bound to the slot. Every slot is rewritten, so no value of an earlier
    /// step survives.
    pub fn reset(&mut self, bucket: usize) {
        self.cached.clear();
        self.cached.resize(bucket, -1);
        self.limit.clear();
        self.limit.resize(bucket, 0);
        self.last.clear();
        self.last.resize(bucket, 0);
        self.params.clear();
        self.params
            .extend((0..bucket as u32).map(RowParams::greedy));
    }

    /// Marks `slot` active with its cached position, KV limit in tokens, input token and
    /// sampling parameters, which are bound to the slot.
    pub fn set(&mut self, slot: usize, cached: i32, limit: i32, last: u32, mut params: RowParams) {
        self.cached[slot] = cached;
        self.limit[slot] = limit;
        self.last[slot] = last;
        params.slot = slot as u32;
        self.params[slot] = params;
    }

    pub fn cached(&self) -> &[i32] {
        &self.cached
    }

    pub fn limit(&self) -> &[i32] {
        &self.limit
    }

    pub fn last(&self) -> &[u32] {
        &self.last
    }

    pub fn params(&self) -> &[RowParams] {
        &self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampled(seed: u64) -> RowParams {
        RowParams {
            seed,
            temperature: 0.7,
            top_k: 40,
            step: 9,
            ..RowParams::greedy(0)
        }
    }

    fn assert_idle(s: &StateScratch, slot: usize) {
        assert_eq!(s.cached()[slot], -1);
        assert_eq!(s.limit()[slot], 0);
        assert_eq!(s.last()[slot], 0);
        let p = s.params()[slot];
        let g = RowParams::greedy(slot as u32);
        assert_eq!((p.seed, p.temperature, p.top_k, p.step), (0, 0.0, 0, 0));
        assert_eq!(p.slot, g.slot);
        assert_eq!(p.repetition, 1.0);
    }

    #[test]
    fn fresh_reset_is_idle_in_every_slot() {
        let mut s = StateScratch::new(8);
        s.reset(8);
        for slot in 0..8 {
            assert_idle(&s, slot);
        }
    }

    #[test]
    fn set_binds_params_to_the_slot() {
        let mut s = StateScratch::new(4);
        s.reset(4);
        s.set(2, 17, 64, 99, sampled(5));
        assert_eq!((s.cached()[2], s.limit()[2], s.last()[2]), (17, 64, 99));
        assert_eq!(s.params()[2].slot, 2);
        assert_eq!(s.params()[2].seed, 5);
        assert_eq!(s.params()[2].top_k, 40);
    }

    #[test]
    fn inactive_slots_are_reset_after_a_fuller_step() {
        let mut s = StateScratch::new(8);
        s.reset(8);
        for slot in 0..8 {
            s.set(slot, slot as i32, 16, 7, sampled(slot as u64 + 1));
        }
        s.reset(8);
        s.set(3, 4, 32, 11, sampled(42));
        for slot in (0..8).filter(|&i| i != 3) {
            assert_idle(&s, slot);
        }
        assert_eq!(s.params()[3].seed, 42);
    }

    #[test]
    fn a_reused_slot_carries_nothing_from_its_previous_owner() {
        let mut s = StateScratch::new(4);
        s.reset(4);
        s.set(1, 100, 128, 5, sampled(7));
        s.reset(4);
        s.set(1, 3, 16, 6, RowParams::greedy(0));
        assert_eq!((s.cached()[1], s.limit()[1], s.last()[1]), (3, 16, 6));
        let p = s.params()[1];
        assert_eq!((p.seed, p.temperature, p.top_k, p.step), (0, 0.0, 0, 0));
        assert_eq!(p.slot, 1);
    }

    #[test]
    fn a_smaller_bucket_shrinks_the_views_to_that_bucket() {
        let mut s = StateScratch::new(8);
        s.reset(8);
        s.reset(2);
        assert_eq!(s.cached().len(), 2);
        assert_eq!(s.limit().len(), 2);
        assert_eq!(s.last().len(), 2);
        assert_eq!(s.params().len(), 2);
    }

    #[test]
    fn storage_is_stable_across_steps() {
        let mut s = StateScratch::new(16);
        s.reset(16);
        let at = (
            s.cached().as_ptr(),
            s.limit().as_ptr(),
            s.last().as_ptr(),
            s.params().as_ptr(),
        );
        for step in 0..32 {
            s.reset(1 + step % 16);
            s.set(0, step as i32, 8, 1, sampled(step as u64));
        }
        assert_eq!(
            at,
            (
                s.cached().as_ptr(),
                s.limit().as_ptr(),
                s.last().as_ptr(),
                s.params().as_ptr()
            )
        );
    }

    #[test]
    fn staged_values_are_owned_copies() {
        let mut s = StateScratch::new(2);
        s.reset(2);
        let source = vec![sampled(3)];
        s.set(0, 5, 16, 9, source[0]);
        drop(source);
        assert_eq!(s.params()[0].seed, 3);
        assert_eq!(s.last()[0], 9);
    }
}
