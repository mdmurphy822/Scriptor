//! Fixed-size KV block pool with per-sequence block tables.

use crate::error::CacheError;
use crate::kv_layout::KvLayout;
use std::collections::{BTreeMap, HashMap};

/// Index of a block within the KV arenas.
pub type BlockId = u32;
/// Caller-chosen sequence identifier.
pub type SeqId = u64;

#[derive(Debug, Clone)]
pub(crate) struct SeqState {
    pub(crate) table: Vec<BlockId>,
    pub(crate) tokens: usize,
    pub(crate) admitted: u64,
}

/// Snapshot of pool occupancy and lifetime counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub used_blocks: usize,
    pub sequences: usize,
    pub allocations: u64,
    pub releases: u64,
    pub preemptions: u64,
}

impl PoolStats {
    /// Fraction of blocks in use, in `[0, 1]`.
    pub fn usage(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            self.used_blocks as f64 / self.total_blocks as f64
        }
    }
}

/// Ordered free runs plus block tables; every block is either free or in exactly one table.
/// Sequences receive physically adjacent blocks wherever the free runs allow.
#[derive(Debug)]
pub struct BlockPool {
    layout: KvLayout,
    /// Free runs keyed by first block, value is the run length; runs never touch each other.
    free: BTreeMap<BlockId, u32>,
    free_count: usize,
    pub(crate) seqs: HashMap<SeqId, SeqState>,
    next_admission: u64,
    allocations: u64,
    releases: u64,
    pub(crate) preemptions: u64,
}

impl BlockPool {
    pub fn new(layout: KvLayout) -> Self {
        let mut free = BTreeMap::new();
        if layout.num_blocks > 0 {
            free.insert(0, layout.num_blocks as u32);
        }
        Self {
            free_count: layout.num_blocks,
            layout,
            free,
            seqs: HashMap::new(),
            next_admission: 0,
            allocations: 0,
            releases: 0,
            preemptions: 0,
        }
    }

    pub fn layout(&self) -> &KvLayout {
        &self.layout
    }

    pub fn free_blocks(&self) -> usize {
        self.free_count
    }

    /// Number of maximal free runs of adjacent blocks.
    pub fn free_runs(&self) -> usize {
        self.free.len()
    }

    /// Length of the longest free run.
    pub fn largest_free_run(&self) -> usize {
        self.free.values().copied().max().unwrap_or(0) as usize
    }

    /// Removes `take` blocks from the front of the free run starting at `start`.
    fn take_front(&mut self, start: BlockId, take: u32, out: &mut Vec<BlockId>) {
        let len = self.free.remove(&start).expect("run exists");
        debug_assert!(take <= len);
        out.extend(start..start + take);
        if take < len {
            self.free.insert(start + take, len - take);
        }
        self.free_count -= take as usize;
    }

    /// Start of the smallest free run holding at least `n` blocks (lowest start on ties).
    fn best_fit(&self, n: u32) -> Option<BlockId> {
        self.free
            .iter()
            .filter(|(_, len)| **len >= n)
            .min_by_key(|(start, len)| (**len, **start))
            .map(|(start, _)| *start)
    }

    /// Start of the longest free run (lowest start on ties).
    fn longest_run(&self) -> Option<BlockId> {
        self.free
            .iter()
            .max_by_key(|(start, len)| (**len, std::cmp::Reverse(**start)))
            .map(|(start, _)| *start)
    }

    /// Returns one block to the free runs, merging with adjacent runs.
    fn free_block(&mut self, block: BlockId) {
        let mut start = block;
        let mut len = 1u32;
        if let Some((&prev, &plen)) = self.free.range(..block).next_back()
            && prev + plen == block
        {
            self.free.remove(&prev);
            start = prev;
            len += plen;
        }
        if let Some(nlen) = self.free.remove(&(block + 1)) {
            len += nlen;
        }
        self.free.insert(start, len);
        self.free_count += 1;
    }

    /// Registers a sequence with an empty block table; later admissions are younger.
    pub fn admit(&mut self, seq: SeqId) -> Result<(), CacheError> {
        let key = self.next_admission;
        self.admit_with_key(seq, key)
    }

    /// Registers a sequence with an empty block table and an explicit ordering key. Lower keys
    /// are older: age listings and eviction order follow the key, so a sequence re-registered
    /// with its earlier key keeps its original standing. Later keyless admissions are younger
    /// than every key given so far.
    pub fn admit_with_key(&mut self, seq: SeqId, key: u64) -> Result<(), CacheError> {
        if self.seqs.contains_key(&seq) {
            return Err(CacheError::DuplicateSequence(seq));
        }
        let admitted = key;
        self.next_admission = self.next_admission.max(key.saturating_add(1));
        self.seqs.insert(
            seq,
            SeqState {
                table: Vec::new(),
                tokens: 0,
                admitted,
            },
        );
        Ok(())
    }

    pub fn contains(&self, seq: SeqId) -> bool {
        self.seqs.contains_key(&seq)
    }

    /// Block table of a sequence in token order.
    pub fn block_table(&self, seq: SeqId) -> Result<&[BlockId], CacheError> {
        self.seqs
            .get(&seq)
            .map(|s| s.table.as_slice())
            .ok_or(CacheError::UnknownSequence(seq))
    }

    /// Tokens the sequence has reserved capacity for.
    pub fn tokens(&self, seq: SeqId) -> Result<usize, CacheError> {
        self.seqs
            .get(&seq)
            .map(|s| s.tokens)
            .ok_or(CacheError::UnknownSequence(seq))
    }

    /// Additional blocks required for the sequence to hold `tokens` tokens.
    pub fn blocks_needed(&self, seq: SeqId, tokens: usize) -> Result<usize, CacheError> {
        let state = self
            .seqs
            .get(&seq)
            .ok_or(CacheError::UnknownSequence(seq))?;
        Ok(self
            .layout
            .blocks_for_tokens(tokens)
            .saturating_sub(state.table.len()))
    }

    /// Grows the table to cover `tokens` tokens without preempting; all-or-nothing.
    /// Returns the newly attached blocks.
    pub fn grow_to(&mut self, seq: SeqId, tokens: usize) -> Result<Vec<BlockId>, CacheError> {
        let needed = self.blocks_needed(seq, tokens)?;
        if needed > self.free_count {
            return Err(CacheError::OutOfBlocks {
                needed,
                free: self.free_count,
            });
        }
        let mut fresh = Vec::with_capacity(needed);
        let last = self.seqs[&seq].table.last().copied();
        if let Some(next) = last.and_then(|b| b.checked_add(1))
            && let Some(&len) = self.free.get(&next)
        {
            self.take_front(next, len.min(needed as u32), &mut fresh);
        }
        while fresh.len() < needed {
            let rest = (needed - fresh.len()) as u32;
            let start = self
                .best_fit(rest)
                .or_else(|| self.longest_run())
                .expect("free count covers the request");
            let len = self.free[&start];
            self.take_front(start, len.min(rest), &mut fresh);
        }
        self.allocations += needed as u64;
        let state = self.seqs.get_mut(&seq).expect("checked above");
        state.table.extend_from_slice(&fresh);
        state.tokens = state.tokens.max(tokens);
        Ok(fresh)
    }

    /// Removes the sequence and returns its blocks to the free list; returns the count freed.
    pub fn release(&mut self, seq: SeqId) -> Result<usize, CacheError> {
        let state = self
            .seqs
            .remove(&seq)
            .ok_or(CacheError::UnknownSequence(seq))?;
        let n = state.table.len();
        for block in state.table {
            self.free_block(block);
        }
        self.releases += n as u64;
        Ok(n)
    }

    /// Registered sequence ids ordered oldest admission first.
    pub fn sequences_by_age(&self) -> Vec<SeqId> {
        let mut v: Vec<_> = self.seqs.iter().map(|(id, s)| (s.admitted, *id)).collect();
        v.sort_unstable();
        v.into_iter().map(|(_, id)| id).collect()
    }

    /// Writes [`Self::sequences_by_age`] into `out`, replacing its contents; `keyed` is
    /// working storage. Both keep their capacity between calls.
    pub fn sequences_by_age_into(&self, keyed: &mut Vec<(u64, SeqId)>, out: &mut Vec<SeqId>) {
        keyed.clear();
        keyed.extend(self.seqs.iter().map(|(id, s)| (s.admitted, *id)));
        keyed.sort_unstable();
        out.clear();
        out.extend(keyed.iter().map(|&(_, id)| id));
    }

    /// Youngest (most recently admitted) registered sequence.
    pub fn youngest(&self) -> Option<SeqId> {
        self.seqs
            .iter()
            .max_by_key(|(_, s)| s.admitted)
            .map(|(id, _)| *id)
    }

    pub fn stats(&self) -> PoolStats {
        let total = self.layout.num_blocks;
        PoolStats {
            total_blocks: total,
            free_blocks: self.free_count,
            used_blocks: total - self.free_count,
            sequences: self.seqs.len(),
            allocations: self.allocations,
            releases: self.releases,
            preemptions: self.preemptions,
        }
    }
}
