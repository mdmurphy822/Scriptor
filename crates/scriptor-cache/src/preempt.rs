//! Youngest-first preemption with recompute when the pool runs out of blocks.

use crate::block_pool::{BlockId, BlockPool, SeqId};
use crate::error::CacheError;

/// Blocks newly attached to the requesting sequence and the sequences evicted to make room.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reservation {
    pub new_blocks: Vec<BlockId>,
    /// Evicted sequences in eviction order; each must be re-admitted and recomputed.
    pub preempted: Vec<SeqId>,
}

/// Result of [`BlockPool::reserve`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReserveOutcome {
    /// The sequence now covers the requested tokens.
    Granted(Reservation),
    /// The requester was itself the youngest sequence and was evicted; it must be recomputed.
    SelfPreempted { preempted: Vec<SeqId> },
}

impl BlockPool {
    /// Grows `seq` to cover `tokens` tokens. When free blocks run out, the youngest registered
    /// sequence is evicted (its blocks freed, its registration removed) until the request fits.
    /// If the requester is the youngest, it is evicted itself. Older sequences are never evicted
    /// while a younger one exists.
    pub fn reserve(&mut self, seq: SeqId, tokens: usize) -> Result<ReserveOutcome, CacheError> {
        let needed = self.blocks_needed(seq, tokens)?;
        let mut preempted = Vec::new();
        while self.free_blocks() < needed {
            let victim = self.youngest().expect("requester is registered");
            self.release(victim)?;
            self.preemptions += 1;
            preempted.push(victim);
            if victim == seq {
                return Ok(ReserveOutcome::SelfPreempted { preempted });
            }
        }
        let new_blocks = self.grow_to(seq, tokens)?;
        Ok(ReserveOutcome::Granted(Reservation {
            new_blocks,
            preempted,
        }))
    }
}
