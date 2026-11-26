//! Block geometry of the paged KV cache derived from the memory plan.

use crate::error::CacheError;
use scriptor_spec::memory_plan::MemoryPlan;

/// Number and size of KV blocks; one block covers `block_tokens` tokens of the main KV cache
/// and of the MTP KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayout {
    pub num_blocks: usize,
    pub block_tokens: usize,
    /// Main-model KV bytes per block.
    pub kv_bytes_per_block: u64,
    /// MTP-layer KV bytes per block (0 when speculative decoding is off).
    pub mtp_kv_bytes_per_block: u64,
}

impl KvLayout {
    /// Builds the layout from the planned KV and MTP KV byte budgets.
    pub fn from_plan(plan: &MemoryPlan) -> Result<Self, CacheError> {
        if plan.kv_blocks == 0 || plan.kv_block_tokens == 0 {
            return Err(CacheError::InvalidLayout(
                "memory plan has no KV blocks".into(),
            ));
        }
        Ok(Self {
            num_blocks: plan.kv_blocks as usize,
            block_tokens: plan.kv_block_tokens,
            kv_bytes_per_block: plan.kv_cache / plan.kv_blocks,
            mtp_kv_bytes_per_block: plan.mtp_kv_cache / plan.kv_blocks,
        })
    }

    /// Total bytes (main plus MTP) of one block.
    pub fn bytes_per_block(&self) -> u64 {
        self.kv_bytes_per_block + self.mtp_kv_bytes_per_block
    }

    /// Blocks needed to hold `tokens` tokens.
    pub fn blocks_for_tokens(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.block_tokens)
    }

    /// Byte offset of a block within the main KV arena.
    pub fn kv_offset(&self, block: u32) -> u64 {
        block as u64 * self.kv_bytes_per_block
    }

    /// Byte offset of a block within the MTP KV arena.
    pub fn mtp_kv_offset(&self, block: u32) -> u64 {
        block as u64 * self.mtp_kv_bytes_per_block
    }
}
