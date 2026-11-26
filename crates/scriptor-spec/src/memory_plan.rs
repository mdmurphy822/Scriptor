//! Device-memory budget for weights, KV cache, embedding cache and workspaces.

use crate::error::SpecError;
use crate::image_proc::ImageProcessorConfig;
use crate::model_spec::ModelSpec;
use crate::runtime_params::{RuntimeParams, VisionBatchPatches};

/// Inputs to [`MemoryPlan::compute`] that come from the checkpoint and the device.
#[derive(Debug, Clone, Copy)]
pub struct MemoryPlanInputs {
    /// Bytes of all checkpoint weights (vision and MTP included).
    pub weight_bytes: u64,
    /// Total device memory in bytes.
    pub device_total_bytes: u64,
    /// Device memory currently free, when known; the budget never exceeds it.
    pub free_bytes: Option<u64>,
}

/// Itemised memory budget; every field is in bytes unless noted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPlan {
    pub weights: u64,
    pub kv_cache: u64,
    pub mtp_kv_cache: u64,
    pub embedding_cache: u64,
    pub vision_workspace: u64,
    pub logits_workspace: u64,
    /// Device-resident decode step: activation scratch, row tables, per-sequence block tables,
    /// sampling bitsets, draft and carry buffers, split-K partials, and the pair-order buffer
    /// at its largest size.
    pub step_workspace: u64,
    /// Captured decode graphs held in device memory.
    pub graph_memory: u64,
    pub total: u64,
    pub available: u64,
    /// KV blocks the budget remainder holds (target and draft layers share one block table).
    pub kv_blocks: u64,
    /// Blocks one full-context sequence occupies.
    pub blocks_per_seq: u64,
    /// Sequences that can hold a full context at once: `kv_blocks / blocks_per_seq`.
    pub full_context_seqs: u64,
    pub kv_block_tokens: usize,
    /// Longest request in tokens the plan was made for.
    pub max_context: usize,
    /// Slot count the slot-scaled buffers were sized for.
    pub slots: usize,
    pub vision_batch_patches: usize,
}

const BF16: u64 = 2;
const F32: u64 = 4;
const I32: u64 = 4;
/// Upper bound on streaming multiprocessors, sizing the split-K partials of paged attention.
const SM_BOUND: u64 = 256;
/// Device bytes of the hoisted pair-order buffer at its largest: 4096 pairs of one `i32`.
pub const PAIR_ORDER_RESERVE_BYTES: u64 = 4096 * I32;
/// Device bytes one captured kernel node is budgeted.
const GRAPH_NODE_BYTES: u64 = 4096;
/// Kernel launches per decoder layer in a captured pass.
const LAUNCHES_PER_LAYER: u64 = 20;

impl MemoryPlan {
    /// Computes the budget: every buffer other than the KV pool is sized for `max_concurrency`
    /// slots, and the KV pool takes the remainder of `device_total_bytes * gpu_share` (capped by
    /// the free memory when known). Refuses when the remainder holds fewer blocks than one
    /// full-context sequence.
    pub fn compute(
        spec: &ModelSpec,
        image: &ImageProcessorConfig,
        params: &RuntimeParams,
        inputs: MemoryPlanInputs,
    ) -> Result<Self, SpecError> {
        params.validate()?;
        let share = (inputs.device_total_bytes as f64 * params.gpu_share) as u64;
        let available = inputs.free_bytes.map_or(share, |free| share.min(free));

        let image = match params.max_pixels {
            Some(p) => image.with_max_pixels(p)?,
            None => image.clone(),
        };
        let vision = &spec.vision;
        let merge = vision.merge_factor() as u64;
        let max_image_patches = image.max_image_tokens() as u64 * merge;
        let batch_patches = match params.vision_batch_patches {
            VisionBatchPatches::Auto => max_image_patches,
            VisionBatchPatches::Fixed(n) => n as u64,
        }
        .max(merge);

        let block = params.kv_block_tokens as u64;
        let blocks_per_seq = (params.max_context as u64).div_ceil(block);
        let block_bytes = block * spec.kv_bytes_per_token();
        let mtp_block_bytes = if params.speculative_tokens > 0 {
            block * spec.mtp_kv_bytes_per_token()
        } else {
            0
        };
        let per_block = block_bytes + mtp_block_bytes;

        let hidden = vision.hidden_size as u64;
        let per_patch = vision.patch_dim() as u64 * BF16
            + 2 * hidden * BF16
            + 3 * hidden * BF16
            + 2 * vision.intermediate_size as u64 * BF16;
        let vision_workspace = batch_patches * per_patch
            + (batch_patches / merge)
                * (spec.text.hidden_size as u64 * 2 + vision.merger_intermediate as u64 * 2)
                * BF16;

        let embedding_cache = 2 * (batch_patches / merge) * spec.text.hidden_size as u64 * BF16;

        let rows = params.max_concurrency as u64 * (1 + params.speculative_tokens as u64);
        let logits_workspace = rows * spec.text.vocab_size as u64 * F32;

        let t = &spec.text;
        let seqs = params.max_concurrency as u64;
        let (hidden_t, inter, vocab) = (
            t.hidden_size as u64,
            t.intermediate_size as u64,
            t.vocab_size as u64,
        );
        let (q_dim, kv_dim) = (
            (t.num_heads * t.head_dim) as u64,
            (t.num_kv_heads * t.head_dim) as u64,
        );
        let per_row = BF16
            * (10 * hidden_t
                + 4 * q_dim
                + 2 * kv_dim
                + 3 * inter
                + 2 * hidden_t
                + vocab
                + hidden_t);
        let words = vocab.div_ceil(32);
        let q_rows = 1 + params.speculative_tokens as u64;
        let split_rows = (2 * SM_BOUND + seqs * t.num_kv_heads as u64)
            * 4
            * q_rows
            * (t.num_heads / t.num_kv_heads.max(1)) as u64;
        let step_workspace = rows * per_row
            + rows * hidden_t * BF16
            + seqs * hidden_t * BF16
            + split_rows * (128 + 2) * F32
            + PAIR_ORDER_RESERVE_BYTES
            + seqs * (blocks_per_seq * I32 + (2 * words + vocab) * 4 + 16 * 4 + 64)
            + rows * (5 * I32 + 3 * F32);
        let buckets = crate::graph_buckets::graph_buckets(params.max_concurrency);
        let passes = 1 + params.speculative_tokens as u64;
        let nodes = (t.num_layers as u64 + 1) * LAUNCHES_PER_LAYER + 16;
        let graph_memory = buckets.len() as u64 * passes * nodes * GRAPH_NODE_BYTES;

        let weights = inputs.weight_bytes;
        let fixed = weights
            + embedding_cache
            + vision_workspace
            + logits_workspace
            + step_workspace
            + graph_memory;
        let kv_blocks = available.saturating_sub(fixed) / per_block.max(1);
        if kv_blocks < blocks_per_seq {
            let one_seq = blocks_per_seq * per_block;
            let suggestion = if available <= fixed {
                format!(
                    "raise gpu_share; fixed buffers alone ({fixed} bytes at max_concurrency {}) exceed the budget",
                    params.max_concurrency
                )
            } else {
                let short = one_seq - (available - fixed);
                format!(
                    "the KV remainder of {} bytes holds {kv_blocks} blocks but one full-context sequence of {} tokens needs {blocks_per_seq}; short by {short} bytes: lower max_context or max_concurrency, or raise gpu_share",
                    available - fixed,
                    params.max_context
                )
            };
            return Err(SpecError::DoesNotFit {
                needed: fixed + one_seq,
                available,
                suggestion,
            });
        }
        let kv_cache = kv_blocks * block_bytes;
        let mtp_kv_cache = kv_blocks * mtp_block_bytes;
        let total = fixed + kv_cache + mtp_kv_cache;

        Ok(Self {
            weights,
            kv_cache,
            mtp_kv_cache,
            embedding_cache,
            vision_workspace,
            logits_workspace,
            step_workspace,
            graph_memory,
            total,
            available,
            kv_blocks,
            blocks_per_seq,
            full_context_seqs: kv_blocks / blocks_per_seq.max(1),
            kv_block_tokens: params.kv_block_tokens,
            max_context: params.max_context,
            slots: params.max_concurrency,
            vision_batch_patches: batch_patches as usize,
        })
    }
}

impl MemoryPlan {
    /// Bytes of buffers that scale with the slot count: logits, decode-step workspace and
    /// captured graphs.
    pub fn slot_buffer_bytes(&self) -> u64 {
        self.logits_workspace + self.step_workspace + self.graph_memory
    }

    /// Bytes of the KV pool, target and draft layers together.
    pub fn kv_pool_bytes(&self) -> u64 {
        self.kv_cache + self.mtp_kv_cache
    }

    /// Tokens the KV pool holds.
    pub fn kv_tokens(&self) -> u64 {
        self.kv_blocks * self.kv_block_tokens as u64
    }

    /// Requests of `tokens` tokens the KV pool holds at once, fractional.
    pub fn concurrency_at(&self, tokens: usize) -> f64 {
        self.kv_tokens() as f64 / tokens.max(1) as f64
    }

    /// The slot count for a share: the largest count whose plan fits and whose KV pool holds
    /// the prompt of the largest image the engine accepts for every slot. Requests share the
    /// pool by reservation, so the count is not bounded by full contexts; the pool always
    /// holds at least one full-context request, so any legal request runs alone.
    /// `params.max_concurrency` is ignored. Refuses with the plan's error when not even one
    /// slot fits.
    pub fn derive_concurrency(
        spec: &ModelSpec,
        image: &ImageProcessorConfig,
        params: &RuntimeParams,
        inputs: MemoryPlanInputs,
    ) -> Result<usize, SpecError> {
        let image = match params.max_pixels {
            Some(p) => image.with_max_pixels(p)?,
            None => image.clone(),
        };
        let slot_tokens = image.max_image_tokens().clamp(1, params.max_context.max(1)) as u64;
        let at = |seqs: usize| {
            let mut p = params.clone();
            p.max_concurrency = seqs;
            Self::compute(spec, &image, &p, inputs)
        };
        let ok = |seqs: usize| at(seqs).is_ok_and(|p| p.kv_tokens() >= seqs as u64 * slot_tokens);
        at(1)?;
        let mut lo = 1usize;
        while lo < usize::MAX / 2 && ok(lo * 2) {
            lo *= 2;
        }
        let mut hi = lo.saturating_mul(2);
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if ok(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Ok(lo)
    }

    /// vLLM-style statement of the KV capacity for requests of the plan's `max_context` tokens.
    pub fn capacity_summary(&self) -> String {
        format!(
            "KV cache size: {} tokens ({} blocks of {}); Maximum concurrency for {} tokens per request: {:.2}x",
            self.kv_tokens(),
            self.kv_blocks,
            self.kv_block_tokens,
            self.max_context,
            self.concurrency_at(self.max_context)
        )
    }
}
