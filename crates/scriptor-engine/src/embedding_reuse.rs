//! Image-embedding reuse: probing which images of a request are already embedded, the
//! `cached_tokens` usage figure, hit-rate metrics and the cache byte budget.
//!
//! Images are keyed by the SHA-256 of their decoded pixels plus grid (see
//! [`scriptor_cache::embedding_key`]). At admission the scheduler resolves each image against the
//! cache; a hit skips the vision tower and the cached `[rows, hidden]` embeddings are
//! scattered into the prompt's image-placeholder positions.
//!
//! Only image embeddings are reused. The short text prefix that precedes the image run
//! (`[gMASK]<sop>`, the user header and the begin-of-image token, about six tokens) is not
//! cached: recomputing it costs less than the bookkeeping a prefix cache would add.

use crate::scheduler::{ModelExecutor, Scheduler};
use crate::sequence::ImageInput;
use scriptor_cache::EmbeddingCacheStats;

/// Per-request outcome of probing the embedding cache.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReusePlan {
    /// For each image in request order, whether its embedding is cached.
    pub hits: Vec<bool>,
    /// Prompt tokens whose embeddings come from the cache (the `cached_tokens` usage figure).
    pub cached_tokens: usize,
    /// Patches the vision tower still has to process for this request.
    pub vision_patches: usize,
}

impl ReusePlan {
    /// Number of images served from the cache.
    pub fn hit_count(&self) -> usize {
        self.hits.iter().filter(|&&h| h).count()
    }
}

/// Builds the plan for `images` given a presence test; `merge` is the patch merge side.
pub fn plan_reuse(
    images: &[ImageInput],
    merge: usize,
    mut cached: impl FnMut(&ImageInput) -> bool,
) -> ReusePlan {
    let mut plan = ReusePlan::default();
    for img in images {
        let hit = cached(img);
        plan.hits.push(hit);
        if hit {
            plan.cached_tokens += img.tokens(merge);
        } else {
            plan.vision_patches += img.patches();
        }
    }
    plan
}

/// The `usage.prompt_tokens_details` object of a chat-completion reply.
pub fn prompt_tokens_details(cached_tokens: usize) -> serde_json::Value {
    serde_json::json!({ "cached_tokens": cached_tokens })
}

/// Lifetime reuse figures for the metrics endpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ReuseMetrics {
    /// Images served from the cache.
    pub hits: u64,
    /// Images that needed the vision tower.
    pub misses: u64,
    /// Entries evicted to stay within the budget.
    pub evictions: u64,
    /// Embeddings larger than the whole budget, never retained.
    pub rejected: u64,
    /// Entries currently held.
    pub entries: usize,
    /// Bytes currently held.
    pub bytes: usize,
    /// Byte budget of the cache.
    pub budget_bytes: usize,
}

impl ReuseMetrics {
    /// Hits divided by lookups; zero before the first lookup.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Prometheus text lines for the reuse figures.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let mut line = |name: &str, kind: &str, value: String| {
            out.push_str(&format!("# TYPE {name} {kind}\n{name} {value}\n"));
        };
        line(
            "oe:image_cache_hits_total",
            "counter",
            self.hits.to_string(),
        );
        line(
            "oe:image_cache_misses_total",
            "counter",
            self.misses.to_string(),
        );
        line(
            "oe:image_cache_evictions_total",
            "counter",
            self.evictions.to_string(),
        );
        line(
            "oe:image_cache_hit_rate",
            "gauge",
            format!("{}", self.hit_rate()),
        );
        line("oe:image_cache_bytes", "gauge", self.bytes.to_string());
        line(
            "oe:image_cache_budget_bytes",
            "gauge",
            self.budget_bytes.to_string(),
        );
        out
    }
}

impl From<EmbeddingCacheStats> for ReuseMetrics {
    fn from(s: EmbeddingCacheStats) -> Self {
        Self {
            hits: s.hits,
            misses: s.misses,
            evictions: s.evictions,
            rejected: s.rejected,
            entries: s.entries,
            bytes: s.bytes,
            budget_bytes: s.budget_bytes,
        }
    }
}

/// Embedding-cache byte budget as `share` (clamped to `[0, 1]`) of the `available_bytes` the
/// runtime memory plan leaves after weights, KV pool and workspaces.
pub fn budget_from_share(available_bytes: u64, share: f64) -> usize {
    let share = if share.is_nan() {
        0.0
    } else {
        share.clamp(0.0, 1.0)
    };
    (available_bytes as f64 * share) as usize
}

impl<E: ModelExecutor> Scheduler<E> {
    /// Probes which of `images` are cached without counting lookups or reordering the cache.
    pub fn plan_reuse(&self, images: &[ImageInput]) -> ReusePlan {
        plan_reuse(images, self.config.spatial_merge, |img| {
            self.embedding_cache.contains(&img.key)
        })
    }

    /// Reuse counters and gauges of the image-embedding cache.
    pub fn reuse_metrics(&self) -> ReuseMetrics {
        self.embedding_cache.stats().into()
    }
}
