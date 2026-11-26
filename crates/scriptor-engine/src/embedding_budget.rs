//! Device budget for image embeddings.
//!
//! The budget `E` is the embedding-cache byte budget of the runtime memory plan; zero leaves
//! the budget unbounded. Every allocation the vision tower makes is charged before it launches,
//! so the bytes held by sequences, by the vision batch in flight and by the LRU cache together
//! never exceed `E`.
//!
//! Charges are derived from live state on every call, so a request that finishes, is
//! cancelled, fails, disconnects or is shed stops being charged without a release call, and an
//! embedding evicted from the cache stops being charged once no sequence holds it.
//!
//! Progress: requests reserve in submission order and stop at the first that does not fit, so
//! every reservation holder is older than every non-holder. Holders never wait on the budget,
//! finish or fail, and release their charge. A request is only accepted when its own
//! embeddings fit `E`, and cache-only entries are evicted on demand, so once no holder remains
//! the oldest non-holder always reserves.

use crate::scheduler::{ModelExecutor, Scheduler, SchedulerConfig};
use crate::sequence::{Phase, RequestId};
use std::collections::HashSet;
use std::sync::Arc;

/// The embedding byte budget `E`, or `None` when it is unbounded.
pub fn embedding_bound(config: &SchedulerConfig) -> Option<usize> {
    (config.embedding_cache_bytes > 0).then_some(config.embedding_cache_bytes)
}

/// Bytes the executor allocates for an image of `patches` patches: one `hidden`-wide row per
/// merge group, at least one element.
pub fn embedding_bytes(config: &SchedulerConfig, patches: usize) -> usize {
    let group = (config.spatial_merge * config.spatial_merge).max(1);
    let elem = config.embedding_elem_bytes.max(1);
    ((patches / group) * config.embedding_hidden).max(1) * elem
}

impl<E: ModelExecutor> Scheduler<E> {
    /// Whether the sequence's pending images may be embedded: always when the budget is
    /// unbounded, otherwise once its reservation is held.
    pub(crate) fn embedding_admitted(&self, id: RequestId) -> bool {
        embedding_bound(&self.config).is_none() || self.embedding_reserved.contains(&id)
    }

    /// Distinct device bytes held by sequences, in-flight vision jobs and the images that
    /// reserved sequences are still to embed; also the identities of the held buffers.
    fn pinned_embeddings(&self) -> (usize, HashSet<usize>) {
        let mut seen: HashSet<usize> = HashSet::new();
        let mut bytes = 0usize;
        let flight: &[crate::scheduler::VisionJob] =
            self.vision_flight.as_ref().map_or(&[], |f| f.jobs());
        for seq in self.waiting.iter().chain(self.active.values()) {
            let reserved = self.embedding_reserved.contains(&seq.id);
            for (i, embed) in seq.embeds.iter().enumerate() {
                match embed {
                    Some(buf) => {
                        if seen.insert(Arc::as_ptr(buf) as usize) {
                            bytes += scriptor_cache::DeviceBuffer::byte_len(&**buf);
                        }
                    }
                    None => {
                        let launched = flight.iter().any(|j| j.seq == seq.id && j.image == i);
                        if reserved && !launched {
                            bytes += embedding_bytes(&self.config, seq.images[i].patches());
                        }
                    }
                }
            }
        }
        bytes += flight
            .iter()
            .map(|j| embedding_bytes(&self.config, j.patches))
            .sum::<usize>();
        (bytes, seen)
    }

    /// Bytes charged to live work: held embeddings counted once however many sequences and
    /// cache entries share them, in-flight vision output, and reserved images yet to embed.
    pub fn pinned_embedding_bytes(&self) -> usize {
        self.pinned_embeddings().0
    }

    /// Device bytes of image embeddings: the charged bytes plus cache entries no live work holds.
    pub fn retained_embedding_bytes(&self) -> usize {
        self.embedding_usage().1
    }

    /// The pinned and the retained device bytes of image embeddings, from one pass over live work.
    pub(crate) fn embedding_usage(&self) -> (usize, usize) {
        let (pinned, seen) = self.pinned_embeddings();
        let cache_only: usize = self
            .embedding_cache
            .iter_lru()
            .filter(|(_, b)| !seen.contains(&(Arc::as_ptr(&b.0) as usize)))
            .map(|(_, b)| scriptor_cache::DeviceBuffer::byte_len(&*b.0))
            .sum();
        (pinned, pinned + cache_only)
    }

    /// Reserves, oldest request first, the embeddings of requests whose images are still to be
    /// embedded, evicting cache-only entries to make room, and stops at the first that does not
    /// fit. Reservations of requests that left the scheduler are dropped.
    pub(crate) fn reserve_embeddings(&mut self) {
        let Some(limit) = embedding_bound(&self.config) else {
            return;
        };
        let live: HashSet<RequestId> = self
            .waiting
            .iter()
            .map(|s| s.id)
            .chain(self.active.keys().copied())
            .collect();
        self.embedding_reserved.retain(|id| live.contains(id));
        let mut candidates: Vec<(u64, RequestId, usize)> = self
            .waiting
            .iter()
            .chain(self.active.values())
            .filter(|s| !self.embedding_reserved.contains(&s.id) && s.phase != Phase::Running)
            .filter_map(|s| {
                let need: usize = s
                    .embeds
                    .iter()
                    .zip(&s.images)
                    .filter(|(e, _)| e.is_none())
                    .map(|(_, img)| embedding_bytes(&self.config, img.patches()))
                    .sum();
                (need > 0).then_some((s.rank, s.id, need))
            })
            .collect();
        candidates.sort_unstable();
        for (_, id, need) in candidates {
            let (pinned, seen) = self.pinned_embeddings();
            if pinned + need > limit {
                self.counters.blocked.embedding += 1;
                break;
            }
            self.embedding_reserved.insert(id);
            let room = limit - pinned - need;
            let shared: usize = self
                .embedding_cache
                .iter_lru()
                .filter(|(_, b)| seen.contains(&(Arc::as_ptr(&b.0) as usize)))
                .map(|(_, b)| scriptor_cache::DeviceBuffer::byte_len(&*b.0))
                .sum();
            self.embedding_cache.evict_lru_unless(
                |_, b| seen.contains(&(Arc::as_ptr(&b.0) as usize)),
                room + shared,
            );
        }
    }
}
