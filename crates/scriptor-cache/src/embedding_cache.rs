//! LRU cache of image embeddings (`[rows, hidden]` bf16) over a byte budget.

use crate::error::CacheError;
use crate::hash::EmbeddingKey;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// A device-resident (or host stand-in) buffer with a known byte size.
pub trait DeviceBuffer {
    fn byte_len(&self) -> usize;
}

/// Host-memory buffer implementing [`DeviceBuffer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostBuffer(pub Vec<u8>);

impl DeviceBuffer for HostBuffer {
    fn byte_len(&self) -> usize {
        self.0.len()
    }
}

/// One cached embedding block.
#[derive(Debug)]
pub struct EmbeddingEntry<B> {
    pub rows: usize,
    pub buffer: B,
}

/// Lifetime counters of the cache.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmbeddingCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub rejected: u64,
    pub entries: usize,
    pub bytes: usize,
    pub budget_bytes: usize,
}

/// What [`EmbeddingCache::insert`] did with the entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertOutcome {
    Stored {
        evicted: usize,
    },
    /// The entry alone exceeds the budget; it is not retained.
    Rejected,
}

struct Slot<B> {
    entry: Arc<EmbeddingEntry<B>>,
    tick: u64,
}

/// LRU embedding cache. Evicted entries still referenced by a running request stay alive
/// through their `Arc` until released.
pub struct EmbeddingCache<B> {
    hidden: usize,
    elem_bytes: usize,
    budget: usize,
    bytes: usize,
    tick: u64,
    slots: HashMap<EmbeddingKey, Slot<B>>,
    order: BTreeMap<u64, EmbeddingKey>,
    hits: u64,
    misses: u64,
    evictions: u64,
    rejected: u64,
}

impl<B: DeviceBuffer> EmbeddingCache<B> {
    /// `hidden` is the embedding width (1536) and `elem_bytes` the element size (2 for bf16).
    pub fn new(budget_bytes: usize, hidden: usize, elem_bytes: usize) -> Self {
        Self {
            hidden,
            elem_bytes,
            budget: budget_bytes,
            bytes: 0,
            tick: 0,
            slots: HashMap::new(),
            order: BTreeMap::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
            rejected: 0,
        }
    }

    fn touch(&mut self, key: &EmbeddingKey) {
        self.tick += 1;
        let tick = self.tick;
        if let Some(slot) = self.slots.get_mut(key) {
            self.order.remove(&slot.tick);
            slot.tick = tick;
            self.order.insert(tick, *key);
        }
    }

    /// Returns the cached embedding and marks it most recently used; counts a hit or a miss.
    pub fn get(&mut self, key: &EmbeddingKey) -> Option<Arc<EmbeddingEntry<B>>> {
        if self.slots.contains_key(key) {
            self.hits += 1;
            self.touch(key);
            Some(Arc::clone(&self.slots[key].entry))
        } else {
            self.misses += 1;
            None
        }
    }

    /// Presence check that neither counts nor reorders.
    pub fn contains(&self, key: &EmbeddingKey) -> bool {
        self.slots.contains_key(key)
    }

    /// Stores an embedding of `rows` rows, evicting least-recently-used entries until it fits.
    /// Re-inserting an existing key replaces it.
    pub fn insert(
        &mut self,
        key: EmbeddingKey,
        rows: usize,
        buffer: B,
    ) -> Result<InsertOutcome, CacheError> {
        let expected = rows * self.hidden * self.elem_bytes;
        let actual = buffer.byte_len();
        if actual != expected {
            return Err(CacheError::EmbeddingSizeMismatch {
                rows,
                expected,
                actual,
            });
        }
        self.remove(&key);
        if actual > self.budget {
            self.rejected += 1;
            return Ok(InsertOutcome::Rejected);
        }
        let mut evicted = 0;
        while self.bytes + actual > self.budget {
            let (&tick, &victim) = self.order.iter().next().expect("bytes > 0 implies entries");
            self.order.remove(&tick);
            let slot = self.slots.remove(&victim).expect("order and slots agree");
            self.bytes -= slot.entry.buffer.byte_len();
            self.evictions += 1;
            evicted += 1;
        }
        self.tick += 1;
        self.order.insert(self.tick, key);
        self.slots.insert(
            key,
            Slot {
                entry: Arc::new(EmbeddingEntry { rows, buffer }),
                tick: self.tick,
            },
        );
        self.bytes += actual;
        Ok(InsertOutcome::Stored { evicted })
    }

    /// Entries from least to most recently used, without counting or reordering.
    pub fn iter_lru(&self) -> impl Iterator<Item = (&EmbeddingKey, &B)> {
        self.order
            .values()
            .map(|key| (key, &self.slots[key].entry.buffer))
    }

    /// Evicts least-recently-used entries for which `keep` is false until the held bytes are at
    /// most `target_bytes` or only kept entries remain; returns the entries evicted.
    pub fn evict_lru_unless(
        &mut self,
        mut keep: impl FnMut(&EmbeddingKey, &B) -> bool,
        target_bytes: usize,
    ) -> usize {
        let victims: Vec<EmbeddingKey> = self
            .iter_lru()
            .filter(|(k, b)| !keep(k, b))
            .map(|(k, _)| *k)
            .collect();
        let mut evicted = 0;
        for key in victims {
            if self.bytes <= target_bytes {
                break;
            }
            self.remove(&key);
            self.evictions += 1;
            evicted += 1;
        }
        evicted
    }

    /// Drops an entry without counting an eviction; returns whether it existed.
    pub fn remove(&mut self, key: &EmbeddingKey) -> bool {
        match self.slots.remove(key) {
            Some(slot) => {
                self.order.remove(&slot.tick);
                self.bytes -= slot.entry.buffer.byte_len();
                true
            }
            None => false,
        }
    }

    /// Drops every entry without counting evictions; entries a running request still holds stay
    /// alive through their `Arc`.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.order.clear();
        self.bytes = 0;
    }

    pub fn stats(&self) -> EmbeddingCacheStats {
        EmbeddingCacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            rejected: self.rejected,
            entries: self.slots.len(),
            bytes: self.bytes,
            budget_bytes: self.budget,
        }
    }
}
