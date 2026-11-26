//! Paged KV block pool and image-embedding cache.

pub mod block_pool;
pub mod embedding_cache;
pub mod error;
pub mod hash;
pub mod kv_layout;
pub mod preempt;

pub use block_pool::{BlockId, BlockPool, PoolStats, SeqId};
pub use embedding_cache::{
    DeviceBuffer, EmbeddingCache, EmbeddingCacheStats, EmbeddingEntry, HostBuffer, InsertOutcome,
};
pub use error::CacheError;
pub use hash::{EmbeddingKey, ImageGrid, embedding_key};
pub use kv_layout::KvLayout;
pub use preempt::{Reservation, ReserveOutcome};
