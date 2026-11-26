//! Error type for the cache crate.

/// Failures reported by the block pool and the embedding cache.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CacheError {
    #[error("sequence {0} is not registered in the block pool")]
    UnknownSequence(u64),
    #[error("sequence {0} is already registered in the block pool")]
    DuplicateSequence(u64),
    #[error("block pool exhausted: need {needed} more blocks, {free} free")]
    OutOfBlocks { needed: usize, free: usize },
    #[error("invalid KV layout: {0}")]
    InvalidLayout(String),
    #[error("embedding buffer holds {actual} bytes, expected {expected} ({rows} rows)")]
    EmbeddingSizeMismatch {
        rows: usize,
        expected: usize,
        actual: usize,
    },
}
