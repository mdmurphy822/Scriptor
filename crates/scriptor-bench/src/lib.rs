//! Equivalence, throughput and mock-seat benchmarks.
//!
//! Region sets are crops with a recognition task each; `equivalence` compares two
//! OpenAI-compatible endpoints token for token, `ladder` measures throughput at
//! rising concurrency, and `mock_seat` answers the Semantik request instantly.

pub mod equivalence;
pub mod ladder;
pub mod mock_seat;
pub mod openai_client;
pub mod regions;
pub mod report;

use std::fmt;

/// Failure of a benchmark operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchError(pub String);

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BenchError {}

impl From<std::io::Error> for BenchError {
    fn from(e: std::io::Error) -> Self {
        Self(e.to_string())
    }
}

impl From<serde_json::Error> for BenchError {
    fn from(e: serde_json::Error) -> Self {
        Self(e.to_string())
    }
}

/// Result alias of the crate.
pub type Result<T> = std::result::Result<T, BenchError>;
