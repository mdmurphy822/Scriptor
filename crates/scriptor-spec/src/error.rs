//! Error type shared by configuration parsing, validation and memory planning.

use thiserror::Error;

/// Failure raised while reading, validating or planning from configuration.
#[derive(Debug, Error)]
pub enum SpecError {
    /// A configuration document is not valid JSON of the expected shape.
    #[error("invalid {what}: {source}")]
    Parse {
        what: &'static str,
        source: serde_json::Error,
    },
    /// A configuration value violates a model or runtime invariant.
    #[error("invalid configuration: {0}")]
    Invalid(String),
    /// The requested runtime parameters need more device memory than the share allows.
    #[error(
        "does not fit: needs {needed} bytes but only {available} bytes are available ({suggestion})"
    )]
    DoesNotFit {
        needed: u64,
        available: u64,
        suggestion: String,
    },
}

impl SpecError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}
