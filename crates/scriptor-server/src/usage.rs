//! Token accounting reported with every completion.

use serde::Serialize;

/// Token counts of one completed request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Usage {
    /// Prompt tokens, including expanded image tokens.
    pub prompt_tokens: u64,
    /// Generated tokens.
    pub completion_tokens: u64,
    /// Sum of prompt and completion tokens.
    pub total_tokens: u64,
}

impl Usage {
    /// Builds usage from prompt and completion counts.
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}
