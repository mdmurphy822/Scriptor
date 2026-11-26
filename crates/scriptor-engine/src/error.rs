//! Failures reported by the scheduler, the executor boundary and the engine handle.

/// Why a request was refused or a step failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EngineError {
    /// The request does not fit the context window; the text is the user-facing 400 message.
    #[error("{0}")]
    ContextOverflow(String),
    /// A single image needs more patches than the vision batch budget holds.
    #[error("image of {patches} patches exceeds the vision batch budget of {budget} patches")]
    ImageTooLarge { patches: usize, budget: usize },
    /// The embeddings the request needs exceed the device embedding budget.
    #[error("request needs {required} bytes of image embeddings but the budget is {budget} bytes")]
    EmbeddingBudget { required: usize, budget: usize },
    /// An image's patch count is not a whole number of merge groups.
    #[error("image of {patches} patches is not a multiple of the {group}-patch merge group")]
    InvalidImage { patches: usize, group: usize },
    /// The prompt's image placeholder count disagrees with the images supplied.
    #[error(
        "prompt holds {placeholders} image placeholder tokens but the images supply {expected}"
    )]
    PlaceholderMismatch {
        placeholders: usize,
        expected: usize,
    },
    /// The prompt has no tokens.
    #[error("the prompt is empty")]
    EmptyPrompt,
    /// The KV cache can never hold the prompt.
    #[error("the KV cache of {blocks} blocks cannot hold a prompt of {tokens} tokens")]
    PromptExceedsCache { blocks: usize, tokens: usize },
    /// The model executor reported a failure.
    #[error("model executor failed: {0}")]
    Executor(String),
    /// The device is in a sticky fault state; the process must be relaunched. `code` is the
    /// driver result code.
    #[error("device fault (code {code}): {message}")]
    Fatal { code: i32, message: String },
    /// A cache operation failed.
    #[error("cache failure: {0}")]
    Cache(String),
    /// The request was cancelled to relieve host memory pressure.
    #[error("shed under memory pressure")]
    Shed,
    /// New requests are refused while host memory is under pressure.
    #[error("host memory is under pressure; retry in {retry_after_secs} s")]
    Pressure { retry_after_secs: u64 },
    #[cfg(feature = "kv-demand-diagnostic")]
    /// The queue of requests waiting for admission is full.
    #[error("the admission queue is full; retry in {retry_after_secs} s")]
    QueueFull { retry_after_secs: u64 },
    /// The engine thread has stopped.
    #[error("the engine has shut down")]
    Closed,
    /// The engine is draining for shutdown and accepts no new requests.
    #[error("the engine is draining and accepts no new requests")]
    Draining,
}

impl EngineError {
    /// A sticky device fault with the driver result `code`.
    pub fn fatal(code: i32, message: impl Into<String>) -> Self {
        Self::Fatal {
            code,
            message: message.into(),
        }
    }

    /// Whether the error leaves the device unusable for the rest of the process.
    pub fn is_fatal(&self) -> bool {
        matches!(self, Self::Fatal { .. })
    }
}

impl From<scriptor_cache::CacheError> for EngineError {
    fn from(e: scriptor_cache::CacheError) -> Self {
        EngineError::Cache(e.to_string())
    }
}
