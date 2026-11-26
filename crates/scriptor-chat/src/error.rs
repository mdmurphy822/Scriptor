//! Errors raised by tokenization and message handling.

/// Failure while loading a tokenizer, encoding, decoding or converting messages.
#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    /// The tokenizer definition could not be read or parsed.
    #[error("tokenizer load failed: {0}")]
    TokenizerLoad(String),
    /// Encoding text to token ids failed.
    #[error("encode failed: {0}")]
    Encode(String),
    /// Decoding token ids to text failed.
    #[error("decode failed: {0}")]
    Decode(String),
    /// A named special token is not part of the vocabulary.
    #[error("token {0:?} is not in the vocabulary")]
    UnknownToken(String),
    /// A message violates the OpenAI chat message contract.
    #[error("invalid message: {0}")]
    InvalidMessage(String),
    /// Parsing or rendering a chat template failed, or the template raised an exception.
    #[error("chat template error: {0}")]
    Template(String),
    /// The images supplied do not match the image placeholders of the prompt, or an image grid is
    /// malformed.
    #[error("image layout error: {0}")]
    ImageLayout(String),
    /// Reading a file failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, ChatError>;
