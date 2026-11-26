//! Tokenizer, chat template, prompt builder and detokenisation.

pub mod detok;
pub mod error;
pub mod messages;
pub mod positions;
pub mod prompt;
pub mod template;
pub mod tokenizer;

pub use detok::{ByteDetokenizer, ByteTable};
pub use error::{ChatError, Result};
pub use messages::{
    ChatMessage, Content, ContentPart, FunctionCall, ImageUrl, ToolCall, messages_to_template,
};
pub use positions::{ImageSpan, MropePositions, decode_position, mrope_positions};
pub use prompt::{ImageGrid, PreparedPrompt, Prompt, PromptBuilder};
pub use template::{ChatTemplate, RenderOptions};
pub use tokenizer::{IncrementalDecoder, Tokenizer};
