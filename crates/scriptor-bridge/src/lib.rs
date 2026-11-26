//! The document OCR bridge: renders the pages of a document, detects layout
//! regions, recognises each region through an OpenAI-compatible engine, and
//! serves the OCR contract over HTTP.

pub mod assemble;
pub mod cli;
pub mod client;
pub mod crop;
pub mod document;
pub mod format;
pub mod http;
pub mod prompt;
pub mod raster;
pub mod recover;
pub mod recovery_finisher;
pub mod render;
pub mod request_policy;
pub mod schedule;
pub mod serve;
pub mod text_layer;
pub mod text_layer_rules;
pub mod textutil;
pub mod wire;
