//! CPU reference model and GPU executors.
//!
//! The `reference` module is a readable f32 forward pass of the whole model: patch
//! embedding, vision transformer, adapter, embedding scatter, decoder and the
//! multi-token-prediction layer.

#[cfg(feature = "cuda")]
pub mod gpu;
pub mod reference;

pub use reference::{
    DecoderLayer, GenerateOutput, GreedyStep, KvCache, Linear, Merger, ModelError, MtpHead,
    ReferenceModel, TextModel, VisionTower,
};
