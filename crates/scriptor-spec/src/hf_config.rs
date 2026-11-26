//! Deserialisation of the checkpoint's `config.json` and `generation_config.json`.

use serde::Deserialize;

use crate::error::SpecError;

/// Top-level `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct HfConfig {
    pub model_type: String,
    #[serde(default)]
    pub architectures: Vec<String>,
    pub text_config: HfTextConfig,
    pub vision_config: HfVisionConfig,
    pub image_token_id: u32,
    pub image_start_token_id: u32,
    pub image_end_token_id: u32,
    #[serde(default)]
    pub video_token_id: Option<u32>,
    #[serde(default)]
    pub video_start_token_id: Option<u32>,
    #[serde(default)]
    pub video_end_token_id: Option<u32>,
}

/// `text_config` section.
#[derive(Debug, Clone, Deserialize)]
pub struct HfTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub num_nextn_predict_layers: usize,
    #[serde(default)]
    pub eos_token_id: EosIds,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    pub rope_parameters: HfTextRope,
}

/// Text `rope_parameters`.
#[derive(Debug, Clone, Deserialize)]
pub struct HfTextRope {
    pub rope_theta: f64,
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
}

fn default_partial_rotary() -> f64 {
    1.0
}

/// `vision_config` section.
#[derive(Debug, Clone, Deserialize)]
pub struct HfVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    #[serde(default = "default_channels")]
    pub in_channels: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_parameters: Option<HfVisionRope>,
}

fn default_channels() -> usize {
    3
}

/// Vision `rope_parameters`.
#[derive(Debug, Clone, Deserialize)]
pub struct HfVisionRope {
    #[serde(default = "default_vision_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_type: Option<String>,
}

fn default_vision_theta() -> f64 {
    10000.0
}

/// One end-of-sequence id or a list of them.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(untagged)]
pub enum EosIds {
    #[default]
    None,
    One(u32),
    Many(Vec<u32>),
}

impl EosIds {
    /// The ids as a vector.
    pub fn to_vec(&self) -> Vec<u32> {
        match self {
            Self::None => Vec::new(),
            Self::One(id) => vec![*id],
            Self::Many(ids) => ids.clone(),
        }
    }
}

/// `generation_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    #[serde(default)]
    pub eos_token_id: EosIds,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub do_sample: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
}

impl HfConfig {
    /// Parses `config.json` text.
    pub fn from_json(text: &str) -> Result<Self, SpecError> {
        serde_json::from_str(text).map_err(|source| SpecError::Parse {
            what: "config.json",
            source,
        })
    }
}

impl GenerationConfig {
    /// Parses `generation_config.json` text.
    pub fn from_json(text: &str) -> Result<Self, SpecError> {
        serde_json::from_str(text).map_err(|source| SpecError::Parse {
            what: "generation_config.json",
            source,
        })
    }
}
