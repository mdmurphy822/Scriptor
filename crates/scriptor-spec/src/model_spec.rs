//! Validated architecture description derived from the checkpoint configuration.

use crate::error::SpecError;
use crate::hf_config::{GenerationConfig, HfConfig};

/// Fraction of the vision head dimension that is rotated by the 2-D axial RoPE.
const VISION_PARTIAL_ROTARY: f64 = 0.5;

/// Text decoder dimensions and behaviour.
#[derive(Debug, Clone, PartialEq)]
pub struct TextSpec {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rotary_dim: usize,
    /// Interleaved (GPT-J style) rotation pairs when `false`.
    pub is_neox_style: bool,
    /// Frequency pairs given to the temporal, height and width axes.
    pub mrope_section: [usize; 3],
    pub max_position_embeddings: usize,
    /// Four RMSNorms per layer (input, post-attention-output, pre-MLP, post-MLP).
    pub sandwich_norms: bool,
    pub tie_word_embeddings: bool,
    pub num_mtp_layers: usize,
    pub eos_token_ids: Vec<u32>,
    pub pad_token_id: Option<u32>,
}

/// Vision encoder and adapter dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionSpec {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub in_channels: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// Rotated dimensions of the vision head (neox layout).
    pub rotary_dim: usize,
    pub is_neox_style: bool,
    /// Per-head RMSNorm on q and k before RoPE.
    pub qk_norm: bool,
    /// Merger MLP intermediate width.
    pub merger_intermediate: usize,
}

impl VisionSpec {
    /// Flattened length of one patch (`channels * temporal * patch^2`).
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// Patches merged into one output token.
    pub fn merge_factor(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size
    }
}

/// Sampling values a request inherits for the fields it omits; `None` leaves a field to the
/// server's built-in default.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SamplingDefaults {
    /// Temperature; `0.0` is greedy.
    pub temperature: Option<f32>,
    /// Nucleus mass.
    pub top_p: Option<f32>,
    /// Candidates kept.
    pub top_k: Option<i32>,
    /// Repetition penalty.
    pub repetition_penalty: Option<f32>,
}

impl SamplingDefaults {
    /// Defaults declared by a generation config: `do_sample: false` selects greedy decoding,
    /// otherwise the declared temperature applies.
    pub fn from_generation(generation: Option<&GenerationConfig>) -> Self {
        let Some(g) = generation else {
            return Self::default();
        };
        let temperature = if g.do_sample == Some(false) {
            Some(0.0)
        } else {
            g.temperature
        };
        Self {
            temperature,
            top_p: g.top_p,
            top_k: g.top_k,
            repetition_penalty: g.repetition_penalty,
        }
    }
}

/// Complete validated model description.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    pub sampling_defaults: SamplingDefaults,
    pub text: TextSpec,
    pub vision: VisionSpec,
    pub image_token_id: u32,
    pub image_start_token_id: u32,
    pub image_end_token_id: u32,
}

impl ModelSpec {
    /// Builds and validates the specification; `generation` supplies the stop ids when the
    /// text config carries none.
    pub fn from_hf(
        config: &HfConfig,
        generation: Option<&GenerationConfig>,
    ) -> Result<Self, SpecError> {
        let t = &config.text_config;
        let v = &config.vision_config;

        let section: [usize; 3] = t
            .rope_parameters
            .mrope_section
            .as_slice()
            .try_into()
            .map_err(|_| SpecError::invalid("mrope_section must have exactly three entries"))?;
        let rotary_dim = (t.head_dim as f64 * t.rope_parameters.partial_rotary_factor) as usize;
        if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > t.head_dim {
            return Err(SpecError::invalid("text rotary dimension is invalid"));
        }
        if section.iter().sum::<usize>() != rotary_dim / 2 {
            return Err(SpecError::invalid(format!(
                "mrope_section {section:?} must sum to {} frequency pairs",
                rotary_dim / 2
            )));
        }
        if t.num_attention_heads == 0
            || t.num_key_value_heads == 0
            || !t.num_attention_heads.is_multiple_of(t.num_key_value_heads)
        {
            return Err(SpecError::invalid(
                "attention heads must be a multiple of KV heads",
            ));
        }
        if v.num_heads == 0 || !v.hidden_size.is_multiple_of(v.num_heads) {
            return Err(SpecError::invalid(
                "vision hidden size must divide into heads",
            ));
        }
        let vision_head_dim = v.hidden_size / v.num_heads;
        let vision_rotary = (vision_head_dim as f64 * VISION_PARTIAL_ROTARY) as usize;
        if vision_rotary == 0 || !(vision_rotary / 2).is_multiple_of(2) {
            return Err(SpecError::invalid(
                "vision rotary pairs must split evenly between height and width",
            ));
        }
        if v.out_hidden_size != t.hidden_size {
            return Err(SpecError::invalid(
                "vision out_hidden_size must equal the text hidden size",
            ));
        }
        if v.out_hidden_size * 3 != t.intermediate_size {
            return Err(SpecError::invalid(format!(
                "merger width mismatch: out_hidden_size {} x 3 != text intermediate {}",
                v.out_hidden_size, t.intermediate_size
            )));
        }
        if t.tie_word_embeddings {
            return Err(SpecError::invalid("tied word embeddings are not supported"));
        }

        let mut eos = t.eos_token_id.to_vec();
        if eos.is_empty()
            && let Some(g) = generation
        {
            eos = g.eos_token_id.to_vec();
        }
        if eos.is_empty() {
            return Err(SpecError::invalid(
                "no end-of-sequence token ids configured",
            ));
        }
        let pad = t
            .pad_token_id
            .or_else(|| generation.and_then(|g| g.pad_token_id));

        let rope_theta = v.rope_parameters.as_ref().map_or(10000.0, |r| r.rope_theta);

        Ok(Self {
            sampling_defaults: SamplingDefaults::from_generation(generation),
            text: TextSpec {
                hidden_size: t.hidden_size,
                intermediate_size: t.intermediate_size,
                num_layers: t.num_hidden_layers,
                num_heads: t.num_attention_heads,
                num_kv_heads: t.num_key_value_heads,
                head_dim: t.head_dim,
                vocab_size: t.vocab_size,
                rms_norm_eps: t.rms_norm_eps,
                rope_theta: t.rope_parameters.rope_theta,
                rotary_dim,
                is_neox_style: false,
                mrope_section: section,
                max_position_embeddings: t.max_position_embeddings,
                sandwich_norms: true,
                tie_word_embeddings: false,
                num_mtp_layers: t.num_nextn_predict_layers,
                eos_token_ids: eos,
                pad_token_id: pad,
            },
            vision: VisionSpec {
                depth: v.depth,
                hidden_size: v.hidden_size,
                num_heads: v.num_heads,
                head_dim: vision_head_dim,
                intermediate_size: v.intermediate_size,
                out_hidden_size: v.out_hidden_size,
                patch_size: v.patch_size,
                spatial_merge_size: v.spatial_merge_size,
                temporal_patch_size: v.temporal_patch_size,
                in_channels: v.in_channels,
                rms_norm_eps: v.rms_norm_eps,
                rope_theta,
                rotary_dim: vision_rotary,
                is_neox_style: true,
                qk_norm: true,
                merger_intermediate: v.out_hidden_size * 3,
            },
            image_token_id: config.image_token_id,
            image_start_token_id: config.image_start_token_id,
            image_end_token_id: config.image_end_token_id,
        })
    }

    /// Parses `config.json` and optional `generation_config.json` text into a specification.
    pub fn from_json(config: &str, generation: Option<&str>) -> Result<Self, SpecError> {
        let config = HfConfig::from_json(config)?;
        let generation = generation.map(GenerationConfig::from_json).transpose()?;
        Self::from_hf(&config, generation.as_ref())
    }

    /// Key-value cache bytes per token for the decoder (BF16).
    pub fn kv_bytes_per_token(&self) -> u64 {
        let t = &self.text;
        (t.num_layers * 2 * t.num_kv_heads * t.head_dim * 2) as u64
    }

    /// Key-value cache bytes per token for the MTP layers (BF16).
    pub fn mtp_kv_bytes_per_token(&self) -> u64 {
        let t = &self.text;
        (t.num_mtp_layers * 2 * t.num_kv_heads * t.head_dim * 2) as u64
    }
}
