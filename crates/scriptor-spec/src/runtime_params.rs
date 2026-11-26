//! Runtime parameters with command-line and `SCRIPTOR_*` environment sources.

use std::str::FromStr;

use clap::Args;

use crate::error::SpecError;

/// Patch budget of one vision batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionBatchPatches {
    /// Sized to one image of the largest permitted resolution.
    Auto,
    Fixed(usize),
}

impl FromStr for VisionBatchPatches {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        match s.parse::<usize>() {
            Ok(n) if n > 0 => Ok(Self::Fixed(n)),
            _ => Err(format!("expected `auto` or a positive integer, got `{s}`")),
        }
    }
}

/// How KV blocks are committed to running sequences.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KvPolicy {
    /// Every admitted sequence holds a promise for the blocks its full output may need, so a
    /// running sequence never waits for blocks.
    #[default]
    Reserve,
    /// Admission is bounded by the blocks running sequences need for their next step, not by
    /// their full output. Only compiled with the `kv-demand-diagnostic` feature.
    #[cfg(feature = "kv-demand-diagnostic")]
    DemandDiagnostic,
}

impl KvPolicy {
    /// The selector spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reserve => "reserve",
            #[cfg(feature = "kv-demand-diagnostic")]
            Self::DemandDiagnostic => "demand-diagnostic",
        }
    }
}

impl std::fmt::Display for KvPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for KvPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "reserve" => Ok(Self::Reserve),
            #[cfg(feature = "kv-demand-diagnostic")]
            "demand-diagnostic" => Ok(Self::DemandDiagnostic),
            #[cfg(not(feature = "kv-demand-diagnostic"))]
            "demand-diagnostic" => Err("kv policy `demand-diagnostic` is not compiled into this \
                 build (it needs the `kv-demand-diagnostic` cargo feature)"
                .to_string()),
            other => Err(format!(
                "unknown kv policy `{other}`; expected `reserve` or `demand-diagnostic`"
            )),
        }
    }
}

/// Serving parameters; none of them names a filesystem path.
#[derive(Debug, Clone, Args, PartialEq)]
pub struct RuntimeParams {
    /// Longest sequence (prompt plus output) in tokens.
    #[arg(long, env = "SCRIPTOR_MAX_CONTEXT", default_value_t = 16384)]
    pub max_context: usize,
    /// Sequences decoded concurrently.
    #[arg(long, env = "SCRIPTOR_MAX_CONCURRENCY", default_value_t = 32)]
    pub max_concurrency: usize,
    /// Fraction of device memory the engine may use.
    #[arg(long, env = "SCRIPTOR_GPU_SHARE", default_value_t = 0.15)]
    pub gpu_share: f64,
    /// Model name reported by the API.
    #[arg(long, env = "SCRIPTOR_SERVED_MODEL_NAME", default_value = "glm-ocr")]
    pub served_model_name: String,
    /// Overrides the image processor's maximum pixel count.
    #[arg(long, env = "SCRIPTOR_MAX_PIXELS")]
    pub max_pixels: Option<usize>,
    /// Draft tokens proposed per step by the MTP head (0 disables).
    #[arg(long, env = "SCRIPTOR_SPECULATIVE_TOKENS", default_value_t = 0)]
    pub speculative_tokens: usize,
    /// Patches per vision batch: `auto` or a count.
    #[arg(long, env = "SCRIPTOR_VISION_BATCH_PATCHES", default_value = "auto")]
    pub vision_batch_patches: VisionBatchPatches,
    /// Tokens per KV block (16 or 32).
    #[arg(long, env = "SCRIPTOR_KV_BLOCK_TOKENS", default_value_t = 16)]
    pub kv_block_tokens: usize,
    /// KV commitment policy: `reserve` or `demand-diagnostic` (feature-gated builds only).
    #[arg(long, env = "SCRIPTOR_KV_POLICY", default_value = "reserve")]
    pub kv_policy: KvPolicy,
}

impl Default for RuntimeParams {
    fn default() -> Self {
        Self {
            max_context: 16384,
            max_concurrency: 32,
            gpu_share: 0.15,
            served_model_name: "glm-ocr".to_string(),
            max_pixels: None,
            speculative_tokens: 0,
            vision_batch_patches: VisionBatchPatches::Auto,
            kv_block_tokens: 16,
            kv_policy: KvPolicy::Reserve,
        }
    }
}

impl RuntimeParams {
    /// Rejects values outside their valid ranges.
    pub fn validate(&self) -> Result<(), SpecError> {
        if self.max_context == 0 {
            return Err(SpecError::invalid("max_context must be positive"));
        }
        if self.max_concurrency == 0 {
            return Err(SpecError::invalid("max_concurrency must be positive"));
        }
        if !(self.gpu_share > 0.0 && self.gpu_share <= 1.0) {
            return Err(SpecError::invalid("gpu_share must be in (0, 1]"));
        }
        if self.served_model_name.is_empty() {
            return Err(SpecError::invalid("served_model_name must not be empty"));
        }
        if !matches!(self.kv_block_tokens, 16 | 32) {
            return Err(SpecError::invalid("kv_block_tokens must be 16 or 32"));
        }
        if self.max_pixels == Some(0) {
            return Err(SpecError::invalid("max_pixels must be positive"));
        }
        Ok(())
    }
}
