//! Checkpoint-derived configuration shared by `serve` and `doctor`.

use std::path::Path;

use scriptor_spec::{GenerationConfig, HfConfig, ImageProcessorConfig, ModelSpec, RuntimeParams};
use scriptor_weights::SafetensorsDir;

use crate::args::{DoctorArgs, ServeArgs};

/// Model description read from a checkpoint directory.
pub struct Setup {
    pub spec: ModelSpec,
    pub image: ImageProcessorConfig,
    pub dir: SafetensorsDir,
    pub weight_bytes: u64,
}

impl Setup {
    /// Reads configs and indexes the safetensors shards under `model`.
    pub fn load(model: &Path) -> Result<Self, String> {
        let read = |name: &str| {
            std::fs::read_to_string(model.join(name)).map_err(|e| format!("{name}: {e}"))
        };
        let config = HfConfig::from_json(&read("config.json")?).map_err(|e| e.to_string())?;
        let generation = read("generation_config.json")
            .ok()
            .map(|t| GenerationConfig::from_json(&t))
            .transpose()
            .map_err(|e| e.to_string())?;
        let spec = ModelSpec::from_hf(&config, generation.as_ref()).map_err(|e| e.to_string())?;
        let image = match read("preprocessor_config.json") {
            Ok(t) => ImageProcessorConfig::from_preprocessor_json(&t),
            Err(_) => ImageProcessorConfig::from_processor_json(&read("processor_config.json")?),
        }
        .map_err(|e| e.to_string())?;
        let dir = SafetensorsDir::open(model).map_err(|e| e.to_string())?;
        let weight_bytes = dir.tensors().values().map(|t| t.byte_len()).sum();
        Ok(Self {
            spec,
            image,
            dir,
            weight_bytes,
        })
    }
}

impl ServeArgs {
    /// Runtime parameters the flags select.
    pub fn runtime_params(&self) -> RuntimeParams {
        RuntimeParams {
            max_context: self.max_model_len,
            max_concurrency: self
                .max_num_seqs
                .unwrap_or(RuntimeParams::default().max_concurrency),
            gpu_share: self.gpu_memory_utilization,
            served_model_name: self.served_model_name.clone(),
            max_pixels: self.max_pixels,
            speculative_tokens: self.num_speculative_tokens,
            kv_policy: self.kv_policy,
            ..RuntimeParams::default()
        }
    }
}

impl DoctorArgs {
    /// Runtime parameters the flags select.
    pub fn runtime_params(&self) -> RuntimeParams {
        RuntimeParams {
            max_context: self.max_model_len,
            max_concurrency: self
                .max_num_seqs
                .unwrap_or(RuntimeParams::default().max_concurrency),
            gpu_share: self.gpu_memory_utilization,
            speculative_tokens: self.num_speculative_tokens,
            ..RuntimeParams::default()
        }
    }
}
