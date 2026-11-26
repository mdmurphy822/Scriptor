//! Model configuration, runtime parameters, memory plan and fail-closed validation.

pub mod error;
pub mod graph_buckets;
pub mod hf_config;
pub mod image_proc;
pub mod memory_plan;
pub mod model_spec;
pub mod runtime_params;

pub use error::SpecError;
pub use graph_buckets::{StepShape, bucket_for, graph_buckets, step_shape};
pub use hf_config::{GenerationConfig, HfConfig, HfTextConfig, HfVisionConfig};
pub use image_proc::{ImageProcessorConfig, Resample};
pub use memory_plan::{MemoryPlan, MemoryPlanInputs};
pub use model_spec::{ModelSpec, SamplingDefaults, TextSpec, VisionSpec};
pub use runtime_params::{KvPolicy, RuntimeParams, VisionBatchPatches};
