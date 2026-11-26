//! Checkpoint reader, tensor name map and engine file.

pub mod dtype;
pub mod engine_file;
pub mod error;
pub mod loader;
pub mod names;
pub mod prefetch;
pub mod repack;
pub mod safetensors_dir;

pub use dtype::Dtype;
pub use engine_file::{EngineFileError, EntryMeta, Header, LAYOUT_VERSION, SourceShard, Transform};
pub use error::WeightsError;
pub use loader::{EngineFile, LayerGroup, file_sha256, source_fingerprint};
pub use names::{
    FusedParam, Kind, LayerRole, MergerRole, MtpRole, NameLayout, ParamName, Placement, VisionRole,
    all_params, classify, fused_targets,
};
pub use prefetch::{HeapStaging, LayerPrefetcher, LoadedGroup, Staging, StagingBuffer};
pub use repack::{ConvertReport, convert, fused_name, infer_layout, revision_of};
pub use safetensors_dir::{SafetensorsDir, ShardHeader, TensorRef, parse_header};
