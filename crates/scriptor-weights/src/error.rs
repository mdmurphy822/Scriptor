use std::path::PathBuf;

/// Failures raised while reading safetensors directories and mapping tensor names.
#[derive(Debug, thiserror::Error)]
pub enum WeightsError {
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("no *.safetensors files in {0}")]
    NoShards(PathBuf),
    #[error("{shard}: malformed safetensors header: {reason}")]
    BadHeader { shard: String, reason: String },
    #[error("{shard}: tensor {name}: {reason}")]
    BadTensor {
        shard: String,
        name: String,
        reason: String,
    },
    #[error("tensor {name} appears in both {first} and {second}")]
    DuplicateTensor {
        name: String,
        first: String,
        second: String,
    },
    #[error("index file {path}: {reason}")]
    BadIndex { path: PathBuf, reason: String },
    #[error("unsupported safetensors dtype {0:?}")]
    UnknownDtype(String),
    #[error("unmapped tensor name {0:?}")]
    UnmappedTensor(String),
    #[error("tensor {0:?} not present")]
    MissingTensor(String),
}

impl WeightsError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
