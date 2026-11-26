//! Readable f32 reference forward pass.

pub mod decoder;
pub mod merger;
pub mod model;
pub mod mtp;
pub mod vision;

use scriptor_kernels::reference::gemm::gemm_bias;
use scriptor_weights::{Dtype, SafetensorsDir, WeightsError};

pub use decoder::{DecoderLayer, KvCache, TextModel};
pub use merger::Merger;
pub use model::{GenerateOutput, GreedyStep, ReferenceModel};
pub use mtp::MtpHead;
pub use vision::VisionTower;

/// Failures raised while loading weights or running the reference forward pass.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Weights(#[from] WeightsError),
    #[error("tensor {name}: expected shape {expected:?}, found {found:?}")]
    Shape {
        name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    #[error("tensor {name}: dtype {dtype:?} cannot be read as f32")]
    Dtype { name: String, dtype: Dtype },
    #[error("unsupported model configuration: {0}")]
    Unsupported(String),
    #[error("invalid input: {0}")]
    Input(String),
}

/// A dense layer `y = x * W^T (+ b)` with `W` stored `[out, in]`.
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl Linear {
    /// Applies the layer to `rows` input rows (`x` is `[rows, in_dim]`).
    pub fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        linear(
            x,
            &self.weight,
            self.bias.as_deref(),
            rows,
            self.out_dim,
            self.in_dim,
        )
    }
}

/// Row-major `A [m, k] * W [n, k]^T + bias`, split over output columns across the
/// available cores when the product is large enough to benefit.
pub fn linear(
    a: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    if threads <= 1 || m * n * k < (1 << 22) || n < 2 * threads {
        return gemm_bias(a, w, bias, m, n, k);
    }
    let chunk = n.div_ceil(threads);
    let parts: Vec<(usize, Vec<f32>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|c0| {
                let c1 = (c0 + chunk).min(n);
                scope.spawn(move || {
                    let part = gemm_bias(
                        a,
                        &w[c0 * k..c1 * k],
                        bias.map(|b| &b[c0..c1]),
                        m,
                        c1 - c0,
                        k,
                    );
                    (c0, part)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("gemm worker panicked"))
            .collect()
    });
    let mut out = vec![0.0; m * n];
    for (c0, part) in parts {
        let width = part.len() / m;
        for i in 0..m {
            out[i * n + c0..i * n + c0 + width].copy_from_slice(&part[i * width..(i + 1) * width]);
        }
    }
    out
}

/// Reads checkpoint tensors as f32 with shape validation.
pub struct TensorSource<'a> {
    dir: &'a SafetensorsDir,
}

impl<'a> TensorSource<'a> {
    pub fn new(dir: &'a SafetensorsDir) -> Self {
        Self { dir }
    }

    /// Loads `name`, converting BF16, F16 or F32 storage to f32; the stored shape must equal
    /// `shape`.
    pub fn tensor(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, ModelError> {
        let t = self.dir.get(name)?;
        if t.shape != shape {
            return Err(ModelError::Shape {
                name: name.to_string(),
                expected: shape.to_vec(),
                found: t.shape.clone(),
            });
        }
        let bytes = self.dir.bytes(t);
        let out = match t.dtype {
            Dtype::Bf16 => bytes
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            dtype => {
                return Err(ModelError::Dtype {
                    name: name.to_string(),
                    dtype,
                });
            }
        };
        Ok(out)
    }

    /// Loads a `[out, in]` weight and an optional `[out]` bias as a [`Linear`].
    pub fn linear(
        &self,
        weight: &str,
        bias: Option<&str>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Linear, ModelError> {
        Ok(Linear {
            weight: self.tensor(weight, &[out_dim, in_dim])?,
            bias: bias.map(|b| self.tensor(b, &[out_dim])).transpose()?,
            out_dim,
            in_dim,
        })
    }
}

/// Adds `delta` into `x` elementwise.
pub(crate) fn add_in_place(x: &mut [f32], delta: &[f32]) {
    assert_eq!(x.len(), delta.len());
    for (a, b) in x.iter_mut().zip(delta) {
        *a += *b;
    }
}
