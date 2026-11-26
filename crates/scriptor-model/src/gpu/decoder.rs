//! Device-resident text decoder parameters: geometry, layer weights, embedding table, final norm,
//! language-model head, and the paged key/value pool.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DriverError};
use half::{bf16, f16};
use scriptor_cuda::CudaError;
use scriptor_spec::TextSpec;
use scriptor_weights::{Dtype, SafetensorsDir, WeightsError};

use crate::reference::ModelError;

/// Failures raised by the GPU text executor.
#[derive(Debug, thiserror::Error)]
pub enum GpuTextError {
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Weights(#[from] WeightsError),
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("invalid input: {0}")]
    Input(String),
    #[error("unsupported configuration: {0}")]
    Unsupported(String),
}

impl From<DriverError> for GpuTextError {
    fn from(e: DriverError) -> Self {
        Self::Cuda(CudaError::from(e))
    }
}

/// Dimensions shared by every decoder layer and the MTP layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextGeometry {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub mrope_section: [usize; 3],
}

impl TextGeometry {
    /// Geometry from the text specification. The device attention kernels require a head
    /// dimension of 128, full-head interleaved rotary embedding and a whole GQA ratio.
    pub fn from_spec(spec: &TextSpec) -> Result<Self, GpuTextError> {
        if spec.head_dim != scriptor_kernels::attention::DECODE_HEAD_DIM {
            return Err(GpuTextError::Unsupported(format!(
                "head_dim {} (device attention needs {})",
                spec.head_dim,
                scriptor_kernels::attention::DECODE_HEAD_DIM
            )));
        }
        if spec.rotary_dim != spec.head_dim || spec.is_neox_style {
            return Err(GpuTextError::Unsupported(
                "text decoder requires full-head interleaved rotary embedding".into(),
            ));
        }
        if spec.num_kv_heads == 0 || !spec.num_heads.is_multiple_of(spec.num_kv_heads) {
            return Err(GpuTextError::Unsupported(
                "attention heads must be a multiple of KV heads".into(),
            ));
        }
        Ok(Self {
            hidden: spec.hidden_size,
            intermediate: spec.intermediate_size,
            layers: spec.num_layers,
            heads: spec.num_heads,
            kv_heads: spec.num_kv_heads,
            head_dim: spec.head_dim,
            vocab: spec.vocab_size,
            eps: spec.rms_norm_eps as f32,
            rope_theta: spec.rope_theta as f32,
            mrope_section: spec.mrope_section,
        })
    }

    /// Width of the query projection output.
    pub fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    /// Width of one key (or value) row.
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// Uploads checkpoint tensor `name` (expected `shape`) as bf16 bits, converting F16 or F32
/// storage.
pub(crate) fn upload_bf16(
    stream: &Arc<CudaStream>,
    dir: &SafetensorsDir,
    name: &str,
    shape: &[usize],
) -> Result<CudaSlice<u16>, GpuTextError> {
    Ok(stream.clone_htod(&host_bf16(dir, name, shape)?)?)
}

/// Whether checkpoint tensors `a` and `b` both have `shape`, the same dtype and byte-identical
/// contents.
pub(crate) fn tensors_identical(
    dir: &SafetensorsDir,
    a: &str,
    b: &str,
    shape: &[usize],
) -> Result<bool, GpuTextError> {
    let (ta, tb) = (dir.get(a)?, dir.get(b)?);
    Ok(ta.shape == shape
        && tb.shape == shape
        && ta.dtype == tb.dtype
        && dir.bytes(ta) == dir.bytes(tb))
}

/// Reads checkpoint tensor `name` (expected `shape`) as bf16 bits, converting F16 or F32
/// storage.
fn host_bf16(dir: &SafetensorsDir, name: &str, shape: &[usize]) -> Result<Vec<u16>, GpuTextError> {
    let t = dir.get(name)?;
    if t.shape != shape {
        return Err(ModelError::Shape {
            name: name.to_string(),
            expected: shape.to_vec(),
            found: t.shape.clone(),
        }
        .into());
    }
    let bytes = dir.bytes(t);
    let bits: Vec<u16> = match t.dtype {
        Dtype::Bf16 => bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|b| bf16::from_f32(f16::from_le_bytes([b[0], b[1]]).to_f32()).to_bits())
            .collect(),
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| bf16::from_f32(f32::from_le_bytes([b[0], b[1], b[2], b[3]])).to_bits())
            .collect(),
        dtype => {
            return Err(ModelError::Dtype {
                name: name.to_string(),
                dtype,
            }
            .into());
        }
    };
    Ok(bits)
}

/// Weights of one decoder layer (bf16, `[out, in]` row-major).
pub struct GpuLayerWeights {
    pub input_norm: CudaSlice<u16>,
    pub post_self_attn_norm: CudaSlice<u16>,
    pub post_attention_norm: CudaSlice<u16>,
    pub post_mlp_norm: CudaSlice<u16>,
    /// Query, key and value projections stacked along the output axis:
    /// `[q_dim + 2 * kv_dim, hidden]`.
    pub qkv: CudaSlice<u16>,
    pub o: CudaSlice<u16>,
    pub gate_up: CudaSlice<u16>,
    pub down: CudaSlice<u16>,
}

impl GpuLayerWeights {
    /// Uploads the layer whose tensors live under `prefix`
    /// (for example `model.language_model.layers.3`).
    pub fn load(
        stream: &Arc<CudaStream>,
        dir: &SafetensorsDir,
        prefix: &str,
        g: &TextGeometry,
    ) -> Result<Self, GpuTextError> {
        let (h, i) = (g.hidden, g.intermediate);
        let (qd, kd) = (g.q_dim(), g.kv_dim());
        let up = |suffix: &str, shape: &[usize]| {
            upload_bf16(stream, dir, &format!("{prefix}.{suffix}"), shape)
        };
        Ok(Self {
            input_norm: up("input_layernorm.weight", &[h])?,
            post_self_attn_norm: up("post_self_attn_layernorm.weight", &[h])?,
            post_attention_norm: up("post_attention_layernorm.weight", &[h])?,
            post_mlp_norm: up("post_mlp_layernorm.weight", &[h])?,
            qkv: {
                let part = |suffix: &str, rows: usize| {
                    host_bf16(dir, &format!("{prefix}.{suffix}"), &[rows, h])
                };
                let mut bits = part("self_attn.q_proj.weight", qd)?;
                bits.extend(part("self_attn.k_proj.weight", kd)?);
                bits.extend(part("self_attn.v_proj.weight", kd)?);
                stream.clone_htod(&bits)?
            },
            o: up("self_attn.o_proj.weight", &[h, qd])?,
            gate_up: up("mlp.gate_up_proj.weight", &[2 * i, h])?,
            down: up("mlp.down_proj.weight", &[h, i])?,
        })
    }
}

/// Embedding table, decoder layers, final norm and language-model head on the device.
pub struct GpuDecoderWeights {
    pub geometry: TextGeometry,
    pub embed_tokens: Arc<CudaSlice<u16>>,
    pub layers: Vec<GpuLayerWeights>,
    pub final_norm: CudaSlice<u16>,
    pub lm_head: Arc<CudaSlice<u16>>,
}

impl GpuDecoderWeights {
    /// Uploads the text model from `model.language_model.*` and `lm_head.weight`.
    pub fn load(
        stream: &Arc<CudaStream>,
        dir: &SafetensorsDir,
        spec: &TextSpec,
    ) -> Result<Self, GpuTextError> {
        if spec.tie_word_embeddings {
            return Err(GpuTextError::Unsupported("tied word embeddings".into()));
        }
        let geometry = TextGeometry::from_spec(spec)?;
        let (h, vocab) = (geometry.hidden, geometry.vocab);
        let layers = (0..geometry.layers)
            .map(|i| {
                GpuLayerWeights::load(
                    stream,
                    dir,
                    &format!("model.language_model.layers.{i}"),
                    &geometry,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            geometry,
            embed_tokens: Arc::new(upload_bf16(
                stream,
                dir,
                "model.language_model.embed_tokens.weight",
                &[vocab, h],
            )?),
            layers,
            final_norm: upload_bf16(stream, dir, "model.language_model.norm.weight", &[h])?,
            lm_head: Arc::new(upload_bf16(stream, dir, "lm_head.weight", &[vocab, h])?),
        })
    }
}

/// Key and value storage of one layer, `[blocks, block_size, kv_heads, head_dim]` bf16.
pub struct LayerCache {
    pub key: CudaSlice<u16>,
    pub value: CudaSlice<u16>,
}

/// Paged key/value storage for a stack of layers. A token at position `p` of a sequence whose
/// block table is `t` lives in slot `t[p / block_size] * block_size + p % block_size`.
pub struct GpuKvPool {
    layers: Vec<LayerCache>,
    block_size: usize,
    num_blocks: usize,
}

impl GpuKvPool {
    /// Allocates zeroed storage for `layers` layers of `num_blocks` blocks.
    pub fn new(
        stream: &Arc<CudaStream>,
        layers: usize,
        num_blocks: usize,
        block_size: usize,
        g: &TextGeometry,
    ) -> Result<Self, GpuTextError> {
        if block_size == 0 || num_blocks == 0 {
            return Err(GpuTextError::Input(
                "a key/value pool needs at least one block of at least one token".into(),
            ));
        }
        let elems = num_blocks * block_size * g.kv_dim();
        let layers = (0..layers)
            .map(|_| -> Result<LayerCache, GpuTextError> {
                Ok(LayerCache {
                    key: stream.alloc_zeros::<u16>(elems)?,
                    value: stream.alloc_zeros::<u16>(elems)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layers,
            block_size,
            num_blocks,
        })
    }

    /// Tokens per block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Blocks per layer.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Layers held.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// The cache of layer `index`, for inspecting stored keys and values.
    pub fn layer(&self, index: usize) -> Option<&LayerCache> {
        self.layers.get(index)
    }

    pub(crate) fn layers_mut(&mut self) -> &mut [LayerCache] {
        &mut self.layers
    }
}
