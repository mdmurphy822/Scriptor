//! Vision transformer: patch embedding, pre-norm blocks with per-head q/k RMSNorm and 2-D
//! axial RoPE, then the post layer norm.

use scriptor_kernels::reference::activation::silu_mul;
use scriptor_kernels::reference::attention::varlen_attention;
use scriptor_kernels::reference::conv::PATCH_ELEMS;
use scriptor_kernels::reference::norm::{head_rms_norm, rms_norm};
use scriptor_kernels::reference::rope::{rope_vision_2d, vision_positions};
use scriptor_spec::VisionSpec;

use super::{Linear, ModelError, TensorSource, add_in_place};

/// One vision block.
#[derive(Debug, Clone)]
pub struct VisionBlock {
    pub norm1: Vec<f32>,
    pub norm2: Vec<f32>,
    pub qkv: Linear,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub proj: Linear,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

/// The vision tower from flattened patches to post-layer-norm hidden states.
#[derive(Debug, Clone)]
pub struct VisionTower {
    pub patch_embed: Linear,
    pub blocks: Vec<VisionBlock>,
    pub post_layernorm: Vec<f32>,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub rope_theta: f64,
}

impl VisionTower {
    /// Loads the tower from checkpoint tensors named `model.visual.*`.
    pub fn load(src: &TensorSource, spec: &VisionSpec) -> Result<Self, ModelError> {
        if spec.patch_dim() != PATCH_ELEMS {
            return Err(ModelError::Unsupported(format!(
                "patch dimension {} differs from {PATCH_ELEMS}",
                spec.patch_dim()
            )));
        }
        if !spec.qk_norm || !spec.is_neox_style {
            return Err(ModelError::Unsupported(
                "vision tower requires q/k norm and NeoX-style rotary embedding".into(),
            ));
        }
        let (h, i) = (spec.hidden_size, spec.intermediate_size);
        let mut blocks = Vec::with_capacity(spec.depth);
        for b in 0..spec.depth {
            let p = format!("model.visual.blocks.{b}");
            blocks.push(VisionBlock {
                norm1: src.tensor(&format!("{p}.norm1.weight"), &[h])?,
                norm2: src.tensor(&format!("{p}.norm2.weight"), &[h])?,
                qkv: src.linear(
                    &format!("{p}.attn.qkv.weight"),
                    Some(&format!("{p}.attn.qkv.bias")),
                    3 * h,
                    h,
                )?,
                q_norm: src.tensor(&format!("{p}.attn.q_norm.weight"), &[spec.head_dim])?,
                k_norm: src.tensor(&format!("{p}.attn.k_norm.weight"), &[spec.head_dim])?,
                proj: src.linear(
                    &format!("{p}.attn.proj.weight"),
                    Some(&format!("{p}.attn.proj.bias")),
                    h,
                    h,
                )?,
                gate: src.linear(
                    &format!("{p}.mlp.gate_proj.weight"),
                    Some(&format!("{p}.mlp.gate_proj.bias")),
                    i,
                    h,
                )?,
                up: src.linear(
                    &format!("{p}.mlp.up_proj.weight"),
                    Some(&format!("{p}.mlp.up_proj.bias")),
                    i,
                    h,
                )?,
                down: src.linear(
                    &format!("{p}.mlp.down_proj.weight"),
                    Some(&format!("{p}.mlp.down_proj.bias")),
                    h,
                    i,
                )?,
            });
        }
        Ok(Self {
            patch_embed: Linear {
                weight: src.tensor(
                    "model.visual.patch_embed.proj.weight",
                    &[
                        h,
                        spec.in_channels,
                        spec.temporal_patch_size,
                        spec.patch_size,
                        spec.patch_size,
                    ],
                )?,
                bias: Some(src.tensor("model.visual.patch_embed.proj.bias", &[h])?),
                out_dim: h,
                in_dim: PATCH_ELEMS,
            },
            blocks,
            post_layernorm: src.tensor("model.visual.post_layernorm.weight", &[h])?,
            hidden: h,
            heads: spec.num_heads,
            head_dim: spec.head_dim,
            eps: spec.rms_norm_eps as f32,
            rope_theta: spec.rope_theta,
        })
    }

    /// Runs the tower over `patches` (`[N, 1176]`, merge-window order) belonging to images with
    /// `(t, h, w)` patch grids; attention stays inside each frame. Returns `[N, hidden]`
    /// after the post layer norm.
    pub fn forward(
        &self,
        patches: &[f32],
        grids: &[(usize, usize, usize)],
    ) -> Result<Vec<f32>, ModelError> {
        let total: usize = grids.iter().map(|&(t, h, w)| t * h * w).sum();
        if patches.len() != total * PATCH_ELEMS {
            return Err(ModelError::Input(format!(
                "{} patch values for grids totalling {total} patches",
                patches.len()
            )));
        }
        let mut positions = Vec::with_capacity(total);
        let mut cu_seqlens = vec![0usize];
        for &(t, h, w) in grids {
            if h % 2 != 0 || w % 2 != 0 {
                return Err(ModelError::Input(format!(
                    "grid {t}x{h}x{w} is not divisible by the merge size"
                )));
            }
            positions.extend(vision_positions(t, h, w));
            for _ in 0..t {
                cu_seqlens.push(cu_seqlens.last().copied().unwrap_or(0) + h * w);
            }
        }
        let (n, hd, heads) = (total, self.head_dim, self.heads);
        let mut x = self.patch_embed.apply(patches, n);
        for block in &self.blocks {
            let normed = rms_norm(&x, &block.norm1, self.hidden, self.eps);
            let qkv = block.qkv.apply(&normed, n);
            let mut q = Vec::with_capacity(n * self.hidden);
            let mut k = Vec::with_capacity(n * self.hidden);
            let mut v = Vec::with_capacity(n * self.hidden);
            for row in qkv.chunks_exact(3 * self.hidden) {
                q.extend_from_slice(&row[..self.hidden]);
                k.extend_from_slice(&row[self.hidden..2 * self.hidden]);
                v.extend_from_slice(&row[2 * self.hidden..]);
            }
            let mut q = head_rms_norm(&q, &block.q_norm, self.eps);
            let mut k = head_rms_norm(&k, &block.k_norm, self.eps);
            rope_vision_2d(&mut q, &positions, heads, hd, self.rope_theta);
            rope_vision_2d(&mut k, &positions, heads, hd, self.rope_theta);
            let attn = varlen_attention(
                &q,
                &k,
                &v,
                &cu_seqlens,
                heads,
                heads,
                hd,
                false,
                (hd as f32).powf(-0.5),
            );
            add_in_place(&mut x, &block.proj.apply(&attn, n));
            let normed = rms_norm(&x, &block.norm2, self.hidden, self.eps);
            let act = silu_mul(&block.gate.apply(&normed, n), &block.up.apply(&normed, n));
            add_in_place(&mut x, &block.down.apply(&act, n));
        }
        Ok(rms_norm(&x, &self.post_layernorm, self.hidden, self.eps))
    }
}
