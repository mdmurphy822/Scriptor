//! M-RoPE and vision 2-D RoPE launchers (bf16 as raw `u16` bits).

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use scriptor_cuda::Result;

use crate::norm::load_module;

/// Width of one vision attention head.
pub const VISION_HEAD_DIM: usize = 64;

/// Rotary embedding kernels.
pub struct RopeKernels {
    mrope: CudaFunction,
    vision: CudaFunction,
    vision_k: CudaFunction,
}

impl RopeKernels {
    /// Loads the kernels into `ctx`.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let text = load_module(ctx, "rope_mrope")?;
        let vision = load_module(ctx, "rope_vision2d")?;
        Ok(Self {
            mrope: text.kernel("scriptor_mrope_gptj_bf16")?,
            vision: vision.kernel("scriptor_vision_qk_norm_rope_bf16")?,
            vision_k: vision.kernel("scriptor_vision_k_norm_rope_bf16")?,
        })
    }

    /// GPT-J interleaved M-RoPE applied in place to `q` (`[tokens, q_heads,
    /// head_dim]`) and `k` (`[tokens, k_heads, head_dim]`). `positions` is a
    /// device buffer of `[t, h, w]` triples per token; `sections` are the pair
    /// counts for t, h and w and sum to `head_dim / 2`.
    #[allow(clippy::too_many_arguments)]
    pub fn mrope(
        &self,
        stream: &Arc<CudaStream>,
        q: &mut CudaSlice<u16>,
        k: &mut CudaSlice<u16>,
        positions: &CudaSlice<i32>,
        tokens: usize,
        q_heads: usize,
        k_heads: usize,
        head_dim: usize,
        theta: f32,
        sections: [usize; 3],
    ) -> Result<()> {
        assert_eq!(sections.iter().sum::<usize>() * 2, head_dim);
        let cfg = LaunchConfig {
            grid_dim: (tokens as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (qh, kh, hd, s0, s1) = (
            q_heads as i32,
            k_heads as i32,
            head_dim as i32,
            sections[0] as i32,
            sections[1] as i32,
        );
        // SAFETY: argument order and types match the kernel signature; `q`/`k`
        // cover `tokens` rows and `positions` `3 * tokens` entries.
        unsafe {
            stream
                .launch_builder(&self.mrope)
                .arg(q)
                .arg(k)
                .arg(positions)
                .arg(&qh)
                .arg(&kh)
                .arg(&hd)
                .arg(&theta)
                .arg(&s0)
                .arg(&s1)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Per-head RMSNorm (`q_w`, `k_w` of width 64) followed by 2-D axial RoPE on
    /// the q and k sections of `qkv` (`[tokens, 3, heads, 64]`), in place.
    /// `positions` is a device buffer of `[row, col]` pairs per token.
    #[allow(clippy::too_many_arguments)]
    pub fn vision_qk_norm_rope(
        &self,
        stream: &Arc<CudaStream>,
        qkv: &mut CudaSlice<u16>,
        q_w: &CudaSlice<u16>,
        k_w: &CudaSlice<u16>,
        positions: &CudaSlice<i32>,
        tokens: usize,
        heads: usize,
        theta: f32,
        eps: f32,
    ) -> Result<()> {
        let warps = (2 * heads).clamp(1, 16) as u32;
        let cfg = LaunchConfig {
            grid_dim: (tokens as u32, 1, 1),
            block_dim: (32 * warps, 1, 1),
            shared_mem_bytes: 0,
        };
        let heads = heads as i32;
        // SAFETY: argument order and types match the kernel signature; `qkv`
        // covers `tokens * 3 * heads * 64` elements and `positions` `2 * tokens`.
        unsafe {
            stream
                .launch_builder(&self.vision)
                .arg(qkv)
                .arg(q_w)
                .arg(k_w)
                .arg(positions)
                .arg(&heads)
                .arg(&theta)
                .arg(&eps)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// As [`Self::vision_qk_norm_rope`] over the k heads only; the q section is left raw.
    #[allow(clippy::too_many_arguments)]
    pub fn vision_k_norm_rope(
        &self,
        stream: &Arc<CudaStream>,
        qkv: &mut CudaSlice<u16>,
        q_w: &CudaSlice<u16>,
        k_w: &CudaSlice<u16>,
        positions: &CudaSlice<i32>,
        tokens: usize,
        heads: usize,
        theta: f32,
        eps: f32,
    ) -> Result<()> {
        let warps = heads.clamp(1, 16) as u32;
        let cfg = LaunchConfig {
            grid_dim: (tokens as u32, 1, 1),
            block_dim: (32 * warps, 1, 1),
            shared_mem_bytes: 0,
        };
        let heads = heads as i32;
        // SAFETY: argument order and types match the kernel signature; `qkv`
        // covers `tokens * 3 * heads * 64` elements and `positions` `2 * tokens`.
        unsafe {
            stream
                .launch_builder(&self.vision_k)
                .arg(qkv)
                .arg(q_w)
                .arg(k_w)
                .arg(positions)
                .arg(&heads)
                .arg(&theta)
                .arg(&eps)
                .launch(cfg)?;
        }
        Ok(())
    }
}
