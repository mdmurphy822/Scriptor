//! Embedding gather and image-embedding scatter launchers (bf16 as raw `u16` bits).

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use scriptor_cuda::Result;

use crate::norm::load_module;

/// Row gather and scatter kernels.
pub struct ScatterKernels {
    gather: CudaFunction,
    scatter: CudaFunction,
}

impl ScatterKernels {
    /// Loads the kernels into `ctx`.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let m = load_module(ctx, "gather_scatter")?;
        Ok(Self {
            gather: m.kernel("scriptor_embedding_gather_bf16")?,
            scatter: m.kernel("scriptor_image_scatter_bf16")?,
        })
    }

    /// `out[t] = table[ids[t]]` for `tokens` ids; ids outside `0..vocab` give
    /// zero rows.
    #[allow(clippy::too_many_arguments)]
    pub fn embedding_gather(
        &self,
        stream: &Arc<CudaStream>,
        table: &CudaSlice<u16>,
        ids: &CudaSlice<i32>,
        out: &mut CudaSlice<u16>,
        tokens: usize,
        dim: usize,
        vocab: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (tokens as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (dim, vocab) = (dim as i32, vocab as i32);
        // SAFETY: argument order and types match the kernel signature; `ids`
        // holds `tokens` entries and `out` `tokens * dim` elements.
        unsafe {
            stream
                .launch_builder(&self.gather)
                .arg(table)
                .arg(ids)
                .arg(out)
                .arg(&dim)
                .arg(&vocab)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// `embeds[positions[i]] = image_embeds[i]` for `count` rows; negative
    /// positions are skipped.
    pub fn image_scatter(
        &self,
        stream: &Arc<CudaStream>,
        embeds: &mut CudaSlice<u16>,
        image_embeds: &CudaSlice<u16>,
        positions: &CudaSlice<i32>,
        count: usize,
        dim: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (count as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let dim = dim as i32;
        // SAFETY: argument order and types match the kernel signature; every
        // position indexes a row inside `embeds` by the caller's contract.
        unsafe {
            stream
                .launch_builder(&self.scatter)
                .arg(embeds)
                .arg(image_embeds)
                .arg(positions)
                .arg(&dim)
                .launch(cfg)?;
        }
        Ok(())
    }
}
