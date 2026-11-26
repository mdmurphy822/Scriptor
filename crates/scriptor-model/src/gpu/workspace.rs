//! Scratch buffers of the vision executor, sized from a patch budget.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use scriptor_cuda::Result;

use super::params::GpuVisionParams;

/// Patches merged into one output token.
pub const MERGE_GROUP: usize = 4;

/// Device scratch for one vision batch of at most `patch_budget` patches.
pub struct VisionWorkspace {
    /// Largest total patch count a batch may carry.
    pub patch_budget: usize,
    pub(crate) x_a: CudaSlice<u16>,
    pub(crate) x_b: CudaSlice<u16>,
    pub(crate) branch: CudaSlice<u16>,
    pub(crate) normed: CudaSlice<u16>,
    pub(crate) qkv: CudaSlice<u16>,
    pub(crate) q: CudaSlice<u16>,
    pub(crate) k: CudaSlice<u16>,
    pub(crate) v: CudaSlice<u16>,
    pub(crate) attn: CudaSlice<u16>,
    pub(crate) gate_up: CudaSlice<u16>,
    pub(crate) act: CudaSlice<u16>,
    pub(crate) merged: CudaSlice<u16>,
    pub(crate) projected: CudaSlice<u16>,
    pub(crate) activated: CudaSlice<u16>,
    pub(crate) merger_gate_up: CudaSlice<u16>,
    pub(crate) merger_act: CudaSlice<u16>,
    pub(crate) embeddings: CudaSlice<u16>,
    pub(crate) positions: CudaSlice<i32>,
    pub(crate) cu_seqlens: CudaSlice<i32>,
}

impl VisionWorkspace {
    /// Allocates scratch for batches of up to `patch_budget` patches shaped like `params`.
    pub fn new(
        stream: &Arc<CudaStream>,
        params: &GpuVisionParams,
        patch_budget: usize,
    ) -> Result<Self> {
        let p = patch_budget.max(MERGE_GROUP);
        let tokens = p / MERGE_GROUP;
        let (h, i) = (params.hidden, params.intermediate);
        let (out, mid) = (params.merger.out_dim, params.merger.intermediate);
        let z = |n: usize| stream.alloc_zeros::<u16>(n);
        Ok(Self {
            patch_budget: p,
            x_a: z(p * h)?,
            x_b: z(p * h)?,
            branch: z(p * h)?,
            normed: z(p * h)?,
            qkv: z(p * 3 * h)?,
            q: z(p * h)?,
            k: z(p * h)?,
            v: z(p * h)?,
            attn: z(p * h)?,
            gate_up: z(p * 2 * i)?,
            act: z(p * i)?,
            merged: z(tokens * out)?,
            projected: z(tokens * out)?,
            activated: z(tokens * out)?,
            merger_gate_up: z(tokens * 2 * mid)?,
            merger_act: z(tokens * mid)?,
            embeddings: z(tokens * out)?,
            positions: stream.alloc_zeros::<i32>(p * 2)?,
            cu_seqlens: stream.alloc_zeros::<i32>(tokens + 1)?,
        })
    }

    /// Frames a batch may carry: every frame holds at least one merge window.
    pub fn frame_budget(&self) -> usize {
        self.patch_budget / MERGE_GROUP
    }
}
