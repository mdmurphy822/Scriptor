//! SiLU-and-mul launchers (bf16 as raw `u16` bits, fp32 math).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, PushKernelArg};
use scriptor_cuda::Result;

use crate::launch::{elementwise_launch, row_launch};
use crate::norm::{load_module, to_config};

/// Gated-activation kernels.
pub struct ActivationKernels {
    fused: CudaFunction,
    split: CudaFunction,
}

impl ActivationKernels {
    /// Loads the kernels into `ctx`.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let m = load_module(ctx, "silu_mul")?;
        Ok(Self {
            fused: m.kernel("scriptor_silu_mul_fused_bf16")?,
            split: m.kernel("scriptor_silu_mul_bf16")?,
        })
    }

    /// Per row of `gate_up` (`[rows, 2 * width]`, gate then up): writes
    /// `silu(gate) * up` to `out` (`[rows, width]`). Works for any width, e.g.
    /// 4096 in the vision MLP and 4608 in the text and merger MLPs.
    pub fn silu_mul_fused(
        &self,
        stream: &Arc<CudaStream>,
        gate_up: &CudaSlice<u16>,
        out: &mut CudaSlice<u16>,
        rows: usize,
        width: usize,
    ) -> Result<()> {
        let cfg = to_config(row_launch(rows, width, 4, 1024));
        let width = width as i32;
        // SAFETY: argument order and types match the kernel signature; `gate_up`
        // covers `rows * 2 * width` and `out` `rows * width` elements.
        unsafe {
            stream
                .launch_builder(&self.fused)
                .arg(gate_up)
                .arg(out)
                .arg(&width)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Elementwise `silu(gate) * up` over `count` elements.
    pub fn silu_mul(
        &self,
        stream: &Arc<CudaStream>,
        gate: &CudaSlice<u16>,
        up: &CudaSlice<u16>,
        out: &mut CudaSlice<u16>,
        count: usize,
    ) -> Result<()> {
        let cfg = to_config(elementwise_launch(count, 256));
        let count = count as i64;
        // SAFETY: argument order and types match the kernel signature; every
        // buffer covers `count` elements.
        unsafe {
            stream
                .launch_builder(&self.split)
                .arg(gate)
                .arg(up)
                .arg(out)
                .arg(&count)
                .launch(cfg)?;
        }
        Ok(())
    }
}
