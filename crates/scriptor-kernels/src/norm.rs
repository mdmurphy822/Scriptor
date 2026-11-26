//! RMSNorm, residual-add RMSNorm and LayerNorm+GELU launchers (bf16 as raw
//! `u16` bits, fp32 math).

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use scriptor_cuda::{CudaError, KernelModule, Result};

use crate::launch::{LaunchDims, row_launch};

/// Loads the embedded module compiled from `kernels/<stem>.cu`.
pub(crate) fn load_module(ctx: &Arc<CudaContext>, stem: &str) -> Result<KernelModule> {
    let image = crate::fatbins::FATBINS
        .iter()
        .find(|(name, _)| *name == stem)
        .map(|(_, bytes)| *bytes)
        .ok_or_else(|| CudaError::Module(format!("kernel module {stem} is not embedded")))?;
    KernelModule::load(ctx, image)
}

/// Converts launch extents to the driver's launch configuration.
pub(crate) fn to_config(d: LaunchDims) -> LaunchConfig {
    LaunchConfig {
        grid_dim: d.grid,
        block_dim: d.block,
        shared_mem_bytes: d.shared_bytes,
    }
}

/// Row-wise normalisation kernels over `[rows, n]` bf16 activations.
pub struct NormKernels {
    rms: CudaFunction,
    add_rms: CudaFunction,
    layer_gelu: CudaFunction,
}

impl NormKernels {
    /// Loads the kernels into `ctx`.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let norm = load_module(ctx, "rmsnorm")?;
        let ln = load_module(ctx, "layernorm_gelu")?;
        Ok(Self {
            rms: norm.kernel("scriptor_rmsnorm_bf16")?,
            add_rms: norm.kernel("scriptor_add_rmsnorm_bf16")?,
            layer_gelu: ln.kernel("scriptor_layernorm_gelu_bf16")?,
        })
    }

    /// `out = x / sqrt(mean(x^2) + eps) * w` per row.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<u16>,
        w: &CudaSlice<u16>,
        out: &mut CudaSlice<u16>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        let cfg = to_config(row_launch(rows, n, 4, 1024));
        let n = n as i32;
        // SAFETY: argument order and types match the kernel signature; buffers
        // cover `rows * n` elements (`w`: `n`) by the caller's contract.
        unsafe {
            stream
                .launch_builder(&self.rms)
                .arg(x)
                .arg(w)
                .arg(out)
                .arg(&n)
                .arg(&eps)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// `sum_out = bf16(x + residual)` and `out = rms_norm(sum_out) * w` per row.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rms_norm(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<u16>,
        residual: &CudaSlice<u16>,
        w: &CudaSlice<u16>,
        sum_out: &mut CudaSlice<u16>,
        out: &mut CudaSlice<u16>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        let cfg = to_config(row_launch(rows, n, 4, 1024));
        let n = n as i32;
        // SAFETY: argument order and types match the kernel signature; buffers
        // cover `rows * n` elements (`w`: `n`) by the caller's contract.
        unsafe {
            stream
                .launch_builder(&self.add_rms)
                .arg(x)
                .arg(residual)
                .arg(w)
                .arg(sum_out)
                .arg(out)
                .arg(&n)
                .arg(&eps)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// LayerNorm with `w`/`b` per row followed by erf-GELU.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm_gelu(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<u16>,
        w: &CudaSlice<u16>,
        b: &CudaSlice<u16>,
        out: &mut CudaSlice<u16>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        let cfg = to_config(row_launch(rows, n, 4, 1024));
        let n = n as i32;
        // SAFETY: argument order and types match the kernel signature; buffers
        // cover `rows * n` elements (`w`, `b`: `n`) by the caller's contract.
        unsafe {
            stream
                .launch_builder(&self.layer_gelu)
                .arg(x)
                .arg(w)
                .arg(b)
                .arg(out)
                .arg(&n)
                .arg(&eps)
                .launch(cfg)?;
        }
        Ok(())
    }
}

/// Launch extents of [`MtpPrologue::launch`]: one block per row and half, with the block width
/// of the [`NormKernels::rms_norm`] launch over `n` elements.
pub fn mtp_prologue_launch(rows: usize, n: usize) -> LaunchDims {
    let row = row_launch(rows, n, 4, 1024);
    LaunchDims {
        grid: (rows as u32, 2, 1),
        ..row
    }
}

/// The fused MTP prologue: embedding gather, both RMSNorms and the concatenation into the
/// `[rows, 2 * n]` input of the `eh_proj` product, written by one kernel.
pub struct MtpPrologue {
    kernel: CudaFunction,
}

impl MtpPrologue {
    /// Loads the kernel into `ctx`.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = load_module(ctx, "mtp_prologue")?;
        Ok(Self {
            kernel: module.kernel("scriptor_mtp_prologue_bf16")?,
        })
    }

    /// `cat[r] = [rms_norm(table[ids[r]]) * enorm, rms_norm(hidden[r]) * hnorm]` for `rows`
    /// rows of width `n`; ids outside `0..vocab` embed as zero rows.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        stream: &Arc<CudaStream>,
        table: &CudaSlice<u16>,
        ids: &CudaSlice<i32>,
        hidden: &CudaSlice<u16>,
        enorm: &CudaSlice<u16>,
        hnorm: &CudaSlice<u16>,
        cat: &mut CudaSlice<u16>,
        rows: usize,
        n: usize,
        vocab: usize,
        eps: f32,
    ) -> Result<()> {
        let cfg = to_config(mtp_prologue_launch(rows, n));
        let (n, vocab) = (n as i32, vocab as i32);
        // SAFETY: argument order and types match the kernel signature; `ids` holds `rows`
        // entries, `hidden` `rows * n`, the norm weights `n` and `cat` `rows * 2 * n` elements
        // by the caller's contract.
        unsafe {
            stream
                .launch_builder(&self.kernel)
                .arg(table)
                .arg(ids)
                .arg(hidden)
                .arg(enorm)
                .arg(hnorm)
                .arg(cat)
                .arg(&n)
                .arg(&vocab)
                .arg(&eps)
                .launch(cfg)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prologue_block_matches_rms_norm_launch_for_any_width() {
        for n in [1usize, 31, 32, 100, 1000, 1536, 2048, 4096, 8192, 12345] {
            for rows in [1usize, 8, 128] {
                let norm = row_launch(rows, n, 4, 1024);
                let fused = mtp_prologue_launch(rows, n);
                assert_eq!(fused.block, norm.block, "n={n}");
                assert_eq!(fused.shared_bytes, norm.shared_bytes);
                assert_eq!(fused.grid, (rows as u32, 2, 1));
            }
        }
    }

    #[test]
    fn prologue_block_for_the_target_hidden_width() {
        assert_eq!(mtp_prologue_launch(64, 1536).block, (512, 1, 1));
    }
}
