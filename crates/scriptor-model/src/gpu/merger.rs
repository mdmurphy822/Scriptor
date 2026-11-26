//! Device vision adapter: downsample as a GEMM over merge groups, then the patch merger.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use scriptor_cuda::Result;
use scriptor_kernels::activation::ActivationKernels;
use scriptor_kernels::cublaslt::GemmEngine;
use scriptor_kernels::norm::NormKernels;

use super::params::GpuLinear;
use super::vision::{Marker, gemm_cost};
use super::workspace::{MERGE_GROUP, VisionWorkspace};
use super::{bf16_bits, dptr, linear};
use crate::reference::Merger;

/// Epsilon of the merger's post-projection layer norm.
const MERGER_NORM_EPS: f32 = 1e-5;

/// Adapter weights on the device.
pub struct GpuMerger {
    /// Downsample convolution as a `[out_dim, 4 * in_dim]` weight whose columns follow the
    /// merge-group row order (`slot * in_dim + channel`), with bias.
    pub downsample: GpuLinear,
    pub proj: GpuLinear,
    pub norm_weight: CudaSlice<u16>,
    pub norm_bias: CudaSlice<u16>,
    pub gate_up: GpuLinear,
    pub down: GpuLinear,
    pub in_dim: usize,
    pub out_dim: usize,
    pub intermediate: usize,
}

impl GpuMerger {
    /// Uploads the reference adapter as BF16, permuting the downsample kernel so that four
    /// consecutive tower rows form one GEMM row without a gather.
    pub fn from_reference(stream: &Arc<CudaStream>, m: &Merger) -> Result<Self> {
        let (inp, out) = (m.in_dim, m.out_dim);
        let mut ds = vec![0.0f32; out * MERGE_GROUP * inp];
        for o in 0..out {
            for c in 0..inp {
                for slot in 0..MERGE_GROUP {
                    ds[o * MERGE_GROUP * inp + slot * inp + c] =
                        m.downsample_weight[(o * inp + c) * MERGE_GROUP + slot];
                }
            }
        }
        Ok(Self {
            downsample: GpuLinear::from_parts(
                stream,
                &ds,
                Some(&m.downsample_bias),
                out,
                MERGE_GROUP * inp,
            )?,
            proj: GpuLinear::from_reference(stream, &m.proj)?,
            norm_weight: stream.clone_htod(&bf16_bits(&m.norm_weight))?,
            norm_bias: stream.clone_htod(&bf16_bits(&m.norm_bias))?,
            gate_up: GpuLinear::stacked(stream, &[&m.gate, &m.up])?,
            down: GpuLinear::from_reference(stream, &m.down)?,
            in_dim: inp,
            out_dim: out,
            intermediate: m.gate.out_dim,
        })
    }

    /// Maps the post-layer-norm tower states in `ws.normed` (`[patches, in_dim]`, merge-window
    /// order) to `[patches / 4, out_dim]` embeddings in `ws.embeddings`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &self,
        stream: &Arc<CudaStream>,
        gemm: &mut GemmEngine,
        norm: &NormKernels,
        act: &ActivationKernels,
        ws: &mut VisionWorkspace,
        patches: usize,
        mk: &mut Marker,
    ) -> Result<()> {
        let groups = patches / MERGE_GROUP;
        linear(
            gemm,
            &self.downsample,
            dptr(stream, &ws.normed),
            dptr(stream, &ws.merged),
            groups,
            stream,
        )?;
        let (f, by) = gemm_cost(groups, self.out_dim, MERGE_GROUP * self.in_dim);
        mk.mark("merger_downsample", f, by)?;
        linear(
            gemm,
            &self.proj,
            dptr(stream, &ws.merged),
            dptr(stream, &ws.projected),
            groups,
            stream,
        )?;
        let (f, by) = gemm_cost(groups, self.out_dim, self.out_dim);
        mk.mark("merger_proj", f, by)?;
        norm.layer_norm_gelu(
            stream,
            &ws.projected,
            &self.norm_weight,
            &self.norm_bias,
            &mut ws.activated,
            groups,
            self.out_dim,
            MERGER_NORM_EPS,
        )?;
        mk.mark(
            "merger_norm_gelu",
            0,
            2 * 2 * (groups * self.out_dim) as u64,
        )?;
        linear(
            gemm,
            &self.gate_up,
            dptr(stream, &ws.activated),
            dptr(stream, &ws.merger_gate_up),
            groups,
            stream,
        )?;
        let (f, by) = gemm_cost(groups, 2 * self.intermediate, self.out_dim);
        mk.mark("merger_gate_up", f, by)?;
        act.silu_mul_fused(
            stream,
            &ws.merger_gate_up,
            &mut ws.merger_act,
            groups,
            self.intermediate,
        )?;
        mk.mark(
            "merger_silu_mul",
            0,
            2 * 3 * (groups * self.intermediate) as u64,
        )?;
        linear(
            gemm,
            &self.down,
            dptr(stream, &ws.merger_act),
            dptr(stream, &ws.embeddings),
            groups,
            stream,
        )?;
        let (f, by) = gemm_cost(groups, self.out_dim, self.intermediate);
        mk.mark("merger_down", f, by)
    }
}
