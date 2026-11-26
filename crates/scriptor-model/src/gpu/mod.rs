//! GPU executors for the model graph.
//!
//! Activations and weights are BF16 held as raw `u16` bits; every executor runs on a
//! caller-supplied stream and never synchronises it.

pub mod decoder;
pub mod forward;
pub mod fused_forward;
pub mod logits;
pub mod merger;
pub mod mtp;
pub mod params;
pub mod resident;
pub mod vision;
pub mod workspace;

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use scriptor_cuda::Result;
use scriptor_kernels::cublaslt::{GemmEngine, GemmOperands, GemmProblem};
use scriptor_kernels::heuristics::{Epilogue, GemmDtype};

pub use decoder::{
    GpuDecoderWeights, GpuKvPool, GpuLayerWeights, GpuTextError, LayerCache, TextGeometry,
};
pub use forward::{DecodeSeq, PrefillSeq, StepInput, StepOutput, TextExecutor};
pub use fused_forward::{Fusion, StageProfiler, StageStat};
pub use logits::RowOps;
pub use merger::GpuMerger;
pub use mtp::{GpuMtpWeights, MtpOutput};
pub use params::{GpuLinear, GpuVisionBlock, GpuVisionParams};
pub use resident::StepPlan;
pub use vision::{GpuVision, VisionBatchStats, VisionStageStat};
pub use workspace::VisionWorkspace;

/// Timed launches per candidate in the boot-time autotune.
pub(crate) const TUNE_ITERATIONS: usize = 7;

/// Outcome of a boot-time autotune pass.
#[derive(Debug, Default)]
pub struct TuneReport {
    /// cuBLASLt candidate choices per tuning key.
    pub gemm: Vec<scriptor_kernels::cublaslt::BootTuned>,
}

/// Row buckets (powers of two) from the bucket holding `lowest` up to the one holding `highest`.
pub(crate) fn tune_buckets(lowest: usize, highest: usize) -> Vec<usize> {
    let top = scriptor_kernels::heuristics::m_bucket(highest.max(lowest));
    let mut buckets = Vec::new();
    let mut m = scriptor_kernels::heuristics::m_bucket(lowest);
    while m <= top {
        buckets.push(m);
        m *= 2;
    }
    buckets
}

/// Converts f32 values to BF16 bits with round-to-nearest-even.
pub fn bf16_bits(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_bits())
        .collect()
}

/// Converts BF16 bits to f32.
pub fn bf16_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::bf16::from_bits(b).to_f32())
        .collect()
}

/// Device address of `slice` on `stream`.
pub(crate) fn dptr(stream: &Arc<CudaStream>, slice: &CudaSlice<u16>) -> u64 {
    slice.device_ptr(stream).0
}

/// Runs `out[rows, out_dim] = x[rows, in_dim] * W^T (+ bias)` through cuBLASLt.
pub(crate) fn linear(
    gemm: &mut GemmEngine,
    layer: &GpuLinear,
    x: u64,
    out: u64,
    rows: usize,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    linear_with(gemm, layer, x, out, rows, stream, true)
}

/// As [`linear`]; with `bias` false the layer's bias is left out of the GEMM.
pub(crate) fn linear_with(
    gemm: &mut GemmEngine,
    layer: &GpuLinear,
    x: u64,
    out: u64,
    rows: usize,
    stream: &Arc<CudaStream>,
    bias: bool,
) -> Result<()> {
    let epilogue = if bias && layer.bias.is_some() {
        Epilogue::Bias
    } else {
        Epilogue::None
    };
    let problem = GemmProblem {
        m: rows,
        n: layer.out_dim,
        k: layer.in_dim,
        dtype: GemmDtype::Bf16,
        epilogue,
    };
    let ops = GemmOperands {
        x,
        w: dptr(stream, &layer.weight),
        bias: layer
            .bias
            .as_ref()
            .filter(|_| bias)
            .map_or(0, |b| dptr(stream, b)),
        out,
        ..GemmOperands::default()
    };
    gemm.gemm(&problem, &ops)
}
