//! Vision tower kernels: attention straight over the packed `qkv` buffer and the
//! bias + residual + RMSNorm join (bf16 as raw `u16` bits, fp32 math).

use std::sync::Arc;

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use scriptor_cuda::{CudaError, Result};

use crate::launch::ceil_div;
use crate::norm::load_module;
use crate::rope::VISION_HEAD_DIM;

/// Query rows per thread block of the vision attention kernel.
pub const VISION_ATTN_BLOCK_Q: usize = 128;
/// Largest row width the join kernel accepts.
pub const JOIN_MAX_WIDTH: usize = 4096;
/// Threads per block of the join kernel.
const JOIN_THREADS: u32 = 128;
/// Output (activation) columns per block of the SiLU-fused gate/up projection.
const GATE_UP_BLOCK_COLS: usize = 128;
/// Rows per block of the SiLU-fused gate/up projection.
const GATE_UP_BLOCK_M: usize = 128;
/// Reduction step of the SiLU-fused gate/up projection.
const GATE_UP_BLOCK_K: usize = 32;
/// Dynamic shared memory of the SiLU-fused gate/up projection (four stages of activation and weight tiles).
const GATE_UP_SMEM: u32 = 4 * 24576;
/// Output (activation) columns per block of the two-blocks-per-multiprocessor gate/up variant.
const GATE_UP_SMALL_COLS: usize = 64;
/// Dynamic shared memory of the small gate/up variant (three stages of activation and weight tiles).
const GATE_UP_SMALL_SMEM: u32 = 3 * 16384;

/// Execution path of the gate/up projection followed by SiLU-and-multiply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GateUpPath {
    /// Vendor GEMM followed by the separate `silu_mul` pass.
    Vendor,
    /// Fused kernel with a 128 x 128-output tile, one block per multiprocessor.
    Fused,
    /// Fused kernel with a 128 x 64-output tile, two blocks per multiprocessor.
    FusedSmall,
}

impl GateUpPath {
    /// Short name for logs.
    pub fn name(self) -> &'static str {
        match self {
            Self::Vendor => "vendor",
            Self::Fused => "fused",
            Self::FusedSmall => "fused-small",
        }
    }
}

/// Fewest rows at which the SiLU-fused gate/up projection replaces the vendor pair.
const GATE_UP_FUSED_MIN_ROWS: usize = 256;

/// Vision attention and join launchers.
pub struct VisionKernels {
    attn: CudaFunction,
    join: CudaFunction,
    gate_up: CudaFunction,
    gate_up_small: CudaFunction,
}

impl VisionKernels {
    /// Loads the kernels into `ctx`.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let attn = load_module(ctx, "attn_vision")?;
        let join = load_module(ctx, "vision_join")?;
        let gate_up_module = load_module(ctx, "vision_gate_up_gemm")?;
        let gate_up = gate_up_module.kernel("scriptor_vision_gate_up_silu_bf16")?;
        gate_up.set_attribute(
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            GATE_UP_SMEM as i32,
        )?;
        let gate_up_small = gate_up_module.kernel("scriptor_vision_gate_up_silu_bf16_small")?;
        gate_up_small.set_attribute(
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            GATE_UP_SMALL_SMEM as i32,
        )?;
        Ok(Self {
            gate_up,
            gate_up_small,
            attn: attn.kernel("scriptor_attn_vision_d64")?,
            join: join.kernel("scriptor_vision_bias_add_rmsnorm_bf16")?,
        })
    }

    /// Whether [`Self::bias_add_rms_norm`] accepts rows of `n` elements.
    pub fn join_supports(n: usize) -> bool {
        n > 0 && n.is_multiple_of(8) && n <= JOIN_MAX_WIDTH
    }

    /// Whether [`Self::gate_up_silu`] accepts an activation width `inter` and reduction width `k`.
    pub fn gate_up_supports(inter: usize, k: usize) -> bool {
        inter > 0
            && k > 0
            && inter.is_multiple_of(GATE_UP_BLOCK_COLS)
            && k.is_multiple_of(GATE_UP_BLOCK_K)
    }

    /// Whether [`Self::gate_up_silu`] replaces the vendor GEMM plus `silu_mul` for `rows` rows
    /// (see [`Self::gate_up_path`]).
    pub fn gate_up_profitable(&self, rows: usize, inter: usize, k: usize) -> bool {
        Self::gate_up_path(rows, inter, k) != GateUpPath::Vendor
    }

    /// The path for `rows` rows, a pure function of the shape: the small-tile fused kernel when
    /// the shape is supported and `rows` is at least [`GATE_UP_FUSED_MIN_ROWS`], the vendor pair
    /// otherwise. The large-tile kernel produces bit-identical output to the small tile.
    pub fn gate_up_path(rows: usize, inter: usize, k: usize) -> GateUpPath {
        if Self::gate_up_supports(inter, k) && rows >= GATE_UP_FUSED_MIN_ROWS {
            GateUpPath::FusedSmall
        } else {
            GateUpPath::Vendor
        }
    }

    /// `out[rows, inter] = bf16(silu(gate) * up)` with `gate` and `up` the bf16-rounded halves
    /// of `x[rows, k] * w^T + bias`; `w` is the stacked `[2 * inter, k]` gate-then-up weight and
    /// `bias` the optional `[2 * inter]` bias.
    /// The `[rows, 2 * inter]` intermediate is never materialised. The result matches the
    /// vendor GEMM followed by `silu_mul_fused` up to the GEMM's fp32 accumulation order.
    #[allow(clippy::too_many_arguments)]
    pub fn gate_up_silu(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<u16>,
        w: &CudaSlice<u16>,
        bias: Option<&CudaSlice<u16>>,
        out: &mut CudaSlice<u16>,
        rows: usize,
        inter: usize,
        k: usize,
    ) -> Result<()> {
        if !Self::gate_up_supports(inter, k) {
            return Err(CudaError::Module(format!(
                "gate/up projection does not support inter {inter}, k {k}"
            )));
        }
        let small = Self::gate_up_path(rows, inter, k) == GateUpPath::FusedSmall;
        self.gate_up_silu_with(stream, small, x, w, bias, out, rows, inter, k)
    }

    /// As [`Self::gate_up_silu`] on the large (`small` false) or small tile variant.
    #[allow(clippy::too_many_arguments)]
    pub fn gate_up_silu_with(
        &self,
        stream: &Arc<CudaStream>,
        small: bool,
        x: &CudaSlice<u16>,
        w: &CudaSlice<u16>,
        bias: Option<&CudaSlice<u16>>,
        out: &mut CudaSlice<u16>,
        rows: usize,
        inter: usize,
        k: usize,
    ) -> Result<()> {
        if !Self::gate_up_supports(inter, k) {
            return Err(CudaError::Module(format!(
                "gate/up projection does not support inter {inter}, k {k}"
            )));
        }
        if rows == 0 {
            return Ok(());
        }
        let (cols, smem, kernel) = if small {
            (GATE_UP_SMALL_COLS, GATE_UP_SMALL_SMEM, &self.gate_up_small)
        } else {
            (GATE_UP_BLOCK_COLS, GATE_UP_SMEM, &self.gate_up)
        };
        let cfg = LaunchConfig {
            grid_dim: (
                (inter / cols) as u32,
                ceil_div(rows, GATE_UP_BLOCK_M) as u32,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem,
        };
        let (rows, inter, k) = (rows as i32, inter as i32, k as i32);
        // SAFETY: argument order and types match the kernel signature; `x` covers `rows * k`,
        // `w` `2 * inter * k` and `out` `rows * inter` elements by the caller's contract, and
        // the kernel bounds row accesses by `rows`.
        let null_bias = 0u64;
        unsafe {
            let mut launch = stream.launch_builder(kernel);
            launch.arg(x).arg(w);
            match bias {
                Some(b) => launch.arg(b),
                None => launch.arg(&null_bias),
            };
            launch.arg(out).arg(&rows).arg(&inter).arg(&k).launch(cfg)?;
        }
        Ok(())
    }

    /// Non-causal attention of every head over the packed `qkv` (`[tokens, 3, heads, 64]`),
    /// written to `out` (`[tokens, heads, 64]`). Sequences are the `num_seqs` ranges of
    /// `cu_seqlens`; attention never crosses a range, and `max_seqlen` bounds the longest.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        stream: &Arc<CudaStream>,
        qkv: &CudaSlice<u16>,
        out: &mut CudaSlice<u16>,
        cu_seqlens: &CudaSlice<i32>,
        num_seqs: usize,
        max_seqlen: usize,
        heads: usize,
        scale: f32,
    ) -> Result<()> {
        self.launch_attention(
            stream, qkv, out, cu_seqlens, num_seqs, max_seqlen, heads, scale, None,
        )
    }

    /// As [`Self::attention`] over a `qkv` whose q section is raw (not yet normalised or
    /// rotated): each query row is RMS-normalised with `q_w` (`[64]`) and rotated by 2-D axial
    /// RoPE from `positions` (`[row, col]` pairs per token) while it is loaded. The k section
    /// must already be normalised and rotated.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_fold_q(
        &self,
        stream: &Arc<CudaStream>,
        qkv: &CudaSlice<u16>,
        out: &mut CudaSlice<u16>,
        cu_seqlens: &CudaSlice<i32>,
        num_seqs: usize,
        max_seqlen: usize,
        heads: usize,
        scale: f32,
        q_w: &CudaSlice<u16>,
        positions: &CudaSlice<i32>,
        theta: f32,
        eps: f32,
    ) -> Result<()> {
        self.launch_attention(
            stream,
            qkv,
            out,
            cu_seqlens,
            num_seqs,
            max_seqlen,
            heads,
            scale,
            Some((q_w, positions, theta, eps)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_attention(
        &self,
        stream: &Arc<CudaStream>,
        qkv: &CudaSlice<u16>,
        out: &mut CudaSlice<u16>,
        cu_seqlens: &CudaSlice<i32>,
        num_seqs: usize,
        max_seqlen: usize,
        heads: usize,
        scale: f32,
        fold: Option<(&CudaSlice<u16>, &CudaSlice<i32>, f32, f32)>,
    ) -> Result<()> {
        if num_seqs == 0 || max_seqlen == 0 {
            return Ok(());
        }
        let cfg = LaunchConfig {
            grid_dim: (
                ceil_div(max_seqlen, VISION_ATTN_BLOCK_Q) as u32,
                heads as u32,
                num_seqs as u32,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let heads = heads as i32;
        debug_assert_eq!(VISION_HEAD_DIM, 64);
        let null = 0u64;
        // SAFETY: argument types match the kernel signature; the kernel bounds every access
        // by `cu_seqlens`, and `qkv` / `out` cover `tokens * 3 * heads * 64` / `tokens * heads * 64`
        // elements by the caller's contract; `positions` covers `2 * tokens` entries and `q_w`
        // 64 when given, and null pointers select the as-stored q path.
        unsafe {
            let mut l = stream.launch_builder(&self.attn);
            l.arg(qkv).arg(out).arg(cu_seqlens).arg(&heads).arg(&scale);
            match fold {
                Some((q_w, positions, theta, eps)) => {
                    l.arg(positions)
                        .arg(q_w)
                        .arg(&theta)
                        .arg(&eps)
                        .launch(cfg)?;
                }
                None => {
                    let (theta, eps) = (0.0f32, 0.0f32);
                    l.arg(&null).arg(&null).arg(&theta).arg(&eps).launch(cfg)?;
                }
            }
        }
        Ok(())
    }

    /// Per row: `y = bf16(x + bias)` (when `bias` is given), `sum_out = bf16(y + residual)` and
    /// `out = rms_norm(sum_out) * w`. `x` is a bias-free GEMM output; `sum_out` and `out` must
    /// not alias `x`.
    #[allow(clippy::too_many_arguments)]
    pub fn bias_add_rms_norm(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<u16>,
        bias: Option<&CudaSlice<u16>>,
        residual: &CudaSlice<u16>,
        w: &CudaSlice<u16>,
        sum_out: &mut CudaSlice<u16>,
        out: &mut CudaSlice<u16>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        if !Self::join_supports(n) {
            return Err(CudaError::Module(format!(
                "vision join does not support rows of {n} elements"
            )));
        }
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (JOIN_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let n = n as i32;
        let none = 0u64;
        // SAFETY: argument order and types match the kernel signature; buffers cover
        // `rows * n` elements (`bias`, `w`: `n`); a null bias pointer disables the bias.
        unsafe {
            let mut b = stream.launch_builder(&self.join);
            b.arg(x);
            match bias {
                Some(bias) => b.arg(bias),
                None => b.arg(&none),
            };
            b.arg(residual)
                .arg(w)
                .arg(sum_out)
                .arg(out)
                .arg(&n)
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
    fn gate_up_path_is_a_pure_function_of_the_shape() {
        for (rows, inter, k) in [(255, 4096, 1024), (256, 4096, 1024), (65536, 4608, 1536)] {
            assert_eq!(
                VisionKernels::gate_up_path(rows, inter, k),
                VisionKernels::gate_up_path(rows, inter, k)
            );
        }
    }

    #[test]
    fn gate_up_path_uses_fused_small_from_the_row_threshold() {
        assert_eq!(
            VisionKernels::gate_up_path(GATE_UP_FUSED_MIN_ROWS - 1, 4096, 1024),
            GateUpPath::Vendor
        );
        assert_eq!(
            VisionKernels::gate_up_path(GATE_UP_FUSED_MIN_ROWS, 4096, 1024),
            GateUpPath::FusedSmall
        );
        assert_eq!(
            VisionKernels::gate_up_path(16384, 4608, 1536),
            GateUpPath::FusedSmall
        );
    }

    #[test]
    fn gate_up_path_is_vendor_for_unsupported_shapes() {
        assert_eq!(
            VisionKernels::gate_up_path(4096, 4100, 1024),
            GateUpPath::Vendor
        );
        assert_eq!(
            VisionKernels::gate_up_path(4096, 4096, 1000),
            GateUpPath::Vendor
        );
    }
}
