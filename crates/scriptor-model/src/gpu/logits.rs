//! Row utilities and the fp32 logits head: strided row copies and bf16 to fp32 conversion
//! (loaded from embedded PTX), and the language-model-head GEMM over gathered rows.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, PushKernelArg,
};
use scriptor_cuda::{CudaError, KernelModule};
use scriptor_kernels::cublaslt::{GemmEngine, GemmOperands, GemmProblem};
use scriptor_kernels::heuristics::{Epilogue, GemmDtype};
use scriptor_kernels::scatter::ScatterKernels;

use super::decoder::GpuTextError;

/// PTX of the row copy and the bf16 to fp32 conversion kernels.
const ROW_OPS_PTX: &str = r#".version 8.0
.target sm_80
.address_size 64

.visible .entry scriptor_copy_rows_u16(
    .param .u64 p_src,
    .param .u64 p_dst,
    .param .u64 p_total,
    .param .u64 p_width,
    .param .u64 p_sstride,
    .param .u64 p_dstride
)
{
    .reg .pred %p<2>;
    .reg .b16 %h<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<20>;
    ld.param.u64 %rd1, [p_src];
    ld.param.u64 %rd2, [p_dst];
    ld.param.u64 %rd3, [p_total];
    ld.param.u64 %rd4, [p_width];
    ld.param.u64 %rd5, [p_sstride];
    ld.param.u64 %rd6, [p_dstride];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    cvt.u64.u32 %rd7, %r3;
    mad.wide.u32 %rd8, %r1, %r2, %rd7;
    setp.ge.u64 %p1, %rd8, %rd3;
    @%p1 bra DONE;
    div.u64 %rd9, %rd8, %rd4;
    mul.lo.u64 %rd10, %rd9, %rd4;
    sub.u64 %rd11, %rd8, %rd10;
    mad.lo.u64 %rd12, %rd9, %rd5, %rd11;
    mad.lo.u64 %rd13, %rd9, %rd6, %rd11;
    shl.b64 %rd14, %rd12, 1;
    shl.b64 %rd15, %rd13, 1;
    add.u64 %rd16, %rd1, %rd14;
    add.u64 %rd17, %rd2, %rd15;
    ld.global.u16 %h1, [%rd16];
    st.global.u16 [%rd17], %h1;
DONE:
    ret;
}

.visible .entry scriptor_bf16_to_f32(
    .param .u64 p_src,
    .param .u64 p_dst,
    .param .u64 p_n
)
{
    .reg .pred %p<2>;
    .reg .b16 %h<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<14>;
    ld.param.u64 %rd1, [p_src];
    ld.param.u64 %rd2, [p_dst];
    ld.param.u64 %rd3, [p_n];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    cvt.u64.u32 %rd4, %r3;
    mad.wide.u32 %rd5, %r1, %r2, %rd4;
    setp.ge.u64 %p1, %rd5, %rd3;
    @%p1 bra DONE;
    shl.b64 %rd6, %rd5, 1;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.u16 %h1, [%rd7];
    cvt.u32.u16 %r4, %h1;
    shl.b32 %r5, %r4, 16;
    shl.b64 %rd8, %rd5, 2;
    add.u64 %rd9, %rd2, %rd8;
    st.global.u32 [%rd9], %r5;
DONE:
    ret;
}
"#;

const BLOCK: u32 = 256;

/// Device address of element `offset` of a bf16 buffer.
pub(crate) fn address(stream: &Arc<CudaStream>, buf: &CudaSlice<u16>, offset: usize) -> u64 {
    buf.device_ptr(stream).0 + 2 * offset as u64
}

/// Strided row copy and bf16 to fp32 conversion kernels.
pub struct RowOps {
    copy: CudaFunction,
    to_f32: CudaFunction,
}

impl RowOps {
    /// Loads the kernels into `ctx`.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self, GpuTextError> {
        let module = KernelModule::load(ctx, ROW_OPS_PTX.as_bytes())?;
        Ok(Self {
            copy: module.kernel("scriptor_copy_rows_u16")?,
            to_f32: module.kernel("scriptor_bf16_to_f32")?,
        })
    }

    /// Copies `rows` rows of `width` elements from `src` (row pitch `src_stride`, starting at
    /// element `src_offset`) to `dst` (row pitch `dst_stride`, starting at `dst_offset`).
    #[allow(clippy::too_many_arguments)]
    pub fn copy_rows(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<u16>,
        src_offset: usize,
        src_stride: usize,
        dst: &mut CudaSlice<u16>,
        dst_offset: usize,
        dst_stride: usize,
        rows: usize,
        width: usize,
    ) -> Result<(), GpuTextError> {
        let total = (rows * width) as u64;
        if total == 0 {
            return Ok(());
        }
        let s = address(stream, src, src_offset);
        let d = address(stream, dst, dst_offset);
        let (width, ss, ds) = (width as u64, src_stride as u64, dst_stride as u64);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total.div_ceil(u64::from(BLOCK)) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: argument order and types match the kernel signature; the caller keeps both
        // ranges inside their buffers.
        unsafe {
            stream
                .launch_builder(&self.copy)
                .arg(&s)
                .arg(&d)
                .arg(&total)
                .arg(&width)
                .arg(&ss)
                .arg(&ds)
                .launch(cfg)
                .map_err(CudaError::from)?;
        }
        Ok(())
    }

    /// Widens the first `n` bf16 values of `src` to fp32 in `dst`.
    pub fn bf16_to_f32(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<u16>,
        dst: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GpuTextError> {
        if n == 0 {
            return Ok(());
        }
        let s = src.device_ptr(stream).0;
        let d = dst.device_ptr(stream).0;
        let n = n as u64;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n.div_ceil(u64::from(BLOCK)) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: argument order and types match the kernel signature; `src` holds `n` bf16
        // values and `dst` `n` fp32 values.
        unsafe {
            stream
                .launch_builder(&self.to_f32)
                .arg(&s)
                .arg(&d)
                .arg(&n)
                .launch(cfg)
                .map_err(CudaError::from)?;
        }
        Ok(())
    }
}

/// Selects `ids.len()`-many rows of `hidden` (`[rows, dim]`) by index into `out`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_rows(
    scatter: &ScatterKernels,
    stream: &Arc<CudaStream>,
    hidden: &CudaSlice<u16>,
    ids: &CudaSlice<i32>,
    out: &mut CudaSlice<u16>,
    selected: usize,
    rows: usize,
    dim: usize,
) -> Result<(), GpuTextError> {
    if selected == 0 {
        return Ok(());
    }
    scatter.embedding_gather(stream, hidden, ids, out, selected, dim, rows)?;
    Ok(())
}

/// Row count the head GEMM is issued with: a single row is padded to two so the weight is
/// streamed by the tensor-core kernel instead of the slower bf16 GEMV. Callers keep at least
/// this many rows allocated in both the hidden rows and the scratch.
pub(crate) const fn head_gemm_rows(rows: usize) -> usize {
    if rows < 2 { 2 } else { rows }
}

/// The head GEMM of `rows` post-norm hidden rows against `head` (`[vocab, hidden]`) into the
/// bf16 `scratch` (`[rows, vocab]`); both buffers hold at least [`head_gemm_rows`] rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lm_head_bf16(
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    hidden_rows: &CudaSlice<u16>,
    rows: usize,
    head: &CudaSlice<u16>,
    vocab: usize,
    hidden: usize,
    scratch: &mut CudaSlice<u16>,
) -> Result<(), GpuTextError> {
    if rows == 0 {
        return Ok(());
    }
    let problem = GemmProblem {
        m: head_gemm_rows(rows),
        n: vocab,
        k: hidden,
        dtype: GemmDtype::Bf16,
        epilogue: Epilogue::None,
    };
    let operands = GemmOperands {
        x: hidden_rows.device_ptr(stream).0,
        w: head.device_ptr(stream).0,
        out: scratch.device_ptr(stream).0,
        ..Default::default()
    };
    gemm.gemm(&problem, &operands)?;
    Ok(())
}

/// The greedy token of each of `rows` post-norm hidden rows against `head` (`[vocab, hidden]`):
/// the lowest index of the maximum bf16-rounded logit, written to `tokens` without storing the
/// logits. The shape must satisfy `GemmEngine::supports_argmax` for `rows`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lm_head_argmax(
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    hidden_rows: &CudaSlice<u16>,
    rows: usize,
    head: &CudaSlice<u16>,
    vocab: usize,
    hidden: usize,
    tokens: &mut CudaSlice<u32>,
) -> Result<(), GpuTextError> {
    if rows == 0 {
        return Ok(());
    }
    gemm.argmax_rows(
        rows,
        vocab,
        hidden,
        hidden_rows.device_ptr(stream).0,
        head.device_ptr(stream).0,
        tokens.device_ptr_mut(stream).0,
    )?;
    Ok(())
}

/// fp32 logits `[rows, vocab]` of `rows` post-norm hidden rows: a bf16 GEMM against `head`
/// (`[vocab, hidden]`) into `scratch`, widened to fp32; both buffers hold at least
/// [`head_gemm_rows`] rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lm_head_logits(
    gemm: &mut GemmEngine,
    ops: &RowOps,
    stream: &Arc<CudaStream>,
    hidden_rows: &CudaSlice<u16>,
    rows: usize,
    head: &CudaSlice<u16>,
    vocab: usize,
    hidden: usize,
    scratch: &mut CudaSlice<u16>,
) -> Result<CudaSlice<f32>, GpuTextError> {
    let mut logits = stream.alloc_zeros::<f32>((rows * vocab).max(1))?;
    if rows == 0 {
        return Ok(logits);
    }
    let problem = GemmProblem {
        m: head_gemm_rows(rows),
        n: vocab,
        k: hidden,
        dtype: GemmDtype::Bf16,
        epilogue: Epilogue::None,
    };
    let operands = GemmOperands {
        x: hidden_rows.device_ptr(stream).0,
        w: head.device_ptr(stream).0,
        out: scratch.device_ptr(stream).0,
        ..Default::default()
    };
    gemm.gemm(&problem, &operands)?;
    ops.bf16_to_f32(stream, scratch, &mut logits, rows * vocab)?;
    Ok(logits)
}
