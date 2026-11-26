//! Batched variable-length vision transformer over the concatenated patches of many images.

use std::sync::{Arc, Mutex};

use cudarc::driver::sys;
use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use scriptor_cuda::{CudaError, Result};
use scriptor_kernels::activation::ActivationKernels;
use scriptor_kernels::attention::{VarlenAttention, VarlenGeometry};
use scriptor_kernels::cublaslt::{GemmEngine, GemmProblem};
use scriptor_kernels::heuristics::{Epilogue, GemmDtype};
use scriptor_kernels::norm::NormKernels;
use scriptor_kernels::reference::conv::PATCH_ELEMS;
use scriptor_kernels::reference::rope::vision_positions;
use scriptor_kernels::rope::{RopeKernels, VISION_HEAD_DIM};
use scriptor_kernels::vision::VisionKernels;

use super::fused_forward::Fusion;
use super::params::GpuVisionParams;
use super::workspace::{MERGE_GROUP, VisionWorkspace};
use super::{TUNE_ITERATIONS, dptr, linear, linear_with};

/// Shape facts of one executed vision batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionBatchStats {
    /// Patches processed by the tower (the GEMM row count).
    pub patches: usize,
    /// Attention sequences (one per frame).
    pub frames: usize,
    /// Output tokens across all images.
    pub tokens: usize,
}

/// Time, arithmetic and planned memory traffic of one named vision stage, summed over calls.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionStageStat {
    /// Stage name.
    pub name: &'static str,
    /// Calls recorded.
    pub calls: u32,
    /// Planned floating-point operations, summed over calls.
    pub flops: u64,
    /// Planned bytes moved, summed over calls.
    pub bytes: u64,
    /// Measured device milliseconds, summed over calls.
    pub millis: f64,
}

impl VisionStageStat {
    /// Achieved TFLOP/s from planned operations over measured time.
    pub fn tflops(&self) -> f64 {
        if self.millis > 0.0 {
            self.flops as f64 / (self.millis * 1e-3) / 1e12
        } else {
            0.0
        }
    }

    /// Achieved bandwidth in GB/s from planned bytes over measured time.
    pub fn gbps(&self) -> f64 {
        if self.millis > 0.0 {
            self.bytes as f64 / (self.millis * 1e-3) / 1e9
        } else {
            0.0
        }
    }
}

type Mark = (CudaEvent, &'static str, u64, u64);

struct RunMarks {
    start: CudaEvent,
    marks: Vec<Mark>,
}

/// CUDA-event stage timer for one `run`; inert when profiling is off.
pub(crate) struct Marker {
    stream: Arc<CudaStream>,
    run: Option<RunMarks>,
}

impl Marker {
    fn new(stream: &Arc<CudaStream>, on: bool) -> Result<Self> {
        let run = if on {
            let start = stream
                .context()
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            start.record(stream)?;
            Some(RunMarks {
                start,
                marks: Vec::new(),
            })
        } else {
            None
        };
        Ok(Self {
            stream: stream.clone(),
            run,
        })
    }

    /// Closes the stage that ended at this point in the stream.
    pub(crate) fn mark(&mut self, name: &'static str, flops: u64, bytes: u64) -> Result<()> {
        if let Some(run) = self.run.as_mut() {
            let ev = self
                .stream
                .context()
                .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            ev.record(&self.stream)?;
            run.marks.push((ev, name, flops, bytes));
        }
        Ok(())
    }
}

/// Planned operations and bytes of `out[m, n] = x[m, k] * W[n, k]^T` in BF16.
pub(crate) fn gemm_cost(m: usize, n: usize, k: usize) -> (u64, u64) {
    (2 * (m * n * k) as u64, 2 * (m * k + n * k + m * n) as u64)
}

/// Vision executor: loaded kernels plus device weights.
pub struct GpuVision {
    params: Arc<GpuVisionParams>,
    norm: NormKernels,
    rope: RopeKernels,
    act: ActivationKernels,
    attn: VarlenAttention,
    fused_kernels: VisionKernels,
    fusion: Fusion,
    gate_up_silu: bool,
    profiling: bool,
    recorded: Mutex<Vec<RunMarks>>,
}

/// Kernel set a new executor starts with: [`Fusion::Fused`] unless `SCRIPTOR_VISION_FUSION=unfused`.
fn default_fusion() -> Fusion {
    match std::env::var("SCRIPTOR_VISION_FUSION").as_deref() {
        Ok("unfused") => Fusion::Unfused,
        _ => Fusion::Fused,
    }
}

fn invalid(msg: String) -> CudaError {
    CudaError::Module(msg)
}

/// Copies column block `index` (width `width` elements) of the row-interleaved `qkv`
/// (`[rows, 3 * width]`) into the dense `dst` (`[rows, width]`).
fn split_columns(
    stream: &Arc<CudaStream>,
    qkv: &CudaSlice<u16>,
    dst: &mut CudaSlice<u16>,
    rows: usize,
    width: usize,
    index: usize,
) -> Result<()> {
    let elem = std::mem::size_of::<u16>();
    let src_base = qkv.device_ptr(stream).0 + (index * width * elem) as u64;
    let dst_base = dst.device_ptr_mut(stream).0;
    let desc = sys::CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: src_base,
        srcArray: std::ptr::null_mut(),
        srcPitch: 3 * width * elem,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: dst_base,
        dstArray: std::ptr::null_mut(),
        dstPitch: width * elem,
        WidthInBytes: width * elem,
        Height: rows,
    };
    // SAFETY: both regions are live device allocations on `stream`'s context and cover
    // `rows` pitched rows of `width` elements.
    let code = unsafe { sys::cuMemcpy2DAsync_v2(&desc, stream.cu_stream()) };
    if code == sys::cudaError_enum::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaError::Driver(format!("cuMemcpy2DAsync: {code:?}")))
    }
}

impl GpuVision {
    /// Measures, in memory only, the cuBLASLt candidate choices for the tower's linear layers at
    /// every row bucket from 512 to the bucket holding `max_rows`. Only candidates whose output
    /// is bit-identical to the first candidate's are eligible (see
    /// [`GemmEngine::autotune_in_memory`]).
    pub fn autotune(&self, gemm: &mut GemmEngine, max_rows: usize) -> Result<super::TuneReport> {
        let p = &self.params;
        let Some(block) = p.blocks.first() else {
            return Ok(super::TuneReport::default());
        };
        let buckets = super::tune_buckets(512, max_rows);
        let epilogue = |l: &super::GpuLinear, bias: bool| {
            if bias && l.bias.is_some() {
                Epilogue::Bias
            } else {
                Epilogue::None
            }
        };
        let fused = self.fusion == Fusion::Fused && VisionKernels::join_supports(p.hidden);
        let layers = [
            (&block.qkv, true),
            (&block.proj, true),
            (&block.down, !fused),
            (&block.gate_up, true),
        ];
        let problems: Vec<GemmProblem> = layers
            .iter()
            .flat_map(|(l, bias)| {
                buckets.iter().map(move |&m| GemmProblem {
                    m,
                    n: l.out_dim,
                    k: l.in_dim,
                    dtype: GemmDtype::Bf16,
                    epilogue: epilogue(l, *bias),
                })
            })
            .collect();
        let gemm_tuned = gemm.autotune_in_memory(&problems, TUNE_ITERATIONS)?;
        Ok(super::TuneReport { gemm: gemm_tuned })
    }

    /// Loads the kernels into `ctx` and takes ownership of the device weights.
    pub fn new(ctx: &Arc<CudaContext>, params: GpuVisionParams) -> Result<Self> {
        Self::with_params(ctx, Arc::new(params))
    }

    /// Loads the kernels into `ctx` around device weights that other instances may share. The
    /// weights must have finished uploading, and must outlive every stream that reads them.
    pub fn with_params(ctx: &Arc<CudaContext>, params: Arc<GpuVisionParams>) -> Result<Self> {
        if params.head_dim != VISION_HEAD_DIM {
            return Err(invalid(format!(
                "vision kernels require head_dim {VISION_HEAD_DIM}, found {}",
                params.head_dim
            )));
        }
        if params.heads * params.head_dim != params.hidden {
            return Err(invalid(
                "vision heads * head_dim differs from hidden".into(),
            ));
        }
        Ok(Self {
            params,
            norm: NormKernels::load(ctx)?,
            rope: RopeKernels::load(ctx)?,
            act: ActivationKernels::load(ctx)?,
            attn: VarlenAttention::load(ctx)?,
            fused_kernels: VisionKernels::load(ctx)?,
            fusion: default_fusion(),
            gate_up_silu: true,
            profiling: false,
            recorded: Mutex::new(Vec::new()),
        })
    }

    /// Selects the kernel set used by [`Self::run`].
    pub fn set_fusion(&mut self, fusion: Fusion) {
        self.fusion = fusion;
    }

    /// Enables (`true`, the default) or disables the SiLU-fused gate/up projection of the fused
    /// kernel set; disabled, the MLP runs the vendor GEMM and `silu_mul`.
    pub fn set_gate_up_silu(&mut self, on: bool) {
        self.gate_up_silu = on;
    }

    /// Kernel set currently selected.
    pub fn fusion(&self) -> Fusion {
        self.fusion
    }

    /// Starts (`true`) or stops and discards (`false`) per-stage timing of [`Self::run`].
    pub fn set_profile(&mut self, on: bool) {
        self.profiling = on;
        if let Ok(mut r) = self.recorded.lock() {
            r.clear();
        }
    }

    /// Stages recorded since profiling started or the last call, in first-seen order, with
    /// times from CUDA events. Synchronises the stream.
    pub fn take_profile(&self, stream: &Arc<CudaStream>) -> Result<Vec<VisionStageStat>> {
        stream.synchronize()?;
        let runs = self
            .recorded
            .lock()
            .map(|mut r| std::mem::take(&mut *r))
            .unwrap_or_default();
        let mut stats: Vec<VisionStageStat> = Vec::new();
        for run in runs {
            let mut prev = run.start;
            for (ev, name, flops, bytes) in run.marks {
                let millis = f64::from(prev.elapsed_ms(&ev)?);
                match stats.iter_mut().find(|s| s.name == name) {
                    Some(s) => {
                        s.calls += 1;
                        s.flops += flops;
                        s.bytes += bytes;
                        s.millis += millis;
                    }
                    None => stats.push(VisionStageStat {
                        name,
                        calls: 1,
                        flops,
                        bytes,
                        millis,
                    }),
                }
                prev = ev;
            }
        }
        Ok(stats)
    }

    /// The device weights.
    pub fn params(&self) -> &GpuVisionParams {
        &self.params
    }

    /// Embedding width produced per output token.
    pub fn out_dim(&self) -> usize {
        self.params.merger.out_dim
    }

    /// Runs the tower and adapter over a batch of images.
    ///
    /// `patches` holds BF16 `[total_patches, 1176]` rows (merge-window order, images
    /// concatenated); `grids` lists each image's `(t, h, w)` patch grid. Attention stays inside
    /// each frame. Image `i`'s `t * h * w / 4` embedding rows of `out_dim` are written to
    /// `outs[i]`. All work is enqueued on `stream`, which must also drive `gemm`.
    pub fn run(
        &self,
        stream: &Arc<CudaStream>,
        gemm: &mut GemmEngine,
        ws: &mut VisionWorkspace,
        patches: &CudaSlice<u16>,
        grids: &[(usize, usize, usize)],
        outs: &mut [&mut CudaSlice<u16>],
    ) -> Result<VisionBatchStats> {
        let p = &self.params;
        if grids.len() != outs.len() {
            return Err(invalid(format!(
                "{} grids but {} output buffers",
                grids.len(),
                outs.len()
            )));
        }
        let out_dim = p.merger.out_dim;
        let mut positions: Vec<i32> = Vec::new();
        let mut cu_seqlens: Vec<i32> = vec![0];
        let mut max_seqlen = 0usize;
        let mut total = 0usize;
        for (i, &(t, h, w)) in grids.iter().enumerate() {
            if t == 0 || h == 0 || w == 0 || h % 2 != 0 || w % 2 != 0 {
                return Err(invalid(format!(
                    "grid {t}x{h}x{w} is not divisible by the merge size"
                )));
            }
            if outs[i].len() < t * h * w / MERGE_GROUP * out_dim {
                return Err(invalid(format!("output buffer {i} is too small")));
            }
            positions.extend(vision_positions(t, h, w).iter().flatten());
            for _ in 0..t {
                let last = cu_seqlens.last().copied().unwrap_or(0);
                cu_seqlens.push(last + (h * w) as i32);
            }
            max_seqlen = max_seqlen.max(h * w);
            total += t * h * w;
        }
        let frames = cu_seqlens.len() - 1;
        if total == 0 {
            return Ok(VisionBatchStats {
                patches: 0,
                frames: 0,
                tokens: 0,
            });
        }
        if total > ws.patch_budget {
            return Err(invalid(format!(
                "{total} patches exceed the workspace budget of {}",
                ws.patch_budget
            )));
        }
        if frames > ws.frame_budget() {
            return Err(invalid(format!(
                "{frames} frames exceed the workspace budget of {}",
                ws.frame_budget()
            )));
        }
        if patches.len() < total * PATCH_ELEMS {
            return Err(invalid(format!(
                "patch buffer holds {} values for {total} patches",
                patches.len()
            )));
        }
        stream.memcpy_htod(&positions, &mut ws.positions.slice_mut(0..positions.len()))?;
        stream.memcpy_htod(
            &cu_seqlens,
            &mut ws.cu_seqlens.slice_mut(0..cu_seqlens.len()),
        )?;

        let (hidden, heads, inter) = (p.hidden, p.heads, p.intermediate);
        let geometry = VarlenGeometry::vision(heads);
        let scale = (p.head_dim as f32).powf(-0.5);
        let fused = self.fusion == Fusion::Fused && VisionKernels::join_supports(hidden);
        let gate_up_fused = fused
            && self.gate_up_silu
            && self.fused_kernels.gate_up_profitable(total, inter, hidden);
        let mut mk = Marker::new(stream, self.profiling)?;
        let rows_h = (total * hidden) as u64;
        let norm_bytes = 2 * 2 * rows_h;
        let join_bytes = 2 * 4 * rows_h;
        let attn_flops: u64 = grids
            .iter()
            .flat_map(|&(t, h, w)| std::iter::repeat_n((h * w) as u64, t))
            .map(|n| 4 * n * n * (heads * p.head_dim) as u64)
            .sum();

        linear(
            gemm,
            &p.patch_embed,
            dptr(stream, patches),
            dptr(stream, &ws.x_a),
            total,
            stream,
        )?;
        let (f, by) = gemm_cost(total, hidden, p.patch_embed.in_dim);
        mk.mark("patch_embed", f, by)?;
        if let Some(first) = p.blocks.first() {
            self.norm.rms_norm(
                stream,
                &ws.x_a,
                &first.norm1,
                &mut ws.normed,
                total,
                hidden,
                p.eps,
            )?;
            mk.mark("norm", 0, norm_bytes)?;
        }
        for (b, block) in p.blocks.iter().enumerate() {
            linear(
                gemm,
                &block.qkv,
                dptr(stream, &ws.normed),
                dptr(stream, &ws.qkv),
                total,
                stream,
            )?;
            let (f, by) = gemm_cost(total, 3 * hidden, hidden);
            mk.mark("qkv_gemm", f, by)?;
            if fused {
                self.rope.vision_k_norm_rope(
                    stream,
                    &mut ws.qkv,
                    &block.q_norm,
                    &block.k_norm,
                    &ws.positions,
                    total,
                    heads,
                    p.rope_theta,
                    p.eps,
                )?;
            } else {
                self.rope.vision_qk_norm_rope(
                    stream,
                    &mut ws.qkv,
                    &block.q_norm,
                    &block.k_norm,
                    &ws.positions,
                    total,
                    heads,
                    p.rope_theta,
                    p.eps,
                )?;
            }
            mk.mark(
                "qk_norm_rope",
                0,
                if fused {
                    2 * 2 * rows_h
                } else {
                    2 * 2 * 2 * rows_h
                },
            )?;
            if fused {
                self.fused_kernels.attention_fold_q(
                    stream,
                    &ws.qkv,
                    &mut ws.attn,
                    &ws.cu_seqlens,
                    frames,
                    max_seqlen,
                    heads,
                    scale,
                    &block.q_norm,
                    &ws.positions,
                    p.rope_theta,
                    p.eps,
                )?;
            } else {
                split_columns(stream, &ws.qkv, &mut ws.q, total, hidden, 0)?;
                split_columns(stream, &ws.qkv, &mut ws.k, total, hidden, 1)?;
                split_columns(stream, &ws.qkv, &mut ws.v, total, hidden, 2)?;
                mk.mark("qkv_split", 0, 2 * 2 * 3 * rows_h)?;
                self.attn.run(
                    stream,
                    &geometry,
                    &ws.q,
                    &ws.k,
                    &ws.v,
                    &mut ws.attn,
                    &ws.cu_seqlens,
                    frames,
                    max_seqlen,
                    scale,
                )?;
            }
            mk.mark("attention", attn_flops, 2 * 4 * rows_h)?;
            linear(
                gemm,
                &block.proj,
                dptr(stream, &ws.attn),
                dptr(stream, &ws.branch),
                total,
                stream,
            )?;
            let (f, by) = gemm_cost(total, hidden, hidden);
            mk.mark("proj_gemm", f, by)?;
            if fused {
                self.fused_kernels.bias_add_rms_norm(
                    stream,
                    &ws.branch,
                    None,
                    &ws.x_a,
                    &block.norm2,
                    &mut ws.x_b,
                    &mut ws.normed,
                    total,
                    hidden,
                    p.eps,
                )?;
            } else {
                self.norm.add_rms_norm(
                    stream,
                    &ws.branch,
                    &ws.x_a,
                    &block.norm2,
                    &mut ws.x_b,
                    &mut ws.normed,
                    total,
                    hidden,
                    p.eps,
                )?;
            }
            mk.mark("join_attn", 0, join_bytes)?;
            if gate_up_fused {
                self.fused_kernels.gate_up_silu(
                    stream,
                    &ws.normed,
                    &block.gate_up.weight,
                    block.gate_up.bias.as_ref(),
                    &mut ws.act,
                    total,
                    inter,
                    hidden,
                )?;
                let (f, by) = gemm_cost(total, inter, hidden);
                mk.mark("gate_up_silu_gemm", 2 * f, 2 * by)?;
            } else {
                linear(
                    gemm,
                    &block.gate_up,
                    dptr(stream, &ws.normed),
                    dptr(stream, &ws.gate_up),
                    total,
                    stream,
                )?;
                let (f, by) = gemm_cost(total, 2 * inter, hidden);
                mk.mark("gate_up_gemm", f, by)?;
                self.act
                    .silu_mul_fused(stream, &ws.gate_up, &mut ws.act, total, inter)?;
                mk.mark("silu_mul", 0, 2 * 3 * (total * inter) as u64)?;
            }
            linear_with(
                gemm,
                &block.down,
                dptr(stream, &ws.act),
                dptr(stream, &ws.branch),
                total,
                stream,
                !fused,
            )?;
            let (f, by) = gemm_cost(total, hidden, inter);
            mk.mark("down_gemm", f, by)?;
            let next_norm = p
                .blocks
                .get(b + 1)
                .map(|n| &n.norm1)
                .unwrap_or(&p.post_layernorm);
            if fused {
                self.fused_kernels.bias_add_rms_norm(
                    stream,
                    &ws.branch,
                    block.down.bias.as_ref(),
                    &ws.x_b,
                    next_norm,
                    &mut ws.x_a,
                    &mut ws.normed,
                    total,
                    hidden,
                    p.eps,
                )?;
            } else {
                self.norm.add_rms_norm(
                    stream,
                    &ws.branch,
                    &ws.x_b,
                    next_norm,
                    &mut ws.x_a,
                    &mut ws.normed,
                    total,
                    hidden,
                    p.eps,
                )?;
            }
            mk.mark("join_mlp", 0, join_bytes)?;
        }
        if p.blocks.is_empty() {
            self.norm.rms_norm(
                stream,
                &ws.x_a,
                &p.post_layernorm,
                &mut ws.normed,
                total,
                hidden,
                p.eps,
            )?;
            mk.mark("norm", 0, norm_bytes)?;
        }
        p.merger
            .run(stream, gemm, &self.norm, &self.act, ws, total, &mut mk)?;

        let mut row = 0usize;
        for (i, &(t, h, w)) in grids.iter().enumerate() {
            let rows = t * h * w / MERGE_GROUP;
            let src = ws.embeddings.slice(row * out_dim..(row + rows) * out_dim);
            let mut dst = outs[i].slice_mut(0..rows * out_dim);
            stream.memcpy_dtod(&src, &mut dst)?;
            row += rows;
        }
        mk.mark("output_copy", 0, 2 * 2 * (row * out_dim) as u64)?;
        if let (Some(run), Ok(mut rec)) = (mk.run.take(), self.recorded.lock()) {
            rec.push(run);
        }
        Ok(VisionBatchStats {
            patches: total,
            frames,
            tokens: row,
        })
    }
}
