//! Decoder layer built from the fused kernels, and the optional per-stage profiler.
//!
//! A layer runs `fused_rope_kv` in place of the separate RoPE and KV-append launches and
//! `norm_add_norm` for each of the two sandwich-norm joins (post-sublayer norm, residual add and
//! the following norm), so a layer issues four non-GEMM launches besides attention instead of
//! seven. [`Fusion::Unfused`] keeps the separate launches in the same staged body so the
//! profiler can compare both.

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaEvent, CudaSlice};

use super::decoder::{GpuLayerWeights, GpuTextError, LayerCache};
use scriptor_kernels::rope_kv::{RopeKvSelect, RopeKvShape};

use super::forward::{DevicePlan, TextExecutor, gemm_bf16};

/// Which kernels a decoder layer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fusion {
    /// Separate RoPE, KV-append, RMSNorm and add-RMSNorm launches.
    Unfused,
    /// Fused RoPE + KV append and fused norm + residual + norm joins.
    #[default]
    Fused,
}

/// Time and planned memory traffic of one named stage, summed over every recorded call.
#[derive(Debug, Clone, PartialEq)]
pub struct StageStat {
    /// Stage name.
    pub name: &'static str,
    /// Kernel launches issued by the stage per call.
    pub launches: u32,
    /// Calls recorded.
    pub calls: u32,
    /// Planned bytes moved, summed over calls.
    pub bytes: u64,
    /// Measured device milliseconds, summed over calls.
    pub millis: f64,
}

impl StageStat {
    /// Achieved bandwidth in GB/s from planned bytes over measured time.
    pub fn gbps(&self) -> f64 {
        if self.millis > 0.0 {
            self.bytes as f64 / (self.millis * 1e-3) / 1e9
        } else {
            0.0
        }
    }
}

/// CUDA-event stage timer; each mark closes the stage that ended at that point.
pub struct StageProfiler {
    marks: Vec<(CudaEvent, &'static str, u32, u64)>,
    start: Option<CudaEvent>,
}

impl StageProfiler {
    fn new() -> Self {
        Self {
            marks: Vec::new(),
            start: None,
        }
    }
}

fn event(ex: &TextExecutor) -> Result<CudaEvent, GpuTextError> {
    Ok(ex
        .stream
        .context()
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?)
}

impl TextExecutor {
    /// Selects the kernels used by [`Self::forward`] and the resident decode step.
    pub fn set_fusion(&mut self, fusion: Fusion) {
        self.fusion = fusion;
    }

    /// Enables (`true`, the default) or disables the SiLU-fused gate/up projection of the fused
    /// kernel set; disabled, the MLP runs the vendor GEMM and `silu_mul`.
    pub fn set_gate_up_silu(&mut self, on: bool) {
        self.gate_up_silu = on;
    }

    /// Selects the fused RoPE + KV-append variants for the fused kernel set; both are off by
    /// default. Each applies to the passes launched after the call; a captured graph keeps the
    /// variant it was captured with, so recapture after changing the selection.
    pub fn set_rope_kv(&mut self, select: RopeKvSelect) {
        self.rope_kv = select;
    }

    /// Variants currently selected.
    pub fn rope_kv(&self) -> RopeKvSelect {
        self.rope_kv
    }

    /// Kernels currently selected.
    pub fn fusion(&self) -> Fusion {
        self.fusion
    }

    /// Starts (`true`) or stops and discards (`false`) per-stage timing of fused-path layers.
    pub fn set_profile(&mut self, on: bool) {
        self.profiler = on.then(StageProfiler::new);
    }

    /// Stages recorded since profiling started or the last call, in first-seen order, with
    /// times from CUDA events. Synchronises the stream.
    pub fn take_profile(&mut self) -> Result<Vec<StageStat>, GpuTextError> {
        let Some(p) = self.profiler.take() else {
            return Ok(Vec::new());
        };
        self.stream.synchronize()?;
        let mut stats: Vec<StageStat> = Vec::new();
        let mut prev = p.start;
        for (ev, name, launches, bytes) in p.marks {
            let millis = prev.as_ref().map_or(Ok(0.0), |s| s.elapsed_ms(&ev))? as f64;
            match stats.iter_mut().find(|s| s.name == name) {
                Some(s) => {
                    s.calls += 1;
                    s.bytes += bytes;
                    s.millis += millis;
                }
                None => stats.push(StageStat {
                    name,
                    launches,
                    calls: 1,
                    bytes,
                    millis,
                }),
            }
            prev = Some(ev);
        }
        self.profiler = Some(StageProfiler::new());
        Ok(stats)
    }

    fn begin_profile(&mut self) -> Result<(), GpuTextError> {
        if self.profiler.as_ref().is_some_and(|p| p.start.is_none()) {
            let ev = event(self)?;
            ev.record(&self.stream)?;
            if let Some(p) = self.profiler.as_mut() {
                p.start = Some(ev);
            }
        }
        Ok(())
    }

    fn mark(&mut self, name: &'static str, launches: u32, bytes: u64) -> Result<(), GpuTextError> {
        if self.profiler.is_some() {
            let ev = event(self)?;
            ev.record(&self.stream)?;
            if let Some(p) = self.profiler.as_mut() {
                p.marks.push((ev, name, launches, bytes));
            }
        }
        Ok(())
    }

    /// Runs one layer with `fusion`, recording stage marks when profiling is on.
    pub(super) fn layer_staged(
        &mut self,
        fusion: Fusion,
        w: &GpuLayerWeights,
        cache: &mut LayerCache,
        next_norm: &CudaSlice<u16>,
        plan: &DevicePlan,
    ) -> Result<(), GpuTextError> {
        let g = self.geometry;
        let rows = plan.rows;
        let r = rows as u64;
        let (h, qd, kd, inter) = (
            g.hidden as u64,
            g.q_dim() as u64,
            g.kv_dim() as u64,
            g.intermediate as u64,
        );
        let gemm_bytes = |out: u64, inp: u64| 2 * (out * inp + r * (out + inp));
        self.begin_profile()?;
        let s = self.stream.clone();
        gemm_bf16(
            &mut self.gemm,
            &s,
            &self.work.hn,
            &w.qkv,
            &mut self.work.qkv,
            rows,
            g.q_dim() + 2 * g.kv_dim(),
            g.hidden,
        )?;
        self.mark("qkv_gemm", 1, gemm_bytes(qd + 2 * kd, h))?;
        let rope_bytes = 2 * r * (2 * (qd + kd));
        match fusion {
            Fusion::Fused => {
                let variant = self.rope_kv.plan(&RopeKvShape {
                    rows,
                    prefill_rows: plan.prefill_rows,
                    head_dim: g.head_dim,
                    sections: g.mrope_section,
                });
                debug_assert!(variant.serves(plan.prefill_rows));
                let scratch =
                    (!variant.skip_kv_scratch).then_some((&mut self.work.k, &mut self.work.v));
                self.fused.split_rope_kv_append_variant(
                    &s,
                    variant,
                    &self.work.qkv,
                    &mut self.work.q,
                    scratch,
                    &mut cache.key,
                    &mut cache.value,
                    &plan.positions,
                    &plan.slots,
                    rows,
                    g.heads,
                    g.kv_heads,
                    g.head_dim,
                    g.rope_theta,
                    g.mrope_section,
                )?;
                let skipped = if variant.skip_kv_scratch {
                    4 * r * kd
                } else {
                    0
                };
                self.mark(
                    "split_rope_kv_append",
                    1,
                    rope_bytes + 2 * r * 5 * kd - skipped,
                )?;
            }
            Fusion::Unfused => {
                self.fused.split_qkv(
                    &s,
                    &self.work.qkv,
                    &mut self.work.q,
                    &mut self.work.k,
                    &mut self.work.v,
                    rows,
                    g.q_dim(),
                    g.kv_dim(),
                )?;
                self.mark("qkv_split", 1, 2 * r * (qd + 2 * kd))?;
                self.rope.mrope(
                    &s,
                    &mut self.work.q,
                    &mut self.work.k,
                    &plan.positions,
                    rows,
                    g.heads,
                    g.kv_heads,
                    g.head_dim,
                    g.rope_theta,
                    g.mrope_section,
                )?;
                self.mark("rope", 1, rope_bytes)?;
                self.kv_append.append(
                    &s,
                    &mut cache.key,
                    &mut cache.value,
                    &self.work.k,
                    &self.work.v,
                    &plan.slots,
                    rows,
                    g.kv_heads,
                    g.head_dim,
                )?;
                self.mark("kv_append", 1, 2 * r * 4 * kd)?;
            }
        }
        self.attention(plan, cache)?;
        let ctx_bytes = (plan.decode_seqs * plan.max_ctx) as u64 * kd * 2 * 2;
        self.mark(
            "attention",
            1,
            ctx_bytes + 2 * r * 2 * qd + 2 * plan.prefill_rows as u64 * 2 * kd,
        )?;
        gemm_bf16(
            &mut self.gemm,
            &s,
            &self.work.attn,
            &w.o,
            &mut self.work.o,
            rows,
            g.hidden,
            g.q_dim(),
        )?;
        self.mark("o_gemm", 1, gemm_bytes(h, qd))?;
        let join_bytes = 2 * r * h * 4 + 2 * 2 * h;
        match fusion {
            Fusion::Fused => {
                self.fused.norm_add_norm(
                    &s,
                    &self.work.o,
                    &self.work.h,
                    &w.post_self_attn_norm,
                    &w.post_attention_norm,
                    &mut self.work.h2,
                    &mut self.work.hn2,
                    rows,
                    g.hidden,
                    g.eps,
                )?;
                self.mark("attn_join", 1, join_bytes)?;
            }
            Fusion::Unfused => {
                self.norm.rms_norm(
                    &s,
                    &self.work.o,
                    &w.post_self_attn_norm,
                    &mut self.work.o2,
                    rows,
                    g.hidden,
                    g.eps,
                )?;
                self.norm.add_rms_norm(
                    &s,
                    &self.work.h,
                    &self.work.o2,
                    &w.post_attention_norm,
                    &mut self.work.h2,
                    &mut self.work.hn2,
                    rows,
                    g.hidden,
                    g.eps,
                )?;
                self.mark("attn_join", 2, join_bytes + 2 * r * h * 3)?;
            }
        }
        if fusion == Fusion::Fused
            && self.gate_up_silu
            && self
                .gate_up_kernels
                .gate_up_profitable(rows, g.intermediate, g.hidden)
        {
            self.gate_up_kernels.gate_up_silu(
                &s,
                &self.work.hn2,
                &w.gate_up,
                None,
                &mut self.work.act,
                rows,
                g.intermediate,
                g.hidden,
            )?;
            self.mark(
                "gate_up_silu_gemm",
                1,
                gemm_bytes(2 * inter, h) - 2 * r * inter,
            )?;
        } else {
            gemm_bf16(
                &mut self.gemm,
                &s,
                &self.work.hn2,
                &w.gate_up,
                &mut self.work.gu,
                rows,
                2 * g.intermediate,
                g.hidden,
            )?;
            self.mark("gate_up_gemm", 1, gemm_bytes(2 * inter, h))?;
            self.act
                .silu_mul_fused(&s, &self.work.gu, &mut self.work.act, rows, g.intermediate)?;
            self.mark("silu_mul", 1, 2 * r * 3 * inter)?;
        }
        gemm_bf16(
            &mut self.gemm,
            &s,
            &self.work.act,
            &w.down,
            &mut self.work.o,
            rows,
            g.hidden,
            g.intermediate,
        )?;
        self.mark("down_gemm", 1, gemm_bytes(h, inter))?;
        match fusion {
            Fusion::Fused => {
                self.fused.norm_add_norm(
                    &s,
                    &self.work.o,
                    &self.work.h2,
                    &w.post_mlp_norm,
                    next_norm,
                    &mut self.work.h,
                    &mut self.work.hn,
                    rows,
                    g.hidden,
                    g.eps,
                )?;
                self.mark("mlp_join", 1, join_bytes)?;
            }
            Fusion::Unfused => {
                self.norm.rms_norm(
                    &s,
                    &self.work.o,
                    &w.post_mlp_norm,
                    &mut self.work.o2,
                    rows,
                    g.hidden,
                    g.eps,
                )?;
                self.norm.add_rms_norm(
                    &s,
                    &self.work.h2,
                    &self.work.o2,
                    next_norm,
                    &mut self.work.h,
                    &mut self.work.hn,
                    rows,
                    g.hidden,
                    g.eps,
                )?;
                self.mark("mlp_join", 2, join_bytes + 2 * r * h * 3)?;
            }
        }
        Ok(())
    }
}
