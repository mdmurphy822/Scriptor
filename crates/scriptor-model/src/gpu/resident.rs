//! Forward passes over device-resident row tables: the row indices, positions, KV slots and
//! sequence lengths live in buffers owned by a [`StepPlan`], rewritten on the device between
//! passes, so a pass performs no host upload, allocation or synchronisation and can be captured
//! into a graph.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use scriptor_kernels::attention::{
    DecodeCapacity, DecodeGeometry, DecodeWorkspace, PagedDecodeAttention,
};
use scriptor_kernels::step::PassRows;

use super::decoder::{GpuDecoderWeights, GpuKvPool, GpuTextError};
use super::forward::{DevicePlan, HiddenSource, TextExecutor};
use super::logits::{gather_rows, lm_head_argmax, lm_head_bf16};
use super::mtp::GpuMtpWeights;

/// How the resident MTP pass builds the `eh_proj` input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MtpPrologueRoute {
    /// Embedding gather, two RMSNorms and two row copies.
    Reference,
    /// One kernel writing both halves.
    Fused,
}

impl MtpPrologueRoute {
    pub(super) fn select(fused: bool) -> Self {
        if fused { Self::Fused } else { Self::Reference }
    }
}

fn input_err(msg: impl Into<String>) -> GpuTextError {
    GpuTextError::Input(msg.into())
}

/// Device row tables for decode passes of up to `max_rows` rows over `max_seqs` sequences. The
/// block tables are persistent per-sequence state written by the owner; every other table is
/// filled by the step bookkeeping kernels before each pass.
pub struct StepPlan {
    plan: DevicePlan,
    max_rows: usize,
    max_seqs: usize,
}

impl StepPlan {
    /// Allocates tables for `max_rows` rows, `max_seqs` sequences and `max_blocks` block-table
    /// entries per sequence.
    pub fn new(
        stream: &Arc<CudaStream>,
        kv: &GpuKvPool,
        max_rows: usize,
        max_seqs: usize,
        max_blocks: usize,
    ) -> Result<Self, GpuTextError> {
        let rows = max_rows.max(1);
        let seqs = max_seqs.max(1);
        let z = |n: usize| stream.alloc_zeros::<i32>(n.max(1));
        let plan = DevicePlan {
            rows,
            prefill_rows: 0,
            prefill_seqs: 0,
            max_prefill_len: 0,
            decode_rows: rows,
            decode_seqs: seqs,
            decode_q_len: 1,
            max_ctx: 1,
            max_blocks: max_blocks.max(1),
            block_size: kv.block_size(),
            image_count: 0,
            sample_count: rows,
            zero_rows: Vec::new(),
            ids: z(rows)?,
            positions: z(rows * 3)?,
            slots: z(rows)?,
            cu_seqlens: z(1)?,
            block_tables: z(seqs * max_blocks.max(1))?,
            seq_lens: z(seqs)?,
            image_slots: z(1)?,
            sample_rows: z(rows)?,
        };
        Ok(Self {
            plan,
            max_rows: rows,
            max_seqs: seqs,
        })
    }

    /// Selects the shape of the next passes: `seqs` sequences of `q_len` rows each, sampling
    /// `samples` rows, with attention split-K planned for contexts up to `max_ctx` tokens.
    pub fn shape(
        &mut self,
        seqs: usize,
        q_len: usize,
        samples: usize,
        max_ctx: usize,
    ) -> Result<(), GpuTextError> {
        let rows = seqs * q_len;
        if seqs == 0 || q_len == 0 || rows > self.max_rows || seqs > self.max_seqs {
            return Err(input_err(format!(
                "a step of {seqs} sequences at {q_len} rows exceeds the plan ({} rows, {} sequences)",
                self.max_rows, self.max_seqs
            )));
        }
        self.plan.rows = rows;
        self.plan.decode_rows = rows;
        self.plan.decode_seqs = seqs;
        self.plan.decode_q_len = q_len;
        self.plan.sample_count = samples;
        self.plan.max_ctx = max_ctx.max(1);
        Ok(())
    }

    /// The block-table buffer, `[max_seqs, max_blocks]`.
    pub fn block_tables(&self) -> &CudaSlice<i32> {
        &self.plan.block_tables
    }

    /// The block-table buffer for host writes.
    pub fn block_tables_mut(&mut self) -> &mut CudaSlice<i32> {
        &mut self.plan.block_tables
    }

    /// Entries per block-table row.
    pub fn max_blocks(&self) -> usize {
        self.plan.max_blocks
    }

    /// The block tables together with the row tables a prepare kernel fills.
    pub fn tables_and_rows(&mut self) -> (&CudaSlice<i32>, PassRows<'_>) {
        (
            &self.plan.block_tables,
            PassRows {
                ids: &mut self.plan.ids,
                positions: &mut self.plan.positions,
                slots: &mut self.plan.slots,
                seq_lens: &mut self.plan.seq_lens,
                sample_rows: &mut self.plan.sample_rows,
            },
        )
    }

    /// The row tables a prepare kernel fills.
    pub fn rows_mut(&mut self) -> PassRows<'_> {
        PassRows {
            ids: &mut self.plan.ids,
            positions: &mut self.plan.positions,
            slots: &mut self.plan.slots,
            seq_lens: &mut self.plan.seq_lens,
            sample_rows: &mut self.plan.sample_rows,
        }
    }
}

impl TextExecutor {
    /// Grows the activation scratch and the split-K workspace to cover every `(seqs, q_len)`
    /// shape in `shapes` at contexts up to `max_ctx`, so later passes never reallocate.
    pub fn reserve_resident(
        &mut self,
        shapes: &[(usize, usize)],
        max_ctx: usize,
    ) -> Result<(), GpuTextError> {
        let g = self.geometry;
        let rows = shapes.iter().map(|&(s, q)| s * q).max().unwrap_or(1);
        self.work.ensure(&self.stream, &g, rows, rows)?;
        let geo = DecodeGeometry {
            q_heads: g.heads,
            kv_heads: g.kv_heads,
            block_size: 1,
        };
        let launches: Vec<(usize, usize, usize)> = shapes
            .iter()
            .map(|&(seqs, q_len)| {
                let (splits, _) =
                    PagedDecodeAttention::plan_splits(&geo, seqs, max_ctx, self.sm_count());
                (seqs, splits, q_len)
            })
            .collect();
        let need = DecodeCapacity::covering(launches, g.heads, self.workspace_pair_mode());
        let held = self.decode_ws_capacity();
        if !held.holds(&need) {
            let ws = DecodeWorkspace::with_capacity(
                &self.stream,
                DecodeCapacity {
                    rows: need.rows.max(held.rows),
                    order_pairs: need.order_pairs.max(held.order_pairs),
                },
            )?;
            self.set_decode_ws(ws);
        }
        Ok(())
    }

    /// Main-model pass over the rows of `plan`: embeds `plan.ids`, runs every layer and writes
    /// the logits of the plan's sampled rows: fp32 into `logits` when given, otherwise left as
    /// bf16 in [`TextExecutor::logits_bf16`]. When `hidden` is given it receives the post-norm
    /// hidden state of every row.
    pub fn forward_resident(
        &mut self,
        weights: &GpuDecoderWeights,
        kv: &mut GpuKvPool,
        plan: &StepPlan,
        logits: Option<&mut CudaSlice<f32>>,
        hidden: Option<&mut CudaSlice<u16>>,
    ) -> Result<(), GpuTextError> {
        let g = self.geometry;
        let plan = &plan.plan;
        self.gemm
            .set_shape_phase(super::forward::gemm_phase(plan, false));
        self.scatter.embedding_gather(
            &self.stream,
            &weights.embed_tokens,
            &plan.ids,
            &mut self.work.h,
            plan.rows,
            g.hidden,
            g.vocab,
        )?;
        self.norm.rms_norm(
            &self.stream,
            &self.work.h,
            &weights.layers[0].input_norm,
            &mut self.work.hn,
            plan.rows,
            g.hidden,
            g.eps,
        )?;
        self.run_stack(&weights.layers, kv.layers_mut(), &weights.final_norm, plan)?;
        self.finish_resident(
            plan,
            &weights.lm_head,
            HiddenSource::Normed,
            logits,
            None,
            hidden,
        )
    }

    /// MTP layer pass over the rows of `plan`, pairing each row's token embedding with the
    /// matching row of `hidden`. Writes draft logits of the sampled rows as for
    /// [`Self::forward_resident`]; when `carry` is given it receives the layer output at each
    /// sampled row. When `tokens` is given (requires [`Self::fused_draft_head`]) the head's
    /// product is reduced on the device to the greedy token of each sampled row, written to
    /// `tokens`, and no logits are produced.
    #[allow(clippy::too_many_arguments)]
    pub fn mtp_forward_resident(
        &mut self,
        mtp: &GpuMtpWeights,
        kv: &mut GpuKvPool,
        plan: &StepPlan,
        hidden: &CudaSlice<u16>,
        logits: Option<&mut CudaSlice<f32>>,
        tokens: Option<&mut CudaSlice<u32>>,
        carry: Option<&mut CudaSlice<u16>>,
    ) -> Result<(), GpuTextError> {
        let g = self.geometry;
        let plan = &plan.plan;
        self.gemm
            .set_shape_phase(super::forward::gemm_phase(plan, true));
        let rows = plan.rows;
        match MtpPrologueRoute::select(self.fused_mtp_prologue) {
            MtpPrologueRoute::Fused => self.mtp_prologue.launch(
                &self.stream,
                &mtp.embed_tokens,
                &plan.ids,
                hidden,
                &mtp.enorm,
                &mtp.hnorm,
                &mut self.work.cat,
                rows,
                g.hidden,
                g.vocab,
                g.eps,
            )?,
            MtpPrologueRoute::Reference => {
                self.scatter.embedding_gather(
                    &self.stream,
                    &mtp.embed_tokens,
                    &plan.ids,
                    &mut self.work.t1,
                    rows,
                    g.hidden,
                    g.vocab,
                )?;
                self.norm.rms_norm(
                    &self.stream,
                    &self.work.t1,
                    &mtp.enorm,
                    &mut self.work.t2,
                    rows,
                    g.hidden,
                    g.eps,
                )?;
                self.ops.copy_rows(
                    &self.stream,
                    &self.work.t2,
                    0,
                    g.hidden,
                    &mut self.work.cat,
                    0,
                    2 * g.hidden,
                    rows,
                    g.hidden,
                )?;
                self.norm.rms_norm(
                    &self.stream,
                    hidden,
                    &mtp.hnorm,
                    &mut self.work.t1,
                    rows,
                    g.hidden,
                    g.eps,
                )?;
                self.ops.copy_rows(
                    &self.stream,
                    &self.work.t1,
                    0,
                    g.hidden,
                    &mut self.work.cat,
                    g.hidden,
                    2 * g.hidden,
                    rows,
                    g.hidden,
                )?;
            }
        }
        super::forward::gemm_bf16(
            &mut self.gemm,
            &self.stream,
            &self.work.cat,
            &mtp.eh_proj,
            &mut self.work.h,
            rows,
            g.hidden,
            2 * g.hidden,
        )?;
        self.norm.rms_norm(
            &self.stream,
            &self.work.h,
            &mtp.layer.input_norm,
            &mut self.work.hn,
            rows,
            g.hidden,
            g.eps,
        )?;
        self.run_stack(
            std::slice::from_ref(&mtp.layer),
            kv.layers_mut(),
            &mtp.shared_norm,
            plan,
        )?;
        self.finish_resident(
            plan,
            &mtp.shared_head,
            HiddenSource::Residual,
            logits,
            tokens,
            carry,
        )
    }

    /// Enables (`true`) or disables (`false`, the default) the single-kernel MTP prologue of
    /// [`Self::mtp_forward_resident`], which writes both halves of the `eh_proj` input directly
    /// instead of running the embedding gather, two RMSNorms and two row copies.
    pub fn set_fused_mtp_prologue(&mut self, on: bool) {
        self.fused_mtp_prologue = on;
    }

    /// Enables the wide fused draft head for 65..=128 sampled rows (disabled by default, where
    /// those rows materialise bf16 logits instead).
    pub fn set_wide_draft_head(&mut self, on: bool) {
        self.gemm.set_wide_argmax_enabled(on);
    }

    /// Whether the draft head of `rows` sampled rows can be reduced to token indices on the
    /// device without materialising logits ([`Self::mtp_forward_resident`] with `tokens`).
    pub fn fused_draft_head(&self, rows: usize) -> bool {
        let g = self.geometry;
        self.gemm.supports_argmax(rows, g.vocab, g.hidden)
    }

    /// Gathers the sampled rows, computes their logits (widened into `logits` when given), and copies the
    /// matching rows (or every row, for the normed source) of `source` into `rows_out`.
    fn finish_resident(
        &mut self,
        plan: &DevicePlan,
        head: &CudaSlice<u16>,
        source: HiddenSource,
        logits: Option<&mut CudaSlice<f32>>,
        tokens: Option<&mut CudaSlice<u32>>,
        rows_out: Option<&mut CudaSlice<u16>>,
    ) -> Result<(), GpuTextError> {
        let g = self.geometry;
        let s = plan.sample_count;
        gather_rows(
            &self.scatter,
            &self.stream,
            &self.work.hn,
            &plan.sample_rows,
            &mut self.work.sh,
            s,
            plan.rows,
            g.hidden,
        )?;
        if let Some(tokens) = tokens {
            lm_head_argmax(
                &mut self.gemm,
                &self.stream,
                &self.work.sh,
                s,
                head,
                g.vocab,
                g.hidden,
                tokens,
            )?;
        } else {
            lm_head_bf16(
                &mut self.gemm,
                &self.stream,
                &self.work.sh,
                s,
                head,
                g.vocab,
                g.hidden,
                &mut self.work.lg,
            )?;
        }
        if let Some(logits) = logits {
            self.ops
                .bf16_to_f32(&self.stream, &self.work.lg, logits, s * g.vocab)?;
        }
        if let Some(out) = rows_out {
            match source {
                HiddenSource::Normed => self.ops.copy_rows(
                    &self.stream,
                    &self.work.hn,
                    0,
                    g.hidden,
                    out,
                    0,
                    g.hidden,
                    plan.rows,
                    g.hidden,
                )?,
                HiddenSource::Residual => gather_rows(
                    &self.scatter,
                    &self.stream,
                    &self.work.h,
                    &plan.sample_rows,
                    out,
                    s,
                    plan.rows,
                    g.hidden,
                )?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod prologue_route_tests {
    use super::*;

    #[test]
    fn prologue_is_reference_unless_selected() {
        assert_eq!(MtpPrologueRoute::select(false), MtpPrologueRoute::Reference);
        assert_eq!(MtpPrologueRoute::select(true), MtpPrologueRoute::Fused);
    }
}
