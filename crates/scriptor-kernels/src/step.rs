//! Launchers for the row bookkeeping kernels of the device-resident decode step.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, PushKernelArg};
use scriptor_cuda::Result;

use crate::launch::elementwise_launch;
use crate::norm::{load_module, to_config};

const BLOCK: usize = 128;

/// Row index buffers one prepare kernel fills for a forward pass.
pub struct PassRows<'a> {
    pub ids: &'a mut CudaSlice<i32>,
    pub positions: &'a mut CudaSlice<i32>,
    pub slots: &'a mut CudaSlice<i32>,
    pub seq_lens: &'a mut CudaSlice<i32>,
    pub sample_rows: &'a mut CudaSlice<i32>,
}

/// Per-sequence state the prepare kernels read, one entry per batch row.
pub struct RowState<'a> {
    /// Tokens already cached before the step, negative for an inactive row.
    pub cached: &'a CudaSlice<i32>,
    /// Tokens each row's block table backs.
    pub limit: &'a CudaSlice<i32>,
    /// Rotary offset added to a sequence index.
    pub delta: &'a CudaSlice<i32>,
    /// Paged block tables, `[rows, stride]`.
    pub tables: &'a CudaSlice<i32>,
    /// Entries per block-table row.
    pub stride: usize,
    /// Tokens per KV block.
    pub block_size: usize,
}

/// Loaded step bookkeeping kernels.
pub struct StepKernels {
    verify: CudaFunction,
    mtp: CudaFunction,
    chain: CudaFunction,
    store: CudaFunction,
    pack: CudaFunction,
}

impl StepKernels {
    /// Loads the kernels into `ctx`.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
        let module = load_module(ctx, "step_ops")?;
        Ok(Self {
            verify: module.kernel("scriptor_step_verify_prepare")?,
            mtp: module.kernel("scriptor_step_mtp_prepare")?,
            chain: module.kernel("scriptor_step_chain_prepare")?,
            store: module.kernel("scriptor_step_store_drafts")?,
            pack: module.kernel("scriptor_step_pack")?,
        })
    }

    /// Fills the rows of the verification pass: each sequence's last token and its `k` drafts.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_prepare(
        &self,
        stream: &Arc<CudaStream>,
        state: &RowState<'_>,
        last: &CudaSlice<u32>,
        drafts: &CudaSlice<u32>,
        k: usize,
        seqs: usize,
        out: PassRows<'_>,
        mark_slots: &mut CudaSlice<u32>,
    ) -> Result<()> {
        let q = k + 1;
        let rows = (seqs * q) as i32;
        let (stride, bs, qi, ki) = (
            state.stride as i32,
            state.block_size as i32,
            q as i32,
            k as i32,
        );
        // SAFETY: argument order and types match the kernel signature; the caller supplies
        // `seqs` state entries, `seqs * q` row entries and block tables covering every row.
        unsafe {
            stream
                .launch_builder(&self.verify)
                .arg(state.cached)
                .arg(state.limit)
                .arg(last)
                .arg(state.delta)
                .arg(drafts)
                .arg(state.tables)
                .arg(&stride)
                .arg(&bs)
                .arg(&qi)
                .arg(&ki)
                .arg(&rows)
                .arg(out.ids)
                .arg(out.positions)
                .arg(out.slots)
                .arg(out.seq_lens)
                .arg(out.sample_rows)
                .arg(mark_slots)
                .launch(to_config(elementwise_launch(rows as usize, BLOCK)))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// Fills the MTP layer rows that consume the verification hidden states of the step.
    pub fn mtp_prepare(
        &self,
        stream: &Arc<CudaStream>,
        state: &RowState<'_>,
        emitted: &CudaSlice<u32>,
        counts: &CudaSlice<u32>,
        k: usize,
        seqs: usize,
        out: PassRows<'_>,
    ) -> Result<()> {
        let q = k + 1;
        let rows = (seqs * q) as i32;
        let (stride, bs, qi) = (state.stride as i32, state.block_size as i32, q as i32);
        // SAFETY: as in `verify_prepare`; `emitted` is `[seqs, q]` and `counts` `[seqs]`.
        unsafe {
            stream
                .launch_builder(&self.mtp)
                .arg(state.cached)
                .arg(state.limit)
                .arg(state.delta)
                .arg(emitted)
                .arg(counts)
                .arg(state.tables)
                .arg(&stride)
                .arg(&bs)
                .arg(&qi)
                .arg(&rows)
                .arg(out.ids)
                .arg(out.positions)
                .arg(out.slots)
                .arg(out.seq_lens)
                .arg(out.sample_rows)
                .launch(to_config(elementwise_launch(rows as usize, BLOCK)))?;
        }
        Ok(())
    }

    /// Fills the rows of MTP chain pass `pass` (1-based): the previous draft of each sequence.
    #[allow(clippy::too_many_arguments)]
    pub fn chain_prepare(
        &self,
        stream: &Arc<CudaStream>,
        state: &RowState<'_>,
        counts: &CudaSlice<u32>,
        drafts: &CudaSlice<u32>,
        k: usize,
        pass: usize,
        seqs: usize,
        out: PassRows<'_>,
    ) -> Result<()> {
        let (stride, bs, ki, pi, si) = (
            state.stride as i32,
            state.block_size as i32,
            k as i32,
            pass as i32,
            seqs as i32,
        );
        // SAFETY: as in `verify_prepare`; `drafts` is `[seqs, k]`.
        unsafe {
            stream
                .launch_builder(&self.chain)
                .arg(state.cached)
                .arg(state.limit)
                .arg(state.delta)
                .arg(counts)
                .arg(drafts)
                .arg(state.tables)
                .arg(&stride)
                .arg(&bs)
                .arg(&ki)
                .arg(&pi)
                .arg(&si)
                .arg(out.ids)
                .arg(out.positions)
                .arg(out.slots)
                .arg(out.seq_lens)
                .arg(out.sample_rows)
                .launch(to_config(elementwise_launch(seqs, BLOCK)))?;
        }
        Ok(())
    }

    /// Stores `tokens[s]` as draft `index` of sequence `s`.
    pub fn store_drafts(
        &self,
        stream: &Arc<CudaStream>,
        tokens: &CudaSlice<u32>,
        drafts: &mut CudaSlice<u32>,
        k: usize,
        index: usize,
        seqs: usize,
    ) -> Result<()> {
        let (ki, ii, si) = (k as i32, index as i32, seqs as i32);
        // SAFETY: `tokens` holds `seqs` entries and `drafts` `[seqs, k]`.
        unsafe {
            stream
                .launch_builder(&self.store)
                .arg(tokens)
                .arg(drafts)
                .arg(&ki)
                .arg(&ii)
                .arg(&si)
                .launch(to_config(elementwise_launch(seqs, BLOCK)))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// Packs counts, emitted tokens and margins into `result`, `1 + 2q` words per sequence.
    pub fn pack(
        &self,
        stream: &Arc<CudaStream>,
        emitted: &CudaSlice<u32>,
        counts: &CudaSlice<u32>,
        margins: &CudaSlice<f32>,
        seqs: usize,
        q: usize,
        result: &mut CudaSlice<u32>,
    ) -> Result<()> {
        let (si, qi) = (seqs as i32, q as i32);
        // SAFETY: `emitted` and `margins` hold `[seqs, q]`, `counts` `[seqs]` and `result`
        // `seqs * (1 + 2q)` words.
        unsafe {
            stream
                .launch_builder(&self.pack)
                .arg(emitted)
                .arg(counts)
                .arg(margins)
                .arg(&si)
                .arg(&qi)
                .arg(result)
                .launch(to_config(elementwise_launch(seqs * (1 + 2 * q), BLOCK)))?;
        }
        Ok(())
    }
}

/// Words of the packed result for one sequence at `q` rows per sequence.
pub const fn result_width(q: usize) -> usize {
    1 + 2 * q
}
