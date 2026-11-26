//! Decode stage: reserves KV for one more token per running sequence (evicting the youngest
//! when the pool is empty) and runs one executor decode step.

use crate::error::EngineError;
use crate::sampling::FinishReason;
use crate::scheduler::{DecodeRow, DeviceRow, ModelExecutor, Scheduler};
use crate::sequence::{Phase, RequestId};
use scriptor_cache::ReserveOutcome;
use std::time::Instant;

/// Buffers one decode step reuses so its host-side preparation allocates nothing once the
/// largest batch has been seen.
#[derive(Default)]
pub(crate) struct DecodeScratch {
    /// Working storage of the age ordering.
    keyed: Vec<(u64, RequestId)>,
    /// Sequences to reserve KV for, read before reservation.
    order: Vec<RequestId>,
    /// Sequences to decode, read after reservation.
    ids: Vec<RequestId>,
    /// Rows of the sequences in `ids`, in the same order.
    rows: Vec<DecodeRow>,
    device_ids: Vec<RequestId>,
    host_ids: Vec<RequestId>,
    device_rows: Vec<DecodeRow>,
    host_rows: Vec<DecodeRow>,
    /// Rows whose block-table storage is recycled for later steps.
    spare: Vec<DecodeRow>,
}

impl<E: ModelExecutor> Scheduler<E> {
    /// Advances every running sequence not in `skip` by one token; returns the rows decoded.
    pub(crate) fn run_decode(&mut self, skip: &[RequestId]) -> usize {
        let mut scratch = std::mem::take(&mut self.decode_scratch);
        let decoded = self.decode_with(skip, &mut scratch);
        self.decode_scratch = scratch;
        decoded
    }

    /// Running sequences not in `skip`, oldest admission first, written into `out`.
    fn running_by_age(
        &self,
        skip: &[RequestId],
        keyed: &mut Vec<(u64, RequestId)>,
        out: &mut Vec<RequestId>,
    ) {
        self.pool.sequences_by_age_into(keyed, out);
        out.retain(|id| {
            !skip.contains(id)
                && self
                    .active
                    .get(id)
                    .is_some_and(|s| s.phase == Phase::Running)
        });
    }

    /// Body of [`Self::run_decode`] over the reusable buffers in `scratch`. The sequence list
    /// is read twice: before reservation, to pick the sequences to reserve for, and after it,
    /// because reservation can evict or requeue sequences.
    fn decode_with(&mut self, skip: &[RequestId], scratch: &mut DecodeScratch) -> usize {
        let prep_started = Instant::now();
        let depth = self.executor.draft_depth();
        let ahead = self.executor.lookahead_tokens().max(depth);
        #[cfg(feature = "kv-demand-diagnostic")]
        let demand_skip;
        #[cfg(feature = "kv-demand-diagnostic")]
        let (skip, demand) = if self.demand_policy() {
            demand_skip = self.reserve_demand(skip, ahead, depth);
            (&demand_skip[..], true)
        } else {
            (skip, false)
        };
        #[cfg(not(feature = "kv-demand-diagnostic"))]
        let demand = false;
        scratch.order.clear();
        if !demand {
            self.running_by_age(skip, &mut scratch.keyed, &mut scratch.order);
        }
        for &id in &scratch.order {
            let Some(seq) = self.active.get(&id) else {
                continue;
            };
            let mut len = seq.len() + depth;
            if self.pool.layout().blocks_for_tokens(len) > self.pool.layout().num_blocks {
                self.finish(id, FinishReason::Length);
                continue;
            }
            if self.pool.layout().blocks_for_tokens(seq.len() + ahead)
                <= self.pool.layout().num_blocks
            {
                len = seq.len() + ahead;
            }
            match self.pool.reserve(id, len) {
                Ok(ReserveOutcome::Granted(r)) => {
                    for victim in r.preempted {
                        self.requeue_preempted(victim);
                    }
                }
                Ok(ReserveOutcome::SelfPreempted { preempted }) => {
                    for victim in preempted {
                        self.requeue_preempted(victim);
                    }
                }
                Err(e) => self.fail(id, e.into()),
            }
        }
        scratch.ids.clear();
        self.running_by_age(skip, &mut scratch.keyed, &mut scratch.ids);
        if scratch.ids.is_empty() {
            self.counters.decode_prep_seconds += prep_started.elapsed().as_secs_f64();
            return 0;
        }
        for &id in &scratch.ids {
            let seq = &self.active[&id];
            let mut row = scratch.spare.pop().unwrap_or_else(|| DecodeRow {
                seq: 0,
                token: 0,
                position: 0,
                blocks: Vec::new(),
            });
            row.seq = id;
            row.token = seq.last_token();
            row.position = seq.len() - 1;
            row.blocks.clear();
            if let Ok(table) = self.pool.block_table(id) {
                row.blocks.extend_from_slice(table);
            }
            scratch.rows.push(row);
        }
        if self.executor.supports_device_step() {
            scratch.device_ids.clear();
            scratch.host_ids.clear();
            for (id, row) in scratch.ids.iter().zip(scratch.rows.drain(..)) {
                let device = self.active.get(id).is_some_and(|s| self.device_bound(s));
                if device {
                    scratch.device_ids.push(*id);
                    scratch.device_rows.push(row);
                } else {
                    scratch.host_ids.push(*id);
                    scratch.host_rows.push(row);
                }
            }
            self.counters.decode_prep_seconds += prep_started.elapsed().as_secs_f64();
            let mut decoded = 0;
            if !scratch.device_ids.is_empty() {
                decoded += self.run_device_decode(&scratch.device_ids, &scratch.device_rows, depth);
            }
            if !scratch.host_ids.is_empty() {
                decoded += self.run_host_decode(&scratch.host_ids, &scratch.host_rows, depth);
            }
            scratch.spare.append(&mut scratch.device_rows);
            scratch.spare.append(&mut scratch.host_rows);
            decoded
        } else {
            self.counters.decode_prep_seconds += prep_started.elapsed().as_secs_f64();
            let decoded = self.run_host_decode(&scratch.ids, &scratch.rows, depth);
            scratch.spare.append(&mut scratch.rows);
            decoded
        }
    }

    /// One device-resident step: the executor drafts, verifies, accepts and samples on the
    /// device and returns each sequence's emitted tokens, which are delivered in order until a
    /// finish rule ends the sequence.
    fn run_device_decode(&mut self, ids: &[RequestId], rows: &[DecodeRow], depth: usize) -> usize {
        let started = Instant::now();
        let begun = {
            let active = &self.active;
            let device_rows: Vec<DeviceRow<'_>> = ids
                .iter()
                .zip(rows)
                .map(|(id, row)| {
                    let sampler = &active[id].sampler;
                    DeviceRow {
                        row,
                        params: sampler.row_params(0),
                        masks: sampler.masks(),
                    }
                })
                .collect();
            match self.executor.decode_device_begin(&device_rows) {
                Ok(true) => {
                    if self.l1.enabled() {
                        let now = self.l1.now();
                        self.l1
                            .on_launch_return(now, ids.len(), self.vision_flight.is_some());
                    }
                    Ok(None)
                }
                Ok(false) => self.executor.decode_device(&device_rows).map(Some),
                Err(e) => Err(e),
            }
        };
        let result = match begun {
            Ok(Some(done)) => Ok(done),
            Ok(None) => {
                #[cfg(feature = "kv-demand-diagnostic")]
                {
                    self.decode_in_flight = true;
                }
                self.stage_next_vision();
                if self.l1.enabled() {
                    let now = self.l1.now();
                    self.l1.on_collect_enter(now);
                }
                let finished = self.executor.decode_device_finish();
                if self.l1.enabled() {
                    let now = self.l1.now();
                    let sample = self.executor.take_host_sample();
                    self.l1.on_collect_return(now, sample);
                }
                #[cfg(feature = "kv-demand-diagnostic")]
                {
                    self.decode_in_flight = false;
                }
                finished
            }
            Err(e) => Err(e),
        };
        let elapsed = started.elapsed();
        self.counters.decode_steps += 1;
        self.counters.decode_seconds += elapsed.as_secs_f64();
        let outputs = match result {
            Ok(o) if o.len() == ids.len() => o,
            Ok(o) => {
                let err = EngineError::Executor(format!(
                    "device decode returned {} outputs for {} sequences",
                    o.len(),
                    ids.len()
                ));
                self.fail_many(ids.iter().copied(), err);
                return 0;
            }
            Err(e) => {
                self.fail_many(ids.iter().copied(), e);
                return 0;
            }
        };
        self.spec_log.begin_span();
        for (id, out) in ids.iter().zip(&outputs) {
            if out.tokens.is_empty() || out.tokens.len() != out.margins.len() {
                let err = EngineError::Executor(format!(
                    "device decode returned {} tokens and {} margins",
                    out.tokens.len(),
                    out.margins.len()
                ));
                self.fail(*id, err);
                continue;
            }
            if let Some(seq) = self.active.get_mut(id) {
                seq.timings.decode += elapsed;
            }
            let seeded = self.active.get_mut(id).is_some_and(|seq| {
                let pending = seq.device_pending;
                seq.device_pending = false;
                pending
            });
            let mut accepted = 0usize;
            for (j, (&token, &margin)) in out.tokens.iter().zip(&out.margins).enumerate() {
                if !self.active.contains_key(id) {
                    break;
                }
                if self.accept_emitted(*id, token, margin) {
                    break;
                }
                if j + 1 < out.tokens.len() {
                    accepted += 1;
                }
            }
            if depth > 0 && !seeded {
                self.record_spec(depth, accepted);
                self.log_spec(*id, depth, accepted);
            }
        }
        self.flush_spec_log();
        rows.len()
    }

    /// Adds one verification to the speculative counters.
    fn record_spec(&mut self, drafted: usize, accepted: usize) {
        self.counters.spec_drafts += 1;
        self.counters.spec_draft_tokens += drafted as u64;
        self.counters.spec_accepted_tokens += accepted as u64;
        if self.spec_accepted_per_pos.len() < drafted {
            self.spec_accepted_per_pos.resize(drafted, 0);
        }
        for slot in self.spec_accepted_per_pos.iter_mut().take(accepted) {
            *slot += 1;
        }
    }

    /// Prints the log lines of requests that finished during the step just run.
    fn flush_spec_log(&mut self) {
        for line in self.spec_log.end_span() {
            eprintln!("{line}");
        }
    }

    /// Adds one verification to request `id`'s acceptance log.
    fn log_spec(&mut self, id: RequestId, drafted: usize, accepted: usize) {
        let active = &self.active;
        self.spec_log
            .record(id, drafted, accepted, |k| active.contains_key(&k));
    }

    /// One host-sampled step: the executor returns logits and the scheduler samples them.
    fn run_host_decode(&mut self, ids: &[RequestId], rows: &[DecodeRow], depth: usize) -> usize {
        if depth > 0 {
            return self.run_spec_decode(ids, rows);
        }
        let started = Instant::now();
        let result = self.executor.decode(rows);
        let elapsed = started.elapsed();
        self.counters.decode_steps += 1;
        self.counters.decode_seconds += elapsed.as_secs_f64();
        let logits = match result {
            Ok(l) if l.len() == ids.len() => l,
            Ok(l) => {
                let err = EngineError::Executor(format!(
                    "decode returned {} logits rows for {} sequences",
                    l.len(),
                    ids.len()
                ));
                self.fail_many(ids.iter().copied(), err);
                return 0;
            }
            Err(e) => {
                self.fail_many(ids.iter().copied(), e);
                return 0;
            }
        };
        for (id, row) in ids.iter().zip(&logits) {
            if let Some(seq) = self.active.get_mut(id) {
                seq.timings.decode += elapsed;
            }
            self.accept_token(*id, row);
        }
        rows.len()
    }

    /// One speculative step: the executor drafts and verifies, then each sequence takes the
    /// longest run of verifier samples that agree with its drafts plus the first disagreeing one.
    fn run_spec_decode(&mut self, ids: &[RequestId], rows: &[DecodeRow]) -> usize {
        let started = Instant::now();
        let result = self.executor.decode_spec(rows);
        let elapsed = started.elapsed();
        self.counters.decode_steps += 1;
        self.counters.decode_seconds += elapsed.as_secs_f64();
        let outputs = match result {
            Ok(o) if o.len() == ids.len() => o,
            Ok(o) => {
                let err = EngineError::Executor(format!(
                    "speculative decode returned {} outputs for {} sequences",
                    o.len(),
                    ids.len()
                ));
                self.fail_many(ids.iter().copied(), err);
                return 0;
            }
            Err(e) => {
                self.fail_many(ids.iter().copied(), e);
                return 0;
            }
        };
        self.spec_log.begin_span();
        for (id, out) in ids.iter().zip(&outputs) {
            if out.logits.len() != out.drafts.len() + 1 {
                let err = EngineError::Executor(format!(
                    "speculative decode returned {} logits rows for {} drafts",
                    out.logits.len(),
                    out.drafts.len()
                ));
                self.fail(*id, err);
                continue;
            }
            if let Some(seq) = self.active.get_mut(id) {
                seq.timings.decode += elapsed;
            }
            let mut accepted = 0usize;
            for (j, row) in out.logits.iter().enumerate() {
                if !self.active.contains_key(id) {
                    break;
                }
                let (token, finished) = self.accept_sampled(*id, row);
                if finished || out.drafts.get(j) != Some(&token) {
                    break;
                }
                accepted += 1;
            }
            self.record_spec(out.drafts.len(), accepted);
            self.log_spec(*id, out.drafts.len(), accepted);
        }
        self.flush_spec_log();
        rows.len()
    }
}
