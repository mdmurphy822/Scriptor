//! Vision stage: resolves image embeddings from the cache and batches the rest up to the
//! patch budget.

use crate::error::EngineError;
use crate::scheduler::{ModelExecutor, Scheduler, VisionJob, VisionToken};
use crate::sequence::{EngineEvent, Phase, RequestId, Sequence};
use scriptor_cache::DeviceBuffer;
use std::sync::Arc;
use std::time::Instant;

/// Cache payload sharing one embedding buffer between the cache and running sequences.
pub struct SharedEmbedding<B>(pub Arc<B>);

impl<B: DeviceBuffer> DeviceBuffer for SharedEmbedding<B> {
    fn byte_len(&self) -> usize {
        self.0.byte_len()
    }
}

/// Number of leading items (in order) whose patch counts fit within `budget`. The first item is
/// always taken, so an image as large as the budget runs alone and nothing is skipped over.
pub fn pack_prefix(patches: &[usize], budget: usize) -> usize {
    let mut total = 0usize;
    let mut taken = 0usize;
    for &p in patches {
        if taken > 0 && total + p > budget {
            break;
        }
        total += p;
        taken += 1;
    }
    taken
}

/// Output-tile edge of the tower's dense matrix multiplications, in rows and columns.
const GEMM_TILE: usize = 128;

/// Waves of tiles a batch must fill before the tail wave stops dominating the tower's time.
const SATURATION_WAVES: usize = 2;

/// Patches per batch from which the tower's matrix multiplications fill the device: the
/// narrowest tower projection (`narrowest_cols` outputs) yields `ceil(cols / tile)` tiles per
/// row block, so `ceil(sm_count / that)` row blocks occupy one wave of `sm_count` SMs, and
/// [`SATURATION_WAVES`] waves keep the partially filled tail wave below a third of the work.
pub fn saturation_patches(sm_count: usize, narrowest_cols: usize) -> usize {
    let tiles_wide = narrowest_cols.div_ceil(GEMM_TILE).max(1);
    SATURATION_WAVES * sm_count.div_ceil(tiles_wide) * GEMM_TILE
}

/// Whether a vision batch of `patches` runs now. It runs when it reaches `saturation` (capped
/// at the batch `budget`), when no prefill or decode work is available to run instead, when
/// no step has been timed yet, or once the oldest pending image has waited as long as the last
/// prefill/decode step took.
pub fn vision_ready(
    patches: usize,
    saturation: usize,
    budget: usize,
    other_work: bool,
    waited: Option<std::time::Duration>,
    last_step: std::time::Duration,
) -> bool {
    if !other_work || patches >= saturation.min(budget) || last_step.is_zero() {
        return true;
    }
    waited.is_some_and(|w| w >= last_step)
}

/// One asynchronous vision batch the executor is working on.
pub(crate) struct VisionFlight {
    token: VisionToken,
    jobs: Vec<VisionJob>,
    started: Instant,
    l1: crate::l1_counters::FlightId,
}

impl VisionFlight {
    /// The jobs of the batch the executor is working on.
    pub(crate) fn jobs(&self) -> &[VisionJob] {
        &self.jobs
    }
}

/// An unresolved image: owning request, index within it, patches.
type Pending = (RequestId, usize, usize);

impl<E: ModelExecutor> Scheduler<E> {
    fn overlap_enabled(&self) -> bool {
        self.config.vision_overlap && self.executor.supports_async_vision()
    }

    /// Advances the vision stage: with overlap, applies a finished batch and starts the next
    /// one without waiting; otherwise embeds one batch to completion.
    pub(crate) fn run_vision(&mut self) {
        self.reserve_embeddings();
        if self.overlap_enabled() {
            self.poll_vision(false);
            if self.vision_flight.is_none() {
                self.begin_vision();
            }
            return;
        }
        let Some(jobs) = self.next_vision_batch() else {
            return;
        };
        let started = Instant::now();
        let result = self.executor.vision(&jobs);
        let elapsed = started.elapsed();
        self.counters.vision_host_seconds += elapsed.as_secs_f64();
        self.finish_vision(&jobs, result, elapsed);
    }

    /// Applies the in-flight batch when it has finished, or after waiting for it when `block`.
    pub(crate) fn poll_vision(&mut self, block: bool) {
        let Some(flight) = self.vision_flight.as_ref() else {
            return;
        };
        let token = flight.token;
        let flight_l1 = flight.l1;
        if !block {
            match self.executor.vision_ready(token) {
                Ok(false) => {
                    self.l1.on_poll(flight_l1, false);
                    return;
                }
                Ok(true) => self.l1.on_poll(flight_l1, true),
                Err(e) => {
                    if let Some(f) = self.vision_flight.take() {
                        self.fail_jobs(&f.jobs, e);
                    }
                    return;
                }
            }
        }
        let Some(flight) = self.vision_flight.take() else {
            return;
        };
        let waited = Instant::now();
        let result = self.executor.vision_finish(flight.token);
        self.counters.vision_host_seconds += waited.elapsed().as_secs_f64();
        let observed = self.l1.enabled().then(|| {
            let now = self.l1.now();
            (now, self.executor.take_vision_timing(flight.token))
        });
        self.finish_vision(&flight.jobs, result, flight.started.elapsed());
        if let Some((now, timing)) = observed {
            let pending = self.pending_images().len();
            self.l1.on_flight_observed(flight.l1, now, timing, pending);
        }
    }

    fn begin_vision(&mut self) {
        let Some(jobs) = self.next_vision_batch() else {
            return;
        };
        let launched = Instant::now();
        let l1_begin = self.l1.now();
        let begun = self.executor.vision_begin(&jobs);
        self.counters.vision_host_seconds += launched.elapsed().as_secs_f64();
        match begun {
            Ok(token) => {
                let l1 = if self.l1.enabled() {
                    let returned = self.l1.now();
                    self.l1.on_flight_begin(l1_begin, returned, &jobs)
                } else {
                    0
                };
                self.vision_flight = Some(VisionFlight {
                    token,
                    jobs,
                    started: Instant::now(),
                    l1,
                });
            }
            Err(e) => self.fail_jobs(&jobs, e),
        }
    }

    /// Gathers the unresolved images of running sequences and then of waiting ones, in order.
    /// Waiting sequences' images are embedded ahead of admission only while the embeddings
    /// already held ahead plus the new ones fit the embedding cache budget.
    fn pending_images(&self) -> Vec<Pending> {
        let mut pending: Vec<Pending> = Vec::new();
        for (id, seq) in &self.active {
            if seq.phase == Phase::Vision && self.embedding_admitted(*id) {
                collect_pending(*id, seq, &mut pending);
            }
        }
        let group = self.config.spatial_merge * self.config.spatial_merge;
        let row_bytes = (self.config.embedding_hidden * self.config.embedding_elem_bytes).max(1);
        let row_cap = self.config.embedding_cache_bytes / row_bytes;
        let mut ahead_rows: usize = self
            .waiting
            .iter()
            .flat_map(|s| s.embeds.iter().zip(&s.images))
            .filter(|(e, _)| e.is_some())
            .map(|(_, img)| img.patches() / group)
            .sum();
        for seq in &self.waiting {
            if !self.embedding_admitted(seq.id) {
                continue;
            }
            let mut mine: Vec<Pending> = Vec::new();
            collect_pending(seq.id, seq, &mut mine);
            let rows: usize = mine.iter().map(|p| p.2 / group).sum();
            if ahead_rows + rows > row_cap {
                break;
            }
            ahead_rows += rows;
            pending.extend(mine);
        }
        pending
    }

    /// The jobs for the leading pending images that fit the patch budget.
    fn jobs_for(&self, pending: &[Pending], take: usize) -> Vec<VisionJob> {
        pending[..take]
            .iter()
            .map(|&(id, i, patches)| {
                let img = &self.seq_ref(id).expect("pending ids exist").images[i];
                VisionJob {
                    seq: id,
                    image: i,
                    key: img.key,
                    grid: img.grid,
                    patches,
                    payload: Arc::clone(&img.payload),
                }
            })
            .collect()
    }

    /// Has the executor build the host inputs of the batch the next vision stage would take,
    /// unless that batch was already prepared. Makes no scheduling decision.
    pub(crate) fn stage_next_vision(&mut self) {
        let pending = self.pending_images();
        let sizes: Vec<usize> = pending.iter().map(|p| p.2).collect();
        let take = pack_prefix(&sizes, self.config.vision_batch_patches);
        let key: Vec<(RequestId, usize)> = pending[..take].iter().map(|p| (p.0, p.1)).collect();
        if key.is_empty() || key == self.staged_vision {
            return;
        }
        let jobs = self.jobs_for(&pending, take);
        if self.executor.vision_prepare(&jobs).is_ok() {
            self.staged_vision = key;
        }
    }

    /// Gathers the next vision batch from the pending images and records the batch size.
    fn next_vision_batch(&mut self) -> Option<Vec<VisionJob>> {
        let pending = self.pending_images();
        if pending.is_empty() {
            self.vision_deferred_since = None;
            return None;
        }
        let sizes: Vec<usize> = pending.iter().map(|p| p.2).collect();
        let take = pack_prefix(&sizes, self.config.vision_batch_patches);
        let batch_patches: usize = sizes[..take].iter().sum();
        let held = self.prefill_deferred_since.is_some();
        let other_work = self.active.values().any(|s| match s.phase {
            Phase::Vision => false,
            Phase::Prefill => !held,
            _ => true,
        });
        let ready = vision_ready(
            batch_patches,
            self.executor.vision_saturation_patches(),
            self.config.vision_batch_patches,
            other_work,
            self.vision_deferred_since.map(|t| t.elapsed()),
            self.last_step_work,
        );
        if !ready {
            self.vision_deferred_since.get_or_insert_with(Instant::now);
            return None;
        }
        self.vision_deferred_since = None;
        let jobs = self.jobs_for(&pending, take);
        self.staged_vision.clear();
        self.counters.vision_batches += 1;
        *self.vision_sizes.entry(jobs.len() as u64).or_insert(0) += 1;
        *self
            .vision_patch_sizes
            .entry(batch_patches.max(1).next_power_of_two() as u64)
            .or_insert(0) += 1;
        Some(jobs)
    }

    fn seq_ref(&self, id: RequestId) -> Option<&Sequence<E::Embedding>> {
        self.active
            .get(&id)
            .or_else(|| self.waiting.iter().find(|s| s.id == id))
    }

    fn seq_mut(&mut self, id: RequestId) -> Option<&mut Sequence<E::Embedding>> {
        if self.active.contains_key(&id) {
            return self.active.get_mut(&id);
        }
        self.waiting.iter_mut().find(|s| s.id == id)
    }

    /// Fails each request owning a job, whether running or waiting.
    fn fail_jobs(&mut self, jobs: &[VisionJob], err: EngineError) {
        let mut seen: Vec<RequestId> = Vec::new();
        for job in jobs {
            if seen.contains(&job.seq) {
                continue;
            }
            seen.push(job.seq);
            self.fail_anywhere(job.seq, err.clone());
        }
    }

    fn fail_anywhere(&mut self, id: RequestId, err: EngineError) {
        if self.active.contains_key(&id) {
            self.fail(id, err);
        } else if let Some(pos) = self.waiting.iter().position(|s| s.id == id) {
            let seq = self.waiting.remove(pos).expect("position is valid");
            self.note_fatal(&err);
            self.counters.failed += 1;
            let _ = seq.tx.send(EngineEvent::Failed(err));
        }
    }

    /// Stores a finished batch's embeddings on their sequences and the cache.
    fn finish_vision(
        &mut self,
        jobs: &[VisionJob],
        result: Result<Vec<E::Embedding>, EngineError>,
        elapsed: std::time::Duration,
    ) {
        self.counters.vision_seconds += elapsed.as_secs_f64();
        let outputs = match result {
            Ok(o) if o.len() == jobs.len() => o,
            Ok(o) => {
                let err = EngineError::Executor(format!(
                    "vision returned {} embeddings for {} images",
                    o.len(),
                    jobs.len()
                ));
                self.fail_jobs(jobs, err);
                return;
            }
            Err(e) => {
                self.fail_jobs(jobs, e);
                return;
            }
        };
        self.counters.vision_images += jobs.len() as u64;
        self.counters.vision_patches += jobs.iter().map(|j| j.patches as u64).sum::<u64>();
        let group = self.config.spatial_merge * self.config.spatial_merge;
        let mut failed: Vec<(RequestId, EngineError)> = Vec::new();
        for (job, buffer) in jobs.iter().zip(outputs) {
            let Some(seq) = self.seq_mut(job.seq) else {
                continue;
            };
            seq.timings.vision += elapsed;
            let rows = job.patches / group;
            let shared = Arc::new(buffer);
            let inserted =
                self.embedding_cache
                    .insert(job.key, rows, SharedEmbedding(Arc::clone(&shared)));
            match inserted {
                Ok(_) => {
                    if let Some(seq) = self.seq_mut(job.seq) {
                        seq.embeds[job.image] = Some(shared);
                    }
                }
                Err(e) => failed.push((job.seq, e.into())),
            }
        }
        for (id, err) in failed {
            self.fail_anywhere(id, err);
        }
        for seq in self.active.values_mut() {
            if seq.phase == Phase::Vision && seq.embeddings_ready() {
                seq.phase = Phase::Prefill;
            }
        }
    }
}

fn collect_pending<B>(id: RequestId, seq: &Sequence<B>, out: &mut Vec<Pending>) {
    for (i, e) in seq.embeds.iter().enumerate() {
        if e.is_none() {
            out.push((id, i, seq.images[i].patches()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const MS: Duration = Duration::from_millis(1);

    #[test]
    fn saturation_fills_two_waves_of_the_narrowest_projection() {
        assert_eq!(saturation_patches(48, 1024), 2 * 6 * 128);
        assert_eq!(saturation_patches(1, 128), 2 * 128);
        assert_eq!(saturation_patches(48, 1), 2 * 48 * 128);
    }

    #[test]
    fn a_batch_never_waits_without_other_work() {
        assert!(vision_ready(1, 1536, 8192, false, None, 30 * MS));
    }

    #[test]
    fn a_saturated_batch_runs_and_the_saturation_is_capped_at_the_budget() {
        assert!(vision_ready(1536, 1536, 8192, true, None, 30 * MS));
        assert!(vision_ready(512, 1536, 512, true, None, 30 * MS));
    }

    #[test]
    fn an_unsaturated_batch_waits_one_measured_step_at_most() {
        assert!(!vision_ready(100, 1536, 8192, true, None, 30 * MS));
        assert!(!vision_ready(100, 1536, 8192, true, Some(10 * MS), 30 * MS));
        assert!(vision_ready(100, 1536, 8192, true, Some(30 * MS), 30 * MS));
    }

    #[test]
    fn a_batch_runs_before_any_step_was_timed_and_without_a_saturation() {
        assert!(vision_ready(100, 1536, 8192, true, None, Duration::ZERO));
        assert!(vision_ready(100, 0, 8192, true, None, 30 * MS));
    }
}
