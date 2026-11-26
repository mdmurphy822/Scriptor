//! Pre-step block reservation under the demand policy: running sequences grow in rank order
//! and a shortfall evicts a sequence of higher rank than the requester.
//!
//! Progress invariant: the lowest-rank active sequence never fails for want of blocks. Every
//! block held by a higher-rank sequence is either reclaimable or bounded:
//! - a running restartable sequence is restarted from its prompt;
//! - a sequence in the vision or prefill phase that owns no image in the vision flight is
//!   returned to the waiting queue with its embeddings and their budget charge intact;
//! - a sequence with an image in the vision flight keeps its blocks only until the flight is
//!   collected, which needs no blocks, and the requester waits for that;
//! - the running non-restartable sequences together never hold more than their reservations,
//!   and admission keeps those reservations plus one full context within the pool whenever
//!   any other sequence is active (see `Scheduler::admit_demand`).
//!
//! A requester that has to wait is never blocked on a holder that itself waits for the
//! requester, so the embedding budget, which is charged to waiting sequences as well, cannot
//! close a cycle.

use crate::sampling::FinishReason;
use crate::scheduler::{ModelExecutor, Scheduler};
use crate::sequence::{Phase, RequestId};
use scriptor_spec::KvPolicy;

impl<E: ModelExecutor> Scheduler<E> {
    /// Whether KV blocks are committed on demand.
    pub(crate) fn demand_policy(&self) -> bool {
        self.kv_policy() == KvPolicy::DemandDiagnostic
    }

    /// Grows every running sequence not in `skip` to cover its next decode step, oldest rank
    /// first. When the free blocks fall short, the running restartable sequence of highest
    /// rank above the requester is evicted and requeued for a restart; the lowest rank is
    /// never evicted. A requester that cannot grow and has no victim waits for a later step,
    /// except the lowest rank, which fails. Returns `skip` extended with the sequences that
    /// must not decode this step. Must run with no decode step in flight.
    pub(crate) fn reserve_demand(
        &mut self,
        skip: &[RequestId],
        ahead: usize,
        depth: usize,
    ) -> Vec<RequestId> {
        debug_assert!(!self.decode_in_flight, "eviction runs between decode steps");
        let mut held_back = skip.to_vec();
        self.last_step_growth = self
            .active
            .values()
            .filter(|s| s.phase == Phase::Running && !skip.contains(&s.id))
            .map(|s| self.step_growth_blocks(s.id, s.len(), ahead))
            .sum();
        let num_blocks = self.pool.layout().num_blocks;
        let order: Vec<RequestId> = self
            .pool
            .sequences_by_age()
            .into_iter()
            .filter(|id| {
                !skip.contains(id)
                    && self
                        .active
                        .get(id)
                        .is_some_and(|s| s.phase == Phase::Running)
            })
            .collect();
        for id in order {
            let Some(seq) = self.active.get(&id) else {
                continue;
            };
            let rank = seq.rank;
            let mut len = seq.len() + depth;
            if self.pool.layout().blocks_for_tokens(len) > num_blocks {
                self.finish(id, FinishReason::Length);
                continue;
            }
            if self.pool.layout().blocks_for_tokens(seq.len() + ahead) <= num_blocks {
                len = seq.len() + ahead;
            }
            loop {
                let needed = match self.pool.blocks_needed(id, len) {
                    Ok(n) => n,
                    Err(e) => {
                        self.fail(id, e.into());
                        break;
                    }
                };
                if self.pool.free_blocks() >= needed {
                    if let Err(e) = self.pool.grow_to(id, len) {
                        self.fail(id, e.into());
                    }
                    break;
                }
                if let Some(victim) = self.eviction_victim(rank) {
                    let _ = self.pool.release(victim);
                    if self.active[&victim].phase == Phase::Running {
                        self.requeue_restart(victim);
                    } else {
                        self.requeue_unstarted(victim);
                    }
                    continue;
                }
                let lowest = self.active.values().all(|s| s.rank >= rank);
                if lowest && !self.holds_vision_flight(rank) {
                    let free = self.pool.free_blocks();
                    self.fail(
                        id,
                        crate::error::EngineError::Cache(format!(
                            "no block can be freed for the oldest request: {needed} needed, {free} free"
                        )),
                    );
                } else {
                    held_back.push(id);
                }
                break;
            }
        }
        held_back
    }

    /// The sequence of highest rank above `requester` whose blocks may be reclaimed now: a
    /// running restartable one, or one in the vision or prefill phase owning no image in the
    /// vision flight.
    fn eviction_victim(&self, requester: u64) -> Option<RequestId> {
        self.active
            .values()
            .filter(|s| s.rank > requester)
            .filter(|s| match s.phase {
                Phase::Running => s.restartable,
                Phase::Vision | Phase::Prefill => !self.owns_flight_image(s.id),
            })
            .max_by_key(|s| s.rank)
            .map(|s| s.id)
    }

    /// Whether an image of the sequence is in the vision flight.
    fn owns_flight_image(&self, id: RequestId) -> bool {
        self.vision_flight
            .as_ref()
            .is_some_and(|f| f.jobs().iter().any(|j| j.seq == id))
    }

    /// Whether a sequence of rank above `rank` owns an image in the vision flight, so its
    /// blocks free up once the flight is collected.
    fn holds_vision_flight(&self, rank: u64) -> bool {
        self.active
            .values()
            .any(|s| s.rank > rank && self.owns_flight_image(s.id))
    }
}
