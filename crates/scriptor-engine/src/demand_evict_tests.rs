//! Demand eviction: rank-ordered pre-step eviction, restart of victims, and the bounded intake.
//! The CPU executor checks scheduler control flow only; it says nothing about device output.

use crate::EngineRequest;
use crate::error::EngineError;
use crate::request::SamplingParams;
use crate::scheduler::{DecodeRow, ModelExecutor, PrefillRow, Scheduler, SchedulerConfig};
use crate::sequence::{EngineEvent, RequestId};
use scriptor_cache::{HostBuffer, KvLayout};
use scriptor_spec::KvPolicy;
use std::collections::HashMap;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

const PROMPT: usize = 8;
const CONTEXT: usize = 160;

/// Generates `100 + (generated % 20)` for every row from the history it holds per sequence;
/// the first prompt id is far above the generation count so only the limits end a request.
#[derive(Default)]
struct History {
    seqs: HashMap<RequestId, Vec<u32>>,
    prefills: usize,
}

fn next_logits(history: &[u32]) -> Vec<f32> {
    let generated = history.iter().filter(|&&t| t >= 100).count();
    let mut l = vec![0.0f32; 128];
    l[100 + generated % 20] = 10.0;
    l
}

impl ModelExecutor for History {
    type Embedding = HostBuffer;

    fn vision(
        &mut self,
        _jobs: &[crate::scheduler::VisionJob],
    ) -> Result<Vec<HostBuffer>, EngineError> {
        Ok(Vec::new())
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        self.prefills += rows.len();
        for r in rows {
            assert!(
                !self.seqs.contains_key(&r.seq),
                "prefill of a live sequence"
            );
            self.seqs.insert(r.seq, r.tokens.clone());
        }
        Ok(rows.iter().map(|r| next_logits(&r.tokens)).collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        let mut out = Vec::new();
        for r in rows {
            let h = self.seqs.get_mut(&r.seq).expect("prefilled");
            assert_eq!(h.len(), r.position, "decode position follows history");
            h.push(r.token);
            out.push(next_logits(h));
        }
        Ok(out)
    }

    fn release(&mut self, seq: RequestId) {
        self.seqs.remove(&seq);
    }
}

fn scheduler(policy: KvPolicy, slots: usize, blocks: usize) -> Scheduler<History> {
    let config = SchedulerConfig {
        max_concurrency: slots,
        max_context: CONTEXT,
        vision_batch_patches: 64,
        vision_overlap: true,
        spatial_merge: 2,
        vocab_size: 128,
        eos_token_ids: vec![0],
        image_token_id: 50,
        embedding_cache_bytes: 1 << 20,
        embedding_hidden: 4,
        embedding_elem_bytes: 2,
        fallback_seed: 1,
        embedding_release_after_prefill: false,
    };
    let layout = KvLayout {
        num_blocks: blocks,
        block_tokens: 16,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    };
    let mut s = Scheduler::new(config, layout, History::default());
    s.set_kv_policy(policy);
    s
}

fn request_of(prompt_len: usize) -> EngineRequest {
    let mut prompt = vec![1u32; prompt_len];
    prompt[0] = 1000;
    EngineRequest {
        prompt,
        images: Vec::new(),
        params: SamplingParams {
            temperature: 0.0,
            max_tokens: Some((CONTEXT - prompt_len) as u32),
            ..SamplingParams::default()
        },
    }
}

/// What one request's channel delivered: the tokens since its last restart, the restarts, and
/// whether a terminal event arrived.
#[derive(Default)]
struct Seen {
    tokens: Vec<u32>,
    restarts: usize,
    finished: Option<usize>,
}

struct Run {
    s: Scheduler<History>,
    rx: Vec<UnboundedReceiver<EngineEvent>>,
    seen: Vec<Seen>,
}

impl Run {
    fn new(policy: KvPolicy, slots: usize, blocks: usize) -> Self {
        Self {
            s: scheduler(policy, slots, blocks),
            rx: Vec::new(),
            seen: Vec::new(),
        }
    }

    fn submit(&mut self, restartable: bool) -> Result<RequestId, EngineError> {
        self.submit_of(PROMPT, restartable)
    }

    fn submit_of(
        &mut self,
        prompt_len: usize,
        restartable: bool,
    ) -> Result<RequestId, EngineError> {
        let (tx, rx) = unbounded_channel();
        let id = self.rx.len() as RequestId;
        self.s
            .submit_restartable_with_id(id, request_of(prompt_len), tx, restartable)?;
        self.rx.push(rx);
        self.seen.push(Seen::default());
        Ok(id)
    }

    fn drain(&mut self) {
        for (rx, seen) in self.rx.iter_mut().zip(&mut self.seen) {
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    EngineEvent::Token(t) => seen.tokens.push(t.token),
                    EngineEvent::Restarted => {
                        seen.tokens.clear();
                        seen.restarts += 1;
                    }
                    EngineEvent::Finished(f) => {
                        assert!(seen.finished.is_none(), "one terminal event");
                        seen.finished = Some(f.completion_tokens);
                    }
                    other => panic!("unexpected event {other:?}"),
                }
            }
        }
    }

    fn run(&mut self) {
        for _ in 0..4000 {
            self.s.step();
            assert!(self.s.pool.stats().used_blocks <= self.s.pool.layout().num_blocks);
            assert!(self.s.accepted() <= self.s.accepted_bound());
            self.drain();
            if self.s.is_idle() {
                break;
            }
        }
        assert!(self.s.is_idle(), "every request terminates");
        assert_eq!(self.s.pool.stats().sequences, 0);
        assert!(self.s.executor().seqs.is_empty(), "executor state released");
    }
}

fn uninterrupted(prompt_len: usize) -> Vec<u32> {
    let mut run = Run::new(KvPolicy::Reserve, 1, 64);
    run.submit_of(prompt_len, false).unwrap();
    run.run();
    run.seen.remove(0).tokens
}

const LENS: [usize; 5] = [8, 40, 12, 30, 9];

#[test]
fn every_request_grows_to_max_tokens_with_a_pool_just_above_one_context() {
    let mut run = Run::new(KvPolicy::DemandDiagnostic, 3, 11);
    for len in LENS {
        run.submit_of(len, true).unwrap();
    }
    run.run();
    for (seen, len) in run.seen.iter().zip(LENS) {
        assert_eq!(seen.finished, Some(CONTEXT - len));
        assert_eq!(seen.tokens, uninterrupted(len));
    }
    assert_eq!(run.seen[0].restarts, 0, "the lowest rank is never evicted");
}

#[test]
fn the_same_victim_is_evicted_repeatedly_and_still_completes() {
    let mut run = Run::new(KvPolicy::DemandDiagnostic, 3, 10);
    for len in &LENS[..4] {
        run.submit_of(*len, true).unwrap();
    }
    run.run();
    assert_eq!(run.seen[0].restarts, 0, "the lowest rank is never evicted");
    assert!(
        run.seen[3].restarts >= 3,
        "victim restarted {} times",
        run.seen[3].restarts
    );
    for (seen, len) in run.seen.iter().zip(LENS) {
        assert_eq!(seen.finished, Some(CONTEXT - len));
        assert_eq!(
            seen.tokens,
            uninterrupted(len),
            "control flow only: CPU executor"
        );
    }
}

#[test]
fn a_request_that_is_not_restartable_is_never_evicted() {
    let mut run = Run::new(KvPolicy::DemandDiagnostic, 2, 24);
    run.submit(false).unwrap();
    run.submit(false).unwrap();
    run.run();
    assert!(run.seen.iter().all(|s| s.restarts == 0));
    assert!(run.seen.iter().all(|s| s.finished.is_some()));
}

#[test]
fn submissions_beyond_twice_the_slots_of_new_requests_are_refused() {
    let mut run = Run::new(KvPolicy::DemandDiagnostic, 2, 24);
    for _ in 0..4 {
        run.submit(true).unwrap();
    }
    assert_eq!(
        run.submit(true).unwrap_err(),
        EngineError::QueueFull {
            retry_after_secs: 1
        }
    );
    run.run();
    assert!(run.seen.iter().all(|s| s.finished.is_some()));
}

#[test]
fn evicted_requests_are_not_counted_against_the_queue_bound() {
    let mut run = Run::new(KvPolicy::DemandDiagnostic, 3, 10);
    for len in &LENS[..4] {
        run.submit_of(*len, true).unwrap();
    }
    let mut saw_victim = false;
    for _ in 0..4000 {
        run.s.step();
        run.drain();
        if run.s.waiting.iter().any(|s| s.admitted_once) {
            saw_victim = true;
            let room = run.s.queue_bound() - run.s.waiting_new();
            for _ in 0..room {
                run.submit(true).unwrap();
            }
            assert!(run.submit(true).is_err());
            break;
        }
    }
    assert!(saw_victim, "an eviction happened");
    run.run();
    assert!(run.seen.iter().all(|s| s.finished.is_some()));
}

#[test]
fn reserve_mode_never_restarts() {
    let mut run = Run::new(KvPolicy::Reserve, 2, 11);
    run.submit(true).unwrap();
    run.submit(true).unwrap();
    run.run();
    assert!(run.seen.iter().all(|s| s.restarts == 0));
}

#[test]
fn non_restartable_requests_never_leave_the_oldest_short_of_blocks() {
    let mut run = Run::new(KvPolicy::DemandDiagnostic, 3, 11);
    run.submit_of(PROMPT, true).unwrap();
    run.submit_of(PROMPT, false).unwrap();
    run.submit_of(PROMPT, false).unwrap();
    run.run();
    for seen in &run.seen {
        assert_eq!(seen.finished, Some(CONTEXT - PROMPT));
        assert_eq!(seen.restarts, 0);
    }
}

#[test]
fn a_non_restartable_request_is_admitted_alone_even_when_it_needs_the_whole_pool() {
    let mut run = Run::new(KvPolicy::DemandDiagnostic, 2, 10);
    run.submit_of(PROMPT, false).unwrap();
    run.run();
    assert_eq!(run.seen[0].finished, Some(CONTEXT - PROMPT));
}

#[test]
fn a_restart_advances_the_restart_counters_and_the_demand_gauges() {
    let mut run = Run::new(KvPolicy::DemandDiagnostic, 3, 10);
    for len in &LENS[..4] {
        run.submit_of(*len, true).unwrap();
    }
    let mut peak_growth = 0;
    for _ in 0..4000 {
        run.s.step();
        run.drain();
        peak_growth = peak_growth.max(run.s.metrics().step_growth_blocks);
        if run.s.is_idle() {
            break;
        }
    }
    let m = run.s.metrics();
    let restarts: usize = run.seen.iter().map(|s| s.restarts).sum();
    assert!(restarts >= 3);
    assert_eq!(m.restarts, restarts as u64);
    assert!(m.restart_discarded_tokens > 0);
    assert_eq!(m.kv_policy, KvPolicy::DemandDiagnostic);
    assert!(peak_growth > 0, "the pre-step growth was recorded");
    assert_eq!(m.accepted, 0);
    assert_eq!(m.rejected_queue, 0);
}

#[test]
fn a_queue_refusal_advances_the_rejected_queue_counter() {
    let mut run = Run::new(KvPolicy::DemandDiagnostic, 2, 24);
    for _ in 0..4 {
        run.submit(true).unwrap();
    }
    assert!(run.submit(true).is_err());
    assert!(run.submit(true).is_err());
    assert_eq!(run.s.metrics().rejected_queue, 2);
    run.run();
}

#[test]
fn admission_blocks_are_counted_by_reason() {
    let mut run = Run::new(KvPolicy::DemandDiagnostic, 1, 24);
    run.submit(true).unwrap();
    run.submit(true).unwrap();
    run.run();
    let b = run.s.metrics().admission_blocked;
    assert!(b.slots > 0, "the second request waited for the only slot");
    assert_eq!(b.embedding, 0);
}

#[test]
fn the_reserve_policy_reports_zero_demand_counters() {
    let mut run = Run::new(KvPolicy::Reserve, 2, 11);
    run.submit(true).unwrap();
    run.submit(true).unwrap();
    run.run();
    let m = run.s.metrics();
    assert_eq!(m.kv_policy, KvPolicy::Reserve);
    assert_eq!(m.restarts, 0);
    assert_eq!(m.restart_discarded_tokens, 0);
    assert_eq!(m.step_growth_blocks, 0);
    assert_eq!(m.rejected_queue, 0);
    assert_eq!(m.admission_blocked, crate::AdmissionBlocked::default());
}
