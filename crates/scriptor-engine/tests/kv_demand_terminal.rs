//! Terminal paths while evicted requests wait for re-admission under the demand policy: every
//! request gets exactly one terminal event and leaves no pool, executor or embedding charge.
//! The CPU executor checks scheduler control flow only; it says nothing about device output.
#![cfg(feature = "kv-demand-diagnostic")]

use scriptor_cache::{HostBuffer, ImageGrid, KvLayout, embedding_key};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::{
    CancelReason, DecodeRow, EngineError, EngineEvent, EngineRequest, ImageInput, ModelExecutor,
    PrefillRow, RequestId, Scheduler, SchedulerConfig, VisionJob, VisionToken,
};
use scriptor_spec::KvPolicy;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

const VOCAB: usize = 128;
const EOS: u32 = 0;
const IMAGE: u32 = 50;
const HIDDEN: usize = 4;
const CONTEXT: usize = 160;
/// Bytes of the embedding of a 16-patch image.
const IMG: usize = 32;
/// Prompt tokens before the image: with the image, three blocks.
const PAD: usize = 40;

/// CPU executor that continues every sequence from the history it holds. The first prompt id
/// of a text request is far above the generation count, so only the length limits end it. An
/// asynchronous vision batch finishes only once `complete` is set or the scheduler waits for it.
#[derive(Default)]
struct Probe {
    seqs: HashMap<RequestId, Vec<u32>>,
    prefills: HashMap<RequestId, usize>,
    releases: HashMap<RequestId, usize>,
    complete: bool,
    next: VisionToken,
    flights: HashMap<VisionToken, Vec<usize>>,
    begun: Vec<Vec<u8>>,
    finish_calls: usize,
    fatal_decode: bool,
    fail_finish: bool,
}

fn next_logits(history: &[u32]) -> Vec<f32> {
    let generated = history.iter().filter(|&&t| t >= 100).count();
    let mut l = vec![0.0f32; VOCAB];
    l[100 + generated % 20] = 10.0;
    l
}

impl ModelExecutor for Probe {
    type Embedding = HostBuffer;

    fn vision(&mut self, _jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        panic!("the overlapped path never embeds synchronously");
    }

    fn supports_async_vision(&self) -> bool {
        true
    }

    fn vision_begin(&mut self, jobs: &[VisionJob]) -> Result<VisionToken, EngineError> {
        assert!(self.flights.is_empty(), "one batch outstanding at a time");
        self.begun.push(jobs.iter().map(|j| j.payload[0]).collect());
        self.complete = false;
        let token = self.next;
        self.next += 1;
        self.flights
            .insert(token, jobs.iter().map(|j| j.patches).collect());
        Ok(token)
    }

    fn vision_ready(&mut self, _token: VisionToken) -> Result<bool, EngineError> {
        Ok(self.complete)
    }

    fn vision_finish(&mut self, token: VisionToken) -> Result<Vec<HostBuffer>, EngineError> {
        self.finish_calls += 1;
        let patches = self.flights.remove(&token).expect("begun");
        if self.fail_finish {
            return Err(EngineError::Executor("injected".into()));
        }
        Ok(patches
            .iter()
            .map(|p| HostBuffer(vec![0u8; p / 4 * HIDDEN * 2]))
            .collect())
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        for r in rows {
            assert!(
                !self.seqs.contains_key(&r.seq),
                "prefill of a live sequence"
            );
            self.seqs.insert(r.seq, r.tokens.clone());
            *self.prefills.entry(r.seq).or_default() += 1;
        }
        Ok(rows.iter().map(|r| next_logits(&r.tokens)).collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        if self.fatal_decode {
            return Err(EngineError::Fatal {
                code: 700,
                message: "injected".into(),
            });
        }
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
        *self.releases.entry(seq).or_default() += 1;
    }
}

fn scheduler(slots: usize, blocks: usize, embedding_bytes: usize) -> Scheduler<Probe> {
    let config = SchedulerConfig {
        max_concurrency: slots,
        max_context: CONTEXT,
        vision_batch_patches: 16,
        vision_overlap: true,
        spatial_merge: 2,
        vocab_size: VOCAB,
        eos_token_ids: vec![EOS],
        image_token_id: IMAGE,
        embedding_cache_bytes: embedding_bytes,
        embedding_hidden: HIDDEN,
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
    let mut s = Scheduler::new(config, layout, Probe::default());
    s.set_kv_policy(KvPolicy::DemandDiagnostic);
    s
}

fn params(prompt_len: usize) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        max_tokens: Some((CONTEXT - prompt_len) as u32),
        ..SamplingParams::default()
    }
}

fn text_request(prompt_len: usize) -> EngineRequest {
    let mut prompt = vec![1u32; prompt_len];
    prompt[0] = 1000;
    EngineRequest {
        prompt,
        images: Vec::new(),
        params: params(prompt_len),
    }
}

fn image_request(seed: u8) -> EngineRequest {
    let grid = ImageGrid { t: 1, h: 4, w: 4 };
    let mut prompt = vec![1u32; PAD];
    prompt[0] = 1000;
    prompt.extend(std::iter::repeat_n(IMAGE, 4));
    prompt.push(1);
    let len = prompt.len();
    EngineRequest {
        prompt,
        images: vec![ImageInput {
            grid,
            key: embedding_key(&[seed], grid),
            payload: Arc::new(vec![seed]),
        }],
        params: params(len),
    }
}

/// One request's observed stream: tokens since its last restart, restarts, terminal events.
#[derive(Default)]
struct Seen {
    max_tokens: usize,
    tokens: usize,
    restarts: usize,
    terminals: Vec<EngineEvent>,
}

struct Fx {
    s: Scheduler<Probe>,
    rx: Vec<Option<UnboundedReceiver<EngineEvent>>>,
    seen: Vec<Seen>,
}

impl Fx {
    fn new(slots: usize, blocks: usize, embedding_bytes: usize) -> Self {
        Self {
            s: scheduler(slots, blocks, embedding_bytes),
            rx: Vec::new(),
            seen: Vec::new(),
        }
    }

    fn submit(&mut self, request: EngineRequest) -> RequestId {
        let (tx, rx) = unbounded_channel();
        let id = self.rx.len() as RequestId;
        let max = request.params.max_tokens.unwrap() as usize;
        self.s
            .submit_restartable_with_id(id, request, tx, true)
            .unwrap();
        self.rx.push(Some(rx));
        self.seen.push(Seen {
            max_tokens: max,
            ..Seen::default()
        });
        id
    }

    fn drain(&mut self) {
        for (rx, seen) in self.rx.iter_mut().zip(&mut self.seen) {
            let Some(rx) = rx else { continue };
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    EngineEvent::Token(_) => seen.tokens += 1,
                    EngineEvent::Restarted => {
                        seen.tokens = 0;
                        seen.restarts += 1;
                    }
                    other => seen.terminals.push(other),
                }
            }
        }
    }

    /// Output tokens the connected requests without a terminal event may still generate.
    fn expected_in_flight(&self) -> u64 {
        self.rx
            .iter()
            .zip(&self.seen)
            .filter(|(rx, seen)| rx.is_some() && seen.terminals.is_empty())
            .map(|(_, seen)| (seen.max_tokens - seen.tokens) as u64)
            .sum()
    }

    fn step(&mut self) {
        self.s.step();
        self.drain();
        assert!(self.s.metrics().pool.used_blocks <= self.s.metrics().pool.total_blocks);
        assert_eq!(self.s.tokens_in_flight(), self.s.metrics().tokens_in_flight);
        if self.rx.iter().all(Option::is_some) {
            assert_eq!(self.s.tokens_in_flight(), self.expected_in_flight());
        }
    }

    /// Steps until the scheduler is idle.
    fn run_out(&mut self) {
        for _ in 0..4000 {
            if self.s.is_idle() {
                return;
            }
            self.step();
        }
        panic!("the scheduler stopped making progress");
    }

    /// Requests evicted and awaiting re-admission.
    fn victims(&self) -> usize {
        let m = self.s.metrics();
        m.waiting - m.waiting_new
    }

    /// The id of the request restarted by the latest step.
    fn restarted_since(&self, before: &[usize]) -> Option<RequestId> {
        self.seen
            .iter()
            .zip(before)
            .position(|(s, &b)| s.restarts > b)
            .map(|i| i as RequestId)
    }

    /// Four restartable text requests on a pool that forces evictions; steps until a running
    /// request is restarted and returns it, waiting for re-admission.
    fn with_restarted_victim() -> (Self, RequestId) {
        let mut fx = Self::new(3, 10, 0);
        for len in [8usize, 40, 12, 30] {
            fx.submit(text_request(len));
        }
        for _ in 0..4000 {
            let before: Vec<usize> = fx.seen.iter().map(|s| s.restarts).collect();
            fx.step();
            if let Some(v) = fx.restarted_since(&before) {
                assert_eq!(fx.victims(), 1);
                return (fx, v);
            }
        }
        panic!("no request was restarted");
    }

    /// A text request, then two image requests that take the free blocks. Steps, completing
    /// each vision batch once a request waits, until the evicted image request waits with its
    /// image in the batch in flight; returns it.
    fn with_flight_victim() -> (Self, RequestId) {
        let mut fx = Self::new(3, 12, 4 * IMG);
        fx.submit(text_request(3));
        fx.step();
        fx.submit(image_request(1));
        fx.submit(image_request(2));
        for _ in 0..400 {
            fx.step();
            if fx.victims() > 0 {
                if fx.s.executor().begun.len() == 2 && !fx.s.executor().flights.is_empty() {
                    assert_eq!(fx.victims(), 1);
                    assert_eq!(fx.s.executor().begun[1], vec![2]);
                    return (fx, 2);
                }
                fx.s.executor_mut().complete = true;
            }
        }
        panic!("no waiting request owned the batch in flight");
    }

    /// Each request ends with exactly one terminal event and every counted request is
    /// accounted for once.
    fn assert_one_terminal_each(&self) {
        for (i, seen) in self.seen.iter().enumerate() {
            if self.rx[i].is_some() {
                assert_eq!(seen.terminals.len(), 1, "request {i}: {:?}", seen.terminals);
            }
        }
        let m = self.s.metrics();
        assert_eq!(
            m.finished + m.aborted + m.failed + m.shed,
            m.submitted,
            "each request is counted once"
        );
    }

    /// Nothing is held: no pool sequence or block, no executor state, no embedding charge, and
    /// every executor state a prefill created was released once.
    fn assert_nothing_held(&self) {
        let m = self.s.metrics();
        assert_eq!(m.pool.sequences, 0);
        assert_eq!(m.pool.used_blocks, 0);
        assert_eq!(m.tokens_in_flight, 0);
        assert!(self.s.executor().seqs.is_empty(), "executor state released");
        assert_eq!(self.s.pinned_embedding_bytes(), 0);
        let ex = self.s.executor();
        for (id, n) in &ex.prefills {
            assert_eq!(
                ex.releases.get(id),
                Some(n),
                "request {id} released once per prefill"
            );
        }
    }
}

#[test]
fn a_restarted_victim_is_charged_its_full_budget_while_it_waits() {
    let (fx, v) = Fx::with_restarted_victim();
    assert_eq!(fx.seen[v as usize].tokens, 0);
    assert_eq!(fx.s.tokens_in_flight(), fx.expected_in_flight());
    assert!(fx.s.tokens_in_flight() >= fx.seen[v as usize].max_tokens as u64);
}

#[test]
fn cancelling_a_waiting_victim_ends_it_once_and_leaks_nothing() {
    let (mut fx, v) = Fx::with_restarted_victim();
    let releases = fx.s.executor().releases.clone();
    let before = fx.s.tokens_in_flight();
    assert!(fx.s.cancel(v));
    fx.drain();
    assert_eq!(fx.seen[v as usize].terminals, vec![EngineEvent::Aborted]);
    assert_eq!(
        fx.s.tokens_in_flight(),
        before - fx.seen[v as usize].max_tokens as u64
    );
    assert_eq!(fx.s.executor().releases, releases, "no second release");
    assert!(!fx.s.cancel(v), "a cancelled request cannot end twice");
    let prefills = fx.s.executor().prefills[&v];
    fx.run_out();
    assert_eq!(fx.s.executor().prefills[&v], prefills, "never re-admitted");
    fx.assert_one_terminal_each();
    fx.assert_nothing_held();
    assert_eq!(fx.seen[v as usize].terminals, vec![EngineEvent::Aborted]);
}

#[test]
fn a_victim_whose_client_disconnected_is_cancelled_at_re_admission_without_prefill() {
    let (mut fx, v) = Fx::with_restarted_victim();
    fx.rx[v as usize] = None;
    let prefills = fx.s.executor().prefills[&v];
    let aborted = fx.s.metrics().aborted;
    fx.run_out();
    assert_eq!(
        fx.s.executor().prefills[&v],
        prefills,
        "never prefilled again"
    );
    assert_eq!(fx.s.metrics().aborted, aborted + 1);
    fx.assert_one_terminal_each();
    fx.assert_nothing_held();
}

#[test]
fn cancel_all_with_victims_present_ends_every_request_once() {
    let (mut fx, _) = Fx::with_restarted_victim();
    let live = fx.s.metrics().running + fx.s.metrics().waiting;
    assert_eq!(fx.s.cancel_all(CancelReason::Drain), live);
    fx.drain();
    for seen in &fx.seen {
        assert!(matches!(
            seen.terminals.as_slice(),
            [EngineEvent::Cancelled(CancelReason::Drain)] | [EngineEvent::Finished(_)]
        ));
    }
    assert!(fx.s.is_idle());
    fx.assert_one_terminal_each();
    fx.assert_nothing_held();
}

#[test]
fn shed_all_with_victims_present_ends_every_request_once() {
    let (mut fx, _) = Fx::with_restarted_victim();
    let live = fx.s.metrics().running + fx.s.metrics().waiting;
    assert_eq!(fx.s.shed_all(), live);
    fx.drain();
    assert!(fx.s.is_idle());
    for seen in &fx.seen {
        assert!(matches!(
            seen.terminals.as_slice(),
            [EngineEvent::Failed(EngineError::Shed)] | [EngineEvent::Finished(_)]
        ));
    }
    fx.assert_one_terminal_each();
    fx.assert_nothing_held();
}

#[test]
fn shed_youngest_never_picks_a_waiting_victim() {
    let (mut fx, v) = Fx::with_restarted_victim();
    assert!(fx.s.shed_youngest());
    fx.drain();
    assert!(fx.seen[v as usize].terminals.is_empty());
    assert_eq!(fx.victims(), 1);
    fx.run_out();
    fx.assert_one_terminal_each();
    fx.assert_nothing_held();
}

#[test]
fn abandon_all_with_victims_present_ends_every_request_once_without_the_executor() {
    let (mut fx, _) = Fx::with_restarted_victim();
    let running = fx.s.metrics().running;
    let live = running + fx.s.metrics().waiting;
    let releases = fx.s.executor().releases.clone();
    assert_eq!(fx.s.abandon_all(&EngineError::Closed), live);
    fx.drain();
    assert!(fx.s.is_idle());
    assert_eq!(fx.s.tokens_in_flight(), 0);
    assert_eq!(fx.s.pinned_embedding_bytes(), 0);
    assert_eq!(
        fx.s.executor().releases,
        releases,
        "the executor is not touched"
    );
    for seen in &fx.seen {
        assert!(matches!(
            seen.terminals.as_slice(),
            [EngineEvent::Failed(EngineError::Closed)] | [EngineEvent::Finished(_)]
        ));
    }
    fx.assert_one_terminal_each();
    assert_eq!(
        fx.s.metrics().pool.sequences,
        running,
        "abandoning leaves the pool of an untrusted device untouched; waiting victims hold none"
    );
}

#[test]
fn a_fatal_fault_with_victims_present_ends_every_request_once() {
    let (mut fx, _) = Fx::with_restarted_victim();
    fx.s.executor_mut().fatal_decode = true;
    fx.step();
    assert!(fx.s.fatal().is_some());
    fx.step();
    assert!(fx.s.is_idle());
    for seen in &fx.seen {
        assert!(matches!(
            seen.terminals.as_slice(),
            [EngineEvent::Failed(_)] | [EngineEvent::Finished(_)]
        ));
    }
    fx.assert_one_terminal_each();
    assert_eq!(fx.s.pinned_embedding_bytes(), 0);
    assert_eq!(fx.s.tokens_in_flight(), 0);
}

#[test]
fn a_flight_victim_waits_with_a_charged_batch_and_a_full_budget() {
    let (fx, v) = Fx::with_flight_victim();
    assert_eq!(fx.seen[v as usize].tokens, 0);
    assert_eq!(fx.s.tokens_in_flight(), fx.expected_in_flight());
    assert_eq!(fx.s.pinned_embedding_bytes(), 2 * IMG);
    assert!(!fx.s.executor().prefills.contains_key(&v));
}

#[test]
fn cancelling_a_victim_in_vision_flight_ends_it_once_and_frees_its_charge_at_collection() {
    let (mut fx, v) = Fx::with_flight_victim();
    assert!(fx.s.cancel(v));
    fx.drain();
    assert_eq!(fx.seen[v as usize].terminals, vec![EngineEvent::Aborted]);
    assert_eq!(
        fx.s.pinned_embedding_bytes(),
        2 * IMG,
        "the batch stays charged"
    );
    fx.s.executor_mut().complete = true;
    fx.step();
    assert_eq!(fx.s.executor().finish_calls, 2);
    assert!(fx.s.executor().flights.is_empty());
    fx.run_out();
    fx.s.clear_embedding_cache();
    assert_eq!(fx.s.retained_embedding_bytes(), 0);
    assert!(!fx.s.executor().prefills.contains_key(&v));
    fx.assert_one_terminal_each();
    fx.assert_nothing_held();
    assert_eq!(fx.seen[v as usize].terminals, vec![EngineEvent::Aborted]);
}

#[test]
fn a_failed_batch_ends_a_waiting_victim_once() {
    let (mut fx, v) = Fx::with_flight_victim();
    fx.s.executor_mut().fail_finish = true;
    fx.s.executor_mut().complete = true;
    fx.step();
    assert_eq!(
        fx.seen[v as usize].terminals,
        vec![EngineEvent::Failed(EngineError::Executor(
            "injected".into()
        ))]
    );
    fx.run_out();
    fx.s.clear_embedding_cache();
    assert_eq!(fx.s.retained_embedding_bytes(), 0);
    fx.assert_one_terminal_each();
    fx.assert_nothing_held();
}

#[test]
fn a_disconnected_victim_in_vision_flight_is_cancelled_at_re_admission_without_prefill() {
    let (mut fx, v) = Fx::with_flight_victim();
    fx.rx[v as usize] = None;
    let aborted = fx.s.metrics().aborted;
    fx.s.executor_mut().complete = true;
    fx.run_out();
    assert!(
        !fx.s.executor().prefills.contains_key(&v),
        "never prefilled"
    );
    assert_eq!(fx.s.metrics().aborted, aborted + 1);
    fx.s.clear_embedding_cache();
    assert_eq!(fx.s.retained_embedding_bytes(), 0);
    fx.assert_one_terminal_each();
    fx.assert_nothing_held();
}

/// Ends the scheduler's work, completes the batch in flight with one step and checks that no
/// embedding bytes remain charged or retained.
fn assert_flight_collected_and_uncharged(fx: &mut Fx) {
    assert_eq!(
        fx.s.pinned_embedding_bytes(),
        IMG,
        "the batch stays charged"
    );
    fx.s.executor_mut().complete = true;
    fx.step();
    assert!(fx.s.executor().flights.is_empty());
    fx.s.clear_embedding_cache();
    assert_eq!(fx.s.pinned_embedding_bytes(), 0);
    assert_eq!(fx.s.retained_embedding_bytes(), 0);
}

#[test]
fn cancel_all_with_a_flight_victim_ends_every_request_once() {
    let (mut fx, _) = Fx::with_flight_victim();
    assert_eq!(fx.s.cancel_all(CancelReason::Drain), 3);
    fx.drain();
    for seen in &fx.seen {
        assert_eq!(
            seen.terminals,
            vec![EngineEvent::Cancelled(CancelReason::Drain)]
        );
    }
    assert_flight_collected_and_uncharged(&mut fx);
    fx.assert_one_terminal_each();
    fx.assert_nothing_held();
}

#[test]
fn shed_all_with_a_flight_victim_ends_every_request_once() {
    let (mut fx, _) = Fx::with_flight_victim();
    assert_eq!(fx.s.shed_all(), 3);
    fx.drain();
    for seen in &fx.seen {
        assert_eq!(seen.terminals, vec![EngineEvent::Failed(EngineError::Shed)]);
    }
    assert_flight_collected_and_uncharged(&mut fx);
    fx.assert_one_terminal_each();
    fx.assert_nothing_held();
}

#[test]
fn abandon_all_with_a_flight_victim_ends_every_request_once() {
    let (mut fx, _) = Fx::with_flight_victim();
    let running = fx.s.metrics().running;
    assert_eq!(fx.s.abandon_all(&EngineError::Closed), 3);
    fx.drain();
    for seen in &fx.seen {
        assert_eq!(
            seen.terminals,
            vec![EngineEvent::Failed(EngineError::Closed)]
        );
    }
    assert_flight_collected_and_uncharged(&mut fx);
    fx.assert_one_terminal_each();
    assert_eq!(
        fx.s.metrics().pool.sequences,
        running,
        "the pool is left untouched"
    );
}

#[test]
fn a_fatal_fault_with_a_flight_victim_ends_every_request_once_and_drops_the_charge() {
    let (mut fx, _) = Fx::with_flight_victim();
    fx.s.executor_mut().fatal_decode = true;
    fx.step();
    fx.step();
    assert!(fx.s.fatal().is_some());
    for seen in &fx.seen {
        assert_eq!(seen.terminals.len(), 1);
    }
    fx.assert_one_terminal_each();
    assert_eq!(fx.s.pinned_embedding_bytes(), 0);
    assert_eq!(fx.s.tokens_in_flight(), 0);
}
