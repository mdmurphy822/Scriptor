//! Demand KV policy at the scheduler's public surface: admission beyond reservation capacity,
//! a seeded randomized property over the pool and progress invariants, and the intake bounds.
//! The CPU executor checks control flow only; it says nothing about device output.
#![cfg(feature = "kv-demand-diagnostic")]

use scriptor_cache::{BlockId, HostBuffer, KvLayout};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::{
    DecodeRow, EngineError, EngineEvent, EngineRequest, ModelExecutor, PrefillRow, RequestId,
    Scheduler, SchedulerConfig, VisionJob,
};
use scriptor_spec::KvPolicy;
use std::collections::{HashMap, HashSet};
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

const VOCAB: usize = 128;
const EOS: u32 = 0;
const BLOCK_TOKENS: usize = 16;
const CONTEXT: usize = 160;
/// Blocks one full context occupies.
const FULL: usize = CONTEXT / BLOCK_TOKENS;
/// First prompt id meaning "never stop on its own".
const UNLIMITED: u32 = 120;

/// Deterministic CPU executor keeping each sequence's history. The first prompt id below 100 is
/// the number of tokens to generate before the end id (120 or more: none); generated ids are
/// `100 + k % 20`, and no prompt id lies in 100..120.
/// Every prefill and decode row is checked to hold blocks covering what it writes, and the rows
/// of one pass to hold disjoint blocks.
#[derive(Default)]
struct Checked {
    seqs: HashMap<RequestId, Vec<u32>>,
}

fn logits_for(history: &[u32]) -> Vec<f32> {
    let limit = history[0];
    let generated = history.iter().filter(|&&t| (100..120).contains(&t)).count();
    let next = if limit < 100 && generated >= limit as usize {
        EOS
    } else {
        100 + (generated % 20) as u32
    };
    let mut l = vec![0.0f32; VOCAB];
    l[next as usize] = 10.0;
    l
}

fn assert_disjoint<'a>(tables: impl Iterator<Item = &'a Vec<BlockId>>) {
    let mut seen = HashSet::new();
    for table in tables {
        for b in table {
            assert!(seen.insert(*b), "block shared between rows of one pass");
        }
    }
}

impl ModelExecutor for Checked {
    type Embedding = HostBuffer;

    fn vision(&mut self, _jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        panic!("text requests only");
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        assert_disjoint(rows.iter().map(|r| &r.blocks));
        for r in rows {
            assert!(
                r.blocks.len() * BLOCK_TOKENS >= r.tokens.len(),
                "prefill blocks cover the tokens"
            );
            assert!(
                self.seqs.insert(r.seq, r.tokens.clone()).is_none(),
                "prefill of a live sequence"
            );
        }
        Ok(rows.iter().map(|r| logits_for(&r.tokens)).collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        assert_disjoint(rows.iter().map(|r| &r.blocks));
        let mut out = Vec::new();
        for r in rows {
            assert!(
                r.blocks.len() * BLOCK_TOKENS > r.position,
                "decode blocks cover the position written"
            );
            let h = self.seqs.get_mut(&r.seq).expect("prefilled");
            assert_eq!(h.len(), r.position, "decode position follows history");
            h.push(r.token);
            out.push(logits_for(h));
        }
        Ok(out)
    }

    fn release(&mut self, seq: RequestId) {
        self.seqs.remove(&seq);
    }
}

fn scheduler(policy: KvPolicy, slots: usize, blocks: usize) -> Scheduler<Checked> {
    let config = SchedulerConfig {
        max_concurrency: slots,
        max_context: CONTEXT,
        vision_batch_patches: 64,
        vision_overlap: true,
        spatial_merge: 2,
        vocab_size: VOCAB,
        eos_token_ids: vec![EOS],
        image_token_id: 50,
        embedding_cache_bytes: 1 << 20,
        embedding_hidden: 4,
        embedding_elem_bytes: 2,
        fallback_seed: 1,
        embedding_release_after_prefill: false,
    };
    let layout = KvLayout {
        num_blocks: blocks,
        block_tokens: BLOCK_TOKENS,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    };
    let mut s = Scheduler::new(config, layout, Checked::default());
    s.set_kv_policy(policy);
    s
}

/// A text request of `prompt_len` ids that generates `limit` tokens (`UNLIMITED` for none) and
/// is capped at `max_tokens`.
fn request(prompt_len: usize, limit: u32, max_tokens: usize) -> EngineRequest {
    let mut prompt = vec![1u32; prompt_len];
    prompt[0] = limit;
    EngineRequest {
        prompt,
        images: Vec::new(),
        params: SamplingParams {
            temperature: 0.0,
            max_tokens: Some(max_tokens as u32),
            ..SamplingParams::default()
        },
    }
}

/// Tokens a request delivers when it finishes uninterrupted.
fn expected(limit: u32, max_tokens: usize) -> Vec<u32> {
    let n = if limit < 100 {
        (limit as usize).min(max_tokens)
    } else {
        max_tokens
    };
    (0..n).map(|k| 100 + (k % 20) as u32).collect()
}

#[derive(Debug, Clone, PartialEq)]
enum End {
    Finished,
    Aborted,
    Other(String),
}

/// What one request's channel delivered: tokens since the last restart, the restarts, and the
/// single terminal event.
#[derive(Default)]
struct Seen {
    tokens: Vec<u32>,
    restarts: usize,
    end: Option<End>,
}

struct Client {
    rx: UnboundedReceiver<EngineEvent>,
    seen: Seen,
    limit: u32,
    max_tokens: usize,
}

struct Harness {
    s: Scheduler<Checked>,
    clients: Vec<Client>,
    ids: Vec<RequestId>,
}

impl Harness {
    fn new(policy: KvPolicy, slots: usize, blocks: usize) -> Self {
        Self {
            s: scheduler(policy, slots, blocks),
            clients: Vec::new(),
            ids: Vec::new(),
        }
    }

    fn submit(
        &mut self,
        prompt_len: usize,
        limit: u32,
        max_tokens: usize,
        restartable: bool,
    ) -> Result<usize, EngineError> {
        let (tx, rx) = unbounded_channel();
        let id = self.clients.len() as RequestId;
        self.s.submit_restartable_with_id(
            id,
            request(prompt_len, limit, max_tokens),
            tx,
            restartable,
        )?;
        self.clients.push(Client {
            rx,
            seen: Seen::default(),
            limit,
            max_tokens,
        });
        self.ids.push(id);
        Ok(self.clients.len() - 1)
    }

    /// Moves every delivered event into its client's record; a second terminal event or an
    /// event after one fails.
    fn drain(&mut self) {
        for c in &mut self.clients {
            while let Ok(ev) = c.rx.try_recv() {
                assert!(c.seen.end.is_none(), "no event follows the terminal event");
                match ev {
                    EngineEvent::Token(t) => {
                        assert_eq!(t.index as usize, c.seen.tokens.len());
                        c.seen.tokens.push(t.token);
                    }
                    EngineEvent::Restarted => {
                        c.seen.tokens.clear();
                        c.seen.restarts += 1;
                    }
                    EngineEvent::Finished(_) => c.seen.end = Some(End::Finished),
                    EngineEvent::Aborted => c.seen.end = Some(End::Aborted),
                    other => c.seen.end = Some(End::Other(format!("{other:?}"))),
                }
            }
        }
    }

    fn live(&self) -> Vec<usize> {
        (0..self.clients.len())
            .filter(|&i| self.clients[i].seen.end.is_none())
            .collect()
    }

    /// Requests the engine has taken on: running plus evicted ones awaiting re-admission.
    fn accepted(&self) -> usize {
        let m = self.s.metrics();
        m.running + (m.waiting - m.waiting_new)
    }

    /// Steps once and checks the pool and executor are consistent with the scheduler.
    fn step_checked(&mut self) {
        self.s.step();
        let m = self.s.metrics();
        assert!(m.pool.used_blocks <= m.pool.total_blocks, "I1 pool bound");
        assert_eq!(
            m.pool.used_blocks + m.pool.free_blocks,
            m.pool.total_blocks,
            "every block is free or held"
        );
        assert!(
            self.s.executor().seqs.len() <= m.pool.sequences,
            "executor state only for sequences holding blocks"
        );
        self.drain();
    }

    fn assert_clean(&self) {
        assert!(self.s.is_idle());
        let m = self.s.metrics();
        assert_eq!(m.pool.sequences, 0, "no pool registration leaks");
        assert_eq!(m.pool.used_blocks, 0, "no blocks leak");
        assert!(self.s.executor().seqs.is_empty(), "no executor state leaks");
    }
}

#[test]
fn demand_admits_beyond_reservation_capacity_without_eviction() {
    // Each request reserves a full context but generates 10 tokens: two blocks of real demand.
    let (slots, blocks, requests) = (4, 2 * FULL + 4, 8);
    let run = |policy: KvPolicy| {
        let mut h = Harness::new(policy, slots, blocks);
        for _ in 0..requests {
            h.submit(8, 10, CONTEXT - 8, true).unwrap();
        }
        let mut peak = 0;
        for _ in 0..10_000 {
            if h.s.is_idle() {
                break;
            }
            h.step_checked();
            peak = peak.max(h.s.metrics().running);
        }
        h.assert_clean();
        for c in &h.clients {
            assert_eq!(c.seen.end, Some(End::Finished));
            assert_eq!(c.seen.tokens, expected(c.limit, c.max_tokens));
            assert_eq!(c.seen.restarts, 0, "no eviction");
        }
        let m = h.s.metrics();
        assert_eq!(m.pool.preemptions, 0);
        peak
    };
    let reserve_peak = run(KvPolicy::Reserve);
    let demand_peak = run(KvPolicy::DemandDiagnostic);
    assert_eq!(
        reserve_peak,
        blocks / FULL,
        "reservation bounds concurrency"
    );
    assert!(
        demand_peak > reserve_peak,
        "demand admitted {demand_peak}, reservation {reserve_peak}"
    );
}

/// Linear congruential generator; the high bits are used.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    /// Uniform in `lo..=hi`.
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() as usize) % (hi - lo + 1)
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.range(0, 99) < percent
    }
}

fn property(seed: u64) {
    let mut rng = Lcg(seed);
    let slots = rng.range(2, 5);
    let blocks = rng.range(FULL, FULL + 4);
    let mut h = Harness::new(KvPolicy::DemandDiagnostic, slots, blocks);
    let total = 40;
    let mut cancelled = HashSet::new();
    let mut steps = 0;
    loop {
        steps += 1;
        assert!(steps < 50_000, "every request terminates");
        if h.clients.len() < total {
            for _ in 0..usize::from(rng.chance(40)) {
                let prompt_len = rng.range(4, 60);
                let room = CONTEXT - prompt_len;
                let max_tokens = if rng.chance(50) {
                    room
                } else {
                    rng.range(1, room)
                };
                let limit = if rng.chance(40) {
                    match rng.range(1, 99) as u32 {
                        50 => 51,
                        n => n,
                    }
                } else {
                    UNLIMITED
                };
                let restartable = rng.chance(70);
                // A refused submit is the queue bound at work and is not a request.
                match h.submit(prompt_len, limit, max_tokens, restartable) {
                    Ok(_) | Err(EngineError::QueueFull { .. }) => {}
                    Err(e) => panic!("unexpected refusal {e:?}"),
                }
            }
        }
        if rng.range(0, 999) < 5 {
            let live = h.live();
            if !live.is_empty() {
                let i = live[rng.range(0, live.len() - 1)];
                if h.s.cancel(h.ids[i]) {
                    cancelled.insert(i);
                }
            }
        }
        let finished_before: HashSet<usize> = (0..h.clients.len())
            .filter(|&i| h.clients[i].seen.end.is_some())
            .collect();
        let restarts_before: Vec<usize> = h.clients.iter().map(|c| c.seen.restarts).collect();
        h.step_checked();
        // A request restarts only while a lower-rank request, one submitted earlier, is still
        // in the system.
        for (i, before) in restarts_before.iter().enumerate() {
            if h.clients[i].seen.restarts > *before {
                assert!(
                    (0..i).any(|j| !finished_before.contains(&j)),
                    "the lowest rank is never evicted"
                );
            }
        }
        assert!(h.accepted() <= 2 * slots, "accepted bound");
        if h.clients.len() >= total && h.s.is_idle() {
            break;
        }
    }
    h.drain();
    h.assert_clean();
    for (i, c) in h.clients.iter().enumerate() {
        match &c.seen.end {
            Some(End::Finished) => {
                assert_eq!(
                    c.seen.tokens,
                    expected(c.limit, c.max_tokens),
                    "request {i}"
                );
            }
            Some(End::Aborted) => assert!(cancelled.contains(&i), "request {i} aborted unasked"),
            other => panic!("request {i} ended {other:?}: no OutOfBlocks or failure allowed"),
        }
    }
    let m = h.s.metrics();
    assert_eq!(
        m.finished + m.aborted,
        h.clients.len() as u64,
        "one terminal per request"
    );
    assert_eq!(m.failed, 0);
}

macro_rules! seeds {
    ($($name:ident: $seed:expr,)*) => {
        $(
            #[test]
            fn $name() {
                property($seed);
            }
        )*
    };
}

seeds! {
    random_workload_keeps_pool_and_progress_invariants_seed_1: 1,
    random_workload_keeps_pool_and_progress_invariants_seed_2: 0x9e37_79b9,
    random_workload_keeps_pool_and_progress_invariants_seed_3: 20_251_113,
    random_workload_keeps_pool_and_progress_invariants_seed_4: 0xdead_beef,
    random_workload_keeps_pool_and_progress_invariants_seed_5: 7_777_777,
    random_workload_keeps_pool_and_progress_invariants_seed_6: 0x1234_5678_9abc,
}

#[test]
fn a_full_queue_refuses_new_requests_but_never_evicted_ones() {
    let slots = 2;
    let mut h = Harness::new(KvPolicy::DemandDiagnostic, slots, FULL);
    let bound = 2 * slots;
    // Long unbounded requests on a pool of one context force evictions.
    for _ in 0..bound {
        h.submit(8, UNLIMITED, CONTEXT - 8, true).unwrap();
    }
    assert!(matches!(
        h.submit(8, UNLIMITED, CONTEXT - 8, true),
        Err(EngineError::QueueFull { .. })
    ));
    let (mut refused, mut accepted_late, mut saw_victim) = (0, 0, false);
    for _ in 0..20_000 {
        if h.s.is_idle() {
            break;
        }
        h.step_checked();
        let m = h.s.metrics();
        saw_victim |= m.waiting > m.waiting_new;
        assert!(h.accepted() <= 2 * slots, "accepted bound");
        // Offer more work every step: refusal exactly when the new-request queue is full.
        if accepted_late + refused < 30 {
            let full = m.waiting_new >= bound;
            match h.submit(8, UNLIMITED, CONTEXT - 8, true) {
                Ok(_) => {
                    assert!(!full, "a full queue accepted a request");
                    accepted_late += 1;
                }
                Err(EngineError::QueueFull { .. }) => {
                    assert!(full, "a queue with room refused a request");
                    refused += 1;
                }
                Err(e) => panic!("unexpected refusal {e:?}"),
            }
        }
    }
    assert!(saw_victim, "evicted requests waited for re-admission");
    assert!(refused > 0, "the bound refused new requests");
    assert!(accepted_late > 0, "room reopened as the queue drained");
    h.assert_clean();
    for (i, c) in h.clients.iter().enumerate() {
        assert_eq!(
            c.seen.end,
            Some(End::Finished),
            "request {i}: accepted requests, evicted ones included, are never rejected"
        );
        assert_eq!(c.seen.tokens, expected(c.limit, c.max_tokens));
    }
    assert!(
        h.clients.iter().any(|c| c.seen.restarts > 0),
        "victims were restarted"
    );
}
