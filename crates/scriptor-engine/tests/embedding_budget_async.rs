//! The embedding budget on the overlapped vision path, where a launched batch stays charged
//! until the test completes or fails it.

use scriptor_cache::{HostBuffer, ImageGrid, KvLayout, embedding_key};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::{
    DecodeRow, EngineError, EngineEvent, EngineRequest, ImageInput, ModelExecutor, PrefillRow,
    RequestId, Scheduler, SchedulerConfig, VisionJob, VisionToken,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

const VOCAB: usize = 128;
const EOS: u32 = 0;
const IMAGE: u32 = 50;
const HIDDEN: usize = 4;
/// Bytes of the embedding of a 16-patch image.
const IMG: usize = 32;
/// The budget: two 16-patch images.
const E: usize = 2 * IMG;

/// CPU executor whose asynchronous vision batch finishes only when the test says so.
struct Gated {
    complete: bool,
    fail_finish: bool,
    next: VisionToken,
    flights: HashMap<VisionToken, Vec<usize>>,
    begun: Vec<Vec<u8>>,
    finish_calls: usize,
    /// Sequence length at which generation ends.
    end: usize,
}

impl Default for Gated {
    fn default() -> Self {
        Self {
            complete: false,
            fail_finish: false,
            next: 0,
            flights: HashMap::new(),
            begun: Vec::new(),
            finish_calls: 0,
            end: 100,
        }
    }
}

fn logits(end: usize, len: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; VOCAB];
    l[if len >= end { EOS } else { 100 } as usize] = 10.0;
    l
}

impl ModelExecutor for Gated {
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
        Ok(rows
            .iter()
            .map(|r| logits(self.end, r.tokens.len()))
            .collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        Ok(rows
            .iter()
            .map(|r| logits(self.end, r.position + 1))
            .collect())
    }
}

/// A scheduler with a running text-only request, so every step has decode work and a pending
/// vision batch is not waited for.
fn busy(max_concurrency: usize) -> Scheduler<Gated> {
    let mut s = scheduler(max_concurrency);
    let (tx, rx) = unbounded_channel();
    std::mem::forget(rx);
    let text = EngineRequest {
        prompt: vec![1, 2, 1],
        images: Vec::new(),
        params: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
    };
    s.submit(text, tx).unwrap();
    s.step();
    s
}

fn scheduler(max_concurrency: usize) -> Scheduler<Gated> {
    let config = SchedulerConfig {
        max_concurrency,
        max_context: 256,
        vision_batch_patches: 64,
        vision_overlap: true,
        spatial_merge: 2,
        vocab_size: VOCAB,
        eos_token_ids: vec![EOS],
        image_token_id: IMAGE,
        embedding_cache_bytes: E,
        embedding_hidden: HIDDEN,
        embedding_elem_bytes: 2,
        fallback_seed: 1,
        embedding_release_after_prefill: false,
    };
    let layout = KvLayout {
        num_blocks: 64,
        block_tokens: 16,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    };
    Scheduler::new(config, layout, Gated::default())
}

fn request(seed: u8) -> EngineRequest {
    let grid = ImageGrid { t: 1, h: 4, w: 4 };
    let mut prompt = vec![1u32, 2];
    prompt.extend(std::iter::repeat_n(IMAGE, 4));
    prompt.push(1);
    EngineRequest {
        prompt,
        images: vec![ImageInput {
            grid,
            key: embedding_key(&[seed], grid),
            payload: Arc::new(vec![seed]),
        }],
        params: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
    }
}

type Rx = UnboundedReceiver<EngineEvent>;

fn submit(s: &mut Scheduler<Gated>, seed: u8) -> (RequestId, Rx) {
    let (tx, rx) = unbounded_channel();
    (s.submit(request(seed), tx).unwrap(), rx)
}

fn terminal(rx: &mut Rx) -> Option<EngineEvent> {
    let mut last = None;
    while let Ok(ev) = rx.try_recv() {
        if !matches!(ev, EngineEvent::Token(_)) {
            last = Some(ev);
        }
    }
    last
}

/// Steps until the executor has begun `batches` batches, then returns with that flight pending.
fn until_begun(s: &mut Scheduler<Gated>, batches: usize) {
    for _ in 0..50 {
        if s.executor().begun.len() >= batches {
            return;
        }
        s.step();
    }
    panic!("no batch began");
}

fn run_out(s: &mut Scheduler<Gated>) {
    for _ in 0..500 {
        if s.is_idle() {
            return;
        }
        s.executor_mut().complete = true;
        s.step();
    }
    panic!("the scheduler stopped making progress");
}

#[test]
fn the_charge_is_held_while_the_launched_batch_is_pending() {
    let mut s = busy(2);
    let (_a, mut rx) = submit(&mut s, 1);
    until_begun(&mut s, 1);
    assert_eq!(s.executor().begun, vec![vec![1]]);
    assert_eq!(s.pinned_embedding_bytes(), IMG);
    assert_eq!(s.retained_embedding_bytes(), IMG);
    for _ in 0..5 {
        s.step();
        assert_eq!(s.pinned_embedding_bytes(), IMG);
        assert_eq!(s.executor().finish_calls, 0);
    }
    assert!(terminal(&mut rx).is_none());
}

#[test]
fn reservations_stop_at_the_budget_while_a_batch_is_pending() {
    let mut s = busy(2);
    let _rxs: Vec<_> = (1..=3u8).map(|seed| submit(&mut s, seed)).collect();
    until_begun(&mut s, 1);
    for _ in 0..5 {
        s.step();
        assert_eq!(s.pinned_embedding_bytes(), E);
        assert_eq!(s.executor().begun.len(), 1);
    }
}

#[test]
fn a_reservation_that_does_not_fit_the_budget_counts_as_an_embedding_block() {
    let mut s = busy(2);
    let _rxs: Vec<_> = (1..=3u8).map(|seed| submit(&mut s, seed)).collect();
    until_begun(&mut s, 1);
    let before = s.metrics().admission_blocked.embedding;
    for _ in 0..5 {
        s.step();
    }
    let m = s.metrics();
    assert!(
        m.admission_blocked.embedding >= before + 5,
        "one count per blocked step: {} then {}",
        before,
        m.admission_blocked.embedding
    );
    assert_eq!(m.admission_blocked.slots, 0);
    assert_eq!(m.pinned_embedding_bytes, E);
    assert!(m.retained_embedding_bytes >= m.pinned_embedding_bytes);
}

#[test]
fn cancelling_a_request_keeps_its_pending_batch_charged_until_collection() {
    let mut s = busy(2);
    let (a, _rx) = submit(&mut s, 1);
    until_begun(&mut s, 1);
    assert!(s.cancel(a));
    assert_eq!(s.pinned_embedding_bytes(), IMG);
    s.step();
    assert_eq!(s.pinned_embedding_bytes(), IMG);

    s.executor_mut().complete = true;
    s.step();
    assert_eq!(s.executor().finish_calls, 1);
    assert_eq!(s.pinned_embedding_bytes(), 0);
    assert_eq!(s.retained_embedding_bytes(), 0);
}

#[test]
fn a_disconnected_client_keeps_its_pending_batch_charged_until_collection() {
    let mut s = busy(2);
    let (_a, rx) = submit(&mut s, 1);
    until_begun(&mut s, 1);
    drop(rx);
    s.step();
    assert_eq!(s.pinned_embedding_bytes(), IMG);

    s.executor_mut().complete = true;
    run_out(&mut s);
    assert_eq!(s.executor().finish_calls, 1);
    assert_eq!(s.pinned_embedding_bytes(), 0);
}

#[test]
fn a_later_request_cannot_allocate_against_a_live_flight() {
    let mut s = busy(2);
    let (a, _rx_a) = submit(&mut s, 1);
    until_begun(&mut s, 1);
    assert!(s.cancel(a));
    let (_b, mut rx_b) = submit(&mut s, 2);
    let (_c, mut rx_c) = submit(&mut s, 3);
    for _ in 0..5 {
        s.step();
        assert_eq!(
            s.pinned_embedding_bytes(),
            E,
            "flight plus the one that fits"
        );
        assert_eq!(s.executor().begun.len(), 1);
    }

    s.executor_mut().complete = true;
    s.step();
    assert!(s.pinned_embedding_bytes() <= E);
    run_out(&mut s);
    assert!(matches!(
        terminal(&mut rx_b),
        Some(EngineEvent::Finished(_))
    ));
    assert!(matches!(
        terminal(&mut rx_c),
        Some(EngineEvent::Finished(_))
    ));
    assert_eq!(s.pinned_embedding_bytes(), 0);
    assert!(s.retained_embedding_bytes() <= E);
}

#[test]
fn collecting_a_batch_releases_its_charge_once() {
    let mut s = busy(2);
    let (_a, mut rx) = submit(&mut s, 1);
    until_begun(&mut s, 1);
    assert_eq!(s.pinned_embedding_bytes(), IMG);
    s.executor_mut().complete = true;
    run_out(&mut s);
    assert!(matches!(terminal(&mut rx), Some(EngineEvent::Finished(_))));
    assert_eq!(s.executor().finish_calls, 1);
    assert_eq!(s.pinned_embedding_bytes(), 0);
    assert_eq!(s.retained_embedding_bytes(), IMG);
    s.step();
    assert_eq!(s.pinned_embedding_bytes(), 0);
    assert_eq!(s.retained_embedding_bytes(), IMG);
}

#[test]
fn a_failed_batch_releases_its_charge_once() {
    let mut s = busy(2);
    let (_a, mut rx) = submit(&mut s, 1);
    until_begun(&mut s, 1);
    assert_eq!(s.pinned_embedding_bytes(), IMG);
    s.executor_mut().fail_finish = true;
    s.executor_mut().complete = true;
    s.step();
    assert!(matches!(terminal(&mut rx), Some(EngineEvent::Failed(_))));
    assert_eq!(s.executor().finish_calls, 1);
    assert_eq!(s.pinned_embedding_bytes(), 0);
    assert_eq!(s.retained_embedding_bytes(), 0);
    s.step();
    assert_eq!(s.executor().finish_calls, 1);
    assert_eq!(s.pinned_embedding_bytes(), 0);
}

#[test]
fn a_collected_output_shared_with_the_cache_is_charged_once() {
    let mut s = busy(3);
    let (_a, _rx) = submit(&mut s, 1);
    until_begun(&mut s, 1);
    s.executor_mut().complete = true;
    s.step();
    assert_eq!(s.executor().finish_calls, 1);
    assert_eq!(s.pinned_embedding_bytes(), IMG);
    assert_eq!(s.retained_embedding_bytes(), IMG);
    assert_eq!(s.metrics().embedding_cache.entries, 1);
}
