//! Capacity behaviour: more concurrent slots than full-context sequences the KV pool holds.

use scriptor_cache::{HostBuffer, KvLayout};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::{
    DecodeRow, EngineError, EngineEvent, EngineRequest, ModelExecutor, PrefillRow, RequestId,
    Scheduler, SchedulerConfig, VisionJob,
};
use std::collections::HashMap;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

const VOCAB: usize = 128;
const EOS: u32 = 0;
const IMAGE: u32 = 50;
const HIDDEN: usize = 4;

/// Deterministic CPU executor. The first prompt token is the number of tokens to generate;
/// generated ids are `100 + (k % 20)` and every id below 100 belongs to the prompt, so the next
/// token depends only on the row's tokens and recompute reproduces the same stream.
#[derive(Default)]
struct TinyExecutor {
    vision_batches: Vec<Vec<usize>>,
    prefill_rows: Vec<usize>,
    decode_rows: Vec<usize>,
    released: Vec<RequestId>,
    fail_decode: bool,
}

fn logits_for(tokens: &[u32]) -> Vec<f32> {
    let n = tokens[0] as usize;
    let g = tokens.iter().filter(|&&t| t >= 100).count();
    let next = if g >= n { EOS } else { 100 + (g % 20) as u32 };
    let mut l = vec![0.0f32; VOCAB];
    l[next as usize] = 10.0;
    l
}

impl ModelExecutor for TinyExecutor {
    type Embedding = HostBuffer;

    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        self.vision_batches
            .push(jobs.iter().map(|j| j.patches).collect());
        Ok(jobs
            .iter()
            .map(|j| HostBuffer(vec![0u8; j.patches / 4 * HIDDEN * 2]))
            .collect())
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        self.prefill_rows.push(rows.len());
        for r in rows {
            assert!(!r.blocks.is_empty());
            let placeholders = r.tokens.iter().filter(|&&t| t == IMAGE).count();
            let rows_in_images: usize = r.images.iter().map(|b| b.0.len() / (HIDDEN * 2)).sum();
            assert_eq!(placeholders, rows_in_images);
        }
        Ok(rows.iter().map(|r| logits_for(&r.tokens)).collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        if self.fail_decode {
            return Err(EngineError::Executor("injected".into()));
        }
        self.decode_rows.push(rows.len());
        Ok(rows.iter().map(|_| logits_for(&[0])).collect())
    }

    fn release(&mut self, seq: RequestId) {
        self.released.push(seq);
    }
}

/// Executor that tracks per-sequence histories so decode rows resolve to the same stream.
#[derive(Default)]
struct HistoryExecutor {
    inner: TinyExecutor,
    history: HashMap<RequestId, Vec<u32>>,
}

impl ModelExecutor for HistoryExecutor {
    type Embedding = HostBuffer;

    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        self.inner.vision(jobs)
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        for r in rows {
            self.history.insert(r.seq, r.tokens.clone());
        }
        self.inner.prefill(rows)
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        if self.inner.fail_decode {
            return Err(EngineError::Executor("injected".into()));
        }
        self.inner.decode_rows.push(rows.len());
        let mut out = Vec::new();
        for r in rows {
            let h = self.history.get_mut(&r.seq).expect("prefilled");
            assert_eq!(h.len(), r.position, "decode position follows history");
            h.push(r.token);
            assert!(!r.blocks.is_empty());
            out.push(logits_for(h));
        }
        Ok(out)
    }

    fn release(&mut self, seq: RequestId) {
        self.history.remove(&seq);
        self.inner.release(seq);
    }
}

fn config(max_concurrency: usize, max_context: usize, patches: usize) -> SchedulerConfig {
    SchedulerConfig {
        max_concurrency,
        max_context,
        vision_batch_patches: patches,
        vision_overlap: true,
        spatial_merge: 2,
        vocab_size: VOCAB,
        eos_token_ids: vec![EOS],
        image_token_id: IMAGE,
        embedding_cache_bytes: 1 << 20,
        embedding_hidden: HIDDEN,
        embedding_elem_bytes: 2,
        fallback_seed: 1,
        embedding_release_after_prefill: false,
    }
}

fn layout(num_blocks: usize, block_tokens: usize) -> KvLayout {
    KvLayout {
        num_blocks,
        block_tokens,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    }
}

fn greedy(max_tokens: Option<u32>) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        max_tokens,
        ..SamplingParams::default()
    }
}

fn text_request(generate: u32, prompt_len: usize, max_tokens: Option<u32>) -> EngineRequest {
    let mut prompt = vec![1u32; prompt_len];
    prompt[0] = generate;
    EngineRequest {
        prompt,
        images: Vec::new(),
        params: greedy(max_tokens),
    }
}

struct Outcome {
    tokens: Vec<u32>,
    terminal: EngineEvent,
}

fn drain(rx: &mut UnboundedReceiver<EngineEvent>) -> Outcome {
    let mut tokens = Vec::new();
    loop {
        match rx.try_recv().expect("terminal event delivered") {
            EngineEvent::Token(t) => {
                assert_eq!(t.index as usize, tokens.len());
                tokens.push(t.token);
            }
            other => {
                assert!(rx.try_recv().is_err(), "nothing follows the terminal event");
                return Outcome {
                    tokens,
                    terminal: other,
                };
            }
        }
    }
}

fn run_bounded<E: ModelExecutor>(s: &mut Scheduler<E>) {
    for _ in 0..100_000 {
        if s.is_idle() {
            return;
        }
        s.step();
    }
    panic!("scheduler did not drain");
}

fn submit_all(
    s: &mut Scheduler<HistoryExecutor>,
    count: u32,
    bounded: bool,
) -> (Vec<UnboundedReceiver<EngineEvent>>, Vec<u32>) {
    let mut rxs = Vec::new();
    let mut wanted = Vec::new();
    for i in 0..count {
        let generate = 21 + (i % 5) * 10;
        let (tx, rx) = unbounded_channel();
        let bound = bounded.then_some(generate);
        s.submit(text_request(generate, 5 + (i as usize % 7), bound), tx)
            .unwrap();
        rxs.push(rx);
        wanted.push(generate);
    }
    (rxs, wanted)
}

fn assert_streams(rxs: &mut [UnboundedReceiver<EngineEvent>], wanted: &[u32]) {
    for (rx, &generate) in rxs.iter_mut().zip(wanted) {
        let out = drain(rx);
        let expect: Vec<u32> = (0..generate).map(|k| 100 + k % 20).collect();
        assert_eq!(out.tokens, expect);
        assert!(matches!(out.terminal, EngineEvent::Finished(_)));
    }
}

#[test]
fn bounded_requests_share_the_pool_by_reservation_without_preemption() {
    // The pool holds 8 full contexts; each request reserves a fraction of one.
    let (block_tokens, max_context, slots) = (4usize, 128usize, 64usize);
    let pool_blocks = 8 * (max_context / block_tokens);
    let mut s = Scheduler::new(
        config(slots, max_context, 4096),
        layout(pool_blocks, block_tokens),
        HistoryExecutor::default(),
    );
    let (mut rxs, wanted) = submit_all(&mut s, slots as u32, true);
    let mut peak = 0;
    for _ in 0..100_000 {
        if s.is_idle() {
            break;
        }
        s.step();
        let m = s.metrics();
        assert!(m.reserved_blocks <= pool_blocks);
        peak = peak.max(m.running);
    }
    assert!(
        peak > 8,
        "streams exceed full-context capacity, peak {peak}"
    );
    assert_streams(&mut rxs, &wanted);
    let m = s.metrics();
    assert_eq!(m.pool.preemptions, 0);
    assert_eq!(m.finished, slots as u64);
    assert_eq!(m.pool.used_blocks, 0);
}

#[test]
fn unbounded_requests_reserve_a_full_context_each() {
    let (block_tokens, max_context) = (4usize, 128usize);
    let pool_blocks = 3 * (max_context / block_tokens);
    let mut s = Scheduler::new(
        config(16, max_context, 4096),
        layout(pool_blocks, block_tokens),
        HistoryExecutor::default(),
    );
    let (mut rxs, wanted) = submit_all(&mut s, 12, false);
    let mut peak = 0;
    for _ in 0..100_000 {
        if s.is_idle() {
            break;
        }
        s.step();
        peak = peak.max(s.metrics().running);
    }
    assert_eq!(peak, 3);
    assert_streams(&mut rxs, &wanted);
    assert_eq!(s.metrics().pool.preemptions, 0);
}

#[test]
fn requests_above_the_context_are_refused_and_a_full_one_runs_alone() {
    let (block_tokens, max_context) = (4usize, 128usize);
    let mut s = Scheduler::new(
        config(8, max_context, 4096),
        layout(max_context / block_tokens, block_tokens),
        HistoryExecutor::default(),
    );
    let (tx, _rx) = unbounded_channel();
    let err = s
        .submit(text_request(5, 100, Some(max_context as u32)), tx)
        .unwrap_err();
    assert!(matches!(err, EngineError::ContextOverflow(_)));
    let (tx, mut rx) = unbounded_channel();
    s.submit(text_request(60, 60, Some(68)), tx).unwrap();
    run_bounded(&mut s);
    assert_eq!(drain(&mut rx).tokens.len(), 60);
    assert_eq!(s.metrics().pool.preemptions, 0);
}
