use scriptor_cache::{HostBuffer, ImageGrid, KvLayout, embedding_key};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::sampling::FinishReason;
use scriptor_engine::{
    DecodeRow, DeviceRow, EngineError, EngineEvent, EngineHandle, EngineRequest, ImageInput,
    ModelExecutor, PrefillRow, RequestId, Scheduler, SchedulerConfig, StepRowResult, VisionJob,
};
use std::collections::HashMap;
use std::sync::Arc;
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

/// History executor with a device-resident step: rows prefilled without logits get their first
/// token from the first step, which feeds the last prompt token.
#[derive(Default)]
struct DeviceExecutor {
    inner: HistoryExecutor,
    deferred: usize,
    sampled: usize,
}

impl ModelExecutor for DeviceExecutor {
    type Embedding = HostBuffer;

    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        self.inner.vision(jobs)
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        let all = self.inner.prefill(rows)?;
        self.deferred += rows.iter().filter(|r| !r.logits).count();
        self.sampled += rows.iter().filter(|r| r.logits).count();
        Ok(rows
            .iter()
            .zip(all)
            .map(|(r, l)| if r.logits { l } else { Vec::new() })
            .collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        self.inner.decode(rows)
    }

    fn release(&mut self, seq: RequestId) {
        self.inner.release(seq);
    }

    fn supports_device_step(&self) -> bool {
        true
    }

    fn decode_device(&mut self, rows: &[DeviceRow<'_>]) -> Result<Vec<StepRowResult>, EngineError> {
        let mut out = Vec::new();
        for r in rows {
            let h = self.inner.history.get_mut(&r.row.seq).expect("prefilled");
            if h.len() != r.row.position + 1 {
                assert_eq!(h.len(), r.row.position, "decode position follows history");
                h.push(r.row.token);
            }
            assert_eq!(h.last(), Some(&r.row.token));
            let l = logits_for(h);
            let token = l.iter().position(|&x| x == 10.0).unwrap() as u32;
            out.push(StepRowResult {
                tokens: vec![token],
                margins: vec![10.0],
            });
        }
        Ok(out)
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

fn image(seed: u8, patches: usize) -> ImageInput {
    let grid = ImageGrid {
        t: 1,
        h: (patches / 4) as u32,
        w: 4,
    };
    ImageInput {
        grid,
        key: embedding_key(&[seed], grid),
        payload: Arc::new(vec![seed]),
    }
}

fn image_request(generate: u32, seed: u8, patches: usize) -> EngineRequest {
    let tokens = patches / 4;
    let mut prompt = vec![1u32; 2];
    prompt[0] = generate;
    prompt.extend(std::iter::repeat_n(IMAGE, tokens));
    prompt.push(1);
    EngineRequest {
        prompt,
        images: vec![image(seed, patches)],
        params: greedy(None),
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

#[test]
fn scheduler_completes_64_concurrent_requests_with_exact_token_counts() {
    let mut s = Scheduler::new(
        config(64, 512, 4096),
        layout(4096, 16),
        HistoryExecutor::default(),
    );
    let mut rxs = Vec::new();
    let mut wanted = Vec::new();
    for i in 0..64u32 {
        let generate = 1 + (i % 9) * 3;
        let cap = (i % 4 == 0).then_some(2);
        let (tx, rx) = unbounded_channel();
        s.submit(text_request(generate, 5 + (i as usize % 7), cap), tx)
            .unwrap();
        rxs.push(rx);
        wanted.push((generate, cap));
    }
    run_bounded(&mut s);
    for (rx, (generate, cap)) in rxs.iter_mut().zip(wanted) {
        let out = drain(rx);
        match (cap, out.terminal) {
            (Some(c), EngineEvent::Finished(f)) if c < generate => {
                assert_eq!(f.reason, FinishReason::Length);
                assert_eq!(out.tokens.len(), c as usize);
                assert_eq!(f.completion_tokens, c as usize);
            }
            (_, EngineEvent::Finished(f)) => {
                assert_eq!(f.reason, FinishReason::Stop);
                assert_eq!(out.tokens.len(), generate as usize);
                assert_eq!(f.completion_tokens, generate as usize + 1);
                let expect: Vec<u32> = (0..generate).map(|k| 100 + k % 20).collect();
                assert_eq!(out.tokens, expect);
            }
            (_, other) => panic!("unexpected terminal {other:?}"),
        }
    }
    let m = s.metrics();
    assert_eq!(m.finished, 64);
    assert_eq!(m.pool.used_blocks, 0);
    assert_eq!(m.running, 0);
    assert!(s.executor().inner.decode_rows.iter().any(|&n| n > 32));
    assert!(m.decode_steps > 0 && m.prefill_passes > 0);
}

#[test]
fn scheduler_admits_a_budget_sized_image_alone() {
    let budget = 24_000;
    let mut s = Scheduler::new(
        config(8, 8192, budget),
        layout(2048, 16),
        HistoryExecutor::default(),
    );
    let mut rxs = Vec::new();
    for req in [
        image_request(2, 1, 400),
        image_request(2, 2, budget),
        image_request(2, 3, 400),
        image_request(2, 4, 400),
    ] {
        let (tx, rx) = unbounded_channel();
        s.submit(req, tx).unwrap();
        rxs.push(rx);
    }
    run_bounded(&mut s);
    for rx in &mut rxs {
        let out = drain(rx);
        assert_eq!(out.tokens.len(), 2);
        assert!(matches!(out.terminal, EngineEvent::Finished(_)));
    }
    let batches = &s.executor().inner.vision_batches;
    assert_eq!(batches[0], vec![400]);
    assert_eq!(batches[1], vec![budget]);
    assert_eq!(batches[2], vec![400, 400]);
    assert!(batches.iter().all(|b| b.iter().sum::<usize>() <= budget));
}

#[test]
fn scheduler_refuses_an_image_beyond_the_budget() {
    let mut s = Scheduler::new(
        config(2, 8192, 1000),
        layout(2048, 16),
        HistoryExecutor::default(),
    );
    let (tx, _rx) = unbounded_channel();
    assert!(matches!(
        s.submit(image_request(1, 1, 1004), tx),
        Err(EngineError::ImageTooLarge {
            patches: 1004,
            budget: 1000
        })
    ));
}

#[test]
fn scheduler_queues_by_reservation_and_completes_without_preemption() {
    // 12 blocks of 4 tokens against 8 unbounded sequences: each reserves the whole context, so
    // they run one at a time instead of evicting each other.
    let mut s = Scheduler::new(
        config(8, 64, 4096),
        layout(12, 4),
        HistoryExecutor::default(),
    );
    let mut rxs = Vec::new();
    for _ in 0..8 {
        let (tx, rx) = unbounded_channel();
        s.submit(text_request(10, 4, None), tx).unwrap();
        rxs.push(rx);
    }
    run_bounded(&mut s);
    let expect: Vec<u32> = (0..10).map(|k| 100 + k).collect();
    for rx in &mut rxs {
        let out = drain(rx);
        assert_eq!(out.tokens, expect);
        match out.terminal {
            EngineEvent::Finished(f) => assert_eq!(f.completion_tokens, 11),
            other => panic!("unexpected terminal {other:?}"),
        }
    }
    let m = s.metrics();
    assert_eq!(m.pool.preemptions, 0, "reservation prevents preemption");
    assert_eq!(m.finished, 8);
    assert_eq!(m.failed, 0);
    assert_eq!(m.pool.used_blocks, 0);
}

#[test]
fn scheduler_refuses_context_overflow_with_vllm_wording() {
    let mut s = Scheduler::new(
        config(2, 64, 4096),
        layout(64, 16),
        HistoryExecutor::default(),
    );
    let (tx, _rx) = unbounded_channel();
    let err = s.submit(text_request(1, 60, Some(10)), tx).unwrap_err();
    assert_eq!(
        err.to_string(),
        "This model's maximum context length is 64 tokens. However, you requested 10 output \
         tokens and your prompt contains at least 60 input tokens, for a total of at least 70 \
         tokens. Please reduce the length of the input prompt or the number of requested output \
         tokens."
    );
    let (tx, _rx) = unbounded_channel();
    let err = s.submit(text_request(1, 70, None), tx).unwrap_err();
    assert_eq!(
        err.to_string(),
        "This model's maximum context length is 64 tokens. However, your request has 70 input \
         tokens. Please reduce the length of the input messages."
    );
}

#[test]
fn scheduler_refuses_placeholder_mismatch_and_empty_prompt() {
    let mut s = Scheduler::new(
        config(2, 64, 4096),
        layout(64, 16),
        HistoryExecutor::default(),
    );
    let mut req = image_request(1, 1, 40);
    req.prompt.push(IMAGE);
    let (tx, _rx) = unbounded_channel();
    assert!(matches!(
        s.submit(req, tx),
        Err(EngineError::PlaceholderMismatch {
            placeholders: 11,
            expected: 10
        })
    ));
    let (tx, _rx) = unbounded_channel();
    let empty = EngineRequest {
        prompt: Vec::new(),
        images: Vec::new(),
        params: greedy(None),
    };
    assert_eq!(s.submit(empty, tx), Err(EngineError::EmptyPrompt));
}

#[test]
fn scheduler_cancel_aborts_and_frees_blocks() {
    let mut s = Scheduler::new(
        config(4, 256, 4096),
        layout(64, 16),
        HistoryExecutor::default(),
    );
    let (tx, mut rx) = unbounded_channel();
    let id = s.submit(text_request(60, 8, None), tx).unwrap();
    s.step();
    s.step();
    assert!(s.metrics().pool.used_blocks > 0);
    assert!(s.cancel(id));
    assert!(!s.cancel(id));
    let out = drain(&mut rx);
    assert_eq!(out.terminal, EngineEvent::Aborted);
    assert!(!out.tokens.is_empty());
    assert_eq!(s.metrics().pool.used_blocks, 0);
    assert!(s.is_idle());
    assert!(s.executor().inner.released.contains(&id));
}

#[test]
fn scheduler_cancels_a_request_whose_receiver_dropped() {
    let mut s = Scheduler::new(
        config(4, 256, 4096),
        layout(64, 16),
        HistoryExecutor::default(),
    );
    let (tx, rx) = unbounded_channel();
    s.submit(text_request(60, 8, None), tx).unwrap();
    drop(rx);
    run_bounded(&mut s);
    let m = s.metrics();
    assert_eq!(m.aborted, 1);
    assert_eq!(m.pool.used_blocks, 0);
}

#[test]
fn scheduler_reuses_cached_embeddings_for_repeated_images() {
    let mut s = Scheduler::new(
        config(1, 8192, 4096),
        layout(512, 16),
        HistoryExecutor::default(),
    );
    for _ in 0..3 {
        let (tx, mut rx) = unbounded_channel();
        s.submit(image_request(1, 9, 400), tx).unwrap();
        run_bounded(&mut s);
        assert!(matches!(drain(&mut rx).terminal, EngineEvent::Finished(_)));
    }
    let m = s.metrics();
    assert_eq!(m.vision_images, 1);
    assert_eq!(m.embedding_cache.hits, 2);
    assert_eq!(s.executor().inner.vision_batches.len(), 1);
}

#[test]
fn scheduler_fails_requests_when_the_executor_fails() {
    let mut exec = HistoryExecutor::default();
    exec.inner.fail_decode = true;
    let mut s = Scheduler::new(config(4, 256, 4096), layout(64, 16), exec);
    let (tx, mut rx) = unbounded_channel();
    s.submit(text_request(5, 8, None), tx).unwrap();
    run_bounded(&mut s);
    let out = drain(&mut rx);
    assert!(matches!(
        out.terminal,
        EngineEvent::Failed(EngineError::Executor(_))
    ));
    let m = s.metrics();
    assert_eq!(m.failed, 1);
    assert_eq!(m.pool.used_blocks, 0);
}

#[test]
fn scheduler_records_per_request_stage_timings_in_metrics() {
    let mut s = Scheduler::new(
        config(4, 8192, 4096),
        layout(512, 16),
        HistoryExecutor::default(),
    );
    let (tx, mut rx) = unbounded_channel();
    s.submit(image_request(3, 5, 400), tx).unwrap();
    run_bounded(&mut s);
    let out = drain(&mut rx);
    let EngineEvent::Finished(f) = out.terminal else {
        panic!("finished expected");
    };
    assert_eq!(f.prompt_tokens, 2 + 100 + 1);
    let m = s.metrics();
    assert_eq!(m.vision_batches, 1);
    assert_eq!(m.vision_patches, 400);
    assert_eq!(m.prompt_tokens, 103);
    assert_eq!(m.generation_tokens, 4);
    assert!(m.vision_seconds >= 0.0 && m.prefill_seconds >= 0.0 && m.decode_seconds >= 0.0);
}

#[test]
fn scheduler_handle_serves_requests_from_the_engine_thread() {
    let handle = EngineHandle::spawn(
        config(8, 256, 4096),
        layout(256, 16),
        HistoryExecutor::default(),
    );
    let mut streams = Vec::new();
    for i in 0..8u32 {
        streams.push((i + 1, handle.submit(text_request(i + 1, 6, None)).unwrap()));
    }
    for (generate, stream) in &mut streams {
        let mut n = 0u32;
        loop {
            match stream.events.blocking_recv().expect("terminal event") {
                EngineEvent::Token(_) => n += 1,
                EngineEvent::Finished(f) => {
                    assert_eq!(f.completion_tokens as u32, *generate + 1);
                    break;
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(n, *generate);
    }
    let err = handle.submit(text_request(1, 300, None)).err().unwrap();
    assert!(matches!(err, EngineError::ContextOverflow(_)));
    let doomed = handle.submit(text_request(200, 6, None)).unwrap();
    handle.cancel(doomed.id);
    let mut doomed = doomed;
    loop {
        match doomed.events.blocking_recv().expect("terminal event") {
            EngineEvent::Token(_) => {}
            EngineEvent::Aborted | EngineEvent::Finished(_) => break,
            other => panic!("unexpected {other:?}"),
        }
    }
    handle.shutdown();
    let m = handle.metrics();
    assert!(m.finished >= 8);
    assert_eq!(m.pool.used_blocks, 0);
    assert!(matches!(
        handle.submit(text_request(1, 6, None)),
        Err(EngineError::Closed)
    ));
}

fn outcomes<E: ModelExecutor>(s: &mut Scheduler<E>, requests: &[EngineRequest]) -> Vec<String> {
    let mut rxs = Vec::new();
    for r in requests {
        let (tx, rx) = unbounded_channel();
        s.submit(r.clone(), tx).unwrap();
        rxs.push(rx);
    }
    run_bounded(s);
    rxs.iter_mut()
        .map(|rx| {
            let out = drain(rx);
            match out.terminal {
                EngineEvent::Finished(f) => {
                    format!("{:?} {:?} {}", out.tokens, f.reason, f.completion_tokens)
                }
                other => panic!("unexpected terminal {other:?}"),
            }
        })
        .collect()
}

#[test]
fn device_admission_without_host_logits_matches_the_host_sampled_path() {
    let mut requests = vec![
        text_request(0, 4, None),
        text_request(1, 3, None),
        text_request(7, 6, None),
        text_request(7, 6, Some(1)),
        text_request(7, 6, Some(3)),
        text_request(25, 5, None),
    ];
    requests.push(EngineRequest {
        params: SamplingParams {
            logprobs: Some(2),
            ..greedy(None)
        },
        ..text_request(5, 4, None)
    });
    let mut host = Scheduler::new(
        config(8, 512, 4096),
        layout(512, 16),
        HistoryExecutor::default(),
    );
    let want = outcomes(&mut host, &requests);
    let mut device = Scheduler::new(
        config(8, 512, 4096),
        layout(512, 16),
        DeviceExecutor::default(),
    );
    let got = outcomes(&mut device, &requests);
    assert_eq!(got, want);
    assert_eq!(device.executor().deferred, requests.len() - 1);
    assert_eq!(
        device.executor().sampled,
        1,
        "log-probability rows stay on the host path"
    );
    assert_eq!(device.metrics().pool.used_blocks, 0);
}

/// History executor that records the sequence ids of every decode call and prefill call.
#[derive(Default)]
struct OrderExecutor {
    inner: HistoryExecutor,
    decode_calls: Vec<Vec<RequestId>>,
    prefill_order: Vec<RequestId>,
}

impl ModelExecutor for OrderExecutor {
    type Embedding = HostBuffer;

    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        self.inner.vision(jobs)
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        self.prefill_order.extend(rows.iter().map(|r| r.seq));
        self.inner.prefill(rows)
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        self.decode_calls.push(rows.iter().map(|r| r.seq).collect());
        self.inner.decode(rows)
    }

    fn release(&mut self, seq: RequestId) {
        self.inner.release(seq);
    }
}

#[test]
fn reserve_mode_keeps_submission_admission_and_ascending_decode_row_order() {
    let mut s = Scheduler::new(config(4, 256, 64), layout(64, 16), OrderExecutor::default());
    let lengths = [9u32, 3, 14, 5, 2, 11, 7, 4, 12, 6];
    let mut rxs = Vec::new();
    for (i, &n) in lengths.iter().enumerate() {
        let (tx, rx) = unbounded_channel();
        s.submit(text_request(n, 4 + i % 3, None), tx).unwrap();
        rxs.push(rx);
    }
    run_bounded(&mut s);
    for (rx, &n) in rxs.iter_mut().zip(&lengths) {
        assert_eq!(drain(rx).tokens.len(), n as usize);
    }
    let ex = s.executor();
    assert_eq!(
        ex.prefill_order,
        (0..lengths.len() as u64).collect::<Vec<_>>()
    );
    assert!(!ex.decode_calls.is_empty());
    for call in &ex.decode_calls {
        assert!(call.windows(2).all(|w| w[0] < w[1]), "{call:?}");
        assert!(call.len() <= 4);
    }
    assert_eq!(s.metrics().pool.preemptions, 0);
}
