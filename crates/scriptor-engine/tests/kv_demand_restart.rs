//! Restart behaviour of the demand KV policy: seeded sampling replays, the embedding budget
//! blocks admission without dropping embeddings, and a speculative device-step executor
//! re-seeds after an eviction. CPU executors check scheduler control flow only.
#![cfg(feature = "kv-demand-diagnostic")]

use scriptor_cache::{HostBuffer, ImageGrid, KvLayout, embedding_key};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::{
    DecodeRow, DeviceRow, EngineError, EngineEvent, EngineRequest, ImageInput, ModelExecutor,
    PrefillRow, RequestId, Scheduler, SchedulerConfig, StepRowResult, VisionJob, VisionToken,
};
use scriptor_spec::KvPolicy;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

const VOCAB: usize = 128;
const CONTEXT: usize = 160;
const LENS: [usize; 5] = [8, 40, 12, 30, 9];

fn mix(h: u64, t: u32) -> u64 {
    let mut z = h ^ u64::from(t)
        .wrapping_add(1)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn hash_of(history: &[u32]) -> u64 {
    history.iter().fold(0, |h, &t| mix(h, t))
}

fn config(slots: usize, embedding_bytes: usize) -> SchedulerConfig {
    SchedulerConfig {
        max_concurrency: slots,
        max_context: CONTEXT,
        vision_batch_patches: 16,
        vision_overlap: true,
        spatial_merge: 2,
        vocab_size: VOCAB,
        eos_token_ids: vec![0],
        image_token_id: 50,
        embedding_cache_bytes: embedding_bytes,
        embedding_hidden: 4,
        embedding_elem_bytes: 2,
        fallback_seed: 1,
        embedding_release_after_prefill: false,
    }
}

fn layout(blocks: usize) -> KvLayout {
    KvLayout {
        num_blocks: blocks,
        block_tokens: 16,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    }
}

/// What one request's channel delivered since its last restart.
#[derive(Default)]
struct Seen {
    tokens: Vec<u32>,
    restarts: usize,
    finished: bool,
    preemptions: u32,
}

fn drain(rx: &mut UnboundedReceiver<EngineEvent>, seen: &mut Seen) {
    while let Ok(ev) = rx.try_recv() {
        match ev {
            EngineEvent::Token(t) => seen.tokens.push(t.token),
            EngineEvent::Restarted => {
                seen.tokens.clear();
                seen.restarts += 1;
            }
            EngineEvent::Finished(f) => {
                assert!(!seen.finished, "one terminal event");
                seen.finished = true;
                seen.preemptions = f.preemptions;
            }
            other => panic!("unexpected event {other:?}"),
        }
    }
}

fn run_to_idle<E: ModelExecutor>(
    s: &mut Scheduler<E>,
    rxs: &mut [UnboundedReceiver<EngineEvent>],
    seen: &mut [Seen],
    blocks: usize,
) {
    for _ in 0..4000 {
        s.step();
        assert!(s.metrics().pool.used_blocks <= blocks);
        for (rx, sn) in rxs.iter_mut().zip(seen.iter_mut()) {
            drain(rx, sn);
        }
        if s.is_idle() {
            break;
        }
    }
    assert!(s.is_idle(), "every request terminates");
    assert_eq!(s.metrics().pool.sequences, 0);
}

fn text_request(prompt_len: usize, params: SamplingParams) -> EngineRequest {
    let mut prompt = vec![1u32; prompt_len];
    prompt[0] = 1000;
    EngineRequest {
        prompt,
        images: Vec::new(),
        params,
    }
}

// ---------------------------------------------------------------- T5: seeded sampling replay

/// Executor whose next-token distribution is a fixed function of the whole history, so any
/// divergence between a restarted and an uninterrupted stream comes from the sampler.
#[derive(Default)]
struct Hashed {
    seqs: HashMap<RequestId, Vec<u32>>,
}

fn hashed_logits(history: &[u32]) -> Vec<f32> {
    let h = hash_of(history);
    let mut l = vec![-1.0e4f32; VOCAB];
    for (v, slot) in l.iter_mut().enumerate().skip(100).take(12) {
        *slot = (mix(h, v as u32) % 1000) as f32 / 250.0;
    }
    l
}

impl ModelExecutor for Hashed {
    type Embedding = HostBuffer;

    fn vision(&mut self, _jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        Ok(Vec::new())
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        for r in rows {
            assert!(
                self.seqs.insert(r.seq, r.tokens.clone()).is_none(),
                "prefill of a live sequence"
            );
        }
        Ok(rows.iter().map(|r| hashed_logits(&r.tokens)).collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        Ok(rows
            .iter()
            .map(|r| {
                let h = self.seqs.get_mut(&r.seq).expect("prefilled");
                assert_eq!(h.len(), r.position);
                h.push(r.token);
                hashed_logits(h)
            })
            .collect())
    }

    fn release(&mut self, seq: RequestId) {
        self.seqs.remove(&seq);
    }
}

fn sampled(seed: Option<u64>, prompt_len: usize) -> SamplingParams {
    SamplingParams {
        temperature: 0.8,
        top_p: 0.95,
        seed,
        max_tokens: Some((CONTEXT - prompt_len) as u32),
        ..SamplingParams::default()
    }
}

/// The stream request `id` produces alone under the reserve policy.
fn uninterrupted_sampled(id: RequestId, prompt_len: usize, seed: Option<u64>) -> Vec<u32> {
    let mut s = Scheduler::new(config(1, 1 << 20), layout(64), Hashed::default());
    let (tx, rx) = unbounded_channel();
    s.submit_restartable_with_id(
        id,
        text_request(prompt_len, sampled(seed, prompt_len)),
        tx,
        false,
    )
    .unwrap();
    let mut rxs = [rx];
    let mut seen = [Seen::default()];
    run_to_idle(&mut s, &mut rxs, &mut seen, 64);
    let [only] = seen;
    only.tokens
}

fn restarted_sampled(seed_of: impl Fn(RequestId) -> Option<u64>) {
    let blocks = 11;
    let mut s = Scheduler::new(config(3, 1 << 20), layout(blocks), Hashed::default());
    s.set_kv_policy(KvPolicy::DemandDiagnostic);
    let mut rxs = Vec::new();
    let mut seen = Vec::new();
    for (i, len) in LENS.iter().enumerate() {
        let (tx, rx) = unbounded_channel();
        let id = i as RequestId;
        s.submit_restartable_with_id(id, text_request(*len, sampled(seed_of(id), *len)), tx, true)
            .unwrap();
        rxs.push(rx);
        seen.push(Seen::default());
    }
    run_to_idle(&mut s, &mut rxs, &mut seen, blocks);
    assert!(
        seen.iter().map(|s| s.restarts).sum::<usize>() >= 1,
        "the pool forces at least one restart"
    );
    for (i, len) in LENS.iter().enumerate() {
        let id = i as RequestId;
        assert!(seen[i].finished);
        assert_eq!(
            seen[i].tokens,
            uninterrupted_sampled(id, *len, seed_of(id)),
            "request {id} equals its uninterrupted stream"
        );
    }
}

#[test]
fn a_restart_under_seeded_temperature_sampling_replays_the_uninterrupted_stream() {
    restarted_sampled(|id| Some(40 + id));
}

#[test]
fn a_restart_of_an_unseeded_request_replays_the_stream_of_its_fallback_seed() {
    restarted_sampled(|_| None);
}

// ---------------------------------------------------- T8: embedding bound keeps embeddings

const IMAGE: u32 = 50;
const IMG: usize = 32;
const END: usize = 150;
const PAD: usize = 40;

/// CPU executor whose vision batch is collected only when the scheduler waits for it; it
/// counts the images it embeds.
#[derive(Default)]
struct Gated {
    next: VisionToken,
    flights: HashMap<VisionToken, Vec<usize>>,
    embedded: usize,
}

fn gated_logits(len: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; VOCAB];
    l[if len >= END { 0 } else { 100 } as usize] = 10.0;
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
        let token = self.next;
        self.next += 1;
        self.embedded += jobs.len();
        self.flights
            .insert(token, jobs.iter().map(|j| j.patches).collect());
        Ok(token)
    }

    fn vision_ready(&mut self, _token: VisionToken) -> Result<bool, EngineError> {
        Ok(false)
    }

    fn vision_finish(&mut self, token: VisionToken) -> Result<Vec<HostBuffer>, EngineError> {
        let patches = self.flights.remove(&token).expect("begun");
        Ok(patches
            .iter()
            .map(|p| HostBuffer(vec![0u8; p / 4 * 4 * 2]))
            .collect())
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        for r in rows {
            let placeholders = r.tokens.iter().filter(|&&t| t == IMAGE).count();
            let embedded: usize = r.images.iter().map(|b| b.0.len() / 8).sum();
            assert_eq!(
                placeholders, embedded,
                "every placeholder has its embedding"
            );
        }
        Ok(rows.iter().map(|r| gated_logits(r.tokens.len())).collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        Ok(rows.iter().map(|r| gated_logits(r.position + 1)).collect())
    }
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    }
}

fn image_request(seed: u8) -> EngineRequest {
    let grid = ImageGrid { t: 1, h: 4, w: 4 };
    let mut prompt = vec![1u32; PAD];
    prompt.extend(std::iter::repeat_n(IMAGE, 4));
    prompt.push(1);
    EngineRequest {
        prompt,
        images: vec![ImageInput {
            grid,
            key: embedding_key(&[seed], grid),
            payload: Arc::new(vec![seed]),
        }],
        params: greedy(),
    }
}

/// The oldest text request decodes while two image requests follow it; returns each
/// request's output, preemption count, and the images the executor embedded.
fn gated_run(
    policy: KvPolicy,
    blocks: usize,
    embedding_bytes: usize,
) -> (Vec<Vec<u32>>, Vec<u32>, usize) {
    let mut s = Scheduler::new(config(3, embedding_bytes), layout(blocks), Gated::default());
    s.set_kv_policy(policy);
    let submit = |s: &mut Scheduler<Gated>, id: RequestId, request: EngineRequest| {
        let (tx, rx) = unbounded_channel();
        s.submit_restartable_with_id(id, request, tx, true).unwrap();
        rx
    };
    let mut rxs = vec![submit(&mut s, 0, text_request(3, greedy()))];
    s.step();
    rxs.push(submit(&mut s, 1, image_request(1)));
    rxs.push(submit(&mut s, 2, image_request(2)));
    let mut seen: Vec<Seen> = rxs.iter().map(|_| Seen::default()).collect();
    run_to_idle(&mut s, &mut rxs, &mut seen, blocks);
    assert!(seen.iter().all(|s| s.finished));
    if embedding_bytes > 0 {
        assert_eq!(s.pinned_embedding_bytes(), 0);
    }
    let embedded = s.executor().embedded;
    (
        seen.iter().map(|s| s.tokens.clone()).collect(),
        seen.iter().map(|s| s.preemptions).collect(),
        embedded,
    )
}

#[test]
fn a_request_blocked_by_the_embedding_budget_keeps_its_embeddings_and_its_output() {
    let (free_out, free_preempt, free_embedded) = gated_run(KvPolicy::Reserve, 40, 4 * IMG);
    assert_eq!(free_preempt, vec![0, 0, 0]);
    assert_eq!(free_embedded, 2);
    let (out, preempt, embedded) = gated_run(KvPolicy::DemandDiagnostic, 12, IMG);
    assert!(
        preempt[2] >= 1,
        "the request holding blocks while blocked on the budget was sent back"
    );
    assert_eq!(embedded, 2, "no image is embedded a second time");
    assert_eq!(out, free_out, "outputs equal the unblocked run");
}

// ------------------------------------- T9: speculative device-step executor re-seeds on restart

const DEPTH: usize = 3;
/// Extra lookahead tokens of a surge step.
const SURGE: usize = 40;

#[derive(Default)]
struct DeviceSeq {
    history: Vec<u32>,
    prompt_len: usize,
    draft_state: usize,
    stepped: bool,
}

/// Speculative device-step executor: prefill rows without logits leave the first token to the
/// device step, which emits one to `DEPTH + 1` tokens per step, the count driven by per-sequence
/// draft state. A prefill that joins running sequences makes the following step's lookahead
/// exceed what admission planned for, which forces an eviction of the newest sequence while its
/// first token is still pending on the device. The next token depends only on the history, so output is independent of how many
/// tokens a step emits.
#[derive(Default)]
struct Speculative {
    seqs: HashMap<RequestId, DeviceSeq>,
    prompt_lens: HashMap<RequestId, usize>,
    released_before_first_step: usize,
    /// Set when a prefill joins running sequences: the next decode step then writes KV well
    /// past the lookahead admission planned for.
    surge: bool,
}

fn spec_next(history: &[u32]) -> u32 {
    100 + (hash_of(history) % 20) as u32
}

impl ModelExecutor for Speculative {
    type Embedding = HostBuffer;

    fn vision(&mut self, _jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        Ok(Vec::new())
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        self.surge = !self.seqs.is_empty();
        for r in rows {
            assert!(!r.logits, "the device step chooses the first token");
            assert_eq!(
                r.tokens.len(),
                self.prompt_lens[&r.seq],
                "a restarted sequence prefills its prompt only"
            );
            let seq = DeviceSeq {
                history: r.tokens.clone(),
                prompt_len: r.tokens.len(),
                ..DeviceSeq::default()
            };
            assert!(
                self.seqs.insert(r.seq, seq).is_none(),
                "draft state of a live sequence is never reused"
            );
        }
        Ok(rows.iter().map(|_| Vec::new()).collect())
    }

    fn decode(&mut self, _rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        panic!("every row runs on the device step");
    }

    fn supports_device_step(&self) -> bool {
        true
    }

    fn draft_depth(&self) -> usize {
        DEPTH
    }

    fn lookahead_tokens(&self) -> usize {
        if self.surge { DEPTH + SURGE } else { DEPTH }
    }

    fn decode_device(&mut self, rows: &[DeviceRow<'_>]) -> Result<Vec<StepRowResult>, EngineError> {
        self.surge = false;
        let mut out = Vec::new();
        for r in rows {
            let seq = self.seqs.get_mut(&r.row.seq).expect("prefilled");
            if !seq.stepped {
                assert_eq!(seq.draft_state, 0, "draft state starts fresh");
                assert_eq!(
                    seq.history.len(),
                    seq.prompt_len,
                    "the first step follows the prompt"
                );
                assert_eq!(r.params.step, 0, "the sampler restarts from step zero");
            }
            seq.stepped = true;
            assert_eq!(seq.history.len(), r.row.position + 1);
            assert_eq!(seq.history.last(), Some(&r.row.token));
            assert_eq!(
                r.params.step as usize,
                seq.history.len() - seq.prompt_len,
                "sampler step follows the tokens generated"
            );
            let emit = 1 + seq.draft_state % (DEPTH + 1);
            seq.draft_state += 1;
            let mut tokens = Vec::new();
            for _ in 0..emit {
                let t = spec_next(&seq.history);
                seq.history.push(t);
                tokens.push(t);
            }
            let margins = vec![1.0; tokens.len()];
            out.push(StepRowResult { tokens, margins });
        }
        Ok(out)
    }

    fn release(&mut self, seq: RequestId) {
        if let Some(state) = self.seqs.remove(&seq)
            && !state.stepped
        {
            self.released_before_first_step += 1;
        }
    }
}

fn device_run(policy: KvPolicy, slots: usize, blocks: usize, lens: &[usize]) -> (Vec<Seen>, usize) {
    let mut exec = Speculative::default();
    for (i, len) in lens.iter().enumerate() {
        exec.prompt_lens.insert(i as RequestId, *len);
    }
    let mut s = Scheduler::new(config(slots, 1 << 20), layout(blocks), exec);
    s.set_kv_policy(policy);
    let mut rxs = Vec::new();
    let mut seen = Vec::new();
    for (i, len) in lens.iter().enumerate() {
        let (tx, rx) = unbounded_channel();
        let request = text_request(
            *len,
            SamplingParams {
                temperature: 0.0,
                max_tokens: Some((CONTEXT - len) as u32),
                ..SamplingParams::default()
            },
        );
        s.submit_restartable_with_id(i as RequestId, request, tx, true)
            .unwrap();
        rxs.push(rx);
        seen.push(Seen::default());
    }
    run_to_idle(&mut s, &mut rxs, &mut seen, blocks);
    assert!(s.executor().seqs.is_empty(), "executor state released");
    let unstepped = s.executor().released_before_first_step;
    (seen, unstepped)
}

#[test]
fn evicting_a_sequence_awaiting_its_first_device_step_reseeds_draft_and_sampler_state() {
    let (seen, unstepped) = device_run(KvPolicy::DemandDiagnostic, 3, 14, &LENS);
    assert!(
        seen.iter().map(|s| s.restarts).sum::<usize>() >= 1,
        "the pool forces at least one restart"
    );
    assert!(
        unstepped >= 1,
        "a sequence is evicted while its first token is still pending on the device"
    );
    for (i, len) in LENS.iter().enumerate() {
        let (alone, _) = device_run(KvPolicy::Reserve, 1, 64, &[*len]);
        assert!(seen[i].finished);
        assert_eq!(
            seen[i].tokens, alone[0].tokens,
            "request {i} equals its uninterrupted run"
        );
    }
}
