//! Releasing image embeddings after prefill: the budget charge, the generated tokens and the
//! re-resolution of a sequence that is recomputed.

use crate::error::EngineError;
use crate::request::SamplingParams;
use crate::scheduler::{
    DecodeRow, ModelExecutor, PrefillRow, Scheduler, SchedulerConfig, VisionJob,
};
use crate::sequence::{EngineEvent, Phase};
use crate::{EngineRequest, ImageInput, RequestId};
use scriptor_cache::{HostBuffer, ImageGrid, KvLayout, embedding_key};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

const VOCAB: usize = 128;
const EOS: u32 = 0;
const IMAGE: u32 = 50;
const HIDDEN: usize = 4;
/// Bytes of the embedding of a 16-patch image.
const IMG: usize = 32;
const END: usize = 14;

#[derive(Default)]
struct Exec {
    vision_images: usize,
    seeds: HashMap<RequestId, u32>,
}

fn logits(seed: u32, len: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; VOCAB];
    let next = if len >= END {
        EOS
    } else {
        100 + (seed + len as u32) % 20
    };
    l[next as usize] = 10.0;
    l
}

impl ModelExecutor for Exec {
    type Embedding = HostBuffer;

    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        self.vision_images += jobs.len();
        Ok(jobs
            .iter()
            .map(|j| HostBuffer(vec![j.payload[0]; j.patches / 4 * HIDDEN * 2]))
            .collect())
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        Ok(rows
            .iter()
            .map(|r| {
                let seed: u32 = r.images.iter().map(|b| b.0[0] as u32).sum();
                self.seeds.insert(r.seq, seed);
                logits(seed, r.tokens.len())
            })
            .collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        Ok(rows
            .iter()
            .map(|r| logits(self.seeds[&r.seq], r.position + 1))
            .collect())
    }

    fn release(&mut self, seq: RequestId) {
        self.seeds.remove(&seq);
    }
}

fn scheduler(max_concurrency: usize, budget: usize, release: bool) -> Scheduler<Exec> {
    let config = SchedulerConfig {
        max_concurrency,
        max_context: 256,
        vision_batch_patches: 64,
        vision_overlap: false,
        spatial_merge: 2,
        vocab_size: VOCAB,
        eos_token_ids: vec![EOS],
        image_token_id: IMAGE,
        embedding_cache_bytes: budget,
        embedding_hidden: HIDDEN,
        embedding_elem_bytes: 2,
        fallback_seed: 1,
        embedding_release_after_prefill: release,
    };
    let layout = KvLayout {
        num_blocks: 64,
        block_tokens: 16,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    };
    Scheduler::new(config, layout, Exec::default())
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

fn submit(s: &mut Scheduler<Exec>, seed: u8) -> (RequestId, Rx) {
    let (tx, rx) = unbounded_channel();
    let id = s.submit(request(seed), tx).unwrap();
    (id, rx)
}

fn tokens(rx: &mut Rx) -> Vec<u32> {
    let mut out = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        if let EngineEvent::Token(t) = ev {
            out.push(t.token);
        }
    }
    out
}

/// Steps until `done` holds, failing after `limit` steps.
fn step_until(s: &mut Scheduler<Exec>, limit: usize, done: impl Fn(&Scheduler<Exec>) -> bool) {
    for _ in 0..limit {
        if done(s) {
            return;
        }
        s.step();
    }
    assert!(done(s), "condition not reached in {limit} steps");
}

fn prefilled(s: &Scheduler<Exec>) -> bool {
    s.metrics().prefill_passes >= 1
}

fn all_decoding(s: &Scheduler<Exec>, n: usize) -> bool {
    s.active.len() == n && s.active.values().all(|q| q.phase == Phase::Running)
}

#[test]
fn release_frees_the_charge_after_prefill() {
    let mut s = scheduler(2, 2 * IMG, true);
    let _a = submit(&mut s, 1);
    let _b = submit(&mut s, 2);
    step_until(&mut s, 50, |s| all_decoding(s, 2));
    assert_eq!(s.pinned_embedding_bytes(), 0);
    let m = s.metrics();
    assert_eq!(m.embedding_released, 2);
    assert_eq!(m.embedding_released_bytes, 2 * IMG as u64);
    assert!(s.embedding_reserved.is_empty());
    s.run_until_idle();
    assert_eq!(s.executor().vision_images, 2);
}

#[test]
fn the_charge_is_held_until_the_sequence_ends_when_the_switch_is_off() {
    let mut s = scheduler(2, 2 * IMG, false);
    let _a = submit(&mut s, 1);
    let _b = submit(&mut s, 2);
    step_until(&mut s, 50, |s| all_decoding(s, 2));
    assert_eq!(s.pinned_embedding_bytes(), 2 * IMG);
    let m = s.metrics();
    assert_eq!((m.embedding_released, m.embedding_released_bytes), (0, 0));
    s.run_until_idle();
    assert_eq!(s.pinned_embedding_bytes(), 0);
}

/// Runs three single-image requests through a budget of two images; returns each request's
/// tokens and the most sequences decoding at once.
fn three_through_two(release: bool) -> (Vec<Vec<u32>>, usize) {
    let mut s = scheduler(3, 2 * IMG, release);
    let mut rxs: Vec<Rx> = (1..=3u8).map(|seed| submit(&mut s, seed).1).collect();
    let mut peak = 0;
    for _ in 0..200 {
        if s.is_idle() {
            break;
        }
        s.step();
        peak = peak.max(s.metrics().active_running);
    }
    assert!(s.is_idle());
    (rxs.iter_mut().map(tokens).collect(), peak)
}

#[test]
fn a_released_charge_lets_a_further_request_embed_while_the_others_decode() {
    let (off_tokens, off_peak) = three_through_two(false);
    let (on_tokens, on_peak) = three_through_two(true);
    assert_eq!(on_tokens, off_tokens);
    assert!(off_tokens.iter().all(|t| !t.is_empty()));
    assert!(off_peak < 3, "off peak {off_peak}");
    assert_eq!(on_peak, 3);
}

#[test]
fn a_released_embedding_is_resolved_from_the_cache_after_preemption() {
    let mut reference = scheduler(1, 4 * IMG, true);
    let (_, mut ref_rx) = submit(&mut reference, 7);
    reference.run_until_idle();
    let want = tokens(&mut ref_rx);

    let mut s = scheduler(1, 4 * IMG, true);
    let (id, mut rx) = submit(&mut s, 7);
    step_until(&mut s, 50, prefilled);
    assert!(s.active[&id].embeds.iter().all(Option::is_none));
    s.pool.release(id).unwrap();
    s.requeue_preempted(id);
    let queued = &s.waiting[0];
    assert!(queued.embeds.iter().all(Option::is_some));
    assert_eq!(queued.phase, Phase::Prefill);
    s.run_until_idle();
    assert_eq!(s.executor().vision_images, 1);
    assert_eq!(tokens(&mut rx), want);
}

#[test]
fn an_evicted_released_embedding_is_embedded_again_after_preemption() {
    let mut reference = scheduler(2, IMG, true);
    let (_, mut ref_rx) = submit(&mut reference, 7);
    reference.run_until_idle();
    let want = tokens(&mut ref_rx);

    let mut s = scheduler(2, IMG, true);
    let (a, mut rx) = submit(&mut s, 7);
    let _b = submit(&mut s, 9);
    step_until(&mut s, 50, |s| s.executor().vision_images >= 2);
    assert!(
        s.active.contains_key(&a),
        "request finished before preemption"
    );
    assert!(s.active[&a].embeds.iter().all(Option::is_none));
    s.pool.release(a).unwrap();
    s.requeue_preempted(a);
    let queued = s.waiting.iter().find(|w| w.id == a).expect("requeued");
    assert!(queued.embeds.iter().all(Option::is_none));
    assert_eq!(queued.phase, Phase::Vision);
    s.run_until_idle();
    assert_eq!(s.executor().vision_images, 3);
    assert_eq!(tokens(&mut rx), want);
}
