//! Progress of the oldest request under the demand policy while younger requests hold prompt
//! KV in the vision or prefill phase. The vision batch finishes only when the scheduler waits
//! for it, so younger requests keep their blocks until the oldest request has to wait.
#![cfg(feature = "kv-demand-diagnostic")]

use scriptor_cache::{HostBuffer, ImageGrid, KvLayout, embedding_key};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::{
    DecodeRow, EngineError, EngineEvent, EngineRequest, ImageInput, ModelExecutor, PrefillRow,
    RequestId, Scheduler, SchedulerConfig, VisionJob, VisionToken,
};
use scriptor_spec::KvPolicy;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

const VOCAB: usize = 128;
const EOS: u32 = 0;
const IMAGE: u32 = 50;
const HIDDEN: usize = 4;
/// Bytes of the embedding of a 16-patch image.
const IMG: usize = 32;
const BLOCKS: usize = 12;
const CONTEXT: usize = 160;
/// Sequence length at which generation ends.
const END: usize = 150;
/// Text tokens before the image in a younger request: three blocks with the image.
const PAD: usize = 40;

/// CPU executor whose vision batch is never ready to poll, so it is collected only when the
/// scheduler waits for it.
#[derive(Default)]
struct Gated {
    next: VisionToken,
    flights: HashMap<VisionToken, Vec<usize>>,
}

fn logits(len: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; VOCAB];
    l[if len >= END { EOS } else { 100 } as usize] = 10.0;
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
        let token = self.next;
        self.next += 1;
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
            .map(|p| HostBuffer(vec![0u8; p / 4 * HIDDEN * 2]))
            .collect())
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        Ok(rows.iter().map(|r| logits(r.tokens.len())).collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        Ok(rows.iter().map(|r| logits(r.position + 1)).collect())
    }
}

/// A demand-policy scheduler with `embedding_bytes` of embedding budget and one image per
/// vision batch.
fn scheduler(embedding_bytes: usize) -> Scheduler<Gated> {
    let config = SchedulerConfig {
        max_concurrency: 3,
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
        num_blocks: BLOCKS,
        block_tokens: 16,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    };
    let mut s = Scheduler::new(config, layout, Gated::default());
    s.set_kv_policy(KvPolicy::DemandDiagnostic);
    s
}

fn params() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    }
}

fn text_request() -> EngineRequest {
    EngineRequest {
        prompt: vec![1, 2, 1],
        images: Vec::new(),
        params: params(),
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
        params: params(),
    }
}

/// What a request's stream ended with.
#[derive(Default)]
struct Outcome {
    finished: bool,
    preemptions: u32,
}

fn submit(
    s: &mut Scheduler<Gated>,
    id: RequestId,
    request: EngineRequest,
) -> UnboundedReceiver<EngineEvent> {
    let (tx, rx) = unbounded_channel();
    s.submit_restartable_with_id(id, request, tx, true).unwrap();
    rx
}

fn drain(rx: &mut UnboundedReceiver<EngineEvent>, out: &mut Outcome) {
    while let Ok(ev) = rx.try_recv() {
        match ev {
            EngineEvent::Token(_) | EngineEvent::Restarted => {}
            EngineEvent::Finished(f) => {
                out.finished = true;
                out.preemptions = f.preemptions;
            }
            other => panic!("the request must not end with {other:?}"),
        }
    }
}

/// Runs the oldest text request until it decodes, adds two younger image requests that take
/// the free blocks, and steps to idle. Every request must finish: the oldest never fails when
/// it crosses block boundaries while the younger ones hold prompt blocks.
fn run(embedding_bytes: usize) -> (Vec<Outcome>, scriptor_engine::EngineMetrics) {
    let mut s = scheduler(embedding_bytes);
    let mut rxs = vec![submit(&mut s, 0, text_request())];
    s.step();
    rxs.push(submit(&mut s, 1, image_request(1)));
    rxs.push(submit(&mut s, 2, image_request(2)));
    let mut outs: Vec<Outcome> = rxs.iter().map(|_| Outcome::default()).collect();
    for _ in 0..4000 {
        s.step();
        assert!(s.metrics().pool.used_blocks <= BLOCKS);
        if embedding_bytes > 0 {
            assert!(s.pinned_embedding_bytes() <= embedding_bytes);
        }
        for (rx, out) in rxs.iter_mut().zip(&mut outs) {
            drain(rx, out);
        }
        if s.is_idle() {
            break;
        }
    }
    assert!(s.is_idle(), "every request terminates");
    assert_eq!(s.metrics().pool.sequences, 0);
    assert_eq!(s.pinned_embedding_bytes(), 0);
    (outs, s.metrics())
}

#[test]
fn the_oldest_decoder_crosses_block_boundaries_while_younger_requests_hold_prompt_kv() {
    let (outs, m) = run(4 * IMG);
    assert!(outs.iter().all(|o| o.finished));
    assert_eq!(
        outs[0].preemptions, 0,
        "the oldest request is never evicted"
    );
    assert!(
        outs[2].preemptions >= 1,
        "the youngest holder gave its blocks back"
    );
    assert_eq!(
        m.restarts + m.yields,
        outs.iter().map(|o| u64::from(o.preemptions)).sum::<u64>()
    );
}

#[test]
fn the_oldest_decoder_progresses_when_the_embedding_budget_holds_one_image() {
    let (outs, m) = run(IMG);
    assert!(outs.iter().all(|o| o.finished));
    assert_eq!(
        outs[0].preemptions, 0,
        "the oldest request is never evicted"
    );
    assert!(
        outs[2].preemptions >= 1,
        "the holder waiting on the budget gave its blocks back"
    );
    assert!(
        m.yields >= 1,
        "a vision or prefill holder yielded its blocks"
    );
}
