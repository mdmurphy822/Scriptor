use scriptor_cache::{HostBuffer, ImageGrid, KvLayout, embedding_key};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::{
    CancelReason, DecodeRow, EngineError, EngineEvent, EngineRequest, ImageInput, ModelExecutor,
    PrefillRow, RequestId, Scheduler, SchedulerConfig, VisionJob,
};
use std::sync::Arc;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

const VOCAB: usize = 128;
const EOS: u32 = 0;
const IMAGE: u32 = 50;
const HIDDEN: usize = 4;
/// Bytes of the embedding of a 16-patch image: 4 rows of `HIDDEN` bf16 elements.
const IMG: usize = 32;
/// The budget: two 16-patch images.
const E: usize = 2 * IMG;
const END: usize = 14;

#[derive(Default)]
struct Exec {
    vision_images: usize,
    fail_vision: bool,
}

/// Logits that end a sequence once it holds `END` tokens.
fn logits(len: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; VOCAB];
    l[if len >= END { EOS } else { 100 } as usize] = 10.0;
    l
}

impl ModelExecutor for Exec {
    type Embedding = HostBuffer;

    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        if self.fail_vision {
            return Err(EngineError::Executor("injected".into()));
        }
        self.vision_images += jobs.len();
        Ok(jobs
            .iter()
            .map(|j| HostBuffer(vec![j.payload[0]; j.patches / 4 * HIDDEN * 2]))
            .collect())
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        Ok(rows.iter().map(|r| logits(r.tokens.len())).collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        Ok(rows.iter().map(|r| logits(r.position + 1)).collect())
    }
}

fn scheduler(max_concurrency: usize, budget: usize) -> Scheduler<Exec> {
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
        embedding_release_after_prefill: false,
    };
    let layout = KvLayout {
        num_blocks: 64,
        block_tokens: 16,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    };
    Scheduler::new(config, layout, Exec::default())
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

fn request(seeds: &[u8], patches: usize) -> EngineRequest {
    let mut prompt = vec![1u32, 2];
    prompt.extend(std::iter::repeat_n(IMAGE, seeds.len() * patches / 4));
    prompt.push(1);
    EngineRequest {
        prompt,
        images: seeds.iter().map(|&s| image(s, patches)).collect(),
        params: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
    }
}

type Rx = UnboundedReceiver<EngineEvent>;

fn submit(s: &mut Scheduler<Exec>, seeds: &[u8]) -> (RequestId, Rx) {
    let (tx, rx) = unbounded_channel();
    let id = s.submit(request(seeds, 16), tx).unwrap();
    (id, rx)
}

/// Two single-image requests on a one-slot scheduler after one step: the first runs and the
/// second waits holding its pre-embedded image, so the whole budget is charged.
fn filled() -> (Scheduler<Exec>, [(RequestId, Rx); 2]) {
    let mut s = scheduler(1, E);
    let a = submit(&mut s, &[1]);
    let b = submit(&mut s, &[2]);
    s.step();
    assert_eq!(s.pinned_embedding_bytes(), E);
    (s, [a, b])
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

#[test]
fn a_request_needing_more_than_the_budget_is_refused_before_acceptance() {
    let mut s = scheduler(2, E);
    let (tx, _rx) = unbounded_channel();
    let three = s.submit(request(&[1, 2, 3], 16), tx.clone());
    assert_eq!(
        three.unwrap_err(),
        EngineError::EmbeddingBudget {
            required: 3 * IMG,
            budget: E
        }
    );
    let big = s.submit(request(&[4], 64), tx.clone());
    assert_eq!(
        big.unwrap_err(),
        EngineError::EmbeddingBudget {
            required: 2 * E,
            budget: E
        }
    );
    assert!(s.is_idle());
    assert_eq!(s.metrics().submitted, 0);
    s.submit(request(&[5, 6], 16), tx).unwrap();
    s.run_until_idle();
}

#[test]
fn an_unbounded_budget_accepts_what_a_bounded_one_refuses() {
    let mut s = scheduler(2, 0);
    let (tx, mut rx) = unbounded_channel();
    s.submit(request(&[1, 2, 3], 16), tx).unwrap();
    s.run_until_idle();
    assert!(matches!(terminal(&mut rx), Some(EngineEvent::Finished(_))));
}

#[test]
fn waiting_requests_complete_vision_and_the_queue_drains_when_the_budget_is_exhausted() {
    let mut s = scheduler(1, E);
    let mut rxs: Vec<Rx> = (1..=6u8).map(|seed| submit(&mut s, &[seed]).1).collect();
    let mut steps = 0;
    while !s.is_idle() {
        s.step();
        steps += 1;
        assert!(steps < 500, "the queue stopped making progress");
        assert!(s.retained_embedding_bytes() <= E);
    }
    for rx in &mut rxs {
        assert!(matches!(terminal(rx), Some(EngineEvent::Finished(_))));
    }
    assert_eq!(s.executor().vision_images, 6);
}

#[test]
fn requests_that_do_not_fit_wait_for_a_holder_instead_of_embedding() {
    let mut s = scheduler(1, E);
    let _rxs: Vec<Rx> = (1..=3u8).map(|seed| submit(&mut s, &[seed]).1).collect();
    s.step();
    assert_eq!(s.executor().vision_images, 2);
    assert_eq!(s.pinned_embedding_bytes(), E);
}

#[test]
fn a_shared_embedding_is_charged_once() {
    let mut s = scheduler(4, 4 * IMG);
    let (_, mut first) = submit(&mut s, &[9]);
    s.run_until_idle();
    assert!(matches!(
        terminal(&mut first),
        Some(EngineEvent::Finished(_))
    ));
    assert_eq!(s.pinned_embedding_bytes(), 0);
    assert_eq!(s.retained_embedding_bytes(), IMG);

    let _a = submit(&mut s, &[9]);
    let _b = submit(&mut s, &[9]);
    s.step();
    assert_eq!(s.metrics().running, 2);
    assert_eq!(s.pinned_embedding_bytes(), IMG);
    assert_eq!(s.retained_embedding_bytes(), IMG);
    assert_eq!(s.executor().vision_images, 1);
}

#[test]
fn finishing_releases_the_charge() {
    let (mut s, mut rxs) = filled();
    s.run_until_idle();
    for (_, rx) in &mut rxs {
        assert!(matches!(terminal(rx), Some(EngineEvent::Finished(_))));
    }
    assert_eq!(s.pinned_embedding_bytes(), 0);
    assert_eq!(s.retained_embedding_bytes(), E);
}

#[test]
fn cancelling_running_and_waiting_requests_releases_the_charge() {
    let (mut s, rxs) = filled();
    assert!(s.cancel(rxs[1].0));
    assert_eq!(s.pinned_embedding_bytes(), IMG);
    assert!(s.cancel(rxs[0].0));
    assert_eq!(s.pinned_embedding_bytes(), 0);
}

#[test]
fn cancel_all_releases_the_charge() {
    let (mut s, _rxs) = filled();
    assert_eq!(s.cancel_all(CancelReason::Drain), 2);
    assert_eq!(s.pinned_embedding_bytes(), 0);
}

#[test]
fn shedding_releases_the_charge() {
    let (mut s, _rxs) = filled();
    assert_eq!(s.shed_all(), 2);
    assert_eq!(s.pinned_embedding_bytes(), 0);
}

#[test]
fn abandoning_releases_the_charge() {
    let (mut s, _rxs) = filled();
    assert_eq!(s.abandon_all(&EngineError::Closed), 2);
    assert_eq!(s.pinned_embedding_bytes(), 0);
}

#[test]
fn a_vision_failure_releases_the_charge() {
    let mut s = scheduler(1, E);
    s.executor_mut().fail_vision = true;
    let mut rxs: Vec<Rx> = (1..=2u8).map(|seed| submit(&mut s, &[seed]).1).collect();
    s.run_until_idle();
    for rx in &mut rxs {
        assert!(matches!(terminal(rx), Some(EngineEvent::Failed(_))));
    }
    assert_eq!(s.pinned_embedding_bytes(), 0);
}

#[test]
fn a_disconnected_client_releases_the_charge() {
    let (mut s, rxs) = filled();
    drop(rxs);
    s.run_until_idle();
    assert_eq!(s.pinned_embedding_bytes(), 0);
}

#[test]
fn cache_entries_no_request_holds_are_evicted_to_make_room() {
    let mut s = scheduler(1, E);
    let (_, mut first) = submit(&mut s, &[1]);
    let (_, mut second) = submit(&mut s, &[2]);
    s.run_until_idle();
    assert_eq!(s.retained_embedding_bytes(), E);
    let _ = (terminal(&mut first), terminal(&mut second));

    let mut rxs: Vec<Rx> = (3..=4u8).map(|seed| submit(&mut s, &[seed]).1).collect();
    while !s.is_idle() {
        s.step();
        assert!(s.retained_embedding_bytes() <= E);
    }
    for rx in &mut rxs {
        assert!(matches!(terminal(rx), Some(EngineEvent::Finished(_))));
    }
    assert!(s.metrics().embedding_cache.evictions >= 2);
}
