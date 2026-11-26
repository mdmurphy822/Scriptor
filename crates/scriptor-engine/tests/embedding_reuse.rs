use scriptor_cache::{HostBuffer, ImageGrid, KvLayout, embedding_key};
use scriptor_engine::embedding_reuse::{ReuseMetrics, budget_from_share, prompt_tokens_details};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::{
    DecodeRow, EngineError, EngineEvent, EngineRequest, ImageInput, ModelExecutor, PrefillRow,
    RequestId, Scheduler, SchedulerConfig, VisionJob,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc::unbounded_channel;

const VOCAB: usize = 128;
const EOS: u32 = 0;
const IMAGE: u32 = 50;
const HIDDEN: usize = 4;

/// Embeddings carry the first payload byte; the next token depends on the sum of those bytes
/// found in the prefilled images, so output identity proves the cached rows were scattered.
#[derive(Default)]
struct Exec {
    vision_calls: usize,
    vision_images: usize,
    history: HashMap<RequestId, Vec<u32>>,
}

fn next_logits(seed: u32, generated: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; VOCAB];
    let next = if generated >= 4 {
        EOS
    } else {
        100 + (seed + generated as u32) % 20
    };
    l[next as usize] = 10.0;
    l
}

fn seed_of(images: &[Arc<HostBuffer>]) -> u32 {
    images.iter().map(|b| b.0[0] as u32).sum()
}

impl ModelExecutor for Exec {
    type Embedding = HostBuffer;

    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        self.vision_calls += 1;
        self.vision_images += jobs.len();
        Ok(jobs
            .iter()
            .map(|j| {
                let mut v = vec![j.payload[0]; j.patches / 4 * HIDDEN * 2];
                v.shrink_to_fit();
                HostBuffer(v)
            })
            .collect())
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        Ok(rows
            .iter()
            .map(|r| {
                let seed = seed_of(&r.images);
                self.history.insert(r.seq, vec![seed]);
                next_logits(seed, 0)
            })
            .collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        Ok(rows
            .iter()
            .map(|r| {
                let h = self.history.get_mut(&r.seq).expect("prefilled");
                h.push(r.token);
                next_logits(h[0], h.len() - 1)
            })
            .collect())
    }

    fn release(&mut self, seq: RequestId) {
        self.history.remove(&seq);
    }
}

fn scheduler(cache_bytes: usize) -> Scheduler<Exec> {
    let config = SchedulerConfig {
        max_concurrency: 4,
        max_context: 256,
        vision_batch_patches: 64,
        vision_overlap: true,
        spatial_merge: 2,
        vocab_size: VOCAB,
        eos_token_ids: vec![EOS],
        image_token_id: IMAGE,
        embedding_cache_bytes: cache_bytes,
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

fn request(seed: u8, patches: usize) -> EngineRequest {
    let mut prompt = vec![1u32, 2];
    prompt.extend(std::iter::repeat_n(IMAGE, patches / 4));
    prompt.push(1);
    EngineRequest {
        prompt,
        images: vec![image(seed, patches)],
        params: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
    }
}

fn run(s: &mut Scheduler<Exec>, req: EngineRequest) -> Vec<u32> {
    let (tx, mut rx) = unbounded_channel();
    s.submit(req, tx).unwrap();
    s.run_until_idle();
    let mut tokens = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        match ev {
            EngineEvent::Token(t) => tokens.push(t.token),
            EngineEvent::Finished(_) => {}
            other => panic!("unexpected event {other:?}"),
        }
    }
    tokens
}

#[test]
fn repeated_image_skips_vision_with_identical_output() {
    let mut s = scheduler(1 << 20);
    let first = run(&mut s, request(7, 16));
    assert_eq!(s.executor().vision_calls, 1);
    let plan = s.plan_reuse(&request(7, 16).images);
    assert_eq!(plan.hit_count(), 1);
    assert_eq!(plan.cached_tokens, 4);
    assert_eq!(plan.vision_patches, 0);

    let second = run(&mut s, request(7, 16));
    assert_eq!(s.executor().vision_calls, 1, "no second vision pass");
    assert_eq!(first, second);
    assert!(!first.is_empty());

    let other = run(&mut s, request(9, 16));
    assert_eq!(s.executor().vision_calls, 2);
    assert_ne!(first, other);

    let m = s.reuse_metrics();
    assert_eq!((m.hits, m.misses), (1, 2));
    assert!((m.hit_rate() - 1.0 / 3.0).abs() < 1e-9);
    assert!(m.render().contains("oe:image_cache_hits_total 1"));
}

#[test]
fn probe_does_not_count_lookups() {
    let mut s = scheduler(1 << 20);
    run(&mut s, request(3, 8));
    let before = s.reuse_metrics();
    let miss = s.plan_reuse(&request(4, 8).images);
    assert_eq!(miss.hit_count(), 0);
    assert_eq!(miss.vision_patches, 8);
    assert_eq!(miss.cached_tokens, 0);
    assert_eq!(s.reuse_metrics(), before);
}

#[test]
fn eviction_honours_the_budget() {
    // One 16-patch image holds 4 rows * 4 hidden * 2 bytes = 32 bytes; the budget holds two.
    let mut s = scheduler(64);
    for seed in 1..=3u8 {
        run(&mut s, request(seed, 16));
    }
    let m = s.reuse_metrics();
    assert!(m.bytes <= m.budget_bytes);
    assert_eq!((m.entries, m.evictions), (2, 1));
    assert!(
        !s.plan_reuse(&request(1, 16).images).hits[0],
        "oldest evicted"
    );
    assert!(s.plan_reuse(&request(3, 16).images).hits[0]);

    let calls = s.executor().vision_calls;
    run(&mut s, request(1, 16));
    assert_eq!(
        s.executor().vision_calls,
        calls + 1,
        "evicted image re-embedded"
    );
    assert!(s.reuse_metrics().bytes <= 64);
}

#[test]
fn zero_budget_never_retains() {
    let mut s = scheduler(0);
    run(&mut s, request(5, 16));
    run(&mut s, request(5, 16));
    assert_eq!(s.executor().vision_calls, 2);
    assert_eq!(s.reuse_metrics().entries, 0);
}

#[test]
fn budget_and_usage_helpers() {
    assert_eq!(budget_from_share(1000, 0.25), 250);
    assert_eq!(budget_from_share(1000, 2.0), 1000);
    assert_eq!(budget_from_share(1000, -1.0), 0);
    assert_eq!(budget_from_share(1000, f64::NAN), 0);
    assert_eq!(
        prompt_tokens_details(12),
        serde_json::json!({"cached_tokens": 12})
    );
    assert_eq!(ReuseMetrics::default().hit_rate(), 0.0);
}
