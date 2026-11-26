use scriptor_cache::{HostBuffer, ImageGrid, KvLayout, embedding_key};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::{
    DecodeRow, EngineError, EngineEvent, EngineRequest, ImageInput, ModelExecutor, PrefillRow,
    RequestId, Scheduler, SchedulerConfig, VisionJob,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::unbounded_channel;

const VOCAB: usize = 128;
const EOS: u32 = 0;
const IMAGE: u32 = 50;
const HIDDEN: usize = 4;

const SATURATION: usize = 256;

/// CPU executor whose tower reports a saturation size and whose decode takes a fixed time.
#[derive(Default)]
struct BatchingExecutor {
    batches: Vec<usize>,
    history: HashMap<RequestId, Vec<u32>>,
}

fn logits_for(tokens: &[u32]) -> Vec<f32> {
    let n = tokens[0] as usize;
    let g = tokens.iter().filter(|&&t| t >= 100).count();
    let next = if g >= n { EOS } else { 100 + (g % 20) as u32 };
    let mut l = vec![0.0f32; VOCAB];
    l[next as usize] = 10.0;
    l
}

impl ModelExecutor for BatchingExecutor {
    type Embedding = HostBuffer;

    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        self.batches.push(jobs.iter().map(|j| j.patches).sum());
        Ok(jobs
            .iter()
            .map(|j| HostBuffer(vec![0u8; j.patches / 4 * HIDDEN * 2]))
            .collect())
    }

    fn vision_saturation_patches(&self) -> usize {
        SATURATION
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        for r in rows {
            self.history.insert(r.seq, r.tokens.clone());
        }
        Ok(rows.iter().map(|r| logits_for(&r.tokens)).collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        std::thread::sleep(Duration::from_millis(2));
        Ok(rows
            .iter()
            .map(|r| {
                let h = self.history.get_mut(&r.seq).expect("prefilled");
                h.push(r.token);
                logits_for(h)
            })
            .collect())
    }

    fn release(&mut self, seq: RequestId) {
        self.history.remove(&seq);
    }
}

fn config() -> SchedulerConfig {
    SchedulerConfig {
        max_concurrency: 8,
        max_context: 512,
        vision_batch_patches: 4096,
        vision_overlap: false,
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

fn layout() -> KvLayout {
    KvLayout {
        num_blocks: 1024,
        block_tokens: 16,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    }
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    }
}

fn text_request(generate: u32) -> EngineRequest {
    EngineRequest {
        prompt: vec![generate, 1, 1, 1],
        images: Vec::new(),
        params: greedy(),
    }
}

fn image_request(generate: u32, seed: u8, patches: usize) -> EngineRequest {
    let grid = ImageGrid {
        t: 1,
        h: (patches / 4) as u32,
        w: 4,
    };
    let mut prompt = vec![generate, 1];
    prompt.extend(std::iter::repeat_n(IMAGE, patches / 4));
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

fn drain(rx: &mut tokio::sync::mpsc::UnboundedReceiver<EngineEvent>) -> usize {
    let mut tokens = 0;
    loop {
        match rx.try_recv().expect("terminal event delivered") {
            EngineEvent::Token(_) => tokens += 1,
            EngineEvent::Finished(_) => return tokens,
            other => panic!("unexpected {other:?}"),
        }
    }
}

#[test]
fn a_lone_small_image_is_embedded_in_the_step_it_arrives() {
    let mut s = Scheduler::new(config(), layout(), BatchingExecutor::default());
    let (tx, mut rx) = unbounded_channel();
    s.submit(image_request(2, 1, 64), tx).unwrap();
    let report = s.step();
    assert_eq!(report.vision_images, 1);
    s.run_until_idle();
    assert_eq!(drain(&mut rx), 2);
    assert_eq!(s.executor().batches, vec![64]);
}

#[test]
fn a_saturating_batch_is_embedded_at_once_beside_decode_work() {
    let mut s = Scheduler::new(config(), layout(), BatchingExecutor::default());
    let (tx_a, mut rx_a) = unbounded_channel();
    s.submit(text_request(30), tx_a).unwrap();
    for _ in 0..3 {
        s.step();
    }
    let mut rxs = Vec::new();
    for seed in 0..4u8 {
        let (tx, rx) = unbounded_channel();
        s.submit(image_request(2, seed, 64), tx).unwrap();
        rxs.push(rx);
    }
    let report = s.step();
    assert_eq!(report.vision_images, 4);
    assert_eq!(s.executor().batches, vec![256]);
    s.run_until_idle();
    assert_eq!(drain(&mut rx_a), 30);
    for rx in &mut rxs {
        assert_eq!(drain(rx), 2);
    }
}

#[test]
fn a_small_image_beside_decode_work_is_never_starved() {
    let mut s = Scheduler::new(config(), layout(), BatchingExecutor::default());
    let (tx_a, mut rx_a) = unbounded_channel();
    s.submit(text_request(60), tx_a).unwrap();
    for _ in 0..3 {
        s.step();
    }
    let (tx_b, mut rx_b) = unbounded_channel();
    s.submit(image_request(2, 9, 64), tx_b).unwrap();
    let mut steps = 0;
    while s.metrics().vision_images == 0 {
        s.step();
        steps += 1;
        assert!(steps <= 3, "the image waited {steps} steps");
    }
    s.run_until_idle();
    assert_eq!(drain(&mut rx_a), 60);
    assert_eq!(drain(&mut rx_b), 2);
}
