use scriptor_cache::{HostBuffer, ImageGrid, KvLayout, embedding_key};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::{
    DecodeRow, EngineError, EngineEvent, EngineRequest, ImageInput, ModelExecutor, PrefillRow,
    RequestId, Scheduler, SchedulerConfig, VisionJob, VisionToken,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc::unbounded_channel;

const VOCAB: usize = 128;
const EOS: u32 = 0;
const IMAGE: u32 = 50;
const HIDDEN: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Call {
    Vision(usize),
    Begin(usize),
    Finish,
    Decode,
    Prefill,
}

/// CPU executor whose asynchronous vision batch completes only after `decodes_needed` decode
/// steps ran since it began.
struct AsyncExecutor {
    asynchronous: bool,
    decodes_needed: usize,
    decodes_since_begin: usize,
    pending: HashMap<VisionToken, Vec<usize>>,
    next: VisionToken,
    history: HashMap<RequestId, Vec<u32>>,
    log: Vec<Call>,
}

impl AsyncExecutor {
    fn new(asynchronous: bool, decodes_needed: usize) -> Self {
        Self {
            asynchronous,
            decodes_needed,
            decodes_since_begin: 0,
            pending: HashMap::new(),
            next: 0,
            history: HashMap::new(),
            log: Vec::new(),
        }
    }
}

fn logits_for(tokens: &[u32]) -> Vec<f32> {
    let n = tokens[0] as usize;
    let g = tokens.iter().filter(|&&t| t >= 100).count();
    let next = if g >= n { EOS } else { 100 + (g % 20) as u32 };
    let mut l = vec![0.0f32; VOCAB];
    l[next as usize] = 10.0;
    l
}

fn buffers(patches: &[usize]) -> Vec<HostBuffer> {
    patches
        .iter()
        .map(|p| HostBuffer(vec![0u8; p / 4 * HIDDEN * 2]))
        .collect()
}

impl ModelExecutor for AsyncExecutor {
    type Embedding = HostBuffer;

    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        self.log.push(Call::Vision(jobs.len()));
        Ok(buffers(&jobs.iter().map(|j| j.patches).collect::<Vec<_>>()))
    }

    fn supports_async_vision(&self) -> bool {
        self.asynchronous
    }

    fn vision_begin(&mut self, jobs: &[VisionJob]) -> Result<VisionToken, EngineError> {
        assert!(self.pending.is_empty(), "one batch outstanding at a time");
        self.log.push(Call::Begin(jobs.len()));
        self.decodes_since_begin = 0;
        let token = self.next;
        self.next += 1;
        self.pending
            .insert(token, jobs.iter().map(|j| j.patches).collect());
        Ok(token)
    }

    fn vision_ready(&mut self, _token: VisionToken) -> Result<bool, EngineError> {
        Ok(self.decodes_since_begin >= self.decodes_needed)
    }

    fn vision_finish(&mut self, token: VisionToken) -> Result<Vec<HostBuffer>, EngineError> {
        self.log.push(Call::Finish);
        Ok(buffers(&self.pending.remove(&token).expect("begun")))
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        self.log.push(Call::Prefill);
        for r in rows {
            self.history.insert(r.seq, r.tokens.clone());
        }
        Ok(rows.iter().map(|r| logits_for(&r.tokens)).collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        self.log.push(Call::Decode);
        self.decodes_since_begin += 1;
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

fn config(overlap: bool, max_concurrency: usize) -> SchedulerConfig {
    SchedulerConfig {
        max_concurrency,
        max_context: 512,
        vision_batch_patches: 4096,
        vision_overlap: overlap,
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

fn finished(rx: &mut tokio::sync::mpsc::UnboundedReceiver<EngineEvent>) -> usize {
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
fn decode_steps_proceed_while_a_vision_batch_is_in_flight() {
    let mut s = Scheduler::new(config(true, 8), layout(), AsyncExecutor::new(true, 3));
    let (tx_a, mut rx_a) = unbounded_channel();
    s.submit(text_request(40), tx_a).unwrap();
    for _ in 0..3 {
        s.step();
    }
    let (tx_b, mut rx_b) = unbounded_channel();
    s.submit(image_request(2, 7, 64), tx_b).unwrap();
    while !s.is_idle() {
        s.step();
    }
    assert_eq!(finished(&mut rx_a), 40);
    assert_eq!(finished(&mut rx_b), 2);
    let log = &s.executor().log;
    let begin = log
        .iter()
        .position(|c| matches!(c, Call::Begin(_)))
        .expect("an asynchronous batch began");
    let finish = log
        .iter()
        .position(|c| *c == Call::Finish)
        .expect("the batch finished");
    let during = log[begin..finish]
        .iter()
        .filter(|c| **c == Call::Decode)
        .count();
    assert!(
        during >= 3,
        "decode ran {during} times while vision was in flight: {log:?}"
    );
    assert!(!log.iter().any(|c| matches!(c, Call::Vision(_))));
}

#[test]
fn disabled_overlap_embeds_images_before_any_decode_resumes() {
    let mut s = Scheduler::new(config(false, 8), layout(), AsyncExecutor::new(true, 3));
    let (tx_a, mut rx_a) = unbounded_channel();
    s.submit(text_request(40), tx_a).unwrap();
    for _ in 0..3 {
        s.step();
    }
    let (tx_b, mut rx_b) = unbounded_channel();
    s.submit(image_request(2, 7, 64), tx_b).unwrap();
    while !s.is_idle() {
        s.step();
    }
    assert_eq!(finished(&mut rx_a), 40);
    assert_eq!(finished(&mut rx_b), 2);
    let log = &s.executor().log;
    assert!(log.contains(&Call::Vision(1)));
    assert!(!log.iter().any(|c| matches!(c, Call::Begin(_))));
}

#[test]
fn waiting_images_join_one_batch_and_the_sizes_are_recorded() {
    for overlap in [false, true] {
        let mut s = Scheduler::new(config(overlap, 2), layout(), AsyncExecutor::new(true, 0));
        let mut rxs = Vec::new();
        for seed in 0..6u8 {
            let (tx, rx) = unbounded_channel();
            s.submit(image_request(2, seed, 64), tx).unwrap();
            rxs.push(rx);
        }
        while !s.is_idle() {
            s.step();
        }
        for rx in &mut rxs {
            assert_eq!(finished(rx), 2);
        }
        let m = s.metrics();
        assert_eq!(m.vision_images, 6);
        assert_eq!(m.vision_batches, 1, "overlap {overlap}");
        assert_eq!(m.vision_batch_sizes, vec![(6, 1)]);
    }
}
