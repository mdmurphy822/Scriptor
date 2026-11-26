use scriptor_cache::{HostBuffer, ImageGrid, KvLayout, embedding_key};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::scheduler::{DeviceRow, StepRowResult};
use scriptor_engine::{
    DecodeRow, EngineError, EngineEvent, EngineRequest, ImageInput, ModelExecutor, PrefillRow,
    RequestId, Scheduler, SchedulerConfig, VisionJob,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

const VOCAB: usize = 128;
const EOS: u32 = 0;
const IMAGE: u32 = 50;
const HIDDEN: usize = 4;

/// CPU executor recording prefill pass sizes and the order of device-step calls.
#[derive(Default)]
struct PackingExecutor {
    saturation: usize,
    device: bool,
    passes: Vec<Vec<RequestId>>,
    events: Vec<&'static str>,
    history: HashMap<RequestId, Vec<u32>>,
    pending: Vec<StepRowResult>,
}

fn logits_for(tokens: &[u32]) -> Vec<f32> {
    let n = tokens[0] as usize;
    let g = tokens.iter().filter(|&&t| t >= 100).count();
    let next = if g >= n { EOS } else { 100 + (g % 20) as u32 };
    let mut l = vec![0.0f32; VOCAB];
    l[next as usize] = 10.0;
    l
}

fn argmax(l: &[f32]) -> u32 {
    l.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

impl ModelExecutor for PackingExecutor {
    type Embedding = HostBuffer;

    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        self.events.push("vision");
        std::thread::sleep(Duration::from_millis(3));
        Ok(jobs
            .iter()
            .map(|j| HostBuffer(vec![0u8; j.patches / 4 * HIDDEN * 2]))
            .collect())
    }

    fn prefill_saturation_tokens(&self) -> usize {
        self.saturation
    }

    fn vision_prepare(&mut self, _jobs: &[VisionJob]) -> Result<(), EngineError> {
        self.events.push("prepare");
        Ok(())
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        self.passes.push(rows.iter().map(|r| r.seq).collect());
        for r in rows {
            let keep = if r.logits {
                r.tokens.len()
            } else {
                r.tokens.len() - 1
            };
            self.history.insert(r.seq, r.tokens[..keep].to_vec());
        }
        Ok(rows
            .iter()
            .map(|r| {
                if r.logits {
                    logits_for(&r.tokens)
                } else {
                    Vec::new()
                }
            })
            .collect())
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

    fn supports_device_step(&self) -> bool {
        self.device
    }

    fn decode_device_begin(&mut self, rows: &[DeviceRow<'_>]) -> Result<bool, EngineError> {
        self.events.push("begin");
        std::thread::sleep(Duration::from_millis(2));
        self.pending = rows
            .iter()
            .map(|r| {
                let h = self.history.get_mut(&r.row.seq).expect("prefilled");
                h.push(r.row.token);
                StepRowResult {
                    tokens: vec![argmax(&logits_for(h))],
                    margins: vec![1.0],
                }
            })
            .collect();
        Ok(true)
    }

    fn decode_device_finish(&mut self) -> Result<Vec<StepRowResult>, EngineError> {
        self.events.push("finish");
        Ok(std::mem::take(&mut self.pending))
    }

    fn release(&mut self, seq: RequestId) {
        self.history.remove(&seq);
    }
}

fn config(max_context: usize) -> SchedulerConfig {
    SchedulerConfig {
        max_concurrency: 8,
        max_context,
        vision_batch_patches: 64,
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

fn text_request(generate: u32, len: usize) -> EngineRequest {
    let mut prompt = vec![generate];
    prompt.resize(len, 1);
    EngineRequest {
        prompt,
        images: Vec::new(),
        params: greedy(),
    }
}

fn image_request(generate: u32, seed: u8) -> EngineRequest {
    let grid = ImageGrid { t: 1, h: 16, w: 4 };
    let mut prompt = vec![generate, 1];
    prompt.extend(std::iter::repeat_n(IMAGE, 16));
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

fn drain(rx: &mut UnboundedReceiver<EngineEvent>) -> Vec<u32> {
    let mut tokens = Vec::new();
    loop {
        match rx.try_recv().expect("terminal event delivered") {
            EngineEvent::Token(t) => tokens.push(t.token),
            EngineEvent::Finished(_) => return tokens,
            other => panic!("unexpected {other:?}"),
        }
    }
}

/// Runs a decoding text request beside four one-image requests that reach prefill one per
/// step, and returns the prefill passes and each request's tokens.
fn staggered(saturation: usize, device: bool) -> (Vec<Vec<RequestId>>, Vec<Vec<u32>>) {
    let exec = PackingExecutor {
        saturation,
        device,
        ..PackingExecutor::default()
    };
    let mut s = Scheduler::new(config(512), layout(), exec);
    let (tx, rx_text) = unbounded_channel();
    s.submit(text_request(60, 4), tx).unwrap();
    for _ in 0..3 {
        s.step();
    }
    let mut rxs = vec![rx_text];
    for seed in 0..4u8 {
        let (tx, rx) = unbounded_channel();
        s.submit(image_request(3, seed), tx).unwrap();
        rxs.push(rx);
    }
    s.run_until_idle();
    let tokens = rxs.iter_mut().map(drain).collect();
    (s.executor().passes.clone(), tokens)
}

#[test]
fn without_a_saturation_size_every_ready_row_runs_at_once() {
    let (passes, _) = staggered(0, false);
    assert!(passes.iter().all(|p| p.len() == 1), "{passes:?}");
}

#[test]
fn vision_complete_rows_are_held_for_a_fuller_pass_beside_decode_work() {
    let (eager, eager_tokens) = staggered(0, false);
    let (held, held_tokens) = staggered(1000, false);
    assert!(held.iter().any(|p| p.len() >= 2), "{held:?}");
    assert!(held.len() < eager.len(), "{held:?} vs {eager:?}");
    assert_eq!(held.iter().map(Vec::len).sum::<usize>(), 5);
    assert_eq!(held_tokens, eager_tokens);
}

#[test]
fn a_held_row_is_not_starved_when_nothing_else_is_coming() {
    let exec = PackingExecutor {
        saturation: 100_000,
        ..PackingExecutor::default()
    };
    let mut s = Scheduler::new(config(512), layout(), exec);
    let (tx, mut rx) = unbounded_channel();
    s.submit(image_request(2, 7), tx).unwrap();
    let mut steps = 0;
    while !s.is_idle() {
        s.step();
        steps += 1;
        assert!(steps < 20);
    }
    assert_eq!(drain(&mut rx).len(), 2);
    assert_eq!(s.executor().passes.len(), 1);
}

#[test]
fn passes_split_at_the_token_budget() {
    let mut s = Scheduler::new(config(64), layout(), PackingExecutor::default());
    let mut rxs = Vec::new();
    for g in 1..=4u32 {
        let (tx, rx) = unbounded_channel();
        s.submit(text_request(g, 30), tx).unwrap();
        rxs.push(rx);
    }
    s.step();
    let sizes: Vec<usize> = s.executor().passes.iter().map(Vec::len).collect();
    assert_eq!(sizes, vec![2, 2]);
    s.run_until_idle();
    for (g, rx) in (1..=4usize).zip(&mut rxs) {
        assert_eq!(drain(rx).len(), g);
    }
}

#[test]
fn next_vision_inputs_are_prepared_while_the_device_step_runs() {
    let (_, tokens) = staggered(0, true);
    let (_, host_tokens) = staggered(0, false);
    assert_eq!(tokens, host_tokens);

    let exec = PackingExecutor {
        device: true,
        ..PackingExecutor::default()
    };
    let mut s = Scheduler::new(config(512), layout(), exec);
    let (tx, _rx_a) = unbounded_channel();
    s.submit(text_request(60, 4), tx).unwrap();
    for _ in 0..3 {
        s.step();
    }
    let (tx, _rx_b) = unbounded_channel();
    s.submit(image_request(2, 1), tx).unwrap();
    let (tx, _rx_c) = unbounded_channel();
    s.submit(image_request(2, 2), tx).unwrap();
    s.step();
    let events = s.executor().events.clone();
    let begin = events.iter().rposition(|e| *e == "begin").unwrap();
    assert_eq!(&events[begin..], ["begin", "prepare", "finish"]);
}
