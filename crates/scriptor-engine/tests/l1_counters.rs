use scriptor_cache::{HostBuffer, ImageGrid, KvLayout, embedding_key};
use scriptor_engine::l1_counters::{L1Stats, L1Switches, StepHostSample, VisionTiming};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::scheduler::{DeviceRow, StepRowResult};
use scriptor_engine::{
    DecodeRow, EngineError, EngineEvent, EngineRequest, ImageInput, ModelExecutor, PrefillRow,
    RequestId, Scheduler, SchedulerConfig, VisionJob, VisionToken,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::unbounded_channel;

const VOCAB: usize = 128;
const EOS: u32 = 0;
const IMAGE: u32 = 50;
const HIDDEN: usize = 4;

/// CPU executor whose asynchronous vision batch completes after `decodes_needed` decode steps
/// and reports scripted timing.
struct TimedExecutor {
    decodes_needed: usize,
    decodes_since_begin: usize,
    pending: HashMap<VisionToken, Vec<usize>>,
    next: VisionToken,
    history: HashMap<RequestId, Vec<u32>>,
    timing: Option<VisionTiming>,
    device: bool,
    prefill_sleep: Duration,
    finish_sleep: Duration,
    pending_device: Vec<StepRowResult>,
}

const SYNC_WALL_NS: u64 = 100_000;
const SYNC_CPU_NS: u64 = 90_000;
const HOST_CPU: u32 = 3;

impl TimedExecutor {
    /// Serves decode through the device-step calls, sleeping `prefill_sleep` per prefill and
    /// `finish_sleep` per vision finish.
    fn device(
        timing: Option<VisionTiming>,
        prefill_sleep: Duration,
        finish_sleep: Duration,
    ) -> Self {
        Self {
            decodes_needed: 1_000,
            device: true,
            prefill_sleep,
            finish_sleep,
            ..Self::new(timing)
        }
    }
}

impl TimedExecutor {
    fn new(timing: Option<VisionTiming>) -> Self {
        Self {
            decodes_needed: 2,
            decodes_since_begin: 0,
            pending: HashMap::new(),
            next: 0,
            history: HashMap::new(),
            timing,
            device: false,
            prefill_sleep: Duration::ZERO,
            finish_sleep: Duration::ZERO,
            pending_device: Vec::new(),
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

impl ModelExecutor for TimedExecutor {
    type Embedding = HostBuffer;

    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        Ok(buffers(&jobs.iter().map(|j| j.patches).collect::<Vec<_>>()))
    }

    fn supports_async_vision(&self) -> bool {
        true
    }

    fn vision_begin(&mut self, jobs: &[VisionJob]) -> Result<VisionToken, EngineError> {
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
        std::thread::sleep(self.finish_sleep);
        Ok(buffers(&self.pending.remove(&token).expect("begun")))
    }

    fn take_vision_timing(&mut self, _token: VisionToken) -> Option<VisionTiming> {
        self.timing
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        std::thread::sleep(self.prefill_sleep);
        for r in rows {
            self.history.insert(r.seq, r.tokens.clone());
        }
        Ok(rows.iter().map(|r| logits_for(&r.tokens)).collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
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

    fn supports_device_step(&self) -> bool {
        self.device
    }

    fn decode_device_begin(&mut self, rows: &[DeviceRow<'_>]) -> Result<bool, EngineError> {
        if !self.device {
            return Ok(false);
        }
        self.decodes_since_begin += 1;
        self.pending_device = rows
            .iter()
            .map(|r| {
                let h = self.history.get_mut(&r.row.seq).expect("prefilled");
                h.push(r.row.token);
                let logits = logits_for(h);
                let best = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map_or(0, |(i, _)| i as u32);
                StepRowResult {
                    tokens: vec![best],
                    margins: vec![1.0],
                }
            })
            .collect();
        Ok(true)
    }

    fn decode_device_finish(&mut self) -> Result<Vec<StepRowResult>, EngineError> {
        Ok(std::mem::take(&mut self.pending_device))
    }

    fn take_host_sample(&mut self) -> Option<StepHostSample> {
        self.device.then_some(StepHostSample {
            sync_wall_ns: SYNC_WALL_NS,
            sync_cpu_ns: SYNC_CPU_NS,
            cpu: HOST_CPU,
        })
    }

    fn release(&mut self, seq: RequestId) {
        self.history.remove(&seq);
    }
}

fn config() -> SchedulerConfig {
    SchedulerConfig {
        max_concurrency: 4,
        max_context: 512,
        vision_batch_patches: 4096,
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

/// Runs three image requests to completion and returns each request's token events.
fn run(
    switches: L1Switches,
    timing: Option<VisionTiming>,
) -> (Vec<String>, Scheduler<TimedExecutor>) {
    let mut s = Scheduler::new(config(), layout(), TimedExecutor::new(timing));
    s.set_l1_switches(switches);
    let mut rxs = Vec::new();
    for seed in 0..3u8 {
        let (tx, rx) = unbounded_channel();
        s.submit(image_request(12, seed, 64 + 32 * seed as usize), tx)
            .unwrap();
        rxs.push(rx);
    }
    while !s.is_idle() {
        s.step();
    }
    let mut events = Vec::new();
    for rx in &mut rxs {
        loop {
            match rx.try_recv().expect("terminal event delivered") {
                EngineEvent::Token(t) => events.push(format!("{t:?}")),
                EngineEvent::Finished(_) => break,
                other => panic!("unexpected {other:?}"),
            }
        }
    }
    (events, s)
}

fn all_on() -> L1Switches {
    L1Switches {
        counters: true,
        step_lines: true,
        pass_lines: false,
    }
}

#[test]
fn l1_on_and_off_emit_identical_event_streams() {
    let (off, _) = run(L1Switches::default(), None);
    let (on, _) = run(all_on(), None);
    assert!(!off.is_empty());
    assert_eq!(off, on);
}

#[test]
fn l1_off_registers_nothing_in_engine_metrics() {
    let (_, s) = run(L1Switches::default(), None);
    assert!(s.metrics().l1.is_none());
}

#[test]
fn l1_flights_are_counted_with_exact_patch_sums() {
    let timing = VisionTiming {
        copy_ns: 1_000,
        device_ns: 5_000,
        start_host_ns: Some(1),
        ready_host_ns: Some(2),
    };
    let (_, s) = run(all_on(), Some(timing));
    let m = s.metrics();
    let l1 = m.l1.expect("counters on");
    assert_eq!(l1.flights, m.vision_batches);
    assert_eq!(l1.flight_patches, m.vision_patches);
    assert!(l1.flights > 0);
    assert_eq!(l1.vision_device_ns, 5_000 * l1.flights);
    assert_eq!(l1.vision_copy_ns, 1_000 * l1.flights);
    assert_eq!(l1.callbacks_missing, 0);
    assert_eq!(l1.steps, 0, "the host decode path records no device steps");
}

#[test]
fn l1_flights_without_timing_count_missing_callbacks() {
    let (_, s) = run(all_on(), None);
    let l1 = s.metrics().l1.expect("counters on");
    assert!(l1.flights > 0);
    assert_eq!(l1.callbacks_missing, l1.flights);
    assert_eq!(l1.vision_device_ns, 0);
}

fn text_request(generate: u32) -> EngineRequest {
    EngineRequest {
        prompt: vec![generate, 1, 1, 1],
        images: Vec::new(),
        params: greedy(),
    }
}

fn l1_of(s: &Scheduler<TimedExecutor>) -> L1Stats {
    s.metrics().l1.expect("counters on")
}

/// Asserts the closed turnaround windows partition into prefill, blocking vision wait and host
/// time, and that every device step carried the executor's host sample.
fn assert_window_partition(l1: &L1Stats) {
    assert!(l1.windows >= 1, "a window closed");
    assert_eq!(
        l1.windows + 1,
        l1.steps,
        "the last step's window stays open"
    );
    assert_eq!(
        l1.turnaround_ns,
        l1.turnaround_prefill_ns + l1.turnaround_vision_wait_ns + l1.turnaround_host_ns
    );
    assert_eq!(l1.sync_ns, SYNC_WALL_NS * l1.steps);
    assert_eq!(l1.sync_cpu_ns, SYNC_CPU_NS * l1.steps);
    assert_eq!(l1.engine_cpu_steps.get(&HOST_CPU), Some(&l1.steps));
    assert_eq!(l1.engine_cpu_migrations, 0);
}

#[test]
fn l1_step_windows_exclude_prefill_and_blocking_vision() {
    let prefill = Duration::from_millis(5);
    let vision = Duration::from_millis(6);

    let mut s = Scheduler::new(
        config(),
        layout(),
        TimedExecutor::device(None, prefill, Duration::ZERO),
    );
    s.set_l1_switches(all_on());
    let (tx_a, _rx_a) = unbounded_channel();
    s.submit(text_request(8), tx_a).unwrap();
    while l1_of(&s).steps < 1 {
        s.step();
    }
    let (tx_c, _rx_c) = unbounded_channel();
    s.submit(text_request(4), tx_c).unwrap();
    s.run_until_idle();
    let l1 = l1_of(&s);
    assert_window_partition(&l1);
    assert!(l1.turnaround_prefill_ns >= prefill.as_nanos() as u64);
    assert_eq!(l1.turnaround_vision_wait_ns, 0);

    let mut s = Scheduler::new(
        config(),
        layout(),
        TimedExecutor::device(None, Duration::ZERO, vision),
    );
    s.set_l1_switches(all_on());
    let (tx_a, _rx_a) = unbounded_channel();
    let (tx_b, _rx_b) = unbounded_channel();
    s.submit(text_request(6), tx_a).unwrap();
    s.submit(image_request(3, 1, 64), tx_b).unwrap();
    s.run_until_idle();
    let l1 = l1_of(&s);
    assert_window_partition(&l1);
    assert!(l1.turnaround_vision_wait_ns >= vision.as_nanos() as u64);
    assert!(l1.turnaround_host_ns + vision.as_nanos() as u64 <= l1.turnaround_ns);
}
