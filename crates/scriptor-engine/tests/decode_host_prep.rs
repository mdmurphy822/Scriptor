//! Decode-step host preparation on CPU-only fake executors: the rows an executor receives are
//! complete, ordered by admission age and independent of earlier steps, and the preparation
//! cost is measured against executor time. Timings and allocation counts describe this
//! fixture's host work only and say nothing about GPU speed. The measurement tests are ignored
//! by default; run them with
//! `cargo test --release -p scriptor-engine --test decode_host_prep -- --ignored --nocapture`.

use scriptor_cache::{BlockId, HostBuffer, KvLayout};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::{
    DecodeRow, DeviceRow, EngineError, EngineRequest, ModelExecutor, PrefillRow, Scheduler,
    SchedulerConfig, StepRowResult, VisionJob,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc::unbounded_channel;

const VOCAB: usize = 64;
const BLOCK_TOKENS: usize = 16;
const NEXT: u32 = 7;

thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
}

struct Counting;

// SAFETY: every call forwards to the system allocator unchanged; the counters are
// destructor-free thread-locals.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        let _ = BYTES.try_with(|c| c.set(c.get() + layout.size() as u64));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        let _ = BYTES.try_with(|c| c.set(c.get() + new as u64));
        unsafe { System.realloc(ptr, layout, new) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn allocs() -> (u64, u64) {
    (ALLOCS.with(Cell::get), BYTES.with(Cell::get))
}

/// One row as an executor saw it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Seen {
    seq: u64,
    token: u32,
    position: usize,
    blocks: Vec<BlockId>,
}

impl From<&DecodeRow> for Seen {
    fn from(r: &DecodeRow) -> Self {
        Self {
            seq: r.seq,
            token: r.token,
            position: r.position,
            blocks: r.blocks.clone(),
        }
    }
}

/// Rows of one executor call.
#[derive(Clone, Debug)]
struct Call {
    device: bool,
    rows: Vec<Seen>,
}

#[derive(Default)]
struct Log {
    calls: Mutex<Vec<Call>>,
}

struct Fake {
    device: bool,
    record: bool,
    log: Arc<Log>,
}

fn logits() -> Vec<f32> {
    let mut l = vec![0.0f32; VOCAB];
    l[NEXT as usize] = 10.0;
    l
}

impl ModelExecutor for Fake {
    type Embedding = HostBuffer;
    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        Ok(jobs.iter().map(|_| HostBuffer(Vec::new())).collect())
    }
    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        Ok(rows.iter().map(|_| logits()).collect())
    }
    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        if self.record {
            self.log.calls.lock().unwrap().push(Call {
                device: false,
                rows: rows.iter().map(Seen::from).collect(),
            });
        }
        Ok(rows.iter().map(|_| logits()).collect())
    }
    fn supports_device_step(&self) -> bool {
        self.device
    }
    fn decode_device(&mut self, rows: &[DeviceRow<'_>]) -> Result<Vec<StepRowResult>, EngineError> {
        if self.record {
            self.log.calls.lock().unwrap().push(Call {
                device: true,
                rows: rows.iter().map(|r| Seen::from(r.row)).collect(),
            });
        }
        Ok(rows
            .iter()
            .map(|_| StepRowResult {
                tokens: vec![NEXT],
                margins: vec![1.0],
            })
            .collect())
    }
}

fn scheduler(n: usize, blocks: usize, device: bool, record: bool) -> (Scheduler<Fake>, Arc<Log>) {
    let log = Arc::new(Log::default());
    let config = SchedulerConfig {
        max_concurrency: n,
        max_context: 8192,
        vision_batch_patches: 4096,
        vision_overlap: false,
        spatial_merge: 2,
        vocab_size: VOCAB,
        eos_token_ids: vec![0],
        image_token_id: 50,
        embedding_cache_bytes: 0,
        embedding_hidden: 8,
        embedding_elem_bytes: 2,
        fallback_seed: 1,
        embedding_release_after_prefill: false,
    };
    let layout = KvLayout {
        num_blocks: blocks,
        block_tokens: BLOCK_TOKENS,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    };
    let exec = Fake {
        device,
        record,
        log: log.clone(),
    };
    (Scheduler::new(config, layout, exec), log)
}

fn request(prompt_len: usize, max_tokens: u32, logprobs: Option<u32>) -> EngineRequest {
    EngineRequest {
        prompt: vec![1; prompt_len],
        images: Vec::new(),
        params: SamplingParams {
            temperature: 0.0,
            max_tokens: Some(max_tokens),
            logprobs,
            ..SamplingParams::default()
        },
    }
}

type Rx = tokio::sync::mpsc::UnboundedReceiver<scriptor_engine::EngineEvent>;

fn submit(s: &mut Scheduler<Fake>, req: EngineRequest, keep: &mut Vec<Rx>) -> u64 {
    let (tx, rx) = unbounded_channel();
    keep.push(rx);
    s.submit(req, tx).unwrap()
}

/// Checks the invariants every decode call must satisfy: rows ascend by admission order
/// (request ids are assigned in submission order), hold a block table exactly as long as the
/// sequence's length needs, and share no block with another row of the call.
fn check_call(call: &Call) {
    assert!(
        call.rows.windows(2).all(|w| w[0].seq < w[1].seq),
        "rows out of age order: {:?}",
        call.rows.iter().map(|r| r.seq).collect::<Vec<_>>()
    );
    let mut all: Vec<BlockId> = Vec::new();
    for r in &call.rows {
        assert_eq!(
            r.blocks.len(),
            (r.position + 1).div_ceil(BLOCK_TOKENS),
            "stale or short block table for seq {}",
            r.seq
        );
        all.extend(&r.blocks);
    }
    let n = all.len();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), n, "a block is shared between rows");
}

#[test]
fn device_rows_are_complete_ordered_and_fed_the_previous_token() {
    let (mut s, log) = scheduler(8, 4096, true, true);
    let mut keep = Vec::new();
    let ids: Vec<u64> = (0..6)
        .map(|i| submit(&mut s, request(20 + i * 13, 10, None), &mut keep))
        .collect();
    s.run_until_idle();
    let calls = log.calls.lock().unwrap().clone();
    assert!(!calls.is_empty());
    let mut last_position = std::collections::HashMap::new();
    for call in &calls {
        assert!(call.device);
        check_call(call);
        for r in &call.rows {
            if let Some(prev) = last_position.insert(r.seq, r.position) {
                assert_eq!(r.position, prev + 1, "position skipped for {}", r.seq);
                assert_eq!(r.token, NEXT, "input is not the previous output");
            }
        }
    }
    for id in ids {
        assert!(last_position.contains_key(&id));
    }
}

#[test]
fn host_rows_match_device_rows_after_the_seeded_first_step() {
    let run = |device: bool| {
        let (mut s, log) = scheduler(8, 4096, device, true);
        let mut keep = Vec::new();
        for i in 0..6 {
            submit(&mut s, request(20 + i * 13, 10, None), &mut keep);
        }
        s.run_until_idle();
        let calls = log.calls.lock().unwrap().clone();
        calls.into_iter().map(|c| c.rows).collect::<Vec<_>>()
    };
    // The device step starts one position earlier, on the prompt's last token, because the
    // prefill returned no logits; every later call must equal the host-sampled call.
    let (device, host) = (run(true), run(false));
    assert_eq!(device.len(), host.len() + 1);
    assert!(device[1..] == host[..], "device and host rows differ");
}

#[test]
fn mixed_partition_keeps_age_order_inside_each_side() {
    let (s, log) = scheduler(8, 4096, true, true);
    let mut keep = Vec::new();
    // Even requests ask for log-probabilities and so decode on the host.
    let mut s2 = s;
    let mut ids = Vec::new();
    for i in 0..8usize {
        let lp = (i % 2 == 0).then_some(1);
        ids.push((
            submit(&mut s2, request(18 + i, 6, lp), &mut keep),
            lp.is_some(),
        ));
    }
    s2.run_until_idle();
    let calls = log.calls.lock().unwrap().clone();
    let host_ids: Vec<u64> = ids.iter().filter(|(_, h)| *h).map(|(i, _)| *i).collect();
    let mut saw_host = false;
    let mut saw_device = false;
    for call in &calls {
        check_call(call);
        for r in &call.rows {
            let expect_host = host_ids.contains(&r.seq);
            assert_eq!(call.device, !expect_host, "seq {} on the wrong side", r.seq);
        }
        saw_host |= !call.device;
        saw_device |= call.device;
    }
    assert!(saw_host && saw_device, "both partitions must run");
}

#[test]
fn recycled_rows_carry_no_state_of_earlier_steps() {
    // Staggered lengths shrink the batch step by step, so rows return to the spare list with
    // long tables and are reused by sequences with shorter ones.
    let (mut s, log) = scheduler(16, 8192, true, true);
    let mut keep = Vec::new();
    for i in 0..12u32 {
        submit(&mut s, request(600, 2 + i * 3, None), &mut keep);
    }
    submit(&mut s, request(5, 40, None), &mut keep);
    s.run_until_idle();
    let calls = log.calls.lock().unwrap().clone();
    assert!(
        calls.windows(2).any(|w| w[1].rows.len() < w[0].rows.len()),
        "batch never shrank"
    );
    for call in &calls {
        check_call(call);
    }
}

#[test]
fn churn_with_continuous_admission_keeps_every_call_consistent() {
    let (mut s, log) = scheduler(8, 700, true, true);
    let mut keep = Vec::new();
    let mut submitted = 0u32;
    for round in 0..120u32 {
        if round % 3 == 0 && submitted < 40 {
            submit(
                &mut s,
                request(
                    30 + (submitted as usize * 17) % 200,
                    3 + submitted % 9,
                    None,
                ),
                &mut keep,
            );
            submitted += 1;
        }
        s.step();
    }
    s.run_until_idle();
    let m = s.metrics();
    assert_eq!(m.running + m.waiting, 0);
    for call in log.calls.lock().unwrap().iter() {
        check_call(call);
    }
}

#[test]
fn preparation_time_is_reported_apart_from_executor_time() {
    let (mut s, _) = scheduler(8, 4096, true, false);
    let mut keep = Vec::new();
    for i in 0..8 {
        submit(&mut s, request(100 + i, 12, None), &mut keep);
    }
    s.run_until_idle();
    let m = s.metrics();
    assert!(m.decode_steps > 0);
    assert!(m.decode_prep_seconds > 0.0);
    assert!(m.decode_seconds >= 0.0);
}

/// Steady decode of `n` long sequences; per-step host preparation against executor time and
/// the allocations of the whole step.
fn measure(n: usize, prompt: usize, device: bool) {
    let blocks = n * ((prompt + 200) / BLOCK_TOKENS + 2);
    let (mut s, _) = scheduler(n, blocks, device, false);
    let mut keep = Vec::new();
    for _ in 0..n {
        submit(&mut s, request(prompt, 200, None), &mut keep);
    }
    let mut guard = 0;
    while s.metrics().running < n || s.metrics().waiting > 0 {
        s.step();
        guard += 1;
        assert!(guard < 10_000);
    }
    const WARM: usize = 8;
    const STEPS: usize = 64;
    for _ in 0..WARM {
        s.step();
    }
    let (m0, wall, a0) = (s.metrics(), Instant::now(), allocs());
    for _ in 0..STEPS {
        assert_eq!(s.step().decode_rows, n);
    }
    let (wall, a1, m1) = (wall.elapsed(), allocs(), s.metrics());
    let per = |x: f64| x / STEPS as f64 * 1e6;
    println!(
        "{} n={n:<4} blocks/seq={:<4} step {:>8.1} us | prep {:>8.1} us | executor {:>8.1} us | allocs/step {:>6} | bytes/step {:>9}",
        if device { "device" } else { "host  " },
        prompt / BLOCK_TOKENS,
        wall.as_secs_f64() / STEPS as f64 * 1e6,
        per(m1.decode_prep_seconds - m0.decode_prep_seconds),
        per(m1.decode_seconds - m0.decode_seconds),
        (a1.0 - a0.0) / STEPS as u64,
        (a1.1 - a0.1) / STEPS as u64,
    );
}

#[test]
#[ignore = "host-cost measurement"]
fn measure_prep_vs_executor_fixture_host_cost() {
    println!("fixture host cost on a CPU fake executor; not GPU timing");
    for device in [true, false] {
        for n in [64, 128] {
            for prompt in [64usize, 1024, 4096] {
                measure(n, prompt, device);
            }
        }
    }
}
