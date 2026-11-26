//! Engine-thread supervision: panics, drain, interrupt and heartbeat on a CPU executor.

use scriptor_cache::{HostBuffer, KvLayout};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::{
    CancelReason, DecodeRow, EngineError, EngineEvent, EngineExit, EngineHandle, EngineRequest,
    ModelExecutor, PrefillRow, Scheduler, SchedulerConfig, VisionJob,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;
use std::time::Duration;
use tokio::sync::mpsc::unbounded_channel;

const VOCAB: usize = 128;

/// Generates token 100 forever; decode can be slowed or made to panic.
struct Endless {
    decode_delay: Duration,
    panic_on_decode: Arc<AtomicBool>,
}

impl Endless {
    fn new() -> Self {
        Self {
            decode_delay: Duration::ZERO,
            panic_on_decode: Arc::new(AtomicBool::new(false)),
        }
    }
}

fn logits() -> Vec<f32> {
    let mut l = vec![0.0f32; VOCAB];
    l[100] = 10.0;
    l
}

impl ModelExecutor for Endless {
    type Embedding = HostBuffer;

    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        Ok(jobs.iter().map(|_| HostBuffer(Vec::new())).collect())
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        Ok(rows.iter().map(|_| logits()).collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        if self.panic_on_decode.load(Ordering::SeqCst) {
            panic!("injected decode panic");
        }
        std::thread::sleep(self.decode_delay);
        Ok(rows.iter().map(|_| logits()).collect())
    }
}

fn config() -> SchedulerConfig {
    SchedulerConfig {
        max_concurrency: 4,
        max_context: 4096,
        vision_batch_patches: 64,
        vision_overlap: false,
        spatial_merge: 2,
        vocab_size: VOCAB,
        eos_token_ids: vec![0],
        image_token_id: 50,
        embedding_cache_bytes: 1 << 16,
        embedding_hidden: 4,
        embedding_elem_bytes: 2,
        fallback_seed: 1,
        embedding_release_after_prefill: false,
    }
}

fn layout() -> KvLayout {
    KvLayout {
        num_blocks: 512,
        block_tokens: 16,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    }
}

fn request(max_tokens: Option<u32>) -> EngineRequest {
    EngineRequest {
        prompt: vec![1, 2, 3],
        images: Vec::new(),
        params: SamplingParams {
            temperature: 0.0,
            max_tokens,
            ..SamplingParams::default()
        },
    }
}

fn terminal(stream: &mut scriptor_engine::RequestStream) -> EngineEvent {
    loop {
        match stream.events.blocking_recv().expect("terminal event") {
            EngineEvent::Token(_) => {}
            other => return other,
        }
    }
}

#[test]
fn step_panic_fails_requests_marks_dead_and_reports_exit() {
    let exec = Endless::new();
    let flag = Arc::clone(&exec.panic_on_decode);
    let (exit_tx, exit_rx) = channel();
    let handle = EngineHandle::spawn_supervised(config(), layout(), exec, move |e| {
        let _ = exit_tx.send(e);
    });
    let mut stream = handle.submit(request(Some(50))).unwrap();
    flag.store(true, Ordering::SeqCst);
    let EngineEvent::Failed(err) = terminal(&mut stream) else {
        panic!("failure expected");
    };
    assert!(err.to_string().contains("injected decode panic"), "{err}");
    let EngineExit::Panicked(text) = exit_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(text.contains("injected decode panic"));
    assert!(handle.dead_reason().unwrap().contains("panicked"));
    assert!(matches!(
        handle.submit(request(Some(5))),
        Err(EngineError::Closed)
    ));
    assert!(!handle.shutdown(), "the join reports the dead engine");
}

#[test]
fn queued_submissions_fail_when_the_engine_dies() {
    let mut exec = Endless::new();
    exec.decode_delay = Duration::from_millis(20);
    let flag = Arc::clone(&exec.panic_on_decode);
    let handle = EngineHandle::spawn_supervised(config(), layout(), exec, |_| {});
    let mut first = handle.submit(request(Some(50))).unwrap();
    flag.store(true, Ordering::SeqCst);
    let mut late = Vec::new();
    for _ in 0..3 {
        late.push(handle.submit(request(Some(5))));
    }
    assert!(matches!(terminal(&mut first), EngineEvent::Failed(_)));
    for s in late.into_iter().flatten() {
        let mut s = s;
        assert!(matches!(terminal(&mut s), EngineEvent::Failed(_)));
    }
}

#[test]
fn interrupt_cancels_in_flight_requests_with_the_drain_reason() {
    let mut exec = Endless::new();
    exec.decode_delay = Duration::from_millis(2);
    let handle = EngineHandle::spawn(config(), layout(), exec);
    let mut a = handle.submit(request(None)).unwrap();
    let mut b = handle.submit(request(None)).unwrap();
    assert!(matches!(
        a.events.blocking_recv(),
        Some(EngineEvent::Token(_))
    ));
    handle.interrupt();
    assert_eq!(
        terminal(&mut a),
        EngineEvent::Cancelled(CancelReason::Drain)
    );
    assert_eq!(
        terminal(&mut b),
        EngineEvent::Cancelled(CancelReason::Drain)
    );
    assert!(matches!(
        handle.submit(request(Some(1))),
        Err(EngineError::Draining)
    ));
    assert!(handle.shutdown());
    let m = handle.metrics();
    assert_eq!(
        m.pool.used_blocks, 0,
        "cancelled requests release their blocks"
    );
    assert_eq!(m.running + m.waiting, 0);
}

#[test]
fn draining_refuses_new_requests_and_lets_running_ones_finish() {
    let handle = EngineHandle::spawn(config(), layout(), Endless::new());
    let mut running = handle.submit(request(Some(8))).unwrap();
    handle.begin_drain();
    assert!(handle.is_draining());
    assert!(matches!(
        handle.submit(request(Some(8))),
        Err(EngineError::Draining)
    ));
    assert!(matches!(terminal(&mut running), EngineEvent::Finished(_)));
    assert!(handle.shutdown());
}

#[test]
fn heartbeat_advances_and_goes_quiet_when_idle() {
    let handle = EngineHandle::spawn(config(), layout(), Endless::new());
    let before = handle.heartbeat();
    let mut s = handle.submit(request(Some(16))).unwrap();
    assert!(matches!(terminal(&mut s), EngineEvent::Finished(_)));
    std::thread::sleep(Duration::from_millis(50));
    let after = handle.heartbeat();
    assert!(after.beats > before.beats);
    assert!(!after.busy);
    assert_eq!(after.age, Duration::ZERO);
    handle.shutdown();
}

#[test]
fn drain_deadline_follows_tokens_in_flight_and_measured_rate() {
    let mut exec = Endless::new();
    exec.decode_delay = Duration::from_millis(5);
    let handle = EngineHandle::spawn(config(), layout(), exec);
    assert_eq!(
        handle.drain_deadline(None),
        Duration::ZERO,
        "nothing in flight"
    );
    let mut warm = handle.submit(request(Some(20))).unwrap();
    assert!(matches!(terminal(&mut warm), EngineEvent::Finished(_)));
    let mut long = handle.submit(request(Some(4000))).unwrap();
    assert!(matches!(
        long.events.blocking_recv(),
        Some(EngineEvent::Token(_))
    ));
    std::thread::sleep(Duration::from_millis(50));
    let m = handle.metrics();
    assert!(m.tokens_in_flight > 3000 && m.tokens_in_flight < 4000);
    let deadline = handle.drain_deadline(None);
    assert!(deadline > Duration::from_secs(1), "{deadline:?}");
    let bound = Duration::from_millis(250);
    assert_eq!(handle.drain_deadline(Some(bound)), bound);
    handle.interrupt();
    assert!(matches!(terminal(&mut long), EngineEvent::Cancelled(_)));
    handle.shutdown();
}

#[test]
fn interrupted_step_stops_before_prefill_and_cancel_all_releases_everything() {
    let mut s = Scheduler::new(config(), layout(), Endless::new());
    let (tx, mut rx) = unbounded_channel();
    s.submit(request(Some(10)), tx).unwrap();
    let flag = Arc::new(AtomicBool::new(true));
    s.set_interrupt_flag(Arc::clone(&flag));
    let report = s.step();
    assert_eq!(report.prefill_rows + report.decode_rows, 0);
    assert_eq!(s.cancel_all(CancelReason::Drain), 1);
    assert_eq!(
        rx.try_recv().unwrap(),
        EngineEvent::Cancelled(CancelReason::Drain)
    );
    assert!(s.is_idle());
    assert_eq!(s.metrics().pool.used_blocks, 0);
}

#[test]
fn abandon_all_fails_waiting_and_running_without_the_executor() {
    let mut s = Scheduler::new(config(), layout(), Endless::new());
    let (tx, mut rx) = unbounded_channel();
    s.submit(request(Some(10)), tx).unwrap();
    s.step();
    let (tx2, mut rx2) = unbounded_channel();
    s.submit(request(Some(10)), tx2).unwrap();
    let err = EngineError::Executor("lost".into());
    assert_eq!(s.abandon_all(&err), 2);
    let mut last = None;
    while let Ok(e) = rx.try_recv() {
        last = Some(e);
    }
    assert_eq!(last, Some(EngineEvent::Failed(err.clone())));
    assert_eq!(rx2.try_recv().unwrap(), EngineEvent::Failed(err));
    assert!(s.is_idle());
}
