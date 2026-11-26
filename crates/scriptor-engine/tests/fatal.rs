use scriptor_cache::{HostBuffer, KvLayout};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::{
    DecodeRow, EngineError, EngineEvent, EngineHandle, EngineRequest, ModelExecutor, PrefillRow,
    RequestId, Scheduler, SchedulerConfig, VisionJob, fatal_slot,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::mpsc::unbounded_channel;

const VOCAB: usize = 8;

/// Executor whose `decode` raises `error` and counts every call it receives.
struct Faulty {
    error: EngineError,
    calls: Arc<AtomicUsize>,
}

fn logits() -> Vec<f32> {
    let mut l = vec![0.0f32; VOCAB];
    l[3] = 10.0;
    l
}

impl ModelExecutor for Faulty {
    type Embedding = HostBuffer;

    fn vision(&mut self, _: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        Ok(Vec::new())
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(rows.iter().map(|_| logits()).collect())
    }

    fn decode(&mut self, _: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(self.error.clone())
    }

    fn release(&mut self, _: RequestId) {}
}

fn config() -> SchedulerConfig {
    SchedulerConfig {
        max_concurrency: 4,
        max_context: 256,
        vision_batch_patches: 64,
        vision_overlap: false,
        spatial_merge: 2,
        vocab_size: VOCAB,
        eos_token_ids: vec![0],
        image_token_id: 7,
        embedding_cache_bytes: 0,
        embedding_hidden: 4,
        embedding_elem_bytes: 2,
        fallback_seed: 1,
        embedding_release_after_prefill: false,
    }
}

fn layout() -> KvLayout {
    KvLayout {
        num_blocks: 64,
        block_tokens: 16,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    }
}

fn request() -> EngineRequest {
    EngineRequest {
        prompt: vec![1, 2, 3],
        images: Vec::new(),
        params: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
    }
}

fn fatal() -> EngineError {
    EngineError::fatal(700, "CUDA_ERROR_ILLEGAL_ADDRESS: injected")
}

#[test]
fn fatal_fault_fails_every_request_and_stops_device_calls() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut s = Scheduler::new(
        config(),
        layout(),
        Faulty {
            error: fatal(),
            calls: Arc::clone(&calls),
        },
    );
    let mut rxs = Vec::new();
    for _ in 0..6 {
        let (tx, rx) = unbounded_channel();
        s.submit(request(), tx).unwrap();
        rxs.push(rx);
    }
    for _ in 0..10 {
        s.step();
    }
    assert_eq!(s.fatal(), Some(&fatal()));
    for rx in &mut rxs {
        let mut last = None;
        while let Ok(ev) = rx.try_recv() {
            last = Some(ev);
        }
        match last {
            Some(EngineEvent::Failed(e)) => assert_eq!(e, fatal()),
            other => panic!("expected a fatal failure, got {other:?}"),
        }
    }
    assert!(s.is_idle());
    let before = calls.load(Ordering::SeqCst);
    s.step();
    assert_eq!(calls.load(Ordering::SeqCst), before);
}

#[test]
fn recoverable_executor_error_does_not_stop_the_scheduler() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut s = Scheduler::new(
        config(),
        layout(),
        Faulty {
            error: EngineError::Executor("out of memory".into()),
            calls,
        },
    );
    let (tx, mut rx) = unbounded_channel();
    s.submit(request(), tx).unwrap();
    for _ in 0..10 {
        s.step();
    }
    assert!(s.fatal().is_none());
    let mut last = None;
    while let Ok(ev) = rx.try_recv() {
        last = Some(ev);
    }
    assert!(matches!(
        last,
        Some(EngineEvent::Failed(EngineError::Executor(_)))
    ));
}

#[test]
fn handle_publishes_the_fault_and_refuses_later_requests() {
    let slot = fatal_slot();
    let mut watcher = slot.subscribe();
    let handle = EngineHandle::spawn_with_fatal(
        config(),
        layout(),
        Faulty {
            error: fatal(),
            calls: Arc::new(AtomicUsize::new(0)),
        },
        Arc::clone(&slot),
    );
    assert!(handle.fatal().is_none());
    let mut stream = handle.submit(request()).unwrap();
    let mut terminal = None;
    while let Some(ev) = stream.events.blocking_recv() {
        terminal = Some(ev);
    }
    assert!(matches!(
        terminal,
        Some(EngineEvent::Failed(EngineError::Fatal { code: 700, .. }))
    ));
    assert_eq!(*watcher.borrow_and_update(), Some(fatal()));
    assert_eq!(handle.fatal(), Some(fatal()));
    assert_eq!(handle.submit(request()).err(), Some(fatal()));
    handle.shutdown();
}
