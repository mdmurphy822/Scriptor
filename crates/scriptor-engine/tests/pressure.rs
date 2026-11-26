//! Host-pressure shedding: the scheduler cancels the youngest request, and the monitor thread
//! walks the engine through Shed and Protect from fake host files.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use scriptor_cache::{HostBuffer, KvLayout};
use scriptor_engine::host_mem::HostPaths;
use scriptor_engine::pressure::{PressureMonitor, PressureState, Thresholds, Tier};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::{
    DecodeRow, EngineError, EngineEvent, EngineHandle, EngineRequest, ModelExecutor, PrefillRow,
    RequestId, Scheduler, SchedulerConfig, VisionJob,
};
use tokio::sync::mpsc::unbounded_channel;

const VOCAB: usize = 128;
const EOS: u32 = 0;

fn logits_for(tokens: &[u32]) -> Vec<f32> {
    let n = tokens[0] as usize;
    let g = tokens.iter().filter(|&&t| t >= 100).count();
    let next = if g >= n { EOS } else { 100 + (g % 20) as u32 };
    let mut l = vec![0.0f32; VOCAB];
    l[next as usize] = 10.0;
    l
}

#[derive(Default)]
struct Exec {
    history: HashMap<RequestId, Vec<u32>>,
    trims: Arc<AtomicUsize>,
    delay: Duration,
}

impl ModelExecutor for Exec {
    type Embedding = HostBuffer;

    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        Ok(jobs.iter().map(|_| HostBuffer(vec![0u8; 8])).collect())
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        for r in rows {
            self.history.insert(r.seq, r.tokens.clone());
        }
        Ok(rows.iter().map(|r| logits_for(&r.tokens)).collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        std::thread::sleep(self.delay);
        let mut out = Vec::new();
        for r in rows {
            let h = self.history.get_mut(&r.seq).expect("prefilled");
            h.push(r.token);
            out.push(logits_for(h));
        }
        Ok(out)
    }

    fn release(&mut self, seq: RequestId) {
        self.history.remove(&seq);
    }

    fn trim_memory(&mut self) {
        self.trims.fetch_add(1, Ordering::SeqCst);
    }
}

fn config() -> SchedulerConfig {
    SchedulerConfig {
        max_concurrency: 4,
        max_context: 4096,
        vision_batch_patches: 4096,
        vision_overlap: false,
        spatial_merge: 2,
        vocab_size: VOCAB,
        eos_token_ids: vec![EOS],
        image_token_id: 50,
        embedding_cache_bytes: 1 << 20,
        embedding_hidden: 4,
        embedding_elem_bytes: 2,
        fallback_seed: 1,
        embedding_release_after_prefill: false,
    }
}

fn layout() -> KvLayout {
    KvLayout {
        num_blocks: 1024,
        block_tokens: 4,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    }
}

fn request(generate: u32) -> EngineRequest {
    let mut prompt = vec![1u32; 8];
    prompt[0] = generate;
    EngineRequest {
        prompt,
        images: Vec::new(),
        params: SamplingParams {
            temperature: 0.0,
            max_tokens: Some(generate),
            ..SamplingParams::default()
        },
    }
}

#[test]
fn shedding_cancels_the_youngest_and_the_oldest_finishes() {
    let mut s = Scheduler::new(config(), layout(), Exec::default());
    let (tx_old, mut old) = unbounded_channel();
    let (tx_new, mut new) = unbounded_channel();
    s.submit(request(30), tx_old).unwrap();
    s.submit(request(30), tx_new).unwrap();
    for _ in 0..3 {
        s.step();
    }
    assert!(s.shed_youngest());
    loop {
        match new.try_recv().expect("youngest ends") {
            EngineEvent::Token(_) => {}
            EngineEvent::Failed(EngineError::Shed) => break,
            other => panic!("unexpected {other:?}"),
        }
    }
    while !s.is_idle() {
        s.step();
    }
    let mut tokens = 0;
    let mut finished = false;
    while let Ok(e) = old.try_recv() {
        match e {
            EngineEvent::Token(_) => tokens += 1,
            EngineEvent::Finished(_) => finished = true,
            other => panic!("oldest must not fail: {other:?}"),
        }
    }
    assert!(finished && tokens == 30);
    assert_eq!(s.metrics().shed, 1);
    assert!(!s.shed_youngest());
}

#[test]
fn shed_all_fails_running_and_waiting() {
    let mut s = Scheduler::new(config(), layout(), Exec::default());
    let mut rxs = Vec::new();
    for _ in 0..6 {
        let (tx, rx) = unbounded_channel();
        s.submit(request(40), tx).unwrap();
        rxs.push(rx);
    }
    s.step();
    assert_eq!(s.shed_all(), 6);
    assert!(s.is_idle());
    for mut rx in rxs {
        let mut last = None;
        while let Ok(e) = rx.try_recv() {
            last = Some(e);
        }
        assert!(matches!(last, Some(EngineEvent::Failed(EngineError::Shed))));
    }
}

fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn monitor_walks_watch_shed_protect_from_fake_files() {
    let dir = std::env::temp_dir().join(format!("scriptor-pressure-it-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let paths = HostPaths {
        meminfo: dir.join("meminfo"),
        psi: dir.join("psi"),
        vm_dir: dir.clone(),
    };
    let write_mem = |kb: u64| {
        let tmp = dir.join("meminfo.tmp");
        std::fs::write(
            &tmp,
            format!("MemTotal: 100000 kB\nMemAvailable: {kb} kB\n"),
        )
        .unwrap();
        std::fs::rename(tmp, dir.join("meminfo")).unwrap();
    };
    write_mem(90_000);
    let trims = Arc::new(AtomicUsize::new(0));
    let exec = Exec {
        trims: Arc::clone(&trims),
        delay: Duration::from_millis(2),
        ..Exec::default()
    };
    let engine = EngineHandle::spawn(config(), layout(), exec);
    let state: Arc<PressureState> = engine.pressure();
    let thresholds = Thresholds {
        watch_below: 30_000 * 1024,
        shed_below: 20_000 * 1024,
        protect_below: 5_000 * 1024,
    };
    let protected = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&protected);
    let guard = PressureMonitor::new(thresholds, paths, Arc::clone(&state))
        .spawn(engine.clone(), move || flag.store(true, Ordering::SeqCst));

    let mut running = engine.submit(request(2000)).expect("admitted at Normal");
    wait_for("normal sample", || {
        state.snapshot().host_mem_available_bytes > 0
    });

    write_mem(25_000);
    wait_for("watch tier", || state.tier() == Tier::Watch);
    assert!(matches!(
        engine.submit(request(5)),
        Err(EngineError::Pressure { .. })
    ));
    assert_eq!(state.snapshot().admission_rejected, 1);

    write_mem(10_000);
    wait_for("shed tier", || state.tier() == Tier::Shed);
    loop {
        match running.events.blocking_recv() {
            Some(EngineEvent::Token(_)) => {}
            Some(EngineEvent::Failed(EngineError::Shed)) => break,
            other => panic!("expected the request to be shed, saw {other:?}"),
        }
    }
    wait_for("pool trim", || trims.load(Ordering::SeqCst) >= 1);
    assert!(state.snapshot().shed_actions >= 1);
    wait_for("published shed count", || engine.metrics().shed == 1);

    write_mem(100);
    wait_for("protect tier", || protected.load(Ordering::SeqCst));
    assert_eq!(state.tier(), Tier::Protect);
    assert_eq!(state.snapshot().protect_actions, 1);

    guard.stop();
    engine.shutdown();
    std::fs::remove_dir_all(&dir).unwrap();
}
