//! Demand admission progress: the oldest request starts when nothing can drain, and the
//! optional margin still holds back later requests while work runs.

use crate::EngineRequest;
use crate::error::EngineError;
use crate::request::SamplingParams;
use crate::scheduler::{DecodeRow, ModelExecutor, PrefillRow, Scheduler, SchedulerConfig};
use crate::sequence::EngineEvent;
use scriptor_cache::{HostBuffer, KvLayout};
use scriptor_spec::KvPolicy;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

struct Ahead;

impl ModelExecutor for Ahead {
    type Embedding = HostBuffer;

    fn vision(
        &mut self,
        _jobs: &[crate::scheduler::VisionJob],
    ) -> Result<Vec<HostBuffer>, EngineError> {
        Ok(Vec::new())
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        Ok(rows.iter().map(|_| logits()).collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        Ok(rows.iter().map(|_| logits()).collect())
    }

    fn lookahead_tokens(&self) -> usize {
        4
    }
}

fn logits() -> Vec<f32> {
    let mut l = vec![0.0f32; 128];
    l[100] = 10.0;
    l
}

/// Pool of exactly the blocks one maximal request needs: 64 tokens plus 4 lookahead rows.
fn scheduler(policy: KvPolicy) -> Scheduler<Ahead> {
    let config = SchedulerConfig {
        max_concurrency: 2,
        max_context: 64,
        vision_batch_patches: 64,
        vision_overlap: true,
        spatial_merge: 2,
        vocab_size: 128,
        eos_token_ids: vec![0],
        image_token_id: 50,
        embedding_cache_bytes: 1 << 20,
        embedding_hidden: 4,
        embedding_elem_bytes: 2,
        fallback_seed: 1,
        embedding_release_after_prefill: false,
    };
    let layout = KvLayout {
        num_blocks: 5,
        block_tokens: 16,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    };
    let mut s = Scheduler::new(config, layout, Ahead);
    s.set_kv_policy(policy);
    s
}

fn request(prompt: usize, max_tokens: u32) -> EngineRequest {
    EngineRequest {
        prompt: vec![1; prompt],
        images: Vec::new(),
        params: SamplingParams {
            temperature: 0.0,
            max_tokens: Some(max_tokens),
            ..SamplingParams::default()
        },
    }
}

fn finished(rx: &mut UnboundedReceiver<EngineEvent>) -> usize {
    let mut n = 0;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            EngineEvent::Finished(_) => n += 1,
            EngineEvent::Token(_) => {}
            other => panic!("unexpected terminal event {other:?}"),
        }
    }
    n
}

fn drive(s: &mut Scheduler<Ahead>, rx: &mut UnboundedReceiver<EngineEvent>, want: usize) {
    let mut done = 0;
    for _ in 0..200 {
        s.step();
        done += finished(rx);
        if done == want {
            break;
        }
    }
    assert_eq!(done, want);
    assert!(s.is_idle());
    assert_eq!(s.pool.stats().sequences, 0);
}

#[test]
fn a_near_full_prompt_in_a_pool_of_one_context_admits_and_completes() {
    let mut s = scheduler(KvPolicy::DemandDiagnostic);
    let (tx, mut rx) = unbounded_channel();
    s.submit(request(60, 4), tx).unwrap();
    assert_eq!(s.step().admitted, 1);
    drive(&mut s, &mut rx, 1);
}

#[test]
fn progress_after_an_eviction_needs_no_margin_when_nothing_runs() {
    let mut s = scheduler(KvPolicy::DemandDiagnostic);
    s.evicted_since_admit = true;
    let (tx, mut rx) = unbounded_channel();
    s.submit(request(60, 4), tx).unwrap();
    assert_eq!(s.step().admitted, 1);
    assert!(!s.evicted_since_admit);
    drive(&mut s, &mut rx, 1);
}

#[test]
fn the_margin_holds_back_a_second_request_while_work_runs() {
    let mut s = scheduler(KvPolicy::DemandDiagnostic);
    let (tx, mut rx) = unbounded_channel();
    s.submit(request(60, 4), tx.clone()).unwrap();
    s.submit(request(1, 2), tx).unwrap();
    assert_eq!(s.step().admitted, 1);
    assert_eq!(s.waiting.len(), 1);
    drive(&mut s, &mut rx, 2);
}

#[test]
fn a_front_that_does_not_fit_the_free_blocks_waits() {
    let mut s = scheduler(KvPolicy::DemandDiagnostic);
    let (tx, mut rx) = unbounded_channel();
    s.submit(request(60, 4), tx.clone()).unwrap();
    assert_eq!(s.step().admitted, 1);
    s.submit(request(40, 4), tx).unwrap();
    assert_eq!(s.step().admitted, 0);
    assert_eq!(s.waiting.len(), 1);
    drive(&mut s, &mut rx, 2);
}

#[test]
fn the_default_policy_is_reserve() {
    let s = scheduler(KvPolicy::default());
    assert_eq!(s.kv_policy(), KvPolicy::Reserve);
}
