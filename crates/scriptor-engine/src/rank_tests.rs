//! Submission rank survives requeue and keeps pool age order stable.

use crate::EngineRequest;
use crate::error::EngineError;
use crate::request::SamplingParams;
use crate::scheduler::{DecodeRow, ModelExecutor, PrefillRow, Scheduler, SchedulerConfig};
use scriptor_cache::{HostBuffer, KvLayout};
use tokio::sync::mpsc::unbounded_channel;

struct Echo;

impl ModelExecutor for Echo {
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
}

fn logits() -> Vec<f32> {
    let mut l = vec![0.0f32; 128];
    l[100] = 10.0;
    l
}

fn scheduler() -> Scheduler<Echo> {
    let config = SchedulerConfig {
        max_concurrency: 8,
        max_context: 256,
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
        num_blocks: 64,
        block_tokens: 16,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    };
    Scheduler::new(config, layout, Echo)
}

fn request() -> EngineRequest {
    EngineRequest {
        prompt: vec![1; 8],
        images: Vec::new(),
        params: SamplingParams {
            temperature: 0.0,
            max_tokens: Some(200),
            ..SamplingParams::default()
        },
    }
}

#[test]
fn ranks_follow_submission_order_not_ids() {
    let mut s = scheduler();
    let (tx, _rx) = unbounded_channel();
    for id in [7u64, 3, 5] {
        s.submit_with_id(id, request(), tx.clone()).unwrap();
    }
    let ranks: Vec<(u64, u64)> = s.waiting.iter().map(|q| (q.id, q.rank)).collect();
    assert_eq!(ranks, vec![(7, 0), (3, 1), (5, 2)]);
}

#[test]
fn a_requeued_victim_keeps_its_rank_and_is_not_youngest() {
    let mut s = scheduler();
    let (tx, _rx) = unbounded_channel();
    for _ in 0..3 {
        s.submit(request(), tx.clone()).unwrap();
    }
    s.step();
    assert_eq!(s.pool.sequences_by_age(), vec![0, 1, 2]);
    let rank = s.active[&0].rank;
    s.pool.release(0).unwrap();
    s.requeue_preempted(0);
    assert_eq!(s.waiting.front().unwrap().rank, rank);
    s.step();
    assert!(s.active.contains_key(&0));
    assert_eq!(s.active[&0].rank, rank);
    assert_eq!(s.pool.sequences_by_age(), vec![0, 1, 2]);
    assert_eq!(s.pool.youngest(), Some(2));
}
