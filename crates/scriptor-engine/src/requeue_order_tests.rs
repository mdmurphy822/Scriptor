//! Requeued victims re-enter the waiting queue by submission rank, whatever their ids and
//! eviction order.

use crate::EngineRequest;
use crate::error::EngineError;
use crate::request::SamplingParams;
use crate::scheduler::{DecodeRow, ModelExecutor, PrefillRow, Scheduler, SchedulerConfig};
use scriptor_cache::{HostBuffer, KvLayout};
use tokio::sync::mpsc::unbounded_channel;

struct Fixed;

impl ModelExecutor for Fixed {
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

fn request() -> EngineRequest {
    EngineRequest {
        prompt: vec![1; 4],
        images: Vec::new(),
        params: SamplingParams {
            temperature: 0.0,
            max_tokens: Some(8),
            ..SamplingParams::default()
        },
    }
}

#[test]
fn victims_and_waiting_requests_readmit_in_submission_rank_order() {
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
    let mut s = Scheduler::new(config, layout, Fixed);
    let (tx, _rx) = unbounded_channel();
    for id in [50, 7, 90, 3] {
        s.submit_with_id(id, request(), tx.clone()).unwrap();
    }
    assert_eq!(s.step().admitted, 4);
    s.submit_with_id(11, request(), tx).unwrap();

    for victim in [3, 90, 50] {
        s.pool.release(victim).unwrap();
        s.requeue_preempted(victim);
    }
    let queued: Vec<u64> = s.waiting.iter().map(|w| w.id).collect();
    assert_eq!(queued, [50, 90, 3, 11]);

    assert_eq!(s.step().admitted, 4);
    assert!(s.waiting.is_empty());
    assert_eq!(s.pool.sequences_by_age(), [50, 7, 90, 3, 11]);
}
