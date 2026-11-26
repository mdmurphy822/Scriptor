use scriptor_cache::{HostBuffer, KvLayout};
use scriptor_engine::pass_ledger::PREFILL_RIDGE_TOKENS;
use scriptor_engine::request::SamplingParams;
use scriptor_engine::{
    DecodeRow, EngineError, EngineEvent, EngineRequest, ModelExecutor, PrefillRow, RequestId,
    Scheduler, SchedulerConfig, VisionJob,
};
use std::collections::HashMap;
use tokio::sync::mpsc::unbounded_channel;

const VOCAB: usize = 128;
const EOS: u32 = 0;

/// CPU executor that generates `prompt[0]` tokens and then stops.
#[derive(Default)]
struct Counting {
    history: HashMap<RequestId, Vec<u32>>,
}

fn logits_for(tokens: &[u32]) -> Vec<f32> {
    let n = tokens[0] as usize;
    let g = tokens.iter().filter(|&&t| t >= 100).count();
    let next = if g >= n { EOS } else { 100 };
    let mut l = vec![0.0f32; VOCAB];
    l[next as usize] = 10.0;
    l
}

impl ModelExecutor for Counting {
    type Embedding = HostBuffer;

    fn vision(&mut self, _jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        Ok(Vec::new())
    }

    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        for r in rows {
            self.history.insert(r.seq, r.tokens.clone());
        }
        Ok(rows.iter().map(|r| logits_for(&r.tokens)).collect())
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
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

fn config(max_context: usize) -> SchedulerConfig {
    SchedulerConfig {
        max_concurrency: 8,
        max_context,
        vision_batch_patches: 64,
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
        block_tokens: 16,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    }
}

fn request(generate: u32, len: usize) -> EngineRequest {
    let mut prompt = vec![generate];
    prompt.resize(len, 1);
    EngineRequest {
        prompt,
        images: Vec::new(),
        params: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
    }
}

fn finish(rx: &mut tokio::sync::mpsc::UnboundedReceiver<EngineEvent>) {
    while let Ok(ev) = rx.try_recv() {
        if matches!(ev, EngineEvent::Finished(_)) {
            return;
        }
    }
    panic!("no terminal event");
}

const _: () = assert!(500 >= PREFILL_RIDGE_TOKENS && 30 < PREFILL_RIDGE_TOKENS);

#[test]
fn pass_sizes_total_the_prompt_tokens_and_split_at_the_ridge() {
    let mut s = Scheduler::new(config(512), layout(), Counting::default());
    let mut rxs = Vec::new();
    for (generate, len) in [(1u32, 30usize), (1, 500)] {
        let (tx, rx) = unbounded_channel();
        s.submit(request(generate, len), tx).unwrap();
        rxs.push(rx);
    }
    s.step();
    let m = s.metrics();
    let p = &m.prefill_pass_stats;
    assert_eq!(m.prefill_passes, 2);
    assert_eq!(p.sizes, vec![(32, 1), (512, 1)]);
    assert_eq!(p.tokens, 530);
    assert_eq!(p.below_ridge_passes, 1);
    assert_eq!(p.below_ridge_tokens, 30);
    s.run_until_idle();
    rxs.iter_mut().for_each(finish);
}

#[test]
fn passes_split_at_the_token_budget_are_counted_one_by_one() {
    let mut s = Scheduler::new(config(64), layout(), Counting::default());
    let mut rxs = Vec::new();
    for g in 1..=4u32 {
        let (tx, rx) = unbounded_channel();
        s.submit(request(g, 30), tx).unwrap();
        rxs.push(rx);
    }
    s.step();
    let p = s.metrics().prefill_pass_stats;
    assert_eq!(p.sizes, vec![(64, 2)]);
    assert_eq!(p.tokens, 120);
    assert_eq!(p.below_ridge_passes, 2);
    assert_eq!(p.below_ridge_tokens, 120);
    s.run_until_idle();
    rxs.iter_mut().for_each(finish);
}
