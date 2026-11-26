//! Host-cost measurements for this fixture: the embedding accounting walk, `metrics()` and the
//! scheduler step on a CPU-only executor, at 64/128/256 sequences with a populated embedding
//! cache. The timed state is verified through executor counters: every sequence has finished
//! vision and prefill, and each timed step decodes exactly N sequences with no vision, prefill
//! or termination. Ignored by default; run with
//! `cargo test --release -p scriptor-engine --test embedding_usage_cost -- --ignored --nocapture`.

use scriptor_cache::{HostBuffer, ImageGrid, KvLayout, embedding_key};
use scriptor_engine::request::SamplingParams;
use scriptor_engine::{
    DecodeRow, EngineError, EngineRequest, ImageInput, ModelExecutor, PrefillRow, Scheduler,
    SchedulerConfig, VisionJob,
};
use std::collections::HashSet;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;
use tokio::sync::mpsc::unbounded_channel;

const VOCAB: usize = 128;
const IMAGE: u32 = 50;
const HIDDEN: usize = 1536;
const ELEM: usize = 2;
const BUDGET: usize = 512 << 20;

#[derive(Default)]
struct Counters {
    vision_calls: AtomicU64,
    vision_jobs: AtomicU64,
    prefill_rows: AtomicU64,
    prefill_completions: AtomicU64,
    decode_rows: AtomicU64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Snap {
    vision_calls: u64,
    vision_jobs: u64,
    prefill_rows: u64,
    prefill_completions: u64,
    decode_rows: u64,
}

impl Counters {
    fn snap(&self) -> Snap {
        Snap {
            vision_calls: self.vision_calls.load(Relaxed),
            vision_jobs: self.vision_jobs.load(Relaxed),
            prefill_rows: self.prefill_rows.load(Relaxed),
            prefill_completions: self.prefill_completions.load(Relaxed),
            decode_rows: self.decode_rows.load(Relaxed),
        }
    }
}

struct Cpu(Arc<Counters>);

fn logits() -> Vec<f32> {
    let mut l = vec![0.0f32; VOCAB];
    l[100] = 10.0;
    l
}

impl ModelExecutor for Cpu {
    type Embedding = HostBuffer;
    fn vision(&mut self, jobs: &[VisionJob]) -> Result<Vec<HostBuffer>, EngineError> {
        self.0.vision_calls.fetch_add(1, Relaxed);
        self.0.vision_jobs.fetch_add(jobs.len() as u64, Relaxed);
        Ok(jobs
            .iter()
            .map(|j| HostBuffer(vec![0u8; j.patches / 4 * HIDDEN * ELEM]))
            .collect())
    }
    fn prefill(&mut self, rows: &[PrefillRow<HostBuffer>]) -> Result<Vec<Vec<f32>>, EngineError> {
        self.0.prefill_rows.fetch_add(rows.len() as u64, Relaxed);
        let done = rows.iter().filter(|r| r.logits).count() as u64;
        self.0.prefill_completions.fetch_add(done, Relaxed);
        Ok(rows.iter().map(|_| logits()).collect())
    }
    fn decode(&mut self, rows: &[DecodeRow]) -> Result<Vec<Vec<f32>>, EngineError> {
        self.0.decode_rows.fetch_add(rows.len() as u64, Relaxed);
        Ok(rows.iter().map(|_| logits()).collect())
    }
}

fn scheduler(n: usize, counters: Arc<Counters>) -> Scheduler<Cpu> {
    let config = SchedulerConfig {
        max_concurrency: n,
        max_context: 2048,
        vision_batch_patches: 4096,
        vision_overlap: false,
        spatial_merge: 2,
        vocab_size: VOCAB,
        eos_token_ids: vec![0],
        image_token_id: IMAGE,
        embedding_cache_bytes: BUDGET,
        embedding_hidden: HIDDEN,
        embedding_elem_bytes: ELEM,
        fallback_seed: 1,
        embedding_release_after_prefill: false,
    };
    let layout = KvLayout {
        num_blocks: 128 * n + 64,
        block_tokens: 16,
        kv_bytes_per_block: 1024,
        mtp_kv_bytes_per_block: 0,
    };
    Scheduler::new(config, layout, Cpu(counters))
}

fn image(seed: u32, big: bool) -> ImageInput {
    let grid = if big {
        ImageGrid { t: 1, h: 24, w: 24 }
    } else {
        ImageGrid { t: 1, h: 16, w: 16 }
    };
    ImageInput {
        grid,
        key: embedding_key(&seed.to_le_bytes(), grid),
        payload: Arc::new(vec![0u8]),
    }
}

fn request(images: Vec<ImageInput>, max_tokens: u32) -> EngineRequest {
    let mut prompt = vec![1u32, 2];
    for img in &images {
        prompt.extend(std::iter::repeat_n(
            IMAGE,
            img.grid.t as usize * img.grid.h as usize * img.grid.w as usize / 4,
        ));
        prompt.push(1);
    }
    EngineRequest {
        prompt,
        images,
        params: SamplingParams {
            temperature: 0.0,
            max_tokens: Some(max_tokens),
            ..SamplingParams::default()
        },
    }
}

fn stats(mut us: Vec<f64>) -> (f64, f64) {
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        us[us.len() / 2],
        us[(us.len() * 99 / 100).min(us.len() - 1)],
    )
}

fn time<R>(iters: usize, mut f: impl FnMut() -> R) -> (f64, f64) {
    for _ in 0..iters / 10 + 1 {
        black_box(f());
    }
    let mut v = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        black_box(f());
        v.push(t.elapsed().as_secs_f64() * 1e6);
    }
    stats(v)
}

const SAMPLES: usize = 300;
const WARMUP: usize = 31;

fn bytes_of(img: &ImageInput) -> usize {
    img.grid.t as usize * img.grid.h as usize * img.grid.w as usize / 4 * HIDDEN * ELEM
}

fn measure(n: usize) {
    let counters = Arc::new(Counters::default());
    let mut s = scheduler(n, counters.clone());
    let mut keep = Vec::new();
    // Fill the cache to its budget with finished requests of distinct images.
    let mut seed = 1_000_000u32;
    let mut fill: Vec<(u32, bool)> = Vec::new();
    while s.metrics().embedding_cache.evictions == 0 {
        let (tx, rx) = unbounded_channel();
        let imgs = (0..2)
            .map(|k| {
                seed += 1;
                fill.push((seed, k == 1));
                image(seed, k == 1)
            })
            .collect();
        s.submit(request(imgs, 1), tx).unwrap();
        keep.push(rx);
        s.run_until_idle();
        keep.clear();
    }
    // Fill keys still cached after the fill, newest last.
    let survivors: Vec<(u32, bool)> = fill
        .iter()
        .copied()
        .filter(|&(sd, big)| s.plan_reuse(&[image(sd, big)]).hit_count() == 1)
        .collect();
    assert!(survivors.len() >= 8, "too few cached fill keys");
    let baseline_hits = s.metrics().embedding_cache.hits;
    let cache_before_live = s.metrics().embedding_cache.entries;
    // Live work: half image requests (1-4 images). Per image: a third reuse a key still in
    // the prepopulated cache, a third share keys among live requests, a third are unique.
    let pool = (n as u32).max(8);
    let mut live_keys: Vec<(scriptor_cache::EmbeddingKey, usize, ImageGrid)> = Vec::new();
    let mut live_images = 0usize;
    let mut probe_hits = 0usize;
    for i in 0..n as u32 {
        let (tx, rx) = unbounded_channel();
        let count = if i % 2 == 0 { 0 } else { 1 + (i / 2) % 4 };
        let imgs: Vec<ImageInput> = (0..count)
            .map(|k| {
                let (sd, big) = match (i + k) % 3 {
                    0 => survivors[(i as usize * 5 + k as usize * 11) % survivors.len()],
                    1 => ((i * 7 + k * 13) % pool, (i + k) % 2 == 0),
                    _ => (2_000_000 + i * 8 + k, (i + k) % 2 == 0),
                };
                image(sd, big)
            })
            .collect();
        live_images += imgs.len();
        probe_hits += s.plan_reuse(&imgs).hit_count();
        live_keys.extend(imgs.iter().map(|m| (m.key, bytes_of(m), m.grid)));
        let req = request(imgs, 600);
        s.submit(req, tx).unwrap();
        keep.push(rx);
    }
    // Run until every sequence has completed prefill and no vision or prefill work remains.
    let live_start = counters.snap();
    let mut guard = 0;
    loop {
        let before = counters.snap();
        let report = s.step();
        let after = counters.snap();
        let quiet = after.vision_jobs == before.vision_jobs
            && after.prefill_rows == before.prefill_rows
            && report.vision_images == 0
            && report.prefill_rows == 0;
        if after.prefill_completions - live_start.prefill_completions == n as u64
            && quiet
            && s.metrics().running == n
        {
            break;
        }
        guard += 1;
        assert!(
            guard < 10_000,
            "sequences never finished vision and prefill"
        );
    }
    let settled = counters.snap();
    let m = s.metrics();
    assert_eq!(
        settled.prefill_completions - live_start.prefill_completions,
        n as u64
    );
    assert_eq!((m.running, m.waiting), (n, 0), "not all sequences active");
    let (pinned, retained) = (m.pinned_embedding_bytes, m.retained_embedding_bytes);
    let cache = m.embedding_cache;
    let live_hits = cache.hits - baseline_hits;

    // Distinct live buffers and those also held by the cache.
    let mut seen = HashSet::new();
    let distinct: Vec<_> = live_keys
        .iter()
        .filter(|(k, _, _)| seen.insert(*k))
        .collect();
    let distinct_bytes: usize = distinct.iter().map(|(_, b, _)| b).sum();
    let shared = distinct
        .iter()
        .filter(|(k, _, g)| {
            let img = ImageInput {
                grid: *g,
                key: *k,
                payload: Arc::new(vec![0u8]),
            };
            s.plan_reuse(&[img]).hit_count() == 1
        })
        .count();
    // Requests that embed one key in the same vision batch hold separate buffers, so pinned
    // bytes can exceed one buffer per distinct key.
    assert!(
        pinned >= distinct_bytes,
        "pinned bytes below distinct live keys"
    );
    let duplicate_bytes = pinned - distinct_bytes;
    let cache_only_bytes = retained - pinned;
    let shared_bytes = cache.bytes - cache_only_bytes;

    // Accounting walk and metrics() timed in the verified steady state (no stepping between).
    let (u_med, u_p99) = time(2000, || s.retained_embedding_bytes());
    let (m_med, m_p99) = time(2000, || s.metrics());
    assert_eq!(counters.snap(), settled);

    // Timed steps: every sample must decode exactly N sequences and do nothing else.
    let mut samples = Vec::with_capacity(SAMPLES);
    let (mut dmin, mut dmax) = (u64::MAX, 0u64);
    for i in 0..WARMUP + SAMPLES {
        let before = counters.snap();
        let t = Instant::now();
        let report = s.step();
        let us = t.elapsed().as_secs_f64() * 1e6;
        let after = counters.snap();
        let decoded = after.decode_rows - before.decode_rows;
        assert_eq!(decoded, n as u64, "step {i} decoded {decoded} rows");
        assert_eq!(report.decode_rows, n, "step {i} report decode rows");
        assert_eq!(
            after.vision_calls, before.vision_calls,
            "step {i} ran vision"
        );
        assert_eq!(after.vision_jobs, before.vision_jobs, "step {i} ran vision");
        assert_eq!(
            after.prefill_rows, before.prefill_rows,
            "step {i} ran prefill"
        );
        assert_eq!(
            (report.vision_images, report.prefill_rows, report.admitted),
            (0, 0, 0),
            "step {i} did other work"
        );
        assert_eq!(
            (report.finished, report.preempted),
            (0, 0),
            "step {i} terminated"
        );
        if i >= WARMUP {
            samples.push(us);
            dmin = dmin.min(decoded);
            dmax = dmax.max(decoded);
        }
    }
    let end = s.metrics();
    assert_eq!((end.running, end.waiting, end.finished), (n, 0, m.finished));
    let (st_med, st_p99) = stats(samples);
    println!(
        "N={n:4} | cache_entries={} (before live {cache_before_live}) live_images={live_images} \
         admission_probe_hits={probe_hits} cache_hits_during_live={live_hits} | \
         distinct_live_keys={} live_keys_also_cached={shared} shared_bytes_seq_and_cache={}KiB duplicate_live_buffer_bytes={}KiB cache_only_entries={} ({}MiB) \
         pinned={}MiB retained={}MiB | decode_rows/sample min={dmin} max={dmax} over {SAMPLES} \
         samples | usage med={u_med:8.1}us p99={u_p99:8.1}us | metrics() med={m_med:8.1}us \
         p99={m_p99:8.1}us | step med={st_med:9.1}us p99={st_p99:9.1}us",
        cache.entries,
        distinct.len(),
        shared_bytes >> 10,
        duplicate_bytes >> 10,
        cache.entries - shared,
        cache_only_bytes >> 20,
        pinned >> 20,
        retained >> 20,
    );
    drop(keep);
}

#[test]
#[ignore = "measurement"]
fn embedding_usage_cost() {
    println!("host-cost measurements for this CPU-only fixture; no projections to other hardware");
    for n in [64, 128, 256] {
        measure(n);
    }
}
