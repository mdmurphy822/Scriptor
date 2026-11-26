//! Scratch measurement (speedup investigation): decode-step and paged-attention time against
//! the sequence count, the query rows per sequence (MTP verify width) and the cached context.
//! Shapes come from `SCRIPTOR_SHAPE_BATCH`, `SCRIPTOR_SHAPE_QLEN` and `SCRIPTOR_SHAPE_CONTEXT` (comma lists);
//! every combination is timed with the fused layers. Run with
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda --release --test gpu_decode_attention_shapes -- --ignored --nocapture`.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use scriptor_cuda::Device;
use scriptor_kernels::cublaslt::GemmOptions;
use scriptor_kernels::heuristics::TuningTable;
use scriptor_model::gpu::{
    DecodeSeq, Fusion, GpuDecoderWeights, GpuKvPool, StepInput, TextExecutor, TextGeometry,
};
use scriptor_spec::{GenerationConfig, HfConfig, ModelSpec};
use scriptor_weights::SafetensorsDir;

const BLOCK: usize = 16;
const WARMUP: usize = 3;
const ITERS: usize = 20;

fn list(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SCRIPTOR_SNAPSHOT_DIR") {
        return p.into();
    }
    let hf = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".cache/huggingface")
        });
    let snaps = hf.join("hub/models--zai-org--GLM-OCR/snapshots");
    let mut dirs: Vec<_> = std::fs::read_dir(&snaps)
        .expect("snapshot cache")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("config.json").exists() && p.join("generation_config.json").exists())
        .collect();
    dirs.sort_by_key(|p| {
        !p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("2e85a62"))
    });
    dirs.into_iter().next().expect("a snapshot")
}

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn decode_attention_by_shape() {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let batches = list("SCRIPTOR_SHAPE_BATCH", &[32]);
    let qlens = list("SCRIPTOR_SHAPE_QLEN", &[1, 4, 8]);
    let contexts = list("SCRIPTOR_SHAPE_CONTEXT", &[350, 1000, 4000]);
    // SCRIPTOR_SHAPE_LONG="count:ctx" gives the first `count` sequences `ctx` cached tokens;
    // SCRIPTOR_SHAPE_IDLE=n gives the last n sequences no cached tokens (idle graph slots).
    let long: Option<(usize, usize)> = std::env::var("SCRIPTOR_SHAPE_LONG").ok().and_then(|v| {
        let (a, b) = v.split_once(':')?;
        Some((a.parse().ok()?, b.parse().ok()?))
    });
    let idle: usize = std::env::var("SCRIPTOR_SHAPE_IDLE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let max_batch = *batches.iter().max().unwrap();
    let max_ctx = contexts
        .iter()
        .copied()
        .chain(long.map(|l| l.1))
        .max()
        .unwrap()
        + 8;
    println!(
        "long {:?} idle {idle} SCRIPTOR_ATTN_SPLITS {:?}",
        long,
        std::env::var("SCRIPTOR_ATTN_SPLITS").ok()
    );

    let dir = snapshot();
    let read = |n: &str| std::fs::read_to_string(dir.join(n)).expect(n);
    let config = HfConfig::from_json(&read("config.json")).expect("config");
    let generation = GenerationConfig::from_json(&read("generation_config.json")).ok();
    let spec = ModelSpec::from_hf(&config, generation.as_ref()).expect("spec");
    let files = SafetensorsDir::open(&dir).expect("safetensors");

    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let mut arena = dev.arena(256 << 20, &stream).expect("arena");
    let geometry = TextGeometry::from_spec(&spec.text).expect("geometry");
    let sm_count = dev
        .context()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
        .expect("sm count") as usize;
    let mut exec = TextExecutor::new(
        dev.context(),
        &stream,
        &mut arena,
        geometry,
        sm_count,
        TuningTable::default(),
        GemmOptions {
            fp8: false,
            workspace_bytes: 64 << 20,
        },
    )
    .expect("executor");
    exec.set_fusion(Fusion::Fused);
    let weights = GpuDecoderWeights::load(&stream, &files, &spec.text).expect("weights");

    // SCRIPTOR_SHAPE_SCATTER=n allocates a pool of n blocks per layer and hands every sequence
    // pseudo-random blocks from it (the live pool after churn) instead of contiguous runs.
    let scatter: usize = std::env::var("SCRIPTOR_SHAPE_SCATTER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let blocks_per_seq = max_ctx.div_ceil(BLOCK);
    let pool_blocks = (max_batch * blocks_per_seq).max(scatter);
    println!("pool blocks per layer {pool_blocks} (scatter {scatter})");
    let mut kv =
        GpuKvPool::new(&stream, geometry.layers, pool_blocks, BLOCK, &geometry).expect("pool");
    let tables: Vec<Vec<u32>> = if scatter > 0 {
        let mut perm: Vec<u32> = (0..pool_blocks as u32).collect();
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        for i in (1..perm.len()).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            perm.swap(i, (x % (i as u64 + 1)) as usize);
        }
        (0..max_batch)
            .map(|s| perm[s * blocks_per_seq..(s + 1) * blocks_per_seq].to_vec())
            .collect()
    } else {
        (0..max_batch)
            .map(|s| {
                (0..blocks_per_seq)
                    .map(|b| (s * blocks_per_seq + b) as u32)
                    .collect()
            })
            .collect()
    };
    let ctx = dev.context();
    println!("sm_count {sm_count}; fused layers; {ITERS} timed steps after {WARMUP} warm-ups");
    println!(
        "| seqs | q_len | rows | context | step ms | attention ms | attn launches/layer | attn planned MB | attn GB/s | non-attention ms |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for &batch in &batches {
        for &q_len in &qlens {
            for &context in &contexts {
                let tokens: Vec<Vec<u32>> = (0..batch)
                    .map(|s| {
                        (0..q_len)
                            .map(|j| ((s * 7 + 11 + j) % 4000) as u32)
                            .collect()
                    })
                    .collect();
                let ctx_of = |s: usize| -> usize {
                    if let Some((n, c)) = long
                        && s < n
                    {
                        c
                    } else if s + idle >= batch {
                        0
                    } else {
                        context
                    }
                };
                let positions: Vec<Vec<[i32; 3]>> = (0..batch)
                    .map(|s| (0..q_len).map(|j| [(ctx_of(s) + j) as i32; 3]).collect())
                    .collect();
                let decode: Vec<DecodeSeq> = (0..batch)
                    .map(|s| DecodeSeq {
                        tokens: &tokens[s],
                        positions: &positions[s],
                        block_table: &tables[s],
                        cached: ctx_of(s),
                    })
                    .collect();
                let rows: Vec<u32> = (0..(batch * q_len) as u32).collect();
                let input = StepInput {
                    prefill: &[],
                    decode: &decode,
                    image_rows: &[],
                    sample_rows: &rows,
                    keep_hidden: false,
                };
                for _ in 0..WARMUP {
                    exec.forward(&weights, &mut kv, &input, None).expect("warm");
                }
                stream.synchronize().expect("sync");
                let (a, b) = (
                    ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .expect("event"),
                    ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .expect("event"),
                );
                a.record(&stream).expect("record");
                for _ in 0..ITERS {
                    exec.forward(&weights, &mut kv, &input, None).expect("step");
                }
                b.record(&stream).expect("record");
                let step_ms = f64::from(a.elapsed_ms(&b).expect("elapsed")) / ITERS as f64;
                exec.set_profile(true);
                for _ in 0..ITERS {
                    exec.forward(&weights, &mut kv, &input, None).expect("step");
                }
                let stats = exec.take_profile().expect("profile");
                exec.set_profile(false);
                let attn = stats
                    .iter()
                    .find(|s| s.name == "attention")
                    .expect("attention stage");
                let layer_ms: f64 = stats.iter().map(|s| s.millis / ITERS as f64).sum();
                let attn_ms = attn.millis / ITERS as f64;
                println!(
                    "| {batch} | {q_len} | {} | {context} | {step_ms:.3} | {attn_ms:.3} | {} | {:.1} | {:.0} | {:.3} |",
                    batch * q_len,
                    attn.launches,
                    attn.bytes as f64 / ITERS as f64 / 1e6,
                    attn.gbps(),
                    layer_ms - attn_ms
                );
            }
        }
    }
}
