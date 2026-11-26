//! Per-stage decode-step timing on the real checkpoint at B = 128, fused against unfused
//! layers. Set `SCRIPTOR_SNAPSHOT_DIR` to the checkpoint directory (or leave it unset to use the
//! Hugging Face cache under `$HF_HOME` or `~/.cache/huggingface`) and run
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda --release --test gpu_decode_profile_real -- --ignored --nocapture`.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use scriptor_cuda::Device;
use scriptor_kernels::cublaslt::GemmOptions;
use scriptor_kernels::heuristics::TuningTable;
use scriptor_model::gpu::{
    DecodeSeq, Fusion, GpuDecoderWeights, GpuKvPool, StageStat, StepInput, TextExecutor,
    TextGeometry,
};
use scriptor_spec::{GenerationConfig, HfConfig, ModelSpec};
use scriptor_weights::SafetensorsDir;

const BATCH: usize = 128;
/// Cached tokens per sequence; `SCRIPTOR_PROFILE_CONTEXT` overrides the default of 300.
fn context() -> usize {
    std::env::var("SCRIPTOR_PROFILE_CONTEXT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300)
}
const BLOCK: usize = 16;
const WEIGHT_FLOOR_MS: f64 = 4.8;
const WARMUP: usize = 3;
const ITERS: usize = 20;

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

fn print_table(label: &str, step_ms: f64, stats: &[StageStat]) {
    println!(
        "\n{label}: decode step {step_ms:.3} ms at B={BATCH}, context {} ({:.2}x the {WEIGHT_FLOOR_MS} ms weight floor)",
        context(),
        step_ms / WEIGHT_FLOOR_MS
    );
    println!("| stage | launches/layer | ms/step | planned MB/step | GB/s |");
    println!("|---|---|---|---|---|");
    let mut layer_ms = 0.0;
    for s in stats {
        let ms = s.millis / ITERS as f64;
        layer_ms += ms;
        println!(
            "| {} | {} | {:.3} | {:.1} | {:.0} |",
            s.name,
            s.launches,
            ms,
            s.bytes as f64 / ITERS as f64 / 1e6,
            s.gbps()
        );
    }
    println!("| layers total | | {layer_ms:.3} | | |");
}

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn decode_step_stage_profile_fused_vs_unfused() {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
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
    let weights = GpuDecoderWeights::load(&stream, &files, &spec.text).expect("weights");

    let blocks_per_seq = (context() + 1).div_ceil(BLOCK);
    let mut kv = GpuKvPool::new(
        &stream,
        geometry.layers,
        BATCH * blocks_per_seq,
        BLOCK,
        &geometry,
    )
    .expect("pool");
    let tables: Vec<Vec<u32>> = (0..BATCH)
        .map(|s| {
            (0..blocks_per_seq)
                .map(|b| (s * blocks_per_seq + b) as u32)
                .collect()
        })
        .collect();
    let tokens: Vec<[u32; 1]> = (0..BATCH).map(|s| [(s * 7 + 11) as u32 % 4000]).collect();
    let positions = [[[context() as i32; 3]]; 1][0];
    let decode: Vec<DecodeSeq> = (0..BATCH)
        .map(|s| DecodeSeq {
            tokens: &tokens[s],
            positions: &positions,
            block_table: &tables[s],
            cached: context(),
        })
        .collect();
    let rows: Vec<u32> = (0..BATCH as u32).collect();
    let input = StepInput {
        prefill: &[],
        decode: &decode,
        image_rows: &[],
        sample_rows: &rows,
        keep_hidden: false,
    };

    let ctx = dev.context();
    for fusion in [Fusion::Unfused, Fusion::Fused] {
        exec.set_fusion(fusion);
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
        print_table(&format!("{fusion:?}"), step_ms, &stats);
    }
}
