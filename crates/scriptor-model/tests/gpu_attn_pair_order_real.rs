//! Decode attention staging removal and hoisted pair order against the real checkpoint:
//! bitwise-equal logits across the staged, direct and hoisted routes, a mixed prefill/decode
//! regression, and an old/new cost report. Shapes come from `SCRIPTOR_SHAPE_BATCH`,
//! `SCRIPTOR_SHAPE_QLEN` and `SCRIPTOR_SHAPE_CONTEXT` (comma lists). Run with
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda --release --test gpu_attn_pair_order_real -- --ignored --nocapture --test-threads 1`.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use scriptor_cuda::Device;
use scriptor_kernels::attention::PairOrderMode;
use scriptor_kernels::cublaslt::GemmOptions;
use scriptor_kernels::heuristics::TuningTable;
use scriptor_model::gpu::{
    DecodeSeq, Fusion, GpuDecoderWeights, GpuKvPool, PrefillSeq, StepInput, TextExecutor,
    TextGeometry,
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

struct Rig {
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    exec: TextExecutor,
    weights: GpuDecoderWeights,
    kv: GpuKvPool,
    tables: Vec<Vec<u32>>,
    // Keeps the device alive for the stream and the executor.
    _dev: Device,
}

fn rig(max_batch: usize, max_ctx: usize) -> Rig {
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
    exec.set_fusion(Fusion::Fused);
    let weights = GpuDecoderWeights::load(&stream, &files, &spec.text).expect("weights");
    let per_seq = max_ctx.div_ceil(BLOCK) + 2;
    let kv = GpuKvPool::new(
        &stream,
        geometry.layers,
        max_batch * per_seq,
        BLOCK,
        &geometry,
    )
    .expect("pool");
    let tables = (0..max_batch)
        .map(|s| (0..per_seq).map(|b| (s * per_seq + b) as u32).collect())
        .collect();
    Rig {
        ctx: dev.context().clone(),
        stream,
        exec,
        weights,
        kv,
        tables,
        _dev: dev,
    }
}

/// Cached length of slot `s`: varied contexts with every third slot idle, like a padded bucket.
fn cached(context: usize, s: usize) -> usize {
    if s % 3 == 2 {
        0
    } else {
        context / 2 + (s * 37) % context.max(2)
    }
}

struct Decode {
    tokens: Vec<Vec<u32>>,
    positions: Vec<Vec<[i32; 3]>>,
    cached: Vec<usize>,
}

fn decode_input(batch: usize, q_len: usize, context: usize, first: usize) -> Decode {
    let cached: Vec<usize> = (0..batch).map(|s| cached(context, s + first)).collect();
    Decode {
        tokens: (0..batch)
            .map(|s| {
                (0..q_len)
                    .map(|j| ((s * 7 + 11 + j) % 4000) as u32)
                    .collect()
            })
            .collect(),
        positions: (0..batch)
            .map(|s| (0..q_len).map(|j| [(cached[s] + j) as i32; 3]).collect())
            .collect(),
        cached,
    }
}

fn logits(rig: &mut Rig, input: &StepInput<'_>) -> Vec<f32> {
    let out = rig
        .exec
        .forward(&rig.weights, &mut rig.kv, input, None)
        .expect("forward");
    rig.stream.synchronize().expect("sync");
    rig.stream.clone_dtoh(&out.logits).expect("logits")
}

fn set_route(rig: &mut Rig, direct: bool, mode: PairOrderMode) {
    rig.exec.set_direct_decode(direct);
    rig.exec.set_pair_order_mode(mode).expect("mode");
}

const ROUTES: [(&str, bool, PairOrderMode); 3] = [
    ("staged", false, PairOrderMode::Off),
    ("direct", true, PairOrderMode::Off),
    (
        "direct+hoisted",
        true,
        PairOrderMode::Hoisted { min_pairs: 1 },
    ),
];

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn pure_decode_routes_are_bitwise_equal() {
    let batches = list("SCRIPTOR_SHAPE_BATCH", &[1, 8, 16, 32, 64, 128]);
    let qlens = list("SCRIPTOR_SHAPE_QLEN", &[1, 4]);
    let context = list("SCRIPTOR_SHAPE_CONTEXT", &[700])[0];
    let max_batch = *batches.iter().max().unwrap();
    let mut rig = rig(max_batch, context * 2 + 16);
    let tables = rig.tables.clone();
    for &batch in &batches {
        for &q_len in &qlens {
            let d = decode_input(batch, q_len, context, 0);
            let seqs: Vec<DecodeSeq> = (0..batch)
                .map(|s| DecodeSeq {
                    tokens: &d.tokens[s],
                    positions: &d.positions[s],
                    block_table: &tables[s],
                    cached: d.cached[s],
                })
                .collect();
            let rows: Vec<u32> = (0..(batch * q_len) as u32).collect();
            let input = StepInput {
                prefill: &[],
                decode: &seqs,
                image_rows: &[],
                sample_rows: &rows,
                keep_hidden: false,
            };
            let mut reference: Option<Vec<f32>> = None;
            for (name, direct, mode) in ROUTES {
                set_route(&mut rig, direct, mode);
                let got = logits(&mut rig, &input);
                match &reference {
                    None => reference = Some(got),
                    Some(want) => assert!(
                        want.iter()
                            .zip(&got)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{name} differs from staged at batch {batch} q_len {q_len}"
                    ),
                }
            }
        }
    }
}

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn mixed_prefill_decode_is_unchanged_by_the_selector() {
    let context = 300;
    let mut rig = rig(8, context * 2 + 64);
    let tables = rig.tables.clone();
    let prompt: Vec<u32> = (0..48).map(|i| (i * 13 + 5) % 4000).collect();
    let prompt_pos: Vec<[i32; 3]> = (0..prompt.len()).map(|i| [i as i32; 3]).collect();
    let prefill = [PrefillSeq {
        tokens: &prompt,
        positions: &prompt_pos,
        block_table: &tables[0],
    }];
    let d = decode_input(6, 1, context, 1);
    let seqs: Vec<DecodeSeq> = (0..6)
        .map(|s| DecodeSeq {
            tokens: &d.tokens[s],
            positions: &d.positions[s],
            block_table: &tables[s + 1],
            cached: d.cached[s],
        })
        .collect();
    let rows: Vec<u32> = (0..(prompt.len() + 6) as u32).collect();
    let input = StepInput {
        prefill: &prefill,
        decode: &seqs,
        image_rows: &[],
        sample_rows: &rows,
        keep_hidden: false,
    };
    let mut reference: Option<Vec<f32>> = None;
    for (name, direct, mode) in ROUTES {
        set_route(&mut rig, direct, mode);
        let got = logits(&mut rig, &input);
        match &reference {
            None => reference = Some(got),
            Some(want) => assert!(
                want.iter()
                    .zip(&got)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name} changed a mixed batch"
            ),
        }
    }
}

/// Old/new cost of attention and of the whole eager decode step (the hoisted order's launch is
/// inside the step time). Rows are labelled with the active streams, the context, `q_len`, the
/// eager launch mode and the padded bucket (the batch itself: idle slots are every third).
#[test]
#[ignore = "needs GPU and the checkpoint"]
fn attention_and_step_cost_by_route() {
    let batches = list("SCRIPTOR_SHAPE_BATCH", &[1, 8, 16, 32, 64, 128]);
    let qlens = list("SCRIPTOR_SHAPE_QLEN", &[1, 4]);
    let contexts = list("SCRIPTOR_SHAPE_CONTEXT", &[350, 1000, 4000]);
    let max_batch = *batches.iter().max().unwrap();
    let max_ctx = *contexts.iter().max().unwrap() * 2 + 16;
    let mut rig = rig(max_batch, max_ctx);
    let tables = rig.tables.clone();
    println!(
        "| active | bucket | context | q_len | mode | route | step ms | attention ms | attn launches/layer |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for &batch in &batches {
        let active = (0..batch).filter(|s| s % 3 != 2).count();
        for &q_len in &qlens {
            for &context in &contexts {
                let d = decode_input(batch, q_len, context, 0);
                let seqs: Vec<DecodeSeq> = (0..batch)
                    .map(|s| DecodeSeq {
                        tokens: &d.tokens[s],
                        positions: &d.positions[s],
                        block_table: &tables[s],
                        cached: d.cached[s],
                    })
                    .collect();
                let rows: Vec<u32> = (0..(batch * q_len) as u32).collect();
                let input = StepInput {
                    prefill: &[],
                    decode: &seqs,
                    image_rows: &[],
                    sample_rows: &rows,
                    keep_hidden: false,
                };
                for (name, direct, mode) in ROUTES {
                    set_route(&mut rig, direct, mode);
                    for _ in 0..WARMUP {
                        rig.exec
                            .forward(&rig.weights, &mut rig.kv, &input, None)
                            .expect("warm");
                    }
                    rig.stream.synchronize().expect("sync");
                    let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
                    let (a, b) = (
                        rig.ctx.new_event(flags).expect("event"),
                        rig.ctx.new_event(flags).expect("event"),
                    );
                    a.record(&rig.stream).expect("record");
                    for _ in 0..ITERS {
                        rig.exec
                            .forward(&rig.weights, &mut rig.kv, &input, None)
                            .expect("step");
                    }
                    b.record(&rig.stream).expect("record");
                    let step_ms = f64::from(a.elapsed_ms(&b).expect("elapsed")) / ITERS as f64;
                    rig.exec.set_profile(true);
                    for _ in 0..ITERS {
                        rig.exec
                            .forward(&rig.weights, &mut rig.kv, &input, None)
                            .expect("step");
                    }
                    let stats = rig.exec.take_profile().expect("profile");
                    rig.exec.set_profile(false);
                    let attn = stats
                        .iter()
                        .find(|s| s.name == "attention")
                        .expect("attention stage");
                    println!(
                        "| {active} | {batch} | {context} | {q_len} | eager | {name} | {step_ms:.3} | {:.3} | {} |",
                        attn.millis / ITERS as f64,
                        attn.launches
                    );
                }
            }
        }
    }
}
