//! Scratch measurement: per-candidate time of the decode QKV and language-model-head GEMMs at
//! the decode row counts. Run with
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda --release --test gpu_head_gemm_shapes -- --ignored --nocapture`.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use cudarc::driver::DevicePtr;
use scriptor_cuda::Device;
use scriptor_kernels::cublaslt::{
    GemmEngine, GemmOperands, GemmOptions, GemmProblem, SyntheticOperands,
};
use scriptor_kernels::heuristics::{Epilogue, GemmDtype, TuningTable, median};
use scriptor_kernels::sampling::{RowParams, SamplingKernels, TokenMasks};
use scriptor_model::gpu::{
    DecodeSeq, GpuDecoderWeights, GpuKvPool, StepInput, TextExecutor, TextGeometry,
};
use scriptor_spec::{GenerationConfig, HfConfig, ModelSpec};
use scriptor_weights::SafetensorsDir;

#[test]
#[ignore = "needs GPU"]
fn head_gemm_candidates() {
    assert_eq!(std::env::var("SCRIPTOR_GPU_TESTS").as_deref(), Ok("1"));
    let dev = Device::open().expect("gpu");
    let stream = dev.new_stream().expect("stream");
    let mut arena = dev.arena(256 << 20, &stream).expect("arena");
    let mut gemm = GemmEngine::new(
        dev.context(),
        &stream,
        &mut arena,
        TuningTable::default(),
        GemmOptions {
            fp8: false,
            workspace_bytes: 64 << 20,
        },
    )
    .expect("engine");
    let mut flush = stream.alloc_zeros::<u8>(256 << 20).expect("flush");
    println!("| shape | m | candidate | us | weight GB/s |");
    println!("|---|---|---|---|---|");
    for (name, n, k) in [
        ("qkv", 4096usize, 1536usize),
        ("o", 1536, 2048),
        ("gate_up", 9216, 1536),
        ("down", 1536, 4608),
        ("lm_head", 59392, 1536),
    ] {
        for m in [1usize, 2, 32, 128] {
            let p = GemmProblem {
                m,
                n,
                k,
                dtype: GemmDtype::Bf16,
                epilogue: Epilogue::None,
            };
            let ops = SyntheticOperands::new(&stream, &p, &gemm).expect("operands");
            let o = ops.operands();
            let count = gemm.candidate_count(&p).expect("candidates");
            for idx in 0..count.min(8) {
                let mut t = Vec::new();
                for i in 0..40 {
                    stream.memset_zeros(&mut flush).expect("flush");
                    let a = stream
                        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .expect("event");
                    gemm.gemm_with(&p, &o, idx).expect("gemm");
                    let b = stream
                        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .expect("event");
                    b.synchronize().expect("sync");
                    if i >= 5 {
                        t.push(f64::from(a.elapsed_ms(&b).expect("elapsed")) * 1e3);
                    }
                }
                let us = median(&t).unwrap();
                let gbps = 2.0 * (n * k) as f64 / us / 1e3;
                println!(
                    "| {name} | {m} | {idx}{} | {us:.1} | {gbps:.0} |",
                    if idx == 0 { " (default)" } else { "" }
                );
            }
        }
    }
}

const BATCH: usize = 128;
const CONTEXT: usize = 300;
const BLOCK: usize = 16;

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
    std::fs::read_dir(&snaps)
        .expect("snapshot cache")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.join("config.json").exists() && p.join("generation_config.json").exists())
        .expect("a snapshot")
}

/// Wide fused draft head on the real checkpoint: parity of its tokens with the logits-plus-
/// sampler path on real head weights and real decoder hidden rows at 65, 96 and 128 rows, then
/// the head's share of a real B=128 decode step with and without the wide kernel. Set
/// `SCRIPTOR_SNAPSHOT_DIR` to the checkpoint directory (or leave it unset to use the Hugging
/// Face cache) and run `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda
/// --profile gpu-test --test gpu_head_gemm_shapes -- --ignored --nocapture wide_draft_head`.
#[test]
#[ignore = "needs GPU and the checkpoint"]
fn wide_draft_head_real_parity_and_step_impact() {
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
    let blocks_per_seq = (CONTEXT + 1).div_ceil(BLOCK);
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
    let positions = [[CONTEXT as i32; 3]];
    let decode: Vec<DecodeSeq> = (0..BATCH)
        .map(|s| DecodeSeq {
            tokens: &tokens[s],
            positions: &positions,
            block_table: &tables[s],
            cached: CONTEXT,
        })
        .collect();
    let rows: Vec<u32> = (0..BATCH as u32).collect();
    let input = StepInput {
        prefill: &[],
        decode: &decode,
        image_rows: &[],
        sample_rows: &rows,
        keep_hidden: true,
    };
    let out = exec.forward(&weights, &mut kv, &input, None).expect("step");
    stream.synchronize().expect("sync");
    let hidden = out.hidden.expect("hidden rows");
    let (n, k) = (geometry.vocab, geometry.hidden);
    let head = weights.lm_head.clone();

    let mut gemm = GemmEngine::new(
        dev.context(),
        &stream,
        &mut arena,
        TuningTable::default(),
        GemmOptions {
            fp8: false,
            workspace_bytes: 64 << 20,
        },
    )
    .expect("engine");
    let sampler = SamplingKernels::load(dev.context()).expect("sampler");
    let (xp, wp) = (hidden.device_ptr(&stream).0, head.device_ptr(&stream).0);
    let mut flush = stream.alloc_zeros::<u8>(256 << 20).expect("flush");
    let mut head_us = [[0.0f64; 2]; 3];
    for (row, m) in [65usize, 96, 128].into_iter().enumerate() {
        let logits = stream.alloc_zeros::<u16>(m * n).expect("logits");
        let masks = TokenMasks::new(m, n);
        let params: Vec<RowParams> = (0..m as u32).map(RowParams::greedy).collect();
        let d_params = stream.clone_htod(&params).expect("params");
        let prompt = stream.clone_htod(masks.prompt_words()).expect("p");
        let output = stream.clone_htod(masks.output_words()).expect("o");
        let counts = stream.clone_htod(masks.count_words()).expect("c");
        let mut want = stream.alloc_zeros::<u32>(m).expect("tokens");
        let mut margins = stream.alloc_zeros::<f32>(m).expect("margins");
        let problem = GemmProblem {
            m,
            n,
            k,
            dtype: GemmDtype::Bf16,
            epilogue: Epilogue::None,
        };
        let ops = GemmOperands {
            x: xp,
            w: wp,
            out: logits.device_ptr(&stream).0,
            ..Default::default()
        };
        let mut logits_path = |gemm: &mut GemmEngine, want: &mut cudarc::driver::CudaSlice<u32>| {
            gemm.gemm(&problem, &ops).expect("gemm");
            sampler
                .choose_bf16(
                    &stream,
                    true,
                    &logits,
                    m,
                    n,
                    &d_params,
                    &prompt,
                    &output,
                    &counts,
                    masks.words(),
                    want,
                    &mut margins,
                )
                .expect("choose");
        };
        logits_path(&mut gemm, &mut want);
        stream.synchronize().expect("sync");
        let want_host = stream.clone_dtoh(&want).expect("dtoh");

        assert!(
            !gemm.supports_argmax(m, n, k),
            "wide head is off by default"
        );
        gemm.set_wide_argmax_enabled(true);
        assert!(gemm.supports_argmax(m, n, k));
        let got = stream.alloc_zeros::<u32>(m).expect("tokens");
        gemm.argmax_rows(m, n, k, xp, wp, got.device_ptr(&stream).0)
            .expect("wide");
        stream.synchronize().expect("sync");
        assert_eq!(stream.clone_dtoh(&got).expect("dtoh"), want_host, "m={m}");

        let mut time = |f: &mut dyn FnMut()| -> f64 {
            let mut t = Vec::new();
            for i in 0..45 {
                stream.memset_zeros(&mut flush).expect("flush");
                stream.synchronize().expect("sync");
                let start = std::time::Instant::now();
                f();
                stream.synchronize().expect("sync");
                if i >= 5 {
                    t.push(start.elapsed().as_secs_f64() * 1e6);
                }
            }
            median(&t).expect("samples")
        };
        let old_us = time(&mut || logits_path(&mut gemm, &mut want));
        let wide_us = time(&mut || {
            gemm.argmax_rows(m, n, k, xp, wp, got.device_ptr(&stream).0)
                .expect("wide");
        });
        gemm.set_wide_argmax_enabled(false);
        head_us[row] = [old_us, wide_us];
        println!(
            "real head m={m}: tokens identical; logits+sampler {old_us:.1} us, wide fused {wide_us:.1} us"
        );
    }

    exec.forward(&weights, &mut kv, &input, None).expect("warm");
    stream.synchronize().expect("sync");
    let start = std::time::Instant::now();
    for _ in 0..20 {
        exec.forward(&weights, &mut kv, &input, None).expect("step");
    }
    stream.synchronize().expect("sync");
    let step_us = start.elapsed().as_secs_f64() * 1e6 / 20.0;
    let [old_us, wide_us] = head_us[2];
    for passes in [1.0f64, 2.0, 3.0] {
        let saved = passes * (old_us - wide_us);
        println!(
            "B=128 decode step {step_us:.0} us; {passes} draft pass(es) at 128 rows: head+selection saves {saved:.0} us ({:.2}% of the step)",
            100.0 * saved / step_us
        );
    }
}
