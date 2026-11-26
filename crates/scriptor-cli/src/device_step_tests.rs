//! GPU tests of the device-resident decode step on a synthetic checkpoint: the device step
//! against the host-sampled speculative step, and graph replay against eager execution.
//! Run with `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-cli --features cuda device_step -- --ignored`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use half::bf16;
use scriptor_cuda::Device;
use scriptor_engine::{DecodeRow, DeviceRow, ModelExecutor, PrefillRow};
use scriptor_kernels::heuristics::TuningTable;
use scriptor_kernels::sampling::{RowParams, TokenMasks, process_row};
use scriptor_model::gpu::{GpuDecoderWeights, GpuMtpWeights, GpuVisionParams};
use scriptor_model::reference::vision::VisionBlock;
use scriptor_model::{Linear, Merger, VisionTower};
use scriptor_spec::{MemoryPlan, ModelSpec};
use scriptor_weights::SafetensorsDir;

mod l1_gpu;
mod real_fixture_dump;
mod scrambled_drafts;
mod sm_split_gate;
mod sm_split_production;
mod split_graph;

use crate::device_step::StepLimits;
use crate::gpu_executor::{GpuEmbedding, GpuExecutor, GpuWeights};

const HIDDEN: usize = 256;
const INTER: usize = 768;
const VOCAB: usize = 512;
const LAYERS: usize = 2;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 128;
const IMAGE_ID: u32 = 510;
const BLOCK: usize = 16;
const BLOCKS_PER_SEQ: usize = 6;
const MAX_SEQS: usize = 8;
const MAX_CONTEXT: usize = BLOCK * BLOCKS_PER_SEQ - 16;
const TOKENS: usize = 20;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn below(&mut self, n: usize) -> usize {
        (((self.next() + 1.0) * 0.5 * n as f32) as usize).min(n - 1)
    }
}

fn bf16_bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| bf16::from_f32(x).to_bits()).collect()
}

fn config_json() -> String {
    format!(
        r#"{{
  "architectures": ["GlmOcrForConditionalGeneration"],
  "image_end_token_id": 509, "image_start_token_id": 508, "image_token_id": {IMAGE_ID},
  "model_type": "glm_ocr",
  "text_config": {{
    "attention_bias": false, "eos_token_id": [500, 501], "head_dim": {HEAD_DIM},
    "hidden_act": "silu", "hidden_size": {HIDDEN}, "intermediate_size": {INTER},
    "max_position_embeddings": 4096, "model_type": "glm_ocr_text",
    "num_attention_heads": {HEADS}, "num_hidden_layers": {LAYERS},
    "num_key_value_heads": {KV_HEADS}, "num_nextn_predict_layers": 1, "pad_token_id": 500,
    "rms_norm_eps": 1e-05,
    "rope_parameters": {{"mrope_section": [16, 24, 24], "partial_rotary_factor": 1.0,
      "rope_theta": 10000, "rope_type": "default"}},
    "tie_word_embeddings": false, "vocab_size": {VOCAB}
  }},
  "tie_word_embeddings": false,
  "vision_config": {{
    "attention_bias": true, "depth": 2, "hidden_act": "silu", "hidden_size": 32,
    "in_channels": 3, "intermediate_size": 64, "model_type": "glm_ocr_vision", "num_heads": 2,
    "out_hidden_size": {HIDDEN}, "patch_size": 14,
    "rms_norm_eps": 1e-05, "rope_parameters": {{"rope_theta": 10000.0, "rope_type": "axial"}},
    "spatial_merge_size": 2, "temporal_patch_size": 2
  }}
}}"#
    )
}

/// Writes a deterministic BF16 safetensors checkpoint of the text decoder and MTP layer.
fn write_checkpoint(dir: &Path, rng: &mut Rng) {
    let mut tensors: Vec<(String, Vec<usize>, Vec<u16>)> = Vec::new();
    let mut matrix = |name: String, out: usize, inp: usize, rng: &mut Rng| {
        let amp = 1.7 / (inp as f32).sqrt();
        let v: Vec<f32> = (0..out * inp).map(|_| rng.next() * amp).collect();
        tensors.push((name, vec![out, inp], bf16_bits(&v)));
    };
    let norm = |name: String, rng: &mut Rng| {
        let v: Vec<f32> = (0..HIDDEN).map(|_| 1.0 + 0.1 * rng.next()).collect();
        (name, vec![HIDDEN], bf16_bits(&v))
    };
    let (qd, kd) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let layer_names = [
        "input_layernorm",
        "post_self_attn_layernorm",
        "post_attention_layernorm",
        "post_mlp_layernorm",
    ];
    let layer = |tensors_prefix: String,
                 rng: &mut Rng,
                 matrix: &mut dyn FnMut(String, usize, usize, &mut Rng)|
     -> Vec<(String, Vec<usize>, Vec<u16>)> {
        let mut norms = Vec::new();
        for n in layer_names {
            norms.push(norm(format!("{tensors_prefix}.{n}.weight"), rng));
        }
        matrix(
            format!("{tensors_prefix}.self_attn.q_proj.weight"),
            qd,
            HIDDEN,
            rng,
        );
        matrix(
            format!("{tensors_prefix}.self_attn.k_proj.weight"),
            kd,
            HIDDEN,
            rng,
        );
        matrix(
            format!("{tensors_prefix}.self_attn.v_proj.weight"),
            kd,
            HIDDEN,
            rng,
        );
        matrix(
            format!("{tensors_prefix}.self_attn.o_proj.weight"),
            HIDDEN,
            qd,
            rng,
        );
        matrix(
            format!("{tensors_prefix}.mlp.gate_up_proj.weight"),
            2 * INTER,
            HIDDEN,
            rng,
        );
        matrix(
            format!("{tensors_prefix}.mlp.down_proj.weight"),
            HIDDEN,
            INTER,
            rng,
        );
        norms
    };
    let mut extra = Vec::new();
    matrix(
        "model.language_model.embed_tokens.weight".into(),
        VOCAB,
        HIDDEN,
        rng,
    );
    matrix("lm_head.weight".into(), VOCAB, HIDDEN, rng);
    extra.push(norm("model.language_model.norm.weight".into(), rng));
    for l in 0..=LAYERS {
        let prefix = format!("model.language_model.layers.{l}");
        extra.extend(layer(prefix.clone(), rng, &mut matrix));
        if l == LAYERS {
            matrix(format!("{prefix}.embed_tokens.weight"), VOCAB, HIDDEN, rng);
            matrix(format!("{prefix}.eh_proj.weight"), HIDDEN, 2 * HIDDEN, rng);
            matrix(
                format!("{prefix}.shared_head.head.weight"),
                VOCAB,
                HIDDEN,
                rng,
            );
            extra.push(norm(format!("{prefix}.enorm.weight"), rng));
            extra.push(norm(format!("{prefix}.hnorm.weight"), rng));
            extra.push(norm(format!("{prefix}.shared_head.norm.weight"), rng));
        }
    }
    tensors.extend(extra);

    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, shape, bits) in &tensors {
        let start = data.len();
        for b in bits {
            data.extend_from_slice(&b.to_le_bytes());
        }
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut text = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
    while !text.len().is_multiple_of(8) {
        text.push(' ');
    }
    let mut file = (text.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(text.as_bytes());
    file.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
}

fn gpu_enabled() -> bool {
    std::env::var("SCRIPTOR_GPU_TESTS").as_deref() == Ok("1")
}

/// What a split gate case does after asking the driver for its split.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CaseOutcome {
    /// The split was granted: run the comparison.
    Run,
    /// The split cannot be formed and no one requires it: skip with this message.
    Skip(String),
    /// The split is required and was refused: fail with this message.
    Fail(String),
}

/// Decides a case from the result of forming `split`. A refused split fails when it is
/// `always_required` or equals `env_required`; otherwise it is skipped. `env_required` only
/// adds a required split, it never relaxes `always_required`.
fn grant_outcome(
    split: (u32, u32),
    always_required: (u32, u32),
    env_required: Option<(u32, u32)>,
    partition: Result<(), String>,
) -> CaseOutcome {
    let (vision, decode) = split;
    match partition {
        Ok(()) => CaseOutcome::Run,
        Err(e) if split == always_required || Some(split) == env_required => CaseOutcome::Fail(
            format!("required split {vision}/{decode} was not granted: {e}"),
        ),
        Err(e) => CaseOutcome::Skip(format!("split {vision}/{decode} cannot be formed: {e}")),
    }
}

/// The split `SCRIPTOR_REQUIRE_SPLIT` names; a malformed value panics before any GPU work.
fn required_split_from_env() -> Option<(u32, u32)> {
    scriptor_cuda::required_split(std::env::var_os(scriptor_cuda::REQUIRE_SPLIT_ENV).as_deref())
        .unwrap_or_else(|e| panic!("{e}"))
}

/// Checks that a required split names one of `covered`; a pair no case matches is an error so
/// that a mistyped split cannot turn a gate into a run that checks nothing.
fn require_covered(
    required: Option<(u32, u32)>,
    covered: &[(u32, u32)],
) -> Result<Option<(u32, u32)>, String> {
    match required {
        Some((v, d)) if !covered.contains(&(v, d)) => Err(format!(
            "{}={v},{d} names no covered case (covered: {covered:?})",
            scriptor_cuda::REQUIRE_SPLIT_ENV
        )),
        other => Ok(other),
    }
}

/// Resolves the result of forming `split` into the case's outcome, printing a skip with the
/// literal prefix `SKIPPED:` and panicking on a refused required split.
fn settle_grant(
    split: (u32, u32),
    always_required: (u32, u32),
    env_required: Option<(u32, u32)>,
    partition: Result<(), String>,
) -> bool {
    match grant_outcome(split, always_required, env_required, partition) {
        CaseOutcome::Run => true,
        CaseOutcome::Skip(message) => {
            println!("SKIPPED: {message}");
            false
        }
        CaseOutcome::Fail(message) => panic!("{message}"),
    }
}

struct Fixture {
    dir: PathBuf,
    spec: ModelSpec,
}

fn fixture(tag: &str) -> Fixture {
    let dir =
        std::env::temp_dir().join(format!("scriptor-device-step-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    write_checkpoint(&dir, &mut rng);
    let spec = ModelSpec::from_json(&config_json(), None).unwrap();
    Fixture { dir, spec }
}

/// A tiny vision tower; the tests never run it, but the executor requires one.
struct Gen(Rng);

impl Gen {
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| bf16::from_f32(self.0.next() * 0.05).to_f32())
            .collect()
    }

    fn linear(&mut self, out_dim: usize, in_dim: usize) -> Linear {
        Linear {
            weight: self.vec(out_dim * in_dim),
            bias: Some(self.vec(out_dim)),
            out_dim,
            in_dim,
        }
    }
}

fn vision(stream: &Arc<cudarc::driver::CudaStream>) -> GpuVisionParams {
    const V_HIDDEN: usize = 128;
    const V_HEADS: usize = 2;
    const V_HEAD_DIM: usize = 64;
    let mut g = Gen(Rng(0x1234_5678_9abc_def1));
    let ones = |n: usize| vec![1.0_f32; n];
    let blocks = (0..1)
        .map(|_| VisionBlock {
            norm1: ones(V_HIDDEN),
            norm2: ones(V_HIDDEN),
            qkv: g.linear(3 * V_HIDDEN, V_HIDDEN),
            q_norm: ones(V_HEAD_DIM),
            k_norm: ones(V_HEAD_DIM),
            proj: g.linear(V_HIDDEN, V_HIDDEN),
            gate: g.linear(192, V_HIDDEN),
            up: g.linear(192, V_HIDDEN),
            down: g.linear(V_HIDDEN, 192),
        })
        .collect();
    let tower = VisionTower {
        patch_embed: g.linear(V_HIDDEN, scriptor_kernels::reference::conv::PATCH_ELEMS),
        blocks,
        post_layernorm: ones(V_HIDDEN),
        hidden: V_HIDDEN,
        heads: V_HEADS,
        head_dim: V_HEAD_DIM,
        eps: 1e-5,
        rope_theta: 10000.0,
    };
    let merger = Merger {
        downsample_weight: g.vec(HIDDEN * V_HIDDEN * 4),
        downsample_bias: g.vec(HIDDEN),
        proj: g.linear(HIDDEN, HIDDEN),
        norm_weight: ones(HIDDEN),
        norm_bias: g.vec(HIDDEN),
        gate: g.linear(160, HIDDEN),
        up: g.linear(160, HIDDEN),
        down: g.linear(HIDDEN, 160),
        in_dim: V_HIDDEN,
        out_dim: HIDDEN,
    };
    GpuVisionParams::from_reference(stream, &tower, &merger).expect("vision weights")
}

fn executor(fx: &Fixture, depth: usize) -> GpuExecutor {
    assert!(gpu_enabled(), "set SCRIPTOR_GPU_TESTS=1 to run GPU tests");
    let device = Device::open().expect("gpu");
    let stream = device.new_stream().expect("stream");
    let files = SafetensorsDir::open(Path::new(&fx.dir)).unwrap();
    let decoder = GpuDecoderWeights::load(&stream, &files, &fx.spec.text).unwrap();
    let mtp = (depth > 0).then(|| GpuMtpWeights::load(&stream, &files, &fx.spec.text).unwrap());
    let weights = GpuWeights::from_parts(decoder, mtp, vision(&stream));
    executor_with(&device, &stream, weights, &fx.spec, depth)
}

/// An executor over `weights` with the test step limits.
fn executor_with(
    device: &Device,
    stream: &Arc<cudarc::driver::CudaStream>,
    weights: GpuWeights,
    spec: &ModelSpec,
    depth: usize,
) -> GpuExecutor {
    executor_sized(
        device,
        stream,
        weights,
        spec,
        depth,
        Sizing {
            seqs: MAX_SEQS,
            max_context: MAX_CONTEXT,
            kv_blocks: MAX_SEQS * BLOCKS_PER_SEQ + 2,
            blocks_per_seq: BLOCKS_PER_SEQ,
        },
    )
}

/// Sequence slots, context bound and KV pool size of a test executor.
#[derive(Clone, Copy)]
struct Sizing {
    seqs: usize,
    max_context: usize,
    kv_blocks: usize,
    blocks_per_seq: usize,
}

/// An executor over `weights` sized by `sizing`.
fn executor_sized(
    device: &Device,
    stream: &Arc<cudarc::driver::CudaStream>,
    weights: GpuWeights,
    spec: &ModelSpec,
    depth: usize,
    sizing: Sizing,
) -> GpuExecutor {
    let plan = MemoryPlan {
        weights: 0,
        kv_cache: 0,
        mtp_kv_cache: 0,
        embedding_cache: 0,
        vision_workspace: 0,
        logits_workspace: 0,
        step_workspace: 0,
        graph_memory: 0,
        total: 0,
        available: 0,
        kv_blocks: sizing.kv_blocks as u64,
        blocks_per_seq: sizing.blocks_per_seq as u64,
        full_context_seqs: sizing.seqs as u64,
        max_context: 0,
        slots: sizing.seqs,
        kv_block_tokens: BLOCK,
        vision_batch_patches: 64,
    };
    GpuExecutor::new(
        device,
        stream,
        weights,
        spec,
        &plan,
        depth,
        TuningTable::default(),
        StepLimits {
            max_seqs: sizing.seqs,
            max_context: sizing.max_context,
        },
    )
    .expect("executor")
}

/// One sequence driven by a test: its prompt, KV blocks, output so far and sampling state.
struct Seq {
    id: u64,
    prompt: Vec<u32>,
    blocks: Vec<u32>,
    out: Vec<u32>,
    masks: TokenMasks,
}

impl Seq {
    /// Sequence `i` over `prompt` in a `vocab`-sized id space, with its own KV blocks.
    fn new(i: usize, prompt: Vec<u32>, vocab: usize) -> Self {
        let mut masks = TokenMasks::new(1, vocab);
        masks.reset_slot(0, &prompt);
        Self {
            id: 100 + i as u64,
            prompt,
            blocks: (0..BLOCKS_PER_SEQ)
                .map(|b| (1 + i * BLOCKS_PER_SEQ + b) as u32)
                .collect(),
            out: Vec::new(),
            masks,
        }
    }

    fn params(&self) -> RowParams {
        RowParams {
            seed: 7,
            repetition: 1.1,
            presence: 0.0,
            frequency: 0.0,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            step: self.out.len() as u32,
            slot: 0,
        }
    }

    fn push(&mut self, token: u32) {
        self.masks.push_output(0, token);
        self.out.push(token);
    }

    fn row(&self) -> DecodeRow {
        DecodeRow {
            seq: self.id,
            token: self
                .out
                .last()
                .or(self.prompt.last())
                .copied()
                .expect("prompt"),
            position: self.prompt.len() + self.out.len() - 1,
            blocks: self.blocks.clone(),
        }
    }
}

fn sequences(count: usize) -> Vec<Seq> {
    seeded_sequences(count, 0xfeed_beef_1234_5678)
}

/// `count` sequences whose prompts are drawn from `seed`.
fn seeded_sequences(count: usize, seed: u64) -> Vec<Seq> {
    let mut rng = Rng(seed);
    (0..count)
        .map(|i| {
            let len = 5 + (i * 3) % 7;
            let prompt: Vec<u32> = (0..len).map(|_| 1 + rng.below(480) as u32).collect();
            Seq::new(i, prompt, VOCAB)
        })
        .collect()
}

/// Prefills every sequence and seeds its first token from the host-sampled prefill logits.
fn start(exec: &mut GpuExecutor, seqs: &mut [Seq]) {
    let rows: Vec<PrefillRow<GpuEmbedding>> = seqs
        .iter()
        .map(|s| PrefillRow {
            seq: s.id,
            tokens: s.prompt.clone(),
            images: Vec::new(),
            blocks: s.blocks.clone(),
            logits: true,
        })
        .collect();
    let logits = exec.prefill(&rows).expect("prefill");
    for (s, l) in seqs.iter_mut().zip(&logits) {
        let token = process_row(l, &s.masks, &s.params()).token;
        s.push(token);
    }
}

/// Prefills every sequence for the device step without host logits; the first token comes
/// from the first device step.
fn defer(exec: &mut GpuExecutor, seqs: &mut [Seq]) {
    let rows: Vec<PrefillRow<GpuEmbedding>> = seqs
        .iter()
        .map(|s| PrefillRow {
            seq: s.id,
            tokens: s.prompt.clone(),
            images: Vec::new(),
            blocks: s.blocks.clone(),
            logits: false,
        })
        .collect();
    let logits = exec.prefill(&rows).expect("prefill");
    assert!(logits.len() == seqs.len() && logits.iter().all(Vec::is_empty));
}

/// One host-sampled step: the speculative executor path with the scheduler's sequential
/// verification of each row.
fn host_step(exec: &mut GpuExecutor, seqs: &mut [Seq]) {
    let rows: Vec<DecodeRow> = seqs.iter().map(Seq::row).collect();
    if exec.draft_depth() == 0 {
        let logits = exec.decode(&rows).expect("decode");
        for (s, l) in seqs.iter_mut().zip(&logits) {
            let token = process_row(l, &s.masks, &s.params()).token;
            s.push(token);
        }
        return;
    }
    let outs = exec.decode_spec(&rows).expect("decode_spec");
    for (s, out) in seqs.iter_mut().zip(&outs) {
        for (j, l) in out.logits.iter().enumerate() {
            let token = process_row(l, &s.masks, &s.params()).token;
            s.push(token);
            if out.drafts.get(j) != Some(&token) {
                break;
            }
        }
    }
}

/// One device step: the executor drafts, verifies, accepts and samples itself.
fn device_step(exec: &mut GpuExecutor, seqs: &mut [Seq]) {
    let rows: Vec<DecodeRow> = seqs.iter().map(Seq::row).collect();
    let device_rows: Vec<DeviceRow<'_>> = seqs
        .iter()
        .zip(&rows)
        .map(|(s, row)| DeviceRow {
            row,
            params: s.params(),
            masks: &s.masks,
        })
        .collect();
    let results = exec.decode_device(&device_rows).expect("decode_device");
    drop(device_rows);
    for (s, r) in seqs.iter_mut().zip(&results) {
        assert!(!r.tokens.is_empty() && r.tokens.len() == r.margins.len());
        for &t in &r.tokens {
            s.push(t);
        }
    }
}

fn generate(
    exec: &mut GpuExecutor,
    mut seqs: Vec<Seq>,
    step: fn(&mut GpuExecutor, &mut [Seq]),
) -> Vec<Vec<u32>> {
    start(exec, &mut seqs);
    while seqs.iter().any(|s| s.out.len() < TOKENS) {
        step(exec, &mut seqs);
    }
    seqs.into_iter().map(|s| s.out[..TOKENS].to_vec()).collect()
}

fn check_zero_draft_admission(depth: usize) {
    let fx = fixture(&format!("defer-k{depth}"));
    let mut host = executor(&fx, depth);
    let want = generate(&mut host, sequences(3), host_step);
    drop(host);
    let mut device = executor(&fx, depth);
    let mut seqs = sequences(3);
    defer(&mut device, &mut seqs);
    device_step(&mut device, &mut seqs);
    assert!(
        seqs.iter().all(|s| s.out.len() == 1),
        "the first step verifies the last prompt token alone"
    );
    while seqs.iter().any(|s| s.out.len() < TOKENS) {
        device_step(&mut device, &mut seqs);
    }
    let got: Vec<Vec<u32>> = seqs.into_iter().map(|s| s.out[..TOKENS].to_vec()).collect();
    assert_eq!(got, want, "zero-draft admission diverged at k={depth}");
}

#[test]
#[ignore = "needs GPU"]
fn zero_draft_admission_matches_host_step_without_drafts() {
    check_zero_draft_admission(0);
}

#[test]
#[ignore = "needs GPU"]
fn zero_draft_admission_matches_host_step_with_three_drafts() {
    check_zero_draft_admission(3);
}

fn check_device_matches_host(depth: usize) {
    let fx = fixture(&format!("host-k{depth}"));
    let mut host = executor(&fx, depth);
    let want = generate(&mut host, sequences(3), host_step);
    drop(host);
    let mut device = executor(&fx, depth);
    let got = generate(&mut device, sequences(3), device_step);
    assert_eq!(
        got, want,
        "device step diverged from the host step at k={depth}"
    );
}

#[test]
#[ignore = "needs GPU"]
fn device_step_matches_host_step_without_drafts() {
    check_device_matches_host(0);
}

#[test]
#[ignore = "needs GPU"]
fn device_step_matches_host_step_with_three_drafts() {
    check_device_matches_host(3);
}

/// A request whose prompt is the bytes of public-domain English `text` and whose KV blocks are
/// `first..first + BLOCKS_PER_SEQ`.
fn english_seq(id: u64, text: &str, first: usize) -> Seq {
    let prompt: Vec<u32> = text.bytes().map(|b| b as u32 + 1).collect();
    let mut masks = TokenMasks::new(1, VOCAB);
    masks.reset_slot(0, &prompt);
    Seq {
        id,
        prompt,
        blocks: (first..first + BLOCKS_PER_SEQ).map(|b| b as u32).collect(),
        out: Vec::new(),
        masks,
    }
}

fn check_slot_reuse_with_new_block_mapping(depth: usize) {
    let fx = fixture(&format!("reuse-k{depth}"));
    let first = "It was the best of times, it was the worst of times";
    let second = "Call me Ishmael. Some years ago, never mind how long";
    let mut fresh = executor(&fx, depth);
    let want = generate(&mut fresh, vec![english_seq(2, second, 0)], device_step);
    drop(fresh);
    let mut reused = executor(&fx, depth);
    generate(&mut reused, vec![english_seq(1, first, 1)], device_step);
    reused.release(1);
    let got = generate(&mut reused, vec![english_seq(2, second, 0)], device_step);
    assert_eq!(
        got, want,
        "a reused slot read the previous request's blocks at k={depth}"
    );
}

#[test]
#[ignore = "needs GPU"]
fn reused_slot_with_new_block_mapping_matches_fresh_slot_without_drafts() {
    check_slot_reuse_with_new_block_mapping(0);
}

#[test]
#[ignore = "needs GPU"]
fn reused_slot_with_new_block_mapping_matches_fresh_slot_with_three_drafts() {
    check_slot_reuse_with_new_block_mapping(3);
}

fn check_replay_matches_eager(depth: usize) {
    let fx = fixture(&format!("graph-k{depth}"));
    for count in [1usize, 3, 5] {
        let mut eager = executor(&fx, depth);
        let want = generate(&mut eager, sequences(count), device_step);
        let eager_stats = eager.graph_stats();
        assert_eq!(eager_stats.replays, 0);
        assert!(eager_stats.fallbacks > 0);
        drop(eager);

        let mut graphed = executor(&fx, depth);
        let report = graphed.capture_graphs().expect("capture");
        assert!(report.graphs >= 4, "buckets 1, 2, 4 and 8 are captured");
        let got = generate(&mut graphed, sequences(count), device_step);
        let stats = graphed.graph_stats();
        assert!(stats.replays > 0, "no step replayed a graph");
        assert_eq!(stats.fallbacks, 0, "a covered shape ran eagerly");
        assert_eq!(
            got, want,
            "replay diverged from eager at batch {count}, k={depth}"
        );
    }
}

#[test]
#[ignore = "needs GPU"]
fn graph_replay_matches_eager_without_drafts() {
    check_replay_matches_eager(0);
}

#[test]
#[ignore = "needs GPU"]
fn graph_replay_matches_eager_with_three_drafts() {
    check_replay_matches_eager(3);
}

#[test]
fn grant_outcome_refused_required_split_fails() {
    let got = grant_outcome((36, 12), (40, 8), Some((36, 12)), Err("refused".into()));
    let CaseOutcome::Fail(text) = got else {
        panic!("{got:?}");
    };
    assert!(text.contains("refused") && text.contains("36/12"), "{text}");
}

#[test]
fn grant_outcome_refused_default_split_fails_without_env() {
    let got = grant_outcome((40, 8), (40, 8), None, Err("refused".into()));
    assert!(matches!(got, CaseOutcome::Fail(_)), "{got:?}");
}

#[test]
fn grant_outcome_refused_default_split_still_fails_with_env() {
    let got = grant_outcome((40, 8), (40, 8), Some((36, 12)), Err("refused".into()));
    assert!(matches!(got, CaseOutcome::Fail(_)), "{got:?}");
}

#[test]
fn grant_outcome_refused_other_split_skips() {
    let got = grant_outcome((36, 12), (40, 8), None, Err("refused".into()));
    let CaseOutcome::Skip(text) = got else {
        panic!("{got:?}");
    };
    assert!(text.starts_with("split 36/12"), "{text}");
}

#[test]
fn grant_outcome_granted_runs() {
    assert_eq!(
        grant_outcome((36, 12), (40, 8), Some((36, 12)), Ok(())),
        CaseOutcome::Run
    );
    assert_eq!(
        grant_outcome((40, 8), (40, 8), None, Ok(())),
        CaseOutcome::Run
    );
}
