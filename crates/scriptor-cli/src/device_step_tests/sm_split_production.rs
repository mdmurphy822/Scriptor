//! GPU gate for the vision/decode SM split at production shape: the real GLM-OCR checkpoint
//! (hidden 1536) decoding 32 sequences with 3 and 4 MTP drafts, so verification batches reach
//! 128 and 160 rows and the skinny and cuBLASLt GEMM routes. Decode graphs replayed from a
//! decode green-context partition, with vision-shaped GEMM work running on the vision
//! partition, must produce the same tokens, margins and hidden states bit for bit as the
//! unpartitioned run (`--sm-split off`).
//!
//! The cases are the product of [`DEPTHS`] and [`SPLITS`], one test each. The default split
//! 40/8 must be grantable; 36/12 is skipped unless `SCRIPTOR_REQUIRE_SPLIT=36,12` names it,
//! in which case a refused grant fails the test.
//!
//! Run with
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-cli --features cuda --profile gpu-test sm_split_production -- --ignored --nocapture --test-threads=1`.
//! `SCRIPTOR_SNAPSHOT_DIR` names the checkpoint directory; unset uses the Hugging Face cache.

use scriptor_chat::Tokenizer;
use scriptor_cuda::{Device, SmSplit};
use scriptor_engine::{ModelExecutor, PrefillRow};
use scriptor_kernels::heuristics::TuningTable;
use scriptor_kernels::sampling::{TokenMasks, process_row};
use scriptor_spec::{GenerationConfig, HfConfig, MemoryPlan, ModelSpec};
use scriptor_weights::SafetensorsDir;

use super::real_fixture_dump::TEXTS;
use super::sm_split_gate::VisionLoad;
use super::split_graph::{Mode, step_subset};
use super::{Seq, gpu_enabled, require_covered, required_split_from_env, settle_grant};
use crate::device_step::{StateBits, StepLimits};
use crate::gpu_executor::{GpuEmbedding, GpuExecutor, GpuWeights};
use crate::image_feed_tests::snapshot;

/// Draft depths covered: 128 and 160 verification rows at 32 sequences.
const DEPTHS: [usize; 2] = [3, 4];
/// The splits covered, as `(vision, decode)` SMs; the first is the engine default.
const SPLITS: [(u32, u32); 2] = [(40, 8), (36, 12)];
/// The split the device must grant.
const REQUIRED_SPLIT: (u32, u32) = (40, 8);
/// Concurrent sequences, the production decode width.
const SEQS: usize = 32;
/// Tokens in a KV block.
const BLOCK: usize = 16;
/// KV blocks one sequence owns.
const BLOCKS_PER_SEQ: usize = 8;
/// Longest prompt kept, in tokens.
const PROMPT_TOKENS: usize = 40;
/// Tokens each sequence generates.
const GENERATED: usize = 24;
/// Synthetic vision batches enqueued before decoding starts.
const VISION_BATCHES: usize = 4;

/// The `(draft depth, split)` of case `index`.
fn case(index: usize) -> (usize, (u32, u32)) {
    (DEPTHS[index / SPLITS.len()], SPLITS[index % SPLITS.len()])
}

/// Every case index; the per-case tests below are a disjoint cover of it.
const CASES: usize = DEPTHS.len() * SPLITS.len();

/// Sequence `i`: a prompt of two public-domain passages, with its own KV blocks.
fn english_prompt(tokenizer: &Tokenizer, i: usize) -> Vec<u32> {
    let first = TEXTS[i % TEXTS.len()];
    let second = TEXTS[(i % TEXTS.len() + 1 + i / TEXTS.len()) % TEXTS.len()];
    let mut prompt = tokenizer
        .encode(&format!("{first} {second}"))
        .expect("encode");
    prompt.truncate(PROMPT_TOKENS);
    prompt
}

struct Checkpoint {
    spec: ModelSpec,
    files: SafetensorsDir,
    tokenizer: Tokenizer,
}

fn checkpoint() -> Checkpoint {
    let dir = snapshot();
    let read = |n: &str| std::fs::read_to_string(dir.join(n)).expect(n);
    let config = HfConfig::from_json(&read("config.json")).expect("config");
    let generation = GenerationConfig::from_json(&read("generation_config.json")).ok();
    Checkpoint {
        spec: ModelSpec::from_hf(&config, generation.as_ref()).expect("spec"),
        files: SafetensorsDir::open(&dir).expect("safetensors"),
        tokenizer: Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer"),
    }
}

/// A real executor with `SEQS` slots and draft depth `depth`.
fn executor(ck: &Checkpoint, depth: usize) -> GpuExecutor {
    assert!(gpu_enabled(), "set SCRIPTOR_GPU_TESTS=1 to run GPU tests");
    let device = Device::open().expect("gpu");
    let stream = device.new_stream().expect("stream");
    let weights = GpuWeights::load(&stream, &ck.files, &ck.spec, true).expect("weights");
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
        kv_blocks: (SEQS * BLOCKS_PER_SEQ + 2) as u64,
        blocks_per_seq: BLOCKS_PER_SEQ as u64,
        full_context_seqs: SEQS as u64,
        max_context: 0,
        slots: SEQS,
        kv_block_tokens: BLOCK,
        vision_batch_patches: 2048,
    };
    GpuExecutor::new(
        &device,
        &stream,
        weights,
        &ck.spec,
        &plan,
        depth,
        TuningTable::default(),
        StepLimits {
            max_seqs: SEQS,
            max_context: BLOCK * BLOCKS_PER_SEQ - 16,
        },
    )
    .expect("executor")
}

fn sequences(ck: &Checkpoint) -> Vec<Seq> {
    let vocab = ck.spec.text.vocab_size;
    (0..SEQS)
        .map(|i| {
            let prompt = english_prompt(&ck.tokenizer, i);
            let mut masks = TokenMasks::new(1, vocab);
            masks.reset_slot(0, &prompt);
            Seq {
                id: 100 + i as u64,
                prompt,
                blocks: (0..BLOCKS_PER_SEQ)
                    .map(|b| (1 + i * BLOCKS_PER_SEQ + b) as u32)
                    .collect(),
                out: Vec::new(),
                masks,
            }
        })
        .collect()
}

/// The sampling mode of step `step`: every fourth step mixes sampled and greedy rows.
fn mode_of(step: usize) -> Mode {
    if step % 4 == 3 {
        Mode::Mixed
    } else {
        Mode::Greedy
    }
}

struct Outcome {
    tokens: Vec<Vec<u32>>,
    bits: Vec<StateBits>,
    probe: Option<Vec<u16>>,
    /// Steps that ran the full 32-row bucket.
    full_steps: usize,
}

/// Captures the executor's graphs, prefills the 32 sequences and decodes them to `GENERATED`
/// tokens, keeping the raw step buffers after every step. With `load`, vision-shaped GEMMs
/// run on the vision stream throughout.
fn run(ck: &Checkpoint, exec: &mut GpuExecutor, load: bool) -> Outcome {
    let report = exec.capture_graphs().expect("capture");
    assert!(
        report.graphs >= 6,
        "buckets 1, 2, 4, 8, 16 and 32 are captured"
    );
    let mut vision = load.then(|| VisionLoad::new(exec));
    let probe = vision.as_mut().map(VisionLoad::probe);
    if let Some(v) = vision.as_mut() {
        v.enqueue(VISION_BATCHES);
    }

    let mut seqs = sequences(ck);
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

    let mut bits = Vec::new();
    let mut full_steps = 0;
    let mut step = 0;
    loop {
        let active: Vec<usize> = (0..SEQS)
            .filter(|&i| seqs[i].out.len() < GENERATED)
            .collect();
        if active.is_empty() {
            break;
        }
        step_subset(exec, &mut seqs, &active, mode_of(step));
        bits.push(exec.step_bits().expect("step bits"));
        full_steps += usize::from(active.len() == SEQS);
        step += 1;
    }
    if vision.is_some() {
        exec.vision_stream().synchronize().expect("vision sync");
    }
    let stats = exec.graph_stats();
    assert_eq!(stats.fallbacks, 0, "a covered shape ran eagerly");
    assert!(stats.replays > 0, "no step replayed a graph");
    Outcome {
        tokens: seqs.into_iter().map(|s| s.out).collect(),
        bits,
        probe,
        full_steps,
    }
}

fn check_equal(label: &str, want: &Outcome, got: &Outcome) {
    assert_eq!(got.tokens, want.tokens, "{label}: sampled tokens diverged");
    assert_eq!(
        got.bits.len(),
        want.bits.len(),
        "{label}: step counts diverged"
    );
    for (i, (g, w)) in got.bits.iter().zip(&want.bits).enumerate() {
        assert_eq!(g.targets, w.targets, "{label}: step {i} targets diverged");
        assert_eq!(g.margins, w.margins, "{label}: step {i} margins diverged");
        assert_eq!(g.hidden, w.hidden, "{label}: step {i} hidden diverged");
    }
}

fn check_case(index: usize) {
    let required = required_split_from_env();
    require_covered(required, &SPLITS).unwrap_or_else(|e| panic!("{e}"));
    let (depth, (vision, decode)) = case(index);
    let split = SmSplit { vision, decode };
    let ck = checkpoint();
    let reference = {
        let mut exec = executor(&ck, depth);
        run(&ck, &mut exec, false)
    };
    assert!(!reference.bits.is_empty());
    assert!(
        reference.full_steps > 0,
        "no step ran the full {SEQS}-sequence bucket"
    );

    let mut exec = executor(&ck, depth);
    let partition = exec.partition(split).map_err(|e| format!("{e:?}"));
    if !settle_grant((vision, decode), REQUIRED_SPLIT, required, partition) {
        return;
    }
    let got = run(&ck, &mut exec, true);
    check_equal(&format!("k={depth} split {split}"), &reference, &got);
    assert!(got.probe.is_some());
    println!(
        "k={depth} rows {} split {split}: {} steps ({} at the full bucket) bitwise identical",
        SEQS * (depth + 1),
        got.bits.len(),
        got.full_steps
    );
    if Some((vision, decode)) == required {
        println!("REQUIRED split {vision}/{decode} granted and bitwise identical");
    }
}

#[test]
#[ignore = "needs GPU and the GLM-OCR checkpoint"]
fn sm_split_production_k3_40_8_is_bitwise_identical_to_the_full_device() {
    check_case(0);
}

#[test]
#[ignore = "needs GPU and the GLM-OCR checkpoint"]
fn sm_split_production_k3_36_12_is_bitwise_identical_to_the_full_device() {
    check_case(1);
}

#[test]
#[ignore = "needs GPU and the GLM-OCR checkpoint"]
fn sm_split_production_k4_40_8_is_bitwise_identical_to_the_full_device() {
    check_case(2);
}

#[test]
#[ignore = "needs GPU and the GLM-OCR checkpoint"]
fn sm_split_production_k4_36_12_is_bitwise_identical_to_the_full_device() {
    check_case(3);
}

#[test]
fn production_cases_cover_every_depth_and_split_exactly_once() {
    let mut seen: Vec<(usize, (u32, u32))> = (0..CASES).map(case).collect();
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), CASES, "a case repeats");
    let mut want: Vec<(usize, (u32, u32))> = DEPTHS
        .iter()
        .flat_map(|&k| SPLITS.iter().map(move |&s| (k, s)))
        .collect();
    want.sort();
    assert_eq!(seen, want);
    assert!(SPLITS.contains(&REQUIRED_SPLIT));
    // The named tests above are cases 0..CASES in order.
    assert_eq!(case(0), (3, (40, 8)));
    assert_eq!(case(1), (3, (36, 12)));
    assert_eq!(case(2), (4, (40, 8)));
    assert_eq!(case(3), (4, (36, 12)));
}

#[test]
fn production_env_must_name_a_covered_case() {
    assert!(require_covered(Some((32, 16)), &SPLITS).is_err());
    assert_eq!(require_covered(Some((36, 12)), &SPLITS), Ok(Some((36, 12))));
    assert_eq!(require_covered(Some((40, 8)), &SPLITS), Ok(Some((40, 8))));
    assert_eq!(require_covered(None, &SPLITS), Ok(None));
}
