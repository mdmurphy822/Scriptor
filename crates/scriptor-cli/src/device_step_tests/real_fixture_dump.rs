//! Dumps the real-model logits fixtures that `scriptor-kernels`' `sampling_split_real_gpu` tests
//! read: bf16 verification-row logits of a GLM-OCR checkpoint with the prompt, output history,
//! penalties and drafts of each row, in the format of `scriptor_kernels::logits_fixture`.
//!
//! Run with
//! `SCRIPTOR_GPU_TESTS=1 SCRIPTOR_MODEL_DIR=<snapshot> SCRIPTOR_SPLIT_FIXTURE_DIR=<out> cargo test -p scriptor-cli --features cuda --profile gpu-test dump_real_logits_fixtures -- --ignored --nocapture --test-threads=1`.
//!
//! `SCRIPTOR_MODEL_DIR` holds `config.json`, `tokenizer.json` and the safetensors files. Two
//! fixtures are written to `SCRIPTOR_SPLIT_FIXTURE_DIR`: `plain.logits` (one row per group, no
//! drafts) from a depth-0 executor and `spec.logits` (`q = 4`, three drafts per group) from a
//! depth-3 executor. Steps run eagerly and the verification logits are copied out as the step
//! produces them, before the draft chain overwrites the shared logits buffer.

use std::path::{Path, PathBuf};

use scriptor_chat::Tokenizer;
use scriptor_cuda::Device;
use scriptor_kernels::logits_fixture::{FixtureGroup, LogitsFixture};
use scriptor_spec::ModelSpec;
use scriptor_weights::SafetensorsDir;

use super::split_graph::{Mode, params_in, step_subset};
use super::{MAX_SEQS, Seq, defer, executor_with, gpu_enabled};
use crate::gpu_executor::GpuWeights;

/// Public-domain English prose, one passage per sequence.
pub(super) const TEXTS: [&str; MAX_SEQS] = [
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, I thought I would sail about a little.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do.",
    "Happy families are all alike; every unhappy family is unhappy in its own way, and the house was in confusion.",
    "In my younger and more vulnerable years my father gave me some advice that I have been turning over in my mind ever since.",
    "There was no possibility of taking a walk that day. We had been wandering, indeed, in the leafless shrubbery an hour in the morning.",
    "Whether I shall turn out to be the hero of my own life, or whether that station will be held by anybody else, these pages must show.",
];
/// Recorded steps per fixture, after the unrecorded first step.
const RECORDED_STEPS: usize = 3;
/// Longest prompt kept, in tokens.
const PROMPT_TOKENS: usize = 30;

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

/// FNV-1a over `bytes`, a stable identity of a small file.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Provenance entries naming the checkpoint: architecture, config identity and the safetensors
/// files with their sizes.
fn provenance(dir: &Path, config: &str, spec: &ModelSpec, depth: usize) -> Vec<(String, String)> {
    let arch = serde_json::from_str::<serde_json::Value>(config)
        .ok()
        .and_then(|v| v["architectures"][0].as_str().map(str::to_string))
        .unwrap_or_default();
    assert!(
        arch.contains("GlmOcr"),
        "{} is not a GLM-OCR checkpoint (architecture {arch:?})",
        dir.display()
    );
    let mut files: Vec<String> = std::fs::read_dir(dir)
        .expect("model dir")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        .map(|e| {
            format!(
                "{}:{}",
                e.file_name().to_string_lossy(),
                e.metadata().map_or(0, |m| m.len())
            )
        })
        .collect();
    files.sort();
    let unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    [
        ("model", format!("GLM-OCR ({arch})")),
        (
            "checkpoint",
            format!(
                "config_fnv1a={:016x} vocab={} files={}",
                fnv1a(config.as_bytes()),
                spec.text.vocab_size,
                files.join(",")
            ),
        ),
        (
            "tool",
            "scriptor-cli device_step_tests::real_fixture_dump::dump_real_logits_fixtures".into(),
        ),
        (
            "prompt",
            format!(
                "{} public-domain passages, fnv1a={:016x}",
                TEXTS.len(),
                fnv1a(TEXTS.concat().as_bytes())
            ),
        ),
        ("draft_depth", depth.to_string()),
        (
            "dtype",
            "bf16 verification logits of eagerly run device steps".into(),
        ),
        ("unix_time", unix.to_string()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

fn dump(dir: &Path, out: &Path, depth: usize, name: &str) {
    let config = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
    let spec = ModelSpec::from_json(&config, generation.as_deref()).expect("model spec");
    let vocab = spec.text.vocab_size;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let device = Device::open().expect("gpu");
    let stream = device.new_stream().expect("stream");
    let files = SafetensorsDir::open(dir).expect("safetensors");
    let weights = GpuWeights::load(&stream, &files, &spec, depth > 0).expect("weights");
    let mut exec = executor_with(&device, &stream, weights, &spec, depth);
    let (k, q) = (depth, depth + 1);
    exec.start_verification_tap(MAX_SEQS * q).expect("tap");

    let mut seqs: Vec<Seq> = TEXTS
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let mut prompt = tokenizer.encode(text).expect("encode");
            prompt.truncate(PROMPT_TOKENS);
            Seq::new(i, prompt, vocab)
        })
        .collect();
    defer(&mut exec, &mut seqs);
    let all: Vec<usize> = (0..MAX_SEQS).collect();
    step_subset(&mut exec, &mut seqs, &all, Mode::Greedy);

    let mut groups = Vec::new();
    for _ in 0..RECORDED_STEPS {
        let before: Vec<(Vec<u32>, RowPenalties)> = seqs
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let p = params_in(s, i, Mode::Greedy);
                (s.out.clone(), (p.repetition, p.presence, p.frequency))
            })
            .collect();
        step_subset(&mut exec, &mut seqs, &all, Mode::Greedy);
        let (logits, drafts) = exec.verification_tap(MAX_SEQS * q).expect("tapped logits");
        for (slot, seq) in seqs.iter().enumerate() {
            let (output, (repetition, presence, frequency)) = before[slot].clone();
            groups.push(FixtureGroup {
                repetition,
                presence,
                frequency,
                prompt: seq.prompt.clone(),
                output,
                drafts: drafts[slot * k..(slot + 1) * k].to_vec(),
                logits: logits[slot * q * vocab..(slot + 1) * q * vocab].to_vec(),
            });
        }
    }
    let fixture = LogitsFixture {
        provenance: provenance(dir, &config, &spec, depth),
        vocab,
        q,
        groups,
    };
    let bytes = fixture.encode().expect("a valid fixture");
    let path = out.join(name);
    std::fs::write(&path, &bytes).expect("write fixture");
    eprintln!(
        "FIXTURE_WRITTEN {} groups {} q {q} vocab {vocab} bytes {}",
        path.display(),
        fixture.groups.len(),
        bytes.len()
    );
}

type RowPenalties = (f32, f32, f32);

#[test]
#[ignore = "needs GPU and a GLM-OCR checkpoint"]
fn dump_real_logits_fixtures() {
    if !gpu_enabled() {
        return;
    }
    let dir = PathBuf::from(required_env("SCRIPTOR_MODEL_DIR"));
    let out = PathBuf::from(required_env("SCRIPTOR_SPLIT_FIXTURE_DIR"));
    std::fs::create_dir_all(&out).expect("fixture dir");
    dump(&dir, &out, 0, "plain.logits");
    dump(&dir, &out, 3, "spec.logits");
}
