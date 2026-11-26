//! Fused RoPE + KV-append variants (shared sin/cos, pure-decode no standalone K/V rows) against
//! the real checkpoint.
//!
//! Correctness gates, each with a synthetic-input and a real-input form:
//! - `*_variants_give_bitwise_equal_logits_across_passes`: bitwise-equal logits over prefill,
//!   mixed, pure-decode and MTP passes; the cache each pass writes is read by the later passes.
//!   Synthetic inputs are arithmetic token ids and an image-like position band; real inputs are
//!   tokenizer-derived OCR text and image-placeholder prompts whose positions come from the
//!   prompt builder's rotary-position computation.
//! - `*_selection_alternates_between_decode_and_prefill_passes`: one executor, repeated scripts.
//! - `graph_decode_replays_*`: captured pure-decode graphs replayed with changed inputs,
//!   positions and slots, one fresh capture per variant. Variant graphs are compared bitwise
//!   with each other; each replay is compared with the eager step by argmax only, and the
//!   logit differences are reported, not asserted.
//!
//! Cost cases come in pairs, `segment_<case>` (profiled stage of the RoPE + KV-append launch)
//! and `end_to_end_<case>` (event-timed whole-forward GPU-stream elapsed, not request or bridge latency), so the two are never mixed. Cases:
//! `decode_b1_q1`, `decode_b1_q4`, `decode_b8_q1`, `decode_b8_q4`, `decode_b32_q1`,
//! `decode_b32_q4`, `padded_b32_q1` (production inactive-row encoding), `prefill_256` and
//! `prefill_1024`. Each test boots one executor and times four variants over
//! `WARMUP + ITERS` forwards per attempt. Attempts run in counterbalanced rounds (rotated, then
//! reversed, variant order; see `rope_schedule`), the round count is planned from a 30 s budget
//! and the bound used is printed, every attempt is printed, and each variant's median follows. Run one case with
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda --profile gpu-test --test gpu_rope_kv_real -- --ignored --nocapture --test-threads=1 --exact <test name>`;
//! `SCRIPTOR_SHAPE_CONTEXT` overrides the decode context length.
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

#[allow(dead_code)]
#[path = "rope_schedule/mod.rs"]
mod schedule;

use cudarc::driver::CudaSlice;
use scriptor_chat::{
    ChatMessage, ChatTemplate, Content, ContentPart, ImageGrid, PromptBuilder, RenderOptions,
    Tokenizer, decode_position,
};
use scriptor_cuda::{CapturedGraph, CudaError, Device};
use scriptor_kernels::cublaslt::GemmOptions;
use scriptor_kernels::heuristics::TuningTable;
use scriptor_kernels::kv::slot_of;
use scriptor_kernels::rope_kv::{RopeKvPlan, RopeKvSelect, RopeKvShape};
use scriptor_kernels::step::PassRows;
use scriptor_model::gpu::{
    DecodeSeq, Fusion, GpuDecoderWeights, GpuKvPool, GpuMtpWeights, PrefillSeq, StepInput,
    StepPlan, TextExecutor, TextGeometry,
};
use scriptor_spec::{GenerationConfig, HfConfig, ModelSpec};
use scriptor_weights::SafetensorsDir;

const BLOCK: usize = 16;
const WARMUP: usize = 2;
const ITERS: usize = 10;
/// Sequences prefilled per forward while building a decode prefix.
const PREFIX_CHUNK: usize = 4;
/// Replays of a captured graph per variant.
const GRAPH_STEPS: usize = 4;
/// Measured-duration bound of one cost case, in seconds.
const COST_BUDGET_SECS: f64 = 30.0;
/// Upper bound on counterbalanced rounds per cost case.
const MAX_ROUNDS: usize = 64;

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

/// Public-domain prose standing in for the text an OCR page carries.
const PASSAGES: [&str; 4] = [
    "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair.",
    "Whether I shall turn out to be the hero of my own life, or whether that station will be held by anybody else, these pages must show. To begin my life with the beginning of my life, I record that I was born, as I have been informed and believe, on a Friday, at twelve o'clock at night.",
];

/// A prompt as the model sees it: token ids, per-row rotary positions and the offset that maps a
/// sequence index to the rotary position of a generated token.
struct Prompt {
    tokens: Vec<u32>,
    positions: Vec<[i32; 3]>,
    delta: i32,
}

impl Prompt {
    fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Rotary position of the token at sequence index `index`.
    fn next_pos(&self, index: usize) -> [i32; 3] {
        [index as i32 + self.delta; 3]
    }
}

/// Inputs shared by the gates and the cost cases.
struct Corpus {
    /// Real prompts: OCR text and image-placeholder prompts from the production preparation path.
    real: Vec<Prompt>,
    /// Tokenised prose continuing a prompt.
    follow: Vec<u32>,
    /// Tokenised prose for prefixes and prefill shapes.
    text: Vec<u32>,
}

impl Corpus {
    fn load(dir: &Path, spec: &ModelSpec) -> Self {
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
        let template =
            ChatTemplate::from_file(dir.join("chat_template.jinja")).expect("chat template");
        let name = tokenizer
            .id_token(spec.image_token_id)
            .expect("image token name");
        let merge = spec.vision.spatial_merge_size;
        let text: Vec<u32> = PASSAGES
            .iter()
            .flat_map(|p| tokenizer.encode(p).expect("encode"))
            .collect();
        let follow = tokenizer
            .encode(" the quick brown fox jumps over the lazy dog 0123456789 Total: $41.50")
            .expect("encode");
        let builder = PromptBuilder::new(tokenizer, template, &name, merge).expect("builder");
        let options = RenderOptions {
            add_generation_prompt: true,
            ..RenderOptions::default()
        };
        let real = (0..12)
            .map(|i| {
                let (message, grids) = if i % 3 == 0 {
                    let page = PASSAGES[(i / 3) % PASSAGES.len()];
                    let cut = page.len() * (2 + i / 3) / 6;
                    let cut = (0..=cut).rev().find(|&c| page.is_char_boundary(c)).unwrap();
                    (
                        ChatMessage {
                            role: "user".into(),
                            content: Some(Content::Text(format!(
                                "Transcribe this page.\n{}",
                                &page[..cut]
                            ))),
                            reasoning_content: None,
                            tool_calls: None,
                            tool_call_id: None,
                            name: None,
                        },
                        Vec::new(),
                    )
                } else {
                    let grid = ImageGrid {
                        t: 1,
                        h: merge * (3 + i % 4),
                        w: merge * (4 + i % 5),
                    };
                    (
                        ChatMessage {
                            role: "user".into(),
                            content: Some(Content::Parts(vec![
                                ContentPart::image_url("image", None),
                                ContentPart::text("Text Recognition:"),
                            ])),
                            reasoning_content: None,
                            tool_calls: None,
                            tool_call_id: None,
                            name: None,
                        },
                        vec![grid],
                    )
                };
                let prompt = builder.build(&[message], &options, &grids).expect("prompt");
                let rotary = prompt.positions().expect("positions");
                let positions: Vec<[i32; 3]> = (0..rotary.len())
                    .map(|r| {
                        [
                            rotary.rows[0][r] as i32,
                            rotary.rows[1][r] as i32,
                            rotary.rows[2][r] as i32,
                        ]
                    })
                    .collect();
                // The step after the last prompt token reads its position from the same
                // delta the production decode loop applies.
                let delta = (decode_position(prompt.token_ids.len(), rotary.delta)
                    - prompt.token_ids.len() as i64) as i32;
                Prompt {
                    tokens: prompt.token_ids,
                    positions,
                    delta,
                }
            })
            .collect();
        Self { real, follow, text }
    }

    /// Synthetic prompts: arithmetic token ids and a position band whose three axes differ.
    fn synthetic(&self, count: usize, base_len: usize) -> Vec<Prompt> {
        (0..count)
            .map(|s| {
                let len = base_len + (s * 9) % 23;
                Prompt {
                    tokens: (0..len)
                        .map(|i| ((i * 13 + s * 29 + 5) % 4000) as u32)
                        .collect(),
                    positions: band_positions(len, s),
                    delta: 0,
                }
            })
            .collect()
    }

    fn synthetic_follow(&self) -> Vec<u32> {
        (0..64).map(|i| ((i * 11 + 5) % 4000) as u32).collect()
    }

    /// `len` tokens of prose starting at an offset chosen by `seed`.
    fn text_tokens(&self, len: usize, seed: usize) -> Vec<u32> {
        (0..len)
            .map(|i| self.text[(seed * 17 + i) % self.text.len()])
            .collect()
    }
}

/// Rotary positions of a synthetic prompt of `len` rows with an image-like band.
fn band_positions(len: usize, seed: usize) -> Vec<[i32; 3]> {
    (0..len)
        .map(|i| {
            if i >= 4 && i < len.saturating_sub(4) {
                let t = 4 + (seed % 3) as i32;
                [t, t + (i % 5) as i32, t + (i % 7) as i32]
            } else {
                [i as i32; 3]
            }
        })
        .collect()
}

struct Rig {
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    exec: TextExecutor,
    weights: GpuDecoderWeights,
    mtp: GpuMtpWeights,
    kv: GpuKvPool,
    mtp_kv: GpuKvPool,
    tables: Vec<Vec<u32>>,
    geometry: TextGeometry,
    corpus: Corpus,
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
    let corpus = Corpus::load(&dir, &spec);
    let files = SafetensorsDir::open(&dir).expect("safetensors");
    let dev = Device::open().expect("gpu");
    // SAFETY: the rig allocates, uploads, runs and reads back on the one stream below, so
    // stream order alone serialises its buffers, as in the executor; without tracking no
    // event recorded before a capture becomes a dependency of the captured work.
    unsafe { dev.context().disable_event_tracking() };
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
    let mtp = GpuMtpWeights::load(&stream, &files, &spec.text).expect("mtp weights");
    let per_seq = max_ctx.div_ceil(BLOCK) + 2;
    let kv = GpuKvPool::new(
        &stream,
        geometry.layers,
        max_batch * per_seq,
        BLOCK,
        &geometry,
    )
    .expect("pool");
    let mtp_kv =
        GpuKvPool::new(&stream, 1, max_batch * per_seq, BLOCK, &geometry).expect("mtp pool");
    let tables = (0..max_batch)
        .map(|s| (0..per_seq).map(|b| (s * per_seq + b) as u32).collect())
        .collect();
    Rig {
        ctx: dev.context().clone(),
        stream,
        exec,
        weights,
        mtp,
        kv,
        mtp_kv,
        tables,
        geometry,
        corpus,
        _dev: dev,
    }
}

const REFERENCE: RopeKvSelect = RopeKvSelect {
    shared_trig: false,
    skip_kv_scratch: false,
};

const SELECTS: [(&str, RopeKvSelect); 4] = [
    ("reference", REFERENCE),
    (
        "A shared",
        RopeKvSelect {
            shared_trig: true,
            skip_kv_scratch: false,
        },
    ),
    (
        "B decode",
        RopeKvSelect {
            shared_trig: false,
            skip_kv_scratch: true,
        },
    ),
    (
        "A+B",
        RopeKvSelect {
            shared_trig: true,
            skip_kv_scratch: true,
        },
    ),
];

fn logits(
    rig: &mut Rig,
    input: &StepInput<'_>,
) -> (Vec<f32>, Option<cudarc::driver::CudaSlice<u16>>) {
    let out = rig
        .exec
        .forward(&rig.weights, &mut rig.kv, input, None)
        .expect("forward");
    rig.stream.synchronize().expect("sync");
    (
        rig.stream.clone_dtoh(&out.logits).expect("logits"),
        out.hidden,
    )
}

/// Asserts two pass lists have the same pass count and per-pass lengths, then that every logit
/// is bit-identical.
fn assert_bitwise(want: &[Vec<f32>], got: &[Vec<f32>], what: &str) {
    assert_eq!(want.len(), got.len(), "{what}: pass count differs");
    for (pass, (w, g)) in want.iter().zip(got).enumerate() {
        assert!(!w.is_empty(), "{what}: pass {pass} produced no logits");
        assert_eq!(
            w.len(),
            g.len(),
            "{what}: pass {pass} logits length differs"
        );
        let first = w
            .iter()
            .zip(g)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert!(first.is_none(), "{what}: pass {pass} differs at {first:?}");
    }
}

/// Token of sequence `s` at sequence index `index`, `j` rows into a pass.
fn follow_token(follow: &[u32], s: usize, index: usize, j: usize) -> u32 {
    follow[(s * 7 + index + j * 3) % follow.len()]
}

/// One scripted run over `prompts[..prompts.len() - 1]` plus one prompt that prefills in the
/// mixed pass. `committed[s]` is the number of positions of sequence `s` already written to the
/// target cache, advanced by exactly the rows each pass writes, so every pass reads only
/// positions an earlier pass of this run produced.
struct Run<'a> {
    prompts: &'a [Prompt],
    follow: &'a [u32],
    seqs: usize,
    committed: Vec<usize>,
    all: Vec<Vec<f32>>,
}

impl<'a> Run<'a> {
    fn new(prompts: &'a [Prompt], follow: &'a [u32]) -> Self {
        let seqs = prompts.len() - 1;
        Self {
            prompts,
            follow,
            seqs,
            committed: vec![0; seqs],
            all: Vec::new(),
        }
    }

    /// Prefill of every sequence, then the MTP pass over the same rows.
    fn prefill(&mut self, rig: &mut Rig, with_mtp: bool) {
        let tables = rig.tables.clone();
        let prefill: Vec<PrefillSeq> = (0..self.seqs)
            .map(|s| PrefillSeq {
                tokens: &self.prompts[s].tokens,
                positions: &self.prompts[s].positions,
                block_table: &tables[s],
            })
            .collect();
        let rows: Vec<u32> =
            (0..prefill.iter().map(|p| p.tokens.len()).sum::<usize>() as u32).collect();
        let input = StepInput {
            prefill: &prefill,
            decode: &[],
            image_rows: &[],
            sample_rows: &rows,
            keep_hidden: with_mtp,
        };
        let (l, hidden) = logits(rig, &input);
        self.all.push(l);
        if with_mtp {
            self.mtp(rig, &input, &hidden.expect("hidden"));
        }
        for s in 0..self.seqs {
            self.committed[s] = self.prompts[s].len();
        }
    }

    fn mtp(&mut self, rig: &mut Rig, input: &StepInput<'_>, hidden: &CudaSlice<u16>) {
        let mtp = rig
            .exec
            .mtp_forward(&rig.mtp, &mut rig.mtp_kv, input, hidden)
            .expect("mtp");
        rig.stream.synchronize().expect("sync");
        self.all
            .push(rig.stream.clone_dtoh(&mtp.logits).expect("mtp logits"));
    }

    /// Mixed pass: sequence `seqs` prefills while the others decode one row each.
    fn mixed(&mut self, rig: &mut Rig, with_mtp: bool) {
        let tables = rig.tables.clone();
        let seqs = self.seqs;
        let tk: Vec<[u32; 1]> = (0..seqs)
            .map(|s| [follow_token(self.follow, s, self.committed[s], 0)])
            .collect();
        let ps: Vec<[[i32; 3]; 1]> = (0..seqs)
            .map(|s| [self.prompts[s].next_pos(self.committed[s])])
            .collect();
        let fresh = &self.prompts[seqs];
        let new_prefill = [PrefillSeq {
            tokens: &fresh.tokens,
            positions: &fresh.positions,
            block_table: &tables[seqs],
        }];
        let decode: Vec<DecodeSeq> = (0..seqs)
            .map(|s| DecodeSeq {
                tokens: &tk[s],
                positions: &ps[s],
                block_table: &tables[s],
                cached: self.committed[s],
            })
            .collect();
        let rows: Vec<u32> = (0..(fresh.len() + seqs) as u32).collect();
        let input = StepInput {
            prefill: &new_prefill,
            decode: &decode,
            image_rows: &[],
            sample_rows: &rows,
            keep_hidden: with_mtp,
        };
        let (l, hidden) = logits(rig, &input);
        self.all.push(l);
        if with_mtp {
            self.mtp(rig, &input, &hidden.expect("hidden"));
        }
        for c in &mut self.committed {
            *c += 1;
        }
    }

    /// Pure-decode pass of `q_len` rows per sequence at each sequence's committed length.
    fn decode(&mut self, rig: &mut Rig, q_len: usize, with_mtp: bool) {
        let tables = rig.tables.clone();
        let seqs = self.seqs;
        let cached = self.committed.clone();
        let tk: Vec<Vec<u32>> = (0..seqs)
            .map(|s| {
                (0..q_len)
                    .map(|j| follow_token(self.follow, s, cached[s], j))
                    .collect()
            })
            .collect();
        let ps: Vec<Vec<[i32; 3]>> = (0..seqs)
            .map(|s| {
                (0..q_len)
                    .map(|j| self.prompts[s].next_pos(cached[s] + j))
                    .collect()
            })
            .collect();
        let decode: Vec<DecodeSeq> = (0..seqs)
            .map(|s| DecodeSeq {
                tokens: &tk[s],
                positions: &ps[s],
                block_table: &tables[s],
                cached: cached[s],
            })
            .collect();
        let rows: Vec<u32> = (0..(seqs * q_len) as u32).collect();
        let input = StepInput {
            prefill: &[],
            decode: &decode,
            image_rows: &[],
            sample_rows: &rows,
            keep_hidden: with_mtp,
        };
        let (l, hidden) = logits(rig, &input);
        self.all.push(l);
        if with_mtp {
            self.mtp(rig, &input, &hidden.expect("hidden"));
        }
        for c in &mut self.committed {
            *c += q_len;
        }
    }
}

/// The full script: prefill (target then MTP), mixed (target then MTP), pure decode at one row
/// (target then MTP) and at four rows. Returns the logits of every pass in order.
fn script(rig: &mut Rig, prompts: &[Prompt], follow: &[u32]) -> Vec<Vec<f32>> {
    let mut run = Run::new(prompts, follow);
    run.prefill(rig, true);
    run.mixed(rig, true);
    run.decode(rig, 1, true);
    run.decode(rig, 4, false);
    run.all
}

fn variants_bitwise_equal(rig: &mut Rig, prompts: &[Prompt], follow: &[u32]) {
    let mut reference: Option<Vec<Vec<f32>>> = None;
    for (name, select) in SELECTS {
        rig.exec.set_rope_kv(select);
        let got = script(rig, prompts, follow);
        match &reference {
            None => reference = Some(got),
            Some(want) => assert_bitwise(want, &got, &format!("{name} against the reference")),
        }
    }
}

fn max_len(prompts: &[Prompt]) -> usize {
    prompts.iter().map(Prompt::len).max().unwrap() + 16
}

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn synthetic_variants_give_bitwise_equal_logits_across_passes() {
    let mut rig = rig(12, 160);
    let prompts = rig.corpus.synthetic(12, 90);
    let follow = rig.corpus.synthetic_follow();
    variants_bitwise_equal(&mut rig, &prompts, &follow);
}

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn real_input_variants_give_bitwise_equal_logits_across_passes() {
    let mut rig = rig(12, 400);
    let prompts = std::mem::take(&mut rig.corpus.real);
    assert!(max_len(&prompts) <= 400);
    let follow = rig.corpus.follow.clone();
    variants_bitwise_equal(&mut rig, &prompts, &follow);
}

fn alternation(rig: &mut Rig, prompts: &[Prompt], follow: &[u32]) {
    // One executor, B selected throughout: pure-decode passes skip the standalone rows while
    // the prefill passes between them still attend over them.
    rig.exec.set_rope_kv(RopeKvSelect {
        shared_trig: true,
        skip_kv_scratch: true,
    });
    let first = script(rig, prompts, follow);
    let second = script(rig, prompts, follow);
    assert_bitwise(&first, &second, "second script against the first");
}

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn synthetic_selection_alternates_between_decode_and_prefill_passes() {
    let mut rig = rig(4, 160);
    let prompts = rig.corpus.synthetic(4, 70);
    let follow = rig.corpus.synthetic_follow();
    alternation(&mut rig, &prompts, &follow);
}

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn real_input_selection_alternates_between_decode_and_prefill_passes() {
    let mut rig = rig(4, 400);
    let prompts: Vec<Prompt> = std::mem::take(&mut rig.corpus.real)
        .into_iter()
        .take(4)
        .collect();
    let follow = rig.corpus.follow.clone();
    alternation(&mut rig, &prompts, &follow);
}

/// Device row tables of a decode pass, in the layout `StepPlan` holds them.
struct RowTables {
    ids: Vec<i32>,
    positions: Vec<i32>,
    slots: Vec<i32>,
    seq_lens: Vec<i32>,
    sample_rows: Vec<i32>,
}

/// Rows of `tokens[s].len()` per sequence at `cached[s]`; `None` encodes a padded (inactive)
/// sequence the way the step bookkeeping kernels do: token 0, position 0, no KV slot, attention
/// over block zero.
fn row_tables(
    tables: &[Vec<u32>],
    cached: &[Option<usize>],
    tokens: &[Vec<u32>],
    deltas: &[i32],
) -> RowTables {
    let mut t = RowTables {
        ids: Vec::new(),
        positions: Vec::new(),
        slots: Vec::new(),
        seq_lens: Vec::new(),
        sample_rows: Vec::new(),
    };
    for (s, c) in cached.iter().enumerate() {
        let q = tokens[s].len();
        for (j, &token) in tokens[s].iter().enumerate() {
            match c {
                Some(c) => {
                    t.ids.push(token as i32);
                    t.positions.extend([(c + j) as i32 + deltas[s]; 3]);
                    t.slots.push(slot_of(&tables[s], BLOCK, c + j));
                }
                None => {
                    t.ids.push(0);
                    t.positions.extend([0; 3]);
                    t.slots.push(-1);
                }
            }
            t.sample_rows.push(t.ids.len() as i32 - 1);
        }
        t.seq_lens.push(c.map_or(q, |c| c + q) as i32);
    }
    t
}

/// A decode step over device-resident row tables, the form a captured graph replays.
struct Resident {
    plan: StepPlan,
    logits: CudaSlice<f32>,
    rows: usize,
}

impl Resident {
    /// Plan for `seqs` sequences of `q_len` rows; block tables are uploaded for the first
    /// `seqs` sequences, with those listed in `padded` zeroed.
    fn new(rig: &mut Rig, seqs: usize, q_len: usize, max_ctx: usize, padded: &[usize]) -> Self {
        let max_blocks = rig.tables[0].len();
        let rows = seqs * q_len;
        let mut plan =
            StepPlan::new(&rig.stream, &rig.kv, rows, seqs, max_blocks).expect("step plan");
        let flat: Vec<i32> = (0..seqs)
            .flat_map(|s| {
                let zero = padded.contains(&s);
                rig.tables[s]
                    .iter()
                    .map(move |&b| if zero { 0 } else { b as i32 })
            })
            .collect();
        rig.stream
            .memcpy_htod(&flat, plan.block_tables_mut())
            .expect("tables");
        rig.exec
            .reserve_resident(&[(seqs, q_len)], max_ctx)
            .expect("reserve");
        plan.shape(seqs, q_len, rows, max_ctx).expect("shape");
        let logits = rig
            .stream
            .alloc_zeros::<f32>(rows * rig.geometry.vocab)
            .expect("logits");
        Self { plan, logits, rows }
    }

    /// Rewrites the row tables of the next pass.
    fn upload(&mut self, rig: &Rig, t: &RowTables) {
        assert_eq!(t.ids.len(), self.rows);
        let PassRows {
            ids,
            positions,
            slots,
            seq_lens,
            sample_rows,
        } = self.plan.rows_mut();
        let s = &rig.stream;
        s.memcpy_htod(&t.ids, ids).expect("ids");
        s.memcpy_htod(&t.positions, positions).expect("positions");
        s.memcpy_htod(&t.slots, slots).expect("slots");
        s.memcpy_htod(&t.seq_lens, seq_lens).expect("seq lens");
        s.memcpy_htod(&t.sample_rows, sample_rows)
            .expect("sample rows");
    }

    fn run(&mut self, rig: &mut Rig) {
        rig.exec
            .forward_resident(
                &rig.weights,
                &mut rig.kv,
                &self.plan,
                Some(&mut self.logits),
                None,
            )
            .expect("resident forward");
    }

    /// Captures one pass under the executor's current selection.
    fn capture(&mut self, rig: &mut Rig) -> CapturedGraph {
        let stream = rig.stream.clone();
        CapturedGraph::capture(&stream, |_| {
            rig.exec
                .forward_resident(
                    &rig.weights,
                    &mut rig.kv,
                    &self.plan,
                    Some(&mut self.logits),
                    None,
                )
                .map_err(|e| CudaError::Graph(e.to_string()))
        })
        .expect("capture")
    }

    fn read(&self, rig: &Rig) -> Vec<f32> {
        rig.stream.synchronize().expect("sync");
        rig.stream.clone_dtoh(&self.logits).expect("logits")
    }
}

/// Route the executor takes for a pass of `rows` rows of which `prefill_rows` are prefill.
fn route(rig: &Rig, rows: usize, prefill_rows: usize) -> RopeKvPlan {
    rig.exec.rope_kv().plan(&RopeKvShape {
        rows,
        prefill_rows,
        head_dim: rig.geometry.head_dim,
        sections: rig.geometry.mrope_section,
    })
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

/// Per variant: eager prefill and mixed passes (which keep the standalone-row writes), eager
/// reference steps, then a freshly captured pure-decode graph replayed with new tokens,
/// positions, slots and sequence lengths each step. Graph logits are bitwise equal across
/// variants; against the eager steps the check is an argmax comparison with the max logit
/// difference reported, not bitwise validation.
fn graph_replays(mut rig: Rig, prompts: &[Prompt], follow: &[u32], max_ctx: usize) {
    let seqs = prompts.len() - 1;
    let vocab = rig.geometry.vocab;
    let mut reference: Option<Vec<Vec<f32>>> = None;
    for (name, select) in SELECTS {
        rig.exec.set_rope_kv(select);
        let mut run = Run::new(prompts, follow);
        run.prefill(&mut rig, false);
        let prefill_rows: usize = prompts[..seqs].iter().map(Prompt::len).sum();
        let prefill_route = route(&rig, prefill_rows, prefill_rows);
        run.mixed(&mut rig, false);
        let mixed_rows = prompts[seqs].len() + seqs;
        let mixed_route = route(&rig, mixed_rows, prompts[seqs].len());
        assert!(!prefill_route.skip_kv_scratch && !mixed_route.skip_kv_scratch);
        assert!(prefill_route.serves(prefill_rows) && mixed_route.serves(prompts[seqs].len()));
        let start = run.committed.clone();
        let eager_from = run.all.len();
        for _ in 0..GRAPH_STEPS {
            run.decode(&mut rig, 1, false);
        }
        let eager: Vec<Vec<f32>> = run.all[eager_from..].to_vec();

        let decode_route = route(&rig, seqs, 0);
        assert_eq!(decode_route.skip_kv_scratch, select.skip_kv_scratch);
        assert!(decode_route.serves(0));
        let mut res = Resident::new(&mut rig, seqs, 1, max_ctx, &[]);
        let tables = rig.tables.clone();
        let deltas: Vec<i32> = prompts[..seqs].iter().map(|p| p.delta).collect();
        let step_rows = |t: usize| {
            let cached: Vec<Option<usize>> = (0..seqs).map(|s| Some(start[s] + t)).collect();
            let tokens: Vec<Vec<u32>> = (0..seqs)
                .map(|s| vec![follow_token(follow, s, start[s] + t, 0)])
                .collect();
            row_tables(&tables, &cached, &tokens, &deltas)
        };
        res.upload(&rig, &step_rows(0));
        res.run(&mut rig);
        rig.stream.synchronize().expect("sync");
        let graph = res.capture(&mut rig);
        let mut replayed = Vec::new();
        for t in 0..GRAPH_STEPS {
            res.upload(&rig, &step_rows(t));
            graph.launch().expect("replay");
            replayed.push(res.read(&rig));
        }
        println!(
            "route {name}: prefill {prefill_route:?} mixed {mixed_route:?} \
             captured decode {decode_route:?} (selection {:?}, {GRAPH_STEPS} replays)",
            rig.exec.rope_kv()
        );
        for (t, (g, e)) in replayed.iter().zip(&eager).enumerate() {
            assert_eq!(g.len(), seqs * vocab);
            assert_eq!(e.len(), g.len(), "{name}: eager step {t} length");
            let mut worst = 0f32;
            for s in 0..seqs {
                let (gr, er) = (
                    &g[s * vocab..(s + 1) * vocab],
                    &e[s * vocab..(s + 1) * vocab],
                );
                assert_eq!(argmax(gr), argmax(er), "{name}: step {t} row {s} token");
                worst = worst.max(
                    gr.iter()
                        .zip(er)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f32::max),
                );
            }
            println!(
                "  step {t}: graph vs eager argmax equal; max |dlogit| {worst:.4} (reported, not bitwise)"
            );
        }
        match &reference {
            None => reference = Some(replayed),
            Some(want) => assert_bitwise(want, &replayed, &format!("{name} graph replays")),
        }
    }
}

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn graph_decode_replays_real_inputs_match_across_variants() {
    let mut rig = rig(8, 400);
    let prompts: Vec<Prompt> = std::mem::take(&mut rig.corpus.real)
        .into_iter()
        .take(7)
        .collect();
    assert!(max_len(&prompts) + GRAPH_STEPS <= 400);
    let follow = rig.corpus.follow.clone();
    graph_replays(rig, &prompts, &follow, 400);
}

#[test]
#[ignore = "needs GPU and the checkpoint"]
fn graph_decode_replays_synthetic_inputs_match_across_variants() {
    let rig = rig(8, 200);
    let prompts = rig.corpus.synthetic(7, 60);
    let follow = rig.corpus.synthetic_follow();
    assert!(max_len(&prompts) + GRAPH_STEPS <= 200);
    graph_replays(rig, &prompts, &follow, 200);
}

#[derive(Clone, Copy)]
enum Mode {
    /// Profiled stage of the RoPE + KV-append launch.
    Segment,
    /// Whole-forward GPU-stream elapsed time between two events; not request or bridge latency.
    EndToEnd,
}

#[derive(Clone, Copy)]
enum Work {
    /// Every sequence active with a real cached prefix.
    Decode { batch: usize, q_len: usize },
    /// A bucket of `batch` single-row sequences, every third one padded, run through the
    /// device-resident step.
    Padded { batch: usize },
    /// One prompt of `len` rows.
    Prefill { len: usize },
}

fn context() -> usize {
    std::env::var("SCRIPTOR_SHAPE_CONTEXT")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(512)
}

/// Cached length of sequence `s`: varied contexts around `context`.
fn cached_len(context: usize, s: usize) -> usize {
    context / 2 + (s * 37) % context.max(2)
}

/// Writes real-text prefixes of `lens` rows into the cache of sequences `0..lens.len()` under
/// the reference variant, outside any measured interval.
fn fill_prefixes(rig: &mut Rig, lens: &[Option<usize>]) {
    rig.exec.set_rope_kv(REFERENCE);
    let tables = rig.tables.clone();
    let todo: Vec<(usize, usize)> = lens
        .iter()
        .enumerate()
        .filter_map(|(s, l)| l.map(|l| (s, l)))
        .collect();
    for chunk in todo.chunks(PREFIX_CHUNK) {
        let toks: Vec<Vec<u32>> = chunk
            .iter()
            .map(|&(s, l)| rig.corpus.text_tokens(l, s))
            .collect();
        let pos: Vec<Vec<[i32; 3]>> = chunk
            .iter()
            .map(|&(_, l)| (0..l).map(|i| [i as i32; 3]).collect())
            .collect();
        let prefill: Vec<PrefillSeq> = chunk
            .iter()
            .enumerate()
            .map(|(i, &(s, _))| PrefillSeq {
                tokens: &toks[i],
                positions: &pos[i],
                block_table: &tables[s],
            })
            .collect();
        let total: usize = chunk.iter().map(|&(_, l)| l).sum();
        let rows = [(total - 1) as u32];
        logits(
            rig,
            &StepInput {
                prefill: &prefill,
                decode: &[],
                image_rows: &[],
                sample_rows: &rows,
                keep_hidden: false,
            },
        );
    }
}

fn run_cost(mode: Mode, work: Work) {
    let ctx = context();
    let (batch, label) = match work {
        Work::Decode { batch, q_len } => (batch, format!("decode b{batch} q{q_len} ctx{ctx}")),
        Work::Padded { batch } => (batch, format!("padded b{batch} q1 ctx{ctx}")),
        Work::Prefill { len } => (1, format!("prefill {len}")),
    };
    let max_ctx = match work {
        Work::Prefill { len } => len + 16,
        _ => ctx * 2 + 16,
    };
    let mut rig = rig(batch, max_ctx);
    let tables = rig.tables.clone();

    // Per-variant forward; the input is built once, outside the measured interval.
    let mut step: Box<dyn FnMut(&mut Rig)> = match work {
        Work::Decode { batch, q_len } => {
            let cached: Vec<usize> = (0..batch).map(|s| cached_len(ctx, s)).collect();
            fill_prefixes(
                &mut rig,
                &cached.iter().map(|&c| Some(c)).collect::<Vec<_>>(),
            );
            let tokens: Vec<Vec<u32>> = (0..batch)
                .map(|s| rig.corpus.text_tokens(q_len, s + 5))
                .collect();
            let positions: Vec<Vec<[i32; 3]>> = (0..batch)
                .map(|s| (0..q_len).map(|j| [(cached[s] + j) as i32; 3]).collect())
                .collect();
            let rows: Vec<u32> = (0..(batch * q_len) as u32).collect();
            Box::new(move |rig: &mut Rig| {
                let seqs: Vec<DecodeSeq> = (0..batch)
                    .map(|s| DecodeSeq {
                        tokens: &tokens[s],
                        positions: &positions[s],
                        block_table: &tables[s],
                        cached: cached[s],
                    })
                    .collect();
                rig.exec
                    .forward(
                        &rig.weights,
                        &mut rig.kv,
                        &StepInput {
                            prefill: &[],
                            decode: &seqs,
                            image_rows: &[],
                            sample_rows: &rows,
                            keep_hidden: false,
                        },
                        None,
                    )
                    .expect("step");
            })
        }
        Work::Padded { batch } => {
            let cached: Vec<Option<usize>> = (0..batch)
                .map(|s| (s % 3 != 2).then(|| cached_len(ctx, s)))
                .collect();
            fill_prefixes(&mut rig, &cached);
            let padded: Vec<usize> = (0..batch).filter(|s| s % 3 == 2).collect();
            let longest = cached.iter().flatten().max().copied().unwrap() + 1;
            let mut res = Resident::new(&mut rig, batch, 1, longest, &padded);
            let tokens: Vec<Vec<u32>> = (0..batch)
                .map(|s| rig.corpus.text_tokens(1, s + 5))
                .collect();
            res.upload(
                &rig,
                &row_tables(&tables, &cached, &tokens, &vec![0; batch]),
            );
            Box::new(move |rig: &mut Rig| res.run(rig))
        }
        Work::Prefill { len } => {
            let toks = rig.corpus.text_tokens(len, 1);
            let pos: Vec<[i32; 3]> = (0..len).map(|i| [i as i32; 3]).collect();
            Box::new(move |rig: &mut Rig| {
                let prefill = [PrefillSeq {
                    tokens: &toks,
                    positions: &pos,
                    block_table: &tables[0],
                }];
                let rows = [(len - 1) as u32];
                rig.exec
                    .forward(
                        &rig.weights,
                        &mut rig.kv,
                        &StepInput {
                            prefill: &prefill,
                            decode: &[],
                            image_rows: &[],
                            sample_rows: &rows,
                            keep_hidden: false,
                        },
                        None,
                    )
                    .expect("step");
            })
        }
    };

    /// One measured attempt of one variant.
    struct Attempt {
        variant: usize,
        round: usize,
        position: usize,
        /// Whole-forward GPU-stream elapsed ms per forward (end-to-end) or rope stage ms.
        ms: f64,
        bytes_mb: f64,
        launches: u32,
    }

    let mut attempts: Vec<Attempt> = Vec::new();
    let started = std::time::Instant::now();
    let mut rounds = SELECTS.len();
    let mut bound = 0.0;
    let mut round = 0;
    while round < rounds {
        let order = schedule::round_order(SELECTS.len(), round);
        let round_start = std::time::Instant::now();
        for (position, &variant) in order.iter().enumerate() {
            let (_, select) = SELECTS[variant];
            rig.exec.set_rope_kv(select);
            for _ in 0..WARMUP {
                step(&mut rig);
            }
            rig.stream.synchronize().expect("sync");
            let attempt = match mode {
                Mode::EndToEnd => {
                    let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
                    let (a, b) = (
                        rig.ctx.new_event(flags).expect("event"),
                        rig.ctx.new_event(flags).expect("event"),
                    );
                    a.record(&rig.stream).expect("record");
                    for _ in 0..ITERS {
                        step(&mut rig);
                    }
                    b.record(&rig.stream).expect("record");
                    let ms = f64::from(a.elapsed_ms(&b).expect("elapsed")) / ITERS as f64;
                    Attempt {
                        variant,
                        round,
                        position,
                        ms,
                        bytes_mb: 0.0,
                        launches: 0,
                    }
                }
                Mode::Segment => {
                    rig.exec.set_profile(true);
                    for _ in 0..ITERS {
                        step(&mut rig);
                    }
                    let stats = rig.exec.take_profile().expect("profile");
                    rig.exec.set_profile(false);
                    let stage = stats
                        .iter()
                        .find(|s| s.name == "split_rope_kv_append")
                        .expect("rope stage");
                    Attempt {
                        variant,
                        round,
                        position,
                        ms: stage.millis / ITERS as f64,
                        bytes_mb: stage.bytes as f64 / ITERS as f64 / 1e6,
                        launches: stage.launches,
                    }
                }
            };
            attempts.push(attempt);
        }
        if round == 0 {
            let (planned, total) = schedule::plan_rounds(
                SELECTS.len(),
                round_start.elapsed().as_secs_f64(),
                COST_BUDGET_SECS,
                MAX_ROUNDS,
            );
            rounds = planned;
            bound = total;
        }
        round += 1;
    }
    println!(
        "{label}: {rounds} counterbalanced rounds x {} variants, planned bound {bound:.1} s of \
         {COST_BUDGET_SECS:.0} s budget, measured {:.1} s",
        SELECTS.len(),
        started.elapsed().as_secs_f64()
    );
    match mode {
        Mode::Segment => {
            println!(
                "| shape | round | position | variant | rope stage ms | stage bytes MB | launches |"
            );
            println!("|---|---|---|---|---|---|---|");
        }
        Mode::EndToEnd => {
            println!(
                "| shape | round | position | variant | whole-forward GPU-stream elapsed ms |"
            );
            println!("|---|---|---|---|---|");
        }
    }
    for at in &attempts {
        let name = SELECTS[at.variant].0;
        match mode {
            Mode::EndToEnd => println!(
                "| {label} | {} | {} | {name} | {:.3} |",
                at.round, at.position, at.ms
            ),
            Mode::Segment => println!(
                "| {label} | {} | {} | {name} | {:.4} | {:.2} | {} |",
                at.round, at.position, at.ms, at.bytes_mb, at.launches
            ),
        }
    }
    println!("median over {rounds} attempts per variant:");
    for (v, (name, _)) in SELECTS.iter().enumerate() {
        let ms: Vec<f64> = attempts
            .iter()
            .filter(|a| a.variant == v)
            .map(|a| a.ms)
            .collect();
        match mode {
            Mode::EndToEnd => println!(
                "| {label} | {name} | median whole-forward GPU-stream elapsed {:.3} ms |",
                schedule::median(&ms)
            ),
            Mode::Segment => println!(
                "| {label} | {name} | median rope stage {:.4} ms |",
                schedule::median(&ms)
            ),
        }
    }
}

macro_rules! cost_case {
    ($segment:ident, $end_to_end:ident, $work:expr) => {
        #[test]
        #[ignore = "needs GPU and the checkpoint"]
        fn $segment() {
            run_cost(Mode::Segment, $work);
        }

        #[test]
        #[ignore = "needs GPU and the checkpoint"]
        fn $end_to_end() {
            run_cost(Mode::EndToEnd, $work);
        }
    };
}

cost_case!(
    segment_decode_b1_q1,
    end_to_end_decode_b1_q1,
    Work::Decode { batch: 1, q_len: 1 }
);
cost_case!(
    segment_decode_b1_q4,
    end_to_end_decode_b1_q4,
    Work::Decode { batch: 1, q_len: 4 }
);
cost_case!(
    segment_decode_b8_q1,
    end_to_end_decode_b8_q1,
    Work::Decode { batch: 8, q_len: 1 }
);
cost_case!(
    segment_decode_b8_q4,
    end_to_end_decode_b8_q4,
    Work::Decode { batch: 8, q_len: 4 }
);
cost_case!(
    segment_decode_b32_q1,
    end_to_end_decode_b32_q1,
    Work::Decode {
        batch: 32,
        q_len: 1
    }
);
cost_case!(
    segment_decode_b32_q4,
    end_to_end_decode_b32_q4,
    Work::Decode {
        batch: 32,
        q_len: 4
    }
);
cost_case!(
    segment_padded_b32_q1,
    end_to_end_padded_b32_q1,
    Work::Padded { batch: 32 }
);
cost_case!(
    segment_prefill_256,
    end_to_end_prefill_256,
    Work::Prefill { len: 256 }
);
cost_case!(
    segment_prefill_1024,
    end_to_end_prefill_1024,
    Work::Prefill { len: 1024 }
);
