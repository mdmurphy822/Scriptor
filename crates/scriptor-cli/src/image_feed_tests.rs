//! Tests of how image embeddings reach the text forward pass: the CPU decision and cost
//! model, and GPU comparisons of the in-place feed against the packed copy on the real
//! checkpoint with real tokenizer-derived prompts and fixture images.
//!
//! GPU tests are ignored by default. Set `SCRIPTOR_SNAPSHOT_DIR` to the checkpoint directory
//! (or leave it unset to use the Hugging Face cache) and run
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-cli --features cuda --profile gpu-test image_feed -- --ignored --nocapture --test-threads 1`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use scriptor_cache::{ImageGrid as CacheGrid, embedding_key};
use scriptor_chat::{
    ChatMessage, ChatTemplate, Content, ContentPart, ImageGrid, PromptBuilder, RenderOptions,
    Tokenizer,
};
use scriptor_cuda::Device;
use scriptor_engine::{DecodeRow, DeviceRow, ModelExecutor, PrefillRow, VisionJob};
use scriptor_image::{MERGE_SIZE, SmartResizeConfig, preprocess};
use scriptor_kernels::heuristics::TuningTable;
use scriptor_kernels::sampling::{RowParams, TokenMasks};
use scriptor_spec::{GenerationConfig, HfConfig, MemoryPlan, ModelSpec};
use scriptor_weights::SafetensorsDir;

use crate::device_step::{StateBits, StepLimits};
use crate::gpu_executor::{FeedCost, GpuEmbedding, GpuExecutor, GpuWeights, ImageFeed};

const HIDDEN: usize = 1536;

#[test]
fn no_image_reference_needs_no_feed() {
    assert_eq!(ImageFeed::plan(&[], true), ImageFeed::None);
    assert_eq!(ImageFeed::plan(&[], false), ImageFeed::None);
}

#[test]
fn one_reference_is_read_in_place() {
    assert_eq!(ImageFeed::plan(&[20], true), ImageFeed::Direct);
}

#[test]
fn one_reference_stays_packed_when_in_place_reads_are_off() {
    assert_eq!(
        ImageFeed::plan(&[20], false),
        ImageFeed::Packed { sources: 1 }
    );
}

#[test]
fn two_different_images_are_packed() {
    assert_eq!(
        ImageFeed::plan(&[20, 7], true),
        ImageFeed::Packed { sources: 2 }
    );
}

#[test]
fn the_same_image_twice_counts_two_references_and_is_packed() {
    assert_eq!(
        ImageFeed::plan(&[20, 20], true),
        ImageFeed::Packed { sources: 2 }
    );
}

#[test]
fn in_place_feed_costs_nothing() {
    let rows = [20];
    let cost = ImageFeed::plan(&rows, true).cost(&rows, HIDDEN);
    assert_eq!(
        cost,
        FeedCost {
            allocations: 0,
            zero_fills: 0,
            copies: 0,
            copied_bytes: 0,
            temp_bytes: 0
        }
    );
}

#[test]
fn text_only_feed_costs_nothing() {
    let cost = ImageFeed::plan(&[], true).cost(&[], HIDDEN);
    assert_eq!(cost.allocations + cost.copies + cost.temp_bytes, 0);
}

#[test]
fn packed_feed_costs_one_allocation_and_a_copy_per_reference() {
    let rows = [20, 7, 20];
    let cost = ImageFeed::plan(&rows, true).cost(&rows, HIDDEN);
    assert_eq!(cost.allocations, 1);
    assert_eq!(cost.zero_fills, 1);
    assert_eq!(cost.copies, 3);
    assert_eq!(cost.copied_bytes, 47 * HIDDEN * 2);
    assert_eq!(cost.temp_bytes, 47 * HIDDEN * 2);
}

#[test]
fn a_single_packed_reference_still_pays_for_the_copy() {
    let rows = [20];
    let cost = ImageFeed::plan(&rows, false).cost(&rows, HIDDEN);
    assert_eq!(cost.allocations, 1);
    assert_eq!(cost.copies, 1);
    assert_eq!(cost.temp_bytes, 20 * HIDDEN * 2);
}

fn gpu_enabled() -> bool {
    std::env::var("SCRIPTOR_GPU_TESTS").as_deref() == Ok("1")
}

pub(crate) fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SCRIPTOR_SNAPSHOT_DIR") {
        return p.into();
    }
    let hf = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".cache/huggingface")
        });
    let snaps = hf.join("hub/models--zai-org--GLM-OCR/snapshots");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .expect("snapshot cache")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("config.json").exists() && p.join("tokenizer.json").exists())
        .collect();
    dirs.sort();
    dirs.into_iter().next().expect("a snapshot")
}

fn fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/image")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

const BLOCK: usize = 16;
const KV_BLOCKS: usize = 512;
const BLOCKS_PER_SEQ: usize = 64;
const MAX_SEQS: usize = 8;

/// A real executor over the checkpoint, the real tokenizer and chat template, and fixture images.
struct Rig {
    exec: GpuExecutor,
    builder: PromptBuilder,
    image_token: u32,
    vocab: usize,
    next_seq: u64,
    next_block: u32,
}

/// One preprocessed fixture image.
struct Image {
    job: VisionJob,
    grid: ImageGrid,
}

/// Generated tokens each request in the deferred-logits cases runs to.
const GENERATED: usize = 8;
/// Positions past a sequence's length its KV table covers for draft lookahead.
const LOOKAHEAD: usize = 8;

/// A request that decodes on the device step.
struct Request {
    id: u64,
    prompt: Vec<u32>,
    blocks: Vec<u32>,
    out: Vec<u32>,
    masks: TokenMasks,
}

impl Request {
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

    fn done(&self) -> bool {
        self.out.len() >= GENERATED
    }
}

impl Rig {
    fn new() -> Self {
        assert!(gpu_enabled(), "set SCRIPTOR_GPU_TESTS=1 to run GPU tests");
        let dir = snapshot();
        let read = |n: &str| std::fs::read_to_string(dir.join(n)).expect(n);
        let config = HfConfig::from_json(&read("config.json")).expect("config");
        let generation = GenerationConfig::from_json(&read("generation_config.json")).ok();
        let spec = ModelSpec::from_hf(&config, generation.as_ref()).expect("spec");
        let files = SafetensorsDir::open(&dir).expect("safetensors");
        let device = Device::open().expect("gpu");
        let stream = device.new_stream().expect("stream");
        let weights = GpuWeights::load(&stream, &files, &spec, true).expect("weights");
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
            kv_blocks: KV_BLOCKS as u64,
            blocks_per_seq: BLOCKS_PER_SEQ as u64,
            full_context_seqs: MAX_SEQS as u64,
            max_context: 0,
            slots: MAX_SEQS,
            kv_block_tokens: BLOCK,
            vision_batch_patches: 2048,
        };
        let exec = GpuExecutor::new(
            &device,
            &stream,
            weights,
            &spec,
            &plan,
            1,
            TuningTable::default(),
            StepLimits {
                max_seqs: MAX_SEQS,
                max_context: BLOCK * BLOCKS_PER_SEQ,
            },
        )
        .expect("executor");
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
        let template =
            ChatTemplate::from_file(dir.join("chat_template.jinja")).expect("chat template");
        let name = tokenizer
            .id_token(spec.image_token_id)
            .expect("image token name");
        let builder =
            PromptBuilder::new(tokenizer, template, &name, MERGE_SIZE as usize).expect("builder");
        Self {
            exec,
            builder,
            image_token: spec.image_token_id,
            vocab: spec.text.vocab_size,
            next_seq: 1,
            next_block: 1,
        }
    }

    fn image(&mut self, name: &str) -> Image {
        let pre = preprocess(&fixture(name), &SmartResizeConfig::default()).expect(name);
        let mut bytes = Vec::with_capacity(pre.pixels.len() * 2);
        for p in &pre.pixels {
            bytes.extend_from_slice(&p.to_bits().to_le_bytes());
        }
        let g = pre.grid_thw;
        let key = embedding_key(
            &bytes,
            CacheGrid {
                t: g.t,
                h: g.h,
                w: g.w,
            },
        );
        let seq = self.next_seq;
        self.next_seq += 1;
        Image {
            job: VisionJob {
                seq,
                image: 0,
                key,
                grid: CacheGrid {
                    t: g.t,
                    h: g.h,
                    w: g.w,
                },
                patches: pre.pixels.len() / scriptor_image::PATCH_ELEMS,
                payload: Arc::new(bytes),
            },
            grid: ImageGrid {
                t: g.t as usize,
                h: g.h as usize,
                w: g.w as usize,
            },
        }
    }

    fn embed(&mut self, image: &Image) -> Arc<GpuEmbedding> {
        let mut out = self
            .exec
            .vision(std::slice::from_ref(&image.job))
            .expect("vision");
        Arc::new(out.remove(0))
    }

    /// The prompt of a chat message holding `images` then the OCR instruction.
    fn prompt(&self, images: &[&Image]) -> Vec<u32> {
        let mut parts: Vec<ContentPart> = images
            .iter()
            .map(|_| ContentPart::image_url("fixture", None))
            .collect();
        parts.push(ContentPart::text("Text Recognition:"));
        let messages = [ChatMessage {
            role: "user".into(),
            content: Some(Content::Parts(parts)),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let grids: Vec<ImageGrid> = images.iter().map(|i| i.grid).collect();
        let options = RenderOptions {
            add_generation_prompt: true,
            ..RenderOptions::default()
        };
        self.builder
            .build(&messages, &options, &grids)
            .expect("prompt")
            .token_ids
    }

    /// Real tokenizer ids of `n` words of public-domain prose.
    fn words(&self, n: usize) -> Vec<u32> {
        let text = "It was the best of times, it was the worst of times, it was the age of \
                    wisdom, it was the age of foolishness, it was the epoch of belief, it was \
                    the epoch of incredulity, it was the season of Light, it was the season of \
                    Darkness, it was the spring of hope, it was the winter of despair.";
        let ids = self.builder.tokenizer().encode(text).expect("encode");
        assert!(ids.len() >= n);
        ids[..n].to_vec()
    }

    fn row(
        &mut self,
        tokens: Vec<u32>,
        images: Vec<Arc<GpuEmbedding>>,
    ) -> PrefillRow<GpuEmbedding> {
        self.row_with_logits(tokens, images, true)
    }

    /// A prefill row over the whole prompt; `logits` selects whether the row returns host logits.
    fn row_with_logits(
        &mut self,
        tokens: Vec<u32>,
        images: Vec<Arc<GpuEmbedding>>,
        logits: bool,
    ) -> PrefillRow<GpuEmbedding> {
        let seq = self.next_seq;
        self.next_seq += 1;
        let need = tokens.len().div_ceil(BLOCK);
        let blocks = (0..need).map(|i| self.next_block + i as u32).collect();
        self.next_block += need as u32;
        PrefillRow {
            seq,
            tokens,
            images,
            blocks,
            logits,
        }
    }

    /// A generating request over `prompt`: its KV table covers the prompt and `GENERATED`
    /// tokens with draft lookahead, and its sampling masks hold the prompt.
    fn request(&mut self, prompt: Vec<u32>) -> Request {
        let id = self.next_seq;
        self.next_seq += 1;
        let need = (prompt.len() + GENERATED + LOOKAHEAD).div_ceil(BLOCK);
        let blocks = (0..need).map(|i| self.next_block + i as u32).collect();
        self.next_block += need as u32;
        let mut masks = TokenMasks::new(1, self.vocab);
        masks.reset_slot(0, &prompt);
        Request {
            id,
            prompt,
            blocks,
            out: Vec::new(),
            masks,
        }
    }

    /// Prefills the whole prompt of `req` with no host logits, as the image-bearing first
    /// chunk of a request that continues on the device step, and returns the row's logits.
    fn prefill_deferred(
        &mut self,
        req: &Request,
        image: Arc<GpuEmbedding>,
        direct: bool,
    ) -> Vec<f32> {
        let rows = [PrefillRow {
            seq: req.id,
            tokens: req.prompt.clone(),
            images: vec![image],
            blocks: req.blocks.clone(),
            logits: false,
        }];
        let mut out = self.exec.prefill_images(&rows, direct).expect("prefill");
        out.remove(0)
    }

    /// One device-resident step over `reqs`: each request appends its emitted tokens.
    fn device_step(&mut self, reqs: &mut [&mut Request]) {
        let rows: Vec<DecodeRow> = reqs.iter().map(|r| r.row()).collect();
        let device_rows: Vec<DeviceRow<'_>> = reqs
            .iter()
            .zip(&rows)
            .map(|(r, row)| DeviceRow {
                row,
                params: r.params(),
                masks: &r.masks,
            })
            .collect();
        let results = self
            .exec
            .decode_device(&device_rows)
            .expect("decode_device");
        drop(device_rows);
        for (r, res) in reqs.iter_mut().zip(&results) {
            assert!(!res.tokens.is_empty() && res.tokens.len() == res.margins.len());
            for &t in &res.tokens {
                r.push(t);
            }
        }
    }

    /// Runs the packed and in-place prefills on `rows` and requires identical logits, hidden
    /// states and untouched source embeddings.
    fn check(&mut self, label: &str, rows: &[PrefillRow<GpuEmbedding>]) {
        let before: Vec<Vec<u16>> = rows
            .iter()
            .flat_map(|r| r.images.iter())
            .map(|e| self.exec.embedding_bits(e).expect("bits"))
            .collect();
        let old = self
            .exec
            .prefill_images(rows, false)
            .expect("packed prefill");
        let old_hidden = self.exec.kept_hidden_bits(rows[0].seq).expect("hidden");
        let new = self
            .exec
            .prefill_images(rows, true)
            .expect("direct prefill");
        let new_hidden = self.exec.kept_hidden_bits(rows[0].seq).expect("hidden");
        assert_eq!(old.len(), new.len(), "{label}: row count");
        for (i, (a, b)) in old.iter().zip(&new).enumerate() {
            assert_eq!(a.len(), b.len(), "{label}: row {i} logits length");
            assert!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{label}: row {i} logits differ"
            );
        }
        assert!(old_hidden.is_some(), "{label}: hidden was not kept");
        assert!(old_hidden == new_hidden, "{label}: hidden states differ");
        let after: Vec<Vec<u16>> = rows
            .iter()
            .flat_map(|r| r.images.iter())
            .map(|e| self.exec.embedding_bits(e).expect("bits"))
            .collect();
        assert!(before == after, "{label}: a source embedding changed");
    }
}

#[test]
#[ignore = "needs GPU and the real checkpoint"]
fn single_image_in_place_matches_packed() {
    let mut rig = Rig::new();
    let img = rig.image("w140_h112.jpg");
    let emb = rig.embed(&img);
    let tokens = rig.prompt(&[&img]);
    let rows = [rig.row(tokens, vec![emb])];
    rig.check("single image", &rows);
}

#[test]
#[ignore = "needs GPU and the real checkpoint"]
fn repeated_image_matches_packed() {
    let mut rig = Rig::new();
    let img = rig.image("w140_h112.jpg");
    let emb = rig.embed(&img);
    let tokens = rig.prompt(&[&img, &img]);
    let rows = [rig.row(tokens, vec![Arc::clone(&emb), emb])];
    rig.check("same image twice", &rows);
}

#[test]
#[ignore = "needs GPU and the real checkpoint"]
fn two_different_images_match_packed() {
    let mut rig = Rig::new();
    let a = rig.image("w140_h112.jpg");
    let b = rig.image("w28_h28.jpg");
    let (ea, eb) = (rig.embed(&a), rig.embed(&b));
    let tokens = rig.prompt(&[&a, &b]);
    let rows = [rig.row(tokens, vec![ea, eb])];
    rig.check("two images", &rows);
}

#[test]
#[ignore = "needs GPU and the real checkpoint"]
fn requests_packed_in_one_prefill_match() {
    let mut rig = Rig::new();
    let a = rig.image("w140_h112.jpg");
    let b = rig.image("w28_h28.jpg");
    let ea = rig.embed(&a);
    let eb = rig.embed(&b);
    let ta = rig.prompt(&[&a]);
    let tb = rig.prompt(&[&b]);
    let text = rig.words(12);
    let ta2 = ta.clone();
    let rows = [
        rig.row(ta, vec![Arc::clone(&ea)]),
        rig.row(text, Vec::new()),
        rig.row(tb, vec![eb]),
        rig.row(ta2, vec![ea]),
    ];
    rig.check("packed requests", &rows);
}

#[test]
#[ignore = "needs GPU and the real checkpoint"]
fn lone_image_request_beside_text_requests_matches() {
    let mut rig = Rig::new();
    let a = rig.image("w140_h112.jpg");
    let ea = rig.embed(&a);
    let (t1, t2) = (rig.words(9), rig.words(20));
    let ta = rig.prompt(&[&a]);
    let rows = [
        rig.row(t1, Vec::new()),
        rig.row(ta, vec![ea]),
        rig.row(t2, Vec::new()),
    ];
    rig.check("image between text rows", &rows);
}

#[test]
#[ignore = "needs GPU and the real checkpoint"]
fn text_only_matches() {
    let mut rig = Rig::new();
    let t = rig.words(30);
    let rows = [rig.row(t, Vec::new())];
    rig.check("text only", &rows);
}

/// Prompts whose image run starts at the first token, ends at the last token, and whose length
/// lands exactly on and just past a KV block boundary.
fn boundary_rows(rig: &mut Rig, emb: &Arc<GpuEmbedding>) -> Vec<Vec<u32>> {
    let run = emb.row_count();
    let image = rig.image_token;
    let mut out = Vec::new();
    let tail = rig.words(5);
    let mut starts_with_image = vec![image; run];
    starts_with_image.extend(&tail);
    out.push(starts_with_image);
    let mut ends_with_image = rig.words(7);
    ends_with_image.extend(std::iter::repeat_n(image, run));
    out.push(ends_with_image);
    for total in [2 * BLOCK, 2 * BLOCK + 1, 2 * BLOCK - 1] {
        let lead = total - run - tail.len();
        let mut t = rig.words(lead);
        t.extend(std::iter::repeat_n(image, run));
        t.extend(&tail);
        assert_eq!(t.len(), total);
        out.push(t);
    }
    out
}

#[test]
#[ignore = "needs GPU and the real checkpoint"]
fn image_runs_at_row_and_block_boundaries_match() {
    let mut rig = Rig::new();
    let img = rig.image("w140_h112.jpg");
    let emb = rig.embed(&img);
    for (i, tokens) in boundary_rows(&mut rig, &emb).into_iter().enumerate() {
        let rows = [rig.row(tokens, vec![Arc::clone(&emb)])];
        rig.check(&format!("boundary case {i}"), &rows);
    }
}

#[test]
#[ignore = "needs GPU and the real checkpoint"]
fn adjacent_image_runs_across_packed_rows_match() {
    let mut rig = Rig::new();
    let img = rig.image("w140_h112.jpg");
    let emb = rig.embed(&img);
    let cases = boundary_rows(&mut rig, &emb);
    let rows = [
        rig.row(cases[1].clone(), vec![Arc::clone(&emb)]),
        rig.row(cases[0].clone(), vec![Arc::clone(&emb)]),
    ];
    rig.check("image run ends one row and starts the next", &rows);
}

/// What a deferred-logits scenario produced once its surviving request completed.
#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    /// Tokens of every request that ran to completion, in request order.
    tokens: Vec<Vec<u32>>,
    /// The device step's buffers after the last step.
    state: StateBits,
    /// The shared embedding's words after every request finished or was released.
    source: Vec<u16>,
}

/// How the first of two requests sharing one image embedding ends.
#[derive(Clone, Copy)]
enum FirstEnds {
    /// Both run to completion.
    Completes,
    /// It is released after this many device steps while the second keeps generating.
    ReleasedAfter(usize),
    /// It is released right after its prefill, before any decode step.
    CancelledBeforeDecode,
}

/// Two requests share `emb`. Each prefills its whole prompt (the image run followed by the final
/// text tokens) with `logits: false` through the feed selected by `direct`, then generates on
/// the device step. The embedding is read back only after the survivors complete.
fn run_shared(
    rig: &mut Rig,
    emb: &Arc<GpuEmbedding>,
    prompt: &[u32],
    direct: bool,
    first_ends: FirstEnds,
) -> Outcome {
    let mut first = rig.request(prompt.to_vec());
    let mut second = rig.request(prompt.to_vec());
    let holder_first = Arc::clone(emb);
    let holder_second = Arc::clone(emb);
    for req in [&first, &second] {
        let logits = rig.prefill_deferred(req, Arc::clone(emb), direct);
        assert!(logits.is_empty(), "a logits-free row returned logits");
    }
    let mut tokens = Vec::new();
    match first_ends {
        FirstEnds::Completes => {
            while !(first.done() && second.done()) {
                rig.device_step(&mut [&mut first, &mut second]);
            }
            tokens.push(first.out.clone());
            rig.exec.release(first.id);
            drop(holder_first);
        }
        FirstEnds::ReleasedAfter(steps) => {
            for _ in 0..steps {
                rig.device_step(&mut [&mut first, &mut second]);
            }
            rig.exec.release(first.id);
            drop(holder_first);
            assert_eq!(
                Arc::strong_count(emb),
                2,
                "the embedding stays held by the test and the other request"
            );
            while !second.done() {
                rig.device_step(&mut [&mut second]);
            }
        }
        FirstEnds::CancelledBeforeDecode => {
            rig.exec.release(first.id);
            drop(holder_first);
            while !second.done() {
                rig.device_step(&mut [&mut second]);
            }
        }
    }
    tokens.push(second.out.clone());
    rig.exec.release(second.id);
    drop(holder_second);
    let state = rig.exec.step_bits().expect("step state");
    let source = rig.exec.embedding_bits(emb).expect("source bits");
    Outcome {
        tokens,
        state,
        source,
    }
}

/// Runs the scenario through the packed copy and then the in-place feed on one rig and requires
/// identical tokens, device-step state and source embedding.
fn check_shared(label: &str, first_ends: FirstEnds) {
    let mut rig = Rig::new();
    let img = rig.image("w140_h112.jpg");
    let emb = rig.embed(&img);
    let prompt = rig.prompt(&[&img]);
    let run = prompt.iter().filter(|&&t| t == rig.image_token).count();
    assert_eq!(run, emb.row_count(), "{label}: the image run is whole");
    assert!(
        prompt.last() != Some(&rig.image_token),
        "{label}: final text tokens follow the image run"
    );
    let packed = run_shared(&mut rig, &emb, &prompt, false, first_ends);
    let direct = run_shared(&mut rig, &emb, &prompt, true, first_ends);
    assert_eq!(packed.tokens, direct.tokens, "{label}: tokens differ");
    assert!(
        packed.state == direct.state,
        "{label}: device-step state differs"
    );
    assert!(packed.source == direct.source, "{label}: source changed");
}

#[test]
#[ignore = "needs GPU and the real checkpoint"]
fn logits_free_image_prompt_shared_by_two_requests_matches_packed() {
    check_shared("both complete", FirstEnds::Completes);
}

#[test]
#[ignore = "needs GPU and the real checkpoint"]
fn releasing_one_sharing_request_mid_decode_leaves_the_other_matching_packed() {
    check_shared("released mid-decode", FirstEnds::ReleasedAfter(3));
}

#[test]
#[ignore = "needs GPU and the real checkpoint"]
fn cancelling_one_sharing_request_before_decode_leaves_the_other_matching_packed() {
    check_shared("cancelled before decode", FirstEnds::CancelledBeforeDecode);
}

/// Alternating packed/in-place prefill pairs per feed in the cost test.
const COST_PAIRS: usize = 7;

#[test]
#[ignore = "needs GPU and the real checkpoint"]
fn prefill_cost_separates_cold_vision_cache_hit_and_text_prefill() {
    let mut rig = Rig::new();
    let img = rig.image("w448_h420.jpg");
    // Cold vision (embedding cache miss) is timed once and kept apart from the prefill pairs.
    let cold = Instant::now();
    let emb = rig.embed(&img);
    let cold_vision = cold.elapsed();
    let tokens = rig.prompt(&[&img]);
    let rows = [rig.row(tokens, vec![Arc::clone(&emb)])];
    let source_rows = [emb.row_count()];
    // The embedding cache hit reuses `emb`: no vision work precedes either prefill below.
    // Both feeds are warmed before any timing.
    for direct in [false, true] {
        rig.exec.prefill_images(&rows, direct).expect("warm up");
    }
    rig.exec.context().synchronize().expect("sync");
    // Whole-prefill time: includes the logits download and an explicit context sync. Pairs run
    // in alternating order over the same rows, so both feeds see the same prompt and KV state.
    let mut times: [Vec<std::time::Duration>; 2] = [Vec::new(), Vec::new()];
    for pair in 0..COST_PAIRS {
        let order = if pair % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for direct in order {
            let start = Instant::now();
            rig.exec.prefill_images(&rows, direct).expect("prefill");
            rig.exec.context().synchronize().expect("sync");
            times[usize::from(direct)].push(start.elapsed());
        }
    }
    println!("cold vision (embedding cache miss, separate): {cold_vision:?}");
    let mut copies = [0usize; 2];
    let mut temp = [0usize; 2];
    for (i, name) in ["packed copy", "in place"].into_iter().enumerate() {
        let feed = ImageFeed::plan(&source_rows, i == 1);
        let cost = feed.cost(&source_rows, emb.width());
        copies[i] = cost.copies;
        temp[i] = cost.temp_bytes;
        times[i].sort();
        println!(
            "cache-hit whole prefill incl. logits download and context sync, {name}: median {:?} over {COST_PAIRS} alternating pairs",
            times[i][COST_PAIRS / 2]
        );
        println!(
            "  FeedCost (source-derived expected work, not allocator or DRAM measurements): feed {feed:?}, allocations {}, zero fills {}, copies {}, copied bytes {}, temporary bytes {}",
            cost.allocations, cost.zero_fills, cost.copies, cost.copied_bytes, cost.temp_bytes
        );
    }
    assert_eq!(temp[1], 0);
    assert_eq!(copies[0], 1);
}
