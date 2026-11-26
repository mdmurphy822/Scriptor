//! Losslessness of the device-resident verify pass against scrambled drafts on a real GLM-OCR
//! checkpoint.
//!
//! For a fixed step shape (draft depth `k`, bucket `B`), the bf16 verification-logit row, the
//! hidden row, the target token and the margin of the row that emitted each committed token
//! are compared bit for bit between a reference run driven by the MTP head's own drafts and
//! runs whose drafts are overwritten before every step (random ids, permuted ids, and forced
//! accepted lengths `m` per row), with padding rows inside the bucket in some runs. No
//! tolerance is applied anywhere.
//!
//! Each test runs one `(k, B)` shape on its own executor with `max_context` 1024, so the
//! decode attention splits a 700-token and a 300-token context into several chunks. Run with
//! `SCRIPTOR_GPU_TESTS=1 SCRIPTOR_MODEL_DIR=<snapshot> HF_HUB_OFFLINE=1 cargo test -p scriptor-cli --features cuda --profile gpu-test scrambled_drafts_k3_b32 -- --ignored --nocapture --test-threads=1`.
//! The `_part_a` and `_part_b` tests of a shape split its arms into two disjoint halves, each
//! recomputing the reference run.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use scriptor_chat::Tokenizer;
use scriptor_cuda::Device;
use scriptor_engine::ModelExecutor;
use scriptor_kernels::attention::{choose_splits, split_chunk};
use scriptor_spec::{ModelSpec, graph_buckets, step_shape};
use scriptor_weights::SafetensorsDir;

use super::real_fixture_dump::TEXTS;
use super::split_graph::{Mode, step_subset};
use super::{BLOCK, Seq, Sizing, TOKENS, defer, executor_sized, gpu_enabled};
use crate::gpu_executor::{GpuExecutor, GpuWeights};

/// Device slots of the executor under test.
const SEQS: usize = 32;
/// Longest sequence the executor serves; fixes the attention split count.
const MAX_CONTEXT: usize = 1024;
/// Prompt tokens of the two longest sequences.
const LONG_PROMPT: usize = 700;
/// Prompt tokens of the two mid-length sequences.
const MID_PROMPT: usize = 300;
/// Absolute position from which a keyed row lies beyond the first split chunk of a bucket-8 step.
const HIGH_POS: usize = 576;
/// Absolute position from which a keyed row lies beyond the first split chunk of a bucket-1 step.
const MID_POS: usize = 128;
/// Prompt tokens prefilled per executor call at most.
const PREFILL_GROUP_TOKENS: usize = 2048;
/// Times each accepted length must occur in a ramp arm.
const RAMP_MIN: u32 = 4;
/// Keyed sequences whose raw bf16 logit rows are kept for mismatch reports.
const RAW_SEQS: usize = 2;
/// Mismatches printed per comparison.
const REPORT_LIMIT: usize = 8;
/// Key-value heads of the text decoder.
const TEXT_KV_HEADS: usize = 8;

/// Fixed accepted-length plans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Forced {
    /// `(slot + step) mod (k + 1)` accepted drafts per row.
    Ramp,
    /// No draft accepted.
    Zero,
    /// Every draft accepted.
    Full,
    /// Even slots accept none, odd slots accept all.
    Alternate,
    /// One rotating row accepts all, the rest none.
    OneHot,
    /// Uniformly random accepted length per row.
    Mixed,
}

/// How the drafts of each step are produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Plan {
    /// The MTP head's own drafts.
    Normal,
    /// Uniformly random token ids.
    Random,
    /// The reference run's drafts of the same step, shuffled across rows and positions.
    Permuted,
    /// Drafts built to be accepted exactly `m` times.
    Forced(Forced),
}

/// Declares [`Arm`] together with the list of every variant.
macro_rules! arms {
    ($(#[$enum_meta:meta])* enum Arm { $($(#[$meta:meta])* $name:ident,)* }) => {
        $(#[$enum_meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Arm {
            $($(#[$meta])* $name,)*
        }

        impl Arm {
            /// Every variant in declaration order.
            const EVERY: &'static [Arm] = &[$(Arm::$name,)*];
        }
    };
}

arms! {
    /// One run of a shape.
    enum Arm {
        /// The reference: the MTP head's own drafts, run until every stream passes the longest
        /// accepted window of any plan.
        Ref,
        /// A second reference run in the same process.
        Repeat,
        Random,
        Permuted,
        Ramp,
        RampSparse,
        Zero,
        Full,
        Alternate,
        OneHot,
        Mixed,
        MixedSparse,
    }
}

/// Arms of the first half of a shape.
const PART_A: [Arm; 3] = [Arm::Repeat, Arm::Random, Arm::Permuted];
/// Arms of the second half of a shape.
const PART_B: [Arm; 8] = [
    Arm::Ramp,
    Arm::RampSparse,
    Arm::Zero,
    Arm::Full,
    Arm::Alternate,
    Arm::OneHot,
    Arm::Mixed,
    Arm::MixedSparse,
];

/// Every arm after the reference, in run order.
fn all_arms() -> Vec<Arm> {
    PART_A.iter().chain(&PART_B).copied().collect()
}

impl Arm {
    fn plan(self) -> Plan {
        match self {
            Self::Ref | Self::Repeat => Plan::Normal,
            Self::Random => Plan::Random,
            Self::Permuted => Plan::Permuted,
            Self::Ramp | Self::RampSparse => Plan::Forced(Forced::Ramp),
            Self::Zero => Plan::Forced(Forced::Zero),
            Self::Full => Plan::Forced(Forced::Full),
            Self::Alternate => Plan::Forced(Forced::Alternate),
            Self::OneHot => Plan::Forced(Forced::OneHot),
            Self::Mixed | Self::MixedSparse => Plan::Forced(Forced::Mixed),
        }
    }

    /// Whether three sequences leave the bucket after the first step.
    fn sparse(self) -> bool {
        matches!(self, Self::RampSparse | Self::MixedSparse)
    }

    /// Committed tokens each sequence must reach before the run ends.
    fn stop(self, k: usize) -> usize {
        match self.plan() {
            Plan::Normal => TOKENS + k + 1,
            _ => TOKENS,
        }
    }
}

/// Deterministic 64-bit generator (splitmix64).
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Seed of the generator of one arm of a shape.
fn arm_seed(arm: Arm, k: usize, bucket: usize) -> u64 {
    0x5ced_0000 ^ ((arm as u64) << 24) ^ ((k as u64) << 8) ^ bucket as u64
}

/// Prompt tokens of sequence `i`.
fn prompt_len(i: usize) -> usize {
    match i {
        0 | 1 => LONG_PROMPT,
        2 | 3 => MID_PROMPT,
        _ => 40 + (i * 7) % 80,
    }
}

/// Tokens of KV capacity sequence `i` needs at draft depth `k`: its prompt, the largest growth
/// of a run, the lookahead of a step and a margin.
fn capacity_tokens(i: usize, k: usize) -> usize {
    prompt_len(i) + (TOKENS + k + 1) * (k + 1) + 2 * (k + 1) + 16
}

/// Consecutive block ids, from 1, covering each sequence's capacity.
fn block_layout(k: usize) -> Vec<Vec<u32>> {
    let mut next = 1u32;
    (0..SEQS)
        .map(|i| {
            let n = capacity_tokens(i, k).div_ceil(BLOCK);
            let blocks: Vec<u32> = (next..next + n as u32).collect();
            next += n as u32;
            blocks
        })
        .collect()
}

/// Slots whose sequences leave a sparse arm's bucket after its first step; the highest slot
/// stays occupied.
fn hole_slots(bucket: usize) -> Vec<usize> {
    vec![bucket - 4, bucket - 3, bucket - 2]
}

/// Accepted drafts the forced plan `kind` asks of the row in `slot` at `step`; `hot` is the
/// rotating row of the one-hot plan.
fn forced_m(kind: Forced, k: usize, slot: usize, step: usize, hot: usize, rng: &mut Mix) -> usize {
    match kind {
        Forced::Ramp => (slot + step) % (k + 1),
        Forced::Zero => 0,
        Forced::Full => k,
        Forced::Alternate => {
            if slot.is_multiple_of(2) {
                0
            } else {
                k
            }
        }
        Forced::OneHot => {
            if slot == hot {
                k
            } else {
                0
            }
        }
        Forced::Mixed => rng.below(k + 1),
    }
}

/// Whether a draft window of `k` built for `m` accepted drafts at committed count `c` lies
/// entirely inside a reference stream of `len` tokens.
fn covered(len: usize, c: usize, m: usize, k: usize) -> bool {
    if m == k { c + m <= len } else { c + m < len }
}

/// Drafts that a verifier following `reference` accepts exactly `m` times from committed count
/// `c`: the next `m` reference tokens, then a token that differs from the next one, then random
/// ids. Positions beyond the reference are random.
fn forced_drafts(
    k: usize,
    vocab: usize,
    reference: &[u32],
    c: usize,
    m: usize,
    rng: &mut Mix,
) -> Vec<u32> {
    (0..k)
        .map(|j| match reference.get(c + j) {
            Some(&t) if j < m => t,
            Some(&t) if j == m => ((t as usize + 1) % vocab) as u32,
            _ => rng.below(vocab) as u32,
        })
        .collect()
}

fn random_drafts(k: usize, vocab: usize, rng: &mut Mix) -> Vec<u32> {
    (0..k).map(|_| rng.below(vocab) as u32).collect()
}

/// Fisher-Yates shuffle of `pool`.
fn permute(pool: &mut [u32], rng: &mut Mix) {
    for i in (1..pool.len()).rev() {
        pool.swap(i, rng.below(i + 1));
    }
}

/// Committed token indices below `TOKENS` a step emits when `c` tokens were committed before it
/// and it emitted `e`.
fn keyed_range(c: usize, e: usize) -> std::ops::Range<usize> {
    c.min(TOKENS)..(c + e).min(TOKENS)
}

/// Occurrences of each accepted length.
struct Histogram(Vec<u32>);

impl Histogram {
    fn new(k: usize) -> Self {
        Self(vec![0; k + 1])
    }

    fn observe(&mut self, m: usize) {
        self.0[m] += 1;
    }

    /// Accepted lengths seen fewer than `min` times.
    fn short(&self, min: u32) -> Vec<usize> {
        (0..self.0.len()).filter(|&m| self.0[m] < min).collect()
    }
}

/// One step of the 64-bit mixer shared by every digest.
fn fold(h: u64, w: u64) -> u64 {
    let h = (h ^ w).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^ (h >> 29)
}

/// 64-bit digest of a run of bf16 words; any single-bit change alters it.
fn hash_bits(bits: &[u16]) -> u64 {
    let mut h = 0x243f_6a88_85a3_08d3u64 ^ bits.len() as u64;
    for chunk in bits.chunks(4) {
        let w = chunk
            .iter()
            .enumerate()
            .fold(0u64, |w, (n, &b)| w | (u64::from(b) << (16 * n)));
        h = fold(h, w);
    }
    h
}

/// Distance in bf16 steps between two words.
fn ulp_distance(a: u16, b: u16) -> u32 {
    let key = |v: u16| {
        let bits = i32::from(v);
        if bits & 0x8000 != 0 {
            -(bits & 0x7fff)
        } else {
            bits
        }
    };
    key(a).abs_diff(key(b))
}

/// Split count and keys per split of the decode attention of a step of `bucket` sequences at
/// draft depth `k` on a device with `sm_count` multiprocessors.
fn attention_plan(k: usize, bucket: usize, sm_count: usize) -> (usize, usize) {
    let stride = (MAX_CONTEXT + 2 * k).div_ceil(BLOCK) + 1;
    let max_ctx = stride * BLOCK;
    let splits = choose_splits(max_ctx, bucket, TEXT_KV_HEADS, sm_count);
    (splits, split_chunk(max_ctx, splits))
}

/// What one verification row produced.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RowKey {
    logits: u64,
    hidden: u64,
    target: u32,
    margin: u32,
}

/// Everything one run recorded.
struct RunOut {
    /// Committed tokens per sequence index.
    streams: Vec<Vec<u32>>,
    /// Verification row of each `(sequence, committed index)` below `TOKENS`.
    keys: HashMap<(usize, usize), RowKey>,
    /// Raw logit rows of the first `RAW_SEQS` sequences' keys.
    raw: HashMap<(usize, usize), Vec<u16>>,
    /// Accepted lengths of rows that emitted a token below `TOKENS`.
    hist: Histogram,
    /// Keyed rows at an absolute position of at least `HIGH_POS`.
    high_rows: u32,
    /// Keyed rows at an absolute position of at least `MID_POS`.
    mid_rows: u32,
    /// Keyed rows lying beyond the first attention split chunk of the step shape.
    split_rows: u32,
    /// Drafts of every step as the device step read them, `[slot * k + j]`.
    step_drafts: Vec<Vec<u32>>,
    /// Rows whose accepted length differed from the forced plan.
    plan_misses: Vec<String>,
    /// First step at which a forced run left the reference stream.
    diverged: Option<String>,
    /// Sequences that left the bucket after the first step.
    holes: Vec<usize>,
    steps: usize,
    /// Steps whose drafts were overwritten.
    patched_steps: usize,
    /// Steps of a scrambling arm that ran on the engine's own drafts.
    unpatched_steps: usize,
    /// Wall time spent inside the device steps.
    step_time: Duration,
    /// Wall time spent reading the verification tap and the step bits back.
    tap_time: Duration,
    /// Wall time spent hashing verification rows.
    hash_time: Duration,
}

impl RunOut {
    fn new(k: usize, holes: Vec<usize>) -> Self {
        Self {
            streams: Vec::new(),
            keys: HashMap::new(),
            raw: HashMap::new(),
            hist: Histogram::new(k),
            high_rows: 0,
            mid_rows: 0,
            split_rows: 0,
            step_drafts: Vec::new(),
            plan_misses: Vec::new(),
            diverged: None,
            holes,
            steps: 0,
            patched_steps: 0,
            unpatched_steps: 0,
            step_time: Duration::ZERO,
            tap_time: Duration::ZERO,
            hash_time: Duration::ZERO,
        }
    }
}

/// Digests of the first `TOKENS` committed tokens and of the verification rows of the
/// sequences `seqs`, folded in ascending `(sequence, index)` order so that insertion order
/// does not matter. Equal digests of the same arm from different processes or binaries mean
/// bit-identical streams and rows.
fn run_digest(run: &RunOut, seqs: &[usize]) -> (u64, u64) {
    let mut streams = 0x5eed_0001u64;
    let mut keys = 0x5eed_0002u64;
    for &i in seqs {
        let tokens = &run.streams[i][..run.streams[i].len().min(TOKENS)];
        streams = fold(streams, i as u64);
        streams = fold(streams, tokens.len() as u64);
        for &t in tokens {
            streams = fold(streams, u64::from(t));
        }
        for t in 0..TOKENS {
            keys = fold(keys, ((i as u64) << 32) | t as u64);
            match run.keys.get(&(i, t)) {
                Some(key) => {
                    for w in [
                        key.logits,
                        key.hidden,
                        u64::from(key.target),
                        u64::from(key.margin),
                    ] {
                        keys = fold(keys, w);
                    }
                }
                None => keys = fold(keys, u64::MAX),
            }
        }
    }
    (streams, keys)
}

/// The digest line of one run.
fn digest_line(name: &str, k: usize, bucket: usize, digest: (u64, u64)) -> String {
    format!(
        "digest arm={name} k={k} bucket={bucket} streams={:016x} keys={:016x}",
        digest.0, digest.1
    )
}

/// Prints a phase marker with the seconds elapsed since `started`.
fn harness_phase(started: Instant, name: &str) {
    eprintln!(
        "phase {name} elapsed_s={:.3}",
        started.elapsed().as_secs_f64()
    );
}

/// The loaded checkpoint and the executor of one shape.
struct Harness {
    exec: GpuExecutor,
    _device: Device,
    k: usize,
    q: usize,
    vocab: usize,
    prompts: Vec<Vec<u32>>,
    blocks: Vec<Vec<u32>>,
    sm_count: usize,
    started: Instant,
}

impl Harness {
    /// Loads the checkpoint named by `SCRIPTOR_MODEL_DIR` into an executor of depth `k`.
    fn new(k: usize) -> Self {
        assert!(gpu_enabled(), "set SCRIPTOR_GPU_TESTS=1 to run GPU tests");
        let started = Instant::now();
        let dir = PathBuf::from(
            std::env::var("SCRIPTOR_MODEL_DIR").expect("SCRIPTOR_MODEL_DIR must be set"),
        );
        let config = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
        let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
        let spec = ModelSpec::from_json(&config, generation.as_deref()).expect("model spec");
        let vocab = spec.text.vocab_size;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
        let device = Device::open().expect("gpu");
        let stream = device.new_stream().expect("stream");
        let files = SafetensorsDir::open(&dir).expect("safetensors");
        let weights = GpuWeights::load(&stream, &files, &spec, true).expect("weights");
        let sm_count = device
            .context()
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .expect("multiprocessor count") as usize;
        let blocks = block_layout(k);
        harness_phase(started, "weights");
        let total: usize = blocks.iter().map(Vec::len).sum();
        let widest = blocks.iter().map(Vec::len).max().unwrap_or(1);
        let mut exec = executor_sized(
            &device,
            &stream,
            weights,
            &spec,
            k,
            Sizing {
                seqs: SEQS,
                max_context: MAX_CONTEXT,
                kv_blocks: total + 2,
                blocks_per_seq: widest,
            },
        );
        exec.start_verification_tap(SEQS * (k + 1)).expect("tap");
        let texts: Vec<Vec<u32>> = TEXTS
            .iter()
            .map(|t| tokenizer.encode(t).expect("encode"))
            .collect();
        let prompts = (0..SEQS)
            .map(|i| {
                let base: Vec<u32> = (0..texts.len())
                    .flat_map(|n| texts[(n + i) % texts.len()].iter().copied())
                    .collect();
                (0..prompt_len(i)).map(|n| base[n % base.len()]).collect()
            })
            .collect();
        let harness = Self {
            exec,
            _device: device,
            k,
            q: k + 1,
            vocab,
            prompts,
            blocks,
            sm_count,
            started,
        };
        harness.phase("load");
        harness
    }

    fn phase(&self, name: &str) {
        harness_phase(self.started, name);
    }

    /// Prefills `seqs` without host logits, in groups bounded by `PREFILL_GROUP_TOKENS`.
    fn prefill(&mut self, seqs: &mut [Seq]) {
        let mut start = 0;
        while start < seqs.len() {
            let mut end = start;
            let mut tokens = 0;
            while end < seqs.len()
                && (end == start || tokens + seqs[end].prompt.len() <= PREFILL_GROUP_TOKENS)
            {
                tokens += seqs[end].prompt.len();
                end += 1;
            }
            defer(&mut self.exec, &mut seqs[start..end]);
            start = end;
        }
    }

    /// Drives `bucket` sequences in slots `0..bucket` through one arm and records every
    /// verification row that emitted a token below `TOKENS`.
    fn run(&mut self, arm: Arm, bucket: usize, reference: Option<&RunOut>) -> RunOut {
        let (k, q, vocab) = (self.k, self.q, self.vocab);
        let name = format!("{arm:?}");
        let holes = if arm.sparse() {
            hole_slots(bucket)
        } else {
            Vec::new()
        };
        let mut seqs: Vec<Seq> = (0..bucket)
            .map(|i| {
                let mut s = Seq::new(i, self.prompts[i].clone(), vocab);
                s.blocks = self.blocks[i].clone();
                s
            })
            .collect();
        self.prefill(&mut seqs);
        self.phase(&format!("prefill_{name}"));
        let stop = arm.stop(k);
        let buckets = graph_buckets(SEQS);
        let mut alive = vec![true; bucket];
        let mut rng = Mix(arm_seed(arm, k, bucket));
        let mut last: Vec<u32> = reference
            .and_then(|r| r.step_drafts.last().cloned())
            .unwrap_or_else(|| vec![0; SEQS * k]);
        let mut out = RunOut::new(k, holes.clone());
        let (_, chunk) = attention_plan(k, bucket, self.sm_count);
        let mut step = 0usize;
        loop {
            let active: Vec<usize> = (0..bucket).filter(|&i| alive[i]).collect();
            if step > 0 && active.iter().all(|&i| seqs[i].out.len() >= stop) {
                break;
            }
            assert!(
                step <= stop,
                "{name}: {step} steps without reaching {stop} tokens"
            );
            let before: Vec<usize> = active.iter().map(|&i| seqs[i].out.len()).collect();
            let mut wants: Vec<Option<usize>> = vec![None; active.len()];
            let mut patches: Vec<(u64, Vec<u32>)> = Vec::new();
            // A sequence admitted from a logits-free prefill proposes no drafts on its first
            // step, so only later steps can be overwritten.
            let patchable = step > 0;
            let plan = if patchable { arm.plan() } else { Plan::Normal };
            if arm.plan() != Plan::Normal {
                if patchable {
                    out.patched_steps += 1;
                } else {
                    out.unpatched_steps += 1;
                }
            }
            match plan {
                Plan::Normal => {}
                Plan::Random => {
                    for &i in &active {
                        patches.push((seqs[i].id, random_drafts(k, vocab, &mut rng)));
                    }
                }
                Plan::Permuted => {
                    let source: &[u32] = reference
                        .and_then(|r| r.step_drafts.get(step))
                        .map_or(&last[..], |d| &d[..]);
                    let mut pool: Vec<u32> = active
                        .iter()
                        .flat_map(|&i| source[i * k..(i + 1) * k].iter().copied())
                        .collect();
                    permute(&mut pool, &mut rng);
                    for (r, &i) in active.iter().enumerate() {
                        patches.push((seqs[i].id, pool[r * k..(r + 1) * k].to_vec()));
                    }
                }
                Plan::Forced(kind) => {
                    let reference = reference.expect("a forced arm needs the reference run");
                    let hot = active[step % active.len()];
                    for (r, &i) in active.iter().enumerate() {
                        let m = forced_m(kind, k, i, step, hot, &mut rng);
                        let c = seqs[i].out.len();
                        let stream = &reference.streams[i];
                        patches.push((seqs[i].id, forced_drafts(k, vocab, stream, c, m, &mut rng)));
                        wants[r] = covered(stream.len(), c, m, k).then_some(m);
                    }
                }
            }
            if !patches.is_empty() {
                self.exec.overwrite_drafts(patches);
            }
            let fallbacks = self.exec.graph_stats().fallbacks;
            let timer = Instant::now();
            let results = step_subset(&mut self.exec, &mut seqs, &active, Mode::Greedy);
            out.step_time += timer.elapsed();
            assert_eq!(
                self.exec.graph_stats().fallbacks,
                fallbacks + 1,
                "{name} step {step}: the step did not run eagerly"
            );
            let slots: Vec<usize> = active
                .iter()
                .map(|&i| self.exec.device_slot(seqs[i].id).expect("tracked slot"))
                .collect();
            assert_eq!(
                slots, active,
                "{name} step {step}: sequences left their slots"
            );
            let shape = step_shape(&buckets, slots.iter().copied()).expect("step shape");
            assert_eq!(shape.bucket, bucket, "{name} step {step}: bucket");
            let timer = Instant::now();
            let (logits, drafts) = self.exec.verification_tap(bucket * q).expect("tap");
            let bits = self.exec.step_bits().expect("step bits");
            out.tap_time += timer.elapsed();
            let hidden_width = bits.hidden.len() / (SEQS * q);
            for (r, &i) in active.iter().enumerate() {
                let (c, e) = (before[r], results[r].tokens.len());
                if c < TOKENS && patchable {
                    out.hist.observe(e - 1);
                }
                if let Some(m) = wants[r]
                    && e - 1 != m
                    && out.plan_misses.len() < REPORT_LIMIT
                {
                    out.plan_misses.push(format!(
                        "{name} step {step} seq {i}: forced {m} accepted, got {}",
                        e - 1
                    ));
                }
                for j in 0..e {
                    let row = i * q + j;
                    assert_eq!(
                        results[r].tokens[j], bits.targets[row],
                        "{name} step {step} seq {i} row {j}: emitted token and target row disagree"
                    );
                }
                for t in keyed_range(c, e) {
                    let j = t - c;
                    let row = i * q + j;
                    let logit_row = &logits[row * vocab..(row + 1) * vocab];
                    let timer = Instant::now();
                    let key = RowKey {
                        logits: hash_bits(logit_row),
                        hidden: hash_bits(
                            &bits.hidden[row * hidden_width..(row + 1) * hidden_width],
                        ),
                        target: bits.targets[row],
                        margin: bits.margins[row],
                    };
                    out.hash_time += timer.elapsed();
                    assert!(
                        out.keys.insert((i, t), key).is_none(),
                        "{name}: sequence {i} index {t} emitted twice"
                    );
                    if i < RAW_SEQS {
                        out.raw.insert((i, t), logit_row.to_vec());
                    }
                    let pos = self.prompts[i].len() + t - 1;
                    out.high_rows += u32::from(pos >= HIGH_POS);
                    out.mid_rows += u32::from(pos >= MID_POS);
                    out.split_rows += u32::from(pos >= chunk);
                }
            }
            last.clone_from(&drafts);
            out.step_drafts.push(drafts);
            if let Plan::Forced(_) = arm.plan() {
                let reference = reference.expect("reference");
                for &i in &active {
                    let (a, b) = (&seqs[i].out, &reference.streams[i]);
                    let n = a.len().min(b.len());
                    if let Some(t) = (0..n).find(|&t| a[t] != b[t]) {
                        out.diverged = Some(format!(
                            "{name} step {step} seq {i}: stream leaves the reference at index {t}"
                        ));
                        break;
                    }
                }
            }
            if step == 0 {
                for &h in &holes {
                    self.exec.release(seqs[h].id);
                    alive[h] = false;
                }
            }
            step += 1;
            if out.diverged.is_some() {
                break;
            }
        }
        for (i, s) in seqs.iter().enumerate() {
            if alive[i] {
                self.exec.release(s.id);
            }
        }
        out.streams = seqs.iter().map(|s| s.out.clone()).collect();
        out.steps = step;
        self.phase(&format!("run_{name}"));
        eprintln!(
            "timing arm={name} step_s={:.3} tap_s={:.3} hash_s={:.3}",
            out.step_time.as_secs_f64(),
            out.tap_time.as_secs_f64(),
            out.hash_time.as_secs_f64()
        );
        out
    }
}

/// Mismatches between a run and the reference over the sequences `seqs`.
fn compare_runs(reference: &RunOut, run: &RunOut, seqs: &[usize]) -> Vec<String> {
    let mut report = Vec::new();
    let mut differing = 0usize;
    for &i in seqs {
        let (a, b) = (&reference.streams[i], &run.streams[i]);
        if a.len() < TOKENS || b.len() < TOKENS {
            report.push(format!("seq {i}: a stream is shorter than {TOKENS} tokens"));
            continue;
        }
        if let Some(t) = (0..TOKENS).find(|&t| a[t] != b[t]) {
            report.push(format!(
                "seq {i}: committed stream diverges at index {t}: reference {} run {}",
                a[t], b[t]
            ));
        }
        for t in 0..TOKENS {
            match (reference.keys.get(&(i, t)), run.keys.get(&(i, t))) {
                (Some(x), Some(y)) if x == y => {}
                (Some(x), Some(y)) => {
                    differing += 1;
                    if differing <= REPORT_LIMIT {
                        report.push(describe(reference, run, i, t, x, y));
                    }
                }
                _ => report.push(format!("seq {i} index {t}: a verification row is missing")),
            }
        }
    }
    if differing > 0 {
        report.push(format!("{differing} differing verification rows"));
    }
    report
}

/// The fields of a verification row that differ between two runs.
fn describe(
    reference: &RunOut,
    run: &RunOut,
    i: usize,
    t: usize,
    x: &RowKey,
    y: &RowKey,
) -> String {
    let mut parts = Vec::new();
    if x.logits != y.logits {
        let detail = match (reference.raw.get(&(i, t)), run.raw.get(&(i, t))) {
            (Some(a), Some(b)) => a
                .iter()
                .zip(b)
                .position(|(p, q)| p != q)
                .map(|n| {
                    format!(
                        " first at word {n}: {:#06x} vs {:#06x}, {} ulp",
                        a[n],
                        b[n],
                        ulp_distance(a[n], b[n])
                    )
                })
                .unwrap_or_default(),
            _ => String::new(),
        };
        parts.push(format!("logits{detail}"));
    }
    if x.hidden != y.hidden {
        parts.push("hidden".into());
    }
    if x.target != y.target {
        parts.push(format!("target {} vs {}", x.target, y.target));
    }
    if x.margin != y.margin {
        parts.push(format!(
            "margin bits {:#010x} vs {:#010x}",
            x.margin, y.margin
        ));
    }
    format!("seq {i} index {t}: {}", parts.join(", "))
}

/// Every failed bar of one arm against the reference.
fn verify(
    arm: Arm,
    k: usize,
    reference: &RunOut,
    run: &RunOut,
    bucket: usize,
    sm_count: usize,
) -> Vec<String> {
    let name = format!("{arm:?}");
    let seqs: Vec<usize> = (0..bucket).filter(|i| !run.holes.contains(i)).collect();
    let mut failures: Vec<String> = compare_runs(reference, run, &seqs)
        .into_iter()
        .map(|f| format!("{name}: {f}"))
        .collect();
    if let Some(d) = &run.diverged {
        failures.push(d.clone());
    }
    failures.extend(run.plan_misses.iter().cloned());
    let digest = run_digest(run, &seqs);
    if digest != run_digest(reference, &seqs) {
        failures.push(format!(
            "{name}: digest differs from the reference over the same sequences"
        ));
    }
    eprintln!("{}", digest_line(&name, k, bucket, digest));
    failures.extend(split_guard(&name, run, k, bucket, sm_count));
    if arm.plan() != Plan::Normal && run.patched_steps == 0 {
        failures.push(format!(
            "{name}: VACUOUS, no step had its drafts overwritten"
        ));
    }
    if run.high_rows < 2 || run.mid_rows < 2 {
        failures.push(format!(
            "{name}: VACUOUS coverage, {} keyed rows at or beyond position {HIGH_POS} and {} at or beyond {MID_POS}",
            run.high_rows, run.mid_rows
        ));
    }
    let all: Vec<usize> = (0..=k).collect();
    let (min, which): (u32, Vec<usize>) = match arm {
        Arm::Ramp | Arm::RampSparse => (RAMP_MIN, all),
        Arm::Mixed | Arm::MixedSparse => (1, vec![0, k]),
        Arm::Random | Arm::Permuted => (1, vec![0]),
        _ => (0, Vec::new()),
    };
    let missing: Vec<usize> = run
        .hist
        .short(min)
        .into_iter()
        .filter(|m| which.contains(m))
        .collect();
    if !missing.is_empty() {
        failures.push(format!(
            "{name}: VACUOUS accepted-length histogram {:?}, lengths {missing:?} below {min}",
            run.hist.0
        ));
    }
    eprintln!(
        "arm {name} bucket {bucket} k {k} patched steps {} unpatched steps {} steps {} accepted-length histogram {:?} rows beyond {HIGH_POS}: {} beyond {MID_POS}: {}",
        run.patched_steps, run.unpatched_steps, run.steps, run.hist.0, run.high_rows, run.mid_rows
    );
    failures
}

/// Failures of the split-merge coverage bar of one run: when the step shape's decode attention
/// uses more than one split, at least two keyed rows must lie beyond the first split chunk.
/// Prints which attention path the shape exercises.
fn split_guard(name: &str, run: &RunOut, k: usize, bucket: usize, sm_count: usize) -> Vec<String> {
    let (splits, chunk) = attention_plan(k, bucket, sm_count);
    if splits > 1 {
        eprintln!(
            "split-merge exercised: yes arm={name} bucket {bucket} k {k} splits {splits} chunk {chunk} rows beyond chunk {}",
            run.split_rows
        );
        if run.split_rows < 2 {
            return vec![format!(
                "{name}: VACUOUS split coverage, {} keyed rows beyond the first split chunk {chunk} of {splits} splits",
                run.split_rows
            )];
        }
    } else {
        eprintln!(
            "split-merge exercised: no arm={name} bucket {bucket} k {k} single split chunk {chunk}"
        );
    }
    Vec::new()
}

/// Runs the reference and `arms` of the shape `(k, bucket)` and asserts every bar.
fn check_shape(k: usize, bucket: usize, arms: &[Arm]) {
    let mut h = Harness::new(k);
    let reference = h.run(Arm::Ref, bucket, None);
    let mut failures: Vec<String> = Vec::new();
    if reference.keys.len() != bucket * TOKENS {
        failures.push(format!(
            "Ref: {} keyed rows, expected {}",
            reference.keys.len(),
            bucket * TOKENS
        ));
    }
    if reference.high_rows < 2 || reference.mid_rows < 2 {
        failures.push("Ref: VACUOUS coverage".into());
    }
    eprintln!(
        "arm Ref bucket {bucket} k {k} steps {} accepted-length histogram {:?}",
        reference.steps, reference.hist.0
    );
    let all: Vec<usize> = (0..bucket).collect();
    eprintln!(
        "{}",
        digest_line("Ref", k, bucket, run_digest(&reference, &all))
    );
    failures.extend(split_guard("Ref", &reference, k, bucket, h.sm_count));
    h.phase("compare_Ref");
    for &arm in arms {
        let run = h.run(arm, bucket, Some(&reference));
        failures.extend(verify(arm, k, &reference, &run, bucket, h.sm_count));
        h.phase(&format!("compare_{arm:?}"));
        if arm == Arm::Repeat && !failures.is_empty() {
            break;
        }
    }
    h.phase("done");
    failures.truncate(64);
    assert!(
        failures.is_empty(),
        "scrambled drafts k {k} bucket {bucket}:\n{}",
        failures.join("\n")
    );
}

macro_rules! shape_tests {
    ($($full:ident, $part_a:ident, $part_b:ident: $k:expr, $bucket:expr;)*) => {$(
        #[test]
        #[ignore = "needs GPU and a GLM-OCR checkpoint"]
        fn $full() {
            if !gpu_enabled() {
                return;
            }
            check_shape($k, $bucket, &all_arms());
        }

        #[test]
        #[ignore = "needs GPU and a GLM-OCR checkpoint"]
        fn $part_a() {
            if !gpu_enabled() {
                return;
            }
            check_shape($k, $bucket, &PART_A);
        }

        #[test]
        #[ignore = "needs GPU and a GLM-OCR checkpoint"]
        fn $part_b() {
            if !gpu_enabled() {
                return;
            }
            check_shape($k, $bucket, &PART_B);
        }
    )*};
}

shape_tests! {
    scrambled_drafts_k3_b8, scrambled_drafts_k3_b8_part_a, scrambled_drafts_k3_b8_part_b: 3, 8;
    scrambled_drafts_k3_b16, scrambled_drafts_k3_b16_part_a, scrambled_drafts_k3_b16_part_b: 3, 16;
    scrambled_drafts_k3_b32, scrambled_drafts_k3_b32_part_a, scrambled_drafts_k3_b32_part_b: 3, 32;
    scrambled_drafts_k4_b8, scrambled_drafts_k4_b8_part_a, scrambled_drafts_k4_b8_part_b: 4, 8;
    scrambled_drafts_k4_b16, scrambled_drafts_k4_b16_part_a, scrambled_drafts_k4_b16_part_b: 4, 16;
    scrambled_drafts_k4_b32, scrambled_drafts_k4_b32_part_a, scrambled_drafts_k4_b32_part_b: 4, 32;
}

#[cfg(test)]
mod tests {
    use scriptor_kernels::reference::scatter::mtp_accept;

    use super::*;

    fn random_stream(len: usize, vocab: usize, rng: &mut Mix) -> Vec<u32> {
        (0..len).map(|_| rng.below(vocab) as u32).collect()
    }

    #[test]
    fn forced_plan_accepts_exactly_m() {
        for vocab in [512, 151_552] {
            for k in 1..=7 {
                let mut rng = Mix(0x1234 + k as u64);
                for _ in 0..1000 {
                    let c = rng.below(40);
                    let stream = random_stream(c + k + 2, vocab, &mut rng);
                    for m in 0..=k {
                        let drafts = forced_drafts(k, vocab, &stream, c, m, &mut rng);
                        assert_eq!(drafts.len(), k);
                        assert!(covered(stream.len(), c, m, k));
                        let target = &stream[c..=c + k];
                        let emitted = mtp_accept(&drafts, target);
                        assert_eq!(emitted.len(), m + 1, "k {k} m {m}");
                        assert_eq!(emitted, stream[c..=c + m].to_vec());
                        if m < k {
                            assert_ne!(drafts[m], stream[c + m]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn forced_plan_beyond_the_reference_is_unforced_and_full_length() {
        let mut rng = Mix(7);
        let stream = random_stream(10, 512, &mut rng);
        assert!(!covered(stream.len(), 9, 1, 3));
        assert!(!covered(stream.len(), 8, 2, 3));
        assert!(covered(stream.len(), 7, 3, 3));
        assert!(!covered(stream.len(), 8, 3, 3));
        let drafts = forced_drafts(3, 512, &stream, 9, 1, &mut rng);
        assert_eq!(drafts.len(), 3);
        assert!(drafts.iter().all(|&d| d < 512));
    }

    #[test]
    fn keyed_rows_cover_each_index_once() {
        let mut rng = Mix(99);
        for k in [3usize, 4] {
            for _ in 0..500 {
                let mut c = 0usize;
                let mut seen = vec![0u32; TOKENS];
                while c < TOKENS {
                    let e = rng.below(k + 1) + 1;
                    for t in keyed_range(c, e) {
                        seen[t] += 1;
                    }
                    c += e;
                }
                assert!(seen.iter().all(|&n| n == 1), "{seen:?}");
                assert!(keyed_range(c, 3).is_empty());
            }
        }
    }

    #[test]
    fn histogram_vacuity_guard_flags_a_missing_length() {
        let mut h = Histogram::new(3);
        for m in [0, 1, 3, 0, 1, 3, 3] {
            h.observe(m);
        }
        assert_eq!(h.short(1), vec![2]);
        h.observe(2);
        assert!(h.short(1).is_empty());
        assert_eq!(h.short(2), vec![2]);
    }

    #[test]
    fn permutation_is_a_permutation() {
        let mut rng = Mix(5);
        let original: Vec<u32> = (0..96).collect();
        let mut pool = original.clone();
        permute(&mut pool, &mut rng);
        assert_ne!(pool, original);
        let mut sorted = pool.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, original);
    }

    #[test]
    fn random_plan_is_seed_deterministic() {
        let draw = |seed: u64| {
            let mut rng = Mix(seed);
            (0..8)
                .map(|_| random_drafts(4, 59_392, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(11), draw(11));
        assert_ne!(draw(11), draw(12));
        let mixed = |seed: u64| {
            let mut rng = Mix(seed);
            (0..32)
                .map(|s| forced_m(Forced::Mixed, 3, s, 0, 0, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(mixed(3), mixed(3));
        assert!(mixed(3).iter().all(|&m| m <= 3));
    }

    #[test]
    fn forced_lengths_follow_their_plans() {
        let mut rng = Mix(1);
        for k in [3, 4] {
            for slot in 0..SEQS {
                for step in 0..12 {
                    assert_eq!(
                        forced_m(Forced::Ramp, k, slot, step, 0, &mut rng),
                        (slot + step) % (k + 1)
                    );
                    assert_eq!(forced_m(Forced::Zero, k, slot, step, 0, &mut rng), 0);
                    assert_eq!(forced_m(Forced::Full, k, slot, step, 0, &mut rng), k);
                    let alt = forced_m(Forced::Alternate, k, slot, step, 0, &mut rng);
                    assert_eq!(alt, if slot.is_multiple_of(2) { 0 } else { k });
                    let hot = forced_m(Forced::OneHot, k, slot, step, 5, &mut rng);
                    assert_eq!(hot, if slot == 5 { k } else { 0 });
                }
            }
        }
    }

    #[test]
    fn arm_partition_covers_all_arms() {
        let all = all_arms();
        let mut union: Vec<Arm> = PART_A.iter().chain(&PART_B).copied().collect();
        assert_eq!(union, all);
        union.sort_by_key(|&a| a as u8);
        union.dedup();
        assert_eq!(union.len(), all.len(), "an arm is in both parts");
        assert!(!all.contains(&Arm::Ref));
        assert_eq!(all.len(), 11);
        assert_eq!(all[0], Arm::Repeat);
    }

    #[test]
    fn arm_parts_and_reference_are_exactly_every_variant() {
        let mut listed = vec![Arm::Ref];
        listed.extend(PART_A);
        listed.extend(PART_B);
        assert_eq!(
            listed.len(),
            Arm::EVERY.len(),
            "an arm is unlisted or listed twice"
        );
        for arm in Arm::EVERY {
            let n = listed.iter().filter(|&l| l == arm).count();
            assert_eq!(n, 1, "{arm:?} appears {n} times across Ref and the parts");
        }
        assert_eq!(all_arms().len(), Arm::EVERY.len() - 1);
        for arm in PART_A {
            assert!(!PART_B.contains(&arm), "{arm:?} is in both parts");
        }
    }

    fn sample_run(seed: u64) -> RunOut {
        let mut rng = Mix(seed);
        let mut run = RunOut::new(3, Vec::new());
        for i in 0..4 {
            run.streams
                .push((0..TOKENS + 3).map(|_| rng.below(1000) as u32).collect());
            for t in 0..TOKENS {
                run.keys.insert(
                    (i, t),
                    RowKey {
                        logits: rng.next(),
                        hidden: rng.next(),
                        target: rng.below(1000) as u32,
                        margin: rng.next() as u32,
                    },
                );
            }
        }
        run
    }

    #[test]
    fn digest_changes_on_any_single_field_or_token_flip() {
        let seqs = [0usize, 1, 2, 3];
        let base = sample_run(1);
        let digest = run_digest(&base, &seqs);
        assert_eq!(digest, run_digest(&sample_run(1), &seqs));
        for key in [(0usize, 0usize), (2, 7), (3, TOKENS - 1)] {
            for field in 0..4 {
                let mut run = sample_run(1);
                let row = run.keys.get_mut(&key).unwrap();
                match field {
                    0 => row.logits ^= 1,
                    1 => row.hidden ^= 1 << 63,
                    2 => row.target ^= 1,
                    _ => row.margin ^= 1 << 31,
                }
                assert_ne!(run_digest(&run, &seqs).1, digest.1, "{key:?} field {field}");
                assert_eq!(run_digest(&run, &seqs).0, digest.0);
            }
        }
        for (i, t) in [(0usize, 0usize), (1, 9), (3, TOKENS - 1)] {
            let mut run = sample_run(1);
            run.streams[i][t] ^= 1;
            assert_ne!(run_digest(&run, &seqs).0, digest.0, "token {i} {t}");
            assert_eq!(run_digest(&run, &seqs).1, digest.1);
        }
        let mut missing = sample_run(1);
        missing.keys.remove(&(1, 4));
        assert_ne!(run_digest(&missing, &seqs).1, digest.1);
        assert_ne!(run_digest(&base, &seqs[..3]), digest);
    }

    #[test]
    fn digest_ignores_insertion_order_and_tokens_beyond_the_keyed_prefix() {
        let seqs = [0usize, 1, 2, 3];
        let base = sample_run(2);
        let digest = run_digest(&base, &seqs);
        let mut entries: Vec<_> = base.keys.iter().map(|(k, v)| (*k, v.clone())).collect();
        entries.reverse();
        let mut shuffled = RunOut::new(3, Vec::new());
        shuffled.streams = base.streams.clone();
        for (key, row) in entries {
            shuffled.keys.insert(key, row);
        }
        assert_eq!(run_digest(&shuffled, &seqs), digest);
        let mut longer = sample_run(2);
        for stream in &mut longer.streams {
            stream.push(7);
        }
        assert_eq!(run_digest(&longer, &seqs), digest);
        let mut other = sample_run(2);
        other.streams[1].truncate(TOKENS - 1);
        assert_ne!(run_digest(&other, &seqs).0, digest.0);
    }

    #[test]
    fn digest_line_names_arm_shape_and_both_digests() {
        assert_eq!(
            digest_line("Ramp", 3, 32, (0xab, 0xcd)),
            "digest arm=Ramp k=3 bucket=32 streams=00000000000000ab keys=00000000000000cd"
        );
    }

    #[test]
    fn attention_plan_is_multi_split_only_at_bucket_8() {
        for k in [3usize, 4] {
            assert_eq!(attention_plan(k, 8, 48), (2, HIGH_POS), "k {k}");
            for bucket in [16usize, 32] {
                let (splits, chunk) = attention_plan(k, bucket, 48);
                assert_eq!(splits, 1, "k {k} bucket {bucket}");
                assert!(chunk > LONG_PROMPT + TOKENS);
            }
        }
    }

    #[test]
    fn split_guard_requires_rows_beyond_the_chunk_only_when_split() {
        let mut run = RunOut::new(3, Vec::new());
        assert!(split_guard("Ramp", &run, 3, 16, 48).is_empty());
        assert!(split_guard("Ramp", &run, 3, 32, 48).is_empty());
        assert_eq!(split_guard("Ramp", &run, 3, 8, 48).len(), 1);
        run.split_rows = 1;
        assert_eq!(split_guard("Ramp", &run, 3, 8, 48).len(), 1);
        run.split_rows = 2;
        assert!(split_guard("Ramp", &run, 3, 8, 48).is_empty());
    }

    #[test]
    fn long_prompt_rows_lie_beyond_the_bucket_8_chunk() {
        let (_, chunk) = attention_plan(3, 8, 48);
        let beyond = (0..TOKENS).filter(|&t| LONG_PROMPT + t > chunk).count();
        assert_eq!(beyond, TOKENS);
    }

    #[test]
    fn layout_holes_give_the_requested_bucket() {
        let buckets = graph_buckets(SEQS);
        for bucket in [8usize, 16, 32] {
            let dense = step_shape(&buckets, 0..bucket).expect("dense");
            assert_eq!((dense.bucket, dense.padded()), (bucket, 0));
            let holes = hole_slots(bucket);
            assert_eq!(holes.len(), 3);
            assert!(holes.iter().all(|&h| (4..bucket - 1).contains(&h)));
            let sparse =
                step_shape(&buckets, (0..bucket).filter(|s| !holes.contains(s))).expect("sparse");
            assert_eq!(sparse.bucket, bucket);
            assert_eq!(sparse.active, bucket - 3);
            assert_eq!(sparse.padded(), 3);
        }
    }

    #[test]
    fn prompts_and_blocks_cover_the_split_geometry() {
        assert_eq!((prompt_len(0), prompt_len(1)), (LONG_PROMPT, LONG_PROMPT));
        assert_eq!((prompt_len(2), prompt_len(3)), (MID_PROMPT, MID_PROMPT));
        const { assert!(LONG_PROMPT > HIGH_POS && MID_PROMPT > MID_POS) };
        for i in 4..SEQS {
            assert!((40..120).contains(&prompt_len(i)));
        }
        for k in [3usize, 4] {
            let layout = block_layout(k);
            let mut ids: Vec<u32> = layout.iter().flatten().copied().collect();
            let total = ids.len();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), total, "block ids repeat");
            assert_eq!(ids[0], 1);
            for (i, blocks) in layout.iter().enumerate() {
                assert!(blocks.len() * BLOCK >= capacity_tokens(i, k));
                assert!(capacity_tokens(i, k) <= MAX_CONTEXT, "seq {i} k {k}");
                let stride = (MAX_CONTEXT + 2 * k).div_ceil(BLOCK) + 1;
                assert!(blocks.len() <= stride);
            }
        }
    }

    #[test]
    fn hash_detects_single_bit_flips_and_length() {
        let base: Vec<u16> = (0..1001).map(|n| (n * 37) as u16).collect();
        let h = hash_bits(&base);
        for at in [0, 3, 4, 500, 1000] {
            for bit in [0, 7, 15] {
                let mut flipped = base.clone();
                flipped[at] ^= 1 << bit;
                assert_ne!(hash_bits(&flipped), h, "word {at} bit {bit}");
            }
        }
        assert_ne!(hash_bits(&base[..1000]), h);
        assert_eq!(ulp_distance(0x3f80, 0x3f81), 1);
        assert_eq!(ulp_distance(0x0000, 0x8000), 0);
    }
}
