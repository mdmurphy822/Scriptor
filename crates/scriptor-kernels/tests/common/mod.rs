//! Shared helpers of the split-selector GPU test binaries: text-shaped batches, device
//! uploads and the original-versus-split comparison.
#![allow(dead_code)]

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use scriptor_cuda::Device;
use scriptor_kernels::sampling::{
    RowParams, SamplingKernels, SplitScratch, SplitSelect, TokenMasks,
};

pub const VOCAB: usize = 59_392;
pub const SELECTS: [(u32, u32); 6] = [(128, 8), (256, 4), (128, 4), (256, 8), (128, 1), (32, 3)];

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    pub fn below(&mut self, n: usize) -> usize {
        (((self.next() + 1.0) * 0.5 * n as f32) as usize).min(n - 1)
    }
}

pub fn setup() -> Option<(Device, Arc<CudaStream>)> {
    if std::env::var("SCRIPTOR_GPU_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let device = Device::open().expect("gpu expected");
    let stream = device.new_stream().expect("stream");
    Some((device, stream))
}

pub fn bf16_bits(v: &[f32]) -> Vec<u16> {
    v.iter()
        .map(|&x| half::bf16::from_f32(x).to_bits())
        .collect()
}

/// A passage of public-domain English prose.
pub const PASSAGE: &str = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters. My dear Mr. Bennet, said his lady to him one day, have you heard that Netherfield Park is let at last? Mr. Bennet replied that he had not.";

/// Token ids of the passage's words, hashed into the vocabulary.
pub fn passage_tokens(vocab: usize) -> Vec<u32> {
    PASSAGE
        .split_whitespace()
        .map(|w| {
            let h = w.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
            });
            (h % vocab as u64) as u32
        })
        .collect()
}

/// One verification batch: logits, masks, parameters and drafts.
pub struct Batch {
    pub vocab: usize,
    pub seqs: usize,
    pub k: usize,
    pub bits: Vec<u16>,
    pub masks: TokenMasks,
    pub params: Vec<RowParams>,
    pub drafts: Vec<u32>,
}

/// Logits shaped like a language model's: a unigram log-frequency floor over the passage's
/// words, a boost on each word's successor in the passage, and noise; each sequence is a
/// different window of the passage, with that window's words as prompt and output history and
/// drafts that repeat tokens.
pub fn text_batch(seqs: usize, k: usize, vocab: usize, seed: u64) -> Batch {
    let toks = passage_tokens(vocab);
    let mut rng = Rng(seed | 1);
    let rows = seqs * (k + 1);
    let mut flat = vec![0f32; rows * vocab];
    let mut masks = TokenMasks::new(seqs + 3, vocab);
    let mut params = Vec::new();
    let mut drafts = Vec::new();
    for s in 0..seqs {
        let start = (s * 13) % (toks.len() - 20);
        let slot = (s * 5 + 3) % (seqs + 3);
        masks.reset_slot(slot, &toks[start..start + 12]);
        for &t in &toks[start + 6..start + 18] {
            masks.push_output(slot, t);
        }
        masks.push_output(slot, toks[start + 6]);
        for d in 0..k {
            drafts.push(if d % 2 == 0 {
                toks[start + 8]
            } else {
                toks[start + 9 + d]
            });
        }
        let mut p = RowParams::greedy(slot as u32);
        p.repetition = [1.0, 1.15, 1.3][s % 3];
        p.presence = [0.0, 0.4, 0.2][(s / 2) % 3];
        p.frequency = [0.0, 0.25][s % 2];
        params.push(p);
        for j in 0..=k {
            let row = &mut flat[(s * (k + 1) + j) * vocab..][..vocab];
            for v in row.iter_mut() {
                *v = rng.next() * 3.0 - 4.0;
            }
            for (i, &t) in toks[start..start + 14].iter().enumerate() {
                row[t as usize] += 3.0 + 0.1 * i as f32;
            }
            row[toks[start + 14 + j % 3] as usize] += 5.0;
            row[toks[start + 8] as usize] += 4.0;
        }
    }
    Batch {
        vocab,
        seqs,
        k,
        bits: bf16_bits(&flat),
        masks,
        params,
        drafts,
    }
}

pub struct Dev {
    pub logits: CudaSlice<u16>,
    pub params: CudaSlice<RowParams>,
    pub prompt: CudaSlice<u32>,
    pub output: CudaSlice<u32>,
    pub counts: CudaSlice<u32>,
    pub drafts: CudaSlice<u32>,
    pub tokens: CudaSlice<u32>,
    pub margins: CudaSlice<f32>,
}

pub fn upload(s: &Arc<CudaStream>, b: &Batch) -> Dev {
    let rows = b.seqs * (b.k + 1);
    Dev {
        logits: s.clone_htod(&b.bits).expect("upload"),
        params: s.clone_htod(&b.params).expect("upload"),
        prompt: s.clone_htod(b.masks.prompt_words()).expect("upload"),
        output: s.clone_htod(b.masks.output_words()).expect("upload"),
        counts: s.clone_htod(b.masks.count_words()).expect("upload"),
        drafts: s
            .clone_htod(&if b.drafts.is_empty() {
                vec![0]
            } else {
                b.drafts.clone()
            })
            .expect("upload"),
        tokens: s.clone_htod(&vec![0u32; rows]).expect("upload"),
        margins: s.clone_htod(&vec![0f32; rows]).expect("upload"),
    }
}

pub fn original(kern: &SamplingKernels, s: &Arc<CudaStream>, b: &Batch, d: &mut Dev) {
    kern.choose_spec_bf16(
        s,
        &d.logits,
        b.seqs,
        b.k,
        b.vocab,
        &d.params,
        &d.prompt,
        &d.output,
        &d.counts,
        b.masks.words(),
        &d.drafts,
        &mut d.tokens,
        &mut d.margins,
    )
    .expect("launch");
}

pub fn split(
    kern: &SamplingKernels,
    s: &Arc<CudaStream>,
    sel: &SplitSelect,
    scratch: &mut SplitScratch,
    b: &Batch,
    d: &mut Dev,
) -> bool {
    kern.choose_spec_bf16_split(
        s,
        sel,
        scratch,
        &d.logits,
        b.seqs,
        b.k,
        b.vocab,
        &d.params,
        &d.prompt,
        &d.output,
        &d.counts,
        b.masks.words(),
        &d.drafts,
        &mut d.tokens,
        &mut d.margins,
    )
    .expect("launch")
}

pub fn result(s: &Arc<CudaStream>, d: &Dev) -> (Vec<u32>, Vec<u32>) {
    (
        s.clone_dtoh(&d.tokens).expect("download"),
        s.clone_dtoh(&d.margins)
            .expect("download")
            .iter()
            .map(|m| m.to_bits())
            .collect(),
    )
}

pub fn select(block: u32, ctas: u32) -> SplitSelect {
    SplitSelect {
        block_threads: block,
        ctas_per_row: ctas,
        max_rows: 64,
    }
}

/// Asserts every partition choice reproduces the single-block kernel's tokens and margin bits.
pub fn assert_equal_to_original(
    kern: &SamplingKernels,
    s: &Arc<CudaStream>,
    b: &Batch,
    what: &str,
) {
    let mut d = upload(s, b);
    original(kern, s, b, &mut d);
    let want = result(s, &d);
    let mut scratch = kern.alloc_split_scratch(s, 64).expect("scratch");
    for (block, ctas) in SELECTS {
        d.tokens = s
            .clone_htod(&vec![0xdead_beefu32; want.0.len()])
            .expect("upload");
        d.margins = s
            .clone_htod(&vec![f32::from_bits(0x7fc0_1234); want.0.len()])
            .expect("upload");
        assert!(
            split(kern, s, &select(block, ctas), &mut scratch, b, &mut d),
            "{what} {block}x{ctas} did not run"
        );
        let got = result(s, &d);
        assert_eq!(got.0, want.0, "{what} {block}x{ctas} tokens");
        assert_eq!(got.1, want.1, "{what} {block}x{ctas} margin bits");
    }
}

/// The batch of the first `groups` groups of a fixture: one sequence per group, with the
/// group's penalties, prompt and output history, drafts and logits rows.
pub fn fixture_batch(f: &scriptor_kernels::logits_fixture::LogitsFixture, groups: usize) -> Batch {
    let k = f.q - 1;
    let mut masks = TokenMasks::new(groups, f.vocab);
    let mut params = Vec::new();
    let mut drafts = Vec::new();
    let mut bits = Vec::new();
    for (slot, g) in f.groups[..groups].iter().enumerate() {
        masks.reset_slot(slot, &g.prompt);
        for &t in &g.output {
            masks.push_output(slot, t);
        }
        let mut p = RowParams::greedy(slot as u32);
        p.repetition = g.repetition;
        p.presence = g.presence;
        p.frequency = g.frequency;
        params.push(p);
        drafts.extend_from_slice(&g.drafts);
        bits.extend_from_slice(&g.logits);
    }
    Batch {
        vocab: f.vocab,
        seqs: groups,
        k,
        bits,
        masks,
        params,
        drafts,
    }
}
