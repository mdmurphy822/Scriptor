//! Checks of the shared row-scan code behind every selection kernel against a host model
//! written independently of `kernels/row_scan.cuh`. The split selector and the single-block
//! kernels share that header, so their agreement with each other cannot reveal a regression in
//! it; these tests compare the shared path with a host implementation of the original
//! selection semantics instead:
//!
//! * the per-thread element order (16-byte groups of eight for bf16 rows whose width is a
//!   multiple of eight, element by element otherwise), the lowest-index tie rule, and the
//!   warp/block merge tree, so NaN and infinity rows are exact too;
//! * the f32 path, whose in-place penalised logits are compared bit for bit;
//! * stochastic rows of both precisions against the host CDF within the inverse-CDF tolerance.
//!
//! Frequency penalties here are multiples of 0.25 so a fused multiply-subtract on the device
//! and a separate multiply and subtract on the host round identically.
//!
//! Run with
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda --profile gpu-test --test sampling_reference_gpu -- --ignored --nocapture --test-threads=1`.
#![cfg(feature = "cuda")]

mod common;

use common::*;
use cudarc::driver::sys::CUdevice_attribute;
use scriptor_kernels::sampling::{
    RowParams, SamplingKernels, TokenMasks, choose_token, row_block_threads, uniform,
};

/// Relative perturbation of the uniform draw within which a sampled token may differ from the
/// host's: device and host sum the softmax mass in different orders.
const CDF_TOLERANCE: f32 = 1e-3;

#[derive(Clone, Copy)]
struct Top2 {
    best: f32,
    index: u32,
    second: f32,
}

const NONE: u32 = u32::MAX;

fn empty() -> Top2 {
    Top2 {
        best: f32::NEG_INFINITY,
        index: NONE,
        second: f32::NEG_INFINITY,
    }
}

fn add(t: &mut Top2, v: f32, i: u32) {
    if v > t.best {
        t.second = t.best;
        t.best = v;
        t.index = i;
    } else if v == t.best {
        if t.index == NONE {
            t.index = i;
        } else {
            t.second = v;
        }
    } else if v > t.second {
        t.second = v;
    }
}

fn merge(a: Top2, b: Top2) -> Top2 {
    if a.best > b.best {
        Top2 {
            best: a.best,
            index: a.index,
            second: a.second.max(b.best),
        }
    } else if b.best > a.best {
        Top2 {
            best: b.best,
            index: b.index,
            second: b.second.max(a.best),
        }
    } else {
        Top2 {
            best: a.best,
            index: a.index.min(b.index),
            second: a.best,
        }
    }
}

/// The xor-shuffle reduction over the 32 lanes of a warp; every lane's result.
fn warp(lanes: &[Top2; 32]) -> [Top2; 32] {
    let mut t = *lanes;
    for offset in [16usize, 8, 4, 2, 1] {
        let prev = t;
        for (l, lane) in t.iter_mut().enumerate() {
            *lane = merge(prev[l], prev[l ^ offset]);
        }
    }
    t
}

/// Token and margin of one penalised row as the single-block kernel computes them with
/// `width` threads: thread `t` scans its elements in order into a running top-2, then warps
/// reduce and warp results reduce through one more warp.
fn model_row(vals: &[f32], width: usize, grouped: bool) -> (u32, f32) {
    let vocab = vals.len();
    let mut tops = vec![empty(); width];
    for (t, top) in tops.iter_mut().enumerate() {
        if grouped {
            let mut g = t;
            while g < vocab / 8 {
                for j in 0..8 {
                    add(top, vals[g * 8 + j], (g * 8 + j) as u32);
                }
                g += width;
            }
        } else {
            let mut i = t;
            while i < vocab {
                add(top, vals[i], i as u32);
                i += width;
            }
        }
    }
    let warps = width.div_ceil(32);
    let per_warp: Vec<Top2> = (0..warps)
        .map(|w| {
            let mut lanes = [empty(); 32];
            for (l, lane) in lanes.iter_mut().enumerate() {
                if let Some(&t) = tops.get(w * 32 + l) {
                    *lane = t;
                }
            }
            warp(&lanes)[0]
        })
        .collect();
    let mut lanes = [empty(); 32];
    for (l, lane) in lanes.iter_mut().enumerate() {
        if l < warps {
            *lane = per_warp[l];
        }
    }
    let top = warp(&lanes)[0];
    (top.index, top.best - top.second)
}

/// The penalised row `row` of a batch: penalties from the prompt and output history of the
/// row's sequence plus its first `row % q` drafts as emitted output.
fn penalised_row(b: &Batch, vals: &[f32], row: usize) -> Vec<f32> {
    let q = b.k + 1;
    let (seq, ahead) = (row / q, row % q);
    let p = &b.params[seq];
    let slot = p.slot as usize;
    let drafts = &b.drafts[seq * b.k..seq * b.k + ahead];
    vals.iter()
        .enumerate()
        .map(|(i, &raw)| {
            let id = i as u32;
            let extra = drafts.iter().filter(|&&d| d == id).count() as u32;
            let in_output = b.masks.in_output(slot, id) || extra > 0;
            let mut v = raw;
            if p.repetition != 1.0 && (b.masks.in_prompt(slot, id) || in_output) {
                v = if v > 0.0 {
                    v / p.repetition
                } else {
                    v * p.repetition
                };
            }
            let count = if p.frequency != 0.0 {
                (b.masks.output_count(slot, id) + extra) as f32
            } else if in_output {
                1.0
            } else {
                0.0
            };
            v -= p.frequency * count;
            if in_output {
                v -= p.presence;
            }
            v
        })
        .collect()
}

fn widen(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| f32::from_bits(u32::from(b) << 16))
        .collect()
}

fn same_bits(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn block_width(ctx: &std::sync::Arc<cudarc::driver::CudaContext>, rows: usize) -> usize {
    let attr = |a| ctx.attribute(a).expect("attribute") as usize;
    row_block_threads(
        rows,
        attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR),
    ) as usize
}

/// A batch of tie-heavy rows with NaN, infinity and planted maximum ties, penalties from real
/// prompt words, output history that repeats words, and drafts that repeat history.
fn reference_batch(seqs: usize, k: usize, vocab: usize, seed: u64, sampled: bool) -> Batch {
    let toks = passage_tokens(vocab);
    let mut rng = Rng(seed | 1);
    let q = k + 1;
    let mut masks = TokenMasks::new(seqs, vocab);
    let mut params = Vec::new();
    let mut drafts = Vec::new();
    let mut flat = vec![0f32; seqs * q * vocab];
    for s in 0..seqs {
        let start = (s * 11) % (toks.len() - 24);
        masks.reset_slot(s, &toks[start..start + 14]);
        for &t in &toks[start + 7..start + 19] {
            masks.push_output(s, t);
        }
        masks.push_output(s, toks[start + 7]);
        for d in 0..k {
            drafts.push(toks[start + 7 + (d * 3) % 9]);
        }
        let mut p = RowParams::greedy(s as u32);
        p.repetition = [1.0, 1.25, 1.5][s % 3];
        p.presence = [0.0, 0.5, 0.25][(s / 2) % 3];
        p.frequency = [0.0, 0.25, 0.5][(s / 3) % 3];
        if sampled {
            p.seed = 4000 + s as u64;
            p.step = (s * 3) as u32;
            p.temperature = [0.7, 1.0, 1.3][s % 3];
            p.top_k = [0, 50, 7, 0][s % 4];
            p.top_p = [1.0, 0.9, 0.5, 0.95][(s / 2) % 4];
        }
        params.push(p);
        for j in 0..q {
            let r = s * q + j;
            let row = &mut flat[r * vocab..(r + 1) * vocab];
            for v in row.iter_mut() {
                *v = if sampled {
                    rng.next() * 6.0
                } else {
                    (rng.next() * 6.0).round() / 4.0
                };
            }
            if sampled {
                continue;
            }
            for t in [vocab - 1, vocab / 2, (r * 7) % vocab] {
                row[t] = 9.0;
            }
            match r % 5 {
                0 => row[(r * 37) % vocab] = f32::NAN,
                1 => {
                    row[vocab / 3] = f32::INFINITY;
                    row[vocab - 2] = f32::INFINITY;
                }
                2 => {
                    for v in row.iter_mut().take(7) {
                        *v = f32::NEG_INFINITY;
                    }
                }
                _ => {}
            }
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

struct Run {
    tokens: Vec<u32>,
    margins: Vec<f32>,
    logits: Vec<f32>,
}

/// Runs the f32 speculative kernel over the widened rows; `logits` is the penalised result.
fn run_f32(
    kern: &SamplingKernels,
    s: &std::sync::Arc<cudarc::driver::CudaStream>,
    b: &Batch,
) -> Run {
    let rows = b.seqs * (b.k + 1);
    let mut d = upload(s, b);
    let mut logits = s.clone_htod(&widen(&b.bits)).expect("upload");
    kern.choose_spec(
        s,
        &mut logits,
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
    assert_eq!(d.tokens.len(), rows);
    Run {
        tokens: s.clone_dtoh(&d.tokens).expect("download"),
        margins: s.clone_dtoh(&d.margins).expect("download"),
        logits: s.clone_dtoh(&logits).expect("download"),
    }
}

fn run_bf16(
    kern: &SamplingKernels,
    s: &std::sync::Arc<cudarc::driver::CudaStream>,
    b: &Batch,
) -> Run {
    let mut d = upload(s, b);
    original(kern, s, b, &mut d);
    Run {
        tokens: s.clone_dtoh(&d.tokens).expect("download"),
        margins: s.clone_dtoh(&d.margins).expect("download"),
        logits: Vec::new(),
    }
}

const SHAPES: [(usize, usize); 4] = [(1, 0), (3, 0), (2, 3), (5, 2)];
const VOCABS: [usize; 4] = [4096, 4099, 777, 59_392];

#[test]
#[ignore = "needs GPU"]
fn bf16_speculative_greedy_matches_the_host_model_with_nan_infinity_ties_and_penalties() {
    let Some((dev, s)) = setup() else { return };
    let kern = SamplingKernels::load(dev.context()).expect("load");
    for vocab in VOCABS {
        for (seqs, k) in SHAPES {
            let b = reference_batch(seqs, k, vocab, (vocab * 3 + seqs * 5 + k) as u64, false);
            let rows = seqs * (k + 1);
            let width = block_width(dev.context(), rows);
            let got = run_bf16(&kern, &s, &b);
            let wide = widen(&b.bits);
            for r in 0..rows {
                let pen = penalised_row(&b, &wide[r * vocab..(r + 1) * vocab], r);
                let (token, margin) = model_row(&pen, width, vocab % 8 == 0);
                let what = format!("bf16 vocab {vocab} seqs {seqs} k {k} row {r}");
                assert_eq!(got.tokens[r], token, "{what} token");
                assert!(same_bits(got.margins[r], margin), "{what} margin");
            }
        }
    }
}

#[test]
#[ignore = "needs GPU"]
fn f32_speculative_greedy_matches_the_host_model_and_writes_penalised_logits_in_place() {
    let Some((dev, s)) = setup() else { return };
    let kern = SamplingKernels::load(dev.context()).expect("load");
    for vocab in VOCABS {
        for (seqs, k) in SHAPES {
            let b = reference_batch(seqs, k, vocab, (vocab * 7 + seqs * 3 + k) as u64, false);
            let rows = seqs * (k + 1);
            let width = block_width(dev.context(), rows);
            let got = run_f32(&kern, &s, &b);
            let wide = widen(&b.bits);
            for r in 0..rows {
                let pen = penalised_row(&b, &wide[r * vocab..(r + 1) * vocab], r);
                let (token, margin) = model_row(&pen, width, false);
                let what = format!("f32 vocab {vocab} seqs {seqs} k {k} row {r}");
                assert_eq!(got.tokens[r], token, "{what} token");
                assert!(same_bits(got.margins[r], margin), "{what} margin");
                for (i, (&g, &w)) in got.logits[r * vocab..(r + 1) * vocab]
                    .iter()
                    .zip(&pen)
                    .enumerate()
                {
                    assert!(same_bits(g, w), "{what} penalised element {i}: {g} vs {w}");
                }
            }
        }
    }
}

fn check_stochastic(b: &Batch, got: &Run, wide: &[f32], what: &str) -> (usize, usize) {
    let q = b.k + 1;
    let vocab = b.vocab;
    let (mut exact, mut total) = (0, 0);
    for r in 0..b.seqs * q {
        let p = &b.params[r / q];
        let pen = penalised_row(b, &wide[r * vocab..(r + 1) * vocab], r);
        let u = uniform(p.seed, p.step + (r % q) as u32);
        let at = |u: f32| choose_token(&pen, p.temperature, p.top_k, p.top_p, u);
        let want = at(u);
        total += 1;
        if got.tokens[r] == want {
            exact += 1;
        } else {
            let lo = at((u * (1.0 - CDF_TOLERANCE)).max(0.0));
            let hi = at((u * (1.0 + CDF_TOLERANCE)).min(0.999_999));
            assert!(
                got.tokens[r] == lo || got.tokens[r] == hi,
                "{what} row {r}: device {} not within tolerance of host {want} ({lo}, {hi})",
                got.tokens[r]
            );
        }
    }
    (exact, total)
}

#[test]
#[ignore = "needs GPU"]
fn stochastic_speculative_rows_match_the_host_cdf_in_both_precisions() {
    let Some((dev, s)) = setup() else { return };
    let kern = SamplingKernels::load(dev.context()).expect("load");
    let (mut exact, mut total) = (0, 0);
    for vocab in [4096usize, 4099, 777] {
        for (seqs, k) in [(4usize, 0usize), (3, 3), (6, 2)] {
            let b = reference_batch(seqs, k, vocab, (vocab * 11 + seqs + k) as u64, true);
            let wide = widen(&b.bits);
            let rows = seqs * (k + 1);
            let width = block_width(dev.context(), rows);
            for (name, got, grouped) in [
                ("f32", run_f32(&kern, &s, &b), false),
                ("bf16", run_bf16(&kern, &s, &b), vocab % 8 == 0),
            ] {
                let what = format!("{name} vocab {vocab} seqs {seqs} k {k}");
                let (e, t) = check_stochastic(&b, &got, &wide, &what);
                exact += e;
                total += t;
                for r in 0..rows {
                    let pen = penalised_row(&b, &wide[r * vocab..(r + 1) * vocab], r);
                    let (_, margin) = model_row(&pen, width, grouped);
                    assert!(
                        same_bits(got.margins[r], margin),
                        "{what} row {r}: the margin of a sampled row is the greedy margin"
                    );
                }
            }
        }
    }
    assert!(exact * 100 >= total * 95, "exact {exact} of {total}");
}
