//! GPU checks of the penalty / argmax / sampling kernels against the CPU
//! reference. Run with
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda sampling -- --ignored`.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::CudaStream;
use scriptor_cuda::Device;
use scriptor_kernels::sampling::{
    RowParams, SamplingKernels, TokenMasks, apply_masked_penalties, choose_token, process_row,
    uniform,
};

/// Relative perturbation of the uniform draw within which a sampled token may
/// differ from the CPU choice: the device and CPU accumulate the softmax mass
/// in different orders, so a draw that lands within this distance of a
/// cumulative-mass boundary may resolve to the neighbouring token.
const CDF_TOLERANCE: f32 = 1e-3;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn setup() -> Option<(Device, Arc<CudaStream>)> {
    if std::env::var("SCRIPTOR_GPU_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let device = Device::open().expect("gpu expected");
    let stream = device.new_stream().expect("stream");
    Some((device, stream))
}

struct Case {
    vocab: usize,
    logits: Vec<f32>,
    masks: TokenMasks,
    params: Vec<RowParams>,
}

fn build(rows: usize, vocab: usize, seed: u64, sampled: bool) -> Case {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let logits: Vec<f32> = (0..rows * vocab).map(|_| rng.next() * 6.0).collect();
    let mut masks = TokenMasks::new(rows, vocab);
    let mut params = Vec::new();
    for r in 0..rows {
        let prompt: Vec<u32> = (0..40)
            .map(|i| ((r * 131 + i * 17) % vocab) as u32)
            .collect();
        masks.reset_slot(r, &prompt);
        for i in 0..12 {
            masks.push_output(r, ((r * 71 + i * 29) % vocab) as u32);
        }
        masks.push_output(r, (r % vocab) as u32);
        let mut p = RowParams::greedy(r as u32);
        p.repetition = 1.0 + 0.1 * (r % 3) as f32;
        p.presence = 0.2 * (r % 2) as f32;
        p.frequency = 0.1 * (r % 4) as f32;
        if sampled {
            p.seed = 1000 + r as u64;
            p.step = (r * 3) as u32;
            p.temperature = [0.7, 1.0, 1.3][r % 3];
            p.top_k = [0, 50, 7, 0][r % 4];
            p.top_p = [1.0, 0.9, 0.5, 0.95][(r / 2) % 4];
        }
        params.push(p);
    }
    Case {
        vocab,
        logits,
        masks,
        params,
    }
}

struct Device_ {
    tokens: Vec<u32>,
    margins: Vec<f32>,
    logits: Vec<f32>,
}

fn run(dev: &Device, s: &Arc<CudaStream>, c: &Case, greedy_only: bool) -> Device_ {
    let k = SamplingKernels::load(dev.context()).expect("load");
    let rows = c.params.len();
    let mut logits = s.clone_htod(&c.logits).expect("upload");
    let params = s.clone_htod(&c.params).expect("upload");
    let prompt = s.clone_htod(c.masks.prompt_words()).expect("upload");
    let output = s.clone_htod(c.masks.output_words()).expect("upload");
    let counts = s.clone_htod(c.masks.count_words()).expect("upload");
    let mut tokens = s.clone_htod(&vec![0u32; rows]).expect("upload");
    let mut margins = s.clone_htod(&vec![0f32; rows]).expect("upload");
    k.choose(
        s,
        greedy_only,
        &mut logits,
        rows,
        c.vocab,
        &params,
        &prompt,
        &output,
        &counts,
        c.masks.words(),
        &mut tokens,
        &mut margins,
    )
    .expect("launch");
    Device_ {
        tokens: s.clone_dtoh(&tokens).expect("download"),
        margins: s.clone_dtoh(&margins).expect("download"),
        logits: s.clone_dtoh(&logits).expect("download"),
    }
}

#[test]
#[ignore = "needs GPU"]
fn sampling_penalty_argmax_matches_reference() {
    let Some((dev, s)) = setup() else { return };
    for (rows, vocab) in [(6, 4096), (3, 1000), (5, 59_904)] {
        let c = build(rows, vocab, vocab as u64, false);
        let got = run(&dev, &s, &c, true);
        for r in 0..rows {
            let want = process_row(
                &c.logits[r * vocab..(r + 1) * vocab],
                &c.masks,
                &c.params[r],
            );
            assert_eq!(got.tokens[r], want.token, "row {r} vocab {vocab}");
            assert!(
                (got.margins[r] - want.margin).abs() <= 1e-5 * want.margin.abs().max(1.0),
                "row {r} margin {} vs {}",
                got.margins[r],
                want.margin
            );
            let mut row = c.logits[r * vocab..(r + 1) * vocab].to_vec();
            apply_masked_penalties(&mut row, &c.masks, r, c.params[r].penalties());
            assert_eq!(&got.logits[r * vocab..(r + 1) * vocab], &row[..], "row {r}");
        }
    }
}

#[test]
#[ignore = "needs GPU"]
fn sampling_zero_temperature_ignores_top_filters_on_device() {
    let Some((dev, s)) = setup() else { return };
    let mut c = build(4, 2048, 5, true);
    for p in &mut c.params {
        p.temperature = 0.0;
        p.top_k = 1;
        p.top_p = 1e-5;
    }
    let got = run(&dev, &s, &c, false);
    for r in 0..4 {
        let want = process_row(&c.logits[r * 2048..(r + 1) * 2048], &c.masks, &c.params[r]);
        assert_eq!(got.tokens[r], want.token);
    }
}

#[test]
#[ignore = "needs GPU"]
fn sampling_sampled_rows_match_reference_within_cdf_tolerance() {
    let Some((dev, s)) = setup() else { return };
    let (rows, vocab) = (12, 4096);
    let mut exact = 0usize;
    let mut total = 0usize;
    for round in 0..8u64 {
        let mut c = build(rows, vocab, 77 + round, true);
        for p in &mut c.params {
            p.step += round as u32 * 5;
        }
        let got = run(&dev, &s, &c, false);
        for r in 0..rows {
            let p = &c.params[r];
            let mut row = c.logits[r * vocab..(r + 1) * vocab].to_vec();
            apply_masked_penalties(&mut row, &c.masks, r, p.penalties());
            let u = uniform(p.seed, p.step);
            let at = |u: f32| choose_token(&row, p.temperature, p.top_k, p.top_p, u);
            let want = at(u);
            total += 1;
            if got.tokens[r] == want {
                exact += 1;
                continue;
            }
            let lo = at((u * (1.0 - CDF_TOLERANCE)).max(0.0));
            let hi = at((u * (1.0 + CDF_TOLERANCE)).min(0.999_999));
            assert!(
                got.tokens[r] == lo || got.tokens[r] == hi,
                "row {r}: device {} not within tolerance of cpu {want} ({lo}, {hi})",
                got.tokens[r]
            );
        }
    }
    assert!(exact * 100 >= total * 95, "exact {exact} of {total}");
}

#[test]
#[ignore = "needs GPU"]
fn sampling_sampling_is_reproducible_for_a_seed() {
    let Some((dev, s)) = setup() else { return };
    let c = build(8, 3000, 9, true);
    let a = run(&dev, &s, &c, false);
    let b = run(&dev, &s, &c, false);
    assert_eq!(a.tokens, b.tokens);
}

#[test]
#[ignore = "needs GPU"]
fn sampling_mark_tokens_matches_masks() {
    let Some((dev, s)) = setup() else { return };
    let k = SamplingKernels::load(dev.context()).expect("load");
    let (slots_n, vocab) = (3usize, 200usize);
    let mut masks = TokenMasks::new(slots_n, vocab);
    let slots = [0u32, 1, 1, 2, 1, 0];
    let tokens = [5u32, 7, 7, 199, 33, 5];
    for (&sl, &t) in slots.iter().zip(&tokens) {
        masks.push_output(sl as usize, t);
    }
    let fresh = TokenMasks::new(slots_n, vocab);
    let mut bits = s.clone_htod(fresh.output_words()).expect("upload");
    let mut counts = s.clone_htod(fresh.count_words()).expect("upload");
    let d_slots = s.clone_htod(&slots).expect("upload");
    let d_tokens = s.clone_htod(&tokens).expect("upload");
    k.mark_tokens(
        &s,
        &mut bits,
        &mut counts,
        &d_slots,
        &d_tokens,
        tokens.len(),
        fresh.words(),
        vocab,
    )
    .expect("launch");
    let got_bits: Vec<u32> = s.clone_dtoh(&bits).expect("download");
    let got_counts: Vec<u32> = s.clone_dtoh(&counts).expect("download");
    assert_eq!(got_bits, masks.output_words());
    assert_eq!(got_counts, masks.count_words());
}

fn to_bf16_bits(v: &[f32]) -> Vec<u16> {
    v.iter()
        .map(|&x| half::bf16::from_f32(x).to_bits())
        .collect()
}

fn widen(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| f32::from_bits(u32::from(b) << 16))
        .collect()
}

struct Picks {
    tokens: Vec<u32>,
    margins: Vec<u32>,
}

fn bf16_case(rows: usize, vocab: usize, seed: u64, sampled: bool) -> (Case, Vec<u16>) {
    let mut c = build(rows, vocab, seed, sampled);
    let bits = to_bf16_bits(&c.logits);
    c.logits = widen(&bits);
    (c, bits)
}

fn picks(tokens: &CudaSlice<u32>, margins: &CudaSlice<f32>, s: &Arc<CudaStream>) -> Picks {
    Picks {
        tokens: s.clone_dtoh(tokens).expect("download"),
        margins: s
            .clone_dtoh(margins)
            .expect("download")
            .iter()
            .map(|m| m.to_bits())
            .collect(),
    }
}

use cudarc::driver::CudaSlice;

#[test]
#[ignore = "needs GPU"]
fn sampling_bf16_rows_are_bit_identical_to_widened_rows() {
    let Some((dev, s)) = setup() else { return };
    let k = SamplingKernels::load(dev.context()).expect("load");
    for (greedy_only, sampled) in [(true, false), (false, false), (false, true)] {
        for (rows, vocab) in [(6, 4096), (3, 1000), (5, 59_904), (9, 33)] {
            let (c, bits) = bf16_case(rows, vocab, vocab as u64 + 3, sampled);
            let params = s.clone_htod(&c.params).expect("upload");
            let prompt = s.clone_htod(c.masks.prompt_words()).expect("upload");
            let output = s.clone_htod(c.masks.output_words()).expect("upload");
            let counts = s.clone_htod(c.masks.count_words()).expect("upload");
            let run = |wide: bool| {
                let mut tokens = s.clone_htod(&vec![0u32; rows]).expect("upload");
                let mut margins = s.clone_htod(&vec![0f32; rows]).expect("upload");
                if wide {
                    let mut logits = s.clone_htod(&c.logits).expect("upload");
                    k.choose(
                        &s,
                        greedy_only,
                        &mut logits,
                        rows,
                        vocab,
                        &params,
                        &prompt,
                        &output,
                        &counts,
                        c.masks.words(),
                        &mut tokens,
                        &mut margins,
                    )
                } else {
                    let logits = s.clone_htod(&bits).expect("upload");
                    k.choose_bf16(
                        &s,
                        greedy_only,
                        &logits,
                        rows,
                        vocab,
                        &params,
                        &prompt,
                        &output,
                        &counts,
                        c.masks.words(),
                        &mut tokens,
                        &mut margins,
                    )
                }
                .expect("launch");
                picks(&tokens, &margins, &s)
            };
            let (a, b) = (run(true), run(false));
            let what = format!("greedy {greedy_only} sampled {sampled} rows {rows} vocab {vocab}");
            assert_eq!(a.tokens, b.tokens, "{what}");
            assert_eq!(a.margins, b.margins, "{what}");
        }
    }
}

#[test]
#[ignore = "needs GPU"]
fn sampling_bf16_spec_rows_are_bit_identical_to_widened_rows() {
    let Some((dev, s)) = setup() else { return };
    let kern = SamplingKernels::load(dev.context()).expect("load");
    let (seqs, k) = (5usize, 3usize);
    let q = k + 1;
    for (vocab, sampled) in [(4096, false), (4096, true), (59_904, true)] {
        let mut c = build(seqs, vocab, 41 + vocab as u64, sampled);
        let rows = seqs * q;
        let mut rng = Rng(0xdead_beef ^ vocab as u64);
        let wide: Vec<f32> = (0..rows * vocab).map(|_| rng.next() * 6.0).collect();
        let bits = to_bf16_bits(&wide);
        c.logits = widen(&bits);
        let drafts: Vec<u32> = (0..seqs * k)
            .map(|i| ((i * 37 + 5) % vocab) as u32)
            .collect();
        let params = s.clone_htod(&c.params).expect("upload");
        let prompt = s.clone_htod(c.masks.prompt_words()).expect("upload");
        let output = s.clone_htod(c.masks.output_words()).expect("upload");
        let counts = s.clone_htod(c.masks.count_words()).expect("upload");
        let d_drafts = s.clone_htod(&drafts).expect("upload");
        let run = |wide_path: bool| {
            let mut tokens = s.clone_htod(&vec![0u32; rows]).expect("upload");
            let mut margins = s.clone_htod(&vec![0f32; rows]).expect("upload");
            if wide_path {
                let mut logits = s.clone_htod(&c.logits).expect("upload");
                kern.choose_spec(
                    &s,
                    &mut logits,
                    seqs,
                    k,
                    vocab,
                    &params,
                    &prompt,
                    &output,
                    &counts,
                    c.masks.words(),
                    &d_drafts,
                    &mut tokens,
                    &mut margins,
                )
            } else {
                let logits = s.clone_htod(&bits).expect("upload");
                kern.choose_spec_bf16(
                    &s,
                    &logits,
                    seqs,
                    k,
                    vocab,
                    &params,
                    &prompt,
                    &output,
                    &counts,
                    c.masks.words(),
                    &d_drafts,
                    &mut tokens,
                    &mut margins,
                )
            }
            .expect("launch");
            picks(&tokens, &margins, &s)
        };
        let (a, b) = (run(true), run(false));
        assert_eq!(a.tokens, b.tokens, "vocab {vocab} sampled {sampled}");
        assert_eq!(a.margins, b.margins, "vocab {vocab} sampled {sampled}");
    }
}

#[test]
#[ignore = "needs GPU"]
fn sampling_greedy_ties_and_infinities_match_reference_bitwise() {
    let Some((dev, s)) = setup() else { return };
    let k = SamplingKernels::load(dev.context()).expect("load");
    let vocab = 3000usize;
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut base = vec![0.5f32; vocab];
    base[7] = 2.0;
    base[2999] = 2.0;
    base[1500] = 2.0;
    rows.push(base);
    rows.push(vec![1.25; vocab]);
    rows.push(vec![f32::NEG_INFINITY; vocab]);
    let mut single = vec![f32::NEG_INFINITY; vocab];
    single[1234] = -3.0;
    rows.push(single);
    let mut near = vec![-1.0f32; vocab];
    near[0] = 4.0;
    near[1] = 3.9921875;
    rows.push(near);
    let mut rng = Rng(99);
    rows.push(
        (0..vocab)
            .map(|_| (rng.next() * 8.0).round() / 4.0)
            .collect(),
    );
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let bits = to_bf16_bits(&flat);
    let wide = widen(&bits);
    let n = rows.len();
    let masks = TokenMasks::new(n, vocab);
    let params: Vec<RowParams> = (0..n as u32).map(RowParams::greedy).collect();
    let d_params = s.clone_htod(&params).expect("upload");
    let prompt = s.clone_htod(masks.prompt_words()).expect("upload");
    let output = s.clone_htod(masks.output_words()).expect("upload");
    let counts = s.clone_htod(masks.count_words()).expect("upload");
    for bf16_path in [false, true] {
        let mut tokens = s.clone_htod(&vec![0u32; n]).expect("upload");
        let mut margins = s.clone_htod(&vec![0f32; n]).expect("upload");
        if bf16_path {
            let logits = s.clone_htod(&bits).expect("upload");
            k.choose_bf16(
                &s,
                true,
                &logits,
                n,
                vocab,
                &d_params,
                &prompt,
                &output,
                &counts,
                masks.words(),
                &mut tokens,
                &mut margins,
            )
        } else {
            let mut logits = s.clone_htod(&wide).expect("upload");
            k.choose(
                &s,
                true,
                &mut logits,
                n,
                vocab,
                &d_params,
                &prompt,
                &output,
                &counts,
                masks.words(),
                &mut tokens,
                &mut margins,
            )
        }
        .expect("launch");
        let got = picks(&tokens, &margins, &s);
        for r in 0..n {
            let want = scriptor_kernels::reference::sampling::argmax_with_margin(
                &wide[r * vocab..(r + 1) * vocab],
            );
            assert_eq!(got.tokens[r], want.token, "row {r} bf16 {bf16_path}");
            let (g, w) = (f32::from_bits(got.margins[r]), want.margin);
            assert!(
                g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                "row {r} bf16 {bf16_path}: margin {g} vs {w}"
            );
        }
    }
}

/// Logits with small integer-quarter values so equal maxima occur many times, plus planted
/// exact ties at the first, a middle, and the last index.
fn tied_logits(rows: usize, vocab: usize, seed: u64) -> Vec<u16> {
    let mut rng = Rng(seed | 1);
    let mut flat: Vec<f32> = (0..rows * vocab)
        .map(|_| (rng.next() * 6.0).round() / 4.0)
        .collect();
    for r in 0..rows {
        let row = &mut flat[r * vocab..(r + 1) * vocab];
        for idx in [vocab - 1, vocab / 2, (r * 7) % vocab] {
            row[idx] = 9.0;
        }
    }
    to_bf16_bits(&flat)
}

#[test]
#[ignore = "needs GPU"]
fn sampling_bf16_greedy_matches_host_reference_with_ties_and_penalties() {
    let Some((dev, s)) = setup() else { return };
    let k = SamplingKernels::load(dev.context()).expect("load");
    for vocab in [59_392usize, 777] {
        for rows in [1usize, 2, 32, 128] {
            for frequency_on in [false, true] {
                let mut c = build(rows, vocab, rows as u64 * 31 + vocab as u64, false);
                for (r, p) in c.params.iter_mut().enumerate() {
                    p.frequency = if frequency_on {
                        0.1 * (1 + r % 3) as f32
                    } else {
                        0.0
                    };
                    p.repetition = [1.0, 1.3, 1.0, 0.8][r % 4];
                    p.presence = [0.0, 0.0, 0.5, 0.25][r % 4];
                }
                let bits = tied_logits(rows, vocab, vocab as u64 ^ rows as u64);
                let wide = widen(&bits);
                let params = s.clone_htod(&c.params).expect("upload");
                let prompt = s.clone_htod(c.masks.prompt_words()).expect("upload");
                let output = s.clone_htod(c.masks.output_words()).expect("upload");
                let counts = s.clone_htod(c.masks.count_words()).expect("upload");
                let logits = s.clone_htod(&bits).expect("upload");
                let mut tokens = s.clone_htod(&vec![0u32; rows]).expect("upload");
                let mut margins = s.clone_htod(&vec![0f32; rows]).expect("upload");
                k.choose_bf16(
                    &s,
                    true,
                    &logits,
                    rows,
                    vocab,
                    &params,
                    &prompt,
                    &output,
                    &counts,
                    c.masks.words(),
                    &mut tokens,
                    &mut margins,
                )
                .expect("launch");
                let got = picks(&tokens, &margins, &s);
                for r in 0..rows {
                    let want =
                        process_row(&wide[r * vocab..(r + 1) * vocab], &c.masks, &c.params[r]);
                    let what = format!("rows {rows} vocab {vocab} freq {frequency_on} row {r}");
                    assert_eq!(got.tokens[r], want.token, "{what}");
                    assert_eq!(got.margins[r], want.margin.to_bits(), "{what}");
                }
            }
        }
    }
}

#[test]
#[ignore = "needs GPU"]
fn sampling_bf16_spec_greedy_matches_host_reference_with_drafts() {
    let Some((dev, s)) = setup() else { return };
    let kern = SamplingKernels::load(dev.context()).expect("load");
    let (seqs, k) = (32usize, 3usize);
    let q = k + 1;
    for vocab in [59_392usize, 777] {
        for frequency_on in [false, true] {
            let mut c = build(seqs, vocab, 5 + vocab as u64, false);
            for (r, p) in c.params.iter_mut().enumerate() {
                p.frequency = if frequency_on { 0.2 } else { 0.0 };
                p.repetition = [1.0, 1.2][r % 2];
                p.presence = [0.3, 0.0][(r / 2) % 2];
            }
            let rows = seqs * q;
            let bits = tied_logits(rows, vocab, 17 + vocab as u64);
            let wide = widen(&bits);
            let drafts: Vec<u32> = (0..seqs * k)
                .map(|i| ((i * 37 + 5) % vocab) as u32)
                .collect();
            let params = s.clone_htod(&c.params).expect("upload");
            let prompt = s.clone_htod(c.masks.prompt_words()).expect("upload");
            let output = s.clone_htod(c.masks.output_words()).expect("upload");
            let counts = s.clone_htod(c.masks.count_words()).expect("upload");
            let d_drafts = s.clone_htod(&drafts).expect("upload");
            let logits = s.clone_htod(&bits).expect("upload");
            let mut tokens = s.clone_htod(&vec![0u32; rows]).expect("upload");
            let mut margins = s.clone_htod(&vec![0f32; rows]).expect("upload");
            kern.choose_spec_bf16(
                &s,
                &logits,
                seqs,
                k,
                vocab,
                &params,
                &prompt,
                &output,
                &counts,
                c.masks.words(),
                &d_drafts,
                &mut tokens,
                &mut margins,
            )
            .expect("launch");
            let got = picks(&tokens, &margins, &s);
            for row in 0..rows {
                let (seq, ahead) = (row / q, row % q);
                let mut masks = c.masks.clone();
                for t in 0..ahead {
                    masks.push_output(c.params[seq].slot as usize, drafts[seq * k + t]);
                }
                let want = process_row(
                    &wide[row * vocab..(row + 1) * vocab],
                    &masks,
                    &c.params[seq],
                );
                let what = format!("vocab {vocab} freq {frequency_on} row {row}");
                assert_eq!(got.tokens[row], want.token, "{what}");
                assert_eq!(got.margins[row], want.margin.to_bits(), "{what}");
            }
        }
    }
}

#[test]
#[ignore = "needs GPU"]
fn sampling_bf16_spec_microbench_prints_microseconds_per_call() {
    let Some((dev, s)) = setup() else { return };
    let kern = SamplingKernels::load(dev.context()).expect("load");
    let (seqs, k, vocab) = (32usize, 3usize, 59_392usize);
    let rows = seqs * (k + 1);
    let c = build(seqs, vocab, 3, false);
    let bits = tied_logits(rows, vocab, 11);
    let drafts: Vec<u32> = (0..seqs * k)
        .map(|i| ((i * 37 + 5) % vocab) as u32)
        .collect();
    let params = s.clone_htod(&c.params).expect("upload");
    let prompt = s.clone_htod(c.masks.prompt_words()).expect("upload");
    let output = s.clone_htod(c.masks.output_words()).expect("upload");
    let counts = s.clone_htod(c.masks.count_words()).expect("upload");
    let d_drafts = s.clone_htod(&drafts).expect("upload");
    let logits = s.clone_htod(&bits).expect("upload");
    let mut tokens = s.clone_htod(&vec![0u32; rows]).expect("upload");
    let mut margins = s.clone_htod(&vec![0f32; rows]).expect("upload");
    let mut launch = || {
        kern.choose_spec_bf16(
            &s,
            &logits,
            seqs,
            k,
            vocab,
            &params,
            &prompt,
            &output,
            &counts,
            c.masks.words(),
            &d_drafts,
            &mut tokens,
            &mut margins,
        )
        .expect("launch")
    };
    for _ in 0..20 {
        launch();
    }
    s.synchronize().expect("sync");
    let iters = 400;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        launch();
    }
    s.synchronize().expect("sync");
    eprintln!(
        "SAMPLER_US_PER_CALL rows={rows} {:.1}",
        t.elapsed().as_secs_f64() * 1e6 / iters as f64
    );
}
