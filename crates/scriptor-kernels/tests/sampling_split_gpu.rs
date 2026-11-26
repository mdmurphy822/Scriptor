//! GPU checks of the two-stage greedy target selector against the single-block kernel: tokens
//! and margin bits must be identical for every partition choice. The inputs here are synthetic
//! (text-shaped logits built from public-domain prose); logits from a real checkpoint are
//! checked in `sampling_split_real_gpu`, and costs are measured in `sampling_split_timing_gpu`.
//! Run with
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda --profile gpu-test --test sampling_split_gpu -- --ignored --nocapture --test-threads=1`.
#![cfg(feature = "cuda")]

mod common;

use common::*;
use scriptor_kernels::sampling::{RowParams, SamplingKernels, SplitSelect};

#[test]
#[ignore = "needs GPU"]
fn split_equals_original_on_synthetic_text_shaped_rows_with_penalties_and_duplicate_drafts() {
    let Some((dev, s)) = setup() else { return };
    let kern = SamplingKernels::load(dev.context()).expect("load");
    for (seqs, k) in [
        (1, 0),
        (2, 0),
        (4, 0),
        (8, 0),
        (1, 3),
        (2, 3),
        (1, 7),
        (3, 2),
    ] {
        for vocab in [VOCAB, 59_390, 777, 33] {
            let b = text_batch(seqs, k, vocab, (seqs * 100 + k * 10 + vocab) as u64);
            assert_equal_to_original(&kern, &s, &b, &format!("seqs {seqs} k {k} vocab {vocab}"));
        }
    }
}

/// Builds a one-sequence-per-row batch (`k = 0`) over the given bf16 rows.
fn special_batch(rows: &[Vec<f32>]) -> Batch {
    let vocab = rows[0].len();
    let mut b = text_batch(rows.len(), 0, vocab, 7);
    b.bits = rows.iter().flat_map(|r| bf16_bits(r)).collect();
    for p in &mut b.params {
        p.repetition = 1.0;
        p.presence = 0.0;
        p.frequency = 0.0;
    }
    b
}

#[test]
#[ignore = "needs GPU"]
fn split_equals_original_on_ties_signed_zeros_infinities_nans_and_empty_rows() {
    let Some((dev, s)) = setup() else { return };
    let kern = SamplingKernels::load(dev.context()).expect("load");
    for vocab in [3000usize, 3001, 1024, 8] {
        let mut rows: Vec<Vec<f32>> = Vec::new();
        let mut base = vec![0.5f32; vocab];
        base[vocab - 1] = 2.0;
        base[vocab / 2] = 2.0;
        base[0] = 2.0;
        rows.push(base);
        rows.push(
            (0..vocab)
                .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
                .collect(),
        );
        rows.push(vec![-0.0; vocab]);
        let mut inf = vec![1.0f32; vocab];
        inf[vocab / 3] = f32::INFINITY;
        inf[vocab - 2] = f32::INFINITY;
        rows.push(inf);
        rows.push(vec![f32::NEG_INFINITY; vocab]);
        rows.push(vec![f32::INFINITY; vocab]);
        rows.push(vec![f32::NAN; vocab]);
        let mut nan = vec![-3.0f32; vocab];
        nan[vocab / 4] = f32::NAN;
        nan[vocab / 2] = 1.0;
        nan[vocab - 1] = 1.0;
        rows.push(nan);
        for chunk in rows.chunks(4) {
            let b = special_batch(chunk);
            assert_equal_to_original(&kern, &s, &b, &format!("special vocab {vocab}"));
        }
        let mut penalised = special_batch(&rows[..4]);
        for (i, p) in penalised.params.iter_mut().enumerate() {
            p.repetition = 1.25;
            p.presence = 0.5 * (i % 2) as f32;
            p.frequency = 0.25;
        }
        assert_equal_to_original(&kern, &s, &penalised, &format!("special pen vocab {vocab}"));
    }
}

#[test]
#[ignore = "needs GPU"]
fn split_equals_original_with_permuted_slots_and_inactive_padding() {
    let Some((dev, s)) = setup() else { return };
    let kern = SamplingKernels::load(dev.context()).expect("load");
    for (seqs, k, active) in [(4usize, 3usize, 2usize), (8, 0, 3), (2, 7, 1)] {
        let mut b = text_batch(seqs, k, VOCAB, 4242);
        let slots = b.masks.slots();
        let mut order: Vec<usize> = (0..seqs).collect();
        let mut rng = Rng(0x51ab);
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let params = b.params.clone();
        for (i, &o) in order.iter().enumerate() {
            b.params[i] = params[o];
            b.params[i].slot = params[order[(i + 1) % seqs]].slot % slots as u32;
        }
        for p in b.params.iter_mut().skip(active) {
            *p = RowParams::greedy(p.slot);
        }
        for d in b.drafts.iter_mut().skip(active * k) {
            *d = 0;
        }
        assert_equal_to_original(&kern, &s, &b, &format!("seqs {seqs} k {k} active {active}"));
    }
}

#[test]
#[ignore = "needs GPU"]
fn split_declines_outside_its_bound_and_with_a_small_scratch() {
    let Some((dev, s)) = setup() else { return };
    let kern = SamplingKernels::load(dev.context()).expect("load");
    let b = text_batch(4, 3, 4096, 5);
    let mut d = upload(&s, &b);
    let mut scratch = kern.alloc_split_scratch(&s, 64).expect("scratch");
    let narrow = SplitSelect {
        block_threads: 128,
        ctas_per_row: 8,
        max_rows: 8,
    };
    assert!(!split(&kern, &s, &narrow, &mut scratch, &b, &mut d));
    let mut small = kern.alloc_split_scratch(&s, 1).expect("scratch");
    assert!(!split(&kern, &s, &select(128, 8), &mut small, &b, &mut d));
    assert!(split(&kern, &s, &select(128, 8), &mut scratch, &b, &mut d));
}
