//! CPU checks of the resident-block case table the GPU equality tests share: the per-test
//! partitions cover the full table exactly once, the lengths stay within bounds, and every
//! kernel module the tests load has a source file.
#[path = "attn_resident_table/mod.rs"]
mod table;

use std::collections::HashSet;

use scriptor_kernels::attention::{AttnResidency, decode_module_name};
use table::{BATCHES, LEN_KINDS, LenKind, MAX_LEN, Rng};

#[test]
fn resident3_case_table_is_exhaustive() {
    let mut union = Vec::new();
    for (batch, kind) in table::tests() {
        union.extend(table::partition(batch, kind));
    }
    let unique: HashSet<_> = union.iter().copied().collect();
    assert_eq!(unique.len(), union.len(), "a case appears in two tests");
    let full = table::full();
    let full_set: HashSet<_> = full.iter().copied().collect();
    assert_eq!(full_set.len(), full.len(), "the full table repeats a case");
    assert_eq!(unique, full_set);
    assert_eq!(table::tests().len(), BATCHES.len() * LEN_KINDS.len());
}

#[test]
fn resident3_table_covers_the_specified_axes() {
    assert_eq!(table::BATCHES, [1, 8, 16, 32, 64, 128]);
    assert_eq!(table::BLOCKS, [16, 32]);
    assert_eq!(table::SM_COUNTS, [48, 12, 8]);
    assert_eq!(table::CTX_MULTIPLIERS, [1, 3]);
    assert_eq!(table::q_lens(8), vec![1, 2, 4, 8]);
    for batch in [1, 16, 32, 64, 128] {
        assert_eq!(table::q_lens(batch), vec![1, 2, 4]);
    }
    let full = table::full();
    assert!(full.iter().any(|c| c.ordered) && full.iter().any(|c| !c.ordered));
    // 2 variants x 2 blocks x 2 seeds x 2 ctx x 3 sm x 3 kinds x (5 batches x 3 + 4 q_lens).
    assert_eq!(full.len(), 2 * 2 * 2 * 2 * 3 * 3 * (5 * 3 + 4));
}

#[test]
fn resident3_lengths_hold_the_new_rows_and_stay_bounded() {
    for &batch in &BATCHES {
        for q_len in table::q_lens(batch) {
            for kind in LEN_KINDS {
                for seed in 0..table::SEED_SETS.len() {
                    let mut rng = Rng(table::rng_seed(seed, batch, q_len));
                    let lens = table::lengths(&mut rng, kind, batch, q_len);
                    assert_eq!(lens.len(), batch);
                    assert!(lens.iter().all(|&l| l >= q_len && l <= MAX_LEN));
                    if kind == LenKind::Tail {
                        assert_eq!(lens.iter().max(), Some(&MAX_LEN));
                    }
                }
            }
        }
    }
}

#[test]
fn resident3_modules_have_kernel_sources() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
    for residency in [AttnResidency::Two, AttnResidency::Three] {
        for ordered in [false, true] {
            let name = decode_module_name(residency, ordered);
            assert!(dir.join(format!("{name}.cu")).is_file(), "{name}");
        }
    }
}
