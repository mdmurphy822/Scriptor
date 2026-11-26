//! GPU checks of the two-stage greedy target selector on bf16 logits dumped from a real
//! GLM-OCR checkpoint. Unlike the synthetic cases of `sampling_split_gpu`, every row here is a
//! checkpoint-produced activation with the prompt, output history, penalties and drafts it was
//! chosen under.
//!
//! With `SCRIPTOR_GPU_TESTS=1` the tests fail unless `SCRIPTOR_SPLIT_FIXTURE_DIR` names a
//! directory holding two files in the logits-fixture format of
//! `scriptor_kernels::logits_fixture`:
//!
//! * `plain.logits`: groups of one row (`q = 1`), at least eight groups;
//! * `spec.logits`: groups of `q >= 2` rows with their drafts, at least two groups.
//!
//! Produce both from a checkpoint with
//! `SCRIPTOR_GPU_TESTS=1 SCRIPTOR_MODEL_DIR=<snapshot> SCRIPTOR_SPLIT_FIXTURE_DIR=<out> cargo test -p scriptor-cli --features cuda --profile gpu-test dump_real_logits_fixtures -- --ignored --nocapture --test-threads=1`.
//!
//! Run these checks with
//! `SCRIPTOR_GPU_TESTS=1 SCRIPTOR_SPLIT_FIXTURE_DIR=<dir> cargo test -p scriptor-kernels --features cuda --profile gpu-test --test sampling_split_real_gpu -- --ignored --nocapture --test-threads=1`.
#![cfg(feature = "cuda")]

mod common;

use common::*;
use scriptor_kernels::logits_fixture::LogitsFixture;
use scriptor_kernels::sampling::SamplingKernels;

/// Reads fixture `name`, failing the test when the GPU gate is on and the file is missing or
/// malformed. `None` when GPU tests are not enabled.
fn require(name: &str) -> Option<LogitsFixture> {
    setup()?;
    let dir = std::env::var("SCRIPTOR_SPLIT_FIXTURE_DIR")
        .expect("SCRIPTOR_SPLIT_FIXTURE_DIR must name the real-model logits fixture directory");
    let path = std::path::Path::new(&dir).join(name);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("real-model fixture {} is unreadable: {e}", path.display()));
    let fixture = LogitsFixture::decode(&bytes)
        .unwrap_or_else(|e| panic!("real-model fixture {} is malformed: {e}", path.display()));
    let model = fixture.provenance_value("model").unwrap_or_default();
    assert!(
        model.to_ascii_lowercase().contains("glm-ocr"),
        "fixture {} was not dumped from a GLM-OCR checkpoint (model = {model:?})",
        path.display()
    );
    Some(fixture)
}

fn check_prefix(name: &str, groups: usize) {
    let Some(f) = require(name) else { return };
    assert!(
        f.groups.len() >= groups,
        "{name} holds {} groups, {groups} required",
        f.groups.len()
    );
    let (dev, s) = setup().expect("gpu");
    let kern = SamplingKernels::load(dev.context()).expect("load");
    let b = fixture_batch(&f, groups);
    assert_equal_to_original(&kern, &s, &b, &format!("{name} first {groups} groups"));
}

#[test]
#[ignore = "needs GPU and a real-model fixture"]
fn real_fixtures_are_present_well_formed_and_shaped() {
    let Some(plain) = require("plain.logits") else {
        return;
    };
    assert_eq!(plain.q, 1, "plain.logits holds single-row groups");
    assert!(
        plain.groups.len() >= 8,
        "plain.logits needs at least 8 groups"
    );
    let spec = require("spec.logits").expect("gpu enabled");
    assert!(spec.q >= 2, "spec.logits holds speculative groups");
    assert!(
        spec.groups.len() >= 2,
        "spec.logits needs at least 2 groups"
    );
    assert_eq!(
        plain.vocab, spec.vocab,
        "fixtures disagree on the vocabulary"
    );
}

macro_rules! plain_prefix {
    ($name:ident, $n:expr) => {
        #[test]
        #[ignore = "needs GPU and a real-model fixture"]
        fn $name() {
            check_prefix("plain.logits", $n);
        }
    };
}

plain_prefix!(split_equals_original_on_real_logits_first_1_row, 1);
plain_prefix!(split_equals_original_on_real_logits_first_2_rows, 2);
plain_prefix!(split_equals_original_on_real_logits_first_3_rows, 3);
plain_prefix!(split_equals_original_on_real_logits_first_4_rows, 4);
plain_prefix!(split_equals_original_on_real_logits_first_5_rows, 5);
plain_prefix!(split_equals_original_on_real_logits_first_6_rows, 6);
plain_prefix!(split_equals_original_on_real_logits_first_7_rows, 7);
plain_prefix!(split_equals_original_on_real_logits_first_8_rows, 8);

#[test]
#[ignore = "needs GPU and a real-model fixture"]
fn split_equals_original_on_real_speculative_group_of_one_sequence() {
    check_prefix("spec.logits", 1);
}

#[test]
#[ignore = "needs GPU and a real-model fixture"]
fn split_equals_original_on_real_speculative_groups_of_two_sequences() {
    check_prefix("spec.logits", 2);
}

#[test]
#[ignore = "needs GPU and a real-model fixture"]
fn split_equals_original_on_every_real_speculative_group_within_the_row_bound() {
    let Some(f) = require("spec.logits") else {
        return;
    };
    check_prefix("spec.logits", f.groups.len().min(64 / f.q));
}

#[test]
#[ignore = "needs GPU and a real-model fixture"]
fn real_history_and_drafts_change_the_selection() {
    let Some(f) = require("spec.logits") else {
        return;
    };
    let (dev, s) = setup().expect("gpu");
    let kern = SamplingKernels::load(dev.context()).expect("load");
    let groups = f.groups.len().min(64 / f.q);
    let with = fixture_batch(&f, groups);
    let mut without = fixture_batch(&f, groups);
    for p in &mut without.params {
        p.repetition = 1.0;
        p.presence = 0.0;
        p.frequency = 0.0;
    }
    let run = |b: &Batch| {
        let mut d = upload(&s, b);
        original(&kern, &s, b, &mut d);
        result(&s, &d)
    };
    assert_ne!(
        run(&with),
        run(&without),
        "the fixture's penalties and history never changed a target or margin: it does not \
         exercise the penalty path"
    );
}
