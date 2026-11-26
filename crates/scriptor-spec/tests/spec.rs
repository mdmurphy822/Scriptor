use scriptor_spec::{
    ImageProcessorConfig, MemoryPlan, MemoryPlanInputs, ModelSpec, RuntimeParams, SamplingDefaults,
    SpecError, VisionBatchPatches,
};

fn fixture(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/config")
        .join(name);
    std::fs::read_to_string(path).expect("fixture")
}

fn real_spec() -> ModelSpec {
    ModelSpec::from_json(
        &fixture("glm-ocr.config.json"),
        Some(&fixture("glm-ocr.generation_config.json")),
    )
    .unwrap()
}

fn real_image() -> ImageProcessorConfig {
    ImageProcessorConfig::from_preprocessor_json(&fixture("glm-ocr.preprocessor_config.json"))
        .unwrap()
}

const BLOCK: u64 = 16;
const GB: u64 = 1_000_000_000;
const WEIGHTS: u64 = 2_650_579_464;
const SPARK: u64 = 128 * 1024 * 1024 * 1024;

#[test]
fn real_config_matches_architecture() {
    let s = real_spec();
    assert_eq!(s.text.hidden_size, 1536);
    assert_eq!(s.text.intermediate_size, 4608);
    assert_eq!(s.text.num_layers, 16);
    assert_eq!(
        (s.text.num_heads, s.text.num_kv_heads, s.text.head_dim),
        (16, 8, 128)
    );
    assert_eq!(s.text.mrope_section, [16, 24, 24]);
    assert!(!s.text.is_neox_style);
    assert!(s.text.sandwich_norms);
    assert_eq!(s.text.rotary_dim, 128);
    assert_eq!(s.text.eos_token_ids, vec![59246, 59253]);
    assert_eq!(s.text.vocab_size, 59392);
    assert_eq!(s.text.num_mtp_layers, 1);
    assert_eq!(
        (
            s.image_token_id,
            s.image_start_token_id,
            s.image_end_token_id
        ),
        (59280, 59256, 59257)
    );
    assert_eq!(s.vision.depth, 24);
    assert_eq!(
        (s.vision.hidden_size, s.vision.num_heads, s.vision.head_dim),
        (1024, 16, 64)
    );
    assert_eq!(s.vision.rotary_dim, 32);
    assert!(s.vision.is_neox_style && s.vision.qk_norm);
    assert_eq!(s.vision.patch_dim(), 1176);
    assert_eq!(s.vision.out_hidden_size * 3, s.text.intermediate_size);
    assert_eq!(s.kv_bytes_per_token(), 65536);
    assert_eq!(s.mtp_kv_bytes_per_token(), 4096);
}

#[test]
fn real_image_processor_values() {
    let from_pre = real_image();
    let from_proc =
        ImageProcessorConfig::from_processor_json(&fixture("glm-ocr.processor_config.json"))
            .unwrap();
    assert_eq!(from_pre, from_proc);
    assert_eq!(from_pre.min_pixels, 12544);
    assert_eq!(from_pre.max_pixels, 9_633_792);
    assert_eq!(from_pre.patch_size, 14);
    assert_eq!(from_pre.merge_size, 2);
    assert_eq!(from_pre.temporal_patch_size, 2);
    assert_eq!(from_pre.max_image_tokens(), 12288);
}

#[test]
fn do_sample_false_resolves_to_greedy_defaults() {
    let s = real_spec();
    assert_eq!(s.sampling_defaults.temperature, Some(0.0));
    assert_eq!(s.sampling_defaults.top_p, None);
    let none = ModelSpec::from_json(&fixture("tiny.config.json"), None).unwrap();
    assert_eq!(none.sampling_defaults, SamplingDefaults::default());
    let sampled = ModelSpec::from_json(
        &fixture("glm-ocr.config.json"),
        Some(r#"{"eos_token_id":[1],"do_sample":true,"temperature":0.6,"top_p":0.9,"top_k":20,"repetition_penalty":1.1}"#),
    )
    .unwrap();
    assert_eq!(sampled.sampling_defaults.temperature, Some(0.6));
    assert_eq!(sampled.sampling_defaults.top_p, Some(0.9));
    assert_eq!(sampled.sampling_defaults.top_k, Some(20));
    assert_eq!(sampled.sampling_defaults.repetition_penalty, Some(1.1));
}

#[test]
fn tiny_config_parses() {
    let s = ModelSpec::from_json(&fixture("tiny.config.json"), None).unwrap();
    let img =
        ImageProcessorConfig::from_preprocessor_json(&fixture("tiny.preprocessor_config.json"))
            .unwrap();
    assert_eq!(s.text.num_layers, 2);
    assert_eq!(s.vision.head_dim, 16);
    assert_eq!(s.vision.rotary_dim, 8);
    assert_eq!(img.max_image_tokens(), 128);
}

#[test]
fn merger_mismatch_is_rejected() {
    let text = fixture("tiny.config.json")
        .replace("\"intermediate_size\": 144", "\"intermediate_size\": 96");
    assert!(matches!(
        ModelSpec::from_json(&text, None),
        Err(SpecError::Invalid(m)) if m.contains("merger")
    ));
}

fn plan_at(concurrency: usize, context: usize, share: f64) -> Result<MemoryPlan, SpecError> {
    let params = RuntimeParams {
        gpu_share: share,
        max_concurrency: concurrency,
        max_context: context,
        speculative_tokens: 3,
        ..RuntimeParams::default()
    };
    MemoryPlan::compute(
        &real_spec(),
        &real_image(),
        &params,
        MemoryPlanInputs {
            weight_bytes: WEIGHTS,
            device_total_bytes: SPARK,
            free_bytes: None,
        },
    )
}

#[test]
fn oversized_concurrency_plans_instead_of_refusing() {
    for c in [32usize, 64, 128, 256] {
        let p = plan_at(c, 16384, 0.60).unwrap_or_else(|e| panic!("c{c}: {e}"));
        assert!(p.total <= p.available, "c{c}");
        assert!(
            p.available - p.total < p.blocks_per_seq * 70_000_000,
            "c{c}"
        );
        assert_eq!(p.full_context_seqs, p.kv_blocks / p.blocks_per_seq);
        assert!(p.full_context_seqs >= 1);
    }
}

#[test]
fn kv_blocks_follow_the_share_remainder_not_concurrency() {
    let a = plan_at(32, 16384, 0.60).unwrap();
    let b = plan_at(64, 16384, 0.60).unwrap();
    let c = plan_at(128, 16384, 0.60).unwrap();
    // Slot-sized buffers differ, so the remainder shrinks as slots grow.
    assert!(a.step_workspace < b.step_workspace && b.step_workspace < c.step_workspace);
    assert!(a.kv_blocks > b.kv_blocks && b.kv_blocks > c.kv_blocks);
    let fixed = |p: &MemoryPlan| {
        p.weights
            + p.embedding_cache
            + p.vision_workspace
            + p.logits_workspace
            + p.step_workspace
            + p.graph_memory
    };
    let per_block = BLOCK * (65536 + 4096);
    for p in [&a, &b, &c] {
        assert_eq!(p.kv_blocks, (p.available - fixed(p)) / per_block);
        assert_eq!(p.kv_cache, p.kv_blocks * BLOCK * 65536);
        assert_eq!(p.mtp_kv_cache, p.kv_blocks * BLOCK * 4096);
    }
}

#[test]
fn kv_blocks_barely_depend_on_context_length() {
    let small = plan_at(32, 4096, 0.60).unwrap();
    let long = plan_at(32, 16384, 0.60).unwrap();
    // Only the per-sequence block tables in the step workspace differ.
    assert!(small.kv_blocks >= long.kv_blocks);
    assert!((small.kv_blocks - long.kv_blocks) * 1000 < small.kv_blocks);
    assert_eq!(long.full_context_seqs, long.kv_blocks / 1024);
    assert_eq!(small.full_context_seqs, small.kv_blocks / 256);
}

#[test]
fn free_memory_caps_the_budget() {
    let params = RuntimeParams {
        gpu_share: 0.60,
        max_concurrency: 32,
        max_context: 4096,
        ..RuntimeParams::default()
    };
    let compute = |free| {
        MemoryPlan::compute(
            &real_spec(),
            &real_image(),
            &params,
            MemoryPlanInputs {
                weight_bytes: WEIGHTS,
                device_total_bytes: SPARK,
                free_bytes: free,
            },
        )
        .unwrap()
    };
    let open = compute(None);
    let capped = compute(Some(40 * GB));
    assert_eq!(capped.available, 40 * GB);
    assert!(capped.total <= 40 * GB && capped.kv_blocks < open.kv_blocks);
}

#[test]
fn plan_refuses_when_one_full_context_sequence_does_not_fit() {
    let err = plan_at(32, 16384, 0.04).unwrap_err();
    let SpecError::DoesNotFit {
        needed,
        available,
        suggestion,
    } = &err
    else {
        panic!("expected DoesNotFit, got {err}");
    };
    assert_eq!(*available, (SPARK as f64 * 0.04) as u64);
    assert!(needed > available);
    assert!(
        suggestion.contains("short by") || suggestion.contains("fixed buffers alone"),
        "{suggestion}"
    );
    let msg = err.to_string();
    assert!(msg.contains(&needed.to_string()) && msg.contains(&available.to_string()));

    // A share leaving a remainder just below one sequence names the exact shortfall.
    let reference = plan_at(32, 16384, 0.60).unwrap();
    let fixed = reference.total - reference.kv_cache - reference.mtp_kv_cache;
    let one_seq = 1024u64 * BLOCK * (65536 + 4096);
    let share = (fixed + one_seq - 1_000_000) as f64 / SPARK as f64;
    let err = plan_at(32, 16384, share).unwrap_err();
    let SpecError::DoesNotFit { suggestion, .. } = &err else {
        panic!("{err}");
    };
    assert!(suggestion.contains("short by"), "{suggestion}");
    assert!(suggestion.contains("holds 1023 blocks"), "{suggestion}");
}

#[test]
fn modest_plan_fits_and_itemises() {
    let params = RuntimeParams {
        gpu_share: 0.60,
        max_concurrency: 32,
        max_context: 4096,
        speculative_tokens: 1,
        ..RuntimeParams::default()
    };
    let p = MemoryPlan::compute(
        &real_spec(),
        &real_image(),
        &params,
        MemoryPlanInputs {
            weight_bytes: WEIGHTS,
            device_total_bytes: SPARK,
            free_bytes: None,
        },
    )
    .unwrap();
    assert_eq!(p.kv_cache, p.kv_blocks * BLOCK * 65536);
    assert_eq!(p.mtp_kv_cache, p.kv_blocks * BLOCK * 4096);
    assert_eq!(p.logits_workspace, 32 * 2 * 59392 * 4);
    assert_eq!(p.blocks_per_seq, 256);
    assert_eq!(p.full_context_seqs, p.kv_blocks / 256);
    assert!(p.full_context_seqs > 32);
    assert_eq!(
        p.total,
        p.weights
            + p.kv_cache
            + p.mtp_kv_cache
            + p.embedding_cache
            + p.vision_workspace
            + p.logits_workspace
            + p.step_workspace
            + p.graph_memory
    );
    assert!(p.step_workspace > 0 && p.graph_memory > 0);
    assert!(p.total <= p.available);
}

#[test]
fn params_validation_and_parsing() {
    assert!("auto".parse::<VisionBatchPatches>().unwrap() == VisionBatchPatches::Auto);
    assert_eq!(
        "4096".parse::<VisionBatchPatches>().unwrap(),
        VisionBatchPatches::Fixed(4096)
    );
    assert!("0".parse::<VisionBatchPatches>().is_err());
    let bad = RuntimeParams {
        kv_block_tokens: 24,
        ..RuntimeParams::default()
    };
    assert!(bad.validate().is_err());
    let bad = RuntimeParams {
        gpu_share: 1.5,
        ..RuntimeParams::default()
    };
    assert!(bad.validate().is_err());
}

fn derive_at(context: usize, share: f64, speculative: usize) -> Result<usize, SpecError> {
    let params = RuntimeParams {
        gpu_share: share,
        max_context: context,
        speculative_tokens: speculative,
        ..RuntimeParams::default()
    };
    MemoryPlan::derive_concurrency(
        &real_spec(),
        &real_image(),
        &params,
        MemoryPlanInputs {
            weight_bytes: WEIGHTS,
            device_total_bytes: SPARK,
            free_bytes: None,
        },
    )
}

#[test]
fn derived_concurrency_is_the_largest_count_the_pool_serves() {
    for (context, share) in [(16384, 0.35), (16384, 0.60), (4096, 0.60)] {
        let slot_tokens = real_image().max_image_tokens().min(context) as u64;
        let n = derive_at(context, share, 3).unwrap();
        assert!(n >= 1);
        let at = plan_at(n, context, share).unwrap();
        assert!(at.kv_tokens() >= n as u64 * slot_tokens);
        assert!(at.kv_blocks >= at.blocks_per_seq, "one full context fits");
        if let Ok(next) = plan_at(n + 1, context, share) {
            assert!(next.kv_tokens() < (n as u64 + 1) * slot_tokens);
        }
    }
}

#[test]
fn derived_concurrency_exceeds_full_context_slots() {
    let n = derive_at(16384, 0.35, 3).unwrap();
    let at = plan_at(n, 16384, 0.35).unwrap();
    assert!(n as u64 > at.full_context_seqs);
    assert!(
        at.capacity_summary()
            .contains("Maximum concurrency for 16384")
    );
}

#[test]
fn derived_concurrency_grows_with_share_and_shrinks_with_context() {
    let small = derive_at(16384, 0.30, 3).unwrap();
    let large = derive_at(16384, 0.60, 3).unwrap();
    assert!(large > small);
    assert!(derive_at(4096, 0.60, 3).unwrap() > large);
}

#[test]
fn derived_concurrency_refuses_when_one_slot_does_not_fit() {
    assert!(matches!(
        derive_at(16384, 0.01, 3),
        Err(SpecError::DoesNotFit { .. })
    ));
}
