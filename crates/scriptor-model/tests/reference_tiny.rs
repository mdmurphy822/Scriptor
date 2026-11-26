//! The f32 reference forward pass against float64 goldens on the tiny synthetic model.

use std::path::PathBuf;

use scriptor_kernels::reference::sampling::Penalties;
use scriptor_model::ReferenceModel;
use scriptor_spec::ModelSpec;
use serde::Deserialize;

const TOLERANCE: f32 = 1e-4;
const CASES: [&str; 4] = [
    "one_token_image",
    "two_token_image",
    "four_token_image",
    "two_images",
];

#[derive(Deserialize)]
struct Golden {
    input_ids: Vec<u32>,
    grid_thw: Vec<[usize; 3]>,
    patches: Vec<f32>,
    image_embeds: Vec<f32>,
    positions: Vec<[i32; 3]>,
    mrope_delta: i64,
    hidden: Vec<f32>,
    prefill_logits: Vec<f32>,
    repetition_penalty: f32,
    greedy_tokens: Vec<u32>,
    greedy_margins: Vec<f32>,
    decode_logits: Vec<f32>,
    mtp_hidden: Vec<f32>,
    mtp_logits: Vec<f32>,
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn model() -> ReferenceModel {
    let config = std::fs::read_to_string(fixtures().join("config/tiny.config.json")).unwrap();
    let spec = ModelSpec::from_json(&config, None).unwrap();
    ReferenceModel::load(fixtures().join("model/tiny"), &spec).unwrap()
}

fn golden(name: &str) -> Golden {
    let path = fixtures().join(format!("model/tiny/goldens/{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn max_diff(label: &str, got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "{label}: length");
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    let diff = max_diff(label, got, want);
    assert!(
        diff < TOLERANCE,
        "{label}: max |diff| {diff} >= {TOLERANCE}"
    );
}

fn grids(g: &Golden) -> Vec<(usize, usize, usize)> {
    g.grid_thw.iter().map(|&[t, h, w]| (t, h, w)).collect()
}

#[test]
fn reference_image_embeddings_match_float64() {
    let model = model();
    for name in CASES {
        let g = golden(name);
        let embeds = model.encode_images(&g.patches, &grids(&g)).unwrap();
        assert_close(&format!("{name} image embeds"), &embeds, &g.image_embeds);
    }
}

#[test]
fn reference_prefill_matches_float64() {
    let model = model();
    let hidden = model.text.hidden;
    for name in CASES {
        let g = golden(name);
        let embeds = model.input_embeds(&g.input_ids, &g.image_embeds).unwrap();
        let mut cache = model.text.new_cache();
        let out = model.text.forward(&embeds, &g.positions, &mut cache);
        assert_close(&format!("{name} hidden"), &out, &g.hidden);
        assert_eq!(out.len(), g.input_ids.len() * hidden);
        assert_close(
            &format!("{name} logits"),
            &model.text.logits(&out),
            &g.prefill_logits,
        );
    }
}

#[test]
fn reference_chunked_prefill_equals_single_pass() {
    let model = model();
    for name in CASES {
        let g = golden(name);
        let embeds = model.input_embeds(&g.input_ids, &g.image_embeds).unwrap();
        let hidden = model.text.hidden;
        let split = g.input_ids.len() / 2;
        let mut cache = model.text.new_cache();
        let mut out =
            model
                .text
                .forward(&embeds[..split * hidden], &g.positions[..split], &mut cache);
        out.extend(model.text.forward(
            &embeds[split * hidden..],
            &g.positions[split..],
            &mut cache,
        ));
        assert_close(&format!("{name} chunked"), &out, &g.hidden);
    }
}

#[test]
fn reference_greedy_decode_matches_float64() {
    let model = model();
    let vocab = model.text.vocab;
    for name in CASES {
        let g = golden(name);
        let penalties = Penalties {
            repetition: g.repetition_penalty,
            ..Penalties::default()
        };
        let out = model
            .generate(
                &g.input_ids,
                &g.image_embeds,
                &g.positions,
                g.mrope_delta,
                g.greedy_tokens.len(),
                penalties,
            )
            .unwrap();
        let tokens: Vec<u32> = out.steps.iter().map(|s| s.token).collect();
        assert_eq!(tokens, g.greedy_tokens, "{name} tokens");
        for (step, want) in out.steps.iter().zip(&g.greedy_margins) {
            assert!(
                (step.margin - want).abs() < 1e-3,
                "{name} margin {} vs {want}",
                step.margin
            );
        }

        // Teacher-forced logits through the cache match the golden decode logits.
        let embeds = model.input_embeds(&g.input_ids, &g.image_embeds).unwrap();
        let mut cache = model.text.new_cache();
        let mut hidden = model.text.forward(&embeds, &g.positions, &mut cache);
        for (step, &tok) in g.greedy_tokens.iter().enumerate() {
            let seq_len = g.input_ids.len() + step;
            let last = &hidden[hidden.len() - model.text.hidden..];
            let logits = model.text.logits(last);
            assert_close(
                &format!("{name} decode step {step}"),
                &logits,
                &g.decode_logits[step * vocab..(step + 1) * vocab],
            );
            let at = (seq_len as i64 + g.mrope_delta) as i32;
            let embed = model.input_embeds(&[tok], &[]).unwrap();
            hidden = model.text.forward(&embed, &[[at; 3]], &mut cache);
        }
    }
}

#[test]
fn reference_mtp_matches_float64() {
    let model = model();
    let mtp = model.mtp.as_ref().expect("tiny model has an MTP layer");
    for name in CASES {
        let g = golden(name);
        let n = g.input_ids.len() - 1;
        let mut cache = mtp.new_cache();
        let hidden = mtp.forward(
            &g.input_ids[1..],
            &g.hidden[..n * mtp.hidden],
            &g.positions[..n],
            &mut cache,
        );
        assert_close(&format!("{name} mtp hidden"), &hidden, &g.mtp_hidden);
        assert_close(
            &format!("{name} mtp logits"),
            &mtp.logits(&hidden),
            &g.mtp_logits,
        );
    }
}

#[test]
fn reference_mtp_zeroes_embedding_at_position_zero() {
    let model = model();
    let mtp = model.mtp.as_ref().unwrap();
    let g = golden("one_token_image");
    let hidden = &g.hidden[..mtp.hidden];
    let at_zero = mtp.forward(&[7], hidden, &[[0; 3]], &mut mtp.new_cache());
    let at_zero_other_token = mtp.forward(&[9], hidden, &[[0; 3]], &mut mtp.new_cache());
    assert_eq!(
        at_zero, at_zero_other_token,
        "token embedding must not matter at position 0"
    );
    let later = mtp.forward(&[7], hidden, &[[1; 3]], &mut mtp.new_cache());
    let later_other = mtp.forward(&[9], hidden, &[[1; 3]], &mut mtp.new_cache());
    assert_ne!(later, later_other);
}

#[test]
fn reference_rejects_mismatched_inputs() {
    let model = model();
    let g = golden("one_token_image");
    assert!(model.encode_images(&g.patches[1..], &grids(&g)).is_err());
    assert!(model.input_embeds(&g.input_ids, &[]).is_err());
    assert!(model.input_embeds(&[u32::MAX], &[]).is_err());
}
