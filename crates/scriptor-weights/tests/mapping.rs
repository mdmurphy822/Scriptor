use std::collections::{BTreeMap, BTreeSet};

use scriptor_weights::{
    NameLayout, ParamName, Placement, WeightsError, all_params, classify, fused_targets,
};

const LAYOUT: NameLayout = NameLayout {
    text_layers: 16,
    vision_blocks: 24,
};

fn fixture_names() -> Vec<String> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/weights/glm-ocr-header.json"
    );
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    v.as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn mapping_classifies_every_header_name() {
    let names = fixture_names();
    assert_eq!(names.len(), 526);
    let mut parts: BTreeMap<_, BTreeSet<usize>> = BTreeMap::new();
    let mut direct = BTreeSet::new();
    for n in &names {
        match classify(n, LAYOUT).unwrap_or_else(|e| panic!("{n}: {e}")) {
            Placement::Direct(p) => {
                assert_eq!(p.checkpoint_name(LAYOUT), *n);
                assert!(direct.insert(p), "{n} classified twice");
            }
            Placement::Part { target, index } => {
                assert!(parts.entry(target).or_default().insert(index));
                assert_eq!(target.sources()[index].checkpoint_name(LAYOUT), *n);
            }
        }
    }
    let targets = fused_targets(LAYOUT);
    assert_eq!(
        parts.keys().copied().collect::<BTreeSet<_>>(),
        targets.iter().copied().collect()
    );
    for (t, idx) in &parts {
        assert_eq!(idx.len(), t.sources().len(), "{t:?} incomplete");
    }
}

#[test]
fn mapping_covers_exactly_the_implied_parameters() {
    let names: BTreeSet<String> = fixture_names().into_iter().collect();
    let implied: BTreeSet<String> = all_params(LAYOUT)
        .iter()
        .map(|p| p.checkpoint_name(LAYOUT))
        .collect();
    assert_eq!(names, implied);
}

#[test]
fn mapping_rejects_unknown_names() {
    for bad in [
        "model.language_model.layers.17.self_attn.q_proj.weight",
        "model.language_model.layers.3.enorm.weight",
        "model.visual.blocks.24.norm1.weight",
        "model.visual.merger.proj.bias",
        "foo",
    ] {
        assert!(
            matches!(
                ParamName::parse(bad, LAYOUT),
                Err(WeightsError::UnmappedTensor(_))
            ),
            "{bad}"
        );
    }
}
