//! Conversion of a safetensors checkpoint directory into an engine file.
//!
//! The stored layout is the one the kernels consume: row-major `[out, in]`
//! weights (the operand order of `C = A * W^T`), text `q/k/v` fused into one
//! `qkv_proj`, vision `gate/up` fused into one `gate_up_proj` (`gate` rows then
//! `up` rows), the patch-embed weight flattened to `[hidden, 3*2*14*14]` in
//! patchify order and the downsample weight flattened to `[out, in*2*2]` in
//! `(channel, row, col)` order. Element types are kept as stored.

use std::path::{Path, PathBuf};

use crate::engine_file::{
    EngineFileError, EntryMeta, Header, LAYOUT_VERSION, Transform, write_file,
};
use crate::error::WeightsError;
use crate::loader::{file_sha256, source_fingerprint};
use crate::names::{
    FusedParam, Kind, LayerRole, MergerRole, MtpRole, NameLayout, ParamName, VisionRole, all_params,
};
use crate::safetensors_dir::{SafetensorsDir, TensorRef};

/// Outcome of a conversion.
#[derive(Debug, Clone)]
pub struct ConvertReport {
    pub out_path: PathBuf,
    pub revision: String,
    pub source_bytes: u64,
    pub out_bytes: u64,
    pub tensors: usize,
}

/// The checkpoint revision: the directory name of a HuggingFace snapshot
/// (`.../snapshots/<revision>`), else the directory's own name.
pub fn revision_of(model_dir: &Path) -> String {
    std::path::absolute(model_dir)
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "unknown".into())
}

/// Derives the layer counts from the tensor names: the highest decoder layer
/// index is the MTP layer when it carries `eh_proj`, and vision blocks are
/// counted by their highest index.
pub fn infer_layout(dir: &SafetensorsDir) -> Result<NameLayout, WeightsError> {
    let mut max_layer: Option<usize> = None;
    let mut max_block: Option<usize> = None;
    for name in dir.tensors().keys() {
        let index_after = |prefix: &str| {
            name.strip_prefix(prefix)
                .and_then(|r| r.split_once('.'))
                .and_then(|(i, _)| i.parse::<usize>().ok())
        };
        if let Some(i) = index_after("model.language_model.layers.") {
            max_layer = Some(max_layer.map_or(i, |m| m.max(i)));
        }
        if let Some(i) = index_after("model.visual.blocks.") {
            max_block = Some(max_block.map_or(i, |m| m.max(i)));
        }
    }
    let last = max_layer.ok_or_else(|| WeightsError::MissingTensor("decoder layers".into()))?;
    let blocks = max_block.ok_or_else(|| WeightsError::MissingTensor("vision blocks".into()))?;
    let has_mtp = dir.tensors().contains_key(&format!(
        "model.language_model.layers.{last}.eh_proj.weight"
    ));
    Ok(NameLayout {
        text_layers: if has_mtp { last } else { last + 1 },
        vision_blocks: blocks + 1,
    })
}

/// The engine tensor name of a fused tensor.
pub fn fused_name(fused: FusedParam) -> String {
    match fused {
        FusedParam::TextQkv { layer } => {
            format!("model.language_model.layers.{layer}.self_attn.qkv_proj.weight")
        }
        FusedParam::VisionGateUp { block, kind } => {
            let k = match kind {
                Kind::Weight => "weight",
                Kind::Bias => "bias",
            };
            format!("model.visual.blocks.{block}.mlp.gate_up_proj.{k}")
        }
    }
}

#[derive(Clone, Copy)]
enum Item {
    Direct(ParamName),
    Fused(FusedParam),
}

/// The engine tensors in file order, each tagged with its load unit.
fn plan_items(layout: NameLayout) -> Vec<(String, Item)> {
    let mut items: Vec<(String, Item)> = Vec::new();
    let mut unit = |label: String, list: Vec<Item>| {
        items.extend(list.into_iter().map(|i| (label.clone(), i)));
    };
    unit(
        "vision.patch_embed".into(),
        vec![
            Item::Direct(ParamName::PatchEmbed(Kind::Weight)),
            Item::Direct(ParamName::PatchEmbed(Kind::Bias)),
        ],
    );
    for block in 0..layout.vision_blocks {
        let v = |role| Item::Direct(ParamName::VisionBlock { block, role });
        unit(
            format!("vision.block.{block}"),
            vec![
                v(VisionRole::Norm1),
                v(VisionRole::Norm2),
                v(VisionRole::QNorm),
                v(VisionRole::KNorm),
                v(VisionRole::Qkv(Kind::Weight)),
                v(VisionRole::Qkv(Kind::Bias)),
                v(VisionRole::Proj(Kind::Weight)),
                v(VisionRole::Proj(Kind::Bias)),
                Item::Fused(FusedParam::VisionGateUp {
                    block,
                    kind: Kind::Weight,
                }),
                Item::Fused(FusedParam::VisionGateUp {
                    block,
                    kind: Kind::Bias,
                }),
                v(VisionRole::DownProj(Kind::Weight)),
                v(VisionRole::DownProj(Kind::Bias)),
            ],
        );
    }
    let mut adapter = vec![
        Item::Direct(ParamName::PostLayernorm),
        Item::Direct(ParamName::Downsample(Kind::Weight)),
        Item::Direct(ParamName::Downsample(Kind::Bias)),
    ];
    adapter.extend(
        MergerRole::ALL
            .into_iter()
            .map(|r| Item::Direct(ParamName::Merger(r))),
    );
    unit("vision.adapter".into(), adapter);
    unit(
        "text.embed".into(),
        vec![Item::Direct(ParamName::EmbedTokens)],
    );
    let layer_items = |layer: usize| {
        let t = |role| Item::Direct(ParamName::TextLayer { layer, role });
        vec![
            Item::Fused(FusedParam::TextQkv { layer }),
            t(LayerRole::OProj),
            t(LayerRole::GateUpProj),
            t(LayerRole::DownProj),
            t(LayerRole::InputLayernorm),
            t(LayerRole::PostSelfAttnLayernorm),
            t(LayerRole::PostAttentionLayernorm),
            t(LayerRole::PostMlpLayernorm),
        ]
    };
    for layer in 0..layout.text_layers {
        unit(format!("text.layer.{layer}"), layer_items(layer));
    }
    unit(
        "text.head".into(),
        vec![
            Item::Direct(ParamName::FinalNorm),
            Item::Direct(ParamName::LmHead),
        ],
    );
    let mut mtp = layer_items(layout.text_layers);
    mtp.extend(
        MtpRole::ALL
            .into_iter()
            .map(|r| Item::Direct(ParamName::Mtp(r))),
    );
    unit("mtp".into(), mtp);
    items
}

struct Planned {
    meta: EntryMeta,
    sources: Vec<TensorRef>,
}

fn plan_entry(
    dir: &SafetensorsDir,
    layout: NameLayout,
    unit: &str,
    item: Item,
) -> Result<Planned, EngineFileError> {
    let (name, params, fused) = match item {
        Item::Direct(p) => (p.checkpoint_name(layout), vec![p], false),
        Item::Fused(f) => (fused_name(f), f.sources(), true),
    };
    let names: Vec<String> = params.iter().map(|p| p.checkpoint_name(layout)).collect();
    let unsupported = |reason: String| EngineFileError::Unsupported {
        name: name.clone(),
        reason,
    };
    let mut refs = Vec::with_capacity(names.len());
    for n in &names {
        refs.push(dir.get(n)?.clone());
    }
    let dtype = refs[0].dtype;
    if refs.iter().any(|r| r.dtype != dtype) {
        return Err(unsupported("sources have different element types".into()));
    }
    let (transform, shape) = if fused {
        let tail = &refs[0].shape[1..];
        if refs
            .iter()
            .any(|r| r.shape.is_empty() || &r.shape[1..] != tail)
        {
            return Err(unsupported(
                "sources disagree beyond dimension 0".to_string(),
            ));
        }
        let rows: usize = refs.iter().map(|r| r.shape[0]).sum();
        let mut shape = vec![rows];
        shape.extend_from_slice(tail);
        (Transform::Concat, shape)
    } else {
        match params[0] {
            ParamName::PatchEmbed(Kind::Weight) | ParamName::Downsample(Kind::Weight) => {
                let s = &refs[0].shape;
                if s.len() < 2 {
                    return Err(unsupported("convolution weight has rank < 2".into()));
                }
                (Transform::Reshape, vec![s[0], s[1..].iter().product()])
            }
            _ => (Transform::Copy, refs[0].shape.clone()),
        }
    };
    let len = shape.iter().product::<usize>() as u64 * dtype.size_bytes() as u64;
    Ok(Planned {
        meta: EntryMeta {
            name,
            unit: unit.to_string(),
            dtype: dtype.as_str().to_string(),
            shape,
            transform,
            sources: names,
            offset: 0,
            len,
        },
        sources: refs,
    })
}

/// Converts the checkpoint in `model_dir` into an engine file at `out`. This
/// is the only writer of engine files. Every checkpoint tensor must be
/// consumed and every expected tensor present; otherwise conversion is
/// refused and nothing is written.
pub fn convert(model_dir: &Path, out: &Path) -> Result<ConvertReport, EngineFileError> {
    let dir = SafetensorsDir::open(model_dir)?;
    let layout = infer_layout(&dir)?;
    for p in all_params(layout) {
        dir.get(&p.checkpoint_name(layout))?;
    }
    let mut planned = Vec::new();
    for (unit, item) in plan_items(layout) {
        planned.push(plan_entry(&dir, layout, &unit, item)?);
    }
    let consumed: usize = planned.iter().map(|p| p.sources.len()).sum();
    if consumed != dir.tensors().len() {
        return Err(EngineFileError::Unsupported {
            name: model_dir.display().to_string(),
            reason: format!(
                "{} checkpoint tensors but the layout consumes {consumed}",
                dir.tensors().len()
            ),
        });
    }
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| EngineFileError::io(parent, e))?;
    }

    let mut sources = source_fingerprint(model_dir, false)?;
    for s in &mut sources {
        s.sha256 = "0".repeat(64);
    }
    let mut header = Header {
        layout_version: LAYOUT_VERSION,
        revision: revision_of(model_dir),
        text_layers: layout.text_layers,
        vision_blocks: layout.vision_blocks,
        sources,
        tensors: planned.iter().map(|p| p.meta.clone()).collect(),
        file_len: 0,
    };
    let source_bytes: u64 = header.sources.iter().map(|s| s.size).sum();
    let mut idx = 0;
    std::thread::scope(|scope| {
        let names: Vec<String> = header.sources.iter().map(|s| s.name.clone()).collect();
        let hasher = scope.spawn(move || {
            names
                .iter()
                .map(|n| file_sha256(&model_dir.join(n)))
                .collect::<Result<Vec<_>, _>>()
        });
        write_file(
            out,
            &mut header,
            |entry, w| {
                let p = &planned[idx];
                idx += 1;
                for src in &p.sources {
                    w.write_all(dir.bytes(src))
                        .map_err(|e| EngineFileError::io(&entry.name, e))?;
                }
                Ok(())
            },
            |h| {
                let hashes = hasher.join().map_err(|_| EngineFileError::Unsupported {
                    name: "source hash".into(),
                    reason: "hashing thread panicked".into(),
                })??;
                for (s, hash) in h.sources.iter_mut().zip(hashes) {
                    s.sha256 = hash;
                }
                Ok(())
            },
        )
    })?;
    Ok(ConvertReport {
        out_path: out.to_path_buf(),
        revision: header.revision.clone(),
        source_bytes,
        out_bytes: header.file_len,
        tensors: header.tensors.len(),
    })
}
