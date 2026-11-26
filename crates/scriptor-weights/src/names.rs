//! Checkpoint tensor names: parsing into typed parameters and the fusion plan
//! that turns checkpoint tensors into engine tensors.

use std::fmt;

use crate::error::WeightsError;

/// Layer counts that decide how checkpoint indices are interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NameLayout {
    /// Decoder layers; the layer at this index is the MTP layer.
    pub text_layers: usize,
    /// Vision transformer blocks.
    pub vision_blocks: usize,
}

/// Weight or bias half of an affine tensor pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Kind {
    Weight,
    Bias,
}

impl Kind {
    fn suffix(self) -> &'static str {
        match self {
            Self::Weight => "weight",
            Self::Bias => "bias",
        }
    }
}

/// Tensors that live in every decoder layer, including the MTP layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LayerRole {
    InputLayernorm,
    PostAttentionLayernorm,
    PostSelfAttnLayernorm,
    PostMlpLayernorm,
    QProj,
    KProj,
    VProj,
    OProj,
    GateUpProj,
    DownProj,
}

impl LayerRole {
    pub const ALL: [Self; 10] = [
        Self::InputLayernorm,
        Self::PostAttentionLayernorm,
        Self::PostSelfAttnLayernorm,
        Self::PostMlpLayernorm,
        Self::QProj,
        Self::KProj,
        Self::VProj,
        Self::OProj,
        Self::GateUpProj,
        Self::DownProj,
    ];

    fn path(self) -> &'static str {
        match self {
            Self::InputLayernorm => "input_layernorm",
            Self::PostAttentionLayernorm => "post_attention_layernorm",
            Self::PostSelfAttnLayernorm => "post_self_attn_layernorm",
            Self::PostMlpLayernorm => "post_mlp_layernorm",
            Self::QProj => "self_attn.q_proj",
            Self::KProj => "self_attn.k_proj",
            Self::VProj => "self_attn.v_proj",
            Self::OProj => "self_attn.o_proj",
            Self::GateUpProj => "mlp.gate_up_proj",
            Self::DownProj => "mlp.down_proj",
        }
    }
}

/// Tensors that only the MTP layer carries beyond the decoder-layer roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MtpRole {
    EmbedTokens,
    Enorm,
    Hnorm,
    EhProj,
    SharedHeadNorm,
    SharedHeadHead,
}

impl MtpRole {
    pub const ALL: [Self; 6] = [
        Self::EmbedTokens,
        Self::Enorm,
        Self::Hnorm,
        Self::EhProj,
        Self::SharedHeadNorm,
        Self::SharedHeadHead,
    ];

    fn path(self) -> &'static str {
        match self {
            Self::EmbedTokens => "embed_tokens",
            Self::Enorm => "enorm",
            Self::Hnorm => "hnorm",
            Self::EhProj => "eh_proj",
            Self::SharedHeadNorm => "shared_head.norm",
            Self::SharedHeadHead => "shared_head.head",
        }
    }
}

/// Tensor roles inside one vision block. Roles with a bias carry a [`Kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VisionRole {
    Norm1,
    Norm2,
    QNorm,
    KNorm,
    Qkv(Kind),
    Proj(Kind),
    GateProj(Kind),
    UpProj(Kind),
    DownProj(Kind),
}

impl VisionRole {
    /// Every role a vision block carries in the checkpoint.
    pub fn all() -> Vec<Self> {
        let mut v = vec![Self::Norm1, Self::Norm2, Self::QNorm, Self::KNorm];
        for k in [Kind::Weight, Kind::Bias] {
            v.extend([
                Self::Qkv(k),
                Self::Proj(k),
                Self::GateProj(k),
                Self::UpProj(k),
                Self::DownProj(k),
            ]);
        }
        v
    }

    fn path(self) -> String {
        let (p, k) = match self {
            Self::Norm1 => return "norm1.weight".into(),
            Self::Norm2 => return "norm2.weight".into(),
            Self::QNorm => return "attn.q_norm.weight".into(),
            Self::KNorm => return "attn.k_norm.weight".into(),
            Self::Qkv(k) => ("attn.qkv", k),
            Self::Proj(k) => ("attn.proj", k),
            Self::GateProj(k) => ("mlp.gate_proj", k),
            Self::UpProj(k) => ("mlp.up_proj", k),
            Self::DownProj(k) => ("mlp.down_proj", k),
        };
        format!("{p}.{}", k.suffix())
    }

    fn parse(path: &str) -> Option<Self> {
        Self::all().into_iter().find(|r| r.path() == path)
    }
}

/// Tensors of the patch-merger head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MergerRole {
    Proj,
    GateProj,
    UpProj,
    DownProj,
    PostProjectionNorm(Kind),
}

impl MergerRole {
    pub const ALL: [Self; 6] = [
        Self::Proj,
        Self::GateProj,
        Self::UpProj,
        Self::DownProj,
        Self::PostProjectionNorm(Kind::Weight),
        Self::PostProjectionNorm(Kind::Bias),
    ];

    fn path(self) -> String {
        match self {
            Self::Proj => "proj.weight".into(),
            Self::GateProj => "gate_proj.weight".into(),
            Self::UpProj => "up_proj.weight".into(),
            Self::DownProj => "down_proj.weight".into(),
            Self::PostProjectionNorm(k) => format!("post_projection_norm.{}", k.suffix()),
        }
    }
}

/// One named parameter of the checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParamName {
    LmHead,
    EmbedTokens,
    FinalNorm,
    /// A decoder-layer tensor; `layer == text_layers` is the MTP layer.
    TextLayer {
        layer: usize,
        role: LayerRole,
    },
    /// An MTP-only tensor stored under the MTP layer index.
    Mtp(MtpRole),
    VisionBlock {
        block: usize,
        role: VisionRole,
    },
    PatchEmbed(Kind),
    Downsample(Kind),
    Merger(MergerRole),
    PostLayernorm,
}

impl ParamName {
    /// Parses a checkpoint tensor name; unknown names are an error.
    pub fn parse(name: &str, layout: NameLayout) -> Result<Self, WeightsError> {
        Self::try_parse(name, layout).ok_or_else(|| WeightsError::UnmappedTensor(name.to_string()))
    }

    fn try_parse(name: &str, layout: NameLayout) -> Option<Self> {
        match name {
            "lm_head.weight" => return Some(Self::LmHead),
            "model.language_model.embed_tokens.weight" => return Some(Self::EmbedTokens),
            "model.language_model.norm.weight" => return Some(Self::FinalNorm),
            "model.visual.post_layernorm.weight" => return Some(Self::PostLayernorm),
            _ => {}
        }
        if let Some(rest) = name.strip_prefix("model.language_model.layers.") {
            let (idx, path) = rest.split_once('.')?;
            let layer: usize = idx.parse().ok()?;
            if layer > layout.text_layers {
                return None;
            }
            if let Some(role) = LayerRole::ALL
                .into_iter()
                .find(|r| format!("{}.weight", r.path()) == path)
            {
                return Some(Self::TextLayer { layer, role });
            }
            if layer == layout.text_layers {
                return MtpRole::ALL
                    .into_iter()
                    .find(|r| format!("{}.weight", r.path()) == path)
                    .map(Self::Mtp);
            }
            return None;
        }
        let rest = name.strip_prefix("model.visual.")?;
        if let Some(rest) = rest.strip_prefix("blocks.") {
            let (idx, path) = rest.split_once('.')?;
            let block: usize = idx.parse().ok()?;
            if block >= layout.vision_blocks {
                return None;
            }
            return VisionRole::parse(path).map(|role| Self::VisionBlock { block, role });
        }
        let kind = |s: &str| match s {
            "weight" => Some(Kind::Weight),
            "bias" => Some(Kind::Bias),
            _ => None,
        };
        if let Some(k) = rest.strip_prefix("patch_embed.proj.") {
            return kind(k).map(Self::PatchEmbed);
        }
        if let Some(k) = rest.strip_prefix("downsample.") {
            return kind(k).map(Self::Downsample);
        }
        if let Some(path) = rest.strip_prefix("merger.") {
            return MergerRole::ALL
                .into_iter()
                .find(|r| r.path() == path)
                .map(Self::Merger);
        }
        None
    }

    /// The checkpoint tensor name this parameter is stored under.
    pub fn checkpoint_name(&self, layout: NameLayout) -> String {
        match *self {
            Self::LmHead => "lm_head.weight".into(),
            Self::EmbedTokens => "model.language_model.embed_tokens.weight".into(),
            Self::FinalNorm => "model.language_model.norm.weight".into(),
            Self::PostLayernorm => "model.visual.post_layernorm.weight".into(),
            Self::TextLayer { layer, role } => {
                format!("model.language_model.layers.{layer}.{}.weight", role.path())
            }
            Self::Mtp(role) => format!(
                "model.language_model.layers.{}.{}.weight",
                layout.text_layers,
                role.path()
            ),
            Self::VisionBlock { block, role } => {
                format!("model.visual.blocks.{block}.{}", role.path())
            }
            Self::PatchEmbed(k) => format!("model.visual.patch_embed.proj.{}", k.suffix()),
            Self::Downsample(k) => format!("model.visual.downsample.{}", k.suffix()),
            Self::Merger(r) => format!("model.visual.merger.{}", r.path()),
        }
    }
}

impl fmt::Display for ParamName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// An engine tensor assembled by concatenating several checkpoint tensors along dim 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FusedParam {
    /// Text `qkv` from `q_proj`, `k_proj`, `v_proj` of a decoder layer (MTP layer included).
    TextQkv { layer: usize },
    /// Vision `gate_up` from `gate_proj`, `up_proj` of a block.
    VisionGateUp { block: usize, kind: Kind },
}

impl FusedParam {
    /// The checkpoint parameters concatenated, in order, to build this tensor.
    pub fn sources(&self) -> Vec<ParamName> {
        match *self {
            Self::TextQkv { layer } => [LayerRole::QProj, LayerRole::KProj, LayerRole::VProj]
                .into_iter()
                .map(|role| ParamName::TextLayer { layer, role })
                .collect(),
            Self::VisionGateUp { block, kind } => {
                [VisionRole::GateProj(kind), VisionRole::UpProj(kind)]
                    .into_iter()
                    .map(|role| ParamName::VisionBlock { block, role })
                    .collect()
            }
        }
    }
}

/// How one checkpoint tensor reaches the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Used as stored.
    Direct(ParamName),
    /// The `index`-th piece of a fused engine tensor.
    Part { target: FusedParam, index: usize },
}

/// Classifies a checkpoint tensor name as direct or as a part of a fused tensor.
pub fn classify(name: &str, layout: NameLayout) -> Result<Placement, WeightsError> {
    let p = ParamName::parse(name, layout)?;
    let fused = match p {
        ParamName::TextLayer { layer, role } => match role {
            LayerRole::QProj => Some((FusedParam::TextQkv { layer }, 0)),
            LayerRole::KProj => Some((FusedParam::TextQkv { layer }, 1)),
            LayerRole::VProj => Some((FusedParam::TextQkv { layer }, 2)),
            _ => None,
        },
        ParamName::VisionBlock { block, role } => match role {
            VisionRole::GateProj(kind) => Some((FusedParam::VisionGateUp { block, kind }, 0)),
            VisionRole::UpProj(kind) => Some((FusedParam::VisionGateUp { block, kind }, 1)),
            _ => None,
        },
        _ => None,
    };
    Ok(match fused {
        Some((target, index)) => Placement::Part { target, index },
        None => Placement::Direct(p),
    })
}

/// Every fused engine tensor the layout implies.
pub fn fused_targets(layout: NameLayout) -> Vec<FusedParam> {
    let mut v: Vec<_> = (0..=layout.text_layers)
        .map(|layer| FusedParam::TextQkv { layer })
        .collect();
    for block in 0..layout.vision_blocks {
        for kind in [Kind::Weight, Kind::Bias] {
            v.push(FusedParam::VisionGateUp { block, kind });
        }
    }
    v
}

/// Every checkpoint parameter the layout implies, for completeness checks.
pub fn all_params(layout: NameLayout) -> Vec<ParamName> {
    let mut v = vec![
        ParamName::LmHead,
        ParamName::EmbedTokens,
        ParamName::FinalNorm,
        ParamName::PostLayernorm,
        ParamName::PatchEmbed(Kind::Weight),
        ParamName::PatchEmbed(Kind::Bias),
        ParamName::Downsample(Kind::Weight),
        ParamName::Downsample(Kind::Bias),
    ];
    for layer in 0..=layout.text_layers {
        v.extend(
            LayerRole::ALL
                .into_iter()
                .map(|role| ParamName::TextLayer { layer, role }),
        );
    }
    v.extend(MtpRole::ALL.into_iter().map(ParamName::Mtp));
    for block in 0..layout.vision_blocks {
        v.extend(
            VisionRole::all()
                .into_iter()
                .map(|role| ParamName::VisionBlock { block, role }),
        );
    }
    v.extend(MergerRole::ALL.into_iter().map(ParamName::Merger));
    v
}
