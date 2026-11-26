//! Text decoder: sandwich-norm layers with grouped-query attention and GPT-J style M-RoPE, a
//! growing key/value cache, final norm and language-model head.

use scriptor_kernels::reference::activation::silu_mul_fused;
use scriptor_kernels::reference::norm::rms_norm;
use scriptor_kernels::reference::rope::mrope_gptj;
use scriptor_spec::TextSpec;

use super::{Linear, ModelError, TensorSource, add_in_place};

/// Cached keys and values of one layer, each `[tokens, kv_heads, head_dim]`.
#[derive(Debug, Clone, Default)]
pub struct LayerKv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// Key/value cache of a stack of layers.
#[derive(Debug, Clone)]
pub struct KvCache {
    pub layers: Vec<LayerKv>,
}

impl KvCache {
    /// An empty cache for `layers` layers.
    pub fn new(layers: usize) -> Self {
        Self {
            layers: vec![LayerKv::default(); layers],
        }
    }

    /// Tokens cached so far, given the per-token width `kv_heads * head_dim`.
    pub fn len(&self, kv_width: usize) -> usize {
        self.layers.first().map_or(0, |l| l.k.len() / kv_width)
    }

    /// Whether nothing is cached.
    pub fn is_empty(&self) -> bool {
        self.layers.first().is_none_or(|l| l.k.is_empty())
    }
}

/// Attention and norm geometry shared by every layer.
#[derive(Debug, Clone, Copy)]
pub struct LayerGeometry {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub mrope_section: [usize; 3],
}

impl LayerGeometry {
    /// Geometry from the text specification; the rotary dimension must cover the whole head.
    pub fn from_spec(spec: &TextSpec) -> Result<Self, ModelError> {
        if spec.rotary_dim != spec.head_dim || spec.is_neox_style {
            return Err(ModelError::Unsupported(
                "text decoder requires full-head interleaved rotary embedding".into(),
            ));
        }
        Ok(Self {
            hidden: spec.hidden_size,
            heads: spec.num_heads,
            kv_heads: spec.num_kv_heads,
            head_dim: spec.head_dim,
            eps: spec.rms_norm_eps as f32,
            rope_theta: spec.rope_theta,
            mrope_section: spec.mrope_section,
        })
    }
}

/// One decoder layer:
/// `h += post_self_attn_norm(attn(input_norm(h)))`, then
/// `h += post_mlp_norm(mlp(post_attention_norm(h)))`.
#[derive(Debug, Clone)]
pub struct DecoderLayer {
    pub input_norm: Vec<f32>,
    pub post_self_attn_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub post_mlp_norm: Vec<f32>,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub gate_up: Linear,
    pub down: Linear,
    pub geometry: LayerGeometry,
    pub intermediate: usize,
}

impl DecoderLayer {
    /// Loads the layer whose tensors live under `prefix` (for example
    /// `model.language_model.layers.3`).
    pub fn load(src: &TensorSource, prefix: &str, spec: &TextSpec) -> Result<Self, ModelError> {
        let geometry = LayerGeometry::from_spec(spec)?;
        let (h, i) = (spec.hidden_size, spec.intermediate_size);
        let (qd, kd) = (
            spec.num_heads * spec.head_dim,
            spec.num_kv_heads * spec.head_dim,
        );
        Ok(Self {
            input_norm: src.tensor(&format!("{prefix}.input_layernorm.weight"), &[h])?,
            post_self_attn_norm: src
                .tensor(&format!("{prefix}.post_self_attn_layernorm.weight"), &[h])?,
            post_attention_norm: src
                .tensor(&format!("{prefix}.post_attention_layernorm.weight"), &[h])?,
            post_mlp_norm: src.tensor(&format!("{prefix}.post_mlp_layernorm.weight"), &[h])?,
            q: src.linear(&format!("{prefix}.self_attn.q_proj.weight"), None, qd, h)?,
            k: src.linear(&format!("{prefix}.self_attn.k_proj.weight"), None, kd, h)?,
            v: src.linear(&format!("{prefix}.self_attn.v_proj.weight"), None, kd, h)?,
            o: src.linear(&format!("{prefix}.self_attn.o_proj.weight"), None, h, qd)?,
            gate_up: src.linear(&format!("{prefix}.mlp.gate_up_proj.weight"), None, 2 * i, h)?,
            down: src.linear(&format!("{prefix}.mlp.down_proj.weight"), None, h, i)?,
            geometry,
            intermediate: i,
        })
    }

    /// Advances `h` (`[n, hidden]`, updated in place) through the layer for `n` new tokens with
    /// rotary `positions`, appending their keys and values to `kv`. Each new token attends to
    /// everything cached before it and to earlier new tokens.
    pub fn forward(&self, h: &mut [f32], positions: &[[i32; 3]], kv: &mut LayerKv) {
        let g = self.geometry;
        let n = positions.len();
        assert_eq!(h.len(), n * g.hidden);
        let normed = rms_norm(h, &self.input_norm, g.hidden, g.eps);
        let mut q = self.q.apply(&normed, n);
        let mut k = self.k.apply(&normed, n);
        let v = self.v.apply(&normed, n);
        mrope_gptj(
            &mut q,
            positions,
            g.heads,
            g.head_dim,
            g.rope_theta,
            g.mrope_section,
        );
        mrope_gptj(
            &mut k,
            positions,
            g.kv_heads,
            g.head_dim,
            g.rope_theta,
            g.mrope_section,
        );
        let past = kv.k.len() / (g.kv_heads * g.head_dim);
        kv.k.extend_from_slice(&k);
        kv.v.extend_from_slice(&v);
        let attn = attend(&q, kv, past, n, &g);
        let attn = self.o.apply(&attn, n);
        let attn = rms_norm(&attn, &self.post_self_attn_norm, g.hidden, g.eps);
        add_in_place(h, &attn);

        let normed = rms_norm(h, &self.post_attention_norm, g.hidden, g.eps);
        let act = silu_mul_fused(&self.gate_up.apply(&normed, n), self.intermediate);
        let mlp = self.down.apply(&act, n);
        let mlp = rms_norm(&mlp, &self.post_mlp_norm, g.hidden, g.eps);
        add_in_place(h, &mlp);
    }
}

/// Causal grouped-query attention of `n` new queries over the cache that already holds `past`
/// tokens plus the new ones. Softmax and weighted sums accumulate in f64.
fn attend(q: &[f32], kv: &LayerKv, past: usize, n: usize, g: &LayerGeometry) -> Vec<f32> {
    let group = g.heads / g.kv_heads;
    let scale = (g.head_dim as f64).powf(-0.5);
    let kv_width = g.kv_heads * g.head_dim;
    let mut out = vec![0.0f32; n * g.heads * g.head_dim];
    for i in 0..n {
        let visible = past + i + 1;
        for head in 0..g.heads {
            let kvh = head / group;
            let qrow = &q[(i * g.heads + head) * g.head_dim..][..g.head_dim];
            let mut scores = Vec::with_capacity(visible);
            for j in 0..visible {
                let krow = &kv.k[j * kv_width + kvh * g.head_dim..][..g.head_dim];
                let dot: f64 = qrow
                    .iter()
                    .zip(krow)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum();
                scores.push(dot * scale);
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            let dst = &mut out[(i * g.heads + head) * g.head_dim..][..g.head_dim];
            for (d, slot) in dst.iter_mut().enumerate() {
                let acc: f64 = weights
                    .iter()
                    .enumerate()
                    .map(|(j, w)| w * f64::from(kv.v[j * kv_width + kvh * g.head_dim + d]))
                    .sum();
                *slot = (acc / total) as f32;
            }
        }
    }
    out
}

/// Embedding table, decoder layers, final norm and language-model head.
#[derive(Debug, Clone)]
pub struct TextModel {
    pub embed_tokens: Vec<f32>,
    pub layers: Vec<DecoderLayer>,
    pub final_norm: Vec<f32>,
    pub lm_head: Linear,
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
}

impl TextModel {
    /// Loads the text model from `model.language_model.*` and `lm_head.weight`.
    pub fn load(src: &TensorSource, spec: &TextSpec) -> Result<Self, ModelError> {
        if spec.tie_word_embeddings {
            return Err(ModelError::Unsupported("tied word embeddings".into()));
        }
        let (h, vocab) = (spec.hidden_size, spec.vocab_size);
        let layers = (0..spec.num_layers)
            .map(|i| DecoderLayer::load(src, &format!("model.language_model.layers.{i}"), spec))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            embed_tokens: src.tensor("model.language_model.embed_tokens.weight", &[vocab, h])?,
            layers,
            final_norm: src.tensor("model.language_model.norm.weight", &[h])?,
            lm_head: src.linear("lm_head.weight", None, vocab, h)?,
            hidden: h,
            vocab,
            eps: spec.rms_norm_eps as f32,
        })
    }

    /// A fresh cache sized for this model.
    pub fn new_cache(&self) -> KvCache {
        KvCache::new(self.layers.len())
    }

    /// Runs `embeds` (`[n, hidden]`) with rotary `positions` through every layer, extending
    /// `cache`, and returns the post-final-norm hidden states `[n, hidden]`.
    pub fn forward(&self, embeds: &[f32], positions: &[[i32; 3]], cache: &mut KvCache) -> Vec<f32> {
        let mut h = embeds.to_vec();
        for (layer, kv) in self.layers.iter().zip(&mut cache.layers) {
            layer.forward(&mut h, positions, kv);
        }
        rms_norm(&h, &self.final_norm, self.hidden, self.eps)
    }

    /// Language-model logits `[rows, vocab]` for post-norm hidden rows.
    pub fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        self.lm_head.apply(hidden, hidden.len() / self.hidden)
    }
}
