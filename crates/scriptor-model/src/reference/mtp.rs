//! Multi-token-prediction layer: fuses the next token's embedding with the target model's hidden
//! state and runs one decoder layer with its own key/value cache.

use scriptor_kernels::reference::norm::rms_norm;
use scriptor_kernels::reference::scatter::embedding_gather;
use scriptor_spec::TextSpec;

use super::decoder::{DecoderLayer, KvCache};
use super::{Linear, ModelError, TensorSource};

/// The MTP head stored as decoder layer index `num_hidden_layers`.
#[derive(Debug, Clone)]
pub struct MtpHead {
    pub embed_tokens: Vec<f32>,
    pub enorm: Vec<f32>,
    pub hnorm: Vec<f32>,
    pub eh_proj: Linear,
    pub layer: DecoderLayer,
    pub shared_norm: Vec<f32>,
    pub shared_head: Linear,
    pub hidden: usize,
    pub eps: f32,
}

impl MtpHead {
    /// Loads the head from `model.language_model.layers.{num_layers}.*`.
    pub fn load(src: &TensorSource, spec: &TextSpec) -> Result<Self, ModelError> {
        let p = format!("model.language_model.layers.{}", spec.num_layers);
        let (h, vocab) = (spec.hidden_size, spec.vocab_size);
        Ok(Self {
            embed_tokens: src.tensor(&format!("{p}.embed_tokens.weight"), &[vocab, h])?,
            enorm: src.tensor(&format!("{p}.enorm.weight"), &[h])?,
            hnorm: src.tensor(&format!("{p}.hnorm.weight"), &[h])?,
            eh_proj: src.linear(&format!("{p}.eh_proj.weight"), None, h, 2 * h)?,
            layer: DecoderLayer::load(src, &p, spec)?,
            shared_norm: src.tensor(&format!("{p}.shared_head.norm.weight"), &[h])?,
            shared_head: src.linear(&format!("{p}.shared_head.head.weight"), None, vocab, h)?,
            hidden: h,
            eps: spec.rms_norm_eps as f32,
        })
    }

    /// A fresh single-layer cache.
    pub fn new_cache(&self) -> KvCache {
        KvCache::new(1)
    }

    /// Predicts from `next_tokens` (the token that follows each row) and `hidden` (the target
    /// model's post-final-norm hidden state at each row, `[n, hidden]`) at rotary `positions`.
    /// The token embedding is zeroed for rows whose temporal position is 0. Returns the layer
    /// output `[n, hidden]` before the shared head's norm.
    pub fn forward(
        &self,
        next_tokens: &[u32],
        hidden: &[f32],
        positions: &[[i32; 3]],
        cache: &mut KvCache,
    ) -> Vec<f32> {
        let n = next_tokens.len();
        assert_eq!(hidden.len(), n * self.hidden);
        assert_eq!(positions.len(), n);
        let mut embeds = embedding_gather(&self.embed_tokens, self.hidden, next_tokens);
        for (row, pos) in embeds.chunks_exact_mut(self.hidden).zip(positions) {
            if pos[0] == 0 {
                row.fill(0.0);
            }
        }
        let embeds = rms_norm(&embeds, &self.enorm, self.hidden, self.eps);
        let hidden = rms_norm(hidden, &self.hnorm, self.hidden, self.eps);
        let mut fused = Vec::with_capacity(n * 2 * self.hidden);
        for (e, h) in embeds
            .chunks_exact(self.hidden)
            .zip(hidden.chunks_exact(self.hidden))
        {
            fused.extend_from_slice(e);
            fused.extend_from_slice(h);
        }
        let mut x = self.eh_proj.apply(&fused, n);
        self.layer.forward(&mut x, positions, &mut cache.layers[0]);
        x
    }

    /// Draft logits `[rows, vocab]` for layer outputs from [`MtpHead::forward`].
    pub fn logits(&self, mtp_hidden: &[f32]) -> Vec<f32> {
        let normed = rms_norm(mtp_hidden, &self.shared_norm, self.hidden, self.eps);
        self.shared_head
            .apply(&normed, mtp_hidden.len() / self.hidden)
    }
}
