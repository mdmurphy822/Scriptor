//! The whole model: pixels to image embeddings, embedding scatter, decoder prefill, greedy decode
//! and the MTP head.

use std::path::Path;

use scriptor_kernels::reference::sampling::{Penalties, penalty_argmax};
use scriptor_kernels::reference::scatter::{
    embedding_gather, image_scatter, placeholder_positions,
};
use scriptor_spec::ModelSpec;
use scriptor_weights::SafetensorsDir;

use super::decoder::TextModel;
use super::merger::Merger;
use super::mtp::MtpHead;
use super::vision::VisionTower;
use super::{ModelError, TensorSource};

/// One greedy decoding step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GreedyStep {
    /// The chosen token.
    pub token: u32,
    /// Best logit minus second-best logit after penalties.
    pub margin: f32,
}

/// The result of [`ReferenceModel::generate`].
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateOutput {
    /// One entry per generated token, including a final end-of-sequence token.
    pub steps: Vec<GreedyStep>,
    /// Whether generation stopped on an end-of-sequence token.
    pub hit_eos: bool,
}

/// Vision tower, adapter, text model and optional MTP head, all in f32.
#[derive(Debug, Clone)]
pub struct ReferenceModel {
    pub spec: ModelSpec,
    pub vision: VisionTower,
    pub merger: Merger,
    pub text: TextModel,
    pub mtp: Option<MtpHead>,
}

impl ReferenceModel {
    /// Loads every tensor from the safetensors files in `dir`.
    pub fn load(dir: impl AsRef<Path>, spec: &ModelSpec) -> Result<Self, ModelError> {
        let files = SafetensorsDir::open(dir)?;
        let src = TensorSource::new(&files);
        Ok(Self {
            vision: VisionTower::load(&src, &spec.vision)?,
            merger: Merger::load(&src, &spec.vision)?,
            text: TextModel::load(&src, &spec.text)?,
            mtp: if spec.text.num_mtp_layers > 0 {
                Some(MtpHead::load(&src, &spec.text)?)
            } else {
                None
            },
            spec: spec.clone(),
        })
    }

    /// Image embeddings `[tokens, text_hidden]` for patches `[N, 1176]` of images with `(t, h, w)`
    /// patch grids.
    pub fn encode_images(
        &self,
        patches: &[f32],
        grids: &[(usize, usize, usize)],
    ) -> Result<Vec<f32>, ModelError> {
        Ok(self.merger.forward(&self.vision.forward(patches, grids)?))
    }

    /// Token embeddings `[ids.len(), hidden]` with every image placeholder row replaced by the
    /// next row of `image_embeds`.
    pub fn input_embeds(&self, ids: &[u32], image_embeds: &[f32]) -> Result<Vec<f32>, ModelError> {
        let hidden = self.text.hidden;
        if let Some(bad) = ids.iter().find(|&&id| id as usize >= self.text.vocab) {
            return Err(ModelError::Input(format!(
                "token id {bad} is outside the vocabulary"
            )));
        }
        let mut embeds = embedding_gather(&self.text.embed_tokens, hidden, ids);
        let slots = placeholder_positions(ids, self.spec.image_token_id);
        if slots.len() * hidden != image_embeds.len() {
            return Err(ModelError::Input(format!(
                "{} image placeholders for {} image embedding rows",
                slots.len(),
                image_embeds.len() / hidden
            )));
        }
        image_scatter(&mut embeds, hidden, image_embeds, &slots);
        Ok(embeds)
    }

    /// Greedy generation from an expanded prompt. `positions` holds the prompt's `[t, h, w]`
    /// rotary positions and `delta` offsets the sequence index to the position of generated
    /// tokens. Stops after `max_steps` tokens or after an end-of-sequence token.
    pub fn generate(
        &self,
        prompt: &[u32],
        image_embeds: &[f32],
        positions: &[[i32; 3]],
        delta: i64,
        max_steps: usize,
        penalties: Penalties,
    ) -> Result<GenerateOutput, ModelError> {
        if positions.len() != prompt.len() || prompt.is_empty() {
            return Err(ModelError::Input(
                "positions must cover a non-empty prompt".into(),
            ));
        }
        let hidden = self.text.hidden;
        let embeds = self.input_embeds(prompt, image_embeds)?;
        let mut cache = self.text.new_cache();
        let mut out = self.text.forward(&embeds, positions, &mut cache);
        let mut generated: Vec<u32> = Vec::new();
        let mut steps = Vec::new();
        let mut hit_eos = false;
        while steps.len() < max_steps {
            let last = &out[out.len() - hidden..];
            let logits = self.text.logits(last);
            let pick = penalty_argmax(&logits, prompt, &generated, penalties);
            steps.push(GreedyStep {
                token: pick.token,
                margin: pick.margin,
            });
            generated.push(pick.token);
            if self.spec.text.eos_token_ids.contains(&pick.token) {
                hit_eos = true;
                break;
            }
            if steps.len() == max_steps {
                break;
            }
            let at = (prompt.len() + generated.len() - 1) as i64 + delta;
            let at =
                i32::try_from(at).map_err(|_| ModelError::Input("position overflow".into()))?;
            let embed = embedding_gather(&self.text.embed_tokens, hidden, &[pick.token]);
            out = self.text.forward(&embed, &[[at; 3]], &mut cache);
        }
        Ok(GenerateOutput { steps, hit_eos })
    }
}
