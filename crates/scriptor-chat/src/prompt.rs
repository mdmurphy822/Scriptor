//! Prompt construction: chat template rendering, tokenisation and image placeholder expansion.

use std::ops::Range;

use serde_json::{Value, json};

use crate::error::{ChatError, Result};
use crate::messages::{ChatMessage, ImageUrl, messages_to_template};
use crate::positions::{ImageSpan, MropePositions, mrope_positions};
use crate::template::{ChatTemplate, RenderOptions};
use crate::tokenizer::Tokenizer;

/// Patch grid of one image before spatial merging, as `(t, h, w)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageGrid {
    /// Temporal patch groups.
    pub t: usize,
    /// Patch rows.
    pub h: usize,
    /// Patch columns.
    pub w: usize,
}

impl ImageGrid {
    /// Merged token grid for a spatial merge size, failing when the patch grid is not divisible.
    pub fn llm_grid(&self, merge: usize) -> Result<(usize, usize, usize)> {
        if merge == 0 || self.t == 0 || self.h == 0 || self.w == 0 {
            return Err(ChatError::ImageLayout(format!(
                "degenerate image grid {}x{}x{} with merge {merge}",
                self.t, self.h, self.w
            )));
        }
        if !self.h.is_multiple_of(merge) || !self.w.is_multiple_of(merge) {
            return Err(ChatError::ImageLayout(format!(
                "grid {}x{}x{} is not divisible by merge size {merge}",
                self.t, self.h, self.w
            )));
        }
        Ok((self.t, self.h / merge, self.w / merge))
    }

    /// Number of image placeholder tokens the image occupies.
    pub fn token_count(&self, merge: usize) -> Result<usize> {
        let (t, h, w) = self.llm_grid(merge)?;
        Ok(t * h * w)
    }
}

/// A rendered and tokenised prompt whose images still hold a single placeholder token each.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedPrompt {
    /// The rendered template text.
    pub text: String,
    /// Token ids with one image placeholder per image.
    pub token_ids: Vec<u32>,
    /// Image references in prompt order.
    pub images: Vec<ImageUrl>,
}

/// A fully expanded prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct Prompt {
    /// The rendered template text (one placeholder per image).
    pub text: String,
    /// Token ids with each image placeholder expanded to its token count.
    pub token_ids: Vec<u32>,
    /// Token ranges of the expanded image placeholders, in prompt order.
    pub image_ranges: Vec<Range<usize>>,
    /// Merged token grids, in prompt order.
    pub llm_grids: Vec<(usize, usize, usize)>,
}

impl Prompt {
    /// Computes the rotary positions of the prompt.
    pub fn positions(&self) -> Result<MropePositions> {
        let spans: Vec<ImageSpan> = self
            .image_ranges
            .iter()
            .zip(&self.llm_grids)
            .map(|(range, grid)| ImageSpan {
                range: range.clone(),
                llm_grid: *grid,
            })
            .collect();
        mrope_positions(self.token_ids.len(), &spans)
    }
}

/// Renders and tokenises chat requests for the model.
pub struct PromptBuilder {
    tokenizer: Tokenizer,
    template: ChatTemplate,
    image_token_id: u32,
    merge_size: usize,
}

impl PromptBuilder {
    /// Creates a builder; `image_token_name` is resolved against the tokenizer vocabulary.
    pub fn new(
        tokenizer: Tokenizer,
        template: ChatTemplate,
        image_token_name: &str,
        merge_size: usize,
    ) -> Result<Self> {
        let image_token_id = tokenizer.require_token_id(image_token_name)?;
        Ok(Self {
            tokenizer,
            template,
            image_token_id,
            merge_size,
        })
    }

    /// The token id that marks one image position.
    pub fn image_token_id(&self) -> u32 {
        self.image_token_id
    }

    /// The tokenizer used by the builder.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Renders the template with image parts replaced by bare `{"type":"image"}` markers and
    /// tokenises the result. Image references are returned in prompt order.
    pub fn prepare(
        &self,
        messages: &[ChatMessage],
        options: &RenderOptions,
    ) -> Result<PreparedPrompt> {
        let mut value = messages_to_template(messages)?;
        let images = replace_image_parts(&mut value);
        let text = self.template.render(&value, options)?;
        let token_ids = self.tokenizer.encode(&text)?;
        let placeholders = token_ids
            .iter()
            .filter(|&&id| id == self.image_token_id)
            .count();
        if placeholders != images.len() {
            return Err(ChatError::ImageLayout(format!(
                "template rendered {placeholders} image placeholders for {} images",
                images.len()
            )));
        }
        Ok(PreparedPrompt {
            text,
            token_ids,
            images,
        })
    }

    /// Expands each image placeholder to the token count its grid implies.
    pub fn expand(&self, prepared: &PreparedPrompt, grids: &[ImageGrid]) -> Result<Prompt> {
        if grids.len() != prepared.images.len() {
            return Err(ChatError::ImageLayout(format!(
                "{} grids supplied for {} images",
                grids.len(),
                prepared.images.len()
            )));
        }
        let mut token_ids = Vec::with_capacity(prepared.token_ids.len());
        let mut image_ranges = Vec::with_capacity(grids.len());
        let mut llm_grids = Vec::with_capacity(grids.len());
        let mut next = 0usize;
        for &id in &prepared.token_ids {
            if id == self.image_token_id {
                let grid = &grids[next];
                next += 1;
                let count = grid.token_count(self.merge_size)?;
                let start = token_ids.len();
                token_ids.resize(start + count, id);
                image_ranges.push(start..start + count);
                llm_grids.push(grid.llm_grid(self.merge_size)?);
            } else {
                token_ids.push(id);
            }
        }
        Ok(Prompt {
            text: prepared.text.clone(),
            token_ids,
            image_ranges,
            llm_grids,
        })
    }

    /// Prepares and expands in one step when the image grids are already known.
    pub fn build(
        &self,
        messages: &[ChatMessage],
        options: &RenderOptions,
        grids: &[ImageGrid],
    ) -> Result<Prompt> {
        let prepared = self.prepare(messages, options)?;
        self.expand(&prepared, grids)
    }
}

/// Replaces every `image_url` content part with `{"type":"image"}` and returns the replaced
/// references in order.
fn replace_image_parts(messages: &mut Value) -> Vec<ImageUrl> {
    let mut images = Vec::new();
    let Some(list) = messages.as_array_mut() else {
        return images;
    };
    for message in list {
        let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for part in parts {
            if part.get("type").and_then(Value::as_str) != Some("image_url") {
                continue;
            }
            let url = part
                .pointer("/image_url/url")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let detail = part
                .pointer("/image_url/detail")
                .and_then(Value::as_str)
                .map(str::to_owned);
            images.push(ImageUrl { url, detail });
            *part = json!({"type": "image"});
        }
    }
    images
}
