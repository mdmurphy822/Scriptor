//! Byte-level BPE tokenizer wrapper with special-token handling and incremental detokenisation.

use std::path::Path;

use tokenizers::Tokenizer as Inner;

use crate::error::{ChatError, Result};

/// A loaded `tokenizer.json`.
pub struct Tokenizer {
    inner: Inner,
}

impl Tokenizer {
    /// Loads a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Parses a `tokenizer.json` document held in memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let inner =
            Inner::from_bytes(bytes).map_err(|e| ChatError::TokenizerLoad(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Encodes text to token ids. Added tokens (such as `<|im_end|>`) appearing in the text are
    /// mapped to their ids; no tokens are added around the text.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| ChatError::Encode(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decodes token ids to text. With `skip_special` set, special tokens are omitted.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| ChatError::Decode(e.to_string()))
    }

    /// Returns the id of a token by its exact string, if present in the vocabulary.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Returns the id of a token by its exact string, failing when absent.
    pub fn require_token_id(&self, token: &str) -> Result<u32> {
        self.token_id(token)
            .ok_or_else(|| ChatError::UnknownToken(token.to_owned()))
    }

    /// Returns the token string for an id, if present.
    pub fn id_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Total vocabulary size including added tokens.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Starts an incremental detokeniser that emits text as soon as it is complete UTF-8.
    pub fn incremental(&self, skip_special: bool) -> IncrementalDecoder<'_> {
        IncrementalDecoder {
            tokenizer: self,
            skip_special,
            ids: Vec::new(),
            prefix: 0,
            read: 0,
        }
    }
}

/// Streaming detokeniser. Text is released only once the pending bytes form complete UTF-8
/// characters, so multi-byte characters split across byte-level tokens never surface as
/// replacement characters.
pub struct IncrementalDecoder<'a> {
    tokenizer: &'a Tokenizer,
    skip_special: bool,
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
}

impl IncrementalDecoder<'_> {
    /// Appends a token id and returns the newly completed text, if any.
    pub fn push(&mut self, id: u32) -> Result<String> {
        self.ids.push(id);
        let prefix_text = self
            .tokenizer
            .decode(&self.ids[self.prefix..self.read], self.skip_special)?;
        let full_text = self
            .tokenizer
            .decode(&self.ids[self.prefix..], self.skip_special)?;
        if full_text.len() > prefix_text.len() && !full_text.ends_with('\u{fffd}') {
            let out = full_text[prefix_text.len()..].to_owned();
            self.prefix = self.read;
            self.read = self.ids.len();
            Ok(out)
        } else {
            Ok(String::new())
        }
    }

    /// Releases any text still held back, including incomplete trailing bytes.
    pub fn finish(&mut self) -> Result<String> {
        let prefix_text = self
            .tokenizer
            .decode(&self.ids[self.prefix..self.read], self.skip_special)?;
        let full_text = self
            .tokenizer
            .decode(&self.ids[self.prefix..], self.skip_special)?;
        self.prefix = self.ids.len();
        self.read = self.ids.len();
        Ok(full_text.get(prefix_text.len()..).unwrap_or("").to_owned())
    }
}
