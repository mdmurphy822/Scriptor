//! Incremental byte-level detokenisation driven by a per-id byte table.

use std::collections::HashMap;

use serde_json::Value;

use crate::error::{ChatError, Result};

/// What one token id contributes to the output.
#[derive(Debug, Clone)]
enum Piece {
    /// Raw bytes of a byte-level vocabulary token.
    Bytes(Vec<u8>),
    /// A control token (skipped when special tokens are hidden).
    Special(String),
    /// An added, non-special token emitted verbatim.
    Literal(String),
}

/// Maps each byte-level alphabet character back to the byte it stands for.
fn alphabet_to_byte() -> HashMap<char, u8> {
    let mut keep: Vec<u32> = (33..127).collect();
    keep.extend(161..173);
    keep.extend(174..256);
    let mut map = HashMap::new();
    let mut extra = 0u32;
    for byte in 0u32..256 {
        if keep.contains(&byte) {
            map.insert(char::from_u32(byte).unwrap_or('\u{fffd}'), byte as u8);
        } else {
            map.insert(
                char::from_u32(256 + extra).unwrap_or('\u{fffd}'),
                byte as u8,
            );
            extra += 1;
        }
    }
    map
}

/// Per-id token table built from a byte-level BPE `tokenizer.json`.
#[derive(Debug, Clone)]
pub struct ByteTable {
    pieces: Vec<Option<Piece>>,
}

impl ByteTable {
    /// Builds the table from the bytes of a `tokenizer.json` document.
    pub fn from_tokenizer_json(bytes: &[u8]) -> Result<Self> {
        let doc: Value = serde_json::from_slice(bytes)
            .map_err(|e| ChatError::TokenizerLoad(format!("tokenizer.json: {e}")))?;
        let vocab = doc
            .pointer("/model/vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| ChatError::TokenizerLoad("tokenizer.json has no model.vocab".into()))?;
        let to_byte = alphabet_to_byte();
        let mut pieces: Vec<Option<Piece>> = Vec::new();
        let mut put = |id: usize, piece: Piece| {
            if pieces.len() <= id {
                pieces.resize(id + 1, None);
            }
            pieces[id] = Some(piece);
        };
        for (token, id) in vocab {
            let Some(id) = id.as_u64() else { continue };
            let bytes: Option<Vec<u8>> = token.chars().map(|c| to_byte.get(&c).copied()).collect();
            match bytes {
                Some(b) => put(id as usize, Piece::Bytes(b)),
                None => put(id as usize, Piece::Literal(token.clone())),
            }
        }
        if let Some(added) = doc.get("added_tokens").and_then(Value::as_array) {
            for entry in added {
                let (Some(id), Some(content)) = (
                    entry.get("id").and_then(Value::as_u64),
                    entry.get("content").and_then(Value::as_str),
                ) else {
                    continue;
                };
                let special = entry
                    .get("special")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let piece = if special {
                    Piece::Special(content.to_owned())
                } else {
                    Piece::Literal(content.to_owned())
                };
                put(id as usize, piece);
            }
        }
        Ok(Self { pieces })
    }

    /// Starts a detokeniser over this table.
    pub fn decoder(&self, skip_special: bool) -> ByteDetokenizer<'_> {
        ByteDetokenizer {
            table: self,
            skip_special,
            pending: Vec::new(),
        }
    }

    /// Decodes a whole id sequence in one call.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        let mut decoder = self.decoder(skip_special);
        let mut out = String::new();
        for &id in ids {
            out.push_str(&decoder.push(id)?);
        }
        out.push_str(&decoder.finish());
        Ok(out)
    }
}

/// Streaming detokeniser that releases text as soon as it forms complete UTF-8 characters.
/// Each push costs time proportional to the token's own bytes.
#[derive(Debug)]
pub struct ByteDetokenizer<'a> {
    table: &'a ByteTable,
    skip_special: bool,
    pending: Vec<u8>,
}

impl ByteDetokenizer<'_> {
    /// Appends a token id and returns the newly completed text, if any. Invalid byte sequences
    /// become U+FFFD; an incomplete trailing sequence is held back.
    pub fn push(&mut self, id: u32) -> Result<String> {
        let piece = self
            .table
            .pieces
            .get(id as usize)
            .and_then(Option::as_ref)
            .ok_or_else(|| ChatError::Decode(format!("token id {id} is not in the vocabulary")))?;
        match piece {
            Piece::Bytes(b) => self.pending.extend_from_slice(b),
            Piece::Special(s) => {
                if !self.skip_special {
                    self.pending.extend_from_slice(s.as_bytes());
                }
            }
            Piece::Literal(s) => self.pending.extend_from_slice(s.as_bytes()),
        }
        Ok(self.drain(false))
    }

    /// Releases any held-back bytes, replacing an incomplete sequence with U+FFFD.
    pub fn finish(&mut self) -> String {
        self.drain(true)
    }

    fn drain(&mut self, flush: bool) -> String {
        let mut out = String::new();
        let mut consumed = 0;
        loop {
            match std::str::from_utf8(&self.pending[consumed..]) {
                Ok(valid) => {
                    out.push_str(valid);
                    consumed = self.pending.len();
                    break;
                }
                Err(err) => {
                    let upto = err.valid_up_to();
                    out.push_str(
                        std::str::from_utf8(&self.pending[consumed..consumed + upto])
                            .unwrap_or_default(),
                    );
                    consumed += upto;
                    match err.error_len() {
                        Some(bad) => {
                            out.push('\u{fffd}');
                            consumed += bad;
                        }
                        None => {
                            if flush {
                                out.push('\u{fffd}');
                                consumed = self.pending.len();
                            }
                            break;
                        }
                    }
                }
            }
        }
        self.pending.drain(..consumed);
        out
    }
}
