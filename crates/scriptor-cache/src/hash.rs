//! Content key for the image-embedding cache.

use sha2::{Digest, Sha256};

/// Patch grid of an image after resizing: temporal, height and width patch counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageGrid {
    pub t: u32,
    pub h: u32,
    pub w: u32,
}

/// SHA-256 digest identifying decoded pixels plus grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EmbeddingKey(pub [u8; 32]);

impl EmbeddingKey {
    /// Lowercase hexadecimal form of the digest.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// Keys decoded RGB bytes together with the grid and the byte length, so equal pixels under a
/// different grid or a different length hash differently.
pub fn embedding_key(rgb: &[u8], grid: ImageGrid) -> EmbeddingKey {
    let mut h = Sha256::new();
    h.update(grid.t.to_le_bytes());
    h.update(grid.h.to_le_bytes());
    h.update(grid.w.to_le_bytes());
    h.update((rgb.len() as u64).to_le_bytes());
    h.update(rgb);
    EmbeddingKey(h.finalize().into())
}
