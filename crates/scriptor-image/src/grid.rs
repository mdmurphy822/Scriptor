/// Spatial patch edge in pixels.
pub const PATCH_SIZE: u32 = 14;
/// Patches merged per side into one language-model token.
pub const MERGE_SIZE: u32 = 2;
/// Frames each image is repeated into.
pub const TEMPORAL_PATCH_SIZE: u32 = 2;
/// Channels per pixel.
pub const CHANNELS: u32 = 3;
/// Elements per flattened patch: `3 * 2 * 14 * 14`.
pub const PATCH_ELEMS: usize = (CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE) as usize;

/// The `(t, h, w)` patch grid of one image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridThw {
    /// Temporal grid extent (1 for a still image).
    pub t: u32,
    /// Patch rows.
    pub h: u32,
    /// Patch columns.
    pub w: u32,
}

impl GridThw {
    /// Grid of an image whose sides are multiples of `PATCH_SIZE * MERGE_SIZE`.
    pub fn for_image(height: u32, width: u32) -> Self {
        Self {
            t: 1,
            h: height / PATCH_SIZE,
            w: width / PATCH_SIZE,
        }
    }

    /// Number of patches (rows of the `[N, 1176]` matrix).
    pub fn patches(&self) -> usize {
        self.t as usize * self.h as usize * self.w as usize
    }

    /// Number of language-model tokens the image occupies.
    pub fn tokens(&self) -> usize {
        self.patches() / (MERGE_SIZE * MERGE_SIZE) as usize
    }
}
