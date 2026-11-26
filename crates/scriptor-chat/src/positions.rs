//! Three-dimensional (temporal, height, width) rotary positions for prompts with images.

use std::ops::Range;

use crate::error::{ChatError, Result};

/// One image's span in the expanded prompt and its merged token grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSpan {
    /// Token index range of the image placeholder copies.
    pub range: Range<usize>,
    /// Merged grid `(t, h, w)`; its product must equal the range length.
    pub llm_grid: (usize, usize, usize),
}

/// Rotary positions of a whole prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MropePositions {
    /// Temporal, height and width position rows, each one entry per prompt token.
    pub rows: [Vec<i64>; 3],
    /// `max position + 1 - prompt length`: add to the sequence length to get the position of the
    /// next generated token on all three rows.
    pub delta: i64,
}

impl MropePositions {
    /// Number of prompt tokens covered.
    pub fn len(&self) -> usize {
        self.rows[0].len()
    }

    /// Whether the prompt is empty.
    pub fn is_empty(&self) -> bool {
        self.rows[0].is_empty()
    }
}

/// Position shared by all three rows for the token generated at sequence index `seq_len`.
pub fn decode_position(seq_len: usize, delta: i64) -> i64 {
    seq_len as i64 + delta
}

/// Computes positions for `total_len` tokens: text tokens advance all three rows together; an
/// image's tokens take `(t, h, w)` grid coordinates offset by the position cursor, after which
/// the cursor moves to one past the largest position used.
pub fn mrope_positions(total_len: usize, images: &[ImageSpan]) -> Result<MropePositions> {
    let mut rows: [Vec<i64>; 3] = [
        Vec::with_capacity(total_len),
        Vec::with_capacity(total_len),
        Vec::with_capacity(total_len),
    ];
    let mut cursor = 0i64;
    let mut at = 0usize;
    for image in images {
        let (t, h, w) = image.llm_grid;
        if image.range.start < at || image.range.end > total_len {
            return Err(ChatError::ImageLayout(
                "image spans overlap or exceed the prompt".into(),
            ));
        }
        if image.range.len() != t * h * w {
            return Err(ChatError::ImageLayout(format!(
                "image span of {} tokens does not match grid {t}x{h}x{w}",
                image.range.len()
            )));
        }
        for _ in at..image.range.start {
            for row in &mut rows {
                row.push(cursor);
            }
            cursor += 1;
        }
        for ti in 0..t {
            for hi in 0..h {
                for wi in 0..w {
                    rows[0].push(cursor + ti as i64);
                    rows[1].push(cursor + hi as i64);
                    rows[2].push(cursor + wi as i64);
                }
            }
        }
        cursor += t.max(h).max(w) as i64;
        at = image.range.end;
    }
    for _ in at..total_len {
        for row in &mut rows {
            row.push(cursor);
        }
        cursor += 1;
    }
    Ok(MropePositions {
        delta: cursor - total_len as i64,
        rows,
    })
}
