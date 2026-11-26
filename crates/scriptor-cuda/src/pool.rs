//! Bump-style arena bookkeeping with an explicit byte capacity.

use crate::error::{CudaError, Result};

/// A byte range handed out by an [`Arena`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaRange {
    pub offset: usize,
    pub len: usize,
}

/// Offset allocator over a fixed capacity. Allocations are aligned and
/// released all at once by [`Arena::reset`].
#[derive(Debug, Clone)]
pub struct Arena {
    capacity: usize,
    cursor: usize,
    high_water: usize,
}

impl Arena {
    /// Creates an arena that never hands out more than `capacity` bytes.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cursor: 0,
            high_water: 0,
        }
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes consumed so far, including alignment padding.
    pub fn used(&self) -> usize {
        self.cursor
    }

    /// Bytes still allocatable (before alignment padding).
    pub fn remaining(&self) -> usize {
        self.capacity - self.cursor
    }

    /// Largest `used()` value observed since creation.
    pub fn high_water(&self) -> usize {
        self.high_water
    }

    /// Reserves `len` bytes at an offset that is a multiple of `align`
    /// (a power of two; zero is treated as one).
    pub fn alloc(&mut self, len: usize, align: usize) -> Result<ArenaRange> {
        let align = align.max(1);
        if !align.is_power_of_two() {
            return Err(CudaError::Driver(format!(
                "alignment {align} is not a power of two"
            )));
        }
        let exhausted = || CudaError::PoolExhausted {
            requested: len,
            available: self.remaining(),
            capacity: self.capacity,
        };
        let offset = self.cursor.checked_add(align - 1).ok_or_else(exhausted)? & !(align - 1);
        let end = offset.checked_add(len).ok_or_else(exhausted)?;
        if end > self.capacity {
            return Err(exhausted());
        }
        self.cursor = end;
        self.high_water = self.high_water.max(end);
        Ok(ArenaRange { offset, len })
    }

    /// Releases every allocation.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligns_and_respects_cap() {
        let mut a = Arena::new(100);
        let r0 = a.alloc(10, 1).unwrap();
        assert_eq!(r0, ArenaRange { offset: 0, len: 10 });
        let r1 = a.alloc(20, 16).unwrap();
        assert_eq!(r1.offset, 16);
        assert_eq!(a.used(), 36);
        assert!(matches!(
            a.alloc(65, 1),
            Err(CudaError::PoolExhausted { requested: 65, .. })
        ));
        assert!(a.alloc(64, 1).is_ok());
        assert_eq!(a.remaining(), 0);
    }

    #[test]
    fn reset_reuses_space_and_tracks_high_water() {
        let mut a = Arena::new(64);
        a.alloc(48, 1).unwrap();
        a.reset();
        assert_eq!(a.used(), 0);
        a.alloc(64, 8).unwrap();
        assert_eq!(a.high_water(), 64);
    }

    #[test]
    fn rejects_non_power_of_two_alignment_and_overflow() {
        let mut a = Arena::new(64);
        assert!(a.alloc(1, 3).is_err());
        assert!(a.alloc(usize::MAX, 1).is_err());
    }
}
