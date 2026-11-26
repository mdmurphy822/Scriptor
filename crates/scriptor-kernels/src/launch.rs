//! Launch-geometry helpers shared by the kernel wrappers.

/// Grid, block and dynamic shared-memory extents of one kernel launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchDims {
    /// Thread blocks along x, y, z.
    pub grid: (u32, u32, u32),
    /// Threads per block along x, y, z.
    pub block: (u32, u32, u32),
    /// Dynamic shared memory in bytes.
    pub shared_bytes: u32,
}

/// Integer division rounding up.
pub fn ceil_div(n: usize, d: usize) -> usize {
    assert!(d > 0, "divisor must be positive");
    n.div_ceil(d)
}

/// Rounds `n` up to the next multiple of `multiple`.
pub fn round_up(n: usize, multiple: usize) -> usize {
    ceil_div(n, multiple) * multiple
}

/// Largest power of two not above `n`, at least 1.
pub fn floor_pow2(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        1 << (usize::BITS - 1 - n.leading_zeros())
    }
}

/// Block width for a row-wise reduction over `dim` elements: a power of two
/// that is a multiple of the 32-lane warp, between 32 and `max_threads`, sized
/// so each thread handles about `per_thread` elements.
pub fn reduction_block(dim: usize, per_thread: usize, max_threads: usize) -> usize {
    let wanted = ceil_div(dim, per_thread.max(1)).max(32);
    floor_pow2(wanted.next_power_of_two().min(max_threads.max(32)))
}

/// One block per row, reduction-sized blocks.
pub fn row_launch(rows: usize, dim: usize, per_thread: usize, max_threads: usize) -> LaunchDims {
    let block = reduction_block(dim, per_thread, max_threads) as u32;
    LaunchDims {
        grid: (rows as u32, 1, 1),
        block: (block, 1, 1),
        shared_bytes: 0,
    }
}

/// One thread per element over `elements`, blocks of `block` threads.
pub fn elementwise_launch(elements: usize, block: usize) -> LaunchDims {
    LaunchDims {
        grid: (ceil_div(elements, block) as u32, 1, 1),
        block: (block as u32, 1, 1),
        shared_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_and_round() {
        assert_eq!(ceil_div(0, 4), 0);
        assert_eq!(ceil_div(9, 4), 3);
        assert_eq!(round_up(9, 4), 12);
        assert_eq!(round_up(8, 4), 8);
    }

    #[test]
    fn pow2_floor() {
        assert_eq!(floor_pow2(0), 1);
        assert_eq!(floor_pow2(1), 1);
        assert_eq!(floor_pow2(1000), 512);
        assert_eq!(floor_pow2(1024), 1024);
    }

    #[test]
    fn reduction_block_is_warp_multiple_power_of_two() {
        for dim in [64, 1024, 1536, 4608, 59392] {
            let b = reduction_block(dim, 4, 1024);
            assert!(
                b.is_power_of_two() && (32..=1024).contains(&b),
                "{dim}: {b}"
            );
        }
        assert_eq!(reduction_block(1536, 4, 1024), 512);
        assert_eq!(reduction_block(8, 4, 1024), 32);
    }

    #[test]
    fn launches() {
        let l = row_launch(7, 1536, 4, 1024);
        assert_eq!(l.grid, (7, 1, 1));
        assert_eq!(l.block.0, 512);
        let e = elementwise_launch(1000, 256);
        assert_eq!(e.grid.0, 4);
    }
}
