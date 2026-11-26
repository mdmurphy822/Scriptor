use half::bf16;

use crate::grid::{GridThw, PATCH_ELEMS, PATCH_SIZE, TEMPORAL_PATCH_SIZE};
use crate::jpeg::RgbImage;
use crate::normalize::bf16_tables;

/// Returns the flat pixel `(row, col)` origin of patch row `index` of the output matrix.
///
/// Patches are ordered `(gh/2, gw/2, 2, 2)`: merge-window row, merge-window column,
/// then the row and column inside the window.
pub fn patch_origin(grid: &GridThw, index: usize) -> (u32, u32) {
    let windows_per_row = (grid.w / 2) as usize;
    let window = index / 4;
    let inner = index % 4;
    let patch_row = (window / windows_per_row) as u32 * 2 + (inner / 2) as u32;
    let patch_col = (window % windows_per_row) as u32 * 2 + (inner % 2) as u32;
    (patch_row * PATCH_SIZE, patch_col * PATCH_SIZE)
}

/// Offset within a patch row of element `(c, t, y, x)`: layout `(c, t=2, 14, 14)`.
pub fn inner_offset(c: usize, t: usize, y: usize, x: usize) -> usize {
    let p = PATCH_SIZE as usize;
    ((c * TEMPORAL_PATCH_SIZE as usize + t) * p + y) * p + x
}

/// Normalises `image` and lays it out as the `[N, 1176]` bf16 patch matrix, repeating
/// the frame along the temporal axis. The image sides must be multiples of 28.
pub fn patchify_bf16(image: &RgbImage) -> (GridThw, Vec<bf16>) {
    assert!(
        image.width.is_multiple_of(28) && image.height.is_multiple_of(28),
        "image sides must be multiples of 28"
    );
    let grid = GridThw::for_image(image.height, image.width);
    let tables = bf16_tables();
    let p = PATCH_SIZE as usize;
    let mut out = vec![bf16::ZERO; grid.patches() * PATCH_ELEMS];
    for index in 0..grid.patches() {
        let (row0, col0) = patch_origin(&grid, index);
        let dst = &mut out[index * PATCH_ELEMS..(index + 1) * PATCH_ELEMS];
        for y in 0..p {
            for x in 0..p {
                let px = image.pixel(row0 + y as u32, col0 + x as u32);
                for (c, &value) in px.iter().enumerate() {
                    let v = tables[c][value as usize];
                    dst[inner_offset(c, 0, y, x)] = v;
                    dst[inner_offset(c, 1, y, x)] = v;
                }
            }
        }
    }
    (grid, out)
}
