//! The self-test request that gates readiness.

/// Side of the synthetic self-test image in pixels.
pub const SELF_TEST_SIDE: u32 = 112;

/// Text part of the self-test prompt.
pub const SELF_TEST_PROMPT: &str = "Text Recognition:";

/// Generation limit of the self-test request.
pub const SELF_TEST_MAX_TOKENS: u32 = 8;

/// Row-major RGB bytes of the synthetic self-test image: a diagonal gradient with a dark
/// rectangle, so the vision tower sees structured input.
pub fn self_test_pixels() -> Vec<u8> {
    let side = SELF_TEST_SIDE as usize;
    let mut data = Vec::with_capacity(side * side * 3);
    for y in 0..side {
        for x in 0..side {
            let inside = (32..80).contains(&x) && (44..68).contains(&y);
            let v = if inside {
                24
            } else {
                (200 + (x + y) * 55 / (2 * side)) as u8
            };
            data.extend_from_slice(&[v, v, v]);
        }
    }
    data
}
