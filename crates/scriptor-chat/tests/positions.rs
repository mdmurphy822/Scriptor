//! M-RoPE position tests.

use scriptor_chat::{ImageSpan, decode_position, mrope_positions};

#[test]
fn text_image_text_matrix() {
    let positions = mrope_positions(
        20,
        &[ImageSpan {
            range: 5..17,
            llm_grid: (1, 3, 4),
        }],
    )
    .unwrap();
    let text_before: Vec<i64> = (0..5).collect();
    let h: Vec<i64> = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2].map(|v| v + 5).to_vec();
    let w: Vec<i64> = [0, 1, 2, 3].repeat(3).iter().map(|v| v + 5).collect();
    let text_after: Vec<i64> = vec![9, 10, 11];
    let build = |mid: Vec<i64>| [text_before.clone(), mid, text_after.clone()].concat();
    assert_eq!(positions.rows[0], build(vec![5; 12]));
    assert_eq!(positions.rows[1], build(h));
    assert_eq!(positions.rows[2], build(w));
    assert_eq!(positions.len(), 20);
    assert_eq!(positions.delta, 12 - 20);
    assert_eq!(decode_position(20, positions.delta), 12);
}

#[test]
fn text_only_is_identity() {
    let positions = mrope_positions(4, &[]).unwrap();
    assert_eq!(
        positions.rows,
        [vec![0, 1, 2, 3], vec![0, 1, 2, 3], vec![0, 1, 2, 3]]
    );
    assert_eq!(positions.delta, 0);
}

#[test]
fn consecutive_images_continue_from_the_larger_extent() {
    let positions = mrope_positions(
        6,
        &[
            ImageSpan {
                range: 1..3,
                llm_grid: (1, 1, 2),
            },
            ImageSpan {
                range: 3..5,
                llm_grid: (1, 2, 1),
            },
        ],
    )
    .unwrap();
    // Text 0; image one at 1 (w 1,2) cursor -> 3; image two at 3 (h 3,4) cursor -> 5; text 5.
    assert_eq!(positions.rows[0], vec![0, 1, 1, 3, 3, 5]);
    assert_eq!(positions.rows[1], vec![0, 1, 1, 3, 4, 5]);
    assert_eq!(positions.rows[2], vec![0, 1, 2, 3, 3, 5]);
    assert_eq!(positions.delta, 6 - 6);
}

#[test]
fn mismatched_span_is_rejected() {
    let err = mrope_positions(
        4,
        &[ImageSpan {
            range: 0..3,
            llm_grid: (1, 2, 2),
        }],
    );
    assert!(err.is_err());
}
