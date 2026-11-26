//! Detection postprocessing: NMS, large-image filter, containment merge,
//! unclip, clamp, and 0..1000 normalisation.

use crate::decode::Detection;
use crate::labels::{
    CHART_CLASS, IMAGE_CLASS, NUM_CLASSES, REFERENCE_CLASS, SEAL_CLASS, Task, TaskMapping,
    label_name,
};

/// IoU at which a lower-scored box of the same class is suppressed.
pub const NMS_IOU_SAME_CLASS: f64 = 0.6;
/// IoU at which a lower-scored box of a different class is suppressed.
pub const NMS_IOU_CROSS_CLASS: f64 = 0.98;
/// Fraction of a landscape page above which an `image` box is dropped.
pub const LARGE_IMAGE_AREA_LANDSCAPE: f64 = 0.82;
/// Fraction of a portrait page above which an `image` box is dropped.
pub const LARGE_IMAGE_AREA_PORTRAIT: f64 = 0.93;
/// Fraction of a box that must lie inside another for containment.
pub const CONTAINMENT_RATIO: f64 = 0.8;

/// How nested boxes of a class are merged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeMode {
    /// Keep every box.
    Union,
    /// Drop boxes contained in a larger box.
    Large,
    /// Drop boxes that contain a smaller box of the category.
    Small,
}

/// Containment merge configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeModes {
    /// One mode applied to every class.
    Uniform(MergeMode),
    /// Modes applied one class at a time, in order, to a shared keep mask.
    PerClass(Vec<(usize, MergeMode)>),
}

/// Postprocessing settings.
#[derive(Debug, Clone, PartialEq)]
pub struct PostprocessConfig {
    /// Whether to run NMS.
    pub nms: bool,
    /// Width and height expansion ratios; `None` disables unclipping.
    pub unclip: Option<(f64, f64)>,
    /// Containment merge; `None` disables merging.
    pub merge: Option<MergeModes>,
}

impl Default for PostprocessConfig {
    /// NMS on, unclip ratio `(1.0, 1.0)`, and per-class merge modes with
    /// `large` for every class except `reference`, which is `small`.
    fn default() -> Self {
        let modes = (0..NUM_CLASSES)
            .map(|c| {
                (
                    c,
                    if c == REFERENCE_CLASS {
                        MergeMode::Small
                    } else {
                        MergeMode::Large
                    },
                )
            })
            .collect();
        Self {
            nms: true,
            unclip: Some((1.0, 1.0)),
            merge: Some(MergeModes::PerClass(modes)),
        }
    }
}

/// A postprocessed box in page pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutBox {
    /// Class id.
    pub cls_id: usize,
    /// Class name, `class_<id>` outside the label table.
    pub label: String,
    /// Class probability.
    pub score: f32,
    /// Integer `(x1, y1, x2, y2)` in page pixels, truncated after clamping.
    pub coordinate: [i32; 4],
    /// Reading-order rank when positive.
    pub order: Option<usize>,
}

/// A region with its recognition task and 0..1000 page-relative box.
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutRegion {
    /// Position among the kept regions of the page.
    pub index: usize,
    /// Class name.
    pub label: String,
    /// Class probability.
    pub score: f32,
    /// `(x1, y1, x2, y2)` scaled to 0..1000 by truncation.
    pub bbox_2d: [i32; 4],
    /// What recognition does with the region.
    pub task: Task,
}

#[derive(Debug, Clone, Copy)]
struct Row {
    cls: usize,
    score: f64,
    b: [f64; 4],
    order: usize,
}

fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let inter_w = (a[2].min(b[2]) - a[0].max(b[0]) + 1.0).max(0.0);
    let inter_h = (a[3].min(b[3]) - a[1].max(b[1]) + 1.0).max(0.0);
    let inter = inter_w * inter_h;
    let area_a = (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0);
    let area_b = (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0);
    inter / (area_a + area_b - inter)
}

fn nms(rows: &[Row], iou_same: f64, iou_diff: f64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..rows.len()).collect();
    indices.sort_by(|&x, &y| rows[x].score.total_cmp(&rows[y].score));
    indices.reverse();
    let mut selected = Vec::new();
    while !indices.is_empty() {
        let current = indices.remove(0);
        selected.push(current);
        indices.retain(|&i| {
            let threshold = if rows[current].cls == rows[i].cls {
                iou_same
            } else {
                iou_diff
            };
            iou(&rows[current].b, &rows[i].b) < threshold
        });
    }
    selected
}

fn is_contained(inner: &Row, outer: &Row) -> bool {
    let (a, b) = (&inner.b, &outer.b);
    let area = (a[2] - a[0]) * (a[3] - a[1]);
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let ratio = if area > 0.0 { w * h / area } else { 0.0 };
    ratio >= CONTAINMENT_RATIO
}

fn is_preserved(cls: usize) -> bool {
    matches!(cls, IMAGE_CLASS | SEAL_CLASS | CHART_CLASS)
}

/// Returns `(contains_other, contained_by_other)` flags per row.
fn check_containment(
    rows: &[Row],
    preserve: bool,
    category: Option<(usize, MergeMode)>,
) -> (Vec<bool>, Vec<bool>) {
    let n = rows.len();
    let mut contains_other = vec![false; n];
    let mut contained_by_other = vec![false; n];
    for i in 0..n {
        for j in 0..n {
            if i == j || (preserve && is_preserved(rows[i].cls)) {
                continue;
            }
            let hit = match category {
                Some((cat, MergeMode::Large)) => {
                    rows[j].cls == cat && is_contained(&rows[i], &rows[j])
                }
                Some((cat, MergeMode::Small)) => {
                    rows[i].cls == cat && is_contained(&rows[i], &rows[j])
                }
                Some((_, MergeMode::Union)) => false,
                None => is_contained(&rows[i], &rows[j]),
            };
            if hit {
                contained_by_other[i] = true;
                contains_other[j] = true;
            }
        }
    }
    (contains_other, contained_by_other)
}

fn keep_for(mode: MergeMode, contains: bool, contained: bool) -> bool {
    match mode {
        MergeMode::Union => true,
        MergeMode::Large => !contained,
        MergeMode::Small => !contains || contained,
    }
}

fn apply_merge(rows: Vec<Row>, merge: &MergeModes) -> Vec<Row> {
    let keep: Vec<bool> = match merge {
        MergeModes::Uniform(MergeMode::Union) => return rows,
        MergeModes::Uniform(mode) => {
            let (contains, contained) = check_containment(&rows, true, None);
            (0..rows.len())
                .map(|i| keep_for(*mode, contains[i], contained[i]))
                .collect()
        }
        MergeModes::PerClass(modes) => {
            let mut keep = vec![true; rows.len()];
            for &(category, mode) in modes {
                if mode == MergeMode::Union {
                    continue;
                }
                let (contains, contained) = check_containment(&rows, true, Some((category, mode)));
                for (i, flag) in keep.iter_mut().enumerate() {
                    *flag &= keep_for(mode, contains[i], contained[i]);
                }
            }
            keep
        }
    };
    rows.into_iter()
        .zip(keep)
        .filter_map(|(r, k)| k.then_some(r))
        .collect()
}

fn filter_large_images(rows: Vec<Row>, width: f64, height: f64) -> Vec<Row> {
    if rows.len() <= 1 {
        return rows;
    }
    let threshold = if width > height {
        LARGE_IMAGE_AREA_LANDSCAPE
    } else {
        LARGE_IMAGE_AREA_PORTRAIT
    };
    let page_area = width * height;
    let kept: Vec<Row> = rows
        .iter()
        .copied()
        .filter(|r| {
            if r.cls != IMAGE_CLASS {
                return true;
            }
            let x1 = r.b[0].max(0.0);
            let y1 = r.b[1].max(0.0);
            let x2 = r.b[2].min(width);
            let y2 = r.b[3].min(height);
            (x2 - x1) * (y2 - y1) <= threshold * page_area
        })
        .collect();
    if kept.is_empty() { rows } else { kept }
}

fn unclip(rows: &mut [Row], (rw, rh): (f64, f64)) {
    for r in rows {
        let w = r.b[2] - r.b[0];
        let h = r.b[3] - r.b[1];
        let (new_w, new_h) = (w * rw, h * rh);
        let cx = r.b[0] + w / 2.0;
        let cy = r.b[1] + h / 2.0;
        r.b = [
            cx - new_w / 2.0,
            cy - new_h / 2.0,
            cx + new_w / 2.0,
            cy + new_h / 2.0,
        ];
    }
}

/// Postprocesses one page of detections in the order: NMS (boxes sorted by
/// descending score; a box is dropped when its IoU, computed with the
/// inclusive-pixel `+1` convention, reaches 0.6 against a kept box of its
/// class or 0.98 against one of another class), large-image filter, containment
/// merge, sort by reading-order rank, unclip, clamp to the page, and removal of
/// empty boxes. Coordinates are truncated to integers.
#[must_use]
pub fn postprocess_page(
    detections: &[Detection],
    page_width: u32,
    page_height: u32,
    config: &PostprocessConfig,
) -> Vec<LayoutBox> {
    let (width, height) = (f64::from(page_width), f64::from(page_height));
    let mut rows: Vec<Row> = detections
        .iter()
        .map(|d| Row {
            cls: d.label,
            score: f64::from(d.score),
            b: d.bbox.map(f64::from),
            order: d.order,
        })
        .collect();
    if rows.is_empty() {
        return Vec::new();
    }
    if config.nms {
        let selected = nms(&rows, NMS_IOU_SAME_CLASS, NMS_IOU_CROSS_CLASS);
        rows = selected.into_iter().map(|i| rows[i]).collect();
    }
    rows = filter_large_images(rows, width, height);
    if let Some(merge) = &config.merge {
        rows = apply_merge(rows, merge);
    }
    if rows.is_empty() {
        return Vec::new();
    }
    rows.sort_by_key(|r| r.order);
    if let Some(ratio) = config.unclip {
        unclip(&mut rows, ratio);
    }
    rows.iter()
        .filter_map(|r| {
            let x1 = r.b[0].min(width).max(0.0);
            let y1 = r.b[1].min(height).max(0.0);
            let x2 = r.b[2].min(width).max(0.0);
            let y2 = r.b[3].min(height).max(0.0);
            if x1 >= x2 || y1 >= y2 {
                return None;
            }
            Some(LayoutBox {
                cls_id: r.cls,
                label: label_name(r.cls).map_or_else(|| format!("class_{}", r.cls), str::to_owned),
                score: r.score as f32,
                coordinate: [x1 as i32, y1 as i32, x2 as i32, y2 as i32],
                order: (r.order > 0).then_some(r.order),
            })
        })
        .collect()
}

/// Maps boxes to regions: boxes whose label has no task or is
/// [`Task::Abandon`] are dropped, the rest are renumbered from 0 and their
/// coordinates scaled to 0..1000 by `int(x / extent * 1000)`.
#[must_use]
pub fn to_regions(
    boxes: &[LayoutBox],
    page_width: u32,
    page_height: u32,
    tasks: &TaskMapping,
) -> Vec<LayoutRegion> {
    let (w, h) = (f64::from(page_width), f64::from(page_height));
    let norm = |v: i32, extent: f64| (f64::from(v) / extent * 1000.0) as i32;
    boxes
        .iter()
        .filter_map(|b| {
            let task = tasks.task_for(&b.label).filter(|t| *t != Task::Abandon)?;
            Some((b, task))
        })
        .enumerate()
        .map(|(index, (b, task))| LayoutRegion {
            index,
            label: b.label.clone(),
            score: b.score,
            bbox_2d: [
                norm(b.coordinate[0], w),
                norm(b.coordinate[1], h),
                norm(b.coordinate[2], w),
                norm(b.coordinate[3], h),
            ],
            task,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(label: usize, score: f32, bbox: [f32; 4], order: usize) -> Detection {
        Detection {
            score,
            label,
            bbox,
            order,
        }
    }

    fn run(dets: &[Detection], w: u32, h: u32) -> Vec<LayoutBox> {
        postprocess_page(dets, w, h, &PostprocessConfig::default())
    }

    fn without_merge() -> PostprocessConfig {
        PostprocessConfig {
            merge: None,
            ..PostprocessConfig::default()
        }
    }

    #[test]
    fn iou_uses_the_inclusive_pixel_convention() {
        let a = [0.0, 0.0, 9.0, 9.0];
        assert!((iou(&a, &a) - 1.0).abs() < 1e-12);
        // Touching edge pixels overlap by one row of 10 pixels: 10 / (100 + 100 - 10).
        let b = [0.0, 9.0, 9.0, 18.0];
        assert!((iou(&a, &b) - 10.0 / 190.0).abs() < 1e-12);
    }

    #[test]
    fn nms_suppresses_same_class_overlap_but_keeps_other_classes() {
        let a = det(22, 0.9, [0.0, 0.0, 100.0, 100.0], 1);
        let same = det(22, 0.8, [2.0, 2.0, 100.0, 100.0], 2);
        let other = det(7, 0.7, [2.0, 2.0, 100.0, 100.0], 3);
        let out = postprocess_page(&[same, a, other], 1000, 1000, &without_merge());
        let classes: Vec<_> = out.iter().map(|b| b.cls_id).collect();
        assert_eq!(classes, [22, 7]);
    }

    #[test]
    fn nms_drops_near_identical_cross_class_boxes() {
        let a = det(22, 0.9, [0.0, 0.0, 100.0, 100.0], 1);
        let b = det(7, 0.8, [0.0, 0.0, 100.0, 100.0], 2);
        assert_eq!(
            postprocess_page(&[a, b], 1000, 1000, &without_merge()).len(),
            1
        );
    }

    #[test]
    fn large_images_are_dropped_by_page_orientation() {
        let text = det(22, 0.9, [0.0, 0.0, 10.0, 10.0], 1);
        let image = det(IMAGE_CLASS, 0.9, [0.0, 0.0, 90.0, 90.0], 2);
        // 90% of a portrait page is kept (0.93); of a landscape page it is dropped (0.82).
        let cfg = without_merge();
        assert_eq!(postprocess_page(&[text, image], 100, 100, &cfg).len(), 2);
        let wide = det(IMAGE_CLASS, 0.9, [0.0, 0.0, 190.0, 95.0], 2);
        assert_eq!(postprocess_page(&[text, wide], 200, 100, &cfg).len(), 1);
        // A lone image is never filtered.
        assert_eq!(postprocess_page(&[wide], 200, 100, &cfg).len(), 1);
    }

    #[test]
    fn contained_boxes_are_dropped_unless_preserved() {
        let outer = det(22, 0.9, [0.0, 0.0, 100.0, 100.0], 1);
        let inner = det(6, 0.8, [10.0, 10.0, 50.0, 50.0], 2);
        let out = run(&[outer, inner], 1000, 1000);
        assert_eq!(out.iter().map(|b| b.cls_id).collect::<Vec<_>>(), [22]);
        let image_inner = det(IMAGE_CLASS, 0.8, [10.0, 10.0, 50.0, 50.0], 2);
        assert_eq!(run(&[outer, image_inner], 1000, 1000).len(), 2);
    }

    #[test]
    fn per_class_modes_share_one_keep_mask() {
        // The container's own `large` pass drops the reference box, and the
        // reference category's `small` pass drops the container.
        let container = det(22, 0.9, [0.0, 0.0, 100.0, 100.0], 1);
        let reference = det(REFERENCE_CLASS, 0.8, [10.0, 10.0, 50.0, 50.0], 2);
        assert!(run(&[container, reference], 1000, 1000).is_empty());
    }

    #[test]
    fn uniform_modes_follow_the_reference_rules() {
        let outer = det(22, 0.9, [0.0, 0.0, 100.0, 100.0], 1);
        let inner = det(6, 0.8, [10.0, 10.0, 50.0, 50.0], 2);
        let with = |mode| PostprocessConfig {
            merge: Some(MergeModes::Uniform(mode)),
            ..PostprocessConfig::default()
        };
        let both = [outer, inner];
        let classes = |mode| {
            let out = postprocess_page(&both, 1000, 1000, &with(mode));
            out.iter().map(|b| b.cls_id).collect::<Vec<_>>()
        };
        assert_eq!(classes(MergeMode::Large), [22]);
        assert_eq!(classes(MergeMode::Small), [6]);
        assert_eq!(classes(MergeMode::Union).len(), 2);
    }

    #[test]
    fn output_is_in_reading_order_clamped_and_truncated() {
        let late = det(22, 0.9, [10.5, 10.5, 50.9, 50.9], 5);
        let early = det(6, 0.8, [-20.0, 200.0, 40.0, 900.0], 2);
        let empty = det(22, 0.8, [300.0, 300.0, 300.0, 400.0], 1);
        let out = run(&[late, early, empty], 100, 500);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].coordinate, [0, 200, 40, 500]);
        assert_eq!(out[0].order, Some(2));
        assert_eq!(out[1].coordinate, [10, 10, 50, 50]);
    }

    #[test]
    fn unclip_scales_about_the_centre() {
        let cfg = PostprocessConfig {
            unclip: Some((2.0, 1.0)),
            ..PostprocessConfig::default()
        };
        let out = postprocess_page(&[det(22, 0.9, [40.0, 40.0, 60.0, 60.0], 1)], 100, 100, &cfg);
        assert_eq!(out[0].coordinate, [30, 40, 70, 60]);
    }

    #[test]
    fn regions_apply_tasks_drop_abandoned_and_normalise() {
        let boxes = run(
            &[
                det(22, 0.9, [10.0, 20.0, 110.0, 80.0], 1),
                det(2, 0.9, [200.0, 0.0, 300.0, 50.0], 2),
                det(21, 0.9, [0.0, 100.0, 399.0, 180.0], 3),
            ],
            400,
            200,
        );
        let regions = to_regions(&boxes, 400, 200, &TaskMapping::bridge());
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].label, "text");
        assert_eq!(regions[0].task, Task::Text);
        assert_eq!(regions[0].bbox_2d, [25, 100, 275, 400]);
        assert_eq!((regions[1].index, regions[1].task), (1, Task::Table));
        assert_eq!(regions[1].bbox_2d, [0, 500, 997, 900]);
    }
}
