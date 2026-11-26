//! Page layout detection for Scriptor.
//!
//! The crate turns page rasters into ordered, labelled layout regions in the
//! way the PP-DocLayoutV3 reference pipeline does, as pure functions over
//! plain buffers:
//!
//! * [`labels`] holds the 25-class label table and the label-to-task mapping.
//! * [`preprocess`] converts a page image into the model's 800x800 CHW tensor.
//! * [`decode`] turns raw model outputs (class logits, normalised boxes, and
//!   pairwise reading-order logits) into scored, ranked page-pixel detections.
//! * [`postprocess`] applies NMS, large-image filtering, containment merging,
//!   unclipping, clamping, and 0..1000 normalisation.
//! * [`model`] defines the [`LayoutModel`] backend trait, the [`RawOutputs`]
//!   tensors it returns, and a [`MockLayout`] backend for tests.
//!
//! The optional `onnx` feature adds [`onnx::OnnxLayout`], an ONNX Runtime
//! backend. The crate depends on no other workspace crate.

pub mod decode;
pub mod labels;
pub mod model;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod postprocess;
pub mod preprocess;

pub use decode::{DecodeConfig, Detection, decode_page, order_ranks};
pub use labels::{LABELS, NUM_CLASSES, Task, TaskMapping, label_name};
pub use model::{LayoutError, LayoutModel, MockLayout, RawOutputs};
#[cfg(feature = "onnx")]
pub use onnx::OnnxLayout;
pub use postprocess::{
    LayoutBox, LayoutRegion, MergeMode, MergeModes, PostprocessConfig, postprocess_page, to_regions,
};
pub use preprocess::{INPUT_SIZE, PreprocessedPage, preprocess};

/// Settings for the full detection pipeline.
#[derive(Debug, Clone)]
pub struct LayoutConfig {
    pub decode: DecodeConfig,
    pub postprocess: PostprocessConfig,
    pub tasks: TaskMapping,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            decode: DecodeConfig::default(),
            postprocess: PostprocessConfig::default(),
            tasks: TaskMapping::bridge(),
        }
    }
}

/// Runs `model` over `pages` and returns the ordered regions of each page.
///
/// # Errors
///
/// Returns the backend error, a shape error when the backend returns a
/// different number of outputs than pages, or a decode error for malformed
/// tensors.
pub fn detect_regions(
    model: &dyn LayoutModel,
    pages: &[image::RgbImage],
    config: &LayoutConfig,
) -> Result<Vec<Vec<LayoutRegion>>, LayoutError> {
    let outputs = model.detect(pages)?;
    if outputs.len() != pages.len() {
        return Err(LayoutError::Shape(format!(
            "backend returned {} outputs for {} pages",
            outputs.len(),
            pages.len()
        )));
    }
    pages
        .iter()
        .zip(&outputs)
        .map(|(page, raw)| {
            let (width, height) = page.dimensions();
            let detections = decode_page(raw, width, height, &config.decode)?;
            let boxes = postprocess_page(&detections, width, height, &config.postprocess);
            Ok(to_regions(&boxes, width, height, &config.tasks))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_regions_runs_the_whole_pipeline_on_a_mock() {
        let mut raw = RawOutputs::blank(300, NUM_CLASSES);
        raw.set_query(0, 22, 4.0, [0.5, 0.25, 0.5, 0.125]);
        raw.set_query(1, 6, 4.0, [0.5, 0.75, 0.5, 0.125]);
        raw.set_order(0, 1, 6.0);
        let page = image::RgbImage::new(200, 400);
        let regions = detect_regions(
            &MockLayout::new(vec![raw]),
            std::slice::from_ref(&page),
            &LayoutConfig::default(),
        )
        .unwrap();
        assert_eq!(regions.len(), 1);
        let labels: Vec<_> = regions[0].iter().map(|r| r.label.as_str()).collect();
        assert_eq!(labels, ["text", "doc_title"]);
        assert_eq!(regions[0][0].bbox_2d, [250, 187, 750, 312]);
        assert_eq!(regions[0][1].index, 1);
    }

    #[test]
    fn detect_regions_rejects_output_count_mismatch() {
        let pages = [image::RgbImage::new(4, 4), image::RgbImage::new(4, 4)];
        let mock = MockLayout::new(vec![RawOutputs::blank(300, NUM_CLASSES)]);
        assert!(detect_regions(&mock, &pages, &LayoutConfig::default()).is_err());
    }
}
