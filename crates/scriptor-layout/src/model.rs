//! The layout backend contract and a test backend.

use image::RgbImage;

/// Failure of a layout backend or of tensor validation.
#[derive(Debug, thiserror::Error)]
pub enum LayoutError {
    /// A tensor or output list had an unexpected size.
    #[error("layout tensor shape: {0}")]
    Shape(String),
    /// The backend failed to run.
    #[error("layout backend: {0}")]
    Backend(String),
}

/// Raw model outputs for one page, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct RawOutputs {
    /// Number of object queries.
    pub queries: usize,
    /// Number of classes per query.
    pub classes: usize,
    /// Class logits, `queries x classes`.
    pub logits: Vec<f32>,
    /// Boxes as normalised `(cx, cy, w, h)`, `queries x 4`.
    pub pred_boxes: Vec<f32>,
    /// Pairwise reading-order logits, `queries x queries`.
    pub order_logits: Vec<f32>,
}

impl RawOutputs {
    /// Outputs in which no query fires: every class logit is -20 and every
    /// order logit is 0.
    #[must_use]
    pub fn blank(queries: usize, classes: usize) -> Self {
        Self {
            queries,
            classes,
            logits: vec![-20.0; queries * classes],
            pred_boxes: vec![0.0; queries * 4],
            order_logits: vec![0.0; queries * queries],
        }
    }

    /// Sets the box of `query` and the logit of one `class` for it.
    pub fn set_query(&mut self, query: usize, class: usize, logit: f32, cxcywh: [f32; 4]) {
        self.logits[query * self.classes + class] = logit;
        self.pred_boxes[query * 4..query * 4 + 4].copy_from_slice(&cxcywh);
    }

    /// Sets the order logit at `(row, col)`.
    pub fn set_order(&mut self, row: usize, col: usize, logit: f32) {
        self.order_logits[row * self.queries + col] = logit;
    }

    /// Checks that the buffers match the declared dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::Shape`] naming the mismatched tensor.
    pub fn validate(&self) -> Result<(), LayoutError> {
        let checks = [
            ("logits", self.logits.len(), self.queries * self.classes),
            ("pred_boxes", self.pred_boxes.len(), self.queries * 4),
            (
                "order_logits",
                self.order_logits.len(),
                self.queries * self.queries,
            ),
        ];
        for (name, got, want) in checks {
            if got != want {
                return Err(LayoutError::Shape(format!(
                    "{name} has {got} values, expected {want}"
                )));
            }
        }
        Ok(())
    }
}

/// A layout inference backend.
pub trait LayoutModel {
    /// Runs the model over `pages`, returning one [`RawOutputs`] per page in
    /// order.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::Backend`] when inference fails.
    fn detect(&self, pages: &[RgbImage]) -> Result<Vec<RawOutputs>, LayoutError>;
}

/// A backend returning canned outputs: page `i` receives `outputs[i]`.
#[derive(Debug, Clone, Default)]
pub struct MockLayout {
    outputs: Vec<RawOutputs>,
}

impl MockLayout {
    /// Creates a mock that answers page `i` with `outputs[i]`.
    #[must_use]
    pub fn new(outputs: Vec<RawOutputs>) -> Self {
        Self { outputs }
    }
}

impl LayoutModel for MockLayout {
    fn detect(&self, pages: &[RgbImage]) -> Result<Vec<RawOutputs>, LayoutError> {
        if pages.len() > self.outputs.len() {
            return Err(LayoutError::Backend(format!(
                "mock has {} outputs for {} pages",
                self.outputs.len(),
                pages.len()
            )));
        }
        Ok(self.outputs[..pages.len()].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_answers_pages_in_order_and_fails_when_short() {
        let mut first = RawOutputs::blank(3, 2);
        first.set_query(1, 1, 2.0, [0.5; 4]);
        let mock = MockLayout::new(vec![first.clone(), RawOutputs::blank(3, 2)]);
        let pages = [RgbImage::new(1, 1)];
        assert_eq!(mock.detect(&pages).unwrap(), vec![first]);
        let three = [
            RgbImage::new(1, 1),
            RgbImage::new(1, 1),
            RgbImage::new(1, 1),
        ];
        assert!(matches!(mock.detect(&three), Err(LayoutError::Backend(_))));
    }

    #[test]
    fn validate_names_the_bad_tensor() {
        let mut raw = RawOutputs::blank(4, 2);
        assert!(raw.validate().is_ok());
        raw.order_logits.pop();
        let err = raw.validate().unwrap_err().to_string();
        assert!(err.contains("order_logits"), "{err}");
    }
}
