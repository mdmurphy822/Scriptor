//! The layout class table and the label-to-task mapping.

/// Number of layout classes the detector predicts.
pub const NUM_CLASSES: usize = 25;

/// Class names indexed by class id.
pub const LABELS: [&str; NUM_CLASSES] = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
];

/// Class id of the `image` label.
pub const IMAGE_CLASS: usize = 14;
/// Class id of the `chart` label.
pub const CHART_CLASS: usize = 3;
/// Class id of the `seal` label.
pub const SEAL_CLASS: usize = 20;
/// Class id of the `reference` label.
pub const REFERENCE_CLASS: usize = 18;

/// Returns the name of class `id`, or `None` outside the table.
#[must_use]
pub fn label_name(id: usize) -> Option<&'static str> {
    LABELS.get(id).copied()
}

/// What recognition does with a region of a given label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Recognised as plain text.
    Text,
    /// Recognised as a table.
    Table,
    /// Recognised as a formula.
    Formula,
    /// Kept as a region but not recognised.
    Skip,
    /// Dropped entirely.
    Abandon,
}

/// Ordered label-to-task assignments; the first task listing a label wins.
#[derive(Debug, Clone)]
pub struct TaskMapping {
    entries: Vec<(Task, Vec<&'static str>)>,
}

impl TaskMapping {
    /// Builds a mapping from `(task, labels)` entries in priority order.
    #[must_use]
    pub fn new(entries: Vec<(Task, Vec<&'static str>)>) -> Self {
        Self { entries }
    }

    /// The mapping the bridge configures: page furniture (`header`, `footer`,
    /// `number`) and `footnote` are recognised as text.
    #[must_use]
    pub fn bridge() -> Self {
        Self::new(vec![
            (
                Task::Text,
                vec![
                    "abstract",
                    "algorithm",
                    "content",
                    "doc_title",
                    "figure_title",
                    "paragraph_title",
                    "reference_content",
                    "text",
                    "vertical_text",
                    "footnote",
                    "vision_footnote",
                    "seal",
                    "formula_number",
                    "header",
                    "footer",
                    "number",
                ],
            ),
            (Task::Table, vec!["table"]),
            (Task::Formula, vec!["display_formula", "inline_formula"]),
            (Task::Skip, vec!["chart", "image"]),
            (
                Task::Abandon,
                vec!["aside_text", "reference", "footer_image", "header_image"],
            ),
        ])
    }

    /// The default mapping, which abandons page furniture and footnotes.
    #[must_use]
    pub fn sdk_default() -> Self {
        Self::new(vec![
            (
                Task::Text,
                vec![
                    "abstract",
                    "algorithm",
                    "content",
                    "doc_title",
                    "figure_title",
                    "paragraph_title",
                    "reference_content",
                    "text",
                    "vertical_text",
                    "vision_footnote",
                    "seal",
                    "formula_number",
                ],
            ),
            (Task::Table, vec!["table"]),
            (Task::Formula, vec!["display_formula", "inline_formula"]),
            (Task::Skip, vec!["chart", "image"]),
            (
                Task::Abandon,
                vec![
                    "header",
                    "footer",
                    "number",
                    "footnote",
                    "aside_text",
                    "reference",
                    "footer_image",
                    "header_image",
                ],
            ),
        ])
    }

    /// Returns the task of `label`, or `None` when no entry lists it.
    #[must_use]
    pub fn task_for(&self, label: &str) -> Option<Task> {
        self.entries
            .iter()
            .find(|(_, labels)| labels.contains(&label))
            .map(|(task, _)| *task)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_has_the_reference_ids() {
        assert_eq!(LABELS[IMAGE_CLASS], "image");
        assert_eq!(LABELS[CHART_CLASS], "chart");
        assert_eq!(LABELS[SEAL_CLASS], "seal");
        assert_eq!(LABELS[REFERENCE_CLASS], "reference");
        assert_eq!(label_name(5), Some("display_formula"));
        assert_eq!(label_name(25), None);
    }

    #[test]
    fn mappings_differ_on_furniture_only() {
        let bridge = TaskMapping::bridge();
        let sdk = TaskMapping::sdk_default();
        for label in LABELS {
            let differs = matches!(label, "header" | "footer" | "number" | "footnote");
            assert_eq!(
                bridge.task_for(label) != sdk.task_for(label),
                differs,
                "{label}"
            );
            assert!(bridge.task_for(label).is_some(), "{label}");
        }
        assert_eq!(bridge.task_for("table"), Some(Task::Table));
        assert_eq!(bridge.task_for("chart"), Some(Task::Skip));
        assert_eq!(bridge.task_for("reference"), Some(Task::Abandon));
        assert_eq!(bridge.task_for("class_30"), None);
    }
}
