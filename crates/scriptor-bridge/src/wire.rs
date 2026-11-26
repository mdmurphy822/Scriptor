//! The reply wire types of the OCR bridge: the regions of a document, their
//! provenance and links, and the reply that carries them. The serde shapes
//! are the wire contract.

use serde::{Deserialize, Serialize};

/// The placeholder text a redacted span is replaced by.
pub const REDACTED_PLACEHOLDER: &str = "\u{e000}redacted\u{e001}";
/// The placeholder text a checked form field is replaced by.
pub const CHECKED_PLACEHOLDER: &str = "\u{e000}checked\u{e001}";
/// The placeholder text an unchecked form field is replaced by.
pub const UNCHECKED_PLACEHOLDER: &str = "\u{e000}unchecked\u{e001}";

/// A record of the stage that produced or changed a region.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Provenance {
    pub stage: String,
    pub provider: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_sha256: Option<String>,
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn one() -> i64 {
    1
}

fn paragraph() -> String {
    "paragraph".into()
}

/// A link over the characters `start..end` (Unicode scalar offsets, end
/// exclusive) of a region's text, pointing at `target`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct LinkSpan {
    pub start: usize,
    pub end: usize,
    pub target: String,
}

/// The sub-kind of a page-furniture region: a running header, a running
/// footer, a page number, or any other furniture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FurnitureKind {
    Header,
    Footer,
    PageNumber,
    Other,
}

/// One region of a document in reading order.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Region {
    pub id: String,
    #[serde(default = "one")]
    pub page: i64,
    #[serde(default = "paragraph")]
    pub kind: String,
    #[serde(default)]
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[f64; 4]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heading_level: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alt_text: Option<String>,
    /// Optional prose for charts, diagrams and images of tables, covering data and trends.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub long_description: Option<String>,
    /// True when the figure is purely ornamental and carries no text alternative.
    #[serde(default, skip_serializing_if = "is_false")]
    pub decorative: bool,
    /// The sub-kind of a `page_furniture` region; absent when unclassified or
    /// when the region is not page furniture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub furniture: Option<FurnitureKind>,
    /// BCP 47 language of this whole region when it differs from the
    /// document language; absent means the region inherits it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
    /// Links over character ranges of `text`: external addresses and `#` plus a region id for a link to another region.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<LinkSpan>,
    #[serde(default)]
    pub provenance: Vec<Provenance>,
}

/// The reply of a layout request.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct Reply {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub regions: Vec<Region>,
    /// The one-based pages the reader reports ending with no region.
    #[serde(default)]
    pub empty_pages: Vec<i64>,
}
