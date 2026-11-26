//! Assembly of the OCR reply from formatted layout results: region
//! ids, kinds, blank handling, page-pixel boxes, page furniture, split-off
//! page numbers, the document title and the list of empty pages.

use std::fmt;
use std::sync::LazyLock;

use crate::wire::{FurnitureKind, Region};
use regex::Regex;

use crate::format::{PageResult, RegionResult, RegionTask, format_page, is_space, trim, trim_end};

/// The extent of the normalised box space of layout results.
const BBOX_EXTENT: f64 = 1_000.0;
/// The most regions one reply may hold.
const MAX_REGIONS: usize = 100_000;
/// The most bytes one region's text may hold.
const MAX_REGION_TEXT_BYTES: usize = 1024 * 1024;
/// The leading pages on which a heading or paragraph can confirm the
/// metadata title.
const METADATA_TITLE_PAGES: i64 = 3;
const GEOMETRY_ERROR: &str = "GLM-OCR SDK returned invalid geometry";

static WHOLE_FENCE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)\A```[A-Za-z0-9+#._-]*[ \t]*\n(.*?)\n?```\z").expect("whole fence pattern")
});

/// Why a reply could not be assembled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssembleError {
    /// Recognition failed; each pair names a failure category and its count.
    Recognition {
        diagnostics: Vec<(&'static str, usize)>,
    },
    /// The result was refused with a fixed, source-private message.
    Rejected(&'static str),
}

impl fmt::Display for AssembleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Recognition { .. } => formatter.write_str("GLM-OCR SDK recognition failed"),
            Self::Rejected(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for AssembleError {}

/// The assembled reply: the document title, the regions in reading order
/// and the one-based pages that ended with no region.
#[derive(Clone, Debug, PartialEq)]
pub struct Assembled {
    pub title: String,
    pub regions: Vec<Region>,
    pub empty_pages: Vec<i64>,
    /// The count of recognised regions whose text was blank.
    pub successful_blank_omitted: usize,
    /// The non-zero diagnostic counters by category name that the page
    /// recoveries produced.
    pub diagnostics: Vec<(&'static str, usize)>,
}

impl From<Assembled> for crate::wire::Reply {
    fn from(assembled: Assembled) -> Self {
        Self {
            title: assembled.title,
            regions: assembled.regions,
            empty_pages: assembled.empty_pages,
        }
    }
}

fn recognition_failure(diagnostics: Vec<(&'static str, usize)>) -> AssembleError {
    AssembleError::Recognition { diagnostics }
}

fn is_skip(region: &RegionResult) -> bool {
    region.task == RegionTask::Skip || matches!(region.label.as_str(), "chart" | "image")
}

/// Checks that every region that needs recognition has some, and returns
/// the count of recognitions that came back blank.
pub fn validate(pages: &[PageResult]) -> Result<usize, AssembleError> {
    let mut missing = 0usize;
    let mut blank = 0usize;
    for region in pages.iter().flat_map(|page| &page.regions) {
        if is_skip(region) {
            continue;
        }
        match &region.content {
            None => missing += 1,
            Some(content) if trim(content).is_empty() => blank += 1,
            Some(_) => {}
        }
    }
    if missing > 0 {
        return Err(recognition_failure(vec![("missing_recognition", missing)]));
    }
    Ok(blank)
}

/// Scales a box from the normalised 0..1000 space to page pixels; every
/// coordinate must be finite and inside the space and the box must have
/// positive width and height.
pub fn scale_bbox(
    value: Option<[f64; 4]>,
    page_dimensions: (f64, f64),
) -> Result<Option<[f64; 4]>, AssembleError> {
    let Some(coordinates) = value else {
        return Ok(None);
    };
    if coordinates
        .iter()
        .any(|c| !c.is_finite() || !(0.0..=BBOX_EXTENT).contains(c))
        || coordinates[0] >= coordinates[2]
        || coordinates[1] >= coordinates[3]
    {
        return Err(AssembleError::Rejected(GEOMETRY_ERROR));
    }
    let (width, height) = page_dimensions;
    Ok(Some([
        coordinates[0] * width / BBOX_EXTENT,
        coordinates[1] * height / BBOX_EXTENT,
        coordinates[2] * width / BBOX_EXTENT,
        coordinates[3] * height / BBOX_EXTENT,
    ]))
}

/// Removes a leading heading marker of one to six `#` and the whitespace
/// after it.
fn strip_heading_prefix(text: &str) -> &str {
    let hashes = text.bytes().take_while(|b| *b == b'#').count().min(6);
    if hashes == 0 {
        return text;
    }
    text[hashes..].trim_start_matches(is_space)
}

/// Whether a recognition is empty: only whitespace, or the model's empty
/// answer wrapped in a code fence, with or without a heading prefix.
pub fn is_blank_recognition(content: &str) -> bool {
    let stripped = trim(strip_heading_prefix(trim(content)));
    if stripped.is_empty() {
        return true;
    }
    WHOLE_FENCE
        .captures(stripped)
        .and_then(|captures| captures.get(1))
        .is_some_and(|body| trim(body.as_str()).is_empty())
}

/// The region kind of a layout block and its text. Titles are
/// headings, images and charts figures, figure titles captions, algorithm
/// blocks and text that is one fenced code block are code with the fence
/// lines removed, and text starting with `- ` is a list item.
pub fn region_kind(label: &str, native_label: &str, text: &str) -> (&'static str, String) {
    if matches!(native_label, "doc_title" | "paragraph_title") {
        return ("heading", trim(strip_heading_prefix(text)).to_string());
    }
    if label == "image" || matches!(native_label, "chart" | "image") {
        return ("figure", text.to_string());
    }
    if label == "table" {
        return ("table", text.to_string());
    }
    if label == "formula" {
        return ("formula", text.to_string());
    }
    if native_label == "figure_title" {
        return ("caption", text.to_string());
    }
    if let Some(body) = WHOLE_FENCE
        .captures(text)
        .and_then(|captures| captures.get(1))
        .map(|body| body.as_str())
        && !body.contains("```")
    {
        return ("code", body.trim_matches('\n').to_string());
    }
    if native_label == "algorithm" {
        return ("code", text.to_string());
    }
    if let Some(item) = text.strip_prefix("- ") {
        return ("list_item", trim(item).to_string());
    }
    ("paragraph", text.to_string())
}

fn furniture_kind(native_label: &str) -> Option<FurnitureKind> {
    match native_label {
        "header" => Some(FurnitureKind::Header),
        "footer" => Some(FurnitureKind::Footer),
        "number" => Some(FurnitureKind::PageNumber),
        _ => None,
    }
}

fn new_region(page: i64, kind: &str, text: String) -> Region {
    Region {
        id: String::new(),
        page,
        kind: kind.to_string(),
        text,
        bbox: None,
        heading_level: None,
        image: None,
        alt_text: None,
        long_description: None,
        decorative: false,
        furniture: None,
        lang: None,
        links: Vec::new(),
        provenance: Vec::new(),
    }
}

/// The `slot`-th of `slots` equal horizontal bands of `bbox`, top to bottom.
fn slot_of(bbox: [f64; 4], slot: usize, slots: usize) -> [f64; 4] {
    let height = bbox[3] - bbox[1];
    let at = |index: usize| bbox[1] + height * index as f64 / slots as f64;
    [bbox[0], at(slot), bbox[2], at(slot + 1)]
}

/// Replaces each table region whose leading rows are one full-width cell, or
/// that has all-empty rows, by a paragraph region per leading row followed by
/// the table without those rows. The paragraphs keep the table region's page
/// and provenance; its box is divided top to bottom into equal horizontal
/// slots, one per paragraph and one for the remaining table. A table left
/// with no row is replaced by its paragraphs.
fn detach_table_lead_rows(pages: &mut [(i64, Vec<Region>)]) {
    for (_, regions) in pages.iter_mut() {
        if !regions.iter().any(|region| region.kind == "table") {
            continue;
        }
        let mut rebuilt = Vec::with_capacity(regions.len());
        for region in std::mem::take(regions) {
            let detached = (region.kind == "table")
                .then(|| crate::textutil::detach_lead_rows(&region.text))
                .flatten();
            let Some(detached) = detached else {
                rebuilt.push(region);
                continue;
            };
            let slots = detached.paragraphs.len() + usize::from(detached.table.is_some());
            for (slot, paragraph) in detached.paragraphs.into_iter().enumerate() {
                let mut lead = region.clone();
                lead.kind = "paragraph".to_string();
                lead.text = paragraph;
                lead.links = Vec::new();
                lead.heading_level = None;
                lead.bbox = region.bbox.map(|bbox| slot_of(bbox, slot, slots));
                rebuilt.push(lead);
            }
            if let Some(table) = detached.table {
                let mut kept = region;
                kept.bbox = kept.bbox.map(|bbox| slot_of(bbox, slots - 1, slots));
                kept.text = table;
                kept.links = Vec::new();
                rebuilt.push(kept);
            }
        }
        *regions = rebuilt;
    }
}

/// The start of the ASCII digits ending `text` and those digits, when the
/// digits follow whitespace.
fn trailing_number(text: &str) -> Option<(usize, &str)> {
    let digits = text.bytes().rev().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    let start = text.len() - digits;
    text[..start].chars().next_back().filter(|c| is_space(*c))?;
    Some((start, &text[start..]))
}

/// Splits a printed page number the recognition glued onto the end of a
/// page's last body paragraph into its own `page_furniture` region of
/// `page_number` furniture placed after that paragraph. A page qualifies
/// when its last region is a paragraph ending in whitespace and a decimal
/// number and the page has no page number region. The number is split only
/// when it continues the numbering around it: the difference between it
/// and its page number equals that of the nearest page number read before
/// or after it, whether a page number region or another qualifying number.
fn split_trailing_page_numbers(pages: &mut [(i64, Vec<Region>)]) {
    let mut numbers: Vec<(i64, i128)> = Vec::new();
    let mut candidates: Vec<(usize, i128)> = Vec::new();
    for (index, (page_number, regions)) in pages.iter().enumerate() {
        let mut numbered = false;
        for region in regions
            .iter()
            .filter(|region| region.furniture == Some(FurnitureKind::PageNumber))
        {
            numbered = true;
            let text = trim(&region.text);
            if !text.is_empty()
                && text.bytes().all(|b| b.is_ascii_digit())
                && let Ok(value) = text.parse::<i128>()
            {
                numbers.push((*page_number, value - i128::from(*page_number)));
            }
        }
        if numbered {
            continue;
        }
        let Some(last) = regions.last() else {
            continue;
        };
        if last.kind != "paragraph" {
            continue;
        }
        let Some((_, digits)) = trailing_number(&last.text) else {
            continue;
        };
        let Ok(value) = digits.parse::<i128>() else {
            continue;
        };
        let offset = value - i128::from(*page_number);
        numbers.push((*page_number, offset));
        candidates.push((index, offset));
    }
    for (index, offset) in candidates {
        let page_number = pages[index].0;
        let before = numbers.iter().filter(|number| number.0 < page_number).max();
        let after = numbers.iter().filter(|number| number.0 > page_number).min();
        if ![before, after]
            .into_iter()
            .flatten()
            .any(|nearest| nearest.1 == offset)
        {
            continue;
        }
        let regions = &mut pages[index].1;
        let Some(last) = regions.last_mut() else {
            continue;
        };
        let Some((start, digits)) = trailing_number(&last.text) else {
            continue;
        };
        let digits = digits.to_string();
        let body = trim_end(&last.text[..start]).to_string();
        if body.is_empty() {
            continue;
        }
        last.text = body;
        let page = last.page;
        let mut number = new_region(page, "page_furniture", digits);
        number.furniture = Some(FurnitureKind::PageNumber);
        regions.push(number);
    }
}

/// Reduces text to its lowercase letters and digits.
fn comparable(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

/// The reply title: the metadata title when a heading or paragraph on one
/// of the first pages prints it, else the recognised document title.
pub fn confirmed_title(title: &str, regions: &[Region], metadata_title: &str) -> String {
    let wanted = comparable(metadata_title);
    let confirmed = !wanted.is_empty()
        && regions.iter().any(|region| {
            region.page <= METADATA_TITLE_PAGES
                && matches!(region.kind.as_str(), "heading" | "paragraph")
                && comparable(&region.text) == wanted
        });
    if confirmed {
        metadata_title.to_string()
    } else {
        title.to_string()
    }
}

/// One page's regions as built from its layout results, before ids,
/// page-number splitting and the document title are settled.
#[derive(Clone, Debug, PartialEq)]
pub struct BuiltPage {
    /// The text of the page's first document title, or empty.
    pub title: String,
    /// The index in `regions` of the region the title came from.
    pub title_region: Option<usize>,
    pub regions: Vec<Region>,
}

impl BuiltPage {
    /// Sets the title to the current text of the region it came from.
    pub fn refresh_title(&mut self) {
        if let Some(region) = self.title_region.and_then(|index| self.regions.get(index)) {
            self.title.clone_from(&region.text);
        }
    }
}

/// Builds the regions of one-based `page_number` from its layout results.
/// `dimensions` is the page size in pixels. Blank regions are omitted,
/// headers, footers and page numbers become `page_furniture` regions, and
/// every box must be valid. Region ids are left empty.
pub fn build_page(
    page: &PageResult,
    page_number: i64,
    dimensions: (f64, f64),
) -> Result<BuiltPage, AssembleError> {
    let mut built = BuiltPage {
        title: String::new(),
        title_region: None,
        regions: Vec::new(),
    };
    for item in format_page(&page.regions) {
        let is_figure =
            item.label == "image" || matches!(item.native_label.as_str(), "chart" | "image");
        let mut content = item.content.unwrap_or_default();
        if content.len() > MAX_REGION_TEXT_BYTES {
            return Err(AssembleError::Rejected(
                "GLM-OCR SDK region exceeds the allowed size",
            ));
        }
        if !is_figure && is_blank_recognition(&content) {
            content.clear();
        }
        let (kind, text) = region_kind(&item.label, &item.native_label, trim(&content));
        let geometry = scale_bbox(item.bbox, dimensions)?;
        if kind != "figure" && text.is_empty() {
            continue;
        }
        if item.native_label == "doc_title" && kind == "heading" && built.title_region.is_none() {
            built.title.clone_from(&text);
            built.title_region = Some(built.regions.len());
        }
        let furniture = furniture_kind(&item.native_label);
        let kind = if furniture.is_some() {
            "page_furniture"
        } else {
            kind
        };
        let mut region = new_region(page_number, kind, text);
        region.furniture = furniture;
        region.bbox = geometry;
        built.regions.push(region);
    }
    Ok(built)
}

/// Settles the document from its built pages, given as one-based page
/// number and regions: lists the pages with no region, detaches table lead
/// rows, splits glued page
/// numbers, assigns region ids `pNNNN-rNNNNNN`, and picks the title (the
/// first page title found, confirmed against `metadata_title`). The
/// document needs at least one region.
pub fn finalise(
    mut pages: Vec<(i64, Vec<Region>)>,
    title: &str,
    metadata_title: &str,
    successful_blank_omitted: usize,
) -> Result<Assembled, AssembleError> {
    let empty_pages: Vec<i64> = pages
        .iter()
        .filter(|(_, regions)| regions.is_empty())
        .map(|(page_number, _)| *page_number)
        .collect();
    detach_table_lead_rows(&mut pages);
    split_trailing_page_numbers(&mut pages);
    let mut regions: Vec<Region> = Vec::new();
    for (page_number, page_regions) in pages {
        for (index, mut region) in page_regions.into_iter().enumerate() {
            region.id = format!("p{page_number:04}-r{:06}", index + 1);
            regions.push(region);
            if regions.len() > MAX_REGIONS {
                return Err(AssembleError::Rejected(
                    "GLM-OCR SDK returned too many regions",
                ));
            }
        }
    }
    if regions.is_empty() {
        return Err(AssembleError::Rejected("GLM-OCR SDK returned no regions"));
    }
    let title = confirmed_title(title, &regions, metadata_title);
    Ok(Assembled {
        title,
        regions,
        empty_pages,
        successful_blank_omitted,
        diagnostics: Vec::new(),
    })
}

/// Assembles the reply for the recognised pages. `page_dimensions` holds
/// each page's size in pixels, and `metadata_title` is the document's own
/// title or empty. Every region that needs recognition must have some and
/// every box must be valid, else the whole document fails; blank regions
/// are omitted, headers, footers and page numbers become `page_furniture`
/// regions, region ids are `pNNNN-rNNNNNN`, and the document needs at least
/// one region.
pub fn assemble(
    pages: &[PageResult],
    page_dimensions: &[(f64, f64)],
    metadata_title: &str,
) -> Result<Assembled, AssembleError> {
    if pages.len() != page_dimensions.len() {
        return Err(recognition_failure(vec![("page_schema", 1)]));
    }
    let successful_blank_omitted = validate(pages)?;
    let mut title = String::new();
    let mut reply_pages: Vec<(i64, Vec<Region>)> = Vec::new();
    for (offset, page) in pages.iter().enumerate() {
        let page_number = offset as i64 + 1;
        let built = build_page(page, page_number, page_dimensions[offset])?;
        if title.is_empty() {
            title = built.title;
        }
        reply_pages.push((page_number, built.regions));
    }
    finalise(
        reply_pages,
        &title,
        metadata_title,
        successful_blank_omitted,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: (f64, f64) = (100.0, 200.0);

    fn region(label: &str, content: &str, bbox: [f64; 4]) -> RegionResult {
        let task = match label {
            "table" => RegionTask::Table,
            "display_formula" => RegionTask::Formula,
            "image" | "chart" => RegionTask::Skip,
            _ => RegionTask::Text,
        };
        RegionResult {
            label: label.to_string(),
            task,
            bbox: Some(bbox),
            content: Some(content.to_string()),
            repetition_stopped: false,
        }
    }

    fn body(label: &str, content: &str) -> RegionResult {
        region(label, content, [100.0, 300.0, 900.0, 400.0])
    }

    fn figure() -> RegionResult {
        RegionResult {
            label: "image".to_string(),
            task: RegionTask::Skip,
            bbox: Some([100.0, 100.0, 900.0, 900.0]),
            content: None,
            repetition_stopped: false,
        }
    }

    fn page(regions: Vec<RegionResult>) -> PageResult {
        PageResult { regions }
    }

    fn assemble_pages(pages: &[PageResult]) -> Result<Assembled, AssembleError> {
        assemble(pages, &vec![PAGE; pages.len()], "")
    }

    fn numbered_pages(bodies: &[&str]) -> Vec<Region> {
        let pages: Vec<PageResult> = bodies
            .iter()
            .map(|text| page(vec![body("text", text)]))
            .collect();
        assemble_pages(&pages).expect("assembled").regions
    }

    fn summary(regions: &[Region]) -> Vec<(i64, &str, Option<FurnitureKind>, &str)> {
        regions
            .iter()
            .map(|r| (r.page, r.kind.as_str(), r.furniture, r.text.as_str()))
            .collect()
    }

    #[test]
    fn margin_blocks_are_page_furniture() {
        let pages = [page(vec![
            region("header", "Running Title", [300.0, 20.0, 700.0, 50.0]),
            body("text", "Body words"),
            region("number", "17", [480.0, 940.0, 520.0, 970.0]),
        ])];
        let regions = assemble_pages(&pages).unwrap().regions;
        assert_eq!(
            summary(&regions),
            vec![
                (
                    1,
                    "page_furniture",
                    Some(FurnitureKind::Header),
                    "Running Title"
                ),
                (1, "paragraph", None, "Body words"),
                (1, "page_furniture", Some(FurnitureKind::PageNumber), "17"),
            ]
        );
        assert_eq!(regions[0].id, "p0001-r000001");
        assert_eq!(regions[2].id, "p0001-r000003");
        assert_eq!(regions[0].bbox, Some([30.0, 4.0, 70.0, 10.0]));
    }

    #[test]
    fn footnote_blocks_are_read_as_content() {
        let pages = [page(vec![region(
            "footnote",
            "1 A note",
            [100.0, 900.0, 900.0, 950.0],
        )])];
        let regions = assemble_pages(&pages).unwrap().regions;
        assert_eq!(summary(&regions), vec![(1, "paragraph", None, "1 A note")]);
    }

    #[test]
    fn glued_trailing_page_numbers_split_into_page_furniture() {
        let regions = numbered_pages(&["alpha beta 11", "gamma delta 12", "epsilon zeta 13"]);
        let number = Some(FurnitureKind::PageNumber);
        assert_eq!(
            summary(&regions),
            vec![
                (1, "paragraph", None, "alpha beta"),
                (1, "page_furniture", number, "11"),
                (2, "paragraph", None, "gamma delta"),
                (2, "page_furniture", number, "12"),
                (3, "paragraph", None, "epsilon zeta"),
                (3, "page_furniture", number, "13"),
            ]
        );
        let mut ids: Vec<&str> = regions.iter().map(|r| r.id.as_str()).collect();
        ids.dedup();
        assert_eq!(ids.len(), regions.len());
    }

    #[test]
    fn trailing_numbers_without_agreeing_sequence_stay_body_text() {
        let regions = numbered_pages(&["in the year 1843", "sold for 12", "a single 7"]);
        assert!(regions.iter().all(|r| r.kind == "paragraph"));
        assert_eq!(regions[0].text, "in the year 1843");
    }

    #[test]
    fn a_lone_trailing_number_is_not_split() {
        let regions = numbered_pages(&["alpha beta 11"]);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].kind, "paragraph");
    }

    #[test]
    fn labelled_page_numbers_vote_for_glued_ones() {
        let pages = [
            page(vec![body("text", "alpha 11")]),
            page(vec![
                body("text", "beta"),
                region("number", "12", [480.0, 940.0, 520.0, 970.0]),
            ]),
        ];
        let regions = assemble_pages(&pages).unwrap().regions;
        let texts: Vec<(i64, &str, &str)> = regions
            .iter()
            .map(|r| (r.page, r.kind.as_str(), r.text.as_str()))
            .collect();
        assert_eq!(
            texts,
            vec![
                (1, "paragraph", "alpha"),
                (1, "page_furniture", "11"),
                (2, "paragraph", "beta"),
                (2, "page_furniture", "12"),
            ]
        );
    }

    #[test]
    fn a_glued_number_follows_the_numbering_beside_it_across_plates() {
        let text_page = |content: &str, number: Option<&str>| {
            let mut regions = vec![body("text", content)];
            if let Some(number) = number {
                regions.push(region("number", number, [480.0, 940.0, 520.0, 970.0]));
            }
            page(regions)
        };
        let pages = [
            text_page("one", Some("10")),
            text_page("two", Some("11")),
            page(vec![figure()]),
            text_page("three", Some("12")),
            page(vec![figure()]),
            text_page("four", Some("13")),
            text_page("five 14", None),
        ];
        let regions = assemble_pages(&pages).unwrap().regions;
        let last: Vec<(&str, &str)> = regions
            .iter()
            .filter(|r| r.page == 7)
            .map(|r| (r.kind.as_str(), r.text.as_str()))
            .collect();
        assert_eq!(last, vec![("paragraph", "five"), ("page_furniture", "14")]);
    }

    #[test]
    fn region_kinds_cover_captions_and_code() {
        assert_eq!(
            region_kind("text", "figure_title", "Figure 1. A map"),
            ("caption", "Figure 1. A map".to_string())
        );
        assert_eq!(
            region_kind("text", "text", "```go\n\npackage main\n\n```"),
            ("code", "package main".to_string())
        );
        assert_eq!(
            region_kind("text", "algorithm", "for x in y"),
            ("code", "for x in y".to_string())
        );
        let mixed = "Run:\n\n```sh\nls\n```";
        assert_eq!(
            region_kind("text", "text", mixed),
            ("paragraph", mixed.to_string())
        );
        assert_eq!(
            region_kind("text", "text", "- item"),
            ("list_item", "item".to_string())
        );
    }

    #[test]
    fn pages_without_regions_are_reported_as_empty() {
        let pages = [
            page(Vec::new()),
            page(vec![region("text", "Body", [10.0, 10.0, 900.0, 90.0])]),
        ];
        let reply = assemble(&pages, &[(612.0, 792.0); 2], "").unwrap();
        assert_eq!(reply.empty_pages, vec![1]);
        assert_eq!(reply.regions.len(), 1);
        assert_eq!(reply.regions[0].id, "p0002-r000001");
    }

    #[test]
    fn a_document_without_regions_fails() {
        let error = assemble_pages(&[page(Vec::new())]).unwrap_err();
        assert_eq!(
            error,
            AssembleError::Rejected("GLM-OCR SDK returned no regions")
        );
    }

    #[test]
    fn invalid_geometry_fails_the_document() {
        let pages = [page(vec![region("text", "text", [5.0, 0.0, 1.0, 2.0])])];
        let error = assemble_pages(&pages).unwrap_err();
        assert_eq!(
            error,
            AssembleError::Rejected("GLM-OCR SDK returned invalid geometry")
        );
        assert!(scale_bbox(Some([0.0, 0.0, 1001.0, 5.0]), PAGE).is_err());
        assert!(scale_bbox(Some([0.0, 0.0, f64::NAN, 5.0]), PAGE).is_err());
        assert_eq!(scale_bbox(None, PAGE), Ok(None));
    }

    #[test]
    fn missing_recognition_fails_without_a_partial_reply() {
        let mut formula = body("display_formula", "x");
        formula.content = None;
        let mut table = body("table", "x");
        table.content = None;
        let pages = [page(vec![
            body("text", "recognized"),
            table,
            formula,
            figure(),
        ])];
        let error = assemble_pages(&pages).unwrap_err();
        assert_eq!(
            error,
            AssembleError::Recognition {
                diagnostics: vec![("missing_recognition", 2)]
            }
        );
        assert_eq!(error.to_string(), "GLM-OCR SDK recognition failed");
    }

    #[test]
    fn a_page_count_mismatch_fails_recognition() {
        let error = assemble(&[page(Vec::new())], &[], "").unwrap_err();
        assert_eq!(
            error,
            AssembleError::Recognition {
                diagnostics: vec![("page_schema", 1)]
            }
        );
    }

    #[test]
    fn successful_blank_regions_are_omitted_and_counted() {
        let pages = [page(vec![
            region("text", "recognized", [0.0, 0.0, 500.0, 500.0]),
            region("table", " \t\n ", [500.0, 0.0, 1000.0, 500.0]),
            RegionResult {
                label: "image".into(),
                task: RegionTask::Skip,
                bbox: Some([0.0, 500.0, 1000.0, 1000.0]),
                content: None,
                repetition_stopped: false,
            },
        ])];
        let reply = assemble_pages(&pages).unwrap();
        assert_eq!(reply.successful_blank_omitted, 1);
        assert_eq!(
            summary(&reply.regions)
                .iter()
                .map(|r| (r.1, r.3))
                .collect::<Vec<_>>(),
            vec![("paragraph", "recognized"), ("figure", "")]
        );
    }

    #[test]
    fn a_table_lead_row_becomes_a_paragraph_before_the_table() {
        let pages = [page(vec![region(
            "table",
            "<table><tr><td colspan=\"2\">Intro</td></tr>\
             <tr><td></td><td></td></tr>\
             <tr><td>a</td><td>b</td></tr></table>",
            [0.0, 0.0, 1000.0, 500.0],
        )])];
        let regions = assemble_pages(&pages).unwrap().regions;
        assert_eq!(
            regions
                .iter()
                .map(|r| (r.id.as_str(), r.kind.as_str()))
                .collect::<Vec<_>>(),
            vec![("p0001-r000001", "paragraph"), ("p0001-r000002", "table")]
        );
        assert_eq!(regions[0].text, "Intro");
        assert_eq!(
            regions[1].text,
            "<table><tr><td>a</td><td>b</td></tr></table>"
        );
        assert_ne!(regions[0].bbox, regions[1].bbox);
        assert_eq!(regions[0].bbox.map(|b| b[3]), regions[1].bbox.map(|b| b[1]));
    }

    #[test]
    fn fence_only_headings_make_no_region() {
        let pages = [page(vec![
            region(
                "doc_title",
                "```markdown\n\n```",
                [100.0, 100.0, 900.0, 200.0],
            ),
            region(
                "paragraph_title",
                "## ```markdown\n\n```",
                [100.0, 220.0, 900.0, 300.0],
            ),
            body("text", "Late."),
        ])];
        let reply = assemble_pages(&pages).unwrap();
        assert_eq!(reply.regions.len(), 1);
        assert_eq!(reply.regions[0].text, "Late.");
        assert_eq!(reply.title, "");
    }

    #[test]
    fn a_fence_with_code_still_becomes_code() {
        let pages = [page(vec![body("text", "```python\nprint(1)\n```")])];
        let regions = assemble_pages(&pages).unwrap().regions;
        assert_eq!(summary(&regions), vec![(1, "code", None, "print(1)")]);
    }

    #[test]
    fn the_first_document_title_is_the_reply_title() {
        let pages = [page(vec![
            region("doc_title", "The Title", [100.0, 10.0, 900.0, 50.0]),
            region("doc_title", "Second", [100.0, 60.0, 900.0, 90.0]),
        ])];
        let reply = assemble_pages(&pages).unwrap();
        assert_eq!(reply.title, "The Title");
        assert_eq!(reply.regions[0].kind, "heading");
        assert_eq!(reply.regions[0].text, "The Title");
    }

    #[test]
    fn metadata_title_is_used_when_an_early_page_prints_it() {
        let mut first = new_region(1, "heading", "Synthetic Field Guide".to_string());
        first.id = "p0001-r000001".to_string();
        let late = new_region(5, "heading", "Table Of Contents".to_string());
        let regions = [first, late];
        assert_eq!(
            confirmed_title("Table Of Contents", &regions, "Synthetic Field Guide"),
            "Synthetic Field Guide"
        );
        assert_eq!(
            confirmed_title("Table Of Contents", &regions, "Microsoft Word - draft.docx"),
            "Table Of Contents"
        );
        assert_eq!(
            confirmed_title("Table Of Contents", &regions, ""),
            "Table Of Contents"
        );
    }

    #[test]
    fn assembly_applies_the_metadata_title() {
        let pages = [page(vec![
            region("doc_title", "Recognised Title", [100.0, 10.0, 900.0, 50.0]),
            body("text", "synthetic field guide"),
        ])];
        let reply = assemble(&pages, &[PAGE], "Synthetic Field Guide").unwrap();
        assert_eq!(reply.title, "Synthetic Field Guide");
    }

    #[test]
    fn blank_detection_sees_through_prefixes_and_empty_fences() {
        assert!(is_blank_recognition(" \n "));
        assert!(is_blank_recognition("## "));
        assert!(is_blank_recognition("```markdown\n\n```"));
        assert!(is_blank_recognition("# ```\n```"));
        assert!(!is_blank_recognition("```\nx\n```"));
        assert!(!is_blank_recognition("text"));
    }

    #[test]
    fn the_reply_converts_into_the_pipeline_reply() {
        let pages = [page(vec![body("text", "Body")])];
        let assembled = assemble_pages(&pages).unwrap();
        let reply = crate::wire::Reply::from(assembled);
        assert_eq!(reply.regions.len(), 1);
    }
}
