//! Recovery of page content the layout and recognition passes miss: text a
//! figure prints (drop caps and captions), bands recognised as blank, the
//! interactive form fields of a PDF, and the document information title.
//!
//! The per-page functions work on the page's assembled regions, in reading
//! order and before ids are assigned, and are applied in this order: figure
//! text, blank band, band above the heading, form fields.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::sync::LazyLock;

use crate::wire::{CHECKED_PLACEHOLDER, REDACTED_PLACEHOLDER, Region, UNCHECKED_PLACEHOLDER};
use hayro::hayro_syntax::Pdf;
use hayro::hayro_syntax::object::{Array, Dict, Name, Number, Object, ObjectIdentifier};
use image::imageops;
use regex::Regex;

use crate::assemble::{AssembleError, is_blank_recognition, scale_bbox};
use crate::client::{OcrClient, RegionError};
use crate::crop::{BBOX_EXTENT, MAX_PIXELS, T_PATCH_SIZE, region_data_url};
use crate::format::{
    RegionResult, RegionTask, clean_content, is_space, longest_leader_run, repetition_collapsed,
    trim,
};
use crate::prompt::{Task, request_body};
use crate::render::RenderedPage;

/// The fewest words a figure's text layer needs to read as a caption.
const MIN_CAPTION_WORDS: usize = 12;
/// The fewest words per line, on average, a caption's text layer has.
const MIN_CAPTION_LINE_WORDS: usize = 5;
/// The leading letters and digits of a caption compared with the page text.
const CAPTION_PROBE_CHARS: usize = 40;
/// Annotation flags that keep a widget from being shown: hidden and no-view.
const ANNOT_FLAGS_NOT_SHOWN: i64 = 0x02 | 0x20;
/// The field flag of a password text field.
const FIELD_FLAG_PASSWORD: i64 = 1 << 13;
/// The field flag of a radio button group.
const FIELD_FLAG_RADIO: i64 = 1 << 15;
/// The field flag of a push button.
const FIELD_FLAG_PUSH_BUTTON: i64 = 1 << 16;
const FORM_CHECKED_GLYPH: char = '\u{2611}';
const FORM_UNCHECKED_GLYPH: char = '\u{2610}';

static FENCE_LINE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\A```[A-Za-z0-9+#._-]*\z").expect("fence line pattern"));

/// The diagnostic counters the recoveries produce.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RecoveryCounts {
    /// Drop-cap figures joined to the paragraph they begin.
    pub text_layer_drop_cap_joined: usize,
    /// Caption regions added from a figure's text layer.
    pub text_layer_caption_recovered: usize,
    /// Paragraphs added from re-read blank bands.
    pub blank_band_recovered: usize,
    /// Paragraphs added from bands re-read below a short recognition.
    pub short_recognition_recovered: usize,
    /// Regions whose runaway recognition a second reading replaced.
    pub repetition_reread: usize,
}

impl RecoveryCounts {
    /// The non-zero counters by category name, in a fixed order.
    pub fn entries(&self) -> Vec<(&'static str, usize)> {
        [
            (
                "text_layer_drop_cap_joined",
                self.text_layer_drop_cap_joined,
            ),
            (
                "text_layer_caption_recovered",
                self.text_layer_caption_recovered,
            ),
            ("blank_band_recovered", self.blank_band_recovered),
            (
                "short_recognition_recovered",
                self.short_recognition_recovered,
            ),
            ("repetition_reread", self.repetition_reread),
        ]
        .into_iter()
        .filter(|(_, count)| *count > 0)
        .collect()
    }

    /// Adds `other`'s counts to these.
    pub fn add(&mut self, other: &Self) {
        self.text_layer_drop_cap_joined += other.text_layer_drop_cap_joined;
        self.text_layer_caption_recovered += other.text_layer_caption_recovered;
        self.blank_band_recovered += other.blank_band_recovered;
        self.short_recognition_recovered += other.short_recognition_recovered;
        self.repetition_reread += other.repetition_reread;
    }
}

/// Why a band could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BandReadError {
    /// The cancellation check held.
    Cancelled,
    /// The band could not be cropped, or the seat gave no text.
    Failed,
}

/// Why a recovery stopped the page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecoverError {
    /// The cancellation check held during a band read.
    Cancelled,
    /// A layout box was invalid.
    Geometry(AssembleError),
}

impl fmt::Display for RecoverError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("the band read was cancelled"),
            Self::Geometry(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RecoverError {}

/// Reads the text of one normalised band (`[x0, y0, x1, y1]` in the 0..1000
/// space) of a page, given the one-based page number.
pub type BandReader<'a> = &'a (dyn Fn(i64, [f64; 4]) -> Result<String, BandReadError> + Sync);

// ---------------------------------------------------------------------
// Text helpers
// ---------------------------------------------------------------------

/// The lines of `text` split at each line boundary: CRLF, CR, LF, vertical tab,
/// form feed, the separators U+001C to U+001E, NEL, and the Unicode line and
/// paragraph separators.
fn split_lines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0usize;
    let mut chars = text.char_indices().peekable();
    while let Some((index, c)) = chars.next() {
        let width = match c {
            '\r' if matches!(chars.peek(), Some((_, '\n'))) => {
                chars.next();
                2
            }
            '\n' | '\r' | '\u{b}' | '\u{c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}'
            | '\u{2028}' | '\u{2029}' => c.len_utf8(),
            _ => continue,
        };
        lines.push(&text[start..index]);
        start = index + width;
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// The words of `text` split at whitespace.
fn words(text: &str) -> impl Iterator<Item = &str> {
    text.split(is_space).filter(|word| !word.is_empty())
}

/// The non-empty trimmed lines of a recognition.
fn text_lines(content: &str) -> Vec<String> {
    split_lines(content)
        .into_iter()
        .map(trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

/// Reduces text to its lowercase letters and digits.
fn comparable(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

/// The lowercase text with whitespace runs collapsed to one space.
fn folded(text: &str) -> String {
    words(&text.to_lowercase()).collect::<Vec<_>>().join(" ")
}

fn paragraph_region(page: i64, text: String, bbox: Option<[f64; 4]>) -> Region {
    Region {
        id: String::new(),
        page,
        kind: "paragraph".to_string(),
        text,
        bbox,
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

fn overlaps(first: &[f64; 4], second: &[f64; 4]) -> bool {
    first[0] < second[2] && second[0] < first[2] && first[1] < second[3] && second[1] < first[3]
}

// ---------------------------------------------------------------------
// Figure text
// ---------------------------------------------------------------------

/// The letter a figure prints when its text layer is one capital letter, as
/// a decorative initial does.
pub fn drop_cap_letter(layer: &str) -> Option<char> {
    let mut chars = layer.chars();
    let letter = chars.next()?;
    (chars.next().is_none() && letter.is_alphabetic() && letter.is_uppercase()).then_some(letter)
}

/// A figure's text-layer text when it reads as running prose, as a printed
/// caption or credit does: at least 12 words averaging 5 or more per line.
/// Diagram labels and axis text give `None`.
pub fn figure_caption_text(layer: &str) -> Option<String> {
    let lines = split_lines(layer);
    let word_count = words(layer).count();
    if word_count >= MIN_CAPTION_WORDS && word_count >= MIN_CAPTION_LINE_WORDS * lines.len() {
        Some(lines.into_iter().map(trim).collect::<Vec<_>>().join(" "))
    } else {
        None
    }
}

/// The page's regions with text that only a figure carries restored. A
/// drop-cap figure joins the paragraph that follows it when that paragraph
/// starts in lower case (the figure is replaced by the joined paragraph), and
/// a figure whose text layer reads as prose that no region on the page
/// already holds gains a caption region after it. `figure_layers` maps the
/// index of a figure region in `page_regions` to its text-layer text.
pub fn recover_figure_text(
    page_regions: Vec<Region>,
    figure_layers: &BTreeMap<usize, String>,
    counts: &mut RecoveryCounts,
) -> Vec<Region> {
    let page_text = comparable(
        &page_regions
            .iter()
            .map(|region| region.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
    );
    let mut recovered: Vec<Region> = Vec::with_capacity(page_regions.len());
    let mut skip_next = false;
    for (index, region) in page_regions.iter().enumerate() {
        if skip_next {
            skip_next = false;
            continue;
        }
        let Some(layer) = figure_layers.get(&index) else {
            recovered.push(region.clone());
            continue;
        };
        let following = page_regions.get(index + 1);
        if let (Some(letter), Some(following)) = (drop_cap_letter(layer), following)
            && following.kind == "paragraph"
            && following
                .text
                .chars()
                .next()
                .is_some_and(char::is_lowercase)
        {
            let mut joined = following.clone();
            joined.text = format!("{letter}{}", following.text);
            for link in &mut joined.links {
                link.start += 1;
                link.end += 1;
            }
            recovered.push(joined);
            skip_next = true;
            counts.text_layer_drop_cap_joined += 1;
            continue;
        }
        recovered.push(region.clone());
        if let Some(caption) = figure_caption_text(layer) {
            let probe: String = comparable(&caption)
                .chars()
                .take(CAPTION_PROBE_CHARS)
                .collect();
            if !page_text.contains(&probe) {
                let mut added = region.clone();
                added.kind = "caption".to_string();
                added.text = caption;
                recovered.push(added);
                counts.text_layer_caption_recovered += 1;
            }
        }
    }
    recovered
}

// ---------------------------------------------------------------------
// Blank bands
// ---------------------------------------------------------------------

fn is_figure_label(label: &str) -> bool {
    matches!(label, "chart" | "image")
}

/// The normalised boxes of the page's text regions that were detected and
/// recognised as blank.
pub fn blank_text_items(raw_page: &[RegionResult]) -> Vec<[f64; 4]> {
    raw_page
        .iter()
        .filter_map(|item| {
            let content = item.content.as_deref()?;
            let bbox = item.bbox?;
            if is_figure_label(&item.label)
                || !is_blank_recognition(content)
                || bbox[0] >= bbox[2]
                || bbox[1] >= bbox[3]
            {
                return None;
            }
            Some(bbox)
        })
        .collect()
}

/// The normalised box spanning the page's blank-recognised text regions and
/// the gap around them, bounded above and below by the nearest regions that
/// hold text, or `None` when the page has none or a figure region touches
/// them.
pub fn blank_band(raw_page: &[RegionResult]) -> Option<[f64; 4]> {
    let blanks = blank_text_items(raw_page);
    if blanks.is_empty() {
        return None;
    }
    let left = blanks.iter().map(|b| b[0]).fold(f64::INFINITY, f64::min);
    let top = blanks.iter().map(|b| b[1]).fold(f64::INFINITY, f64::min);
    let right = blanks
        .iter()
        .map(|b| b[2])
        .fold(f64::NEG_INFINITY, f64::max);
    let bottom = blanks
        .iter()
        .map(|b| b[3])
        .fold(f64::NEG_INFINITY, f64::max);
    let mut upper = 0.0f64;
    let mut lower = BBOX_EXTENT;
    for item in raw_page {
        let Some(bbox) = item.bbox else { continue };
        let is_figure = is_figure_label(&item.label);
        if is_figure && bbox[0] < right && left < bbox[2] && bbox[1] < bottom && top < bbox[3] {
            return None;
        }
        if !is_figure && item.content.as_deref().is_none_or(is_blank_recognition) {
            continue;
        }
        if bbox[1] < top {
            upper = upper.max(bbox[3].min(top));
        } else if bbox[1] >= bottom {
            lower = lower.min(bbox[1]);
        } else if bbox[3] > bottom {
            lower = lower.min(bottom);
        }
    }
    if lower <= upper {
        return None;
    }
    Some([0.0, upper, BBOX_EXTENT, lower])
}

/// The recognised lines of a band that hold a letter or digit, are not a
/// code fence marker and do not repeat text already on the page; `existing`
/// holds the folded texts seen and gains each returned line.
pub fn band_lines(text: &str, existing: &mut HashSet<String>) -> Vec<String> {
    let mut lines = Vec::new();
    for line in text_lines(text) {
        let key = folded(&line);
        if FENCE_LINE.is_match(&line) {
            continue;
        }
        if line.chars().any(char::is_alphanumeric) && !existing.contains(&key) {
            existing.insert(key);
            lines.push(line);
        }
    }
    lines
}

/// The text `band_reader` reads in `band`; `None` when the read failed.
fn read_band_text(
    page_number: i64,
    band: [f64; 4],
    band_reader: BandReader<'_>,
) -> Result<Option<String>, RecoverError> {
    match band_reader(page_number, band) {
        Ok(text) => Ok(Some(text)),
        Err(BandReadError::Cancelled) => Err(RecoverError::Cancelled),
        Err(BandReadError::Failed) => Ok(None),
    }
}

/// The regions with `lines` inserted as paragraphs, in ordered disjoint slots
/// inside `geometry`, before the first region that starts below its top.
fn insert_band_paragraphs(
    page_regions: Vec<Region>,
    page_number: i64,
    geometry: [f64; 4],
    lines: Vec<String>,
) -> Vec<Region> {
    let first_top = geometry[1];
    let position = page_regions
        .iter()
        .position(|region| region.bbox.is_some_and(|b| b[1] > first_top))
        .unwrap_or(page_regions.len());
    let mut result = Vec::with_capacity(page_regions.len() + lines.len());
    let mut rest = page_regions;
    let tail = rest.split_off(position);
    result.extend(rest);
    let slots = lines.len() as f64;
    let slot_height = (geometry[3] - geometry[1]) / slots;
    result.extend(lines.into_iter().enumerate().map(|(index, line)| {
        let top = geometry[1] + slot_height * index as f64;
        let bottom = if index + 1 == slots as usize {
            geometry[3]
        } else {
            geometry[1] + slot_height * (index + 1) as f64
        };
        paragraph_region(
            page_number,
            line,
            Some([geometry[0], top, geometry[2], bottom]),
        )
    }));
    result.extend(tail);
    result
}

fn add_band_paragraphs(
    page_regions: Vec<Region>,
    page_number: i64,
    band: [f64; 4],
    geometry: [f64; 4],
    band_reader: BandReader<'_>,
    counts: &mut RecoveryCounts,
) -> Result<Vec<Region>, RecoverError> {
    let mut existing: HashSet<String> = page_regions
        .iter()
        .flat_map(|region| {
            std::iter::once(folded(&region.text))
                .chain(text_lines(&region.text).iter().map(|line| folded(line)))
                .collect::<Vec<_>>()
        })
        .collect();
    let Some(text) = read_band_text(page_number, band, band_reader)? else {
        return Ok(page_regions);
    };
    let lines = band_lines(&text, &mut existing);
    if lines.is_empty() {
        return Ok(page_regions);
    }
    counts.blank_band_recovered += lines.len();
    Ok(insert_band_paragraphs(
        page_regions,
        page_number,
        geometry,
        lines,
    ))
}

/// The page's regions with the lines `band_reader` reads in the page's blank
/// band added as paragraphs before the first region that starts below the
/// first blank box. A failed band read leaves the regions as they are.
pub fn recover_blank_band(
    page_regions: Vec<Region>,
    raw_page: &[RegionResult],
    page_number: i64,
    dimensions: (f64, f64),
    band_reader: BandReader<'_>,
    counts: &mut RecoveryCounts,
) -> Result<Vec<Region>, RecoverError> {
    let Some(band) = blank_band(raw_page) else {
        return Ok(page_regions);
    };
    let blanks = blank_text_items(raw_page);
    let union = [
        blanks.iter().map(|b| b[0]).fold(f64::INFINITY, f64::min),
        blanks.iter().map(|b| b[1]).fold(f64::INFINITY, f64::min),
        blanks
            .iter()
            .map(|b| b[2])
            .fold(f64::NEG_INFINITY, f64::max),
        blanks
            .iter()
            .map(|b| b[3])
            .fold(f64::NEG_INFINITY, f64::max),
    ];
    let Some(geometry) = scale_bbox(Some(union), dimensions).map_err(RecoverError::Geometry)?
    else {
        return Ok(page_regions);
    };
    add_band_paragraphs(
        page_regions,
        page_number,
        band,
        geometry,
        band_reader,
        counts,
    )
}

/// The fewest lines of text a short recognition's box holds beyond the lines
/// the recognition covers to count as cut short. The covered height is
/// rounded up to whole lines, which can overstate coverage by up to one
/// line, so two lines leave one line of margin.
const SHORT_RECOGNITION_MISSING_LINES: f64 = 2.0;

/// The ratio of a character's average advance to the line pitch of running
/// text. Measured on a scanned newspaper (median 0.411, quartiles
/// 0.394 to 0.446 over 154 paragraph regions, recognised text against
/// tesseract line counts).
const ADVANCE_PER_LINE_PITCH: f64 = 0.41;

/// The ratio of a character's average advance to a leader dot's advance
/// (Helvetica: about 0.5 em over 0.278 em), so a text line can hold up to
/// this many times its character capacity in leader dots.
const LEADER_DOTS_PER_CHARACTER: f64 = 1.8;

/// The most lines of text one strip of a region read in strips holds.
const STRIP_LINES: f64 = 10.0;

/// The fraction of the longer union of character four-grams two texts must
/// share to count as the same text re-recognised.
const SAME_TEXT_OVERLAP: f64 = 0.6;

/// The character count of a text: its trimmed non-empty lines, with their
/// inner spaces.
fn text_chars(content: &str) -> usize {
    text_lines(content)
        .iter()
        .map(|line| line.chars().count())
        .sum()
}

/// A region's width and height in page pixels.
fn region_pixels(bbox: [f64; 4], dimensions: (f64, f64)) -> (f64, f64) {
    (
        (bbox[2] - bbox[0]) * dimensions.0 / BBOX_EXTENT,
        (bbox[3] - bbox[1]) * dimensions.1 / BBOX_EXTENT,
    )
}

/// Whether a raw region is a text recognition with a usable box.
fn is_text_region(item: &RegionResult) -> bool {
    item.task == RegionTask::Text
        && !is_figure_label(&item.label)
        && item
            .content
            .as_deref()
            .is_some_and(|c| !is_blank_recognition(c))
        && item.bbox.is_some_and(|b| b[0] < b[2] && b[1] < b[3])
}

/// The typography of a page's running text, taken from its other text
/// regions.
#[derive(Clone, Copy, Debug)]
struct PageType {
    /// The line pitch in page pixels.
    pitch: f64,
}

impl PageType {
    /// The characters one line of a region `width` pixels wide holds.
    fn line_capacity(&self, width: f64) -> f64 {
        (width / (ADVANCE_PER_LINE_PITCH * self.pitch)).max(1.0)
    }

    /// The height in pixels a recognition of `content` covers in a region
    /// `width` pixels wide: each recognised line takes as many text lines as
    /// its characters fill, at least one.
    fn covered_height(&self, content: &str, width: f64) -> f64 {
        let capacity = self.line_capacity(width);
        let lines: f64 = text_lines(content)
            .iter()
            .map(|line| (line.chars().count() as f64 / capacity).ceil().max(1.0))
            .sum();
        lines * self.pitch
    }
}

/// The fraction of the other regions' area, taken from the densest text, at
/// which a page's text density is read. A box can hold no more text than
/// fills it but may hold less, so partly filled boxes only overstate the
/// area per character; reading a quarter of the way up from the densest
/// text keeps those boxes and any one over-recognised box from setting it.
const DENSITY_QUANTILE: f64 = 0.25;

/// The typography of the page from the regions other than the one at
/// `skip`: the area-weighted `DENSITY_QUANTILE` quantile of the page area
/// per character of their recognitions gives the line pitch, since a
/// character cell is `ADVANCE_PER_LINE_PITCH` pitches wide and one pitch
/// tall.
fn page_type(raw_page: &[RegionResult], skip: usize, dimensions: (f64, f64)) -> Option<PageType> {
    let mut cells: Vec<(f64, f64)> = raw_page
        .iter()
        .enumerate()
        .filter(|(index, item)| *index != skip && is_text_region(item) && !item.repetition_stopped)
        .filter_map(|(_, item)| {
            let bbox = item.bbox?;
            let chars = text_chars(item.content.as_deref()?);
            let (width, height) = region_pixels(bbox, dimensions);
            let area = width * height;
            (chars > 0 && area > 0.0).then_some((area / chars as f64, area))
        })
        .collect();
    if cells.is_empty() {
        return None;
    }
    cells.sort_by(|a, b| a.0.total_cmp(&b.0));
    let threshold: f64 = cells.iter().map(|cell| cell.1).sum::<f64>() * DENSITY_QUANTILE;
    let mut seen = 0.0;
    let per_char = cells
        .iter()
        .find(|cell| {
            seen += cell.1;
            seen >= threshold
        })
        .map(|cell| cell.0)?;
    let pitch = (per_char / ADVANCE_PER_LINE_PITCH).sqrt();
    (pitch.is_finite() && pitch > 0.0).then_some(PageType { pitch })
}

/// Whether the recognition of the region at `index` was stopped or cut for
/// repetition: the seat reported its repetition stop, or the content holds
/// repetition cleaning would cut.
fn repetition_stopped_or_cut(raw_page: &[RegionResult], index: usize) -> bool {
    let item = &raw_page[index];
    is_text_region(item)
        && (item.repetition_stopped || item.content.as_deref().is_some_and(repetition_collapsed))
}

/// Whether the recognition of the region at `index` shows a runaway: it was
/// stopped or cut for repetition, or it holds a leader run no line of the
/// region could hold.
fn repetition_runaway(raw_page: &[RegionResult], index: usize, dimensions: (f64, f64)) -> bool {
    if repetition_stopped_or_cut(raw_page, index) {
        return true;
    }
    let item = &raw_page[index];
    if !is_text_region(item) {
        return false;
    }
    let (Some(content), Some(bbox)) = (item.content.as_deref(), item.bbox) else {
        return false;
    };
    page_type(raw_page, index, dimensions).is_some_and(|typography| {
        let (width, _) = region_pixels(bbox, dimensions);
        longest_leader_run(content) as f64
            > typography.line_capacity(width) * LEADER_DOTS_PER_CHARACTER
    })
}

/// The median height, in the normalised space, of one recognised line on the
/// page: each text region's box height over its newline line count, leaving
/// out the region at `skip`.
fn page_line_height(raw_page: &[RegionResult], skip: usize) -> Option<f64> {
    let mut heights: Vec<f64> = raw_page
        .iter()
        .enumerate()
        .filter(|(index, item)| *index != skip && is_text_region(item))
        .filter_map(|(_, item)| {
            let bbox = item.bbox?;
            let lines = text_lines(item.content.as_deref()?).len();
            (lines > 0).then(|| (bbox[3] - bbox[1]) / lines as f64)
        })
        .collect();
    if heights.is_empty() {
        return None;
    }
    heights.sort_by(f64::total_cmp);
    let middle = heights.len() / 2;
    Some(if heights.len() % 2 == 1 {
        heights[middle]
    } else {
        (heights[middle - 1] + heights[middle]) / 2.0
    })
}

/// The normalised boxes below the recognised lines of the page's text regions
/// whose recognition covers far less of the box than it holds. A region is
/// cut short when either test fires. By newline lines, the box is at least
/// two page line heights taller than its recognised lines, each taken as one
/// median line height. By coverage, the recognition's text lines, estimated
/// from the characters each recognised line holds against what one line of
/// the region holds, leave the box at least two lines taller than they fill,
/// so a paragraph returned as one line counts as the lines it fills. A
/// region yields one band running to the bottom of the region, from the
/// higher of the tests' tops: the end of the recognised lines for the
/// newline test, one line above the estimated end for the coverage test. A
/// region whose recognised text is taller than its box has its band top
/// below the box and yields no band.
pub fn short_recognition_bands(raw_page: &[RegionResult], dimensions: (f64, f64)) -> Vec<[f64; 4]> {
    let mut bands = Vec::new();
    for (index, item) in raw_page.iter().enumerate() {
        if !is_text_region(item) {
            continue;
        }
        let (Some(content), Some(bbox)) = (item.content.as_deref(), item.bbox) else {
            continue;
        };
        let by_lines = page_line_height(raw_page, index)
            .filter(|line_height| *line_height > 0.0)
            .map(|line_height| {
                (
                    bbox[1] + text_lines(content).len() as f64 * line_height,
                    line_height,
                )
            })
            .filter(|(top, line_height)| {
                bbox[3] - top >= SHORT_RECOGNITION_MISSING_LINES * line_height
            })
            .map(|(top, _)| top);
        let by_coverage = page_type(raw_page, index, dimensions).and_then(|typography| {
            let (width, height) = region_pixels(bbox, dimensions);
            let covered = typography.covered_height(content, width);
            if height - covered < SHORT_RECOGNITION_MISSING_LINES * typography.pitch {
                return None;
            }
            let pitch = typography.pitch * BBOX_EXTENT / dimensions.1;
            let covered = covered * BBOX_EXTENT / dimensions.1;
            Some((bbox[1] + covered - pitch).max(bbox[1]))
        });
        let top = match (by_lines, by_coverage) {
            (Some(a), Some(b)) => a.min(b),
            (Some(top), None) | (None, Some(top)) => top,
            (None, None) => continue,
        };
        bands.push([bbox[0], top, bbox[2], bbox[3]]);
    }
    bands
}

/// The index of the page region a short recognition's band belongs to: the
/// region whose box shares the band's left, right and bottom edges (within a
/// small fraction of the band's smaller extent) and starts at or above the
/// band's top. The closest such region by top edge wins.
fn short_band_region(regions: &[Region], geometry: [f64; 4]) -> Option<usize> {
    let tolerance =
        SHORT_BAND_EDGE_TOLERANCE * (geometry[2] - geometry[0]).min(geometry[3] - geometry[1]);
    regions
        .iter()
        .enumerate()
        .filter_map(|(index, region)| Some((index, region.bbox?)))
        .filter(|(_, b)| {
            (b[0] - geometry[0]).abs() <= tolerance
                && (b[2] - geometry[2]).abs() <= tolerance
                && (b[3] - geometry[3]).abs() <= tolerance
                && b[1] <= geometry[1] + tolerance
        })
        .max_by(|(_, first), (_, second)| first[1].total_cmp(&second[1]))
        .map(|(index, _)| index)
}

/// The fraction of a band's smaller extent by which a region's edges may
/// differ from the band's and still match it.
const SHORT_BAND_EDGE_TOLERANCE: f64 = 0.05;

/// The text of one held unit prepared for comparison: its letters and
/// digits, and their four-grams.
struct Held {
    key: String,
    grams: HashSet<[char; 4]>,
}

impl Held {
    fn new(text: &str) -> Self {
        let key = comparable(text);
        let chars: Vec<char> = key.chars().collect();
        let grams = chars.windows(4).map(|w| [w[0], w[1], w[2], w[3]]).collect();
        Self { key, grams }
    }

    /// Whether `other` repeats this text: its letters and digits run
    /// together inside this text's, or the two share at least
    /// `SAME_TEXT_OVERLAP` of the four-grams either holds.
    fn repeats(&self, other: &Self) -> bool {
        self.key.contains(&other.key) || self.overlap(other) >= SAME_TEXT_OVERLAP
    }

    /// The share of the four-grams either text holds that both hold.
    fn overlap(&self, other: &Self) -> f64 {
        let shared = self.grams.intersection(&other.grams).count();
        let union = self.grams.len() + other.grams.len() - shared;
        if union == 0 {
            0.0
        } else {
            shared as f64 / union as f64
        }
    }
}

/// Where a held unit lives: a region's whole text, one line of a region's
/// text, or a line this band read has queued to append.
enum Target {
    Region(usize),
    Line(usize, String),
    Queued(usize),
}

/// A held unit with the place it lives.
struct Unit {
    target: Target,
    held: Held,
    spaced: String,
}

impl Unit {
    fn new(target: Target, text: &str) -> Self {
        Self {
            target,
            held: Held::new(text),
            spaced: words(text).collect::<Vec<_>>().join(" "),
        }
    }
}

/// The held units of the regions and the queued lines: each region's whole
/// text and each of its lines, then each queued line.
fn held_units(regions: &[Region], queued: &[String]) -> Vec<Unit> {
    let mut units = Vec::new();
    for (index, region) in regions.iter().enumerate() {
        units.push(Unit::new(Target::Region(index), &region.text));
        for line in text_lines(&region.text) {
            units.push(Unit::new(Target::Line(index, line.clone()), &line));
        }
    }
    for (index, line) in queued.iter().enumerate() {
        units.push(Unit::new(Target::Queued(index), line));
    }
    units
}

/// The most token pairs an alignment compares; a larger pair of texts is
/// joined end to end instead.
const MERGE_CELL_LIMIT: usize = 4_000_000;

/// A whitespace-separated token with the whitespace that followed it and
/// whether it lies inside a redaction span.
struct Token<'a> {
    text: &'a str,
    gap: &'a str,
    redacted: bool,
}

/// The tokens of `text`, each with its trailing whitespace; tokens from a
/// token holding U+E000 through the next holding U+E001 are redacted.
fn tokens(text: &str) -> Vec<Token<'_>> {
    let mut result = Vec::new();
    let mut inside = false;
    let mut rest = text.trim_start_matches(is_space);
    while !rest.is_empty() {
        let end = rest.find(is_space).unwrap_or(rest.len());
        let word = &rest[..end];
        let after = &rest[end..];
        let gap_end = after.find(|c| !is_space(c)).unwrap_or(after.len());
        let opens = word.contains('\u{e000}');
        let closes = word.contains('\u{e001}');
        let redacted = inside || opens || closes;
        if opens && !closes {
            inside = true;
        } else if closes {
            inside = false;
        }
        result.push(Token {
            text: word,
            gap: &after[..gap_end],
            redacted,
        });
        rest = &after[gap_end..];
    }
    result
}

/// One step of the alignment of held tokens against read tokens.
enum Step {
    Both(usize),
    Held(usize),
    Read(usize),
}

/// The alignment of two token lists as a longest common subsequence, in
/// order; ties prefer matching the later tokens. A redacted held token
/// absorbs any number of consecutive read tokens, each scoring as a match and
/// returned as a `Step::Read` beside the redacted token's `Step::Held`.
fn align(held: &[Token<'_>], read: &[Token<'_>]) -> Vec<Step> {
    let width = read.len() + 1;
    let mut table = vec![0u32; (held.len() + 1) * width];
    let mut absorb = vec![0u32; (held.len() + 1) * width];
    for i in 1..=held.len() {
        for j in 1..=read.len() {
            let mut best = if held[i - 1].text == read[j - 1].text {
                table[(i - 1) * width + j - 1] + 1
            } else {
                table[(i - 1) * width + j].max(table[i * width + j - 1])
            };
            if held[i - 1].redacted {
                let run = table[(i - 1) * width + j - 1].max(absorb[i * width + j - 1]) + 1;
                absorb[i * width + j] = run;
                best = best.max(run);
            }
            table[i * width + j] = best;
        }
    }
    let (mut i, mut j) = (held.len(), read.len());
    let mut steps = Vec::new();
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && held[i - 1].redacted && absorb[i * width + j] == table[i * width + j] {
            steps.push(Step::Held(i - 1));
            loop {
                steps.push(Step::Read(j - 1));
                let continues = j > 1 && absorb[i * width + j - 1] > table[(i - 1) * width + j - 1];
                j -= 1;
                if !continues {
                    break;
                }
            }
            i -= 1;
        } else if i > 0 && j > 0 && held[i - 1].text == read[j - 1].text {
            steps.push(Step::Both(i - 1));
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || table[i * width + j - 1] >= table[(i - 1) * width + j]) {
            steps.push(Step::Read(j - 1));
            j -= 1;
        } else {
            steps.push(Step::Held(i - 1));
            i -= 1;
        }
    }
    steps.reverse();
    steps
}

/// Whether a read run replaces a held run: the read run is non-empty and
/// either the held run is empty or the held run's letters and digits
/// (ignoring case, spacing and punctuation) lie inside the longer read run's,
/// or both hold the same letters and digits and the read run has more
/// punctuation characters.
fn extends(held_run: &[&Token<'_>], read_run: &[&Token<'_>]) -> bool {
    if read_run.is_empty() {
        return false;
    }
    if held_run.is_empty() {
        return true;
    }
    let join =
        |run: &[&Token<'_>]| comparable(&run.iter().map(|t| t.text).collect::<Vec<_>>().join(" "));
    let (held, read) = (join(held_run), join(read_run));
    if read.len() != held.len() {
        return read.len() > held.len() && read.contains(&held);
    }
    let marks = |run: &[&Token<'_>]| {
        run.iter()
            .flat_map(|token| token.text.chars())
            .filter(|c| !c.is_alphanumeric() && !c.is_whitespace())
            .count()
    };
    read == held && marks(read_run) > marks(held_run)
}

/// Pushes the read tokens in order, the last carrying `tail_gap` when it
/// holds a line break.
fn push_read(push: &mut impl FnMut(&str, &str), read_run: &[&Token<'_>], tail_gap: &str) {
    for (position, token) in read_run.iter().enumerate() {
        let gap = if position + 1 == read_run.len() && tail_gap.contains('\n') {
            tail_gap
        } else {
            " "
        };
        push(token.text, gap);
    }
}

/// The held text merged with a re-read of the same text: tokens both hold
/// appear once, tokens only one holds appear in order, and where the two
/// differ over the same span the held tokens are kept, unless the read
/// tokens extend them (the held letters and digits, ignoring case, spacing
/// and punctuation, lie inside the longer read ones, or are equal to them
/// with fewer punctuation characters), in which case the read tokens are used. A held redaction span in a differing span is replaced by
/// that span's read tokens; the span's unredacted held tokens are kept. The
/// held text's whitespace is kept.
fn merge_text(held: &str, read: &str) -> String {
    let held_tokens = tokens(held);
    let read_tokens = tokens(read);
    if held_tokens.len().saturating_mul(read_tokens.len()) > MERGE_CELL_LIMIT {
        return format!("{held}\n\n{read}");
    }
    let steps = align(&held_tokens, &read_tokens);
    let mut out = String::new();
    let mut push = |text: &str, gap: &str| {
        if !out.is_empty() && !out.ends_with(is_space) {
            out.push(' ');
        }
        out.push_str(text);
        out.push_str(gap);
    };
    let mut index = 0;
    while index < steps.len() {
        if let Step::Both(h) = steps[index] {
            push(held_tokens[h].text, held_tokens[h].gap);
            index += 1;
            continue;
        }
        let start = index;
        while index < steps.len() && !matches!(steps[index], Step::Both(_)) {
            index += 1;
        }
        let mut held_run = Vec::new();
        let mut read_run = Vec::new();
        for step in &steps[start..index] {
            match step {
                Step::Held(h) => held_run.push(&held_tokens[*h]),
                Step::Read(r) => read_run.push(&read_tokens[*r]),
                Step::Both(_) => {}
            }
        }
        if held_run.iter().any(|token| token.redacted) && !read_run.is_empty() {
            let mut inserted = false;
            for token in &held_run {
                if !token.redacted {
                    push(token.text, token.gap);
                } else if !inserted {
                    inserted = true;
                    push_read(&mut push, &read_run, token.gap);
                }
            }
        } else if extends(&held_run, &read_run) {
            let tail_gap = held_run.last().map_or(" ", |token| token.gap);
            push_read(&mut push, &read_run, tail_gap);
        } else {
            for token in &held_run {
                push(token.text, token.gap);
            }
        }
    }
    out.trim_end_matches(is_space).to_string()
}

/// The lines of `text` that repeat nothing the regions hold, in order. A
/// line that repeats held text, with its letters and digits running together
/// inside a held region or line (ignoring case, spacing and punctuation) or
/// sharing most of its four-grams with one, is merged into that text with
/// `merge_text` instead of returned; the second value counts the merges that
/// changed the regions.
fn unheld_lines(regions: &mut [Region], text: &str) -> (Vec<String>, usize) {
    let mut lines: Vec<String> = Vec::new();
    let mut merged_count = 0usize;
    for line in text_lines(text) {
        if FENCE_LINE.is_match(&line) || !line.chars().any(char::is_alphanumeric) {
            continue;
        }
        let units = held_units(regions, &lines);
        let spaced = words(&line).collect::<Vec<_>>().join(" ");
        if units.iter().any(|unit| unit.spaced.contains(&spaced)) {
            continue;
        }
        let candidate = Held::new(&line);
        let Some(unit) = units
            .iter()
            .filter(|unit| unit.held.repeats(&candidate))
            .max_by(|first, second| {
                first
                    .held
                    .overlap(&candidate)
                    .total_cmp(&second.held.overlap(&candidate))
            })
        else {
            lines.push(line);
            continue;
        };
        match &unit.target {
            Target::Region(index) => {
                let merged = merge_text(&regions[*index].text, &line);
                if merged != regions[*index].text {
                    merged_count += 1;
                    regions[*index].text = merged;
                }
            }
            Target::Line(index, held_line) => {
                let merged = merge_text(held_line, &line);
                if merged != *held_line {
                    merged_count += 1;
                    regions[*index].text = regions[*index].text.replacen(held_line, &merged, 1);
                }
            }
            Target::Queued(index) => {
                lines[*index] = merge_text(&lines[*index], &line);
            }
        }
    }
    (lines, merged_count)
}

/// The page's regions with the lines `band_reader` reads below each short
/// recognition appended to that region's own text after its existing text,
/// formatted as the region's own recognition is; a band no region matches
/// gets its lines as paragraphs before the first region that starts below
/// the band's top. A read line that repeats text already on the page, with
/// its letters and digits running together inside a held region or line
/// (ignoring case, spacing and punctuation) or sharing most of its
/// four-grams with one, is dropped. A failed band read leaves the regions
/// as they are.
pub fn recover_short_recognition(
    page_regions: Vec<Region>,
    raw_page: &[RegionResult],
    page_number: i64,
    dimensions: (f64, f64),
    band_reader: BandReader<'_>,
    counts: &mut RecoveryCounts,
) -> Result<Vec<Region>, RecoverError> {
    let mut regions = page_regions;
    let bands = short_recognition_bands(raw_page, dimensions);
    let geometries: Vec<_> = bands
        .iter()
        .map(|band| scale_bbox(Some(*band), dimensions))
        .collect();
    let texts: Vec<Option<Result<Option<String>, RecoverError>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = bands
            .iter()
            .zip(&geometries)
            .map(|(band, geometry)| {
                matches!(geometry, Ok(Some(_)))
                    .then(|| scope.spawn(move || read_band_text(page_number, *band, band_reader)))
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle.map(|handle| match handle.join() {
                    Ok(result) => result,
                    Err(panic) => std::panic::resume_unwind(panic),
                })
            })
            .collect()
    });
    for (geometry, text) in geometries.into_iter().zip(texts) {
        let Some(geometry) = geometry.map_err(RecoverError::Geometry)? else {
            continue;
        };
        let Some(text) = text.expect("a band with geometry was read")? else {
            continue;
        };
        let (lines, merged) = unheld_lines(&mut regions, &text);
        counts.short_recognition_recovered += merged;
        if lines.is_empty() {
            continue;
        }
        counts.short_recognition_recovered += lines.len();
        if let Some(index) = short_band_region(&regions, geometry) {
            let region = &mut regions[index];
            let joined = crate::format::format_content(&lines.join("\n"), "text", "text");
            if !region.text.is_empty() {
                region.text.push_str("\n\n");
            }
            region.text.push_str(&joined);
        } else {
            regions = insert_band_paragraphs(regions, page_number, geometry, lines);
        }
    }
    Ok(regions)
}

/// How a region whose recognition ran away is read again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reread {
    /// The region's crop enlarged to twice its width and height.
    Upscaled,
    /// The region cut into this many horizontal strips of equal height, each
    /// read on its own and the texts joined in order.
    Strips(usize),
}

/// Reads the text of one normalised region of a page again the way `Reread`
/// names, given the one-based page number.
pub type RereadReader<'a> =
    &'a (dyn Fn(i64, [f64; 4], Reread) -> Result<String, BandReadError> + Sync);

/// The way to read a region again: doubled in size when the doubled crop
/// still fits the seat's pixel budget, otherwise in strips of about
/// `STRIP_LINES` lines each (a region of unknown line count is read doubled).
pub fn reread_plan(bbox: [f64; 4], dimensions: (f64, f64), pitch: Option<f64>) -> Reread {
    let (width, height) = region_pixels(bbox, dimensions);
    let doubled = f64::from(T_PATCH_SIZE) * 4.0 * width * height;
    if doubled <= MAX_PIXELS as f64 {
        return Reread::Upscaled;
    }
    match pitch {
        Some(pitch) if pitch > 0.0 => {
            Reread::Strips(((height / pitch / STRIP_LINES).ceil() as usize).max(2))
        }
        _ => Reread::Upscaled,
    }
}

/// The page's regions with each region whose recognition ran away (the
/// seat reported its repetition stop, repetition cleaning would cut its
/// text, or it holds a leader run longer than a line of the region could
/// hold) replaced by `reader`'s second reading when that reading holds more
/// letters and digits than the region's text. Otherwise the second
/// reading's lines that repeat nothing held are appended to the region and
/// those that repeat held text are merged into it as the band path does. A
/// failed reading leaves the region as it is.
pub fn recover_repetition(
    page_regions: Vec<Region>,
    raw_page: &[RegionResult],
    page_number: i64,
    dimensions: (f64, f64),
    reader: RereadReader<'_>,
    counts: &mut RecoveryCounts,
) -> Result<Vec<Region>, RecoverError> {
    let mut regions = page_regions;
    for index in 0..raw_page.len() {
        if !repetition_runaway(raw_page, index, dimensions) {
            continue;
        }
        let Some(bbox) = raw_page[index].bbox else {
            continue;
        };
        let Some(geometry) = scale_bbox(Some(bbox), dimensions).map_err(RecoverError::Geometry)?
        else {
            continue;
        };
        let Some(target) = short_band_region(&regions, geometry) else {
            continue;
        };
        if matches!(
            regions[target].kind.as_str(),
            "heading" | "table" | "formula" | "code" | "figure"
        ) {
            continue;
        }
        let pitch = page_type(raw_page, index, dimensions).map(|typography| typography.pitch);
        let mode = reread_plan(bbox, dimensions, pitch);
        let text = match reader(page_number, bbox, mode) {
            Ok(text) => text,
            Err(BandReadError::Cancelled) => return Err(RecoverError::Cancelled),
            Err(BandReadError::Failed) => continue,
        };
        let cleaned = clean_content(&text);
        if comparable(&cleaned).chars().count() > comparable(&regions[target].text).chars().count()
        {
            regions[target].text = crate::format::format_content(&text, "text", "text");
            counts.repetition_reread += 1;
            continue;
        }
        let (lines, merged) = unheld_lines(&mut regions, &cleaned);
        if !lines.is_empty() {
            let joined = crate::format::format_content(&lines.join("\n"), "text", "text");
            if !regions[target].text.is_empty() {
                regions[target].text.push_str("\n\n");
            }
            regions[target].text.push_str(&joined);
        }
        if merged > 0 || !lines.is_empty() {
            counts.repetition_reread += 1;
        }
    }
    Ok(regions)
}

/// The normalised full-width band directly above the page's first heading,
/// below the nearest detected box above it, when no box covers it and it is
/// taller than the heading itself.
pub fn uncovered_band_above_heading(raw_page: &[RegionResult]) -> Option<[f64; 4]> {
    let mut boxes: Vec<[f64; 4]> = Vec::new();
    let mut heading: Option<usize> = None;
    for item in raw_page {
        let Some(bbox) = item.bbox else { continue };
        if bbox[0] >= bbox[2] || bbox[1] >= bbox[3] {
            continue;
        }
        boxes.push(bbox);
        if matches!(item.label.as_str(), "doc_title" | "paragraph_title")
            && item
                .content
                .as_deref()
                .is_some_and(|content| !is_blank_recognition(content))
            && heading.is_none_or(|held| bbox[1] < boxes[held][1])
        {
            heading = Some(boxes.len() - 1);
        }
    }
    let heading_index = heading?;
    let heading_box = boxes[heading_index];
    let top = heading_box[1];
    let mut upper = 0.0f64;
    for (index, bbox) in boxes.iter().enumerate() {
        if index == heading_index || bbox[1] >= top {
            continue;
        }
        upper = upper.max(bbox[3].min(top));
    }
    if top - upper <= heading_box[3] - heading_box[1] {
        return None;
    }
    Some([0.0, upper, BBOX_EXTENT, top])
}

/// The page's regions with the lines `band_reader` reads in the tall band no
/// detected box covers above the page's first heading added as paragraphs
/// before that heading.
pub fn recover_band_above_heading(
    page_regions: Vec<Region>,
    raw_page: &[RegionResult],
    page_number: i64,
    dimensions: (f64, f64),
    band_reader: BandReader<'_>,
    counts: &mut RecoveryCounts,
) -> Result<Vec<Region>, RecoverError> {
    let Some(band) = uncovered_band_above_heading(raw_page) else {
        return Ok(page_regions);
    };
    let Some(geometry) = scale_bbox(Some(band), dimensions).map_err(RecoverError::Geometry)? else {
        return Ok(page_regions);
    };
    add_band_paragraphs(
        page_regions,
        page_number,
        band,
        geometry,
        band_reader,
        counts,
    )
}

/// The pixel rectangle `[x0, y0, x1, y1]` of a normalised band of a page
/// raster `width` by `height`: near edges truncated, far edges rounded up to
/// whole pixels; `None` for a band with no pixels.
fn band_pixels(band: [f64; 4], width: u32, height: u32) -> Option<[u32; 4]> {
    let edge = |normalised: f64, extent: u32, up: bool| -> Option<u32> {
        let pixel = normalised * f64::from(extent) / BBOX_EXTENT;
        if !pixel.is_finite() {
            return None;
        }
        let pixel = if up { pixel.ceil() } else { pixel.trunc() };
        Some(pixel.clamp(0.0, f64::from(extent)) as u32)
    };
    let (x0, y0, x1, y1) = (
        edge(band[0], width, false)?,
        edge(band[1], height, false)?,
        edge(band[2], width, true)?,
        edge(band[3], height, true)?,
    );
    (x1 > x0 && y1 > y0).then_some([x0, y0, x1, y1])
}

/// The text the seat reads in an image under the plain text-recognition
/// prompt, sent through `client` (which applies its retry policy).
fn read_image(
    client: &OcrClient,
    model: &str,
    image: &image::RgbImage,
    cancelled: &dyn Fn() -> bool,
) -> Result<String, BandReadError> {
    let url = region_data_url(image).map_err(|_| BandReadError::Failed)?;
    match client.recognize(&request_body(model, Task::Text, &url), cancelled) {
        Ok(text) => Ok(text),
        Err(RegionError::Cancelled) => Err(BandReadError::Cancelled),
        Err(RegionError::Failed(_)) => Err(BandReadError::Failed),
    }
}

/// The text the seat reads in a normalised band of a rendered page under the
/// plain text-recognition prompt. The band is cropped from the page raster
/// with its near edges truncated and its far edges rounded up to whole
/// pixels, and sent through `client` (which applies its retry policy).
pub fn read_band(
    client: &OcrClient,
    model: &str,
    page: &RenderedPage,
    band: [f64; 4],
    cancelled: &dyn Fn() -> bool,
) -> Result<String, BandReadError> {
    let (width, height) = page.image.dimensions();
    let Some([x0, y0, x1, y1]) = band_pixels(band, width, height) else {
        return Err(BandReadError::Failed);
    };
    let crop = imageops::crop_imm(&page.image, x0, y0, x1 - x0, y1 - y0).to_image();
    read_image(client, model, &crop, cancelled)
}

/// The row in `lo..=hi` of `image` between columns `x0..x1` with the fewest
/// dark pixels; among equals the one closest to `near`.
fn quietest_row(image: &image::RgbImage, x0: u32, x1: u32, lo: u32, hi: u32, near: u32) -> u32 {
    let ink = |row: u32| {
        (x0..x1)
            .filter(|column| {
                let [r, g, b] = image.get_pixel(*column, row).0;
                (u32::from(r) * 299 + u32::from(g) * 587 + u32::from(b) * 114) / 1000 < 128
            })
            .count()
    };
    (lo..=hi)
        .min_by_key(|row| (ink(*row), row.abs_diff(near)))
        .unwrap_or(near)
}

/// The row ranges of a crop `height` pixels tall cut into `count` strips of
/// about equal height, each cut moved to the quietest row within a quarter
/// strip of the even cut so it falls between text lines.
fn strip_rows(image: &image::RgbImage, count: usize) -> Vec<(u32, u32)> {
    let (width, height) = image.dimensions();
    let count = count.max(1) as u32;
    let window = height / count / 4;
    let mut cuts = vec![0u32];
    for index in 1..count {
        let even = height * index / count;
        let lo = even.saturating_sub(window).max(cuts[cuts.len() - 1] + 1);
        let hi = (even + window).min(height - 1);
        cuts.push(if lo <= hi {
            quietest_row(image, 0, width, lo, hi, even)
        } else {
            even
        });
    }
    cuts.push(height);
    cuts.windows(2)
        .filter(|pair| pair[1] > pair[0])
        .map(|pair| (pair[0], pair[1]))
        .collect()
}

/// The text the seat reads in a normalised region of a rendered page when
/// read again the way `mode` names: the crop at twice its size, or its
/// strips read in turn and joined with line breaks.
pub fn reread_region(
    client: &OcrClient,
    model: &str,
    page: &RenderedPage,
    band: [f64; 4],
    mode: Reread,
    cancelled: &dyn Fn() -> bool,
) -> Result<String, BandReadError> {
    let (width, height) = page.image.dimensions();
    let Some([x0, y0, x1, y1]) = band_pixels(band, width, height) else {
        return Err(BandReadError::Failed);
    };
    let crop = imageops::crop_imm(&page.image, x0, y0, x1 - x0, y1 - y0).to_image();
    match mode {
        Reread::Upscaled => {
            let enlarged = imageops::resize(
                &crop,
                crop.width() * 2,
                crop.height() * 2,
                imageops::FilterType::CatmullRom,
            );
            read_image(client, model, &enlarged, cancelled)
        }
        Reread::Strips(count) => {
            let mut texts = Vec::new();
            for (top, bottom) in strip_rows(&crop, count) {
                let strip =
                    imageops::crop_imm(&crop, 0, top, crop.width(), bottom - top).to_image();
                texts.push(read_image(client, model, &strip, cancelled)?);
            }
            Ok(texts.join("\n"))
        }
    }
}

// ---------------------------------------------------------------------
// Form fields
// ---------------------------------------------------------------------

/// Whether something the page draws or marks for redaction hides a box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cover {
    /// Nothing hides the box.
    Clear,
    /// A filled box, a redaction mark or the page edge hides the box.
    Covered,
    /// It could not be determined; treated as covered.
    Unknown,
}

/// Tells whether a box is hidden. The arguments are the zero-based page index
/// and the box `[x0, y0, x1, y1]` in the page's points as rendered (rotation
/// and crop box applied, origin at the top left).
pub type CoverCheck<'a> = &'a dyn Fn(usize, [f64; 4]) -> Cover;

/// One visible form field.
#[derive(Clone, Debug, PartialEq)]
pub struct FormField {
    /// The region text: `Label: value`, or a ballot box glyph, the label and
    /// the state placeholder for a check box or radio button.
    pub text: String,
    /// The text alternative, equal to `text`.
    pub alt_text: String,
    /// The entered text; empty for check boxes, radio buttons and withheld
    /// fields.
    pub value: String,
    /// The field box in page pixels.
    pub bbox: [f64; 4],
}

/// The form fields of a PDF by one-based page number and what could not be
/// read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FormScan {
    pub fields: BTreeMap<i64, Vec<FormField>>,
    /// Fields whose cover state was `Unknown` and were withheld.
    pub cover_unknown: usize,
    /// Whether the document or its catalog could not be read, so no field
    /// was listed.
    pub unreadable: bool,
}

impl FormScan {
    /// The non-zero counters by category name.
    pub fn entries(&self) -> Vec<(&'static str, usize)> {
        let mut entries = Vec::new();
        if self.cover_unknown > 0 {
            entries.push(("form_field_cover_unknown", self.cover_unknown));
        }
        if self.unreadable {
            entries.push(("form_fields_unreadable", 1));
        }
        entries
    }
}

const PDF_DOC_HIGH: [char; 33] = [
    '\u{2022}', '\u{2020}', '\u{2021}', '\u{2026}', '\u{2014}', '\u{2013}', '\u{192}', '\u{2044}',
    '\u{2039}', '\u{203a}', '\u{2212}', '\u{2030}', '\u{201e}', '\u{201c}', '\u{201d}', '\u{2018}',
    '\u{2019}', '\u{201a}', '\u{2122}', '\u{fb01}', '\u{fb02}', '\u{141}', '\u{152}', '\u{160}',
    '\u{178}', '\u{17d}', '\u{131}', '\u{142}', '\u{153}', '\u{161}', '\u{17e}', '\u{fffd}',
    '\u{20ac}',
];
const PDF_DOC_LOW: [char; 8] = [
    '\u{2d8}', '\u{2c7}', '\u{2c6}', '\u{2d9}', '\u{2dd}', '\u{2db}', '\u{2da}', '\u{2dc}',
];

/// The text a PDF text string holds: UTF-16BE or UTF-8 behind its byte order
/// mark, else PDFDocEncoding.
fn decode_text_string(bytes: &[u8]) -> String {
    if let Some(body) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        let units: Vec<u16> = body
            .chunks_exact(2)
            .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
            .collect();
        return String::from_utf16_lossy(&units);
    }
    if let Some(body) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8_lossy(body).into_owned();
    }
    bytes
        .iter()
        .map(|&byte| match byte {
            0x18..=0x1F => PDF_DOC_LOW[usize::from(byte - 0x18)],
            0x80..=0xA0 => PDF_DOC_HIGH[usize::from(byte - 0x80)],
            _ => char::from(byte),
        })
        .collect()
}

fn text_of(object: Object<'_>) -> Option<String> {
    match object {
        Object::String(string) => Some(decode_text_string(string.as_bytes())),
        Object::Name(name) => Some(name.as_str().to_string()),
        _ => None,
    }
}

/// `widget` and its `/Parent` ancestors, nearest first.
fn field_chain<'a>(widget: &Dict<'a>, limit: usize) -> Vec<Dict<'a>> {
    let mut chain = vec![widget.clone()];
    let mut seen = HashSet::new();
    let mut current = widget.clone();
    while chain.len() <= limit {
        let Some(parent) = current.get::<Dict>("Parent") else {
            break;
        };
        if let Some(id) = parent.obj_id()
            && !seen.insert(id)
        {
            break;
        }
        chain.push(parent.clone());
        current = parent;
    }
    chain
}

fn inherited_name(chain: &[Dict<'_>], key: &str) -> Option<String> {
    chain
        .iter()
        .find_map(|dict| dict.get::<Name>(key))
        .map(|name| name.as_str().to_string())
}

fn inherited_text(chain: &[Dict<'_>], key: &str) -> Option<String> {
    chain
        .iter()
        .find_map(|dict| {
            dict.get::<Object>(key)
                .filter(|o| !matches!(o, Object::Null(_)))
        })
        .and_then(|object| match object {
            Object::Array(array) => array.iter::<Object>().find_map(text_of),
            other => text_of(other),
        })
}

fn inherited_flags(chain: &[Dict<'_>]) -> i64 {
    chain
        .iter()
        .find_map(|dict| dict.get::<Number>("Ff"))
        .map_or(0, |number| number.as_i64())
}

/// The fully qualified field name: the `/T` of each ancestor from the top
/// down, joined with dots.
fn qualified_name(chain: &[Dict<'_>]) -> String {
    chain
        .iter()
        .rev()
        .filter_map(|dict| dict.get::<Object>("T").and_then(text_of))
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(".")
}

/// The widget's index among its field's `/Kids`, matched by the kid's
/// reference to the widget's object, else by an identical `/Rect`; zero when
/// it is the field or no kid matches.
fn control_index(widget: &Dict<'_>, chain: &[Dict<'_>]) -> usize {
    let Some(parent) = chain.get(1) else {
        return 0;
    };
    let Some(kids) = parent.get::<Array>("Kids") else {
        return 0;
    };
    if let Some(widget_id) = widget.obj_id()
        && let Some(index) = kids.raw_iter().position(|kid| {
            kid.as_obj_ref()
                .is_some_and(|reference| ObjectIdentifier::from(reference) == widget_id)
        })
    {
        return index;
    }
    let Some(rect) = widget_rect(widget) else {
        return 0;
    };
    kids.iter::<Dict>()
        .position(|kid| widget_rect(&kid) == Some(rect))
        .unwrap_or(0)
}

/// Whether a check box or radio button widget shows its on state: its
/// appearance state names an on state its normal appearance defines.
fn is_checked(widget: &Dict<'_>) -> bool {
    let state = widget
        .get::<Name>("AS")
        .map(|name| name.as_str().to_string())
        .unwrap_or_default();
    if state.is_empty() || state == "Off" {
        return false;
    }
    let on_state = widget
        .get::<Dict>("AP")
        .and_then(|appearance| appearance.get::<Dict>("N"))
        .and_then(|normal| {
            let mut keys: Vec<String> = normal
                .keys()
                .map(|key| key.as_str().to_string())
                .filter(|key| key != "Off")
                .collect();
            keys.sort();
            keys.into_iter().next()
        });
    on_state.is_none_or(|on| on == state)
}

fn widget_rect(widget: &Dict<'_>) -> Option<[f64; 4]> {
    let values: Vec<f64> = widget
        .get::<Array>("Rect")?
        .iter::<Number>()
        .map(|number| number.as_f64())
        .collect();
    let [a, b, c, d] = values.get(..4)? else {
        return None;
    };
    Some([a.min(*c), b.min(*d), a.max(*c), b.max(*d)])
}

/// The visible interactive form fields of a PDF by one-based page number.
/// `page_dimensions` holds each page's size in pixels and `cover` tells
/// whether a field's box is hidden.
///
/// A text, combo or list field with a value reads `Label: value`, the label
/// being the field's alternate name, else its qualified name. A check box
/// reads the ballot box glyph (checked or unchecked), its label and the
/// state placeholder; a radio button reads the same with its `/Opt` display
/// value in place of the label when the group has options. A text field whose
/// box is covered reads the redaction placeholder in place of its value, and a
/// check box or radio button there is left out; a box whose cover state is
/// unknown is treated as covered and counted in `cover_unknown`. Hidden
/// fields, password fields, push buttons, signatures and empty text fields
/// are left out. A source that is not a PDF or has no `/AcroForm` yields no
/// fields; one that cannot be read sets `unreadable`.
pub fn pdf_form_fields(
    source: &[u8],
    page_dimensions: &[(f64, f64)],
    cover: CoverCheck<'_>,
) -> FormScan {
    let mut scan = FormScan::default();
    if !source.starts_with(b"%PDF-") {
        return scan;
    }
    let Ok(pdf) = Pdf::new(source.to_vec()) else {
        scan.unreadable = true;
        return scan;
    };
    let xref = pdf.xref();
    let Some(catalog) = xref.get::<Dict>(xref.root_id()) else {
        scan.unreadable = true;
        return scan;
    };
    if !catalog.contains_key("AcroForm") {
        return scan;
    }
    let limit = pdf.len() + 1;
    for (index, page) in pdf.pages().iter().enumerate().take(page_dimensions.len()) {
        let (render_width, render_height) = page.render_dimensions();
        if render_width <= 0.0 || render_height <= 0.0 {
            continue;
        }
        let scale_x = page_dimensions[index].0 / f64::from(render_width);
        let scale_y = page_dimensions[index].1 / f64::from(render_height);
        let [a, b, c, d, e, f] = page.initial_transform(true).as_coeffs();
        let mut found: Vec<FormField> = Vec::new();
        let annots = page
            .raw()
            .get::<Array>("Annots")
            .map(|array| array.iter::<Dict>().collect::<Vec<_>>())
            .unwrap_or_default();
        for widget in &annots {
            if widget
                .get::<Name>("Subtype")
                .map(|n| n.as_str().to_string())
                .as_deref()
                != Some("Widget")
            {
                continue;
            }
            let flags = widget.get::<Number>("F").map_or(0, |n| n.as_i64());
            if flags & ANNOT_FLAGS_NOT_SHOWN != 0 {
                continue;
            }
            let Some(rect) = widget_rect(widget) else {
                continue;
            };
            let chain = field_chain(widget, limit);
            let field_flags = inherited_flags(&chain);
            if field_flags & FIELD_FLAG_PASSWORD != 0 {
                continue;
            }
            let field_type = inherited_name(&chain, "FT").unwrap_or_default();
            let is_box = field_type == "Btn" && field_flags & FIELD_FLAG_PUSH_BUTTON == 0;
            let is_radio = is_box && field_flags & FIELD_FLAG_RADIO != 0;
            let is_value = matches!(field_type.as_str(), "Tx" | "Ch");
            if !is_box && !is_value {
                continue;
            }
            let mut label = inherited_text(&chain, "TU")
                .map(|text| trim(&text).to_string())
                .filter(|text| !text.is_empty())
                .unwrap_or_else(|| trim(&qualified_name(&chain)).to_string());
            let corners = [
                (rect[0], rect[1]),
                (rect[2], rect[1]),
                (rect[0], rect[3]),
                (rect[2], rect[3]),
            ]
            .map(|(x, y)| (a * x + c * y + e, b * x + d * y + f));
            let visual = [
                corners.iter().map(|p| p.0).fold(f64::INFINITY, f64::min),
                corners.iter().map(|p| p.1).fold(f64::INFINITY, f64::min),
                corners
                    .iter()
                    .map(|p| p.0)
                    .fold(f64::NEG_INFINITY, f64::max),
                corners
                    .iter()
                    .map(|p| p.1)
                    .fold(f64::NEG_INFINITY, f64::max),
            ];
            let withheld = match cover(index, visual) {
                Cover::Clear => false,
                Cover::Covered => true,
                Cover::Unknown => {
                    scan.cover_unknown += 1;
                    true
                }
            };
            let bbox = [
                visual[0] * scale_x,
                visual[1] * scale_y,
                visual[2] * scale_x,
                visual[3] * scale_y,
            ];
            let (text, value) = if is_box {
                if is_radio
                    && let Some(options) = chain.iter().find_map(|dict| dict.get::<Array>("Opt"))
                    && options.iter::<Object>().next().is_some()
                {
                    let export = options
                        .iter::<Object>()
                        .nth(control_index(widget, &chain))
                        .and_then(|option| match option {
                            Object::String(string) => Some(decode_text_string(string.as_bytes())),
                            _ => None,
                        })
                        .map(|text| trim(&text).to_string())
                        .filter(|text| !text.is_empty())
                        .unwrap_or_else(|| "Yes".to_string());
                    label = export;
                }
                if label.is_empty() || withheld {
                    continue;
                }
                let checked = is_checked(widget);
                let (glyph, state) = if checked {
                    (FORM_CHECKED_GLYPH, CHECKED_PLACEHOLDER)
                } else {
                    (FORM_UNCHECKED_GLYPH, UNCHECKED_PLACEHOLDER)
                };
                (format!("{glyph} {label}: {state}"), String::new())
            } else {
                let value = inherited_text(&chain, "V")
                    .map(|text| trim(&text).to_string())
                    .unwrap_or_default();
                if value.is_empty() {
                    continue;
                }
                let shown = if withheld {
                    REDACTED_PLACEHOLDER.to_string()
                } else {
                    value.clone()
                };
                let text = if label.is_empty() {
                    shown
                } else {
                    format!("{label}: {shown}")
                };
                (text, if withheld { String::new() } else { value })
            };
            found.push(FormField {
                alt_text: text.clone(),
                text,
                value,
                bbox,
            });
        }
        found.sort_by(|x, y| {
            x.bbox[1]
                .total_cmp(&y.bbox[1])
                .then(x.bbox[0].total_cmp(&y.bbox[0]))
        });
        if !found.is_empty() {
            scan.fields.insert(index as i64 + 1, found);
        }
    }
    scan
}

/// The page's regions with each form field added as a paragraph before the
/// first region that starts lower on the page. A text field whose value an
/// overlapping region already reads is not added again.
pub fn with_form_fields(
    page_regions: Vec<Region>,
    fields: &[FormField],
    page_number: i64,
) -> Vec<Region> {
    let mut merged = page_regions;
    for field in fields {
        let wanted = comparable(&field.value);
        if !wanted.is_empty()
            && merged.iter().any(|region| {
                region.bbox.is_some_and(|bbox| overlaps(&bbox, &field.bbox))
                    && comparable(&region.text).contains(&wanted)
            })
        {
            continue;
        }
        let mut region = paragraph_region(page_number, field.text.clone(), Some(field.bbox));
        region.alt_text = Some(field.alt_text.clone());
        let position = merged
            .iter()
            .position(|existing| existing.bbox.is_some_and(|b| b[1] > field.bbox[1]))
            .unwrap_or(merged.len());
        merged.insert(position, region);
    }
    merged
}

// ---------------------------------------------------------------------
// Metadata title
// ---------------------------------------------------------------------

/// The title a PDF's document information names with whitespace collapsed,
/// or empty when the source is not a PDF or names none.
pub fn pdf_metadata_title(source: &[u8]) -> String {
    if !source.starts_with(b"%PDF-") {
        return String::new();
    }
    let Ok(pdf) = Pdf::new(source.to_vec()) else {
        return String::new();
    };
    match &pdf.metadata().title {
        Some(bytes) => words(&decode_text_string(bytes))
            .collect::<Vec<_>>()
            .join(" "),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assemble::assemble;
    use crate::format::{PageResult, RegionTask};
    use std::sync::Mutex;

    fn item(label: &str, content: &str, bbox: [f64; 4]) -> RegionResult {
        RegionResult {
            label: label.to_string(),
            task: if is_figure_label(label) {
                RegionTask::Skip
            } else {
                RegionTask::Text
            },
            bbox: Some(bbox),
            content: Some(content.to_string()),
            repetition_stopped: false,
        }
    }

    fn figure(bbox: [f64; 4]) -> RegionResult {
        RegionResult {
            content: None,
            repetition_stopped: false,
            ..item("image", "", bbox)
        }
    }

    fn assembled(raw: &[RegionResult], dimensions: (f64, f64)) -> Vec<Region> {
        let page = PageResult {
            regions: raw.to_vec(),
        };
        assemble(&[page], &[dimensions], "")
            .expect("assembles")
            .regions
    }

    fn texts(regions: &[Region]) -> Vec<&str> {
        regions.iter().map(|r| r.text.as_str()).collect()
    }

    fn kinds_and_texts(regions: &[Region]) -> Vec<(&str, &str)> {
        regions
            .iter()
            .map(|r| (r.kind.as_str(), r.text.as_str()))
            .collect()
    }

    // ---- figure text ----

    #[test]
    fn drop_cap_figure_joins_the_paragraph_it_begins() {
        let raw = [
            figure([100.0, 60.0, 300.0, 100.0]),
            item(
                "text",
                "achine learning begins this synthetic paragraph.",
                [100.0, 300.0, 900.0, 400.0],
            ),
        ];
        let regions = assembled(&raw, (612.0, 792.0));
        let layers = BTreeMap::from([(0usize, "M".to_string())]);
        let mut counts = RecoveryCounts::default();
        let recovered = recover_figure_text(regions, &layers, &mut counts);
        assert_eq!(counts.entries(), vec![("text_layer_drop_cap_joined", 1)]);
        assert_eq!(
            kinds_and_texts(&recovered),
            vec![(
                "paragraph",
                "Machine learning begins this synthetic paragraph."
            )]
        );
    }

    #[test]
    fn figure_prose_becomes_a_caption_and_labels_do_not() {
        let credit = [
            "A synthetic instrument photographed in a synthetic laboratory",
            "for this test, with a credit line printed over the image.",
        ];
        let raw = [
            figure([0.0, 60.0, 1000.0, 160.0]),
            figure([0.0, 230.0, 1000.0, 400.0]),
        ];
        let layers = BTreeMap::from([
            (0usize, credit.join("\n")),
            (1usize, "x1\nx2\nM1 M2\nsigma".to_string()),
        ]);
        let mut counts = RecoveryCounts::default();
        let recovered = recover_figure_text(assembled(&raw, (612.0, 792.0)), &layers, &mut counts);
        assert_eq!(counts.entries(), vec![("text_layer_caption_recovered", 1)]);
        assert_eq!(
            kinds_and_texts(&recovered),
            vec![
                ("figure", ""),
                ("caption", &credit.join(" ")),
                ("figure", "")
            ]
        );
    }

    #[test]
    fn a_caption_the_page_already_holds_is_not_added_again() {
        let credit = "A synthetic instrument photographed in a synthetic laboratory for this test, with a credit line.";
        let raw = [
            figure([0.0, 60.0, 1000.0, 160.0]),
            item("text", credit, [0.0, 300.0, 1000.0, 400.0]),
        ];
        let layers = BTreeMap::from([(0usize, credit.to_string())]);
        let mut counts = RecoveryCounts::default();
        let recovered = recover_figure_text(assembled(&raw, (612.0, 792.0)), &layers, &mut counts);
        assert_eq!(counts, RecoveryCounts::default());
        assert_eq!(recovered.len(), 2);
    }

    // ---- blank bands ----

    const PAGE: (f64, f64) = (1000.0, 2000.0);

    fn band_page() -> Vec<RegionResult> {
        vec![
            item("number", "7", [300.0, 50.0, 340.0, 70.0]),
            item("header", "", [150.0, 200.0, 450.0, 370.0]),
            item("header", "", [600.0, 200.0, 900.0, 370.0]),
            item("doc_title", "TITLE LINE", [350.0, 390.0, 700.0, 420.0]),
            item("text", "Body text.", [100.0, 440.0, 900.0, 540.0]),
        ]
    }

    fn chapter_page(extra: Vec<RegionResult>) -> Vec<RegionResult> {
        let mut page = vec![
            item("number", "7", [300.0, 50.0, 340.0, 70.0]),
            item(
                "paragraph_title",
                "CHAPTER TITLE",
                [300.0, 500.0, 700.0, 540.0],
            ),
            item("text", "Body text.", [100.0, 560.0, 900.0, 700.0]),
        ];
        page.extend(extra);
        page
    }

    type Calls = Mutex<Vec<(i64, [f64; 4])>>;

    /// Recovers both band kinds on a page, as the scheduler does, with a
    /// reader answering `answer` and recording its calls.
    fn recover(
        raw: &[RegionResult],
        answer: Result<&str, BandReadError>,
    ) -> (Vec<Region>, RecoveryCounts, Calls) {
        let calls: Calls = Mutex::new(Vec::new());
        let reader = |page: i64, band: [f64; 4]| {
            calls.lock().unwrap().push((page, band));
            answer.clone().map(str::to_string)
        };
        let mut counts = RecoveryCounts::default();
        let regions = recover_blank_band(assembled(raw, PAGE), raw, 1, PAGE, &reader, &mut counts)
            .expect("blank band");
        let regions =
            recover_band_above_heading(regions, raw, 1, PAGE, &reader, &mut counts).expect("above");
        (regions, counts, calls)
    }

    #[test]
    fn reads_the_band_between_blank_artwork_boxes() {
        let (regions, counts, calls) = recover(&band_page(), Ok("LABEL ONE\n\n"));
        assert_eq!(
            *calls.lock().unwrap(),
            vec![(1, [0.0, 70.0, 1000.0, 390.0])]
        );
        assert_eq!(
            texts(&regions),
            vec!["7", "LABEL ONE", "TITLE LINE", "Body text."]
        );
        assert!(counts.entries().contains(&("blank_band_recovered", 1)));
    }

    fn short_page(box_bottom: f64) -> Vec<RegionResult> {
        vec![
            item(
                "text",
                "Line one.\nLine two.\nLine three.",
                [100.0, 100.0, 900.0, 160.0],
            ),
            item(
                "text",
                "Alpha . . . 1\nBeta . . . 2",
                [100.0, 200.0, 900.0, box_bottom],
            ),
        ]
    }

    fn recover_short(
        raw: &[RegionResult],
        answer: Result<&str, BandReadError>,
    ) -> (Vec<Region>, RecoveryCounts, Calls) {
        let calls: Calls = Mutex::new(Vec::new());
        let reader = |page: i64, band: [f64; 4]| {
            calls.lock().unwrap().push((page, band));
            answer.clone().map(str::to_string)
        };
        let mut counts = RecoveryCounts::default();
        let regions =
            recover_short_recognition(assembled(raw, PAGE), raw, 1, PAGE, &reader, &mut counts)
                .expect("short recognition");
        (regions, counts, calls)
    }

    #[test]
    fn rereads_below_a_recognition_far_shorter_than_its_box() {
        let raw = short_page(700.0);
        let (regions, counts, calls) = recover_short(&raw, Ok("Gamma . . . 3\nDelta . . . 4"));
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        let (page, band) = calls[0];
        assert_eq!(page, 1);
        assert_eq!([band[0], band[2], band[3]], [100.0, 900.0, 700.0]);
        assert!(
            (band[1] - 240.0).abs() < 0.1,
            "the band starts at the higher of the newline end 240 and the coverage top 244.9: {band:?}"
        );
        assert_eq!(
            texts(&regions),
            vec![
                "Line one.\n\nLine two.\n\nLine three.",
                "Alpha . . . 1\n\nBeta . . . 2\n\nGamma . . . 3\n\nDelta . . . 4",
            ]
        );
        assert_eq!(counts.entries(), vec![("short_recognition_recovered", 2)]);
    }

    /// A paragraph of 120 characters returned as a single line.
    fn dense_body(index: i32) -> String {
        format!(
            "Body {index} lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore"
        )
    }

    fn multi_short_page() -> Vec<RegionResult> {
        let mut raw: Vec<RegionResult> = (0..5)
            .map(|index| {
                let top = 100.0 + 70.0 * f64::from(index);
                item("text", &dense_body(index), [100.0, top, 900.0, top + 60.0])
            })
            .collect();
        for index in 0..3 {
            let top = 460.0 + 180.0 * f64::from(index);
            raw.push(item(
                "text",
                &format!("Short {index} a.\nShort {index} b."),
                [100.0, top, 900.0, top + 160.0],
            ));
        }
        raw
    }

    fn multi_short_answer(band: [f64; 4]) -> String {
        let index = ((band[1] - 460.0) / 180.0).round() as i64;
        let distinct = [
            "Quiet orchard lantern glowing",
            "Violet harbour tide rising",
            "Granite bridge over river",
        ][usize::try_from(index).unwrap()];
        format!("Shared line\n{distinct}")
    }

    #[test]
    fn short_recognition_bands_are_read_concurrently() {
        let raw = multi_short_page();
        let wanted = short_recognition_bands(&raw, PAGE).len();
        assert_eq!(wanted, 3);
        let state = (Mutex::new(0usize), std::sync::Condvar::new());
        let overlapped = Mutex::new(Vec::new());
        let reader = |_: i64, band: [f64; 4]| {
            let (count, wake) = &state;
            let mut in_flight = count.lock().unwrap();
            *in_flight += 1;
            wake.notify_all();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            while *in_flight < wanted {
                let Some(left) = deadline.checked_duration_since(std::time::Instant::now()) else {
                    break;
                };
                in_flight = wake.wait_timeout(in_flight, left).unwrap().0;
            }
            overlapped.lock().unwrap().push(*in_flight >= wanted);
            Ok(multi_short_answer(band))
        };
        let mut counts = RecoveryCounts::default();
        recover_short_recognition(assembled(&raw, PAGE), &raw, 1, PAGE, &reader, &mut counts)
            .expect("short recognition");
        assert_eq!(*overlapped.lock().unwrap(), vec![true; wanted]);
    }

    #[test]
    fn concurrent_short_reads_merge_in_band_order() {
        let raw = multi_short_page();
        let reader = |_: i64, band: [f64; 4]| Ok(multi_short_answer(band));
        let mut counts = RecoveryCounts::default();
        let regions =
            recover_short_recognition(assembled(&raw, PAGE), &raw, 1, PAGE, &reader, &mut counts)
                .expect("short recognition");
        let held = texts(&regions);
        assert_eq!(
            held[5..],
            [
                "Short 0 a.\n\nShort 0 b.\n\nShared line\n\nQuiet orchard lantern glowing",
                "Short 1 a.\n\nShort 1 b.\n\nViolet harbour tide rising",
                "Short 2 a.\n\nShort 2 b.\n\nGranite bridge over river",
            ]
        );
        assert_eq!(counts.entries(), vec![("short_recognition_recovered", 4)]);
    }

    fn run_together_page() -> Vec<RegionResult> {
        vec![
            item(
                "text",
                "Line one.\nLine two.\nLine three.",
                [100.0, 100.0, 900.0, 160.0],
            ),
            item(
                "text",
                "Dear Peter, Next Thursday at noon works! see you then.",
                [100.0, 200.0, 900.0, 500.0],
            ),
        ]
    }

    #[test]
    fn a_reread_line_already_inside_the_region_is_not_added_and_new_lines_join_it() {
        let raw = run_together_page();
        let (regions, counts, _) = recover_short(
            &raw,
            Ok("Next Thursday at noon works! see you then.\nMarion"),
        );
        assert_eq!(regions.len(), 2);
        assert_eq!(
            regions[1].text,
            "Dear Peter, Next Thursday at noon works! see you then.\n\nMarion"
        );
        assert_eq!(counts.entries(), vec![("short_recognition_recovered", 1)]);
    }

    #[test]
    fn a_reread_line_differing_only_in_spacing_and_punctuation_is_not_added() {
        let raw = run_together_page();
        let (regions, counts, _) = recover_short(
            &raw,
            Ok("Next Thursday at noon works ! see you then .\nMarion"),
        );
        assert_eq!(
            regions[1].text,
            "Dear Peter, Next Thursday at noon works! see you then.\n\nMarion"
        );
        assert_eq!(counts.entries(), vec![("short_recognition_recovered", 1)]);
    }

    #[test]
    fn lines_with_no_matching_region_are_inserted_as_paragraphs() {
        let raw = run_together_page();
        let mut page = assembled(&raw, PAGE);
        page[1].bbox = Some([300.0, 420.0, 700.0, 520.0]);
        let reader = |_: i64, _: [f64; 4]| Ok("Marion\nBest".to_string());
        let mut counts = RecoveryCounts::default();
        let regions = recover_short_recognition(page, &raw, 1, PAGE, &reader, &mut counts)
            .expect("short recognition");
        let held = texts(&regions);
        assert_eq!(held.len(), 4);
        assert_eq!(
            held[1],
            "Dear Peter, Next Thursday at noon works! see you then."
        );
        assert_eq!(held[2..], ["Marion", "Best"]);
        assert_eq!(counts.entries(), vec![("short_recognition_recovered", 2)]);
    }

    #[test]
    fn does_not_reread_a_recognition_that_fills_its_box() {
        let raw = short_page(260.0);
        let (regions, counts, calls) = recover_short(&raw, Ok("Gamma"));
        assert!(calls.lock().unwrap().is_empty());
        assert_eq!(regions.len(), 2);
        assert_eq!(counts, RecoveryCounts::default());
    }

    #[test]
    fn a_failed_reread_below_a_short_recognition_changes_nothing() {
        let raw = short_page(700.0);
        let (regions, counts, calls) = recover_short(&raw, Err(BandReadError::Failed));
        assert_eq!(calls.lock().unwrap().len(), 1);
        assert_eq!(regions.len(), 2);
        assert_eq!(counts, RecoveryCounts::default());
    }

    const PARAGRAPH: &str =
        "The committee approved the annual budget for the coming fiscal year after long debate.";

    /// Two dense reference paragraphs (800 by 120 pixels, 120 characters each
    /// on one line) and a candidate paragraph region 240 pixels tall.
    fn paragraph_page(candidate: &str) -> Vec<RegionResult> {
        vec![
            item("text", &dense_body(1), [100.0, 100.0, 900.0, 160.0]),
            item("text", &dense_body(2), [100.0, 200.0, 900.0, 260.0]),
            item("text", candidate, [100.0, 400.0, 900.0, 520.0]),
        ]
    }

    #[test]
    fn a_one_line_paragraph_that_fills_its_box_is_not_a_short_recognition() {
        let complete = format!("{} {}", dense_body(7), dense_body(8));
        assert!(short_recognition_bands(&paragraph_page(&complete), PAGE).is_empty());
    }

    #[test]
    fn a_paragraph_read_far_shorter_than_its_box_gets_a_band_from_one_line_above_its_end() {
        let bands = short_recognition_bands(&paragraph_page(PARAGRAPH), PAGE);
        assert_eq!(bands.len(), 1);
        let band = bands[0];
        assert_eq!([band[0], band[2], band[3]], [100.0, 900.0, 520.0]);
        assert!(
            (band[1] - 422.1).abs() < 1.0,
            "two read lines of about 22 end near 444; the band starts one line above: {band:?}"
        );
    }

    #[test]
    fn a_paragraph_filling_a_box_many_lines_tall_still_gets_a_band_from_the_newline_test() {
        let full = (1..=6).map(dense_body).collect::<Vec<_>>().join(" ");
        let mut raw = paragraph_page(&full);
        raw[2].bbox = Some([100.0, 400.0, 900.0, 700.0]);
        let bands = short_recognition_bands(&raw, PAGE);
        assert_eq!(bands, vec![[100.0, 460.0, 900.0, 700.0]]);
    }

    #[test]
    fn a_region_both_tests_flag_gets_one_band_from_the_higher_top() {
        let mut raw = paragraph_page(PARAGRAPH);
        raw[2].bbox = Some([100.0, 400.0, 900.0, 700.0]);
        let bands = short_recognition_bands(&raw, PAGE);
        assert_eq!(bands.len(), 1);
        assert!(
            (bands[0][1] - 422.1).abs() < 1.0,
            "the coverage top 422 is above the newline top 460: {bands:?}"
        );
    }

    #[test]
    fn a_reread_of_a_paragraph_the_region_holds_is_deduped_and_a_distinct_paragraph_is_kept() {
        let raw = paragraph_page(PARAGRAPH);
        let variant =
            "The comittee approved the annual budget for the coming fiscal year after long debate";
        let distinct = "Residents objected to the road closure during the summer festival weekend.";
        let (regions, counts, _) = recover_short(&raw, Ok(&format!("{variant}\n{distinct}")));
        assert_eq!(
            regions[2].text,
            format!("{PARAGRAPH}\n\n{distinct}"),
            "the re-recognised variant adds nothing and the distinct paragraph is appended"
        );
        assert_eq!(counts.entries(), vec![("short_recognition_recovered", 1)]);
    }

    #[test]
    fn held_text_repeats_recognition_variants_and_not_distinct_text() {
        let held = Held::new(PARAGRAPH);
        assert!(held.repeats(&Held::new(
            "The comittee approved the annual budget for the coming fiscal year after long debate"
        )));
        assert!(held.repeats(&Held::new("annual budget for the coming fiscal year")));
        assert!(!held.repeats(&Held::new(
            "Residents objected to the road closure during the summer festival weekend."
        )));
        assert!(
            !Held::new("Albany")
                .repeats(&Held::new("A long unrelated sentence about Albany roads."))
        );
    }

    #[test]
    fn a_reread_suffix_with_a_footnote_marker_and_a_hyphen_merges_into_the_held_paragraph() {
        let held = "The statute [21 U.S.C. 355(a)] . Receipt of valid prescriptions for individuallyidentified patients is required.";
        let read = "[21 U.S.C. 355(a)] . $^{1}$ Receipt of valid prescriptions for individually-identified patients is required.";
        let raw = paragraph_page(held);
        let (regions, counts, _) = recover_short(&raw, Ok(read));
        assert_eq!(
            regions[2].text,
            "The statute [21 U.S.C. 355(a)] . $^{1}$ Receipt of valid prescriptions for individually-identified patients is required."
        );
        assert_eq!(counts.entries(), vec![("short_recognition_recovered", 1)]);
    }

    #[test]
    fn a_reread_with_real_words_replaces_a_redaction_span_in_the_held_text() {
        let held = "High Oil and Gas Resource and Technology \u{e000}redacted\u{e001} High Economic Growth case with steady demand across every sector and region";
        let read = "Resource and Technology High Oil Price High Economic Growth case with steady demand across every sector and region";
        let raw = paragraph_page(held);
        let (regions, _, _) = recover_short(&raw, Ok(read));
        assert_eq!(
            regions[2].text,
            "High Oil and Gas Resource and Technology High Oil Price High Economic Growth case with steady demand across every sector and region"
        );
    }

    #[test]
    fn merge_keeps_held_text_a_worse_reread_would_overwrite_and_takes_extensions() {
        assert_eq!(
            merge_text("the fatuity. end", "the futility. end"),
            "the fatuity. end"
        );
        assert_eq!(merge_text("the Revolu", "the Revolution"), "the Revolution");
        assert_eq!(
            merge_text("you come back,", "you, some bank,"),
            "you come back,"
        );
    }

    #[test]
    fn merge_replaces_a_redaction_inside_a_mixed_run_with_the_read_tokens() {
        let held =
            "Low Oil and Gas Resource and Technology \u{e000}redacted\u{e001} High Economic Growth";
        let read = "Technology High Oil Price High Economic Growth";
        let merged = merge_text(held, read);
        assert!(merged.contains("High Oil Price"), "{merged}");
        assert!(
            !merged.contains('\u{e000}') && !merged.contains("redacted"),
            "{merged}"
        );
        assert!(
            merged.starts_with("Low Oil and Gas Resource and Technology"),
            "{merged}"
        );
    }

    #[test]
    fn merge_lets_a_redaction_absorb_the_read_tokens_between_its_neighbours() {
        let held = "High Oil (and Gas) Resource and Technology \u{e000}redacted\u{e001} High Economic Growth";
        let read = "Technology High Oil Price High Economic Growth";
        let merged = merge_text(held, read);
        assert!(
            merged.contains("Technology High Oil Price High Economic"),
            "{merged}"
        );
        assert!(!merged.contains('\u{e000}'), "{merged}");
    }

    #[test]
    fn merge_keeps_the_held_paragraph_separators_and_ties_to_the_held_side() {
        assert_eq!(
            merge_text("alpha beta\n\ngamma delta", "alpha beta gamma delta"),
            "alpha beta\n\ngamma delta"
        );
        assert_eq!(merge_text("one two", "one 2"), "one two");
        assert_eq!(merge_text("one two", "one two three"), "one two three");
    }

    #[test]
    fn merge_beyond_the_cell_limit_keeps_both_texts_whole() {
        let held = vec!["a"; 2100].join(" ");
        let read = vec!["b"; 2100].join(" ");
        let merged = merge_text(&held, &read);
        assert_eq!(merged, format!("{held}\n\n{read}"));
    }

    type RereadCalls = Mutex<Vec<(i64, [f64; 4], Reread)>>;

    fn rereads(
        raw: &[RegionResult],
        answer: Result<&str, BandReadError>,
    ) -> (Vec<Region>, RecoveryCounts, RereadCalls) {
        let calls = Mutex::new(Vec::new());
        let reader = |page: i64, band: [f64; 4], mode: Reread| {
            calls.lock().unwrap().push((page, band, mode));
            answer.clone().map(str::to_string)
        };
        let mut counts = RecoveryCounts::default();
        let regions = recover_repetition(assembled(raw, PAGE), raw, 1, PAGE, &reader, &mut counts)
            .expect("repetition");
        (regions, counts, calls)
    }

    fn stopped(label: &str, content: &str, bbox: [f64; 4]) -> RegionResult {
        RegionResult {
            repetition_stopped: true,
            ..item(label, content, bbox)
        }
    }

    #[test]
    fn a_repetition_stopped_reply_is_read_again_upscaled_and_replaces_the_shorter_text() {
        let mut raw = paragraph_page(PARAGRAPH);
        raw[2] = stopped("text", PARAGRAPH, [100.0, 400.0, 900.0, 520.0]);
        let full = format!(
            "{PARAGRAPH} Residents objected to the road closure during the summer festival weekend."
        );
        let (regions, counts, calls) = rereads(&raw, Ok(&full));
        assert_eq!(
            *calls.lock().unwrap(),
            vec![(1, [100.0, 400.0, 900.0, 520.0], Reread::Upscaled)]
        );
        assert_eq!(regions[2].text, full);
        assert_eq!(counts.entries(), vec![("repetition_reread", 1)]);
    }

    #[test]
    fn a_runaway_the_content_cleaning_cuts_is_read_again_without_the_stop_field() {
        let unit = "0123456789ab ";
        let looped = format!("intro {}", unit.repeat(250));
        let mut raw = paragraph_page(PARAGRAPH);
        raw[2] = item("text", &looped, [100.0, 400.0, 900.0, 520.0]);
        let (_, _, calls) = rereads(&raw, Err(BandReadError::Failed));
        assert_eq!(calls.lock().unwrap().len(), 1);
    }

    #[test]
    fn a_leader_run_no_line_of_the_region_could_hold_is_read_again() {
        let runaway = format!("Chapter one {} 12", ".".repeat(400));
        let mut raw = paragraph_page(PARAGRAPH);
        raw[2] = item("text", &runaway, [100.0, 400.0, 900.0, 520.0]);
        let (_, _, calls) = rereads(&raw, Err(BandReadError::Failed));
        assert_eq!(calls.lock().unwrap().len(), 1);
        let ordinary = format!("Chapter one {} 12", ".".repeat(40));
        raw[2] = item("text", &ordinary, [100.0, 400.0, 900.0, 520.0]);
        let (_, _, calls) = rereads(&raw, Err(BandReadError::Failed));
        assert!(calls.lock().unwrap().is_empty());
    }

    #[test]
    fn a_contents_region_flagged_only_by_its_leaders_still_gets_a_band_below_its_text() {
        let contents = format!(
            "1 Introduction {} 3\n2 Fundamentals of Graph Theory {} 7",
            ".".repeat(400),
            ".".repeat(400)
        );
        let mut raw = paragraph_page(PARAGRAPH);
        raw[2] = item("text", &contents, [100.0, 400.0, 900.0, 900.0]);
        assert!(repetition_runaway(&raw, 2, PAGE));
        assert!(!repetition_stopped_or_cut(&raw, 2));
        let bands = short_recognition_bands(&raw, PAGE);
        assert_eq!(bands.len(), 1);
        assert_eq!(bands[0][2..], [900.0, 900.0]);
    }

    #[test]
    fn a_stopped_region_whose_text_covers_the_top_of_its_box_gets_a_band_below_its_text() {
        let contents = "1 General Notation . . . . . . . . 3\n2 Fundamentals . . . . . . 7";
        let mut raw = paragraph_page(contents);
        raw[2] = stopped("text", contents, [100.0, 400.0, 900.0, 900.0]);
        assert!(repetition_stopped_or_cut(&raw, 2));
        let bands = short_recognition_bands(&raw, PAGE);
        assert_eq!(bands.len(), 1);
        assert!(bands[0][1] > 400.0 && bands[0][1] < 900.0, "{bands:?}");
        assert_eq!(bands[0][2..], [900.0, 900.0]);
    }

    #[test]
    fn a_runaway_taller_than_its_box_gets_no_band() {
        let looped = format!("intro {}", "0123456789ab ".repeat(250));
        let mut raw = paragraph_page(PARAGRAPH);
        raw[2] = stopped("text", &looped, [100.0, 400.0, 900.0, 520.0]);
        assert!(short_recognition_bands(&raw, PAGE).is_empty());
    }

    #[test]
    fn a_second_reading_that_holds_no_more_still_adds_its_lines_the_region_lacks() {
        let held =
            "Alpha chapter one begins here\nBeta chapter two follows it\nGamma chapter three ends";
        let mut raw = paragraph_page(held);
        raw[2] = stopped("text", held, [100.0, 400.0, 900.0, 520.0]);
        let (regions, counts, _) = rereads(
            &raw,
            Ok("Alpha chapter one begins here\nDelta appendix material is new"),
        );
        let before = assembled(&raw, PAGE)[2].text.clone();
        assert_eq!(
            regions[2].text,
            format!("{before}\n\nDelta appendix material is new")
        );
        assert_eq!(counts.entries(), vec![("repetition_reread", 1)]);
    }

    #[test]
    fn a_reply_with_neither_signal_is_never_read_again() {
        let raw = paragraph_page(PARAGRAPH);
        let (regions, counts, calls) = rereads(&raw, Ok("anything at all"));
        assert!(calls.lock().unwrap().is_empty());
        assert_eq!(regions[2].text, PARAGRAPH);
        assert_eq!(counts, RecoveryCounts::default());
    }

    #[test]
    fn a_second_reading_that_holds_no_more_leaves_the_region_and_a_failed_one_too() {
        let mut raw = paragraph_page(PARAGRAPH);
        raw[2] = stopped("text", PARAGRAPH, [100.0, 400.0, 900.0, 520.0]);
        let (regions, counts, _) = rereads(&raw, Ok("The committee approved"));
        assert_eq!(regions[2].text, PARAGRAPH);
        assert_eq!(counts, RecoveryCounts::default());
        let (regions, counts, _) = rereads(&raw, Err(BandReadError::Failed));
        assert_eq!(regions[2].text, PARAGRAPH);
        assert_eq!(counts, RecoveryCounts::default());
    }

    #[test]
    fn a_region_already_at_the_pixel_cap_is_read_in_strips_of_about_ten_lines() {
        let mut raw = paragraph_page(PARAGRAPH);
        raw[2] = stopped("text", PARAGRAPH, [100.0, 100.0, 900.0, 900.0]);
        raw[0].bbox = Some([100.0, 920.0, 900.0, 980.0]);
        raw[1].bbox = Some([100.0, 985.0, 900.0, 1000.0]);
        let (_, _, calls) = rereads(&raw, Err(BandReadError::Failed));
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].2, Reread::Strips(4));
    }

    #[test]
    fn the_reread_plan_doubles_a_crop_that_fits_and_strips_one_that_does_not() {
        let small = [100.0, 100.0, 900.0, 300.0];
        assert_eq!(reread_plan(small, PAGE, Some(40.0)), Reread::Upscaled);
        let tall = [100.0, 100.0, 900.0, 900.0];
        assert_eq!(reread_plan(tall, PAGE, Some(40.0)), Reread::Strips(4));
        assert_eq!(reread_plan(tall, PAGE, None), Reread::Upscaled);
    }

    #[test]
    fn strips_are_cut_on_the_quietest_row_near_the_even_cut() {
        let mut image = image::RgbImage::from_pixel(40, 100, image::Rgb([255, 255, 255]));
        for row in 0..100 {
            if (row % 10) < 6 {
                for column in 0..40 {
                    image.put_pixel(column, row, image::Rgb([0, 0, 0]));
                }
            }
        }
        let strips = strip_rows(&image, 2);
        assert_eq!(strips.len(), 2);
        assert_eq!(strips[0].0, 0);
        assert_eq!(strips[1].1, 100);
        assert_eq!(strips[0].1, strips[1].0);
        assert!(
            strips[0].1 % 10 >= 6,
            "the cut {} falls between text rows",
            strips[0].1
        );
    }

    #[test]
    fn recovered_lines_take_ordered_disjoint_boxes_inside_the_band() {
        let (regions, _, _) = recover(&band_page(), Ok("ONE\nTWO\nTHREE\nFOUR"));
        let boxes: Vec<[f64; 4]> = regions
            .iter()
            .filter(|r| ["ONE", "TWO", "THREE", "FOUR"].contains(&r.text.as_str()))
            .filter_map(|r| r.bbox)
            .collect();
        assert_eq!(boxes.len(), 4);
        let band = (200.0 * 2.0, 370.0 * 2.0);
        for pair in boxes.windows(2) {
            assert!(pair[0][3] <= pair[1][1] && pair[0][1] < pair[1][1]);
        }
        for b in &boxes {
            assert!(b[1] >= band.0 - 1e-6 && b[3] <= band.1 + 1e-6 && b[1] < b[3]);
        }
    }

    #[test]
    fn reads_uncovered_band_above_the_first_heading() {
        let (regions, counts, calls) = recover(&chapter_page(vec![]), Ok("STAVE FIVE"));
        assert_eq!(
            *calls.lock().unwrap(),
            vec![(1, [0.0, 70.0, 1000.0, 500.0])]
        );
        assert_eq!(
            kinds_and_texts(&regions),
            vec![
                ("page_furniture", "7"),
                ("paragraph", "STAVE FIVE"),
                ("heading", "CHAPTER TITLE"),
                ("paragraph", "Body text."),
            ]
        );
        assert!(counts.entries().contains(&("blank_band_recovered", 1)));
    }

    #[test]
    fn does_not_read_a_small_or_covered_band_above_the_heading() {
        let mut small = chapter_page(vec![]);
        small[0].bbox = Some([300.0, 480.0, 340.0, 495.0]);
        assert!(recover(&small, Ok("x")).2.lock().unwrap().is_empty());
        let covered = chapter_page(vec![figure([100.0, 80.0, 900.0, 495.0])]);
        assert!(recover(&covered, Ok("x")).2.lock().unwrap().is_empty());
    }

    #[test]
    fn fence_only_heading_is_blank_and_makes_no_region() {
        let mut raw = chapter_page(vec![]);
        raw[1].content = Some("```markdown\n\n```".to_string());
        raw.push(item("text", "Late.", [100.0, 720.0, 900.0, 760.0]));
        assert_eq!(blank_text_items(&raw), vec![[300.0, 500.0, 700.0, 540.0]]);
        let (regions, _, _) = recover(&raw, Ok(""));
        assert_eq!(texts(&regions), vec!["7", "Body text.", "Late."]);
    }

    #[test]
    fn fence_only_heading_with_a_heading_prefix_is_blank() {
        let mut raw = chapter_page(vec![]);
        raw[1].content = Some("## ```markdown\n\n```".to_string());
        let (regions, _, _) = recover(&raw, Ok(""));
        assert_eq!(texts(&regions), vec!["7", "Body text."]);
    }

    #[test]
    fn band_reads_drop_code_fence_marker_lines() {
        let (regions, _, _) = recover(&chapter_page(vec![]), Ok("```markdown\nSTAVE FIVE\n```"));
        assert_eq!(
            texts(&regions),
            vec!["7", "STAVE FIVE", "CHAPTER TITLE", "Body text."]
        );
        let (regions, _, _) = recover(&chapter_page(vec![]), Ok("```markdown\n\n```"));
        assert_eq!(texts(&regions), vec!["7", "CHAPTER TITLE", "Body text."]);
    }

    #[test]
    fn fence_with_code_still_becomes_code() {
        let raw = [item(
            "text",
            "```python\nprint(1)\n```",
            [100.0, 100.0, 900.0, 200.0],
        )];
        let (regions, _, _) = recover(&raw, Ok(""));
        assert_eq!(kinds_and_texts(&regions), vec![("code", "print(1)")]);
    }

    #[test]
    fn skips_lines_that_repeat_page_text_or_hold_no_letters() {
        let (regions, _, _) = recover(&band_page(), Ok("TITLE LINE\n---"));
        assert_eq!(texts(&regions), vec!["7", "TITLE LINE", "Body text."]);
    }

    #[test]
    fn ignores_pages_without_blank_text_boxes_or_with_figures_on_them() {
        let mut raw: Vec<RegionResult> = band_page()
            .into_iter()
            .filter(|i| i.content.as_deref().is_some_and(|c| !c.trim().is_empty()))
            .collect();
        raw.push(item("text", "Above.", [100.0, 75.0, 900.0, 385.0]));
        let (regions, _, calls) = recover(&raw, Ok("x"));
        assert!(calls.lock().unwrap().is_empty());
        assert_eq!(regions.len(), 4);
        let mut with_figure = vec![figure([100.0, 150.0, 500.0, 380.0])];
        with_figure.extend(band_page());
        with_figure.push(raw[raw.len() - 1].clone());
        let (regions, _, calls) = recover(&with_figure, Ok("LABEL"));
        assert!(calls.lock().unwrap().is_empty());
        assert!(!texts(&regions).join(" ").contains("LABEL"));
    }

    #[test]
    fn a_failed_band_read_keeps_the_recognized_regions() {
        let (regions, counts, _) = recover(&band_page(), Err(BandReadError::Failed));
        assert_eq!(regions.len(), 3);
        assert_eq!(counts, RecoveryCounts::default());
    }

    #[test]
    fn a_cancelled_band_read_stops_the_page() {
        let raw = band_page();
        let reader = |_: i64, _: [f64; 4]| Err(BandReadError::Cancelled);
        let mut counts = RecoveryCounts::default();
        let result = recover_blank_band(assembled(&raw, PAGE), &raw, 1, PAGE, &reader, &mut counts);
        assert_eq!(result, Err(RecoverError::Cancelled));
    }

    // ---- PDFs ----

    fn build_pdf(objects: &[Vec<u8>], trailer_extra: &str) -> Vec<u8> {
        let mut output = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (number, body) in objects.iter().enumerate() {
            offsets.push(output.len());
            output.extend(format!("{} 0 obj\n", number + 1).into_bytes());
            output.extend(body);
            output.extend(b"\nendobj\n");
        }
        let xref = output.len();
        output.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).into_bytes());
        for offset in offsets {
            output.extend(format!("{offset:010} 00000 n \n").into_bytes());
        }
        output.extend(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R {trailer_extra}>>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .into_bytes(),
        );
        output
    }

    fn plain_pdf(info: Option<&str>) -> Vec<u8> {
        let mut objects: Vec<Vec<u8>> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>".to_vec(),
        ];
        match info {
            Some(body) => {
                objects.push(body.as_bytes().to_vec());
                build_pdf(&objects, "/Info 4 0 R ")
            }
            None => build_pdf(&objects, ""),
        }
    }

    #[test]
    fn metadata_title_is_empty_for_images_and_untitled_pdfs() {
        assert_eq!(pdf_metadata_title(b"\x89PNG\r\n\x1a\n"), "");
        assert_eq!(pdf_metadata_title(&plain_pdf(None)), "");
    }

    #[test]
    fn metadata_title_reads_text_strings_and_collapses_whitespace() {
        let plain = plain_pdf(Some("<< /Title (  Synthetic   Field\nGuide ) >>"));
        assert_eq!(pdf_metadata_title(&plain), "Synthetic Field Guide");
        let utf16 = plain_pdf(Some("<< /Title <FEFF00480069> >>"));
        assert_eq!(pdf_metadata_title(&utf16), "Hi");
    }

    fn form_pdf(
        radio_options: bool,
        fill_name: bool,
        redact_name: bool,
        page_extra: &str,
    ) -> Vec<u8> {
        let empty =
            b"<< /Type /XObject /Subtype /Form /BBox [0 0 12 12] /Length 0 >>\nstream\n\nendstream"
                .to_vec();
        let widget = "<< /Type /Annot /Subtype /Widget /F 4 /P 3 0 R ";
        let fill = if fill_name {
            "0 g 90 695 220 30 re f"
        } else {
            ""
        };
        let annots = format!(
            "[6 0 R 7 0 R 8 0 R 10 0 R 11 0 R{}]",
            if redact_name { " 13 0 R" } else { "" }
        );
        let radio = format!(
            "<< /FT /Btn /Ff 49152 /T (size) /V /1 {} /Kids [10 0 R 11 0 R] >>",
            if radio_options {
                "/Opt [(Small) (Large)]"
            } else {
                "/TU (Size choice)"
            }
        );
        let mut objects: Vec<Vec<u8>> = vec![
            b"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [6 0 R 7 0 R 8 0 R 9 0 R] /DA (/F1 12 Tf 0 g) >> >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R /Annots {annots} {page_extra} >>").into_bytes(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
            format!("<< /Length {} >>\nstream\n{fill}\nendstream", fill.len()).into_bytes(),
            format!("{widget}/FT /Tx /T (name) /TU (Full name) /V (Ada Synthetic) /Rect [100 700 300 720] >>").into_bytes(),
            format!("{widget}/FT /Btn /T (agree) /TU (Accept terms) /V /Yes /AS /Yes /Rect [100 650 112 662] /AP << /N << /Yes 12 0 R /Off 12 0 R >> >> >>").into_bytes(),
            format!("{widget}/FT /Btn /T (news) /TU (Send news) /V /Off /AS /Off /Rect [100 600 112 612] /AP << /N << /Yes 12 0 R /Off 12 0 R >> >> >>").into_bytes(),
            radio.into_bytes(),
            format!("{widget}/Parent 9 0 R /AS /Off /Rect [100 550 112 562] /AP << /N << /0 12 0 R /Off 12 0 R >> >> >>").into_bytes(),
            format!("{widget}/Parent 9 0 R /AS /1 /Rect [100 500 112 512] /AP << /N << /1 12 0 R /Off 12 0 R >> >> >>").into_bytes(),
            empty,
        ];
        if redact_name {
            objects.push(
                b"<< /Type /Annot /Subtype /Redact /Rect [90 695 310 725] /QuadPoints [90 725 310 725 90 695 310 695] >>".to_vec(),
            );
        }
        build_pdf(&objects, "")
    }

    fn clear(_: usize, _: [f64; 4]) -> Cover {
        Cover::Clear
    }

    fn fields_of(scan: &FormScan) -> &Vec<FormField> {
        scan.fields.get(&1).expect("page 1 fields")
    }

    fn field_texts(fields: &[FormField]) -> Vec<&str> {
        fields.iter().map(|f| f.text.as_str()).collect()
    }

    fn page_region(text: &str, bbox: [f64; 4]) -> Region {
        paragraph_region(1, text.to_string(), Some(bbox))
    }

    fn px(bbox: [f64; 4]) -> [f64; 4] {
        [
            bbox[0] * 612.0 / 1000.0,
            bbox[1] * 792.0 / 1000.0,
            bbox[2] * 612.0 / 1000.0,
            bbox[3] * 792.0 / 1000.0,
        ]
    }

    #[test]
    fn a_filled_text_field_and_checkboxes_read_as_label_and_state() {
        let scan = pdf_form_fields(&form_pdf(true, false, false, ""), &[(612.0, 792.0)], &clear);
        let fields = fields_of(&scan);
        let merged = with_form_fields(
            vec![page_region("Application", px([100.0, 20.0, 900.0, 60.0]))],
            fields,
            1,
        );
        let all = texts(&merged);
        assert_eq!(all[0], "Application");
        assert!(all.contains(&"Full name: Ada Synthetic"));
        let checked = format!("\u{2611} Accept terms: {CHECKED_PLACEHOLDER}");
        let unchecked = format!("\u{2610} Send news: {UNCHECKED_PLACEHOLDER}");
        assert!(all.contains(&checked.as_str()));
        assert!(all.contains(&unchecked.as_str()));
        for region in &merged[1..] {
            assert_eq!(region.alt_text.as_deref(), Some(region.text.as_str()));
        }
    }

    #[test]
    fn a_radio_group_reads_its_option_display_values() {
        let scan = pdf_form_fields(&form_pdf(true, false, false, ""), &[(612.0, 792.0)], &clear);
        let all = field_texts(fields_of(&scan));
        let small = format!("\u{2610} Small: {UNCHECKED_PLACEHOLDER}");
        let large = format!("\u{2611} Large: {CHECKED_PLACEHOLDER}");
        assert!(all.contains(&small.as_str()), "{all:?}");
        assert!(all.contains(&large.as_str()), "{all:?}");
    }

    #[test]
    fn a_radio_group_without_options_keeps_its_label_not_a_raw_state_name() {
        let scan = pdf_form_fields(
            &form_pdf(false, false, false, ""),
            &[(612.0, 792.0)],
            &clear,
        );
        let all = field_texts(fields_of(&scan));
        let off = format!("\u{2610} Size choice: {UNCHECKED_PLACEHOLDER}");
        let on = format!("\u{2611} Size choice: {CHECKED_PLACEHOLDER}");
        assert!(all.contains(&off.as_str()), "{all:?}");
        assert!(all.contains(&on.as_str()), "{all:?}");
    }

    fn name_field_box() -> [f64; 4] {
        [90.0, 67.0, 310.0, 97.0]
    }

    fn covers_name(_: usize, field: [f64; 4]) -> Cover {
        if overlaps(&field, &name_field_box()) {
            Cover::Covered
        } else {
            Cover::Clear
        }
    }

    #[test]
    fn a_form_field_under_a_filled_box_never_reaches_the_reply() {
        let scan = pdf_form_fields(
            &form_pdf(true, true, false, ""),
            &[(612.0, 792.0)],
            &covers_name,
        );
        let fields = fields_of(&scan);
        assert!(!format!("{fields:?}").contains("Ada"));
        let withheld = format!("Full name: {REDACTED_PLACEHOLDER}");
        assert!(field_texts(fields).contains(&withheld.as_str()));
    }

    #[test]
    fn form_fields_follow_a_rotated_cropped_page_and_a_widget_under_a_filled_box_is_withheld() {
        let rotated_cover = |_: usize, field: [f64; 4]| {
            if overlaps(&field, &[665.0, 70.0, 695.0, 290.0]) {
                Cover::Covered
            } else {
                Cover::Clear
            }
        };
        let pdf = form_pdf(true, true, false, "/Rotate 90 /CropBox [20 30 612 792]");
        let scan = pdf_form_fields(&pdf, &[(762.0, 592.0)], &rotated_cover);
        let fields = fields_of(&scan);
        assert!(!format!("{fields:?}").contains("Ada"));
        let withheld = format!("Full name: {REDACTED_PLACEHOLDER}");
        assert!(field_texts(fields).contains(&withheld.as_str()));
        let accept = fields
            .iter()
            .find(|f| {
                f.text
                    .ends_with(&format!("Accept terms: {CHECKED_PLACEHOLDER}"))
            })
            .expect("accept terms field");
        for (got, want) in accept.bbox.iter().zip([620.0, 80.0, 632.0, 92.0]) {
            assert!((got - want).abs() < 1.0, "{:?}", accept.bbox);
        }
    }

    #[test]
    fn a_form_field_under_a_redact_annotation_never_reaches_the_reply() {
        let scan = pdf_form_fields(
            &form_pdf(true, false, true, ""),
            &[(612.0, 792.0)],
            &covers_name,
        );
        let fields = fields_of(&scan);
        assert!(!format!("{fields:?}").contains("Ada"));
        let withheld = format!("Full name: {REDACTED_PLACEHOLDER}");
        let checked = format!("\u{2611} Accept terms: {CHECKED_PLACEHOLDER}");
        assert!(field_texts(fields).contains(&withheld.as_str()));
        assert!(field_texts(fields).contains(&checked.as_str()));
    }

    #[test]
    fn an_undeterminable_cover_withholds_the_field_and_is_counted() {
        let unknown = |_: usize, _: [f64; 4]| Cover::Unknown;
        let scan = pdf_form_fields(
            &form_pdf(true, false, false, ""),
            &[(612.0, 792.0)],
            &unknown,
        );
        let fields = fields_of(&scan);
        assert!(!format!("{fields:?}").contains("Ada"));
        assert!(fields.iter().all(|f| f.value.is_empty()));
        assert!(
            fields
                .iter()
                .all(|f| !f.text.contains('\u{2611}') && !f.text.contains('\u{2610}'))
        );
        assert_eq!(scan.cover_unknown, 5);
        assert_eq!(scan.entries(), vec![("form_field_cover_unknown", 5)]);
    }

    #[test]
    fn form_fields_follow_reading_order_and_skip_values_already_recognized() {
        let scan = pdf_form_fields(&form_pdf(true, false, false, ""), &[(612.0, 792.0)], &clear);
        let merged = with_form_fields(
            vec![
                page_region("Ada Synthetic", px([150.0, 90.0, 500.0, 110.0])),
                page_region("Closing remarks", px([100.0, 900.0, 900.0, 950.0])),
            ],
            fields_of(&scan),
            1,
        );
        let all = texts(&merged);
        assert!(!all.contains(&"Full name: Ada Synthetic"));
        assert_eq!(all[all.len() - 1], "Closing remarks");
    }

    #[test]
    fn a_source_without_a_form_yields_no_fields() {
        let scan = pdf_form_fields(&plain_pdf(None), &[(612.0, 792.0)], &clear);
        assert!(scan.fields.is_empty() && !scan.unreadable);
        let bad = pdf_form_fields(b"%PDF-1.4 not a pdf", &[(612.0, 792.0)], &clear);
        assert!(bad.fields.is_empty() && bad.unreadable);
        assert_eq!(bad.entries(), vec![("form_fields_unreadable", 1)]);
    }
}
