//! What a PDF page's own objects add to recognised regions: text-layer
//! substitution for regions whose recognition lost text, withholding of
//! words a page covers or redacts, and link spans from link annotations.
//!
//! A [`PageReader`] reads one page: its glyphs, its `/Redact` annotations and
//! its link annotations, and decides per glyph whether the page withholds it.
//! A glyph is withheld when it lies off the page, inside a `/Redact`
//! annotation, or over page pixels without glyph contrast (a filled box, an
//! image painted over it, invisible text over blank paper). Every decision
//! that cannot be made withholds. [`finish_region`] applies the behaviours to
//! one recognised region at the stage the bridge applies them (after the
//! region's kind and text are known, before page furniture is labelled and
//! ids are assigned); [`finish_regions`] applies them to the assembled
//! regions of one page.

use std::cell::{OnceCell, RefCell};
use std::collections::HashSet;
use std::panic::AssertUnwindSafe;

use crate::textutil::is_external_link_target;
use crate::wire::{LinkSpan, REDACTED_PLACEHOLDER, Region};
use hayro::hayro_interpret::{InterpreterSettings, TransformExt};
use hayro::hayro_syntax::Pdf;
use hayro::hayro_syntax::object::{
    Array, Dict, MaybeRef, Name, Object, ObjectIdentifier, Rect, String as PdfString,
};
use hayro::hayro_syntax::page::Page;
use hayro::kurbo::{Affine, Point};
use regex::Regex;

use crate::crop::MAX_PIXELS;
use crate::raster;
use crate::render::RenderedPage;
use crate::text_layer::{
    MarkedGlyph, MarkedWord, TextGlyph, expand_ligatures, join_hyphenated, marked_lines,
    page_glyphs,
};

/// The extent of the normalised box space of layout results.
const BBOX_EXTENT: f64 = 1_000.0;

/// The fewest letters and digits a text layer must hold to replace a
/// recognition.
pub const MIN_ALPHANUMERIC: usize = 40;

/// The most bytes a region's text may hold.
const MAX_REGION_TEXT_BYTES: usize = 1024 * 1024;

/// The height in raster pixels the smallest glyph of a page is rendered at
/// when the pixel budget allows.
const COVER_MIN_GLYPH_PIXELS: f64 = 8.0;

/// The fraction of the page's own ink contrast a glyph's surroundings must
/// show.
const COVER_GLYPH_CONTRAST_FRACTION: f64 = 0.25;

/// The region kinds whose text the text layer may replace.
const TEXT_LAYER_KINDS: [&str; 4] = ["paragraph", "heading", "list_item", "caption"];

/// The text of a region read from the page's text layer, with the words the
/// page withholds already removed from `text`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayerText {
    /// The visible text: one line per printed line, line-break hyphens
    /// joined, withheld words left out.
    pub text: String,
    /// Every word of the region in order with whether it is withheld; empty
    /// when no word is withheld.
    pub tokens: Vec<(String, bool)>,
    /// The characters of the withheld words.
    pub covered_chars: usize,
    /// The region's visible links in reading order.
    pub links: Vec<LayerLink>,
}

/// A link annotation over visible words of a region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerLink {
    /// The linked words separated by single spaces.
    pub words: String,
    /// An http, https or mailto address, or `page:N[@Y]`.
    pub target: String,
    /// The count of earlier occurrences of the same words in the region.
    pub ordinal: usize,
}

/// A page's ink: its luma raster and the contrast thresholds measured on it.
struct Ink {
    width: u32,
    height: u32,
    luma: Vec<u8>,
    threshold: f64,
    background: u8,
}

impl Ink {
    fn render(page: &Page<'_>, width: u64, height: u64, annotations: bool) -> Option<Self> {
        let settings = InterpreterSettings {
            render_annotations: annotations,
            ..InterpreterSettings::default()
        };
        let drawn = std::panic::catch_unwind(AssertUnwindSafe(|| {
            raster::render_sized_with(page, width, height, &settings)
        }))
        .ok()?
        .ok()?;
        let luma: Vec<u8> = drawn
            .rgba
            .chunks_exact(4)
            .map(|pixel| {
                ((299 * u32::from(pixel[0])
                    + 587 * u32::from(pixel[1])
                    + 114 * u32::from(pixel[2]))
                    / 1000) as u8
            })
            .collect();
        if luma.is_empty() || luma.len() != drawn.width as usize * drawn.height as usize {
            return None;
        }
        let low = *luma.iter().min()?;
        let high = *luma.iter().max()?;
        let mut histogram = [0usize; 256];
        for level in &luma {
            histogram[usize::from(*level)] += 1;
        }
        let mut background = 0usize;
        for (level, count) in histogram.iter().enumerate() {
            if *count > histogram[background] {
                background = level;
            }
        }
        Some(Self {
            width: drawn.width,
            height: drawn.height,
            luma,
            threshold: f64::from(high - low) * COVER_GLYPH_CONTRAST_FRACTION,
            background: background as u8,
        })
    }

    /// The darkest and lightest level of pixels `x0..x1` by `y0..y1`, which
    /// must be non-empty and inside the raster.
    fn extrema(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> (u8, u8) {
        let width = self.width as usize;
        let mut low = u8::MAX;
        let mut high = u8::MIN;
        for row in y0..y1 {
            for level in &self.luma[row * width + x0..row * width + x1] {
                low = low.min(*level);
                high = high.max(*level);
            }
        }
        (low, high)
    }

    fn mean(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        let width = self.width as usize;
        let mut sum = 0u64;
        let mut count = 0u64;
        for row in y0..y1 {
            for level in &self.luma[row * width + x0..row * width + x1] {
                sum += u64::from(*level);
                count += 1;
            }
        }
        sum as f64 / count.max(1) as f64
    }
}

/// The reader of one PDF page's glyphs, redactions and links.
pub struct PageReader<'a> {
    document: &'a Pdf,
    page_index: usize,
    draw_annotations: bool,
    scale: f64,
    width: f64,
    height: f64,
    glyphs: Vec<TextGlyph>,
    upright: Vec<bool>,
    quarters: Vec<Option<u8>>,
    redactions: Vec<[f64; 4]>,
    links: Vec<(Vec<[f64; 4]>, String)>,
    ink: OnceCell<Option<Ink>>,
    decided: RefCell<Vec<Option<bool>>>,
}

impl<'a> PageReader<'a> {
    /// Reads zero-based `page_index` of `document`. `rendered` is the page's
    /// raster for OCR: its vertical scale converts the heights of in-document
    /// link destinations to the pixels of region boxes. With
    /// `draw_annotations` false the contrast checks see the page without
    /// annotation appearances (for form widgets, whose own appearance shows
    /// their value). `None` when the page or its glyphs cannot be read.
    pub fn open(
        document: &'a Pdf,
        page_index: usize,
        rendered: &RenderedPage,
        draw_annotations: bool,
    ) -> Option<Self> {
        let pages = document.pages();
        let page = pages.get(page_index)?;
        let (width_pt, height_pt) = page.render_dimensions();
        let (width_pt, height_pt) = (f64::from(width_pt), f64::from(height_pt));
        if !(width_pt > 0.0 && height_pt > 0.0) {
            return None;
        }
        let points = page_glyphs(document, page_index, 1.0)?;
        let heights = points
            .glyphs
            .iter()
            .filter(|glyph| !glyph.space && glyph.y1 > glyph.y0 && glyph.x1 > glyph.x0)
            .map(|glyph| glyph.y1 - glyph.y0);
        let budget_scale = (MAX_PIXELS as f64 / (width_pt * height_pt).max(1.0)).sqrt();
        let scale = match heights.fold(f64::INFINITY, f64::min) {
            smallest if smallest.is_finite() => budget_scale.min(COVER_MIN_GLYPH_PIXELS / smallest),
            _ => budget_scale,
        };
        if !(scale.is_finite() && scale > 0.0) {
            return None;
        }
        let glyphs: Vec<TextGlyph> = points
            .glyphs
            .iter()
            .map(|glyph| scaled(glyph, scale))
            .collect();
        let affine = Affine::scale(scale) * page.initial_transform(true).to_kurbo();
        let box_scale = match rendered.y_scale() {
            value if value.is_finite() && value > 0.0 => value,
            _ => 1.0,
        };
        let annotations = read_annotations(document, page, affine, box_scale);
        let count = glyphs.len();
        Some(Self {
            document,
            page_index,
            draw_annotations,
            scale,
            width: width_pt * scale,
            height: height_pt * scale,
            glyphs,
            upright: points.upright,
            quarters: points.quarters,
            redactions: annotations.redactions,
            links: annotations.links,
            ink: OnceCell::new(),
            decided: RefCell::new(vec![None; count]),
        })
    }

    fn ink(&self) -> Option<&Ink> {
        self.ink
            .get_or_init(|| {
                let pages = self.document.pages();
                let page = pages.get(self.page_index)?;
                Ink::render(
                    page,
                    (self.width.ceil() as u64).max(1),
                    (self.height.ceil() as u64).max(1),
                    self.draw_annotations,
                )
            })
            .as_ref()
    }

    fn covered(&self, index: usize) -> bool {
        if let Some(decision) = self.decided.borrow()[index] {
            return decision;
        }
        let decision = self.decide(index);
        self.decided.borrow_mut()[index] = Some(decision);
        decision
    }

    fn decide(&self, index: usize) -> bool {
        let glyph = &self.glyphs[index];
        let (center_x, center_y) = glyph.center();
        if !(0.0..=self.width).contains(&center_x) || !(0.0..=self.height).contains(&center_y) {
            return true;
        }
        if self
            .redactions
            .iter()
            .any(|r| r[0] <= center_x && center_x <= r[2] && r[1] <= center_y && center_y <= r[3])
        {
            return true;
        }
        let Some(ink) = self.ink() else {
            return true;
        };
        let x0 = (glyph.x0 as i64 - 1).max(0);
        let x1 = (glyph.x1.ceil() as i64 + 1).min(i64::from(ink.width));
        let y0 = (glyph.y0 as i64 - 1).max(0);
        let y1 = (glyph.y1.ceil() as i64 + 1).min(i64::from(ink.height));
        if x1 <= x0 || y1 <= y0 {
            return true;
        }
        let (low, high) = ink.extrema(x0 as usize, y0 as usize, x1 as usize, y1 as usize);
        high == low || f64::from(high - low) < ink.threshold
    }

    /// Whether the box `[left, top, right, bottom]`, in points from the top
    /// left of the page as rendered, is withheld: it overlaps a `/Redact`
    /// annotation, lies off the page, or is painted one flat colour that
    /// differs from the page background by at least the glyph contrast
    /// threshold. A box that cannot be judged is withheld.
    pub fn rect_covered(&self, left: f64, top: f64, right: f64, bottom: f64) -> bool {
        if [left, top, right, bottom].iter().any(|v| !v.is_finite()) {
            return true;
        }
        let (left, right) = (left.min(right) * self.scale, left.max(right) * self.scale);
        let (top, bottom) = (top.min(bottom) * self.scale, top.max(bottom) * self.scale);
        if right < 0.0 || bottom < 0.0 || left > self.width || top > self.height {
            return true;
        }
        if self
            .redactions
            .iter()
            .any(|r| left < r[2] && r[0] < right && top < r[3] && r[1] < bottom)
        {
            return true;
        }
        let Some(ink) = self.ink() else {
            return true;
        };
        let (width, height) = (ink.width as usize, ink.height as usize);
        let x0 = (left.max(0.0) as usize).min(width - 1);
        let x1 = (right.ceil().max(0.0) as usize).clamp(x0 + 1, width);
        let y0 = (top.max(0.0) as usize).min(height - 1);
        let y1 = (bottom.ceil().max(0.0) as usize).clamp(y0 + 1, height);
        let (low, high) = ink.extrema(x0, y0, x1, y1);
        if f64::from(high - low) >= ink.threshold && high != low {
            return false;
        }
        let difference = (ink.mean(x0, y0, x1, y1) - f64::from(ink.background)).abs();
        difference > 0.0 && difference >= ink.threshold
    }

    /// Whether the box `[x0, y0, x1, y1]` given in PDF user space is
    /// withheld, as [`PageReader::rect_covered`] decides.
    pub fn user_rect_covered(&self, rect: [f64; 4]) -> bool {
        let pages = self.document.pages();
        let Some(page) = pages.get(self.page_index) else {
            return true;
        };
        let affine = Affine::scale(self.scale) * page.initial_transform(true).to_kurbo();
        let b = pixel_box(affine, rect);
        self.rect_covered(
            b[0] / self.scale,
            b[1] / self.scale,
            b[2] / self.scale,
            b[3] / self.scale,
        )
    }

    /// The text of the page inside `bbox`, a box in the normalised 0..1000
    /// space of layout results, with withheld words removed from the text
    /// and listed in the tokens, and the visible links over the region.
    pub fn read(&self, bbox: [f64; 4]) -> LayerText {
        if bbox.iter().any(|value| !value.is_finite()) {
            return LayerText::default();
        }
        let left = bbox[0].min(bbox[2]) * self.width / BBOX_EXTENT;
        let right = bbox[0].max(bbox[2]) * self.width / BBOX_EXTENT;
        let top = bbox[1].min(bbox[3]) * self.height / BBOX_EXTENT;
        let bottom = bbox[1].max(bbox[3]) * self.height / BBOX_EXTENT;
        let inside: Vec<usize> = (0..self.glyphs.len())
            .filter(|&index| self.glyphs[index].within([left, top, right, bottom]))
            .collect();
        let mut upright: Vec<MarkedGlyph<'_>> = Vec::new();
        let mut turned: [Vec<(usize, bool, TextGlyph)>; 3] = Default::default();
        let mut rotated: Vec<MarkedWord> = Vec::new();
        let mut previous_rotated: Option<usize> = None;
        for &index in &inside {
            let glyph = &self.glyphs[index];
            let withheld = !glyph.space && self.covered(index);
            if self.upright[index] {
                upright.push(MarkedGlyph {
                    glyph,
                    withheld,
                    id: index,
                });
                continue;
            }
            if let Some(turns @ 1..=3) = self.quarters[index] {
                turned[usize::from(turns) - 1].push((
                    index,
                    withheld,
                    glyph.in_reading_frame(turns),
                ));
                continue;
            }
            if glyph.space {
                continue;
            }
            let text: String = expand_ligatures(&glyph.text)
                .chars()
                .filter(|c| *c != '\u{AD}')
                .collect();
            let joins = previous_rotated.is_some_and(|last| {
                let before = &self.glyphs[last];
                let (x, y) = before.center();
                let (cx, cy) = glyph.center();
                last + 1 == index && (cx - x).hypot(cy - y) <= glyph.em.max(before.em)
            });
            match rotated.last_mut() {
                Some(word) if joins => {
                    word.text.push_str(&text);
                    word.withheld |= withheld;
                    word.ids.push(index);
                }
                _ => rotated.push(MarkedWord {
                    text,
                    withheld,
                    ids: vec![index],
                }),
            }
            previous_rotated = Some(index);
        }
        let mut lines = join_hyphenated(marked_lines(upright));
        for group in &turned {
            let marked: Vec<MarkedGlyph<'_>> = group
                .iter()
                .map(|(id, withheld, glyph)| MarkedGlyph {
                    glyph,
                    withheld: *withheld,
                    id: *id,
                })
                .collect();
            lines.extend(join_hyphenated(marked_lines(marked)));
        }
        let words: Vec<&MarkedWord> = lines.iter().flat_map(|line| line.words.iter()).collect();
        let text = lines
            .iter()
            .map(|line| {
                line.words
                    .iter()
                    .filter(|word| !word.withheld)
                    .map(|word| word.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        let covered_chars: usize = words
            .iter()
            .copied()
            .chain(rotated.iter())
            .filter(|word| word.withheld)
            .map(|word| word.text.chars().count())
            .sum();
        let tokens = if covered_chars == 0 {
            Vec::new()
        } else {
            words
                .iter()
                .copied()
                .chain(rotated.iter())
                .map(|word| (word.text.clone(), word.withheld))
                .collect()
        };
        let links = self.region_links(&inside, &words);
        LayerText {
            text,
            tokens,
            covered_chars,
            links,
        }
    }

    /// The links of the page that cover visible glyphs of the region, in
    /// reading order. A link is left out when any glyph of the page under it
    /// is withheld or its box overlaps a `/Redact` annotation, so withheld
    /// text never reaches a link.
    fn region_links(&self, inside: &[usize], region_words: &[&MarkedWord]) -> Vec<LayerLink> {
        let mut found: Vec<(usize, LayerLink)> = Vec::new();
        for (rects, target) in &self.links {
            let under = |index: usize| {
                let (x, y) = self.glyphs[index].center();
                rects
                    .iter()
                    .any(|r| r[0] <= x && x <= r[2] && r[1] <= y && y <= r[3])
            };
            let members: Vec<usize> = inside.iter().copied().filter(|&i| under(i)).collect();
            if members.is_empty() {
                continue;
            }
            if rects.iter().any(|r| {
                self.redactions
                    .iter()
                    .any(|d| r[0] < d[2] && d[0] < r[2] && r[1] < d[3] && d[1] < r[3])
            }) {
                continue;
            }
            if (0..self.glyphs.len()).any(|i| !self.glyphs[i].space && under(i) && self.covered(i))
            {
                continue;
            }
            let member_glyphs: Vec<MarkedGlyph<'_>> = members
                .iter()
                .filter(|&&i| self.upright[i])
                .map(|&i| MarkedGlyph {
                    glyph: &self.glyphs[i],
                    withheld: false,
                    id: i,
                })
                .collect();
            let member_lines = join_hyphenated(marked_lines(member_glyphs));
            let words: Vec<&str> = member_lines
                .iter()
                .flat_map(|line| line.words.iter().map(|word| word.text.as_str()))
                .collect();
            if words.is_empty() {
                continue;
            }
            let first = member_lines
                .first()
                .and_then(|line| line.words.first())
                .and_then(|word| word.ids.first())
                .copied();
            let position = region_words
                .iter()
                .position(|word| first.is_some_and(|id| word.ids.contains(&id)))
                .unwrap_or(region_words.len());
            let before = region_words[..position]
                .iter()
                .map(|word| word.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let ordinal =
                words_pattern(&words).map_or(0, |pattern| pattern.find_iter(&before).count());
            found.push((
                position,
                LayerLink {
                    words: words.join(" "),
                    target: target.clone(),
                    ordinal,
                },
            ));
        }
        found.sort_by_key(|(position, _)| *position);
        found.into_iter().map(|(_, link)| link).collect()
    }
}

/// `glyph` with every coordinate multiplied by `factor`.
fn scaled(glyph: &TextGlyph, factor: f64) -> TextGlyph {
    TextGlyph {
        text: glyph.text.clone(),
        x0: glyph.x0 * factor,
        x1: glyph.x1 * factor,
        y0: glyph.y0 * factor,
        y1: glyph.y1 * factor,
        baseline: glyph.baseline * factor,
        em: glyph.em * factor,
        space: glyph.space,
        origin: glyph.origin * factor,
        advance_end: glyph.advance_end.map(|end| end * factor),
    }
}

/// The box `[x0, y0, x1, y1]` of PDF user space as a box of the raster
/// `affine` maps to.
fn pixel_box(affine: Affine, rect: [f64; 4]) -> [f64; 4] {
    let a = affine * Point::new(rect[0], rect[1]);
    let b = affine * Point::new(rect[2], rect[3]);
    [a.x.min(b.x), a.y.min(b.y), a.x.max(b.x), a.y.max(b.y)]
}

struct Annotations {
    redactions: Vec<[f64; 4]>,
    links: Vec<(Vec<[f64; 4]>, String)>,
}

/// The `/Redact` boxes and the http, https, mailto and in-document links of
/// `page`, as boxes of the raster `affine` maps to. A redaction takes both
/// its rectangle and each of its quadrilaterals; a link takes its
/// quadrilaterals, else its rectangle. Heights of in-document destinations
/// are in points times `box_scale`.
fn read_annotations(
    document: &Pdf,
    page: &Page<'_>,
    affine: Affine,
    box_scale: f64,
) -> Annotations {
    let mut out = Annotations {
        redactions: Vec::new(),
        links: Vec::new(),
    };
    let Some(annots) = page.raw().get::<Array<'_>>("Annots") else {
        return out;
    };
    for annot in annots.iter::<Dict<'_>>() {
        let Some(subtype) = annot.get::<Name<'_>>("Subtype") else {
            continue;
        };
        let is_redaction = subtype.as_str() == "Redact";
        if !is_redaction && subtype.as_str() != "Link" {
            continue;
        }
        let mut boxes: Vec<[f64; 4]> = Vec::new();
        if let Some(quads) = annot.get::<Vec<f64>>("QuadPoints") {
            for quad in quads.chunks_exact(8) {
                let xs = [quad[0], quad[2], quad[4], quad[6]];
                let ys = [quad[1], quad[3], quad[5], quad[7]];
                boxes.push(pixel_box(
                    affine,
                    [
                        xs.iter().copied().fold(f64::INFINITY, f64::min),
                        ys.iter().copied().fold(f64::INFINITY, f64::min),
                        xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                        ys.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    ],
                ));
            }
        }
        let rectangle = annot
            .get::<Rect>("Rect")
            .map(|r| pixel_box(affine, [r.x0, r.y0, r.x1, r.y1]));
        if is_redaction {
            out.redactions.extend(boxes);
            out.redactions.extend(rectangle);
            continue;
        }
        if boxes.is_empty() {
            boxes.extend(rectangle);
        }
        if boxes.is_empty() {
            continue;
        }
        if let Some(target) = link_target(document, &annot, box_scale) {
            out.links.push((boxes, target));
        }
    }
    out
}

/// The target of a link annotation: its http, https or mailto address, or
/// `page:N` (one-based) with `@Y` when it names a position, `Y` being the
/// distance from the top of that page in points times `box_scale`. `None`
/// for every other action and for a destination that names no page.
fn link_target(document: &Pdf, annot: &Dict<'_>, box_scale: f64) -> Option<String> {
    let action = annot.get::<Dict<'_>>("A");
    if let Some(action) = &action {
        match action.get::<Name<'_>>("S")?.as_str() {
            "URI" => return uri_target(&action.get::<PdfString<'_>>("URI")?),
            "GoTo" => {}
            _ => return None,
        }
    }
    let destination = annot
        .get::<Object<'_>>("Dest")
        .or_else(|| action.as_ref()?.get::<Object<'_>>("D"))?;
    destination_target(document, destination, box_scale)
}

/// `uri` as a link target: stripped, an http or https URL with a host or a
/// mailto address, holding no whitespace or control character.
fn uri_target(uri: &PdfString<'_>) -> Option<String> {
    let text: String = uri
        .as_bytes()
        .iter()
        .map(|byte| char::from(*byte))
        .collect();
    let target = text.trim();
    is_external_link_target(target).then(|| target.to_string())
}

fn destination_target(document: &Pdf, destination: Object<'_>, box_scale: f64) -> Option<String> {
    let array = match destination {
        Object::Array(array) => array,
        Object::Name(name) => named_destination(document, name.as_ref())?,
        Object::String(name) => named_destination(document, name.as_bytes())?,
        Object::Dict(dict) => dict.get::<Array<'_>>("D")?,
        _ => return None,
    };
    let page_ref = array.raw_iter().next()?.as_obj_ref()?;
    let identifier = ObjectIdentifier::from(page_ref);
    let pages = document.pages();
    let index = page_object_ids(document)
        .iter()
        .position(|id| *id == identifier)?;
    let target = format!("page:{}", index + 1);
    let items = array.raw_iter().skip(1).collect::<Vec<_>>();
    let direct = |index: usize| match items.get(index) {
        Some(MaybeRef::NotRef(object)) => Some(object.clone()),
        _ => None,
    };
    let is_xyz = matches!(direct(0), Some(Object::Name(mode)) if mode.as_str() == "XYZ");
    let top = direct(2).and_then(Object::into_f32);
    let (true, Some(top)) = (is_xyz, top) else {
        return Some(target);
    };
    let height = f64::from(pages.get(index)?.render_dimensions().1);
    Some(format!(
        "{target}@{:.1}",
        (height - f64::from(top)).max(0.0) * box_scale
    ))
}

/// The object identifiers of the document's page objects in page order,
/// read from the catalog's page tree; a node seen twice is not walked again.
fn page_object_ids(document: &Pdf) -> Vec<ObjectIdentifier> {
    fn walk(
        document: &Pdf,
        node: &Dict<'_>,
        visited: &mut HashSet<ObjectIdentifier>,
        found: &mut Vec<ObjectIdentifier>,
    ) {
        let Some(kids) = node.get::<Array<'_>>("Kids") else {
            return;
        };
        let xref = document.xref();
        for kid in kids.raw_iter() {
            let Some(reference) = kid.as_obj_ref() else {
                continue;
            };
            let id = ObjectIdentifier::from(reference);
            if !visited.insert(id) {
                continue;
            }
            let Some(dict) = xref.get::<Dict<'_>>(id) else {
                continue;
            };
            if dict.get::<Array<'_>>("Kids").is_some() {
                walk(document, &dict, visited, found);
            } else {
                found.push(id);
            }
        }
    }
    let xref = document.xref();
    let mut found = Vec::new();
    if let Some(pages) = xref
        .get::<Dict<'_>>(xref.root_id())
        .and_then(|catalog| catalog.get::<Dict<'_>>("Pages"))
    {
        walk(document, &pages, &mut HashSet::new(), &mut found);
    }
    found
}

/// The explicit destination array a destination name stands for, from the
/// catalog's `/Dests` dictionary or its `/Names` `/Dests` name tree.
fn named_destination<'a>(document: &'a Pdf, name: &[u8]) -> Option<Array<'a>> {
    let xref = document.xref();
    let catalog = xref.get::<Dict<'a>>(xref.root_id())?;
    let value = catalog
        .get::<Dict<'a>>("Dests")
        .and_then(|dests| dests.get::<Object<'a>>(name))
        .or_else(|| {
            let tree = catalog.get::<Dict<'a>>("Names")?.get::<Dict<'a>>("Dests")?;
            name_tree_lookup(&tree, name, &mut HashSet::new())
        })?;
    match value {
        Object::Array(array) => Some(array),
        Object::Dict(dict) => dict.get::<Array<'a>>("D"),
        _ => None,
    }
}

fn name_tree_lookup<'a>(
    node: &Dict<'a>,
    name: &[u8],
    visited: &mut HashSet<ObjectIdentifier>,
) -> Option<Object<'a>> {
    if let Some(id) = node.obj_id()
        && !visited.insert(id)
    {
        return None;
    }
    if let Some(names) = node.get::<Array<'a>>("Names") {
        let mut items = names.flex_iter();
        while let Some(key) = items.next::<PdfString<'a>>() {
            let value = items.next::<Object<'a>>()?;
            if key.as_bytes() == name {
                return Some(value);
            }
        }
    }
    let kids = node.get::<Array<'a>>("Kids")?;
    for kid in kids.iter::<Dict<'a>>() {
        if let Some(found) = name_tree_lookup(&kid, name, visited) {
            return Some(found);
        }
    }
    None
}

/// Whether `c` is a letter, digit or underscore.
fn is_word_character(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `word` without the characters that are not letters or digits at either
/// end.
fn word_core(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_alphanumeric())
        .to_string()
}

/// The end of the match of `cores` as whole words in order, with any
/// punctuation or space between them, that starts at `start`.
fn match_run_at(text: &str, start: usize, cores: &[String]) -> Option<usize> {
    if text[..start]
        .chars()
        .next_back()
        .is_some_and(is_word_character)
    {
        return None;
    }
    let mut position = start;
    for (number, core) in cores.iter().enumerate() {
        if number > 0 {
            while let Some(c) = text[position..].chars().next() {
                if is_word_character(c) {
                    break;
                }
                position += c.len_utf8();
            }
        }
        if !text[position..].starts_with(core.as_str()) {
            return None;
        }
        position += core.len();
    }
    if text[position..]
        .chars()
        .next()
        .is_some_and(is_word_character)
    {
        return None;
    }
    Some(position)
}

/// The leftmost match of `cores` in `text` that starts at or after byte
/// `from`.
fn find_run(text: &str, cores: &[String], from: usize) -> Option<(usize, usize)> {
    text.char_indices()
        .filter(|(start, _)| *start >= from)
        .find_map(|(start, _)| match_run_at(text, start, cores).map(|end| (start, end)))
}

/// `recognized` with every run of withheld words replaced by the redaction
/// marker. The tokens are walked in reading order against the recognised
/// text: each visible word moves the search position past its match, so a
/// withheld word is replaced at its own position even when the same word is
/// printed visibly earlier. A run found in the recognised text is replaced
/// there; a run absent after the search position is replaced at its earlier
/// occurrence when there is one, and otherwise (a filled box the recognition
/// never read) gets the marker after the recognised text of the word before
/// it, or at the start when it has none, or at the end when that word is not
/// found.
pub fn withhold_covered(recognized: &str, tokens: &[(String, bool)]) -> String {
    let mut text = recognized.to_string();
    let mut cursor = 0usize;
    let mut has_previous = false;
    let mut anchor: Option<usize> = None;
    let mut index = 0usize;
    while index < tokens.len() {
        let (word, withheld) = &tokens[index];
        if !*withheld {
            let core = word_core(word);
            if !core.is_empty() {
                has_previous = true;
                let visible = find_run(&text, &[core], cursor);
                anchor = visible.map(|(_, end)| end);
                if let Some((_, end)) = visible {
                    cursor = end;
                }
            }
            index += 1;
            continue;
        }
        let mut cores: Vec<String> = Vec::new();
        while index < tokens.len() && tokens[index].1 {
            let core = word_core(&tokens[index].0);
            if !core.is_empty() {
                cores.push(core);
            }
            index += 1;
        }
        let found = if cores.is_empty() {
            None
        } else {
            find_run(&text, &cores, cursor).or_else(|| find_run(&text, &cores, 0))
        };
        if let Some((start, end)) = found {
            text.replace_range(start..end, REDACTED_PLACEHOLDER);
            cursor = start + REDACTED_PLACEHOLDER.len();
            anchor = Some(cursor);
            has_previous = true;
        } else if let Some(at) = anchor {
            text.insert_str(at, &format!(" {REDACTED_PLACEHOLDER}"));
            cursor = at + 1 + REDACTED_PLACEHOLDER.len();
            anchor = Some(cursor);
        } else if !has_previous {
            text = if text.is_empty() {
                REDACTED_PLACEHOLDER.to_string()
            } else {
                format!("{REDACTED_PLACEHOLDER} {text}")
            };
            cursor = REDACTED_PLACEHOLDER.len();
            anchor = Some(cursor);
            has_previous = true;
        } else {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(REDACTED_PLACEHOLDER);
            cursor = text.len();
            anchor = Some(cursor);
        }
    }
    text
}

/// Whether the text layer of a region holds substantially more text than its
/// recognition: at least [`MIN_ALPHANUMERIC`] letters and digits, and more
/// than twice as many as the recognised text, within the region size limit.
pub fn recognition_lost_text(recognized: &str, layer: &str) -> bool {
    let layer_count = layer.chars().filter(|c| c.is_alphanumeric()).count();
    let recognized_count = recognized.chars().filter(|c| c.is_alphanumeric()).count();
    layer_count >= MIN_ALPHANUMERIC
        && recognized_count * 2 < layer_count
        && layer.len() <= MAX_REGION_TEXT_BYTES
}

/// The pattern matching `words` separated by any whitespace.
fn words_pattern(words: &[&str]) -> Option<Regex> {
    if words.is_empty() {
        return None;
    }
    let pattern = words
        .iter()
        .map(|word| regex::escape(word))
        .collect::<Vec<_>>()
        .join(r"\s+");
    Regex::new(&pattern).ok()
}

/// The link spans of `text` for the region's layer `links` in reading order.
/// A link whose words the text does not hold at its occurrence, after the
/// previous link, is left out.
/// Offsets are in characters.
pub fn link_spans(text: &str, links: &[LayerLink]) -> Vec<LinkSpan> {
    let mut spans = Vec::new();
    let mut cursor = 0usize;
    for link in links {
        let words: Vec<&str> = link.words.split_whitespace().collect();
        let Some(pattern) = words_pattern(&words) else {
            continue;
        };
        let Some(found) = pattern.find_iter(text).nth(link.ordinal) else {
            continue;
        };
        if found.start() < cursor {
            continue;
        }
        spans.push(LinkSpan {
            start: text[..found.start()].chars().count(),
            end: text[..found.end()].chars().count(),
            target: link.target.clone(),
        });
        cursor = found.end();
    }
    spans
}

/// The outcome of finishing one region.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionFinish {
    pub text: String,
    pub kind: String,
    pub links: Vec<LinkSpan>,
    /// Whether the text is the text layer's.
    pub substituted: bool,
    /// The characters of withheld words; the words themselves are not kept.
    pub covered_chars: usize,
}

/// Finishes one recognised region of `kind` with `text` and the box `bbox`
/// in the normalised 0..1000 space. Words the page withholds are replaced in
/// the text by the redaction marker, and nothing else of the layer is used;
/// otherwise a paragraph, heading, list item or caption whose layer text
/// holds substantially more text takes the layer text, a heading becoming a
/// paragraph when that text spans lines. Visible links become spans of the
/// final text. A figure is returned unchanged.
pub fn finish_region(
    reader: &PageReader<'_>,
    kind: &str,
    text: &str,
    bbox: [f64; 4],
) -> RegionFinish {
    let mut finish = RegionFinish {
        text: text.to_string(),
        kind: kind.to_string(),
        links: Vec::new(),
        substituted: false,
        covered_chars: 0,
    };
    if kind == "figure" {
        return finish;
    }
    let layer = reader.read(bbox);
    if layer.covered_chars > 0 {
        finish.covered_chars = layer.covered_chars;
        finish.text = withhold_covered(text, &layer.tokens);
    } else if TEXT_LAYER_KINDS.contains(&kind) && recognition_lost_text(text, &layer.text) {
        finish.text = layer.text.clone();
        finish.substituted = true;
        if kind == "heading" && finish.text.contains('\n') {
            finish.kind = "paragraph".to_string();
        }
    }
    finish.links = link_spans(&finish.text, &layer.links);
    finish
}

/// The counts of finishing the regions of one page.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageFinish {
    pub substituted: usize,
    pub covered_skipped: usize,
}

/// Finishes the assembled `regions` of one-based `page`, whose boxes are in
/// pixels of a page `page_px` (width, height) in size, with
/// [`finish_region`]. Regions of other pages, figures and regions without a
/// box are left alone, and a page furniture region may be withheld from but
/// never takes layer text. Applied after assembly, a region keeps the kind
/// and place assembly gave it.
pub fn finish_regions(
    reader: &PageReader<'_>,
    page: i64,
    page_px: (f64, f64),
    regions: &mut [Region],
) -> PageFinish {
    let mut total = PageFinish::default();
    let (width, height) = page_px;
    if !(width > 0.0 && height > 0.0) {
        return total;
    }
    for region in regions.iter_mut().filter(|region| region.page == page) {
        let Some(bbox) = region.bbox else {
            continue;
        };
        let normalised = [
            bbox[0] / width * BBOX_EXTENT,
            bbox[1] / height * BBOX_EXTENT,
            bbox[2] / width * BBOX_EXTENT,
            bbox[3] / height * BBOX_EXTENT,
        ];
        let finished = finish_region(reader, &region.kind, &region.text, normalised);
        total.covered_skipped += finished.covered_chars;
        if finished.substituted {
            total.substituted += 1;
        }
        region.text = finished.text;
        region.kind = finished.kind;
        if !finished.links.is_empty() {
            region.links = finished.links;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_layer;
    use image::RgbImage;

    const FONT: &str =
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>";

    fn pdf_with_objects(
        content: &str,
        extra_objects: &[String],
        page_extra: &str,
        resources_extra: &str,
    ) -> Vec<u8> {
        let mut objects: Vec<String> = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".into(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".into(),
            format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> {resources_extra} >> /Contents 5 0 R {page_extra} >>"
            ),
            FONT.into(),
            format!(
                "<< /Length {} >>\nstream\n{content}\nendstream",
                content.len()
            ),
        ];
        objects.extend(extra_objects.iter().cloned());
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (index, body) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend(format!("{} 0 obj\n{body}\nendobj\n", index + 1).bytes());
        }
        let xref = out.len();
        out.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).bytes());
        for offset in offsets {
            out.extend(format!("{offset:010} 00000 n \n").bytes());
        }
        out.extend(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .bytes(),
        );
        out
    }

    fn text_operations(lines: &[&str], render_mode: u8) -> String {
        let mut operations = vec![
            "BT".to_string(),
            "/F1 12 Tf".to_string(),
            format!("{render_mode} Tr"),
            "72 720 Td".to_string(),
            "20 TL".to_string(),
        ];
        operations.extend(lines.iter().map(|line| format!("({line}) Tj T*")));
        operations.push("ET".to_string());
        operations.join("\n")
    }

    fn redaction_lines() -> Vec<&'static str> {
        vec![
            "Account holder Zebulon Quillfeather lives nearby",
            "Contact record for the northern district office",
            "Balance reviewed during the quarterly meeting today",
        ]
    }

    fn rendered(width: f64, height: f64, scale: f64) -> RenderedPage {
        RenderedPage {
            image: RgbImage::new(
                (width * scale).ceil() as u32,
                (height * scale).ceil() as u32,
            ),
            width_pt: width,
            height_pt: height,
            scale,
            pixel_size: (width * scale, height * scale),
        }
    }

    fn finish(
        source: &[u8],
        kind: &str,
        text: &str,
        bbox: [f64; 4],
        size: (f64, f64),
        scale: f64,
    ) -> RegionFinish {
        let document = text_layer::open(source).expect("a readable pdf");
        let reader = PageReader::open(&document, 0, &rendered(size.0, size.1, scale), true)
            .expect("a readable page");
        finish_region(&reader, kind, text, bbox)
    }

    fn paragraph(source: &[u8], text: &str, bbox: [f64; 4]) -> RegionFinish {
        finish(source, "paragraph", text, bbox, (612.0, 792.0), 1.0)
    }

    const DEFAULT_BOX: [f64; 4] = [50.0, 50.0, 950.0, 300.0];

    /// The user-space boxes, one per printed line, that the `occurrence`-th
    /// match of `phrase` covers in the text of `lines`.
    fn phrase_rects(lines: &[&str], phrase: &str, occurrence: usize) -> Vec<[f64; 4]> {
        let source = pdf_with_objects(&text_operations(lines, 0), &[], "", "");
        let document = text_layer::open(&source).unwrap();
        let glyphs = page_glyphs(&document, 0, 1.0).unwrap().glyphs;
        let mut text = String::new();
        let mut owners: Vec<usize> = Vec::new();
        for (index, glyph) in glyphs.iter().enumerate() {
            if index > 0 && (glyph.baseline - glyphs[index - 1].baseline).abs() > glyph.em * 0.5 {
                text.push('\n');
                owners.push(index);
            }
            text.push_str(&glyph.text);
            owners.extend(std::iter::repeat_n(index, glyph.text.len()));
        }
        let words: Vec<&str> = phrase.split(' ').collect();
        let found = words_pattern(&words)
            .unwrap()
            .find_iter(&text)
            .nth(occurrence)
            .expect("the phrase is printed");
        let mut rows: Vec<(i64, [f64; 4])> = Vec::new();
        let mut seen = HashSet::new();
        for &index in &owners[found.start()..found.end()] {
            let glyph = &glyphs[index];
            if glyph.space || !seen.insert(index) {
                continue;
            }
            let (left, right) = (glyph.x0, glyph.x1);
            let (bottom, top) = (792.0 - glyph.y1, 792.0 - glyph.y0);
            let key = (bottom / 10.0).round() as i64;
            match rows.iter_mut().find(|(row, _)| *row == key) {
                Some((_, row)) => {
                    row[0] = row[0].min(left);
                    row[1] = row[1].min(bottom);
                    row[2] = row[2].max(right);
                    row[3] = row[3].max(top);
                }
                None => rows.push((key, [left, bottom, right, top])),
            }
        }
        rows.into_iter().map(|(_, row)| row).collect()
    }

    fn link_object(rects: &[[f64; 4]], target: &str) -> String {
        let outer = [
            rects.iter().map(|r| r[0]).fold(f64::INFINITY, f64::min),
            rects.iter().map(|r| r[1]).fold(f64::INFINITY, f64::min),
            rects.iter().map(|r| r[2]).fold(f64::NEG_INFINITY, f64::max),
            rects.iter().map(|r| r[3]).fold(f64::NEG_INFINITY, f64::max),
        ];
        let quads = rects
            .iter()
            .map(|r| {
                format!(
                    "{} {} {} {} {} {} {} {}",
                    r[0], r[3], r[2], r[3], r[0], r[1], r[2], r[1]
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let action = if target == "goto" {
            "/A << /S /GoTo /D [3 0 R /XYZ 72 400 0] >>".to_string()
        } else {
            format!("/A << /S /URI /URI ({target}) >>")
        };
        format!(
            "<< /Type /Annot /Subtype /Link /Border [0 0 0] /Rect [{} {} {} {}] /QuadPoints [{quads}] {action} >>",
            outer[0], outer[1], outer[2], outer[3]
        )
    }

    fn redact_object(r: [f64; 4]) -> String {
        format!(
            "<< /Type /Annot /Subtype /Redact /Rect [{} {} {} {}] /QuadPoints [{} {} {} {} {} {} {} {}] >>",
            r[0], r[1], r[2], r[3], r[0], r[3], r[2], r[3], r[0], r[1], r[2], r[1]
        )
    }

    fn linked_pdf(
        lines: &[&str],
        links: &[(Vec<[f64; 4]>, String)],
        redact: Option<[f64; 4]>,
        page_extra: &str,
    ) -> Vec<u8> {
        let mut annots: Vec<String> = links
            .iter()
            .map(|(rects, target)| link_object(rects, target))
            .collect();
        if let Some(r) = redact {
            annots.push(redact_object(r));
        }
        let references = (0..annots.len())
            .map(|n| format!("{} 0 R", 6 + n))
            .collect::<Vec<_>>()
            .join(" ");
        pdf_with_objects(
            &text_operations(lines, 0),
            &annots,
            &format!("/Annots [{references}] {page_extra}"),
            "",
        )
    }

    fn link_region(source: &[u8], content: &str, scale: f64) -> RegionFinish {
        finish(
            source,
            "paragraph",
            content,
            [50.0, 50.0, 950.0, 400.0],
            (612.0, 792.0),
            scale,
        )
    }

    fn spans(finish: &RegionFinish) -> Vec<(String, String)> {
        finish
            .links
            .iter()
            .map(|link| {
                let words: String = finish
                    .text
                    .chars()
                    .skip(link.start)
                    .take(link.end - link.start)
                    .collect();
                (words, link.target.clone())
            })
            .collect()
    }

    fn assert_withheld(finished: &RegionFinish) {
        assert!(!finished.text.contains("Zebulon"), "{}", finished.text);
        assert!(!finished.text.contains("Quillfeather"), "{}", finished.text);
        assert!(finished.text.contains(REDACTED_PLACEHOLDER));
        assert!(finished.text.contains("Contact record"));
        assert_eq!(finished.covered_chars, "ZebulonQuillfeather".len());
    }

    const OCR_WITHOUT_NAME: &str = "Account holder lives nearby\nContact record for the northern district office\nBalance reviewed during the quarterly meeting today";

    fn printed_entries() -> Vec<String> {
        (1..=8)
            .map(|n| format!("Section {n}. Synthetic entry title . . . . . {}", n * 3))
            .collect()
    }

    fn as_strs(lines: &[String]) -> Vec<&str> {
        lines.iter().map(String::as_str).collect()
    }

    #[test]
    fn text_layer_replaces_recognition_that_lost_most_of_a_region() {
        let lines = printed_entries();
        let source = pdf_with_objects(&text_operations(&as_strs(&lines), 0), &[], "", "");
        let heading = finish(
            &source,
            "heading",
            "Section 1. **Synthetic entry title** . . . . .",
            [100.0, 50.0, 900.0, 300.0],
            (612.0, 792.0),
            1.0,
        );
        assert_eq!(heading.kind, "paragraph");
        assert_eq!(heading.text, lines.join("\n"));
        assert!(heading.substituted);
        let outside = finish(
            &source,
            "paragraph",
            "A recognized paragraph outside the printed text.",
            [100.0, 500.0, 900.0, 700.0],
            (612.0, 792.0),
            1.0,
        );
        assert_eq!(
            outside.text,
            "A recognized paragraph outside the printed text."
        );
        assert!(!outside.substituted);
    }

    #[test]
    fn text_layer_keeps_complete_recognition() {
        let source = pdf_with_objects(
            &text_operations(
                &[
                    "Synthetic paragraph text that the recognizer read in full,",
                    "across two printed lines of the page.",
                ],
                0,
            ),
            &[],
            "",
            "",
        );
        let recognized = "Synthetic paragraph text that the **recognizer** read in full, across two printed lines of the page.";
        let finished = paragraph(&source, recognized, [100.0, 50.0, 900.0, 300.0]);
        assert_eq!(finished.text, recognized);
        assert!(!finished.substituted);
        assert_eq!(finished.covered_chars, 0);
    }

    #[test]
    fn a_link_annotation_becomes_a_span_with_the_right_offsets() {
        let lines = ["Read the Example Gazette online today."];
        let rects = phrase_rects(&lines, "Example Gazette", 0);
        let source = linked_pdf(
            &lines,
            &[(rects, "https://example.org/gazette".to_string())],
            None,
            "",
        );
        let finished = link_region(&source, lines[0], 1.0);
        assert_eq!(finished.links.len(), 1);
        assert_eq!((finished.links[0].start, finished.links[0].end), (9, 24));
        assert_eq!(
            spans(&finished),
            vec![(
                "Example Gazette".to_string(),
                "https://example.org/gazette".to_string()
            )]
        );
    }

    #[test]
    fn a_link_to_a_place_in_the_same_pdf_becomes_a_page_destination_span() {
        let lines = ["See the Appendix table for the figures."];
        let rects = phrase_rects(&lines, "Appendix table", 0);
        let source = linked_pdf(&lines, &[(rects, "goto".to_string())], None, "");
        let finished = link_region(&source, lines[0], 1.0);
        assert_eq!(
            spans(&finished),
            vec![("Appendix table".to_string(), "page:1@392.0".to_string())]
        );
        assert_eq!((finished.links[0].start, finished.links[0].end), (8, 22));
    }

    #[test]
    fn page_destination_height_is_measured_in_the_pixels_of_the_region_boxes() {
        let lines = ["See the Appendix table for the figures."];
        let rects = phrase_rects(&lines, "Appendix table", 0);
        let source = linked_pdf(&lines, &[(rects, "goto".to_string())], None, "");
        let finished = finish(
            &source,
            "paragraph",
            lines[0],
            [50.0, 50.0, 950.0, 400.0],
            (612.0, 792.0),
            2.0,
        );
        assert_eq!(
            spans(&finished),
            vec![("Appendix table".to_string(), "page:1@784.0".to_string())]
        );
    }

    #[test]
    fn only_http_https_and_mailto_link_targets_are_kept() {
        let lines = ["Open alpha beta gamma delta epsilon now."];
        let targets = [
            ("alpha", "javascript:alert(1)"),
            ("beta", "file:///private/notes"),
            ("gamma", "mailto:team@example.org"),
            ("delta", "http://example.org/d"),
            ("epsilon", "ftp://example.org/e"),
        ];
        let links: Vec<(Vec<[f64; 4]>, String)> = targets
            .iter()
            .map(|(word, target)| (phrase_rects(&lines, word, 0), (*target).to_string()))
            .collect();
        let finished = link_region(&linked_pdf(&lines, &links, None, ""), lines[0], 1.0);
        assert_eq!(
            spans(&finished),
            vec![
                ("gamma".to_string(), "mailto:team@example.org".to_string()),
                ("delta".to_string(), "http://example.org/d".to_string()),
            ]
        );
    }

    #[test]
    fn a_link_wrapped_over_two_lines_is_one_span() {
        let lines = ["Visit the Example", "Gazette for details."];
        let rects = phrase_rects(&lines, "Example Gazette", 0);
        assert_eq!(rects.len(), 2);
        let source = linked_pdf(
            &lines,
            &[(rects, "https://example.org/g".to_string())],
            None,
            "",
        );
        let finished = link_region(&source, &lines.join("\n"), 1.0);
        assert_eq!(
            spans(&finished),
            vec![(
                "Example\nGazette".to_string(),
                "https://example.org/g".to_string()
            )]
        );
    }

    #[test]
    fn a_link_follows_the_occurrence_of_its_words_that_the_page_links() {
        let lines = ["Click here to read, or click here instead."];
        let rects = phrase_rects(&lines, "here", 1);
        let source = linked_pdf(
            &lines,
            &[(rects, "https://example.org/second".to_string())],
            None,
            "",
        );
        let finished = link_region(&source, lines[0], 1.0);
        assert_eq!(finished.links.len(), 1);
        let second = lines[0].rfind("here").unwrap();
        assert_eq!(finished.links[0].start, second);
    }

    #[test]
    fn a_link_over_redacted_text_never_reaches_the_reply() {
        let lines = ["Reach Zebulon Quillfeather through the office."];
        let rects = phrase_rects(&lines, "Zebulon Quillfeather", 0);
        let source = linked_pdf(
            &lines,
            &[(rects.clone(), "https://example.org/person".to_string())],
            Some(rects[0]),
            "",
        );
        let finished = link_region(&source, lines[0], 1.0);
        assert!(finished.links.is_empty());
        assert!(!finished.text.contains("Zebulon"));
        assert!(!format!("{finished:?}").contains("Zebulon"));
    }

    #[test]
    fn a_link_over_redacted_text_on_a_rotated_cropped_page_never_reaches_the_reply() {
        let lines = ["Reach Zebulon Quillfeather through the office."];
        let rects = phrase_rects(&lines, "Zebulon Quillfeather", 0);
        let source = linked_pdf(
            &lines,
            &[(rects.clone(), "https://example.org/person".to_string())],
            Some(rects[0]),
            "/Rotate 90 /CropBox [20 30 612 792]",
        );
        let finished = finish(
            &source,
            "paragraph",
            lines[0],
            [0.0, 0.0, 1000.0, 1000.0],
            (762.0, 592.0),
            1.0,
        );
        assert!(finished.links.is_empty());
        assert!(!format!("{finished:?}").contains("Zebulon"));
    }

    #[test]
    fn a_link_whose_box_overlaps_a_redaction_is_left_out() {
        let lines = ["Reach Zebulon Quillfeather through the office."];
        let rects = phrase_rects(&lines, "Reach Zebulon Quillfeather", 0);
        let hidden = phrase_rects(&lines, "Zebulon Quillfeather", 0);
        let source = linked_pdf(
            &lines,
            &[(rects, "https://example.org/person".to_string())],
            Some(hidden[0]),
            "",
        );
        let finished = link_region(&source, "Reach", 1.0);
        assert!(finished.links.is_empty());
    }

    #[test]
    fn words_under_a_filled_box_are_replaced_by_the_redaction_marker() {
        let content = format!(
            "{}\n0 g 154 716 111 16 re f",
            text_operations(&redaction_lines(), 0)
        );
        let source = pdf_with_objects(&content, &[], "", "");
        let finished = paragraph(
            &source,
            "Account holder lives nearby\nContact record for the northern district office\nBalance reviewed during the quarterly meeting today",
            DEFAULT_BOX,
        );
        assert_withheld(&finished);
        assert!(!finished.substituted);
    }

    #[test]
    fn words_under_a_redact_annotation_are_replaced_by_the_redaction_marker() {
        let source = pdf_with_objects(
            &text_operations(&redaction_lines(), 0),
            &[redact_object([154.0, 716.0, 265.0, 732.0])],
            "/Annots [6 0 R]",
            "",
        );
        let finished = paragraph(&source, &redaction_lines().join("\n"), DEFAULT_BOX);
        assert!(!finished.text.contains("Zebulon"));
        assert!(!finished.text.contains("Quillfeather"));
        assert!(finished.text.contains(REDACTED_PLACEHOLDER));
        assert!(finished.text.contains("Account holder"));
        assert!(finished.text.contains("Balance reviewed"));
    }

    fn assert_covered_words_withheld(source: &[u8], size: (f64, f64)) {
        let finished = finish(
            source,
            "paragraph",
            OCR_WITHOUT_NAME,
            [0.0, 0.0, 1000.0, 1000.0],
            size,
            1.0,
        );
        assert_withheld(&finished);
    }

    #[test]
    fn words_under_a_filled_box_are_withheld_on_a_page_with_a_crop_box_origin() {
        let content = format!(
            "{}\n0 g 154 716 111 16 re f",
            text_operations(&redaction_lines(), 0)
        );
        let source = pdf_with_objects(&content, &[], "/CropBox [20 30 612 792]", "");
        assert_covered_words_withheld(&source, (592.0, 762.0));
    }

    #[test]
    fn words_under_a_filled_box_are_withheld_on_a_rotated_page() {
        let content = format!(
            "{}\n0 g 154 716 111 16 re f",
            text_operations(&redaction_lines(), 0)
        );
        for rotation in [90, 180, 270] {
            let source = pdf_with_objects(&content, &[], &format!("/Rotate {rotation}"), "");
            let size = if rotation == 180 {
                (612.0, 792.0)
            } else {
                (792.0, 612.0)
            };
            assert_covered_words_withheld(&source, size);
        }
    }

    #[test]
    fn words_under_a_redact_annotation_are_withheld_on_a_rotated_cropped_page() {
        let source = pdf_with_objects(
            &text_operations(&redaction_lines(), 0),
            &[redact_object([154.0, 716.0, 265.0, 732.0])],
            "/Annots [6 0 R] /Rotate 90 /CropBox [20 30 612 792]",
            "",
        );
        assert_covered_words_withheld(&source, (762.0, 592.0));
    }

    #[test]
    fn page_boxes_convert_between_user_space_and_the_rotated_page() {
        let source = pdf_with_objects(
            &text_operations(&redaction_lines(), 0),
            &[],
            "/Rotate 90 /CropBox [20 30 612 792]",
            "",
        );
        let document = text_layer::open(&source).unwrap();
        let pages = document.pages();
        let page = pages.first().unwrap();
        let affine = Affine::scale(1.0) * page.initial_transform(true).to_kurbo();
        assert_eq!(
            pixel_box(affine, [20.0, 30.0, 120.0, 80.0]),
            [0.0, 0.0, 50.0, 100.0]
        );
    }

    #[test]
    fn a_covered_word_is_replaced_at_its_own_position_not_its_earlier_twin() {
        let tokens = tokens(&[
            ("Pay", false),
            ("Ada", false),
            ("and", false),
            ("Ada", true),
            ("today", false),
        ]);
        assert_eq!(
            withhold_covered("Pay Ada and Ada today", &tokens),
            format!("Pay Ada and {REDACTED_PLACEHOLDER} today")
        );
    }

    #[test]
    fn a_covered_word_does_not_match_inside_a_longer_word() {
        let tokens = tokens(&[("party", false), ("art", true)]);
        assert_eq!(
            withhold_covered("party art", &tokens),
            format!("party {REDACTED_PLACEHOLDER}")
        );
    }

    #[test]
    fn a_covered_word_the_walk_has_passed_is_still_removed() {
        let tokens = tokens(&[("later", false), ("Ada", true)]);
        assert_eq!(
            withhold_covered("Ada wrote later", &tokens),
            format!("{REDACTED_PLACEHOLDER} wrote later")
        );
    }

    #[test]
    fn a_run_the_recognition_never_read_is_placed_after_the_word_before_it() {
        let after = tokens(&[("first", false), ("hidden", true), ("last", false)]);
        assert_eq!(
            withhold_covered("first last", &after),
            format!("first {REDACTED_PLACEHOLDER} last")
        );
        let start = tokens(&[("hidden", true), ("first", false)]);
        assert_eq!(
            withhold_covered("first", &start),
            format!("{REDACTED_PLACEHOLDER} first")
        );
        let end = tokens(&[("gone", false), ("hidden", true)]);
        assert_eq!(
            withhold_covered("other", &end),
            format!("other {REDACTED_PLACEHOLDER}")
        );
    }

    fn tokens(words: &[(&str, bool)]) -> Vec<(String, bool)> {
        words
            .iter()
            .map(|(word, withheld)| ((*word).to_string(), *withheld))
            .collect()
    }

    #[test]
    fn small_punctuation_and_ordinary_text_are_not_withheld() {
        let content = "BT /F1 12 Tf 0 Tr 72 720 Td (Ordinary line of text) Tj ET\nBT /F1 4 Tf 0 Tr 72 690 Td (. , . ; : . ,) Tj ET";
        let source = pdf_with_objects(content, &[], "", "");
        let finished = paragraph(
            &source,
            "Ordinary line of text\n. , . ; : . ,",
            [50.0, 50.0, 950.0, 150.0],
        );
        assert!(!finished.text.contains(REDACTED_PLACEHOLDER));
        assert_eq!(finished.covered_chars, 0);
    }

    #[test]
    fn invisible_text_over_an_image_of_the_same_text_is_still_substituted() {
        let lines = printed_entries();
        let mut data = String::new();
        for y in 0..200 {
            for x in 0..400 {
                data.push_str(if (x + y) % 2 == 0 { "00" } else { "FF" });
            }
        }
        data.push('>');
        let image = format!(
            "<< /Type /XObject /Subtype /Image /Width 400 /Height 200 /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /ASCIIHexDecode /Length {} >>\nstream\n{data}\nendstream",
            data.len()
        );
        let content = format!(
            "q 400 0 0 200 50 560 cm /Im1 Do Q\n{}",
            text_operations(&as_strs(&lines), 3)
        );
        let source = pdf_with_objects(&content, &[image], "", "/XObject << /Im1 6 0 R >>");
        let finished = paragraph(
            &source,
            "Section 1. Synthetic entry title",
            [50.0, 50.0, 950.0, 300.0],
        );
        assert_eq!(finished.text, lines.join("\n"));
        assert!(finished.substituted);
        assert_eq!(finished.covered_chars, 0);
    }

    #[test]
    fn invisible_text_over_blank_paper_never_reaches_the_reply() {
        let source = pdf_with_objects(&text_operations(&redaction_lines(), 3), &[], "", "");
        let finished = paragraph(&source, "Visible words only", DEFAULT_BOX);
        assert!(!finished.text.contains("Zebulon"));
        assert!(!finished.text.contains("Quillfeather"));
        assert!(!finished.substituted);
        assert!(finished.covered_chars > 0);
    }

    #[test]
    fn the_outcome_counts_skipped_characters_and_holds_no_covered_text() {
        let content = format!(
            "{}\n0 g 154 716 111 16 re f",
            text_operations(&redaction_lines(), 0)
        );
        let source = pdf_with_objects(&content, &[], "", "");
        let finished = paragraph(&source, "Account holder lives nearby", DEFAULT_BOX);
        assert_eq!(finished.covered_chars, "ZebulonQuillfeather".len());
        assert!(!format!("{finished:?}").contains("Zebulon"));
    }

    #[test]
    fn rotated_text_under_a_filled_box_is_withheld_whatever_its_direction() {
        let content = "BT /F1 12 Tf 0 1 -1 0 300 400 Tm (Zebulon Quillfeather) Tj ET\n0 g 280 395 30 130 re f";
        let source = pdf_with_objects(content, &[], "", "");
        let document = text_layer::open(&source).unwrap();
        let reader = PageReader::open(&document, 0, &rendered(612.0, 792.0, 1.0), true).unwrap();
        let layer = reader.read([0.0, 0.0, 1000.0, 1000.0]);
        assert!(layer.covered_chars > 0, "{layer:?}");
        assert!(layer.text.is_empty());
        let finished = finish_region(
            &reader,
            "paragraph",
            "Zebulon Quillfeather",
            [0.0, 0.0, 1000.0, 1000.0],
        );
        assert!(!finished.text.contains("Zebulon"));
        assert!(!finished.text.contains("Quillfeather"));
    }

    fn turned_lines(lines: &[&str], matrix: &str, origins: &[(f64, f64)]) -> String {
        lines
            .iter()
            .zip(origins)
            .map(|(line, (x, y))| format!("BT /F1 12 Tf {matrix} {x} {y} Tm ({line}) Tj ET"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn read_page(source: &[u8], size: (f64, f64), bbox: [f64; 4]) -> LayerText {
        let document = text_layer::open(source).unwrap();
        let reader = PageReader::open(&document, 0, &rendered(size.0, size.1, 1.0), true).unwrap();
        reader.read(bbox)
    }

    #[test]
    fn text_that_reads_upright_on_a_rotated_page_is_substituted_like_unrotated_text() {
        let lines = printed_entries();
        let origins: Vec<(f64, f64)> = (0..lines.len())
            .map(|n| (72.0 + 20.0 * n as f64, 72.0))
            .collect();
        let content = turned_lines(&as_strs(&lines), "0 1 -1 0", &origins);
        let source = pdf_with_objects(&content, &[], "/Rotate 90", "");
        let finished = finish(
            &source,
            "paragraph",
            "Section 1.",
            [0.0, 0.0, 1000.0, 1000.0],
            (792.0, 612.0),
            1.0,
        );
        assert_eq!(finished.text, lines.join("\n"));
        assert!(finished.substituted);
    }

    #[test]
    fn a_vertical_run_is_read_in_its_own_reading_order() {
        let content = turned_lines(
            &["second part of the note", "first part of the note"],
            "0 1 -1 0",
            &[(320.0, 200.0), (300.0, 200.0)],
        );
        let source = pdf_with_objects(&content, &[], "", "");
        let layer = read_page(&source, (612.0, 792.0), [400.0, 300.0, 600.0, 800.0]);
        assert_eq!(
            layer.text,
            "first part of the note\nsecond part of the note"
        );
        assert!(layer.tokens.is_empty());
        let down = turned_lines(&["reads downward"], "0 -1 1 0", &[(300.0, 400.0)]);
        let source = pdf_with_objects(&down, &[], "", "");
        let layer = read_page(&source, (612.0, 792.0), [400.0, 300.0, 600.0, 800.0]);
        assert_eq!(layer.text, "reads downward");
        let upside = turned_lines(&["upside down"], "-1 0 0 -1", &[(400.0, 400.0)]);
        let source = pdf_with_objects(&upside, &[], "", "");
        let layer = read_page(&source, (612.0, 792.0), [0.0, 0.0, 1000.0, 1000.0]);
        assert_eq!(layer.text, "upside down");
    }

    #[test]
    fn a_covered_word_of_a_vertical_run_is_withheld() {
        let content = format!(
            "{}\n0 g 285 395 20 50 re f",
            turned_lines(
                &["Zebulon Quillfeather lives nearby"],
                "0 1 -1 0",
                &[(300.0, 400.0)]
            )
        );
        let source = pdf_with_objects(&content, &[], "", "");
        let layer = read_page(&source, (612.0, 792.0), [0.0, 0.0, 1000.0, 1000.0]);
        assert_eq!(layer.text, "Quillfeather lives nearby");
        assert_eq!(layer.covered_chars, "Zebulon".len());
        assert!(layer.tokens.contains(&("Zebulon".to_string(), true)));
        assert!(layer.tokens.contains(&("lives".to_string(), false)));
    }

    #[test]
    fn a_run_at_no_right_angle_gives_tokens_but_no_layer_text() {
        let run = turned_lines(
            &["Zebulon Quillfeather"],
            "0.7071 0.7071 -0.7071 0.7071",
            &[(200.0, 200.0)],
        );
        let source = pdf_with_objects(&run, &[], "", "");
        let layer = read_page(&source, (612.0, 792.0), [0.0, 0.0, 1000.0, 1000.0]);
        assert!(layer.text.is_empty(), "{layer:?}");
        assert!(layer.tokens.is_empty());
        let covered = format!("{run}\n0 g 150 150 300 300 re f");
        let source = pdf_with_objects(&covered, &[], "", "");
        let layer = read_page(&source, (612.0, 792.0), [0.0, 0.0, 1000.0, 1000.0]);
        assert!(layer.text.is_empty());
        assert!(layer.covered_chars > 0, "{layer:?}");
        assert!(layer.tokens.iter().all(|(_, withheld)| *withheld));
    }

    #[test]
    fn a_figure_is_returned_unchanged_and_other_pages_are_left_alone() {
        let content = format!(
            "{}\n0 g 154 716 111 16 re f",
            text_operations(&redaction_lines(), 0)
        );
        let source = pdf_with_objects(&content, &[], "", "");
        let figure = finish(
            &source,
            "figure",
            "",
            [0.0, 0.0, 1000.0, 1000.0],
            (612.0, 792.0),
            1.0,
        );
        assert_eq!(figure.text, "");
        assert_eq!(figure.covered_chars, 0);
        let document = text_layer::open(&source).unwrap();
        let reader = PageReader::open(&document, 0, &rendered(612.0, 792.0, 1.0), true).unwrap();
        let mut regions = vec![
            region("p0001-r000001", 1, OCR_WITHOUT_NAME),
            region("p0002-r000001", 2, OCR_WITHOUT_NAME),
        ];
        let counts = finish_regions(&reader, 1, (612.0, 792.0), &mut regions);
        assert_eq!(
            counts,
            PageFinish {
                substituted: 0,
                covered_skipped: "ZebulonQuillfeather".len()
            }
        );
        assert!(regions[0].text.contains(REDACTED_PLACEHOLDER));
        assert_eq!(regions[1].text, OCR_WITHOUT_NAME);
    }

    fn region(id: &str, page: i64, text: &str) -> Region {
        Region {
            id: id.into(),
            page,
            kind: "paragraph".into(),
            text: text.into(),
            bbox: Some([0.0, 0.0, 612.0, 792.0]),
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

    #[test]
    fn a_box_under_a_filled_box_or_redaction_is_covered_and_a_clear_one_is_not() {
        let content = "0 g 100 600 100 20 re f";
        let filled = pdf_with_objects(content, &[], "", "");
        let document = text_layer::open(&filled).unwrap();
        let reader = PageReader::open(&document, 0, &rendered(612.0, 792.0, 1.0), false).unwrap();
        assert!(reader.user_rect_covered([120.0, 602.0, 180.0, 618.0]));
        assert!(!reader.user_rect_covered([300.0, 300.0, 360.0, 320.0]));
        assert!(reader.user_rect_covered([-500.0, -500.0, -400.0, -400.0]));

        let redacted = pdf_with_objects(
            "",
            &[redact_object([100.0, 600.0, 200.0, 620.0])],
            "/Annots [6 0 R]",
            "",
        );
        let document = text_layer::open(&redacted).unwrap();
        let reader = PageReader::open(&document, 0, &rendered(612.0, 792.0, 1.0), false).unwrap();
        assert!(reader.user_rect_covered([120.0, 602.0, 180.0, 618.0]));
        assert!(!reader.user_rect_covered([300.0, 300.0, 360.0, 320.0]));
    }

    #[test]
    fn only_a_pdf_has_a_text_layer() {
        assert!(text_layer::open(b"\x89PNG\r\n\x1a\n").is_none());
    }

    #[test]
    fn link_spans_skip_words_the_text_does_not_hold_in_order() {
        let links = [
            LayerLink {
                words: "beta gamma".into(),
                target: "https://example.org/b".into(),
                ordinal: 0,
            },
            LayerLink {
                words: "alpha".into(),
                target: "https://example.org/a".into(),
                ordinal: 0,
            },
            LayerLink {
                words: "missing".into(),
                target: "https://example.org/m".into(),
                ordinal: 0,
            },
        ];
        let spans = link_spans("alpha beta\ngamma", &links);
        assert_eq!(spans.len(), 1);
        assert_eq!((spans[0].start, spans[0].end), (6, 16));
    }

    #[test]
    fn recognition_lost_text_needs_forty_letters_and_twice_the_recognition() {
        let layer = "a".repeat(40);
        assert!(recognition_lost_text("", &layer));
        assert!(recognition_lost_text(&"b".repeat(19), &layer));
        assert!(!recognition_lost_text(&"b".repeat(20), &layer));
        assert!(!recognition_lost_text("", &"a".repeat(39)));
    }
}
