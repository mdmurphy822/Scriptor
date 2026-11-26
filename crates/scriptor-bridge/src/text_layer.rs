//! The text layer of a PDF page as positioned unicode glyphs in the pixel
//! space of the page raster, and the grouping of those glyphs into lines and
//! words in reading order.

use hayro::hayro_interpret::font::{Glyph, GlyphRun};
use hayro::hayro_interpret::hayro_cmap::BfString;
use hayro::hayro_interpret::hayro_syntax::Pdf;
use hayro::hayro_interpret::{
    BlendMode, ClipPath, Context, Device, DrawMode, DrawProps, Image, ImageDrawProps,
    InterpreterCache, InterpreterSettings, SoftMask, TransformExt, interpret_page,
};
use hayro::kurbo::{Affine, BezPath, Rect, Shape};
use std::collections::HashSet;
use std::sync::Arc;

/// The largest gap between the inked edges of two neighbouring glyphs, in
/// ems, that still belongs to one word.
const WORD_GAP_EMS: f64 = 0.27;

/// The largest gap, in ems, between where one glyph's advance ends and where
/// the next glyph is placed that still belongs to one word.
const ADVANCE_GAP_EMS: f64 = 0.15;

/// The unicode of an unmapped glyph.
const UNMAPPED: char = '\u{FFFD}';

/// The soft hyphen character.
const SOFT_HYPHEN: char = '\u{AD}';

/// One visible glyph of a page with its unicode text and its extent in pixels
/// of the page raster (origin at the top left, y growing downward).
#[derive(Clone, Debug, PartialEq)]
pub struct TextGlyph {
    pub text: String,
    pub x0: f64,
    pub x1: f64,
    pub y0: f64,
    pub y1: f64,
    pub baseline: f64,
    pub em: f64,
    pub space: bool,
    /// Where the glyph is placed on its baseline.
    pub origin: f64,
    /// Where the glyph's advance ends on its baseline, when the font gives
    /// the advance. For a glyph that is not upright it is the position along
    /// the reading axis of the glyph's reading frame (see
    /// [`TextGlyph::in_reading_frame`]).
    pub advance_end: Option<f64>,
}

/// Fraction of a glyph's em that its line band extends above the baseline.
const LINE_ASCENT_EM: f64 = 0.75;
/// Fraction of a glyph's em that its line band extends below the baseline.
const LINE_DESCENT_EM: f64 = 0.25;

impl TextGlyph {
    /// Whether the glyph belongs to the region `bbox` (`x0, y0, x1, y1` in
    /// either corner order): its ink center lies within the box's x range and
    /// its line band, from `LINE_ASCENT_EM` of the em above the baseline to
    /// `LINE_DESCENT_EM` below it, overlaps the box vertically by at least half
    /// the band's height. Glyphs sharing a baseline and size get the same
    /// vertical decision. A glyph whose ink center lies outside its line band
    /// (turned text) is decided by its ink center alone.
    pub(crate) fn within(&self, bbox: [f64; 4]) -> bool {
        let (cx, cy) = self.center();
        if cx < bbox[0].min(bbox[2]) || cx > bbox[0].max(bbox[2]) {
            return false;
        }
        let (top, bottom) = (bbox[1].min(bbox[3]), bbox[1].max(bbox[3]));
        let band_top = self.baseline - LINE_ASCENT_EM * self.em;
        let band_bottom = self.baseline + LINE_DESCENT_EM * self.em;
        let height = band_bottom - band_top;
        if height <= 0.0 || height.is_nan() || cy < band_top || cy > band_bottom {
            return cy >= top && cy <= bottom;
        }
        let overlap = band_bottom.min(bottom) - band_top.max(top);
        overlap >= height / 2.0
    }

    pub(crate) fn center(&self) -> (f64, f64) {
        ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)
    }

    /// The glyph in the frame in which a text run turned by `quarter`
    /// counter-clockwise quarter turns reads upright and left to right:
    /// extent, baseline and origin are mapped into that frame.
    pub(crate) fn in_reading_frame(&self, quarter: u8) -> TextGlyph {
        let map = |x: f64, y: f64| reading_point(quarter, x, y);
        let (ax, ay) = map(self.x0, self.y0);
        let (bx, by) = map(self.x1, self.y1);
        let (origin, baseline) = map(self.origin, self.baseline);
        TextGlyph {
            text: self.text.clone(),
            x0: ax.min(bx),
            x1: ax.max(bx),
            y0: ay.min(by),
            y1: ay.max(by),
            baseline,
            em: self.em,
            space: self.space,
            origin,
            advance_end: self.advance_end,
        }
    }
}

/// The point `(x, y)` of the page raster in the reading frame of a text run
/// turned by `quarter` counter-clockwise quarter turns.
fn reading_point(quarter: u8, x: f64, y: f64) -> (f64, f64) {
    match quarter % 4 {
        1 => (-y, x),
        2 => (-x, -y),
        3 => (y, -x),
        _ => (x, y),
    }
}

/// The counter-clockwise quarter turns of a glyph placed by the linear part
/// `[a, b, c, d]` of its transform in the raster (y down), when its baseline
/// runs along a raster axis and the glyph is not mirrored; `None` otherwise.
fn quarter_turns(a: f64, b: f64, c: f64, d: f64) -> Option<u8> {
    if c * b - d * a <= 0.0 {
        return None;
    }
    if b.abs() <= a.abs() * 0.1 {
        Some(if a > 0.0 { 0 } else { 2 })
    } else if a.abs() <= b.abs() * 0.1 {
        Some(if b < 0.0 { 1 } else { 3 })
    } else {
        None
    }
}

/// The parsed PDF `bytes`; `None` when they are not a readable PDF.
pub fn open(bytes: &[u8]) -> Option<Pdf> {
    Pdf::new(Arc::new(bytes.to_vec())).ok()
}

/// Collects the glyphs a page draws, counting those drawn invisibly. With
/// `keep_all` it also keeps invisible glyphs and glyphs that are not
/// upright, recording which glyphs are upright.
struct Collector {
    scale: f64,
    keep_all: bool,
    glyphs: Vec<TextGlyph>,
    upright: Vec<bool>,
    quarters: Vec<Option<u8>>,
    seen: HashSet<(String, i64, i64, i64)>,
    invisible: usize,
}

impl<'a> Device<'a> for Collector {
    fn draw_path(&mut self, _: &BezPath, _: DrawProps<'a>, _: &DrawMode) {}
    fn push_clip_path(&mut self, _: &ClipPath) {}
    fn push_transparency_group(&mut self, _: f32, _: Option<SoftMask<'a>>, _: BlendMode) {}
    fn draw_glyph_run(&mut self, run: &GlyphRun<'_, 'a>, props: DrawProps<'a>, mode: &DrawMode) {
        let invisible = matches!(mode, DrawMode::Invisible);
        for glyph in run.glyphs() {
            if invisible {
                self.invisible += 1;
                if !self.keep_all {
                    continue;
                }
            }
            let placed = Affine::scale(self.scale) * props.transform * glyph.transform();
            let [a, b, c, d, e, f] = placed.as_coeffs();
            let unit = a.hypot(b);
            if unit < 1e-9 {
                continue;
            }
            let upright = a > 0.0 && b.abs() <= a * 0.1 && d < 0.0;
            if !upright && !self.keep_all {
                continue;
            }
            let quarter = if upright {
                Some(0)
            } else {
                quarter_turns(a, b, c, d)
            };
            let em = unit * 1000.0;
            let text = match glyph.as_unicode() {
                Some(BfString::Char(c)) => c.to_string(),
                Some(BfString::String(s)) => s,
                None => UNMAPPED.to_string(),
            };
            if text.chars().any(|c| c.is_control() && !c.is_whitespace()) {
                continue;
            }
            let space = text.chars().all(char::is_whitespace);
            let ink = match &**glyph {
                Glyph::Outline(outline) => {
                    let path = placed * outline.outline();
                    (!path.elements().is_empty()).then(|| path.bounding_box())
                }
                Glyph::Type3(_) => None,
            };
            let advance_end = match &**glyph {
                Glyph::Outline(outline) => outline.advance_width(),
                Glyph::Type3(_) => None,
            }
            .filter(|advance| advance.is_finite() && *advance > 0.0)
            .map(|advance| match quarter {
                Some(turns) if turns != 0 => {
                    reading_point(
                        turns,
                        e + f64::from(advance) * a,
                        f + f64::from(advance) * b,
                    )
                    .0
                }
                _ => e + f64::from(advance) * unit,
            });
            let (x0, x1, y0, y1) = match ink {
                Some(Rect { x0, y0, x1, y1 }) if !space => (x0, x1, y0, y1),
                _ if upright => (e, e + em * if space { 0.25 } else { 0.5 }, f - em * 0.7, f),
                _ => (e - em * 0.5, e + em * 0.5, f - em * 0.5, f + em * 0.5),
            };
            let quantum = (em * 0.1).max(0.5);
            let key = (
                text.clone(),
                (e / quantum).round() as i64,
                (f / quantum).round() as i64,
                (em * 10.0).round() as i64,
            );
            if !self.seen.insert(key) {
                continue;
            }
            self.glyphs.push(TextGlyph {
                text,
                x0,
                x1,
                y0,
                y1,
                baseline: f,
                em,
                space,
                origin: e,
                advance_end,
            });
            self.upright.push(upright);
            self.quarters.push(quarter);
        }
    }
    fn draw_image(&mut self, _: Image<'a, '_>, _: ImageDrawProps<'a>) {}
    fn pop_clip(&mut self) {}
    fn pop_transparency_group(&mut self) {}
}

/// Runs the glyph collector over zero-based `page_index`; `None` when the
/// page is unreadable.
fn collect(document: &Pdf, page_index: usize, scale: f64, keep_all: bool) -> Option<Collector> {
    let pages = document.pages();
    let page = pages.get(page_index)?;
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (width, height) = page.render_dimensions();
        let mut context = Context::new(
            page.initial_transform(true).to_kurbo(),
            Rect::new(0.0, 0.0, f64::from(width), f64::from(height)),
            &InterpreterCache::new(),
            page.xref(),
            InterpreterSettings::default(),
        );
        let mut collector = Collector {
            scale,
            keep_all,
            glyphs: Vec::new(),
            upright: Vec::new(),
            quarters: Vec::new(),
            seen: HashSet::new(),
            invisible: 0,
        };
        interpret_page(page, &mut context, &mut collector);
        collector
    }))
    .ok()
}

/// Every glyph a page draws, in drawing order, with whether each is upright.
pub(crate) struct PageGlyphs {
    pub glyphs: Vec<TextGlyph>,
    /// Whether each glyph is placed upright and left to right.
    pub upright: Vec<bool>,
    /// The counter-clockwise quarter turns of each glyph's baseline in the
    /// rendered page (0 for upright text); `None` when it runs along no
    /// right angle or the glyph is mirrored.
    pub quarters: Vec<Option<u8>>,
}

/// All glyphs of zero-based `page_index` in pixels of the page rendered at
/// `scale` pixels per point, including invisible and non-upright glyphs.
/// `None` when the page is unreadable.
pub(crate) fn page_glyphs(document: &Pdf, page_index: usize, scale: f64) -> Option<PageGlyphs> {
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let collected = collect(document, page_index, scale, true)?;
    Some(PageGlyphs {
        glyphs: collected.glyphs,
        upright: collected.upright,
        quarters: collected.quarters,
    })
}

/// A glyph of a region with whether its page withholds it and an identity
/// its caller assigns.
#[derive(Clone, Copy)]
pub(crate) struct MarkedGlyph<'a> {
    pub glyph: &'a TextGlyph,
    pub withheld: bool,
    pub id: usize,
}

/// A word of a line: its text, whether any of its glyphs is withheld, and
/// the identities of its glyphs.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MarkedWord {
    pub text: String,
    pub withheld: bool,
    pub ids: Vec<usize>,
}

/// A line of words in reading order and whether it ends in a soft hyphen.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MarkedLine {
    pub words: Vec<MarkedWord>,
    pub soft_hyphen_end: bool,
}

/// A run of glyphs sharing a baseline.
struct Line<'a> {
    glyphs: Vec<MarkedGlyph<'a>>,
    baseline_sum: f64,
    em_sum: f64,
}

impl<'a> Line<'a> {
    fn new(marked: MarkedGlyph<'a>) -> Self {
        Self {
            baseline_sum: marked.glyph.baseline,
            em_sum: marked.glyph.em,
            glyphs: vec![marked],
        }
    }

    fn count(&self) -> f64 {
        self.glyphs.len() as f64
    }

    fn baseline(&self) -> f64 {
        self.baseline_sum / self.count()
    }

    fn accepts(&self, glyph: &TextGlyph) -> bool {
        let em = (self.em_sum / self.count()).min(glyph.em);
        (glyph.baseline - self.baseline()).abs() <= em * 0.5
    }

    fn push(&mut self, marked: MarkedGlyph<'a>) {
        self.baseline_sum += marked.glyph.baseline;
        self.em_sum += marked.glyph.em;
        self.glyphs.push(marked);
    }
}

/// The words of `glyphs` as lines in reading order, each word carrying
/// whether any of its glyphs is withheld.
pub(crate) fn marked_lines(mut glyphs: Vec<MarkedGlyph<'_>>) -> Vec<MarkedLine> {
    glyphs.sort_by(|a, b| a.glyph.baseline.total_cmp(&b.glyph.baseline));
    let mut lines: Vec<Line<'_>> = Vec::new();
    for marked in glyphs {
        match lines.last_mut() {
            Some(line) if line.accepts(marked.glyph) => line.push(marked),
            _ => lines.push(Line::new(marked)),
        }
    }
    lines.sort_by(|a, b| a.baseline().total_cmp(&b.baseline()));
    let mut out = Vec::new();
    for mut line in lines {
        line.glyphs
            .sort_by(|a, b| a.glyph.center().0.total_cmp(&b.glyph.center().0));
        let (words, soft_hyphen_end) = render_line(&line.glyphs);
        if words.is_empty() {
            continue;
        }
        out.push(MarkedLine {
            words,
            soft_hyphen_end,
        });
    }
    out
}

/// `lines` with every line that ends in a soft hyphen joined to the line
/// after it, the hyphen's word and the next line's first word becoming one.
pub(crate) fn join_hyphenated(lines: Vec<MarkedLine>) -> Vec<MarkedLine> {
    let mut out: Vec<MarkedLine> = Vec::new();
    let mut joined = false;
    for line in lines {
        match out.last_mut() {
            Some(last) if joined => {
                let mut words = line.words.into_iter();
                if let Some(first) = words.next() {
                    match last.words.last_mut() {
                        Some(tail) => {
                            tail.text.push_str(&first.text);
                            tail.withheld |= first.withheld;
                            tail.ids.extend(first.ids);
                        }
                        None => last.words.push(first),
                    }
                }
                last.words.extend(words);
                last.soft_hyphen_end = line.soft_hyphen_end;
            }
            _ => out.push(line),
        }
        joined = out.last().is_some_and(|last| last.soft_hyphen_end);
    }
    out
}

/// The words of one left-to-right line and whether it ends in a soft hyphen.
fn render_line(glyphs: &[MarkedGlyph<'_>]) -> (Vec<MarkedWord>, bool) {
    let mut words: Vec<MarkedWord> = Vec::new();
    let mut pending_space = false;
    let mut previous: Option<&TextGlyph> = None;
    let mut ends_soft = false;
    for marked in glyphs {
        let glyph = marked.glyph;
        if glyph.space {
            pending_space = true;
            previous = Some(glyph);
            continue;
        }
        let text: String = expand_ligatures(&glyph.text)
            .chars()
            .filter(|c| *c != SOFT_HYPHEN)
            .collect();
        if text.is_empty() {
            ends_soft = true;
            continue;
        }
        ends_soft = false;
        if let Some(before) = previous
            && !before.space
            && word_gap(before, glyph)
        {
            pending_space = true;
        }
        if pending_space || words.is_empty() {
            words.push(MarkedWord {
                text: String::new(),
                withheld: false,
                ids: Vec::new(),
            });
        }
        pending_space = false;
        if let Some(word) = words.last_mut() {
            word.text.push_str(&text);
            word.withheld |= marked.withheld;
            word.ids.push(marked.id);
        }
        previous = Some(glyph);
    }
    for word in &mut words {
        word.text = word.text.trim().to_string();
    }
    words.retain(|word| !word.text.is_empty());
    (words, ends_soft)
}

/// Whether a word break falls between neighbouring glyphs `before` and
/// `after`: the gap from the end of `before`'s advance to `after`'s placement
/// when the font gives the advance, otherwise the gap between their inked
/// edges, against a word gap of their size.
fn word_gap(before: &TextGlyph, after: &TextGlyph) -> bool {
    let em = before.em.max(after.em);
    match before.advance_end {
        Some(end) => after.origin - end > ADVANCE_GAP_EMS * em,
        None => after.x0 - before.x1 > WORD_GAP_EMS * em,
    }
}

/// `text` with the Latin ligature characters replaced by their letters.
pub fn expand_ligatures(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\u{FB00}' => out.push_str("ff"),
            '\u{FB01}' => out.push_str("fi"),
            '\u{FB02}' => out.push_str("fl"),
            '\u{FB03}' => out.push_str("ffi"),
            '\u{FB04}' => out.push_str("ffl"),
            '\u{FB05}' | '\u{FB06}' => out.push_str("st"),
            other => out.push(other),
        }
    }
    out
}
