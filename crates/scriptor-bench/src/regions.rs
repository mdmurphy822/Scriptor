//! Region sets: crops with a recognition task each, either rendered
//! procedurally (synthetic) or loaded from a directory written by
//! `tools/bench/render_regions.sh`.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use ab_glyph::{Font, FontRef, PxScale, ScaleFont, point};
use image::codecs::jpeg::JpegEncoder;
use image::{ExtendedColorType, Rgb, RgbImage};
use serde::{Deserialize, Serialize};

use crate::{BenchError, Result};

/// The synthetic region manifest.
pub const SYNTHETIC_MANIFEST: &str =
    include_str!("../../../tests/fixtures/bench/synthetic-manifest.json");

/// The bundled font (SIL OFL 1.1) used to render synthetic crops.
pub const BUNDLED_FONT: &[u8] = include_bytes!("../../../tests/fixtures/bench/Lato-Medium.ttf");

/// The JPEG quality of every crop.
pub const JPEG_QUALITY: u8 = 75;

/// The pixel edge of one merged vision token.
pub const IMAGE_FACTOR: u32 = 28;

/// What a region is recognised as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    Text,
    Table,
    Formula,
}

impl Task {
    /// The prompt text of the task.
    pub const fn prompt(self) -> &'static str {
        match self {
            Self::Text => "Text Recognition:",
            Self::Table => "Table Recognition:",
            Self::Formula => "Formula Recognition:",
        }
    }

    /// The task named `text`, `table` or `formula`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "text" => Some(Self::Text),
            "table" => Some(Self::Table),
            "formula" => Some(Self::Formula),
            _ => None,
        }
    }

    /// The lower-case task name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Table => "table",
            Self::Formula => "formula",
        }
    }
}

/// How a synthetic region is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Lines,
    Table,
    Formula,
}

/// One entry of the synthetic manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyntheticSpec {
    pub id: String,
    pub kind: Kind,
    pub task: Task,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
}

#[derive(Deserialize)]
struct SyntheticFile {
    regions: Vec<SyntheticSpec>,
}

/// Parses a synthetic manifest.
pub fn parse_synthetic_manifest(json: &str) -> Result<Vec<SyntheticSpec>> {
    Ok(serde_json::from_str::<SyntheticFile>(json)?.regions)
}

/// The approximate vision-token count of a crop: its edges rounded to the nearest
/// multiple of 28, at least one cell each, multiplied.
pub fn approx_image_tokens(width: u32, height: u32) -> u64 {
    let cells = |v: u32| ((v as f64 / IMAGE_FACTOR as f64).round() as u64).max(1);
    cells(width) * cells(height)
}

/// One crop with the tasks it is posted with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub tasks: Vec<Task>,
    /// The JPEG bytes of the crop.
    pub jpeg: Vec<u8>,
}

impl Region {
    /// The `data:image/jpeg;base64,` URL of the crop.
    pub fn data_url(&self) -> String {
        use base64::Engine as _;
        format!(
            "data:image/jpeg;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&self.jpeg)
        )
    }
}

/// One (region, task) request of a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Case {
    pub region: String,
    pub task: Task,
    pub data_url: String,
}

/// Expands regions into one case per (region, task); `tasks` overrides the tasks of
/// every region when given.
pub fn cases(regions: &[Region], tasks: Option<&[Task]>) -> Vec<Case> {
    let mut out = Vec::new();
    for r in regions {
        let url = r.data_url();
        for &task in tasks.unwrap_or(&r.tasks) {
            out.push(Case {
                region: r.id.clone(),
                task,
                data_url: url.clone(),
            });
        }
    }
    out
}

/// `count` cases drawn deterministically from `regions` with the task mix the
/// bridge produces (mostly text, a few formulas and tables).
pub fn bridge_mix(regions: &[Region], count: usize) -> Vec<Case> {
    if regions.is_empty() {
        return Vec::new();
    }
    let urls: Vec<String> = regions.iter().map(Region::data_url).collect();
    let mut rng = SplitMix64(0x9E37_79B9_7F4A_7C15);
    (0..count)
        .map(|i| {
            let idx = i % regions.len();
            let roll = rng.next() % 1000;
            let task = if roll < 34 {
                Task::Formula
            } else if roll < 60 {
                Task::Table
            } else {
                Task::Text
            };
            Case {
                region: regions[idx].id.clone(),
                task,
                data_url: urls[idx].clone(),
            }
        })
        .collect()
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const WORDS: &[&str] = &[
    "the", "model", "reads", "page", "region", "table", "value", "total", "figure", "section",
    "result", "method", "update", "input", "output", "count", "rate", "level", "signal", "design",
    "report", "sample", "series", "window", "memory", "kernel", "vision", "token", "batch",
    "length", "margin", "scale", "layer", "index", "order", "range", "limit", "state", "event",
    "frame", "field", "group", "image", "block", "query", "route", "stage", "trace", "unit",
    "world",
];

const FORMULAS: &[&str] = &[
    "E = m c^2 + (a_i - b_j) / 2",
    "f(x) = sum_{k=0}^{n} a_k x^k",
    "y = W x + b, W in R^{m x n}",
    "p(x | y) = p(y | x) p(x) / p(y)",
    "d/dt x(t) = A x(t) + B u(t)",
    "L = - sum_i y_i log(p_i) + lambda ||w||^2",
];

fn blend(img: &mut RgbImage, x: i64, y: i64, coverage: f32, ink: u8) {
    if x < 0 || y < 0 || x >= img.width() as i64 || y >= img.height() as i64 {
        return;
    }
    let p = img.get_pixel_mut(x as u32, y as u32);
    for c in p.0.iter_mut() {
        let base = *c as f32;
        *c = (base + (ink as f32 - base) * coverage.clamp(0.0, 1.0)).round() as u8;
    }
}

/// Draws `text` with its left edge at `x` and its top at `y`; returns the
/// advance in pixels. Pixels outside the image are clipped.
fn draw_text(img: &mut RgbImage, font: &FontRef<'_>, px: f32, x: f32, y: f32, text: &str) -> f32 {
    let scaled = font.as_scaled(PxScale::from(px));
    let baseline = y + scaled.ascent();
    let mut cursor = x;
    let mut prev = None;
    for ch in text.chars() {
        let id = scaled.glyph_id(ch);
        if let Some(p) = prev {
            cursor += scaled.kern(p, id);
        }
        let glyph = id.with_scale_and_position(PxScale::from(px), point(cursor, baseline));
        if let Some(outlined) = font.outline_glyph(glyph) {
            let b = outlined.px_bounds();
            outlined.draw(|gx, gy, cov| {
                blend(
                    img,
                    b.min.x as i64 + gx as i64,
                    b.min.y as i64 + gy as i64,
                    cov,
                    0,
                );
            });
        }
        cursor += scaled.h_advance(id);
        prev = Some(id);
    }
    cursor - x
}

fn text_width(font: &FontRef<'_>, px: f32, text: &str) -> f32 {
    let scaled = font.as_scaled(PxScale::from(px));
    text.chars()
        .map(|c| scaled.h_advance(scaled.glyph_id(c)))
        .sum()
}

fn hline(img: &mut RgbImage, y: u32, x0: u32, x1: u32) {
    if y < img.height() {
        for x in x0..x1.min(img.width()) {
            img.put_pixel(x, y, Rgb([90, 90, 90]));
        }
    }
}

fn vline(img: &mut RgbImage, x: u32, y0: u32, y1: u32) {
    if x < img.width() {
        for y in y0..y1.min(img.height()) {
            img.put_pixel(x, y, Rgb([90, 90, 90]));
        }
    }
}

/// Renders one synthetic region deterministically from its spec.
pub fn render_synthetic(spec: &SyntheticSpec, font: &FontRef<'_>) -> RgbImage {
    let mut img = RgbImage::from_pixel(spec.width, spec.height, Rgb([255, 255, 255]));
    let mut rng = SplitMix64(spec.seed);
    let (w, h) = (spec.width as f32, spec.height as f32);
    match spec.kind {
        Kind::Lines => {
            let px = (w.min(h) / 6.0).clamp(10.0, 22.0);
            let step = px * 1.4;
            let mut y = 4.0;
            while y + px < h {
                let mut x = 4.0;
                loop {
                    let word = WORDS[rng.below(WORDS.len())];
                    let adv = text_width(font, px, word);
                    if x + adv > w - 4.0 {
                        break;
                    }
                    draw_text(&mut img, font, px, x, y, word);
                    x += adv + px * 0.35;
                }
                y += step;
            }
        }
        Kind::Table => {
            let cols = ((w / 160.0) as u32).max(2);
            let rows = ((h / 36.0) as u32).max(2);
            let (cw, rh) = (spec.width / cols, spec.height / rows);
            let px = (rh as f32 * 0.5).clamp(8.0, 16.0);
            for r in 0..=rows {
                hline(&mut img, (r * rh).min(spec.height - 1), 0, spec.width);
            }
            for c in 0..=cols {
                vline(&mut img, (c * cw).min(spec.width - 1), 0, spec.height);
            }
            for r in 0..rows {
                for c in 0..cols {
                    let cell = if rng.below(2) == 0 {
                        WORDS[rng.below(WORDS.len())].to_owned()
                    } else {
                        format!("{}.{:02}", rng.below(900) + 10, rng.below(100))
                    };
                    draw_text(
                        &mut img,
                        font,
                        px,
                        (c * cw) as f32 + 4.0,
                        (r * rh) as f32 + (rh as f32 - px) / 2.0,
                        &cell,
                    );
                }
            }
        }
        Kind::Formula => {
            let px = (h / 3.0).clamp(10.0, 28.0);
            let mut y = (h - px).max(0.0) / 3.0;
            let mut left = 0;
            while y + px < h && left < 2 {
                let f = FORMULAS[rng.below(FORMULAS.len())];
                let x = ((w - text_width(font, px, f)) / 2.0).max(4.0);
                draw_text(&mut img, font, px, x, y, f);
                y += px * 1.6;
                left += 1;
            }
        }
    }
    img
}

/// Encodes an image as JPEG at [`JPEG_QUALITY`].
pub fn encode_jpeg(img: &RgbImage) -> Result<Vec<u8>> {
    let mut out = Cursor::new(Vec::new());
    JpegEncoder::new_with_quality(&mut out, JPEG_QUALITY)
        .encode(
            img.as_raw(),
            img.width(),
            img.height(),
            ExtendedColorType::Rgb8,
        )
        .map_err(|e| BenchError(e.to_string()))?;
    Ok(out.into_inner())
}

/// Renders every spec of a synthetic manifest into regions.
pub fn synthetic_regions(specs: &[SyntheticSpec], font: &FontRef<'_>) -> Result<Vec<Region>> {
    specs
        .iter()
        .map(|s| {
            Ok(Region {
                id: s.id.clone(),
                width: s.width,
                height: s.height,
                tasks: vec![s.task],
                jpeg: encode_jpeg(&render_synthetic(s, font))?,
            })
        })
        .collect()
}

/// Loads the bundled font, or the font file at `path` when given.
pub fn load_font(path: Option<&Path>) -> Result<FontRef<'static>> {
    match path {
        None => FontRef::try_from_slice(BUNDLED_FONT).map_err(|e| BenchError(e.to_string())),
        Some(p) => {
            let bytes: &'static [u8] = Box::leak(std::fs::read(p)?.into_boxed_slice());
            FontRef::try_from_slice(bytes).map_err(|e| BenchError(e.to_string()))
        }
    }
}

#[derive(Serialize, Deserialize)]
struct DirEntry {
    file: String,
    width: u32,
    height: u32,
    tasks: Vec<Task>,
}

#[derive(Serialize, Deserialize)]
struct DirManifest {
    regions: Vec<DirEntry>,
}

/// Name of the manifest inside a region directory.
pub const DIR_MANIFEST: &str = "manifest.json";

/// Writes regions as `<id>.jpg` files plus [`DIR_MANIFEST`] into `dir`.
pub fn write_region_dir(dir: &Path, regions: &[Region]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut entries = Vec::new();
    for r in regions {
        let file = format!("{}.jpg", r.id);
        std::fs::write(dir.join(&file), &r.jpeg)?;
        entries.push(DirEntry {
            file,
            width: r.width,
            height: r.height,
            tasks: r.tasks.clone(),
        });
    }
    let manifest = DirManifest { regions: entries };
    std::fs::write(
        dir.join(DIR_MANIFEST),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    Ok(())
}

/// Loads a region directory written by [`write_region_dir`] or `render_regions.sh`.
pub fn load_region_dir(dir: &Path) -> Result<Vec<Region>> {
    let manifest: DirManifest = serde_json::from_slice(&std::fs::read(dir.join(DIR_MANIFEST))?)?;
    manifest
        .regions
        .into_iter()
        .map(|e| {
            let path: PathBuf = dir.join(&e.file);
            let id = e.file.strip_suffix(".jpg").unwrap_or(&e.file).to_owned();
            Ok(Region {
                id,
                width: e.width,
                height: e.height,
                tasks: e.tasks,
                jpeg: std::fs::read(&path)
                    .map_err(|err| BenchError(format!("{}: {err}", e.file)))?,
            })
        })
        .collect()
}
