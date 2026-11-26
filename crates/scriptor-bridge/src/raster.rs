//! Pure-Rust PDF page rasterization.

use anyhow::{Result, bail};
use hayro::hayro_interpret::font::GlyphRun;
use hayro::hayro_interpret::{
    BlendMode, ClipPath, Context, Device, DrawMode, DrawProps, Image, ImageData, ImageDrawProps,
    InterpreterCache, InterpreterSettings, SoftMask, TransformExt, interpret_page,
};
use hayro::hayro_syntax::page::Page;
use hayro::hayro_syntax::{LoadPdfError, Pdf};
use hayro::kurbo::{Affine, BezPath, Rect};
use hayro::vello_cpu::color::palette::css::WHITE;
use hayro::{PixmapSettings, RenderCache, RenderSettings, render};
use image::imageops::{self, FilterType};
use image::{RgbImage, RgbaImage};
use std::sync::Arc;

/// An 8-bit RGBA raster, rows top to bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Raster {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

pub(crate) fn load(pdf: &[u8]) -> Result<Pdf> {
    load_shared(Arc::new(pdf.to_vec()))
}

/// Parses `pdf` without copying its bytes.
pub(crate) fn load_shared(pdf: Arc<Vec<u8>>) -> Result<Pdf> {
    Pdf::new(pdf).map_err(|error| match error {
        LoadPdfError::Decryption(_) => {
            anyhow::anyhow!("the PDF is encrypted or password-protected and cannot be rendered")
        }
        LoadPdfError::Invalid => anyhow::anyhow!("the file is not a readable PDF"),
    })
}

/// The number of pages in `pdf`.
pub fn page_count(pdf: &[u8]) -> Result<usize> {
    Ok(load(pdf)?.pages().len())
}

/// The rendered size in points, width then height, of every page of `pdf`
/// in page order.
pub fn page_dimensions(pdf: &[u8]) -> Result<Vec<(f32, f32)>> {
    let document = load(pdf)?;
    let pages = document.pages();
    Ok((0..pages.len())
        .filter_map(|index| pages.get(index))
        .map(|page| page.render_dimensions())
        .collect())
}

/// Renders zero-based page `page_index` of `pdf` at `dpi`, with page
/// rotation and the crop box applied. The raster is opaque, composited on
/// white, and each side is `round(points * dpi / 72)` pixels.
pub fn render_page(pdf: &[u8], page_index: usize, dpi: f32) -> Result<Raster> {
    if !dpi.is_finite() || dpi <= 0.0 {
        bail!("render resolution must be a positive number");
    }
    render_loaded(&load(pdf)?, page_index, dpi)
}

/// Like [`render_page`] over shared document bytes that are parsed without
/// being copied.
pub fn render_page_shared(pdf: &Arc<Vec<u8>>, page_index: usize, dpi: f32) -> Result<Raster> {
    if !dpi.is_finite() || dpi <= 0.0 {
        bail!("render resolution must be a positive number");
    }
    render_loaded(&load_shared(Arc::clone(pdf))?, page_index, dpi)
}

fn render_loaded(document: &Pdf, page_index: usize, dpi: f32) -> Result<Raster> {
    let pages = document.pages();
    let Some(page) = pages.get(page_index) else {
        bail!(
            "page {} is outside the document's {} pages",
            page_index + 1,
            pages.len()
        );
    };
    let (width_pt, height_pt) = page.render_dimensions();
    let factor = dpi / 72.0;
    let width = (width_pt * factor).round().max(1.0);
    let height = (height_pt * factor).round().max(1.0);
    render_sized(page, width as u64, height as u64)
}

/// Renders `page` onto an opaque white raster of exactly `width` by
/// `height` pixels, with page rotation and the crop box applied.
pub(crate) fn render_sized<'a>(page: &'a Page<'a>, width: u64, height: u64) -> Result<Raster> {
    render_sized_with(page, width, height, &InterpreterSettings::default())
}

/// Like [`render_sized`] with the interpreter `settings`.
pub(crate) fn render_sized_with<'a>(
    page: &'a Page<'a>,
    width: u64,
    height: u64,
    settings: &InterpreterSettings,
) -> Result<Raster> {
    if width == 0 || height == 0 {
        bail!("the page raster must be at least one pixel on each side");
    }
    if width > u64::from(u16::MAX) || height > u64::from(u16::MAX) {
        bail!("the page raster exceeds the renderer's size limit at this resolution");
    }
    let (width_pt, height_pt) = page.render_dimensions();
    let (width, height) = (width as f32, height as f32);
    let scale = Affine::scale_non_uniform(
        f64::from((width + 0.5) / width_pt),
        f64::from((height + 0.5) / height_pt),
    );
    let transform = scale * page.initial_transform(true).to_kurbo();
    if let Some(raster) = sole_image_raster(page, transform, width as u32, height as u32, settings)
    {
        return Ok(raster);
    }
    // The renderer truncates the scaled size, so scale to the middle of the
    // target pixel to land on exactly `width` x `height`.
    let pixmap = render(
        page,
        &RenderCache::new(),
        settings,
        &RenderSettings {
            force_image_interpolation: true,
        },
        &PixmapSettings {
            x_scale: (width + 0.5) / width_pt,
            y_scale: (height + 0.5) / height_pt,
            bg_color: WHITE,
        },
    );
    let (out_w, out_h) = (u32::from(pixmap.width()), u32::from(pixmap.height()));
    Ok(Raster {
        width: out_w,
        height: out_h,
        rgba: pixmap.data_as_u8_slice().to_vec(),
    })
}

/// A drawing device that keeps the page's image when everything the page
/// paints is one opaque raster image.
#[derive(Default)]
struct SoleImage {
    image: Option<(ImageData, Affine)>,
    other: bool,
}

impl SoleImage {
    fn paints_other(&mut self) {
        self.other = true;
        self.image = None;
    }
}

impl<'a> Device<'a> for SoleImage {
    fn draw_path(&mut self, _: &BezPath, _: DrawProps<'a>, _: &DrawMode) {
        self.paints_other();
    }

    fn push_clip_path(&mut self, _: &ClipPath) {}

    fn push_transparency_group(
        &mut self,
        opacity: f32,
        mask: Option<SoftMask<'a>>,
        blend_mode: BlendMode,
    ) {
        if opacity < 1.0 || mask.is_some() || blend_mode != BlendMode::Normal {
            self.paints_other();
        }
    }

    fn draw_glyph_run(&mut self, _: &GlyphRun<'_, 'a>, _: DrawProps<'a>, draw_mode: &DrawMode) {
        if !matches!(draw_mode, DrawMode::Invisible) {
            self.paints_other();
        }
    }

    fn draw_image(&mut self, image: Image<'a, '_>, props: ImageDrawProps<'a>) {
        let Image::Raster(raster) = image else {
            return self.paints_other();
        };
        if self.other || self.image.is_some() || props.soft_mask.is_some() {
            return self.paints_other();
        }
        let mut decoded = None;
        raster.with_rgba(
            |data, alpha| {
                if alpha.is_none() {
                    decoded = Some(data);
                }
            },
            None,
        );
        match decoded {
            Some(data) => {
                let (sx, sy) = data.scale_factors();
                let transform =
                    props.transform * Affine::scale_non_uniform(f64::from(sx), f64::from(sy));
                self.image = Some((data, transform));
            }
            None => self.paints_other(),
        }
    }

    fn pop_clip(&mut self) {}

    fn pop_transparency_group(&mut self) {}
}

/// The page raster of `width` by `height` pixels resampled bilinearly
/// straight from the page's image, when the page under `transform` paints
/// exactly one opaque, upright raster image that covers the whole raster to
/// within a pixel; `None` otherwise.
fn sole_image_raster(
    page: &Page<'_>,
    transform: Affine,
    width: u32,
    height: u32,
    settings: &InterpreterSettings,
) -> Option<Raster> {
    if page.resources().x_objects.len() != 1 {
        return None;
    }
    let cache = InterpreterCache::new();
    let bounds = Rect::new(0.0, 0.0, f64::from(width), f64::from(height));
    let mut context = Context::new(transform, bounds, &cache, page.xref(), settings.clone());
    let mut device = SoleImage::default();
    interpret_page(page, &mut context, &mut device);
    let (data, placement) = device.image?;
    let (image_w, image_h) = (data.width(), data.height());
    let [a, b, c, d, e, f] = placement.as_coeffs();
    if b.abs() > 1e-6 || c.abs() > 1e-6 || a == 0.0 || d == 0.0 {
        return None;
    }
    let x = [e, e + a * f64::from(image_w)];
    let y = [f, f + d * f64::from(image_h)];
    let covers = |ends: [f64; 2], extent: u32| {
        let (low, high) = (ends[0].min(ends[1]), ends[0].max(ends[1]));
        low.abs() <= 1.0 && (high - f64::from(extent)).abs() <= 1.0
    };
    if !covers(x, width) || !covers(y, height) {
        return None;
    }
    let mut source = match data {
        ImageData::Rgb(rgb) => RgbImage::from_raw(rgb.width, rgb.height, rgb.data)?,
        ImageData::Luma(luma) => RgbImage::from_fn(luma.width, luma.height, |px, py| {
            let v = luma.data[(py * luma.width + px) as usize];
            image::Rgb([v, v, v])
        }),
    };
    if a < 0.0 {
        imageops::flip_horizontal_in_place(&mut source);
    }
    if d < 0.0 {
        imageops::flip_vertical_in_place(&mut source);
    }
    let resized = imageops::resize(&source, width, height, FilterType::Triangle);
    let rgba = RgbaImage::from_fn(width, height, |px, py| {
        let [r, g, b] = resized.get_pixel(px, py).0;
        image::Rgba([r, g, b, 255])
    });
    Some(Raster {
        width,
        height,
        rgba: rgba.into_raw(),
    })
}

#[cfg(test)]
pub(crate) mod fixtures {
    /// Builds a PDF whose pages are `(media_box, extra_page_entries)` each
    /// painting a filled rectangle at (10,10)-(60,40).
    pub fn pdf(pages: &[(&str, &str)]) -> Vec<u8> {
        let drawn = pages
            .iter()
            .map(|(media, extra)| (*media, *extra, PLAIN_CONTENT))
            .collect::<Vec<_>>();
        pdf_drawn(&drawn)
    }

    /// The content stream of a plain page: one filled rectangle.
    const PLAIN_CONTENT: &str = "0 0 0 rg 10 10 50 30 re f";

    /// Like [`pdf`], with the content stream of each page given as the third
    /// item.
    pub fn pdf_drawn(pages: &[(&str, &str, &str)]) -> Vec<u8> {
        let mut objects: Vec<String> = vec![String::new(); 2 + pages.len() * 2];
        let kids = (0..pages.len())
            .map(|i| format!("{} 0 R", 3 + i * 2))
            .collect::<Vec<_>>()
            .join(" ");
        objects[0] = "<< /Type /Catalog /Pages 2 0 R >>".into();
        objects[1] = format!("<< /Type /Pages /Kids [{kids}] /Count {} >>", pages.len());
        for (i, (media, extra, content)) in pages.iter().enumerate() {
            objects[2 + i * 2] = format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox {media} {extra} /Contents {} 0 R >>",
                4 + i * 2
            );
            objects[3 + i * 2] = format!(
                "<< /Length {} >>\nstream\n{content}\nendstream",
                content.len()
            );
        }
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, body) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend(format!("{} 0 obj\n{body}\nendobj\n", i + 1).bytes());
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

    /// A one-page 100 by 100 point PDF whose resources hold one 2 by 2 RGB
    /// image `/Im0` (red, green / blue, white rows) and whose content stream
    /// is `content`.
    pub fn image_page_pdf(content: &str) -> Vec<u8> {
        let pixels: [u8; 12] = [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        let mut object = |out: &mut Vec<u8>, number: usize, head: String, stream: &[u8]| {
            offsets.push(out.len());
            out.extend(format!("{number} 0 obj\n{head}").bytes());
            if !stream.is_empty() {
                out.extend(b"\nstream\n");
                out.extend(stream);
                out.extend(b"\nendstream");
            }
            out.extend(b"\nendobj\n");
        };
        object(&mut out, 1, "<< /Type /Catalog /Pages 2 0 R >>".into(), b"");
        object(
            &mut out,
            2,
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".into(),
            b"",
        );
        object(
            &mut out,
            3,
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
             /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
                .into(),
            b"",
        );
        object(
            &mut out,
            4,
            format!("<< /Length {} >>", content.len()),
            content.as_bytes(),
        );
        object(
            &mut out,
            5,
            format!(
                "<< /Type /XObject /Subtype /Image /Width 2 /Height 2 \
                 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Length {} >>",
                pixels.len()
            ),
            &pixels,
        );
        let xref = out.len();
        out.extend(b"xref\n0 6\n0000000000 65535 f \n");
        for offset in offsets {
            out.extend(format!("{offset:010} 00000 n \n").bytes());
        }
        out.extend(
            format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").bytes(),
        );
        out
    }

    pub fn three_page_pdf() -> Vec<u8> {
        pdf(&[
            ("[0 0 200 100]", ""),
            ("[0 0 200 100]", "/Rotate 90"),
            ("[0 0 400 300]", "/CropBox [0 0 100 50]"),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    fn pixel(raster: &Raster, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * raster.width + x) * 4) as usize;
        raster.rgba[i..i + 4].try_into().unwrap()
    }

    #[test]
    fn counts_pages() {
        assert_eq!(page_count(&three_page_pdf()).unwrap(), 3);
    }

    #[test]
    fn dimensions_follow_dpi_rotation_and_crop_box() {
        let pdf = three_page_pdf();
        for (dpi, scale) in [(72.0, 1.0), (150.0, 150.0 / 72.0), (300.0, 300.0 / 72.0)] {
            let round = |points: f32| (points * scale).round() as u32;
            let portrait = render_page(&pdf, 0, dpi).unwrap();
            assert_eq!(
                (portrait.width, portrait.height),
                (round(200.0), round(100.0))
            );
            assert_eq!(
                portrait.rgba.len() as u32,
                portrait.width * portrait.height * 4
            );
            let rotated = render_page(&pdf, 1, dpi).unwrap();
            assert_eq!(
                (rotated.width, rotated.height),
                (round(100.0), round(200.0))
            );
            let cropped = render_page(&pdf, 2, dpi).unwrap();
            assert_eq!((cropped.width, cropped.height), (round(100.0), round(50.0)));
        }
    }

    #[test]
    fn paints_content_on_white() {
        let raster = render_page(&three_page_pdf(), 0, 72.0).unwrap();
        // The rectangle spans y 10..40 above the page bottom (height 100).
        assert_eq!(pixel(&raster, 30, 75), [0, 0, 0, 255]);
        assert_eq!(pixel(&raster, 150, 10), [255, 255, 255, 255]);
    }

    #[test]
    fn a_page_that_is_one_image_is_resampled_from_the_image() {
        let pdf = image_page_pdf("q 100 0 0 100 0 0 cm /Im0 Do Q");
        let raster = render_page(&pdf, 0, 72.0).unwrap();
        let source =
            RgbImage::from_raw(2, 2, vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]).unwrap();
        let expected = imageops::resize(&source, 100, 100, FilterType::Triangle);
        assert_eq!((raster.width, raster.height), (100, 100));
        for (px, py) in [(0, 0), (99, 0), (0, 99), (99, 99), (50, 50), (20, 70)] {
            let [r, g, b] = expected.get_pixel(px, py).0;
            assert_eq!(pixel(&raster, px, py), [r, g, b, 255], "pixel {px},{py}");
        }
        assert_eq!(pixel(&raster, 0, 0), [255, 0, 0, 255]);
        assert_eq!(pixel(&raster, 99, 99), [255, 255, 255, 255]);
    }

    #[test]
    fn a_page_painting_more_than_its_image_is_rendered_whole() {
        let content = "q 100 0 0 100 0 0 cm /Im0 Do Q 0 0 0 rg 10 10 30 30 re f";
        let raster = render_page(&image_page_pdf(content), 0, 72.0).unwrap();
        assert_eq!(pixel(&raster, 20, 80), [0, 0, 0, 255]);
        let partial = image_page_pdf("q 50 0 0 50 0 0 cm /Im0 Do Q");
        let raster = render_page(&partial, 0, 72.0).unwrap();
        assert_eq!(pixel(&raster, 75, 25), [255, 255, 255, 255]);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(render_page(&three_page_pdf(), 3, 72.0).is_err());
        assert!(render_page(&three_page_pdf(), 0, 0.0).is_err());
        assert!(page_count(b"not a pdf at all").is_err());
        let mut encrypted = three_page_pdf();
        let text = String::from_utf8(encrypted.clone()).unwrap();
        let o = "0".repeat(32);
        let patched = text.replace(
            "/Root 1 0 R",
            &format!(
                "/Root 1 0 R /ID [<00112233445566778899aabbccddeeff> <00112233445566778899aabbccddeeff>] /Encrypt << /Filter /Standard /V 1 /R 2 /P -4 /O ({o}) /U ({o}) >>"
            ),
        );
        encrypted = patched.into_bytes();
        let error = page_count(&encrypted).unwrap_err().to_string();
        assert!(error.contains("encrypted"), "{error}");
    }
}
