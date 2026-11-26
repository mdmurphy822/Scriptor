//! Page rasters for native OCR: PDF pages rendered at a fixed resolution
//! with a long-side pixel cap, and image files loaded as a single page.

use crate::raster::{self, Raster};
use anyhow::{Context, Result, bail};
use hayro::hayro_syntax::Pdf;
use image::RgbImage;
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// The resolution, in dots per inch, at which PDF pages are rendered.
pub const PDF_DPI: f64 = 300.0;

/// The cap, in pixels, on the longer side of a rendered PDF page.
pub const MAX_LONG_SIDE_PX: f64 = 3500.0;

const PDF_MAGIC: &[u8] = b"%PDF-";

static NEXT_SOURCE_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static OPEN_DOCUMENT: RefCell<Option<(u64, Pdf)>> = const { RefCell::new(None) };
}

/// A rendered page and the scale between its raster and the page's points.
#[derive(Clone, Debug)]
pub struct RenderedPage {
    pub image: RgbImage,
    /// The page's width in points (pixels for an image input).
    pub width_pt: f64,
    /// The page's height in points (pixels for an image input).
    pub height_pt: f64,
    /// The uniform scale the renderer was asked for, in raster pixels per
    /// point.
    pub scale: f64,
    /// The page's size in pixels at the source's dpi before the render cap
    /// (an image input's own size): the space region boxes are reported in.
    pub pixel_size: (f64, f64),
}

impl RenderedPage {
    /// Raster pixels per point along x: `image.width() / width_pt`.
    pub fn x_scale(&self) -> f64 {
        f64::from(self.image.width()) / self.width_pt
    }

    /// Raster pixels per point along y: `image.height() / height_pt`.
    pub fn y_scale(&self) -> f64 {
        f64::from(self.image.height()) / self.height_pt
    }

    /// Maps a raster-pixel point back to page points.
    pub fn to_points(&self, x_px: f64, y_px: f64) -> (f64, f64) {
        (x_px / self.x_scale(), y_px / self.y_scale())
    }
}

enum Content {
    Pdf {
        id: u64,
        bytes: Arc<Vec<u8>>,
        sizes: Vec<(f32, f32)>,
        dpi: f64,
    },
    Image(Arc<RgbImage>),
}

/// A document ready to render page by page. It is `Send + Sync`: PDF bytes
/// are shared and every rendering thread parses its own copy of the
/// document structure, so pages render independently and concurrently.
pub struct PageSource {
    content: Content,
}

impl PageSource {
    /// Opens `bytes` as a PDF when they start with `%PDF-`, otherwise as a
    /// single-page image, rendering PDF pages at [`PDF_DPI`].
    pub fn open(bytes: &[u8]) -> Result<Self> {
        Self::open_at(bytes, PDF_DPI)
    }

    /// Like [`PageSource::open`] with PDF pages rendered at `dpi`.
    pub fn open_at(bytes: &[u8], dpi: f64) -> Result<Self> {
        if !dpi.is_finite() || dpi <= 0.0 {
            bail!("render resolution must be a positive number");
        }
        if bytes.starts_with(PDF_MAGIC) {
            let shared = Arc::new(bytes.to_vec());
            let document = raster::load_shared(Arc::clone(&shared))?;
            let pages = document.pages();
            let sizes: Vec<(f32, f32)> = (0..pages.len())
                .filter_map(|index| pages.get(index))
                .map(|page| page.render_dimensions())
                .collect();
            if sizes.is_empty() {
                bail!("the PDF has no pages");
            }
            Ok(Self {
                content: Content::Pdf {
                    id: NEXT_SOURCE_ID.fetch_add(1, Ordering::Relaxed),
                    bytes: shared,
                    sizes,
                    dpi,
                },
            })
        } else {
            let decoded = image::load_from_memory(bytes)
                .context("the file is neither a PDF nor a readable image")?;
            Ok(Self {
                content: Content::Image(Arc::new(decoded.to_rgb8())),
            })
        }
    }

    /// The number of pages: the PDF's page count, or one for an image.
    pub fn page_count(&self) -> usize {
        match &self.content {
            Content::Pdf { sizes, .. } => sizes.len(),
            Content::Image(_) => 1,
        }
    }

    /// The size in points of zero-based page `index`, after rotation and
    /// crop box; for an image, its size in pixels.
    pub fn page_size_points(&self, index: usize) -> Result<(f64, f64)> {
        match &self.content {
            Content::Pdf { sizes, .. } => {
                let (width, height) = *sizes.get(index).with_context(|| {
                    format!(
                        "page {} is outside the document's {} pages",
                        index + 1,
                        sizes.len()
                    )
                })?;
                Ok((f64::from(width), f64::from(height)))
            }
            Content::Image(image) => {
                if index != 0 {
                    bail!("page {} is outside the document's 1 page", index + 1);
                }
                Ok((f64::from(image.width()), f64::from(image.height())))
            }
        }
    }

    /// The size in pixels of zero-based page `index` at the source's dpi
    /// before the render cap (an image input's own size), as
    /// [`RenderedPage::pixel_size`] reports it.
    pub fn page_pixel_size(&self, index: usize) -> Result<(f64, f64)> {
        let (width_pt, height_pt) = self.page_size_points(index)?;
        match &self.content {
            Content::Pdf { dpi, .. } => Ok((width_pt * dpi / 72.0, height_pt * dpi / 72.0)),
            Content::Image(_) => Ok((width_pt, height_pt)),
        }
    }

    /// Renders zero-based page `index`. A PDF page is rendered at the
    /// source's dpi, its scale reduced so the longer side is at most
    /// [`MAX_LONG_SIDE_PX`]; an image is returned as loaded at scale 1.
    pub fn render_page(&self, index: usize) -> Result<RenderedPage> {
        let (width_pt, height_pt) = self.page_size_points(index)?;
        match &self.content {
            Content::Image(image) => Ok(RenderedPage {
                image: RgbImage::clone(image),
                width_pt,
                height_pt,
                scale: 1.0,
                pixel_size: (width_pt, height_pt),
            }),
            Content::Pdf { id, bytes, dpi, .. } => {
                let scale = capped_scale(width_pt, height_pt, *dpi);
                let width = pixels(width_pt, scale);
                let height = pixels(height_pt, scale);
                let raster = OPEN_DOCUMENT.with(|slot| -> Result<Raster> {
                    let mut slot = slot.borrow_mut();
                    if slot.as_ref().map(|(held, _)| *held) != Some(*id) {
                        *slot = Some((*id, raster::load_shared(Arc::clone(bytes))?));
                    }
                    let (_, document) = slot.as_ref().context("document not loaded")?;
                    let pages = document.pages();
                    let page = pages
                        .get(index)
                        .with_context(|| format!("page {} could not be read", index + 1))?;
                    raster::render_sized(page, width, height)
                })?;
                Ok(RenderedPage {
                    image: to_rgb(&raster)?,
                    width_pt,
                    height_pt,
                    scale,
                    pixel_size: (width_pt * dpi / 72.0, height_pt * dpi / 72.0),
                })
            }
        }
    }
}

/// The render scale in pixels per point for a page of the given size:
/// `dpi / 72`, reduced to `MAX_LONG_SIDE_PX / long_side_pt` when the longer
/// side would exceed [`MAX_LONG_SIDE_PX`].
pub fn capped_scale(width_pt: f64, height_pt: f64, dpi: f64) -> f64 {
    let scale = dpi / 72.0;
    let long_side_pt = width_pt.max(height_pt);
    if long_side_pt * scale > MAX_LONG_SIDE_PX {
        MAX_LONG_SIDE_PX / long_side_pt
    } else {
        scale
    }
}

/// The pixel extent of `points` at `scale`: the ceiling, at least one.
fn pixels(points: f64, scale: f64) -> u64 {
    ((points * scale).ceil() as u64).max(1)
}

fn to_rgb(raster: &Raster) -> Result<RgbImage> {
    let mut rgb = Vec::with_capacity(raster.rgba.len() / 4 * 3);
    for pixel in raster.rgba.chunks_exact(4) {
        rgb.extend_from_slice(&pixel[..3]);
    }
    RgbImage::from_raw(raster.width, raster.height, rgb)
        .context("the page raster has an inconsistent size")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::fixtures::{pdf, three_page_pdf};
    use image::ImageFormat;
    use std::io::Cursor;

    const LETTER: &str = "[0 0 612 792]";

    fn px(points: f64) -> u32 {
        (points * (300.0 / 72.0)).ceil() as u32
    }

    #[test]
    fn letter_page_renders_at_three_hundred_dpi() {
        let source = PageSource::open(&pdf(&[(LETTER, "")])).unwrap();
        assert_eq!(source.page_count(), 1);
        let page = source.render_page(0).unwrap();
        assert_eq!(
            (page.image.width(), page.image.height()),
            (px(612.0), px(792.0))
        );
        assert!((page.scale - 300.0 / 72.0).abs() < 1e-12);
        assert!((page.x_scale() - 300.0 / 72.0).abs() < 1e-9);
        let (x, y) = page.to_points(f64::from(px(612.0)), f64::from(px(792.0)));
        assert!((x - 612.0).abs() < 1e-2 && (y - 792.0).abs() < 1e-2);
    }

    #[test]
    fn large_page_is_capped_on_its_long_side() {
        let source = PageSource::open(&pdf(&[("[0 0 1224 792]", "")])).unwrap();
        let page = source.render_page(0).unwrap();
        assert!(matches!(page.image.width(), 3500 | 3501));
        assert_eq!(
            page.image.height(),
            (792.0f64 * 3500.0 / 1224.0).ceil() as u32
        );
        assert!((page.scale - 3500.0 / 1224.0).abs() < 1e-12);
    }

    #[test]
    fn rotation_and_crop_box_set_the_page_size() {
        let source = PageSource::open(&pdf(&[
            (LETTER, "/Rotate 90"),
            ("[0 0 1000 1000]", "/CropBox [0 0 306 396]"),
        ]))
        .unwrap();
        assert_eq!(source.page_count(), 2);
        let rotated = source.render_page(0).unwrap();
        assert_eq!(
            (rotated.image.width(), rotated.image.height()),
            (px(792.0), px(612.0))
        );
        let cropped = source.render_page(1).unwrap();
        assert_eq!(
            (cropped.image.width(), cropped.image.height()),
            (px(306.0), px(396.0))
        );
        assert_eq!(source.page_size_points(1).unwrap(), (306.0, 396.0));
    }

    #[test]
    fn pages_render_concurrently_and_independently() {
        let source = PageSource::open(&three_page_pdf()).unwrap();
        assert_eq!(source.page_count(), 3);
        let sizes: Vec<(u32, u32)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..3)
                .rev()
                .map(|index| {
                    let source = &source;
                    scope.spawn(move || {
                        let page = source.render_page(index).unwrap();
                        (page.image.width(), page.image.height())
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(
            sizes,
            vec![
                (px(100.0), px(50.0)),
                (px(100.0), px(200.0)),
                (px(200.0), px(100.0))
            ]
        );
    }

    #[test]
    fn image_inputs_load_as_one_page() {
        let source_image = RgbImage::from_fn(30, 20, |x, y| image::Rgb([x as u8, y as u8, 7]));
        for format in [ImageFormat::Png, ImageFormat::Tiff] {
            let mut bytes = Cursor::new(Vec::new());
            source_image.write_to(&mut bytes, format).unwrap();
            let source = PageSource::open(bytes.get_ref()).unwrap();
            assert_eq!(source.page_count(), 1);
            let page = source.render_page(0).unwrap();
            assert_eq!(page.image, source_image);
            assert_eq!(
                (page.scale, page.width_pt, page.height_pt),
                (1.0, 30.0, 20.0)
            );
            assert!(source.render_page(1).is_err());
        }
    }

    #[test]
    fn rejects_unreadable_input_and_missing_pages() {
        assert!(PageSource::open(b"not a document").is_err());
        let source = PageSource::open(&three_page_pdf()).unwrap();
        assert!(source.render_page(3).is_err());
        assert!(PageSource::open_at(&three_page_pdf(), 0.0).is_err());
    }
}
