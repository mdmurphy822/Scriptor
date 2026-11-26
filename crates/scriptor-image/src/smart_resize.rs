use crate::error::ImageError;

/// Spatial factor of the vision tower: patch 14 times the 2x2 merge.
pub const IMAGE_FACTOR: u32 = 28;
/// Temporal patch size of the vision tower.
pub const TEMPORAL_FACTOR: u32 = 2;
/// Fewest pixels (times temporal extent) a resized image may have.
pub const MIN_PIXELS: u64 = 12_544;
/// Most pixels (times temporal extent) a resized image may have.
pub const MAX_PIXELS: u64 = 9_633_792;
/// Largest permitted ratio between the longer and shorter side.
pub const MAX_ASPECT_RATIO: f64 = 200.0;

/// Parameters of [`smart_resize`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmartResizeConfig {
    /// Frames the image is repeated into.
    pub num_frames: u32,
    /// Temporal factor frames are rounded to.
    pub temporal_factor: u32,
    /// Spatial factor both sides are rounded to.
    pub factor: u32,
    /// Lower bound on `frames * height * width`.
    pub min_pixels: u64,
    /// Upper bound on `frames * height * width`.
    pub max_pixels: u64,
    /// Refuse images whose side ratio exceeds this; `None` disables the check.
    pub max_aspect_ratio: Option<f64>,
}

impl Default for SmartResizeConfig {
    fn default() -> Self {
        Self {
            num_frames: 2,
            temporal_factor: TEMPORAL_FACTOR,
            factor: IMAGE_FACTOR,
            min_pixels: MIN_PIXELS,
            max_pixels: MAX_PIXELS,
            max_aspect_ratio: Some(MAX_ASPECT_RATIO),
        }
    }
}

/// Returns the `(height, width)` an image is resized to: both sides rounded half-to-even
/// to a multiple of the factor, shrunk by floor when the temporal-scaled area exceeds
/// `max_pixels`, grown by ceil when it is below `min_pixels`.
pub fn smart_resize_with(
    height: u32,
    width: u32,
    config: &SmartResizeConfig,
) -> Result<(u32, u32), ImageError> {
    if config.num_frames < config.temporal_factor
        || config.temporal_factor == 0
        || config.factor == 0
    {
        return Err(ImageError::InvalidResizeConfig);
    }
    if height == 0 || width == 0 {
        return Err(ImageError::InvalidDimensions { width, height });
    }
    if let Some(limit) = config.max_aspect_ratio {
        let (long, short) = (height.max(width), height.min(width));
        if f64::from(long) / f64::from(short) > limit {
            return Err(ImageError::AspectRatio {
                height,
                width,
                limit,
            });
        }
    }
    let frames = f64::from(config.num_frames);
    let factor = f64::from(config.factor);
    let (h, w) = (f64::from(height), f64::from(width));
    let mut h_bar = (h / factor).round_ties_even() * factor;
    let mut w_bar = (w / factor).round_ties_even() * factor;
    let t_bar = (frames / f64::from(config.temporal_factor)).round_ties_even()
        * f64::from(config.temporal_factor);
    let volume = t_bar * h_bar * w_bar;
    if volume > config.max_pixels as f64 {
        let beta = (frames * h * w / config.max_pixels as f64).sqrt();
        h_bar = factor.max((h / beta / factor).floor() * factor);
        w_bar = factor.max((w / beta / factor).floor() * factor);
    } else if volume < config.min_pixels as f64 {
        let beta = (config.min_pixels as f64 / (frames * h * w)).sqrt();
        h_bar = (h * beta / factor).ceil() * factor;
        w_bar = (w * beta / factor).ceil() * factor;
    }
    Ok((h_bar as u32, w_bar as u32))
}

/// [`smart_resize_with`] using the GLM-OCR processor defaults.
pub fn smart_resize(height: u32, width: u32) -> Result<(u32, u32), ImageError> {
    smart_resize_with(height, width, &SmartResizeConfig::default())
}
