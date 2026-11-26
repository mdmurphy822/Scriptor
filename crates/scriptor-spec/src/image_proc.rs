//! Image-processor settings from `preprocessor_config.json` / `processor_config.json`.

use serde::Deserialize;

use crate::error::SpecError;

/// Resampling filter used by the reference image processor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resample {
    Bicubic,
}

/// Preprocessing parameters of the reference image processor.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageProcessorConfig {
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub rescale_factor: f64,
    pub resample: Resample,
}

#[derive(Deserialize)]
struct RawSize {
    shortest_edge: usize,
    longest_edge: usize,
}

#[derive(Deserialize)]
struct RawProcessor {
    size: RawSize,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    #[serde(default)]
    rescale_factor: Option<f64>,
    #[serde(default)]
    resample: Option<u32>,
}

#[derive(Deserialize)]
struct RawProcessorFile {
    image_processor: RawProcessor,
}

impl RawProcessor {
    fn finish(self) -> Result<ImageProcessorConfig, SpecError> {
        let resample = match self.resample {
            None | Some(3) => Resample::Bicubic,
            Some(other) => {
                return Err(SpecError::invalid(format!(
                    "unsupported resample filter {other}; only bicubic (3) is implemented"
                )));
            }
        };
        if self.size.shortest_edge == 0 || self.size.longest_edge < self.size.shortest_edge {
            return Err(SpecError::invalid("image size bounds are inconsistent"));
        }
        if self.image_std.iter().any(|s| *s <= 0.0) {
            return Err(SpecError::invalid("image_std must be positive"));
        }
        Ok(ImageProcessorConfig {
            min_pixels: self.size.shortest_edge,
            max_pixels: self.size.longest_edge,
            patch_size: self.patch_size,
            merge_size: self.merge_size,
            temporal_patch_size: self.temporal_patch_size,
            image_mean: self.image_mean,
            image_std: self.image_std,
            rescale_factor: self.rescale_factor.unwrap_or(1.0 / 255.0),
            resample,
        })
    }
}

impl ImageProcessorConfig {
    /// Parses `preprocessor_config.json` text.
    pub fn from_preprocessor_json(text: &str) -> Result<Self, SpecError> {
        let raw: RawProcessor = serde_json::from_str(text).map_err(|source| SpecError::Parse {
            what: "preprocessor_config.json",
            source,
        })?;
        raw.finish()
    }

    /// Parses `processor_config.json` text (image processor nested under `image_processor`).
    pub fn from_processor_json(text: &str) -> Result<Self, SpecError> {
        let raw: RawProcessorFile =
            serde_json::from_str(text).map_err(|source| SpecError::Parse {
                what: "processor_config.json",
                source,
            })?;
        raw.image_processor.finish()
    }

    /// Pixels covered by one merged vision token (`(patch_size * merge_size)^2`).
    pub fn pixels_per_token(&self) -> usize {
        let edge = self.patch_size * self.merge_size;
        edge * edge
    }

    /// Returns a copy whose `max_pixels` is replaced by `max_pixels`.
    pub fn with_max_pixels(&self, max_pixels: usize) -> Result<Self, SpecError> {
        if max_pixels < self.min_pixels {
            return Err(SpecError::invalid(format!(
                "max pixels {max_pixels} is below the minimum {}",
                self.min_pixels
            )));
        }
        Ok(Self {
            max_pixels,
            ..self.clone()
        })
    }

    /// Largest number of image tokens one image can produce.
    pub fn max_image_tokens(&self) -> usize {
        self.max_pixels / self.pixels_per_token()
    }
}
