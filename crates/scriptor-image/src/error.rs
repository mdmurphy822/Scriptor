use std::path::PathBuf;

/// Failures raised while parsing, decoding or preprocessing an image.
#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    /// The media reference is neither a supported data URL nor an allowed local path.
    #[error("unsupported media reference: {0}")]
    UnsupportedReference(String),
    /// The data URL declares a media type other than jpeg, png or webp.
    #[error("unsupported media type `{0}`")]
    UnsupportedMediaType(String),
    /// The data URL payload is not valid base64.
    #[error("invalid base64 payload: {0}")]
    Base64(#[from] base64::DecodeError),
    /// A local media path was supplied but no allowed root is configured.
    #[error("local media paths are disabled; no allowed local media path is set")]
    LocalMediaDisabled,
    /// A local media path resolves outside the allowed root.
    #[error("local media path `{0}` is outside the allowed local media path")]
    LocalMediaOutsideRoot(PathBuf),
    /// Reading a local media file failed.
    #[error("cannot read local media `{path}`: {source}")]
    Io {
        /// The path that could not be read.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// The bytes do not start with a jpeg, png or webp signature.
    #[error("unrecognised image format")]
    UnknownFormat,
    /// The decoder rejected the bytes.
    #[error("{format} decode failed: {detail}")]
    Decode {
        /// The format being decoded.
        format: &'static str,
        /// The decoder's message.
        detail: String,
    },
    /// The image has a zero or overflowing dimension.
    #[error("invalid image dimensions {width}x{height}")]
    InvalidDimensions {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// The longer side is more than the permitted multiple of the shorter side.
    #[error("aspect ratio of {height}x{width} exceeds {limit}")]
    AspectRatio {
        /// Height in pixels.
        height: u32,
        /// Width in pixels.
        width: u32,
        /// The permitted ratio.
        limit: f64,
    },
    /// The image has more pixels than the decoder may materialise.
    #[error("image {width}x{height} exceeds the decode bound of max_pixels {max_pixels}: {detail}")]
    TooLarge {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// The configured `max_pixels`.
        max_pixels: u64,
        /// How the bound applies to this format.
        detail: String,
    },
    /// The temporal extent is smaller than the temporal factor, or a factor is zero.
    #[error("invalid smart_resize configuration")]
    InvalidResizeConfig,
}
