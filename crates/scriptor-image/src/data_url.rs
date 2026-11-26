use std::path::{Path, PathBuf};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;

use crate::error::ImageError;

/// Policy for resolving media references that are not data URLs.
#[derive(Clone, Debug, Default)]
pub struct MediaPolicy {
    /// Directory under which `file://` URLs and plain paths may be read; `None` disables them.
    pub allowed_local_media_path: Option<PathBuf>,
}

/// Returns the raw image bytes named by `reference`.
///
/// Accepts `data:image/{jpeg,png,webp};base64,<payload>`; `file://` URLs and plain
/// paths are accepted only when `policy.allowed_local_media_path` is set and the
/// canonical path lies beneath it.
pub fn load_media(reference: &str, policy: &MediaPolicy) -> Result<Vec<u8>, ImageError> {
    if let Some(rest) = strip_prefix_ignore_case(reference, "data:") {
        return parse_data_url_body(rest);
    }
    if reference.contains("://") && !reference.starts_with("file://") {
        return Err(ImageError::UnsupportedReference(scheme_of(reference)));
    }
    let path = reference.strip_prefix("file://").unwrap_or(reference);
    read_local(Path::new(path), policy)
}

/// Decodes a `data:image/...;base64,` URL into its payload bytes.
pub fn parse_data_url(url: &str) -> Result<Vec<u8>, ImageError> {
    match strip_prefix_ignore_case(url, "data:") {
        Some(rest) => parse_data_url_body(rest),
        None => Err(ImageError::UnsupportedReference(scheme_of(url))),
    }
}

/// Decodes at most the first `max_bytes` payload bytes of a base64 `data:image/...` URL, or
/// `None` when the URL is not a base64 image data URL or the prefix is not valid base64.
pub fn data_url_prefix(url: &str, max_bytes: usize) -> Option<Vec<u8>> {
    let rest = strip_prefix_ignore_case(url, "data:")?;
    let (meta, payload) = rest.split_once(',')?;
    if !meta
        .split(';')
        .any(|p| p.trim().eq_ignore_ascii_case("base64"))
    {
        return None;
    }
    let want = max_bytes.div_ceil(3) * 4;
    let compact: Vec<u8> = payload
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .take(want)
        .collect();
    let usable = if compact.len() == want {
        want
    } else {
        compact.len()
    };
    STANDARD.decode(&compact[..usable]).ok()
}

fn parse_data_url_body(rest: &str) -> Result<Vec<u8>, ImageError> {
    let (meta, payload) = rest
        .split_once(',')
        .ok_or_else(|| ImageError::UnsupportedReference("data URL without payload".into()))?;
    let mut parts = meta.split(';');
    let media_type = parts.next().unwrap_or_default().trim().to_ascii_lowercase();
    if !matches!(
        media_type.as_str(),
        "image/jpeg" | "image/jpg" | "image/png" | "image/webp"
    ) {
        return Err(ImageError::UnsupportedMediaType(media_type));
    }
    if !parts.any(|p| p.trim().eq_ignore_ascii_case("base64")) {
        return Err(ImageError::UnsupportedReference(
            "data URL is not base64 encoded".into(),
        ));
    }
    let compact: String = payload.chars().filter(|c| !c.is_whitespace()).collect();
    Ok(STANDARD.decode(compact.as_bytes())?)
}

fn read_local(path: &Path, policy: &MediaPolicy) -> Result<Vec<u8>, ImageError> {
    let root = policy
        .allowed_local_media_path
        .as_ref()
        .ok_or(ImageError::LocalMediaDisabled)?;
    let io_err = |path: &Path, source| ImageError::Io {
        path: path.to_path_buf(),
        source,
    };
    let root = root.canonicalize().map_err(|e| io_err(root, e))?;
    let resolved = path.canonicalize().map_err(|e| io_err(path, e))?;
    if !resolved.starts_with(&root) {
        return Err(ImageError::LocalMediaOutsideRoot(path.to_path_buf()));
    }
    std::fs::read(&resolved).map_err(|e| io_err(&resolved, e))
}

fn strip_prefix_ignore_case<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let head = s.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix)
        .then(|| &s[prefix.len()..])
}

fn scheme_of(reference: &str) -> String {
    reference
        .split_once(':')
        .map_or("(none)", |(scheme, _)| scheme)
        .to_string()
}
