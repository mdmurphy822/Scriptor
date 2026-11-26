//! The engine-native weight file: byte layout, header model, encoding and the
//! error type shared by the writer and the loader.
//!
//! File layout (all integers little endian):
//!
//! | bytes      | content                                              |
//! |------------|------------------------------------------------------|
//! | 0..8       | magic `GLMOCRE\0`                                    |
//! | 8..12      | layout version (u32)                                 |
//! | 12..16     | zero                                                 |
//! | 16..24     | header JSON length (u64)                             |
//! | 24..56     | SHA-256 of the header JSON                           |
//! | 56..       | header JSON, then zero padding to a 4096 boundary    |
//! | per tensor | data at its recorded offset, a multiple of 4096      |
//!
//! Tensor data is stored in the operand layout the GEMM and elementwise
//! kernels read: every 2-D weight is row-major `[out, in]`, fused tensors are
//! the dim-0 concatenation of their sources, and convolution weights are
//! flattened to `[out, in * kernel]` in patchify order. See
//! `docs/engine-file.md`.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::dtype::Dtype;
use crate::error::WeightsError;

pub const MAGIC: [u8; 8] = *b"GLMOCRE\0";
/// The tensor layout generation produced by this crate.
pub const LAYOUT_VERSION: u32 = 1;
/// Alignment of the data region and of every tensor offset.
pub const ALIGN: u64 = 4096;
/// Bytes before the header JSON.
pub const PREFIX_LEN: usize = 56;

/// Failures raised while writing, opening or verifying an engine file.
#[derive(Debug, thiserror::Error)]
pub enum EngineFileError {
    #[error(transparent)]
    Weights(#[from] WeightsError),
    #[error("io error on {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("{path}: not an engine file (bad magic)")]
    BadMagic { path: PathBuf },
    #[error("{path}: layout version {found} but this build reads version {expected}")]
    LayoutVersion {
        path: PathBuf,
        found: u32,
        expected: u32,
    },
    #[error("{path}: truncated or corrupt: {reason}")]
    Corrupt { path: PathBuf, reason: String },
    #[error("engine file does not match its source checkpoint: {}", .0.join("; "))]
    SourceMismatch(Vec<String>),
    #[error("tensor {0:?} not present in engine file")]
    MissingEntry(String),
    #[error("cannot convert {name}: {reason}")]
    Unsupported { name: String, reason: String },
}

impl EngineFileError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

/// How a stored tensor was derived from its source tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transform {
    /// The single source's bytes, verbatim.
    Copy,
    /// The sources' bytes concatenated along dimension 0, in order.
    Concat,
    /// The single source's bytes under a flattened shape.
    Reshape,
}

/// One source shard recorded for staleness checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceShard {
    pub name: String,
    pub size: u64,
    pub mtime_ns: u64,
    /// SHA-256 (hex) of the safetensors length prefix plus header JSON.
    pub header_sha256: String,
    /// SHA-256 (hex) of the whole shard file.
    pub sha256: String,
}

/// One stored tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryMeta {
    /// The engine tensor name.
    pub name: String,
    /// Load unit the tensor belongs to (consecutive entries share a unit).
    pub unit: String,
    /// Stored element type, in safetensors spelling.
    pub dtype: String,
    /// Stored shape.
    pub shape: Vec<usize>,
    pub transform: Transform,
    /// Checkpoint tensor names the bytes derive from, in order.
    pub sources: Vec<String>,
    /// Absolute file offset, a multiple of [`ALIGN`].
    pub offset: u64,
    /// Stored byte length.
    pub len: u64,
}

impl EntryMeta {
    pub fn stored_dtype(&self) -> Result<Dtype, WeightsError> {
        Dtype::parse(&self.dtype)
    }
}

/// The decoded header of an engine file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
    pub layout_version: u32,
    /// The checkpoint revision the file was converted from.
    pub revision: String,
    /// Decoder layers excluding the MTP layer.
    pub text_layers: usize,
    pub vision_blocks: usize,
    pub sources: Vec<SourceShard>,
    pub tensors: Vec<EntryMeta>,
    /// Total byte length of the file.
    pub file_len: u64,
}

pub(crate) fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Assigns 4 KiB aligned offsets to every entry (in order) and sets
/// `file_len`. Returns the offset at which tensor data begins.
pub(crate) fn assign_offsets(header: &mut Header) -> u64 {
    let widest = {
        let mut sizing = header.clone();
        for t in &mut sizing.tensors {
            t.offset = u64::MAX;
            t.len = u64::MAX;
        }
        sizing.file_len = u64::MAX;
        serde_json::to_vec(&sizing)
            .expect("header serialises")
            .len()
    };
    let data_start = align_up((PREFIX_LEN + widest) as u64, ALIGN);
    let mut cursor = data_start;
    for t in &mut header.tensors {
        t.offset = cursor;
        cursor = align_up(cursor + t.len, ALIGN);
    }
    header.file_len = cursor;
    data_start
}

/// Serialises the prefix and header JSON, zero padded to `data_start`.
pub(crate) fn encode_header(header: &Header, data_start: u64) -> Vec<u8> {
    let json = serde_json::to_vec(header).expect("header serialises");
    let mut out = Vec::with_capacity(data_start as usize);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&header.layout_version.to_le_bytes());
    out.extend_from_slice(&[0u8; 4]);
    out.extend_from_slice(&(json.len() as u64).to_le_bytes());
    out.extend_from_slice(&Sha256::digest(&json));
    out.extend_from_slice(&json);
    out.resize(data_start as usize, 0);
    out
}

/// Parses and validates the header at the start of `bytes` (the whole file).
pub(crate) fn decode_header(path: &Path, bytes: &[u8]) -> Result<Header, EngineFileError> {
    let corrupt = |reason: String| EngineFileError::Corrupt {
        path: path.to_path_buf(),
        reason,
    };
    let bad_magic = || EngineFileError::BadMagic {
        path: path.to_path_buf(),
    };
    if bytes.len() < 8 || bytes[..8] != MAGIC {
        return Err(bad_magic());
    }
    if bytes.len() < PREFIX_LEN {
        return Err(corrupt(format!(
            "{} bytes is shorter than the {PREFIX_LEN}-byte prefix",
            bytes.len()
        )));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
    if version != LAYOUT_VERSION {
        return Err(EngineFileError::LayoutVersion {
            path: path.to_path_buf(),
            found: version,
            expected: LAYOUT_VERSION,
        });
    }
    let json_len = u64::from_le_bytes(bytes[16..24].try_into().expect("8 bytes"));
    let end = (PREFIX_LEN as u64)
        .checked_add(json_len)
        .filter(|&e| e <= bytes.len() as u64)
        .ok_or_else(|| corrupt("header length exceeds file size".into()))?;
    let json = &bytes[PREFIX_LEN..end as usize];
    if Sha256::digest(json).as_slice() != &bytes[24..56] {
        return Err(corrupt("header checksum mismatch".into()));
    }
    let header: Header =
        serde_json::from_slice(json).map_err(|e| corrupt(format!("header JSON: {e}")))?;
    if header.layout_version != version {
        return Err(corrupt(
            "header layout version disagrees with prefix".into(),
        ));
    }
    if header.file_len != bytes.len() as u64 {
        return Err(corrupt(format!(
            "file is {} bytes but the header records {}",
            bytes.len(),
            header.file_len
        )));
    }
    let mut floor = align_up(end, ALIGN);
    for t in &header.tensors {
        if t.offset % ALIGN != 0 || t.offset < floor {
            return Err(corrupt(format!("tensor {} has a misplaced offset", t.name)));
        }
        let tail = t
            .offset
            .checked_add(t.len)
            .filter(|&e| e <= header.file_len)
            .ok_or_else(|| corrupt(format!("tensor {} extends past the file", t.name)))?;
        floor = tail;
    }
    Ok(header)
}

/// Writes an engine file to `<path>.partial` and renames it into place.
/// `fill` is called once per entry in order and must write exactly
/// `entry.len` bytes. `finish` runs after all data is written and may replace
/// the `sha256` fields of `header.sources` with values of identical length;
/// the header is then rewritten in place.
pub(crate) fn write_file(
    path: &Path,
    header: &mut Header,
    mut fill: impl FnMut(&EntryMeta, &mut dyn Write) -> Result<(), EngineFileError>,
    finish: impl FnOnce(&mut Header) -> Result<(), EngineFileError>,
) -> Result<(), EngineFileError> {
    let data_start = assign_offsets(header);
    let mut partial = path.as_os_str().to_owned();
    partial.push(".partial");
    let partial = PathBuf::from(partial);
    let io_err = |e| EngineFileError::io(&partial, e);
    let file = File::create(&partial).map_err(io_err)?;
    let mut out = CountingWriter {
        inner: BufWriter::with_capacity(1 << 20, file),
        written: 0,
    };
    let result = (|| {
        out.write_all(&encode_header(header, data_start))
            .map_err(io_err)?;
        for entry in &header.tensors {
            out.pad_to(entry.offset).map_err(io_err)?;
            fill(entry, &mut out)?;
            if out.written != entry.offset + entry.len {
                return Err(EngineFileError::Corrupt {
                    path: partial.clone(),
                    reason: format!(
                        "{} wrote {} bytes, expected {}",
                        entry.name,
                        out.written - entry.offset,
                        entry.len
                    ),
                });
            }
        }
        out.pad_to(header.file_len).map_err(io_err)?;
        out.inner.flush().map_err(io_err)?;
        let json_before = serde_json::to_vec(&*header)
            .expect("header serialises")
            .len();
        finish(header)?;
        let json_after = serde_json::to_vec(&*header)
            .expect("header serialises")
            .len();
        let rewritten = encode_header(header, data_start);
        if json_after != json_before || rewritten.len() as u64 != data_start {
            return Err(EngineFileError::Corrupt {
                path: partial.clone(),
                reason: "finalised header no longer fits its reserved space".into(),
            });
        }
        out.inner
            .get_ref()
            .write_all_at(&rewritten, 0)
            .map_err(io_err)?;
        out.inner.get_ref().sync_all().map_err(io_err)
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&partial);
        return Err(e);
    }
    std::fs::rename(&partial, path).map_err(|e| EngineFileError::io(path, e))
}

struct CountingWriter {
    inner: BufWriter<File>,
    written: u64,
}

impl CountingWriter {
    fn pad_to(&mut self, target: u64) -> io::Result<()> {
        const ZEROS: [u8; 4096] = [0; 4096];
        while self.written < target {
            let n = (target - self.written).min(ZEROS.len() as u64) as usize;
            self.write_all(&ZEROS[..n])?;
        }
        Ok(())
    }
}

impl Write for CountingWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
