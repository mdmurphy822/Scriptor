//! Opening engine files and verifying them against their source checkpoint.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use memmap2::{Advice, Mmap};
use sha2::{Digest, Sha256};

use crate::engine_file::{
    EngineFileError, EntryMeta, Header, SourceShard, decode_header, hex, sha256_hex,
};

fn shard_paths(dir: &Path) -> Result<Vec<PathBuf>, EngineFileError> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| EngineFileError::io(dir, e))?
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<_, _>>()
        .map_err(|e| EngineFileError::io(dir, e))?;
    files.retain(|p| p.is_file() && p.extension().is_some_and(|x| x == "safetensors"));
    files.sort();
    Ok(files)
}

/// SHA-256 (hex) of the whole file at `path`, streamed in large reads.
pub fn file_sha256(path: &Path) -> Result<String, EngineFileError> {
    let io = |e| EngineFileError::io(path, e);
    let mut file = File::open(path).map_err(io)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 4 << 20];
    loop {
        let n = file.read(&mut buf).map_err(io)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// Fingerprints every `*.safetensors` shard in `dir`: size, mtime and the
/// SHA-256 of its length prefix plus header JSON. The whole-file `sha256` is
/// computed only when `full` is set and is empty otherwise.
pub fn source_fingerprint(dir: &Path, full: bool) -> Result<Vec<SourceShard>, EngineFileError> {
    let mut out = Vec::new();
    for path in shard_paths(dir)? {
        let io = |e| EngineFileError::io(&path, e);
        let meta = std::fs::metadata(&path).map_err(io)?;
        let mtime_ns = meta
            .modified()
            .map_err(io)?
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let mut file = File::open(&path).map_err(io)?;
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix).map_err(io)?;
        let json_len = u64::from_le_bytes(prefix);
        if json_len > meta.len().saturating_sub(8) {
            return Err(crate::error::WeightsError::BadHeader {
                shard: path.display().to_string(),
                reason: "header length exceeds file size".into(),
            }
            .into());
        }
        let mut head = prefix.to_vec();
        head.resize(8 + json_len as usize, 0);
        file.read_exact(&mut head[8..]).map_err(io)?;
        out.push(SourceShard {
            name: path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            size: meta.len(),
            mtime_ns,
            header_sha256: sha256_hex(&head),
            sha256: if full {
                file_sha256(&path)?
            } else {
                String::new()
            },
        });
    }
    Ok(out)
}

/// A memory-mapped, validated engine file.
pub struct EngineFile {
    path: PathBuf,
    map: Mmap,
    header: Header,
    index: BTreeMap<String, usize>,
}

impl EngineFile {
    /// Maps `path` and validates magic, layout version, header checksum,
    /// total length and tensor placement.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, EngineFileError> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| EngineFileError::io(&path, e))?;
        // SAFETY: the file is a read-only cache artifact; the engine never
        // writes through the mapping and the converter replaces files by
        // atomic rename.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| EngineFileError::io(&path, e))?;
        let header = decode_header(&path, &map)?;
        let index = header
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        Ok(Self {
            path,
            map,
            header,
            index,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn entries(&self) -> &[EntryMeta] {
        &self.header.tensors
    }

    pub fn entry(&self, name: &str) -> Result<&EntryMeta, EngineFileError> {
        self.index
            .get(name)
            .map(|&i| &self.header.tensors[i])
            .ok_or_else(|| EngineFileError::MissingEntry(name.to_string()))
    }

    /// The stored bytes of an entry, straight from the mapping.
    pub fn bytes(&self, entry: &EntryMeta) -> &[u8] {
        &self.map[entry.offset as usize..(entry.offset + entry.len) as usize]
    }

    /// Asks the kernel to read the whole mapping into the page cache.
    pub fn advise_willneed(&self) -> Result<(), EngineFileError> {
        self.map
            .advise(Advice::WillNeed)
            .map_err(|e| EngineFileError::io(&self.path, e))
    }

    /// Compares the recorded source fingerprint (shard names, sizes,
    /// modification times and header hashes) with the shards currently in
    /// `source_dir`; any difference is refused and names each mismatching
    /// shard and field.
    pub fn verify_source(&self, source_dir: &Path) -> Result<(), EngineFileError> {
        self.compare_sources(&source_fingerprint(source_dir, false)?, false)
    }

    /// As [`Self::verify_source`], and additionally hashes every shard's full
    /// contents against the recorded SHA-256.
    pub fn verify_source_contents(&self, source_dir: &Path) -> Result<(), EngineFileError> {
        self.compare_sources(&source_fingerprint(source_dir, true)?, true)
    }

    fn compare_sources(&self, now: &[SourceShard], full: bool) -> Result<(), EngineFileError> {
        let mut bad = Vec::new();
        let current: BTreeMap<&str, &SourceShard> =
            now.iter().map(|s| (s.name.as_str(), s)).collect();
        for rec in &self.header.sources {
            match current.get(rec.name.as_str()) {
                None => bad.push(format!("{}: missing", rec.name)),
                Some(cur) => {
                    if cur.size != rec.size {
                        bad.push(format!("{}: size {} -> {}", rec.name, rec.size, cur.size));
                    }
                    if cur.mtime_ns != rec.mtime_ns {
                        bad.push(format!("{}: mtime changed", rec.name));
                    }
                    if cur.header_sha256 != rec.header_sha256 {
                        bad.push(format!("{}: header hash changed", rec.name));
                    }
                    if full && cur.sha256 != rec.sha256 {
                        bad.push(format!("{}: content hash changed", rec.name));
                    }
                }
            }
        }
        for cur in now {
            if !self.header.sources.iter().any(|r| r.name == cur.name) {
                bad.push(format!("{}: added", cur.name));
            }
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(EngineFileError::SourceMismatch(bad))
        }
    }

    /// Entry indices grouped by load unit in file order, so a consumer can
    /// upload one unit while the next is being read.
    pub fn layer_groups(&self) -> Vec<LayerGroup> {
        let mut groups: Vec<LayerGroup> = Vec::new();
        for (i, t) in self.header.tensors.iter().enumerate() {
            match groups.last_mut() {
                Some(g) if g.label == t.unit => g.entries.push(i),
                _ => groups.push(LayerGroup {
                    label: t.unit.clone(),
                    entries: vec![i],
                }),
            }
        }
        groups
    }
}

/// Entries that are loaded together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerGroup {
    pub label: String,
    /// Indices into [`EngineFile::entries`].
    pub entries: Vec<usize>,
}
