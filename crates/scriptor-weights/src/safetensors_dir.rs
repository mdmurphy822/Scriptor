use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde::Deserialize;

use crate::dtype::Dtype;
use crate::error::WeightsError;

/// Location and type of one tensor inside a mapped shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRef {
    /// Index into [`SafetensorsDir::shard_names`].
    pub shard: usize,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Absolute byte range within the shard file.
    pub start: u64,
    pub end: u64,
}

impl TensorRef {
    pub fn byte_len(&self) -> u64 {
        self.end - self.start
    }
}

/// The parsed header of one shard: tensor refs by name plus the data offset.
#[derive(Debug, Clone, Default)]
pub struct ShardHeader {
    pub tensors: BTreeMap<String, TensorRef>,
    /// Byte length of the header block including its 8-byte length prefix.
    pub data_start: u64,
}

#[derive(Deserialize)]
struct RawEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// Parses a safetensors header JSON object (without the length prefix).
/// `data_start` is the file offset at which tensor data begins and `data_len`
/// the number of data bytes available after it, when known.
pub fn parse_header(
    shard: &str,
    shard_index: usize,
    json: &[u8],
    data_start: u64,
    data_len: Option<u64>,
) -> Result<ShardHeader, WeightsError> {
    let bad = |reason: String| WeightsError::BadHeader {
        shard: shard.to_string(),
        reason,
    };
    let value: serde_json::Value = serde_json::from_slice(json).map_err(|e| bad(e.to_string()))?;
    let obj = value
        .as_object()
        .ok_or_else(|| bad("header is not a JSON object".into()))?;
    let mut tensors = BTreeMap::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let bad_tensor = |reason: String| WeightsError::BadTensor {
            shard: shard.to_string(),
            name: name.clone(),
            reason,
        };
        let raw: RawEntry =
            serde_json::from_value(v.clone()).map_err(|e| bad_tensor(e.to_string()))?;
        let dtype = Dtype::parse(&raw.dtype)?;
        let [lo, hi] = raw.data_offsets;
        if hi < lo {
            return Err(bad_tensor(format!("data_offsets [{lo},{hi}] reversed")));
        }
        if let Some(len) = data_len
            && hi > len
        {
            return Err(bad_tensor(format!(
                "data_offsets end {hi} exceeds data section of {len} bytes"
            )));
        }
        let elems = raw
            .shape
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .ok_or_else(|| bad_tensor("shape overflows".into()))?;
        let expected = elems as u64 * dtype.size_bytes() as u64;
        if hi - lo != expected {
            return Err(bad_tensor(format!(
                "byte length {} does not match shape {:?} of {}",
                hi - lo,
                raw.shape,
                raw.dtype
            )));
        }
        tensors.insert(
            name.clone(),
            TensorRef {
                shard: shard_index,
                dtype,
                shape: raw.shape,
                start: data_start + lo,
                end: data_start + hi,
            },
        );
    }
    Ok(ShardHeader {
        tensors,
        data_start,
    })
}

struct Shard {
    name: String,
    map: Mmap,
}

/// A directory of memory-mapped `*.safetensors` shards with a merged
/// tensor table built from the headers alone.
pub struct SafetensorsDir {
    dir: PathBuf,
    shards: Vec<Shard>,
    tensors: BTreeMap<String, TensorRef>,
}

impl SafetensorsDir {
    /// Maps every `*.safetensors` file in `dir`. When
    /// `model.safetensors.index.json` is present it must agree with the
    /// shard headers.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, WeightsError> {
        let dir = dir.as_ref().to_path_buf();
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| WeightsError::io(&dir, e))?
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<_, _>>()
            .map_err(|e| WeightsError::io(&dir, e))?;
        files.retain(|p| p.is_file() && p.extension().is_some_and(|x| x == "safetensors"));
        files.sort();
        if files.is_empty() {
            return Err(WeightsError::NoShards(dir));
        }
        let mut shards = Vec::new();
        let mut tensors: BTreeMap<String, TensorRef> = BTreeMap::new();
        for (i, path) in files.iter().enumerate() {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let file = File::open(path).map_err(|e| WeightsError::io(path, e))?;
            // SAFETY: shards are read-only inputs; the engine never writes to
            // the mapped files, and truncation by another process is outside
            // the contract of a model directory.
            let map = unsafe { Mmap::map(&file) }.map_err(|e| WeightsError::io(path, e))?;
            let header = read_shard_header(&name, i, &map)?;
            for (tname, tref) in header.tensors {
                if let Some(prev) = tensors.get(&tname) {
                    return Err(WeightsError::DuplicateTensor {
                        name: tname,
                        first: shards_name(&shards, prev.shard),
                        second: name,
                    });
                }
                tensors.insert(tname, tref);
            }
            shards.push(Shard { name, map });
        }
        let this = Self {
            dir,
            shards,
            tensors,
        };
        this.check_index()?;
        Ok(this)
    }

    fn check_index(&self) -> Result<(), WeightsError> {
        let path = self.dir.join("model.safetensors.index.json");
        if !path.is_file() {
            return Ok(());
        }
        let bad = |reason: String| WeightsError::BadIndex {
            path: path.clone(),
            reason,
        };
        let text = std::fs::read(&path).map_err(|e| WeightsError::io(&path, e))?;
        #[derive(Deserialize)]
        struct Index {
            weight_map: BTreeMap<String, String>,
        }
        let index: Index = serde_json::from_slice(&text).map_err(|e| bad(e.to_string()))?;
        for (name, shard) in &index.weight_map {
            let t = self
                .tensors
                .get(name)
                .ok_or_else(|| bad(format!("lists {name} but no shard header has it")))?;
            if self.shards[t.shard].name != *shard {
                return Err(bad(format!(
                    "{name} indexed in {shard} but found in {}",
                    self.shards[t.shard].name
                )));
            }
        }
        if let Some(extra) = self
            .tensors
            .keys()
            .find(|n| !index.weight_map.contains_key(*n))
        {
            return Err(bad(format!("shard tensor {extra} missing from weight_map")));
        }
        Ok(())
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn shard_names(&self) -> Vec<&str> {
        self.shards.iter().map(|s| s.name.as_str()).collect()
    }

    /// All tensors by checkpoint name, in sorted order.
    pub fn tensors(&self) -> &BTreeMap<String, TensorRef> {
        &self.tensors
    }

    pub fn get(&self, name: &str) -> Result<&TensorRef, WeightsError> {
        self.tensors
            .get(name)
            .ok_or_else(|| WeightsError::MissingTensor(name.to_string()))
    }

    /// The raw bytes of a tensor from its mapped shard.
    pub fn bytes(&self, t: &TensorRef) -> &[u8] {
        &self.shards[t.shard].map[t.start as usize..t.end as usize]
    }
}

fn shards_name(shards: &[Shard], i: usize) -> String {
    shards.get(i).map(|s| s.name.clone()).unwrap_or_default()
}

fn read_shard_header(name: &str, index: usize, map: &[u8]) -> Result<ShardHeader, WeightsError> {
    let bad = |reason: &str| WeightsError::BadHeader {
        shard: name.to_string(),
        reason: reason.to_string(),
    };
    let prefix: [u8; 8] = map
        .get(..8)
        .ok_or_else(|| bad("file shorter than 8 bytes"))?
        .try_into()
        .expect("slice of 8");
    let len = u64::from_le_bytes(prefix);
    let end = 8u64
        .checked_add(len)
        .filter(|&e| e <= map.len() as u64)
        .ok_or_else(|| bad("header length exceeds file size"))?;
    let json = &map[8..end as usize];
    parse_header(name, index, json, end, Some(map.len() as u64 - end))
}
