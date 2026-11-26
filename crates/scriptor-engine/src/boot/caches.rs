//! Persistent derived state. Nothing is written anywhere unless a [`CacheDir`] exists; the only
//! way to obtain one is an explicit directory, so a boot without it leaves the filesystem alone.

use std::path::{Path, PathBuf};

use super::BootError;

/// File recording the stage timings of the last successful boot.
pub const BOOT_RECORD: &str = "boot-record.json";

/// An explicitly requested directory for tuning results and boot records.
#[derive(Debug, Clone)]
pub struct CacheDir {
    root: PathBuf,
}

impl CacheDir {
    /// Names `root` as the cache directory; nothing is created until the first write.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The directory path.
    pub fn path(&self) -> &Path {
        &self.root
    }

    /// Path of a file inside the directory.
    pub fn file(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    /// Whether a file of that name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.file(name).is_file()
    }

    /// Reads a file, `None` when absent.
    pub fn read(&self, name: &str) -> Option<Vec<u8>> {
        std::fs::read(self.file(name)).ok()
    }

    /// Writes a file atomically, creating the directory when needed.
    pub fn write(&self, name: &str, bytes: &[u8]) -> Result<(), BootError> {
        std::fs::create_dir_all(&self.root)
            .map_err(|e| BootError::Cache(format!("creating the cache directory: {e}")))?;
        let tmp = self.file(&format!(".{name}.tmp"));
        std::fs::write(&tmp, bytes)
            .and_then(|()| std::fs::rename(&tmp, self.file(name)))
            .map_err(|e| BootError::Cache(format!("writing {name}: {e}")))
    }
}

/// What the cache stage found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheReport {
    /// Cache entries loaded.
    pub loaded: usize,
    /// Items expected but absent; each is reported, none is fatal.
    pub missing: Vec<String>,
}
