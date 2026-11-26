//! Readahead and parallel positional reads of engine-file entries into
//! caller-provided staging buffers, overlapped across load units.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::mpsc::{Receiver, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::engine_file::{EngineFileError, EntryMeta};
use crate::loader::{EngineFile, LayerGroup};

/// Bytes per positional read; larger tensors are split so several threads
/// share one tensor.
const READ_SEGMENT: usize = 8 << 20;

/// Asks the kernel to start reading `[offset, offset + len)` into the page
/// cache.
pub fn advise_willneed(file: &File, offset: u64, len: u64) -> io::Result<()> {
    // SAFETY: posix_fadvise only reads the descriptor and integer arguments.
    let rc = unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_WILLNEED,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(rc))
    }
}

/// Reads each `(offset, buffer)` job fully, spreading fixed-size segments of
/// all jobs across `threads` workers. Buffers are caller-owned so they may be
/// pinned memory.
pub fn read_parallel(file: &File, jobs: &mut [(u64, &mut [u8])], threads: usize) -> io::Result<()> {
    let mut pieces: Vec<(u64, &mut [u8])> = Vec::new();
    for (offset, buf) in jobs.iter_mut() {
        let mut at = *offset;
        for chunk in buf.chunks_mut(READ_SEGMENT) {
            let len = chunk.len() as u64;
            pieces.push((at, chunk));
            at += len;
        }
    }
    let threads = threads.clamp(1, pieces.len().max(1));
    let queue = Mutex::new(pieces);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    loop {
                        let next = queue.lock().expect("queue lock").pop();
                        match next {
                            Some((offset, buf)) => file.read_exact_at(buf, offset)?,
                            None => return Ok::<(), io::Error>(()),
                        }
                    }
                })
            })
            .collect();
        handles.into_iter().try_for_each(|h| {
            h.join()
                .unwrap_or_else(|_| Err(io::Error::other("reader thread panicked")))
        })
    })
}

/// A destination for one tensor's bytes; implementations may be pinned host
/// memory.
pub trait StagingBuffer: Send {
    fn as_slice(&self) -> &[u8];
    fn as_mut_slice(&mut self) -> &mut [u8];
}

/// A source of staging buffers.
pub trait Staging: Send + Sync {
    /// A zero-initialised buffer of exactly `len` bytes.
    fn alloc(&self, len: usize) -> io::Result<Box<dyn StagingBuffer>>;
}

/// Staging in ordinary heap memory.
pub struct HeapStaging;

struct HeapBuffer(Vec<u8>);

impl StagingBuffer for HeapBuffer {
    fn as_slice(&self) -> &[u8] {
        &self.0
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl Staging for HeapStaging {
    fn alloc(&self, len: usize) -> io::Result<Box<dyn StagingBuffer>> {
        Ok(Box::new(HeapBuffer(vec![0u8; len])))
    }
}

/// The bytes of one loaded layer group.
pub struct LoadedGroup {
    pub label: String,
    /// `(entry, bytes)` in group order.
    pub tensors: Vec<(EntryMeta, Box<dyn StagingBuffer>)>,
}

/// Background loader that reads layer groups in file order, up to `lookahead`
/// finished groups ahead of the consumer, so reads of group N+1 overlap with
/// the consumer's host-to-device copy of group N.
pub struct LayerPrefetcher {
    rx: Receiver<Result<LoadedGroup, EngineFileError>>,
    handle: Option<JoinHandle<()>>,
}

impl LayerPrefetcher {
    /// Starts loading. `threads` bounds per-group read parallelism.
    pub fn start(
        engine: Arc<EngineFile>,
        staging: Arc<dyn Staging>,
        threads: usize,
        lookahead: usize,
    ) -> Self {
        let (tx, rx) = sync_channel(lookahead.max(1));
        let handle = std::thread::spawn(move || {
            let path = engine.path().to_path_buf();
            let file = match File::open(&path) {
                Ok(f) => f,
                Err(e) => {
                    let _ = tx.send(Err(EngineFileError::io(&path, e)));
                    return;
                }
            };
            let groups = engine.layer_groups();
            let advise = |g: &LayerGroup| {
                for &i in &g.entries {
                    let e = &engine.entries()[i];
                    let _ = advise_willneed(&file, e.offset, e.len);
                }
            };
            if let Some(first) = groups.first() {
                advise(first);
            }
            for (n, group) in groups.iter().enumerate() {
                if let Some(next) = groups.get(n + 1) {
                    advise(next);
                }
                let result = load_group(&engine, &*staging, &file, &path, group, threads);
                let failed = result.is_err();
                if tx.send(result).is_err() || failed {
                    return;
                }
            }
        });
        Self {
            rx,
            handle: Some(handle),
        }
    }
}

fn load_group(
    engine: &EngineFile,
    staging: &dyn Staging,
    file: &File,
    path: &Path,
    group: &LayerGroup,
    threads: usize,
) -> Result<LoadedGroup, EngineFileError> {
    let metas: Vec<EntryMeta> = group
        .entries
        .iter()
        .map(|&i| engine.entries()[i].clone())
        .collect();
    let mut bufs = Vec::with_capacity(metas.len());
    for m in &metas {
        bufs.push(
            staging
                .alloc(m.len as usize)
                .map_err(|e| EngineFileError::io(path, e))?,
        );
    }
    {
        let mut jobs: Vec<(u64, &mut [u8])> = metas
            .iter()
            .zip(bufs.iter_mut())
            .map(|(m, b)| (m.offset, b.as_mut_slice()))
            .collect();
        read_parallel(file, &mut jobs, threads).map_err(|e| EngineFileError::io(path, e))?;
    }
    Ok(LoadedGroup {
        label: group.label.clone(),
        tensors: metas.into_iter().zip(bufs).collect(),
    })
}

impl Iterator for LayerPrefetcher {
    type Item = Result<LoadedGroup, EngineFileError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.rx.recv().ok()
    }
}

impl Drop for LayerPrefetcher {
    fn drop(&mut self) {
        drop(std::mem::replace(&mut self.rx, sync_channel(1).1));
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
