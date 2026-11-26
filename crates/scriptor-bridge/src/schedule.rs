//! The document scheduler: renders pages, detects layout in batches, and
//! recognises the regions of every page through one shared pool of worker
//! slots, builds each page's regions as the page completes, and finalises
//! the reply.
//!
//! Pages flow through overlapping stages. A small pool of render threads
//! renders pages ahead of the layout stage within a bounded window and may
//! hand them over out of order. The calling thread is the layout stage: it
//! takes whatever rendered pages are ready, up to `layout_batch_size`, never
//! waiting for a batch to fill, runs the layout model on them and queues one
//! job per recognisable region. A pool of `workers` threads takes jobs from
//! the queue in order, whichever page they belong to, crops and encodes the
//! region, and sends it to the seat. Pages restored from a checkpoint and
//! pages with no region to read are completed by a separate pool of
//! completion threads, so finisher work never blocks rendering, layout or
//! submission. A page is complete when every one of its regions is back;
//! pages complete in any order, and a slow region holds only its own page.
//!
//! Memory is bounded: the render-ahead window holds at most a few layout
//! batches of rasters, the job queue holds at most `2 * workers` jobs (one
//! running and one waiting per worker slot), each job shares its page raster,
//! and a raster is released when its page completes.

use crate::assemble::{AssembleError, Assembled, BuiltPage, build_page, finalise, validate};
use crate::client::{OcrClient, Recognition, RegionError};
use crate::crop::crop_data_url;
use crate::format::{PageResult, RegionResult, RegionTask};
use crate::prompt::{self, request_body};
use crate::render::{PageSource, RenderedPage};
use scriptor_layout::{LayoutConfig, LayoutError, LayoutModel, LayoutRegion, Task as LayoutTask};
use serde_json::{Value, json};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

/// The recognition seat as the scheduler uses it: the text for one request
/// body, honouring the cancellation check.
pub trait RegionRecognizer: Sync {
    /// The text the seat reads for `body`.
    fn recognize(&self, body: &Value, cancelled: &dyn Fn() -> bool) -> Result<String, RegionError>;

    /// The text the seat reads for `body` and whether its repetition stop
    /// ended the reply; a recognizer that cannot report the stop reports
    /// none.
    fn recognize_detailed(
        &self,
        body: &Value,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Recognition, RegionError> {
        self.recognize(body, cancelled).map(|text| Recognition {
            text,
            repetition_stopped: false,
        })
    }
}

impl RegionRecognizer for OcrClient {
    fn recognize(&self, body: &Value, cancelled: &dyn Fn() -> bool) -> Result<String, RegionError> {
        OcrClient::recognize(self, body, cancelled)
    }

    fn recognize_detailed(
        &self,
        body: &Value,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Recognition, RegionError> {
        OcrClient::recognize_detailed(self, body, cancelled)
    }
}

/// Why a finisher hook stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinishError {
    /// The cancellation check held during the hook.
    Cancelled,
    /// The hook refused with a message.
    Failed(String),
}

/// Post-processing applied to recognised pages. Every hook defaults to doing
/// nothing.
pub trait PageFinisher: Sync {
    /// Adjusts the recognised `regions` of zero-based page `page_index`
    /// once every region of the page is back. `rendered` is the page's
    /// raster. An error fails the document.
    fn finish_page(
        &self,
        _page_index: usize,
        _rendered: &RenderedPage,
        _regions: &mut Vec<RegionResult>,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Adjusts the built regions of zero-based page `page_index` after the
    /// page's layout results are turned into regions and before region ids,
    /// page-number splitting and the title are settled. `raw` holds the
    /// page's recognised layout results, `dimensions` its raster size in
    /// pixels. A cancellation ends the run; a failure fails the document.
    fn finish_built(
        &self,
        _page_index: usize,
        _rendered: &RenderedPage,
        _raw: &PageResult,
        _dimensions: (f64, f64),
        _built: &mut BuiltPage,
        _cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<(), FinishError> {
        Ok(())
    }

    /// The document's own title when the finisher knows it; it replaces
    /// the configured one when non-empty.
    fn metadata_title(&self) -> Option<String> {
        None
    }

    /// The non-zero diagnostic counters by category name the finisher
    /// accumulated over the run.
    fn diagnostics(&self) -> Vec<(&'static str, usize)> {
        Vec::new()
    }
}

/// A finisher that leaves every page as recognised.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoFinish;

impl PageFinisher for NoFinish {}

/// Settings of one document run.
#[derive(Clone, Debug)]
pub struct ScheduleConfig {
    /// The size of the shared pool of recognition worker slots: the
    /// configured worker concurrency (`SCRIPTOR_BRIDGE_WORKERS`). Zero counts
    /// as one.
    pub workers: usize,
    /// The most pages the layout model sees in one call: the configured
    /// layout batch size (`SCRIPTOR_BRIDGE_LAYOUT_BATCH_SIZE`). Zero counts as
    /// one.
    pub layout_batch_size: usize,
    /// The `model` field of every region request; empty omits it.
    pub model: String,
    /// The document's own title, or empty.
    pub metadata_title: String,
    /// The directory holding per-page checkpoints, when resuming is wanted.
    pub checkpoint_dir: Option<PathBuf>,
    /// The layout detection settings.
    pub layout: LayoutConfig,
}

impl ScheduleConfig {
    /// A configuration with the given pool and batch sizes, no model name,
    /// no title, no checkpoints, and the default layout settings.
    #[must_use]
    pub fn new(workers: usize, layout_batch_size: usize) -> Self {
        Self {
            workers,
            layout_batch_size,
            model: String::new(),
            metadata_title: String::new(),
            checkpoint_dir: None,
            layout: LayoutConfig::default(),
        }
    }
}

/// Why a document run failed.
#[derive(Debug)]
pub enum ScheduleError {
    /// The cancellation check held before all pages were complete.
    Cancelled,
    /// A page could not be rendered.
    Render(anyhow::Error),
    /// The layout model failed.
    Layout(LayoutError),
    /// A finisher hook refused.
    Finish(String),
    /// The recognised pages could not be assembled.
    Assemble(AssembleError),
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => f.write_str("the document run was cancelled"),
            Self::Render(error) => write!(f, "page rendering failed: {error}"),
            Self::Layout(error) => write!(f, "layout detection failed: {error}"),
            Self::Finish(message) => write!(f, "page post-processing failed: {message}"),
            Self::Assemble(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// The recognition task of a layout task; `None` for a dropped region.
#[must_use]
pub fn region_task(task: LayoutTask) -> Option<RegionTask> {
    match task {
        LayoutTask::Text => Some(RegionTask::Text),
        LayoutTask::Table => Some(RegionTask::Table),
        LayoutTask::Formula => Some(RegionTask::Formula),
        LayoutTask::Skip => Some(RegionTask::Skip),
        LayoutTask::Abandon => None,
    }
}

/// The prompt task of a region task; `None` for a skipped region, which is
/// not sent to the seat.
#[must_use]
pub fn prompt_task(task: RegionTask) -> Option<prompt::Task> {
    match task {
        RegionTask::Text => Some(prompt::Task::Text),
        RegionTask::Table => Some(prompt::Task::Table),
        RegionTask::Formula => Some(prompt::Task::Formula),
        RegionTask::Skip => None,
    }
}

fn region_result(region: &LayoutRegion) -> Option<RegionResult> {
    let task = region_task(region.task)?;
    Some(RegionResult {
        label: region.label.clone(),
        task,
        bbox: Some(region.bbox_2d.map(f64::from)),
        content: None,
        repetition_stopped: false,
    })
}

fn task_name(task: RegionTask) -> &'static str {
    match task {
        RegionTask::Text => "text",
        RegionTask::Table => "table",
        RegionTask::Formula => "formula",
        RegionTask::Skip => "skip",
    }
}

fn task_from_name(name: &str) -> Option<RegionTask> {
    match name {
        "text" => Some(RegionTask::Text),
        "table" => Some(RegionTask::Table),
        "formula" => Some(RegionTask::Formula),
        "skip" => Some(RegionTask::Skip),
        _ => None,
    }
}

/// Whether every region that needs recognition has a recognition.
fn fully_recognised(regions: &[RegionResult]) -> bool {
    regions
        .iter()
        .all(|region| region.task == RegionTask::Skip || region.content.is_some())
}

fn checkpoint_path(dir: &Path, page_index: usize) -> PathBuf {
    dir.join(format!("{page_index:06}.json"))
}

/// The recognised regions saved for `page_index`, or `None` when the file is
/// absent, unreadable, malformed, or lacks a recognition a region needs.
fn read_checkpoint(dir: &Path, page_index: usize) -> Option<Vec<RegionResult>> {
    let text = std::fs::read_to_string(checkpoint_path(dir, page_index)).ok()?;
    let saved: Value = serde_json::from_str(&text).ok()?;
    let items = saved.as_object()?.get("regions")?.as_array()?;
    let mut regions = Vec::with_capacity(items.len());
    for item in items {
        let item = item.as_object()?;
        let bbox = match item.get("bbox")? {
            Value::Null => None,
            Value::Array(values) if values.len() == 4 => {
                let mut edges = [0.0; 4];
                for (edge, value) in edges.iter_mut().zip(values) {
                    *edge = value.as_f64()?;
                }
                Some(edges)
            }
            _ => return None,
        };
        let content = match item.get("content")? {
            Value::Null => None,
            Value::String(text) => Some(text.clone()),
            _ => return None,
        };
        regions.push(RegionResult {
            label: item.get("label")?.as_str()?.to_owned(),
            task: task_from_name(item.get("task")?.as_str()?)?,
            bbox,
            content,
            repetition_stopped: item
                .get("repetition_stopped")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        });
    }
    fully_recognised(&regions).then_some(regions)
}

#[cfg(unix)]
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)
}

#[cfg(not(unix))]
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)
}

#[cfg(unix)]
fn create_private_file(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
}

#[cfg(not(unix))]
fn create_private_file(path: &Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
}

fn write_checkpoint_file(
    dir: &Path,
    page_index: usize,
    regions: &[RegionResult],
) -> std::io::Result<()> {
    use std::io::Write;
    create_private_dir(dir)?;
    let items: Vec<Value> = regions
        .iter()
        .map(|region| {
            json!({
                "label": region.label,
                "task": task_name(region.task),
                "bbox": region.bbox,
                "content": region.content,
                "repetition_stopped": region.repetition_stopped,
            })
        })
        .collect();
    let bytes = serde_json::to_vec(&json!({ "regions": items }))
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let path = checkpoint_path(dir, page_index);
    let temporary = dir.join(format!("{page_index:06}.tmp"));
    let _ = std::fs::remove_file(&temporary);
    let written = create_private_file(&temporary).and_then(|mut file| {
        file.write_all(&bytes)?;
        file.sync_all()
    });
    match written.and_then(|()| std::fs::rename(&temporary, &path)) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = std::fs::remove_file(&temporary);
            Err(error)
        }
    }
}

/// Saves the recognised regions of a page atomically with owner-only access.
/// A page with a missing recognition, or one that cannot be written, is left
/// unsaved.
fn save_checkpoint(dir: &Path, page_index: usize, regions: &[RegionResult]) {
    if fully_recognised(regions) {
        let _ = write_checkpoint_file(dir, page_index, regions);
    }
}

struct Job {
    page: usize,
    slot: usize,
    task: prompt::Task,
    bbox: [f64; 4],
    raster: Arc<RenderedPage>,
}

struct PageSlot {
    rendered: Arc<RenderedPage>,
    regions: Vec<RegionResult>,
    remaining: usize,
}

struct DonePage {
    raw: PageResult,
    dimensions: (f64, f64),
    built: Option<BuiltPage>,
}

struct State {
    pending: Vec<Option<PageSlot>>,
    done: Vec<Option<DonePage>>,
    pages_done: usize,
    /// The pages with a region request sent and not yet done.
    reading: std::collections::BTreeSet<usize>,
    /// The pages that are done.
    read: std::collections::BTreeSet<usize>,
}

/// Receives the one-based numbers of the pages being read and of the pages
/// read, each ascending.
pub type PagesCallback<'a> = dyn Fn(&[i64], &[i64]) + Sync + 'a;

/// Receives `(region requests in flight, worker pool size)` on every change
/// of the number of requests in flight.
pub type LoadCallback<'a> = dyn Fn(usize, usize) + Sync + 'a;

/// A page to finish off the layout and submission threads.
struct Completion {
    page: usize,
    slot: PageSlot,
    save: bool,
}

/// Wall-clock and count totals of the stages of one document run.
#[derive(Default)]
struct Metrics {
    render_ns: AtomicU64,
    layout_ns: AtomicU64,
    layout_calls: AtomicU64,
    layout_pages: AtomicU64,
    queue_wait_ns: AtomicU64,
    worker_idle_ns: AtomicU64,
    finish_ns: AtomicU64,
    crops: AtomicU64,
    crops_reused: AtomicU64,
}

fn add_elapsed(counter: &AtomicU64, since: Instant) {
    let nanos = u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX);
    counter.fetch_add(nanos, Ordering::Relaxed);
}

impl Metrics {
    /// The non-zero schedule counters: stage times in whole milliseconds and
    /// the layout call and page counts.
    fn counters(&self) -> Vec<(&'static str, usize)> {
        let millis = |counter: &AtomicU64| (counter.load(Ordering::Relaxed) / 1_000_000) as usize;
        let count = |counter: &AtomicU64| counter.load(Ordering::Relaxed) as usize;
        [
            ("schedule_render_ms", millis(&self.render_ns)),
            ("schedule_layout_ms", millis(&self.layout_ns)),
            ("schedule_layout_calls", count(&self.layout_calls)),
            ("schedule_layout_pages", count(&self.layout_pages)),
            ("schedule_queue_wait_ms", millis(&self.queue_wait_ns)),
            ("schedule_worker_idle_ms", millis(&self.worker_idle_ns)),
            ("schedule_finish_ms", millis(&self.finish_ns)),
            ("schedule_crops", count(&self.crops)),
            ("schedule_crops_reused", count(&self.crops_reused)),
        ]
        .into_iter()
        .filter(|(_, value)| *value >= 1)
        .collect()
    }
}

/// Rendered items re-ordered by page index. An item is a page, or `None`
/// for an index that will produce no layout work.
struct Reorder<T> {
    held: std::collections::BTreeMap<usize, Option<T>>,
    next: usize,
}

impl<T> Reorder<T> {
    fn new() -> Self {
        Self {
            held: std::collections::BTreeMap::new(),
            next: 0,
        }
    }

    /// The lowest index not yet handed out.
    fn next_index(&self) -> usize {
        self.next
    }

    /// The next batch of consecutive pages, up to `max`, starting at the
    /// next expected index. Everything already waiting in `rx` is taken in
    /// first; it blocks only while the next expected index is missing, never
    /// to fill a batch. `None` once all `total` indices are handed out or
    /// every sender is gone.
    fn take_batch(
        &mut self,
        rx: &Receiver<(usize, Option<T>)>,
        max: usize,
        total: usize,
        advanced: &dyn Fn(usize),
    ) -> Option<Vec<(usize, T)>> {
        loop {
            while let Ok((index, item)) = rx.try_recv() {
                self.held.insert(index, item);
            }
            let mut batch = Vec::new();
            while batch.len() < max.max(1) {
                let Some(entry) = self.held.remove(&self.next) else {
                    break;
                };
                batch.extend(entry.map(|item| (self.next, item)));
                self.next += 1;
            }
            if !batch.is_empty() {
                return Some(batch);
            }
            if self.next >= total {
                return None;
            }
            advanced(self.next);
            if !self.held.contains_key(&self.next) {
                let (index, item) = rx.recv().ok()?;
                self.held.insert(index, item);
            }
        }
    }
}

struct Shared<'a> {
    total: usize,
    config: &'a ScheduleConfig,
    recognizer: &'a dyn RegionRecognizer,
    finisher: &'a dyn PageFinisher,
    cancelled: &'a (dyn Fn() -> bool + Sync),
    progress: &'a (dyn Fn(usize, usize) + Sync),
    pages: &'a PagesCallback<'a>,
    load: &'a LoadCallback<'a>,
    /// The lowest page index the layout stage has not yet taken.
    floor: Mutex<usize>,
    floor_changed: std::sync::Condvar,
    /// How far past the floor a page may be rendered.
    horizon: usize,
    workers: usize,
    in_flight: Mutex<usize>,
    metrics: Metrics,
    /// The region requests of this run by the digest of their bytes.
    requests: Mutex<std::collections::HashMap<[u8; 32], Arc<Entry>>>,
    state: Mutex<State>,
    stopped: AtomicBool,
    fatal: Mutex<Option<ScheduleError>>,
}

/// How the first request carrying a given body is going.
enum Reading {
    /// The request is being answered.
    Pending,
    /// The request was answered with this text.
    Done(Recognition),
    /// The request did not produce text; a waiter asks for itself.
    Abandoned,
}

/// The shared state of one distinct request body: its progress, and a
/// signal for requests with identical bytes waiting on it.
struct Entry {
    reading: Mutex<Reading>,
    changed: std::sync::Condvar,
}

/// How long a request waits on an identical request before it checks for
/// cancellation.
const ENTRY_RECHECK: std::time::Duration = std::time::Duration::from_millis(200);

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Shared<'_> {
    /// Reports the pages being read and the pages read to the pages callback.
    fn publish(&self, state: &State) {
        let numbers = |set: &std::collections::BTreeSet<usize>| -> Vec<i64> {
            set.iter().map(|page| *page as i64 + 1).collect()
        };
        (self.pages)(&numbers(&state.reading), &numbers(&state.read));
    }

    /// Marks `page` as being read from its first region request, unless it
    /// is already being read or done.
    fn begin_reading(&self, page: usize) {
        let mut state = lock(&self.state);
        if state.read.contains(&page) || !state.reading.insert(page) {
            return;
        }
        self.publish(&state);
    }

    /// Whether work must stop: a failure was recorded or the cancellation
    /// check holds.
    fn should_stop(&self) -> bool {
        if self.stopped.load(Ordering::SeqCst) {
            return true;
        }
        if (self.cancelled)() {
            self.stopped.store(true, Ordering::SeqCst);
            return true;
        }
        false
    }

    fn fail(&self, error: ScheduleError) {
        let mut fatal = lock(&self.fatal);
        if fatal.is_none() {
            *fatal = Some(error);
        }
        self.stopped.store(true, Ordering::SeqCst);
    }

    /// Finishes a page whose regions are all back: saves its checkpoint when
    /// `save` holds, applies the finisher, builds the page's regions and
    /// applies the finisher's region hook, stores the result, and reports
    /// progress.
    fn complete(&self, page: usize, slot: PageSlot, save: bool) {
        let PageSlot {
            rendered,
            mut regions,
            ..
        } = slot;
        if let (true, Some(dir)) = (save, &self.config.checkpoint_dir) {
            save_checkpoint(dir, page, &regions);
        }
        let finishing = Instant::now();
        let finished = self.finisher.finish_page(page, &rendered, &mut regions);
        add_elapsed(&self.metrics.finish_ns, finishing);
        if let Err(message) = finished {
            self.fail(ScheduleError::Finish(message));
            return;
        }
        let dimensions = rendered.pixel_size;
        let raw = PageResult { regions };
        let built = if fully_recognised(&raw.regions) {
            let mut built = match build_page(&raw, page as i64 + 1, dimensions) {
                Ok(built) => built,
                Err(error) => {
                    self.fail(ScheduleError::Assemble(error));
                    return;
                }
            };
            let finishing = Instant::now();
            let finished = self.finisher.finish_built(
                page,
                &rendered,
                &raw,
                dimensions,
                &mut built,
                self.cancelled,
            );
            add_elapsed(&self.metrics.finish_ns, finishing);
            match finished {
                Ok(()) => {}
                Err(FinishError::Cancelled) => {
                    self.stopped.store(true, Ordering::SeqCst);
                    return;
                }
                Err(FinishError::Failed(message)) => {
                    self.fail(ScheduleError::Finish(message));
                    return;
                }
            }
            Some(built)
        } else {
            None
        };
        drop(rendered);
        let mut state = lock(&self.state);
        state.done[page] = Some(DonePage {
            raw,
            dimensions,
            built,
        });
        state.pages_done += 1;
        state.reading.remove(&page);
        state.read.insert(page);
        (self.progress)(state.pages_done, self.total);
        self.publish(&state);
    }

    /// Stores the recognition of one region and, when it was the last one
    /// outstanding, hands its page to the completion pool so this thread can
    /// take the next job. The hand-off waits only for room in the bounded
    /// completion channel, which the completion threads always drain.
    fn record(
        &self,
        page: usize,
        slot: usize,
        content: Option<Recognition>,
        completions: &SyncSender<Completion>,
    ) {
        let finished = {
            let mut state = lock(&self.state);
            let Some(entry) = state.pending[page].as_mut() else {
                return;
            };
            if let Some(recognition) = content {
                entry.regions[slot].content = Some(recognition.text);
                entry.regions[slot].repetition_stopped = recognition.repetition_stopped;
            }
            entry.remaining -= 1;
            if entry.remaining == 0 {
                state.pending[page].take()
            } else {
                None
            }
        };
        if let Some(entry) = finished {
            let _ = completions.send(Completion {
                page,
                slot: entry,
                save: true,
            });
        }
    }

    /// Registers a page whose regions are to be recognised and queues its
    /// jobs; returns `false` when work stopped before every job was queued.
    fn submit(
        &self,
        page: usize,
        rendered: RenderedPage,
        regions: Vec<RegionResult>,
        queue: &SyncSender<Job>,
        completions: &SyncSender<Completion>,
    ) -> bool {
        let rendered = Arc::new(rendered);
        let jobs: Vec<(usize, prompt::Task, [f64; 4])> = regions
            .iter()
            .enumerate()
            .filter_map(|(slot, region)| {
                let task = prompt_task(region.task)?;
                Some((slot, task, region.bbox.unwrap_or([0.0; 4])))
            })
            .collect();
        if jobs.is_empty() {
            return completions
                .send(Completion {
                    page,
                    slot: PageSlot {
                        rendered,
                        regions,
                        remaining: 0,
                    },
                    save: true,
                })
                .is_ok();
        }
        lock(&self.state).pending[page] = Some(PageSlot {
            rendered: Arc::clone(&rendered),
            regions,
            remaining: jobs.len(),
        });
        for (slot, task, bbox) in jobs {
            if self.should_stop() {
                return false;
            }
            let job = Job {
                page,
                slot,
                task,
                bbox,
                raster: Arc::clone(&rendered),
            };
            let waiting = Instant::now();
            let sent = queue.send(job);
            add_elapsed(&self.metrics.queue_wait_ns, waiting);
            if sent.is_err() {
                return false;
            }
        }
        true
    }

    /// Runs the layout model over a batch and submits each page's regions;
    /// returns `false` when work stopped.
    fn flush(
        &self,
        batch: &mut Vec<(usize, RenderedPage)>,
        layout: &dyn LayoutModel,
        queue: &SyncSender<Job>,
        completions: &SyncSender<Completion>,
    ) -> bool {
        if batch.is_empty() {
            return true;
        }
        if self.should_stop() {
            return false;
        }
        let mut pages = std::mem::take(batch);
        let images: Vec<image::RgbImage> = pages
            .iter_mut()
            .map(|(_, rendered)| std::mem::replace(&mut rendered.image, image::RgbImage::new(0, 0)))
            .collect();
        let detecting = Instant::now();
        let detected = scriptor_layout::detect_regions(layout, &images, &self.config.layout);
        add_elapsed(&self.metrics.layout_ns, detecting);
        self.metrics.layout_calls.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .layout_pages
            .fetch_add(images.len() as u64, Ordering::Relaxed);
        for ((_, rendered), image) in pages.iter_mut().zip(images) {
            rendered.image = image;
        }
        let detected = match detected {
            Ok(detected) => detected,
            Err(error) => {
                self.fail(ScheduleError::Layout(error));
                return false;
            }
        };
        for ((index, rendered), regions) in pages.into_iter().zip(detected) {
            let regions: Vec<RegionResult> = regions.iter().filter_map(region_result).collect();
            if !self.submit(index, rendered, regions, queue, completions) {
                return false;
            }
        }
        true
    }

    /// One render thread: claims page indices in ascending order, renders
    /// each, and hands it to the layout stage, or to the completion stage
    /// when a valid checkpoint holds the page. Ends when the pages run out,
    /// work stops, or the receiving side is gone; a stop releases every
    /// render thread waiting on the horizon.
    fn render_stage(
        &self,
        source: &PageSource,
        next: &AtomicUsize,
        rendered: &SyncSender<(usize, Option<RenderedPage>)>,
        completions: &SyncSender<Completion>,
    ) {
        self.render_pages(source, next, rendered, completions);
        if self.should_stop() {
            self.advance_floor(usize::MAX);
        }
    }

    fn render_pages(
        &self,
        source: &PageSource,
        next: &AtomicUsize,
        rendered: &SyncSender<(usize, Option<RenderedPage>)>,
        completions: &SyncSender<Completion>,
    ) {
        loop {
            let index = next.fetch_add(1, Ordering::SeqCst);
            if index >= self.total || self.should_stop() {
                return;
            }
            self.await_horizon(index);
            let rendering = Instant::now();
            let page = source.render_page(index);
            add_elapsed(&self.metrics.render_ns, rendering);
            let page = match page {
                Ok(page) => page,
                Err(error) => {
                    self.fail(ScheduleError::Render(error));
                    return;
                }
            };
            let saved = self
                .config
                .checkpoint_dir
                .as_deref()
                .and_then(|dir| read_checkpoint(dir, index));
            let sent = match saved {
                Some(regions) => {
                    completions
                        .send(Completion {
                            page: index,
                            slot: PageSlot {
                                rendered: Arc::new(page),
                                regions,
                                remaining: 0,
                            },
                            save: false,
                        })
                        .is_ok()
                        && rendered.send((index, None)).is_ok()
                }
                None => rendered.send((index, Some(page))).is_ok(),
            };
            if !sent {
                return;
            }
        }
    }

    /// Blocks a render thread until page `index` is within the render-ahead
    /// horizon of the layout stage. The layout stage always needs the lowest
    /// unrendered index, which is never held back, so this cannot deadlock;
    /// the pages rendered but not yet laid out stay within the horizon (the
    /// channel window plus one page in hand per render thread).
    fn await_horizon(&self, index: usize) {
        let mut floor = lock(&self.floor);
        while index >= floor.saturating_add(self.horizon) {
            floor = self
                .floor_changed
                .wait(floor)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Moves the horizon floor to `floor` and wakes waiting render threads.
    fn advance_floor(&self, floor: usize) {
        *lock(&self.floor) = floor;
        self.floor_changed.notify_all();
    }

    /// The calling thread's stage: takes the rendered pages in page order,
    /// up to a layout batch of consecutive pages that are ready, detects
    /// their layout and queues their jobs. Pages that arrive early wait in a
    /// reorder buffer bounded by the render-ahead horizon.
    fn layout_stage(
        &self,
        rendered: Receiver<(usize, Option<RenderedPage>)>,
        layout: &dyn LayoutModel,
        queue: SyncSender<Job>,
        completions: SyncSender<Completion>,
    ) {
        let batch_size = self.config.layout_batch_size.max(1);
        let mut order = Reorder::new();
        loop {
            if self.should_stop() {
                break;
            }
            let advanced = |next: usize| self.advance_floor(next);
            let Some(mut batch) = order.take_batch(&rendered, batch_size, self.total, &advanced)
            else {
                break;
            };
            self.advance_floor(order.next_index());
            if !self.flush(&mut batch, layout, &queue, &completions) {
                break;
            }
        }
        self.advance_floor(usize::MAX);
    }

    /// One completion thread: finishes pages handed over by the render,
    /// layout and worker stages until the channel closes.
    fn complete_stage(&self, completions: &Mutex<Receiver<Completion>>) {
        loop {
            let next = lock(completions).recv();
            let Ok(Completion { page, slot, save }) = next else {
                return;
            };
            if self.should_stop() {
                continue;
            }
            self.complete(page, slot, save);
        }
    }

    /// Writes one line of the run's stage totals to standard error.
    fn log_summary(&self, pages: usize) {
        use std::io::Write;
        let millis = |counter: &AtomicU64| counter.load(Ordering::Relaxed) / 1_000_000;
        let metrics = &self.metrics;
        let _ = writeln!(
            std::io::stderr(),
            "ocr-bridge: schedule pages={pages} workers={} render_ms={} layout_ms={} \
             layout_calls={} layout_pages={} queue_wait_ms={} worker_idle_ms={} finish_ms={} \
             crops={} crops_reused={}",
            self.workers,
            millis(&metrics.render_ns),
            millis(&metrics.layout_ns),
            metrics.layout_calls.load(Ordering::Relaxed),
            metrics.layout_pages.load(Ordering::Relaxed),
            millis(&metrics.queue_wait_ns),
            millis(&metrics.worker_idle_ns),
            millis(&metrics.finish_ns),
            metrics.crops.load(Ordering::Relaxed),
            metrics.crops_reused.load(Ordering::Relaxed),
        );
    }

    /// Counts a region request starting or ending and reports the load.
    fn adjust_load(&self, starting: bool) {
        let mut in_flight = lock(&self.in_flight);
        if starting {
            *in_flight += 1;
        } else {
            *in_flight = in_flight.saturating_sub(1);
        }
        (self.load)(*in_flight, self.workers);
    }

    /// The reading for `body`, asking the seat once per distinct byte sequence
    /// of this run: a request whose bytes equal a request already answered
    /// takes that text, and one whose twin is still being answered waits for
    /// it. A twin that produced no text leaves the waiter to ask for itself.
    fn read_region(&self, body: &Value) -> Result<Recognition, RegionError> {
        use sha2::{Digest, Sha256};
        self.metrics.crops.fetch_add(1, Ordering::Relaxed);
        let Ok(bytes) = serde_json::to_vec(body) else {
            return self.recognizer.recognize_detailed(body, self.cancelled);
        };
        let key: [u8; 32] = Sha256::digest(&bytes).into();
        loop {
            let (entry, owner) = {
                let mut requests = lock(&self.requests);
                match requests.get(&key) {
                    Some(entry) => (Arc::clone(entry), false),
                    None => {
                        let entry = Arc::new(Entry {
                            reading: Mutex::new(Reading::Pending),
                            changed: std::sync::Condvar::new(),
                        });
                        requests.insert(key, Arc::clone(&entry));
                        (entry, true)
                    }
                }
            };
            if owner {
                self.adjust_load(true);
                let outcome = self.recognizer.recognize_detailed(body, self.cancelled);
                self.adjust_load(false);
                match &outcome {
                    Ok(found) => *lock(&entry.reading) = Reading::Done(found.clone()),
                    Err(_) => {
                        lock(&self.requests).remove(&key);
                        *lock(&entry.reading) = Reading::Abandoned;
                    }
                }
                entry.changed.notify_all();
                return outcome;
            }
            let mut reading = lock(&entry.reading);
            loop {
                match &*reading {
                    Reading::Done(found) => {
                        self.metrics.crops_reused.fetch_add(1, Ordering::Relaxed);
                        return Ok(found.clone());
                    }
                    Reading::Abandoned => break,
                    Reading::Pending => {}
                }
                if self.should_stop() {
                    return Err(RegionError::Cancelled);
                }
                reading = entry
                    .changed
                    .wait_timeout(reading, ENTRY_RECHECK)
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0;
            }
        }
    }

    /// One pool slot: takes jobs until the queue closes, handing each page
    /// whose last region it recognises to `completions`.
    fn work(&self, queue: &Mutex<Receiver<Job>>, completions: &SyncSender<Completion>) {
        loop {
            let idle = Instant::now();
            let job = lock(queue).recv();
            add_elapsed(&self.metrics.worker_idle_ns, idle);
            let Ok(job) = job else {
                return;
            };
            if self.should_stop() {
                continue;
            }
            let Job {
                page,
                slot,
                task,
                bbox,
                raster,
            } = job;
            let url = crop_data_url(&raster.image, bbox);
            drop(raster);
            self.begin_reading(page);
            let content = match url {
                Ok(url) => {
                    let body = request_body(&self.config.model, task, &url);
                    match self.read_region(&body) {
                        Ok(found) => Some(found),
                        Err(RegionError::Cancelled) => {
                            self.stopped.store(true, Ordering::SeqCst);
                            continue;
                        }
                        Err(RegionError::Failed(_)) => None,
                    }
                }
                Err(_) => None,
            };
            self.record(page, slot, content, completions);
        }
    }
}

/// Runs a whole document: renders every page of `source`, detects layout with
/// `layout`, recognises every region through `recognizer` on a shared pool of
/// `config.workers` slots, and assembles the reply.
///
/// `progress` receives `(pages done, page total)`: first `(0, total)`, then
/// once per completed page, with the count never decreasing. `cancelled` is
/// checked before every page render, layout batch and region submission; when
/// it holds, no further work is submitted, running requests end, and the run
/// returns [`ScheduleError::Cancelled`]. With `config.checkpoint_dir`, each
/// fully recognised page is saved as `<page index>.json` and a valid saved
/// page is reused without layout or recognition.
///
/// # Errors
///
/// Returns the first render, layout or finisher failure, the cancellation,
/// or the assembly refusal (for example a region that never got text).
pub fn run_document(
    source: &PageSource,
    layout: &dyn LayoutModel,
    recognizer: &dyn RegionRecognizer,
    finisher: &dyn PageFinisher,
    config: &ScheduleConfig,
    cancelled: &(dyn Fn() -> bool + Sync),
    progress: &(dyn Fn(usize, usize) + Sync),
) -> Result<Assembled, ScheduleError> {
    run_document_reporting(
        source,
        layout,
        recognizer,
        finisher,
        config,
        cancelled,
        progress,
        &|_, _| {},
    )
}

/// [`run_document`] that also reports which pages are being read. `pages`
/// receives `(pages being read, pages read)` as one-based ascending numbers:
/// first `([], [])`, then on every change. A page is being read from the
/// first region request sent for it until it is done; a page restored from
/// a checkpoint or without a region to read goes straight to the pages read.
///
/// # Errors
///
/// Fails as [`run_document`] does.
#[allow(clippy::too_many_arguments)]
pub fn run_document_reporting(
    source: &PageSource,
    layout: &dyn LayoutModel,
    recognizer: &dyn RegionRecognizer,
    finisher: &dyn PageFinisher,
    config: &ScheduleConfig,
    cancelled: &(dyn Fn() -> bool + Sync),
    progress: &(dyn Fn(usize, usize) + Sync),
    pages: &PagesCallback<'_>,
) -> Result<Assembled, ScheduleError> {
    run_document_observed(
        source,
        layout,
        recognizer,
        finisher,
        config,
        cancelled,
        progress,
        pages,
        &|_, _| {},
    )
}

/// [`run_document_reporting`] that also reports the load: `load` receives
/// `(region requests in flight, worker pool size)` on every change. The
/// returned diagnostics carry the finisher's counters followed by the
/// non-zero `schedule_*` stage counters (render, layout, queue wait, worker
/// idle and finisher times in milliseconds, layout calls and pages), and one
/// summary line of the same numbers is written to standard error.
///
/// # Errors
///
/// Fails as [`run_document`] does.
#[allow(clippy::too_many_arguments)]
pub fn run_document_observed(
    source: &PageSource,
    layout: &dyn LayoutModel,
    recognizer: &dyn RegionRecognizer,
    finisher: &dyn PageFinisher,
    config: &ScheduleConfig,
    cancelled: &(dyn Fn() -> bool + Sync),
    progress: &(dyn Fn(usize, usize) + Sync),
    pages: &PagesCallback<'_>,
    load: &LoadCallback<'_>,
) -> Result<Assembled, ScheduleError> {
    let total = source.page_count();
    let workers = config.workers.max(1);
    progress(0, total);
    pages(&[], &[]);
    load(0, workers);
    let batch_size = config.layout_batch_size.max(1);
    let renderers = std::thread::available_parallelism()
        .map_or(1, |cpus| (cpus.get() / 4).max(1))
        .min(total.max(1));
    let window = batch_size.saturating_mul(2).max(renderers);
    let shared = Shared {
        total,
        config,
        recognizer,
        finisher,
        cancelled,
        progress,
        pages,
        load,
        floor: Mutex::new(0),
        floor_changed: std::sync::Condvar::new(),
        horizon: window.saturating_add(renderers),
        workers,
        in_flight: Mutex::new(0),
        metrics: Metrics::default(),
        requests: Mutex::new(std::collections::HashMap::new()),
        state: Mutex::new(State {
            pending: (0..total).map(|_| None).collect(),
            done: (0..total).map(|_| None).collect(),
            pages_done: 0,
            reading: Default::default(),
            read: Default::default(),
        }),
        stopped: AtomicBool::new(false),
        fatal: Mutex::new(None),
    };
    let (sender, receiver) = sync_channel::<Job>(workers.saturating_mul(2));
    let receiver = Mutex::new(receiver);
    let (rendered_tx, rendered_rx) = sync_channel::<(usize, Option<RenderedPage>)>(window);
    let (completion_tx, completion_rx) = sync_channel::<Completion>(window);
    let completion_rx = Mutex::new(completion_rx);
    let next_page = AtomicUsize::new(0);
    let shared_ref = &shared;
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let completions = completion_tx.clone();
            let queue = &receiver;
            scope.spawn(move || shared_ref.work(queue, &completions));
        }
        for _ in 0..workers {
            scope.spawn(|| shared_ref.complete_stage(&completion_rx));
        }
        for _ in 0..renderers {
            let rendered = rendered_tx.clone();
            let completions = completion_tx.clone();
            let next = &next_page;
            scope.spawn(move || shared_ref.render_stage(source, next, &rendered, &completions));
        }
        drop(rendered_tx);
        shared_ref.layout_stage(rendered_rx, layout, sender, completion_tx);
    });
    shared.log_summary(total);
    if let Some(error) = lock(&shared.fatal).take() {
        return Err(error);
    }
    let state = shared
        .state
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.pages_done < total {
        return Err(ScheduleError::Cancelled);
    }
    let mut pages = Vec::with_capacity(total);
    let mut dimensions = Vec::with_capacity(total);
    let mut built_pages = Vec::with_capacity(total);
    for entry in state.done {
        let done = entry.ok_or(ScheduleError::Cancelled)?;
        pages.push(done.raw);
        dimensions.push(done.dimensions);
        built_pages.push(done.built);
    }
    let blank_omitted = validate(&pages).map_err(ScheduleError::Assemble)?;
    let mut title = String::new();
    let mut reply_pages = Vec::with_capacity(total);
    for (offset, built) in built_pages.into_iter().enumerate() {
        let page_number = offset as i64 + 1;
        let built = match built {
            Some(built) => built,
            None => build_page(&pages[offset], page_number, dimensions[offset])
                .map_err(ScheduleError::Assemble)?,
        };
        if title.is_empty() {
            title = built.title;
        }
        reply_pages.push((page_number, built.regions));
    }
    let metadata_title = finisher
        .metadata_title()
        .filter(|title| !title.is_empty())
        .unwrap_or_else(|| config.metadata_title.clone());
    let mut assembled = finalise(reply_pages, &title, &metadata_title, blank_omitted)
        .map_err(ScheduleError::Assemble)?;
    assembled.diagnostics = finisher.diagnostics();
    assembled.diagnostics.extend(shared.metrics.counters());
    Ok(assembled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{FailureClass, RegionFailure, TailOutcome};
    use crate::raster::fixtures::{pdf, pdf_drawn};
    use scriptor_layout::{MockLayout, NUM_CLASSES, RawOutputs};
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    const PAGE_BOX: &str = "[0 0 100 100]";

    fn source(pages: usize) -> PageSource {
        let contents: Vec<String> = (0..pages)
            .map(|page| {
                format!(
                    "{:.3} g 0 50 100 50 re f {:.3} g 0 0 100 50 re f",
                    page as f64 / 64.0,
                    0.5 + page as f64 / 64.0
                )
            })
            .collect();
        let entries: Vec<(&str, &str, &str)> = contents
            .iter()
            .map(|content| (PAGE_BOX, "", content.as_str()))
            .collect();
        PageSource::open(&pdf_drawn(&entries)).unwrap()
    }

    fn mock(batch: usize) -> MockLayout {
        let mut raw = RawOutputs::blank(300, NUM_CLASSES);
        raw.set_query(0, 22, 4.0, [0.5, 0.25, 0.5, 0.125]);
        raw.set_query(1, 6, 4.0, [0.5, 0.75, 0.5, 0.125]);
        raw.set_order(0, 1, 6.0);
        MockLayout::new(vec![raw; batch])
    }

    struct CountingLayout {
        inner: MockLayout,
        calls: AtomicUsize,
    }

    impl LayoutModel for CountingLayout {
        fn detect(&self, pages: &[image::RgbImage]) -> Result<Vec<RawOutputs>, LayoutError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.detect(pages)
        }
    }

    type Reply = Box<dyn Fn(usize) -> (Duration, Result<String, RegionError>) + Sync>;

    struct Stub {
        calls: AtomicUsize,
        active: AtomicUsize,
        peak: AtomicUsize,
        reply: Reply,
    }

    impl Stub {
        fn new(
            reply: impl Fn(usize) -> (Duration, Result<String, RegionError>) + Sync + 'static,
        ) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                active: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
                reply: Box::new(reply),
            }
        }
    }

    impl RegionRecognizer for Stub {
        fn recognize(
            &self,
            _body: &Value,
            _cancelled: &dyn Fn() -> bool,
        ) -> Result<String, RegionError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            let (delay, result) = (self.reply)(call);
            std::thread::sleep(delay);
            self.active.fetch_sub(1, Ordering::SeqCst);
            result
        }
    }

    fn ok(delay_ms: u64) -> (Duration, Result<String, RegionError>) {
        (Duration::from_millis(delay_ms), Ok("hello".to_owned()))
    }

    struct Order(Mutex<Vec<usize>>);

    impl PageFinisher for Order {
        fn finish_page(
            &self,
            index: usize,
            _: &RenderedPage,
            _: &mut Vec<RegionResult>,
        ) -> Result<(), String> {
            self.0.lock().unwrap().push(index);
            Ok(())
        }
    }

    fn never() -> bool {
        false
    }

    /// A recognizer whose first request waits until it is opened.
    struct Held {
        state: Mutex<(usize, bool)>,
        changed: std::sync::Condvar,
    }

    impl RegionRecognizer for Held {
        fn recognize(
            &self,
            _body: &Value,
            _cancelled: &dyn Fn() -> bool,
        ) -> Result<String, RegionError> {
            let mut state = self.state.lock().unwrap();
            state.0 += 1;
            let first = state.0 == 1;
            self.changed.notify_all();
            while first && !state.1 {
                state = self.changed.wait(state).unwrap();
            }
            Ok("hello".to_owned())
        }
    }

    #[test]
    fn a_page_is_being_read_from_its_first_request_until_it_is_done() {
        let held = Held {
            state: Mutex::new((0, false)),
            changed: std::sync::Condvar::new(),
        };
        let latest = Mutex::new((Vec::new(), Vec::new()));
        let history = Mutex::new(Vec::new());
        let config = ScheduleConfig::new(1, 1);
        let report = |reading: &[i64], read: &[i64]| {
            *latest.lock().unwrap() = (reading.to_vec(), read.to_vec());
            history
                .lock()
                .unwrap()
                .push((reading.to_vec(), read.to_vec()));
        };
        std::thread::scope(|scope| {
            let run = scope.spawn(|| {
                run_document_reporting(
                    &source(2),
                    &mock(1),
                    &held,
                    &NoFinish,
                    &config,
                    &never,
                    &|_, _| {},
                    &report,
                )
                .unwrap()
            });
            let mut state = held.state.lock().unwrap();
            while state.0 == 0 {
                state = held.changed.wait(state).unwrap();
            }
            drop(state);
            assert_eq!(*latest.lock().unwrap(), (vec![1], vec![]));
            held.state.lock().unwrap().1 = true;
            held.changed.notify_all();
            run.join().unwrap();
        });
        assert_eq!(*latest.lock().unwrap(), (vec![], vec![1, 2]));
        let history = history.into_inner().unwrap();
        assert_eq!(history.first(), Some(&(vec![], vec![])));
        assert!(history.contains(&(vec![2], vec![1])));
    }

    #[test]
    fn task_enums_map_explicitly() {
        assert_eq!(region_task(LayoutTask::Table), Some(RegionTask::Table));
        assert_eq!(region_task(LayoutTask::Skip), Some(RegionTask::Skip));
        assert_eq!(region_task(LayoutTask::Abandon), None);
        assert_eq!(
            prompt_task(RegionTask::Formula),
            Some(prompt::Task::Formula)
        );
        assert_eq!(prompt_task(RegionTask::Skip), None);
    }

    fn identical_pages(pages: usize) -> PageSource {
        let entries: Vec<(&str, &str)> = (0..pages).map(|_| (PAGE_BOX, "")).collect();
        PageSource::open(&pdf(&entries)).unwrap()
    }

    fn counter(result: &Assembled, name: &str) -> usize {
        result
            .diagnostics
            .iter()
            .find(|(key, _)| *key == name)
            .map_or(0, |(_, value)| *value)
    }

    #[test]
    fn byte_identical_regions_are_asked_once_even_while_in_flight() {
        let stub = Stub::new(|_| ok(150));
        let config = ScheduleConfig::new(6, 2);
        let result = run_document(
            &identical_pages(6),
            &mock(2),
            &stub,
            &NoFinish,
            &config,
            &never,
            &|_, _| {},
        )
        .unwrap();
        assert_eq!(stub.calls.load(Ordering::SeqCst), 2);
        assert_eq!(result.regions.len(), 12);
        assert_eq!(counter(&result, "schedule_crops"), 12);
        assert_eq!(counter(&result, "schedule_crops_reused"), 10);
    }

    #[test]
    fn a_failed_twin_is_not_shared() {
        let stub = Stub::new(|call| {
            if call == 0 {
                (
                    Duration::from_millis(100),
                    Err(RegionError::Failed(RegionFailure {
                        class: FailureClass::PermanentClient,
                        status: Some(400),
                        detail: String::new(),
                        tail: TailOutcome::NotRetryable,
                    })),
                )
            } else {
                ok(0)
            }
        });
        let config = ScheduleConfig::new(2, 1);
        let _ = run_document(
            &identical_pages(1),
            &mock(1),
            &stub,
            &NoFinish,
            &config,
            &never,
            &|_, _| {},
        );
        assert_eq!(stub.calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn the_seat_stays_saturated_across_pages() {
        let stub = Stub::new(|_| ok(30));
        let config = ScheduleConfig::new(3, 2);
        let result = run_document(
            &source(6),
            &mock(2),
            &stub,
            &NoFinish,
            &config,
            &never,
            &|_, _| {},
        )
        .unwrap();
        assert_eq!(stub.calls.load(Ordering::SeqCst), 12);
        assert_eq!(stub.peak.load(Ordering::SeqCst), 3);
        assert_eq!(result.regions.len(), 12);
    }

    /// A recognizer whose first request returns only once `others` pages have
    /// been finished, failing the request if that takes longer than ten seconds.
    struct FirstAfterOthers<'a> {
        calls: AtomicUsize,
        finished: &'a Order,
        others: usize,
    }

    impl RegionRecognizer for FirstAfterOthers<'_> {
        fn recognize(
            &self,
            _body: &Value,
            _cancelled: &dyn Fn() -> bool,
        ) -> Result<String, RegionError> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                let deadline = std::time::Instant::now() + Duration::from_secs(10);
                while self.finished.0.lock().unwrap().len() < self.others {
                    if std::time::Instant::now() >= deadline {
                        return Err(RegionError::Cancelled);
                    }
                    std::thread::yield_now();
                }
            }
            Ok("hello".to_owned())
        }
    }

    #[test]
    fn progress_is_monotone_and_counts_out_of_order_pages() {
        let order = Order(Mutex::new(Vec::new()));
        let stub = FirstAfterOthers {
            calls: AtomicUsize::new(0),
            finished: &order,
            others: 3,
        };
        let seen = Mutex::new(Vec::new());
        let config = ScheduleConfig::new(2, 4);
        run_document(
            &source(4),
            &mock(4),
            &stub,
            &order,
            &config,
            &never,
            &|done, total| seen.lock().unwrap().push((done, total)),
        )
        .unwrap();
        let seen = seen.into_inner().unwrap();
        assert_eq!(seen, vec![(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]);
        let finished = order.0.into_inner().unwrap();
        assert_eq!(finished.last(), Some(&0), "{finished:?}");
    }

    #[test]
    fn cancellation_stops_further_submissions() {
        let flag = AtomicBool::new(false);
        let stub = Stub::new(|_| ok(40));
        let config = ScheduleConfig::new(2, 4);
        let cancelled = || flag.load(Ordering::SeqCst);
        let trip = std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(10));
                flag.store(true, Ordering::SeqCst);
            });
            run_document(
                &source(8),
                &mock(4),
                &stub,
                &NoFinish,
                &config,
                &cancelled,
                &|_, _| {},
            )
        });
        assert!(matches!(trip, Err(ScheduleError::Cancelled)));
        assert!(stub.calls.load(Ordering::SeqCst) <= 4);
    }

    #[test]
    fn a_checkpointed_page_skips_layout_and_the_seat() {
        let dir =
            std::env::temp_dir().join(format!("semantik-schedule-resume-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut config = ScheduleConfig::new(2, 2);
        config.checkpoint_dir = Some(dir.clone());
        let first_stub = Stub::new(|_| ok(1));
        let first = run_document(
            &source(3),
            &mock(2),
            &first_stub,
            &NoFinish,
            &config,
            &never,
            &|_, _| {},
        )
        .unwrap();
        assert_eq!(first_stub.calls.load(Ordering::SeqCst), 6);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let dir_mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
            let file_mode = std::fs::metadata(checkpoint_path(&dir, 0))
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!((dir_mode, file_mode), (0o700, 0o600));
        }
        let layout = CountingLayout {
            inner: mock(2),
            calls: AtomicUsize::new(0),
        };
        let second_stub = Stub::new(|_| ok(1));
        let second = run_document(
            &source(3),
            &layout,
            &second_stub,
            &NoFinish,
            &config,
            &never,
            &|_, _| {},
        )
        .unwrap();
        assert_eq!(second_stub.calls.load(Ordering::SeqCst), 0);
        assert_eq!(layout.calls.load(Ordering::SeqCst), 0);
        let (mut second, mut first) = (second, first);
        second.diagnostics.clear();
        first.diagnostics.clear();
        assert_eq!(second, first);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_region_fails_the_document_and_is_not_checkpointed() {
        let dir =
            std::env::temp_dir().join(format!("semantik-schedule-fail-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut config = ScheduleConfig::new(1, 2);
        config.checkpoint_dir = Some(dir.clone());
        let stub = Stub::new(|call| {
            if call == 0 {
                (
                    Duration::ZERO,
                    Err(RegionError::Failed(RegionFailure {
                        class: FailureClass::PermanentClient,
                        status: Some(400),
                        detail: String::new(),
                        tail: TailOutcome::NotRetryable,
                    })),
                )
            } else {
                ok(0)
            }
        });
        let result = run_document(
            &source(2),
            &mock(2),
            &stub,
            &NoFinish,
            &config,
            &never,
            &|_, _| {},
        );
        assert!(matches!(
            result,
            Err(ScheduleError::Assemble(AssembleError::Recognition { .. }))
        ));
        assert!(read_checkpoint(&dir, 0).is_none());
        assert!(read_checkpoint(&dir, 1).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    struct Prepend;

    impl PageFinisher for Prepend {
        fn finish_built(
            &self,
            _: usize,
            _: &RenderedPage,
            _: &PageResult,
            _: (f64, f64),
            built: &mut BuiltPage,
            _: &(dyn Fn() -> bool + Sync),
        ) -> Result<(), FinishError> {
            assert!(built.regions.iter().all(|region| region.id.is_empty()));
            let mut added = built.regions[0].clone();
            added.text = "added by the hook".to_string();
            built.regions.insert(0, added);
            Ok(())
        }
    }

    #[test]
    fn the_region_hook_runs_before_ids_are_assigned() {
        let stub = Stub::new(|_| ok(0));
        let plain = run_document(
            &source(1),
            &mock(1),
            &stub,
            &NoFinish,
            &ScheduleConfig::new(1, 1),
            &never,
            &|_, _| {},
        )
        .unwrap();
        let hooked = run_document(
            &source(1),
            &mock(1),
            &stub,
            &Prepend,
            &ScheduleConfig::new(1, 1),
            &never,
            &|_, _| {},
        )
        .unwrap();
        assert_eq!(hooked.regions.len(), plain.regions.len() + 1);
        assert_eq!(hooked.regions[0].text, "added by the hook");
        let ids: Vec<&str> = hooked.regions.iter().map(|r| r.id.as_str()).collect();
        let expected: Vec<String> = (1..=hooked.regions.len())
            .map(|n| format!("p0001-r{n:06}"))
            .collect();
        assert_eq!(ids, expected);
    }

    #[test]
    fn a_batch_takes_ready_pages_in_index_order_without_waiting_to_fill() {
        let (sender, receiver) = sync_channel::<(usize, Option<usize>)>(8);
        let mut order = Reorder::new();
        let thread = std::thread::spawn(move || {
            sender.send((1, Some(11))).unwrap();
            sender.send((0, Some(10))).unwrap();
            std::thread::sleep(Duration::from_millis(300));
            sender.send((2, None)).unwrap();
            sender.send((3, Some(13))).unwrap();
        });
        let started = std::time::Instant::now();
        let first = order.take_batch(&receiver, 4, 4, &|_| {}).unwrap();
        assert_eq!(first.iter().map(|(i, _)| *i).collect::<Vec<_>>(), [0, 1]);
        assert!(started.elapsed() < Duration::from_millis(250));
        let second = order.take_batch(&receiver, 4, 4, &|_| {}).unwrap();
        assert_eq!(second, vec![(3, 13)]);
        assert!(order.take_batch(&receiver, 4, 4, &|_| {}).is_none());
        thread.join().unwrap();
    }

    #[test]
    fn region_requests_begin_in_page_order_across_render_threads() {
        let recognizer = Stub::new(|_| ok(0));
        let started = Mutex::new(Vec::new());
        let config = ScheduleConfig::new(1, 1);
        run_document_reporting(
            &source(8),
            &mock(1),
            &recognizer,
            &NoFinish,
            &config,
            &never,
            &|_, _| {},
            &|reading, _| {
                let mut started = started.lock().unwrap();
                for page in reading {
                    if !started.contains(page) {
                        started.push(*page);
                    }
                }
            },
        )
        .unwrap();
        let started = started.into_inner().unwrap();
        assert_eq!(started, (1..=8).collect::<Vec<i64>>());
    }

    /// A finisher whose built-page hook waits until released.
    struct BlockedBuilt {
        released: AtomicBool,
        entered: AtomicUsize,
    }

    impl PageFinisher for BlockedBuilt {
        fn finish_built(
            &self,
            _: usize,
            _: &RenderedPage,
            _: &PageResult,
            _: (f64, f64),
            _: &mut BuiltPage,
            _: &(dyn Fn() -> bool + Sync),
        ) -> Result<(), FinishError> {
            self.entered.fetch_add(1, Ordering::SeqCst);
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            while !self.released.load(Ordering::SeqCst) {
                assert!(std::time::Instant::now() < deadline, "never released");
                std::thread::sleep(Duration::from_millis(2));
            }
            Ok(())
        }
    }

    #[test]
    fn a_blocked_finisher_on_a_recognised_page_does_not_stop_submission() {
        let blocked = BlockedBuilt {
            released: AtomicBool::new(false),
            entered: AtomicUsize::new(0),
        };
        let stub = Stub::new(|_| ok(0));
        let config = ScheduleConfig::new(1, 1);
        let layout = mock(1);
        let calls_per_page = {
            let probe = Stub::new(|_| ok(0));
            let probe_config = ScheduleConfig::new(1, 1);
            run_document(
                &source(1),
                &layout,
                &probe,
                &NoFinish,
                &probe_config,
                &never,
                &|_, _| {},
            )
            .unwrap();
            probe.calls.load(Ordering::SeqCst)
        };
        assert!(calls_per_page >= 1);
        std::thread::scope(|scope| {
            let run = scope.spawn(|| {
                run_document(
                    &source(3),
                    &mock(1),
                    &stub,
                    &blocked,
                    &config,
                    &never,
                    &|_, _| {},
                )
            });
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            while stub.calls.load(Ordering::SeqCst) < calls_per_page * 3
                || blocked.entered.load(Ordering::SeqCst) == 0
            {
                assert!(std::time::Instant::now() < deadline, "submission stalled");
                std::thread::sleep(Duration::from_millis(2));
            }
            blocked.released.store(true, Ordering::SeqCst);
            assert!(run.join().unwrap().is_ok());
        });
    }

    /// A finisher whose completion of regionless pages waits until released.
    struct BlockedEmpty {
        released: AtomicBool,
    }

    impl PageFinisher for BlockedEmpty {
        fn finish_page(
            &self,
            _: usize,
            _: &RenderedPage,
            regions: &mut Vec<RegionResult>,
        ) -> Result<(), String> {
            if regions.is_empty() {
                let deadline = std::time::Instant::now() + Duration::from_secs(20);
                while !self.released.load(Ordering::SeqCst) {
                    assert!(std::time::Instant::now() < deadline, "never released");
                    std::thread::sleep(Duration::from_millis(2));
                }
            }
            Ok(())
        }
    }

    #[test]
    fn a_blocked_finisher_on_a_regionless_page_does_not_stop_submission() {
        let blocked = BlockedEmpty {
            released: AtomicBool::new(false),
        };
        let stub = Stub::new(|_| ok(0));
        let config = ScheduleConfig::new(2, 1);
        let blank = MockLayout::new(vec![RawOutputs::blank(300, NUM_CLASSES)]);
        struct Alternating<'a> {
            blank: &'a MockLayout,
            full: MockLayout,
            calls: AtomicUsize,
        }
        impl LayoutModel for Alternating<'_> {
            fn detect(&self, pages: &[image::RgbImage]) -> Result<Vec<RawOutputs>, LayoutError> {
                if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    self.blank.detect(pages)
                } else {
                    self.full.detect(pages)
                }
            }
        }
        let layout = Alternating {
            blank: &blank,
            full: mock(1),
            calls: AtomicUsize::new(0),
        };
        std::thread::scope(|scope| {
            let run = scope.spawn(|| {
                run_document(
                    &source(3),
                    &layout,
                    &stub,
                    &blocked,
                    &config,
                    &never,
                    &|_, _| {},
                )
            });
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            while stub.calls.load(Ordering::SeqCst) < 4 {
                assert!(std::time::Instant::now() < deadline, "submission stalled");
                std::thread::sleep(Duration::from_millis(2));
            }
            blocked.released.store(true, Ordering::SeqCst);
            assert!(run.join().unwrap().is_ok());
        });
    }

    #[test]
    fn the_run_reports_schedule_counters_and_load() {
        let stub = Stub::new(|_| ok(20));
        let peak = AtomicUsize::new(0);
        let pool = AtomicUsize::new(0);
        let config = ScheduleConfig::new(2, 2);
        let result = run_document_observed(
            &source(3),
            &mock(2),
            &stub,
            &NoFinish,
            &config,
            &never,
            &|_, _| {},
            &|_, _| {},
            &|requests, workers| {
                peak.fetch_max(requests, Ordering::SeqCst);
                pool.store(workers, Ordering::SeqCst);
            },
        )
        .unwrap();
        assert_eq!(pool.load(Ordering::SeqCst), 2);
        assert!(peak.load(Ordering::SeqCst) >= 1);
        let names: Vec<&str> = result.diagnostics.iter().map(|(name, _)| *name).collect();
        for name in ["schedule_layout_calls", "schedule_layout_pages"] {
            assert!(names.contains(&name), "{name} missing from {names:?}");
        }
        let pages = result
            .diagnostics
            .iter()
            .find(|(name, _)| *name == "schedule_layout_pages")
            .map(|(_, count)| *count);
        assert_eq!(pages, Some(3));
        assert!(result.diagnostics.iter().all(|(_, count)| *count >= 1));
    }
}
