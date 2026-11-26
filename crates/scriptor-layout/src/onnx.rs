//! ONNX Runtime layout backend.

use crate::model::{LayoutError, LayoutModel, RawOutputs};
use crate::preprocess::{INPUT_SIZE, preprocess};
use image::RgbImage;
use ort::ep::cuda::ConvAlgorithmSearch;
use ort::ep::{ArenaExtendStrategy, CUDA, ExecutionProvider};
use ort::session::Session;
use ort::value::Tensor;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::mpsc::{Sender, channel};
use std::thread::JoinHandle;

/// The execution provider requested by a device string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeviceChoice {
    Auto,
    Cpu,
    Cuda(i32),
}

/// Parses `None`/`auto`, `cpu`, `cuda`, or `cuda:N`.
fn parse_device(device: Option<&str>) -> Result<DeviceChoice, LayoutError> {
    let Some(text) = device else {
        return Ok(DeviceChoice::Auto);
    };
    let text = text.trim().to_ascii_lowercase();
    match text.as_str() {
        "" | "auto" => Ok(DeviceChoice::Auto),
        "cpu" => Ok(DeviceChoice::Cpu),
        "cuda" => Ok(DeviceChoice::Cuda(0)),
        other => other
            .strip_prefix("cuda:")
            .and_then(|id| id.parse::<i32>().ok())
            .filter(|id| *id >= 0)
            .map(DeviceChoice::Cuda)
            .ok_or_else(|| LayoutError::Backend(format!("unsupported layout device `{text}`"))),
    }
}

/// Extracts the hex digest from the first line of a `sha256sum` file.
fn parse_sha256_file(text: &str) -> Option<String> {
    let digest = text.split_whitespace().next()?;
    (digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then(|| digest.to_ascii_lowercase())
}

/// Compares the digest recorded in `sha256sum` text with `actual_hex`.
fn check_digest(recorded: &str, actual_hex: &str) -> Result<(), LayoutError> {
    match parse_sha256_file(recorded) {
        Some(expected) if expected == actual_hex.to_ascii_lowercase() => Ok(()),
        Some(_) => Err(LayoutError::Backend(
            "layout weights hash mismatch".to_string(),
        )),
        None => Err(LayoutError::Backend(
            "layout weights hash file is malformed".to_string(),
        )),
    }
}

/// Verifies `weights` against the `<weights>.sha256` file beside it, if any.
fn verify_weights(weights: &Path) -> Result<(), LayoutError> {
    let mut sidecar = weights.as_os_str().to_owned();
    sidecar.push(".sha256");
    let recorded = match std::fs::read_to_string(PathBuf::from(sidecar)) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(LayoutError::Backend(format!(
                "layout weights hash file unreadable: {error}"
            )));
        }
    };
    let mut file = std::fs::File::open(weights)
        .map_err(|error| LayoutError::Backend(format!("layout weights unreadable: {error}")))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| LayoutError::Backend(format!("layout weights unreadable: {error}")))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual: String = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    check_digest(&recorded, &actual)
}

/// Splits batched flat outputs into one validated [`RawOutputs`] per page.
///
/// Shapes are `[pages, queries, classes]`, `[pages, queries, 4]`, and
/// `[pages, queries, queries]`.
fn split_batch(
    pages: usize,
    logits: (&[i64], &[f32]),
    boxes: (&[i64], &[f32]),
    order: (&[i64], &[f32]),
) -> Result<Vec<RawOutputs>, LayoutError> {
    let dims = |name: &str, shape: &[i64]| -> Result<[usize; 3], LayoutError> {
        match shape {
            [a, b, c] if *a >= 0 && *b >= 0 && *c >= 0 => {
                Ok([*a as usize, *b as usize, *c as usize])
            }
            _ => Err(LayoutError::Shape(format!(
                "{name} has shape {shape:?}, expected rank 3"
            ))),
        }
    };
    let [lp, queries, classes] = dims("logits", logits.0)?;
    let [bp, bq, bw] = dims("pred_boxes", boxes.0)?;
    let [op, oq, ow] = dims("order_logits", order.0)?;
    if lp != pages || bp != pages || op != pages {
        return Err(LayoutError::Shape(format!(
            "model returned {lp}/{bp}/{op} pages for {pages} inputs"
        )));
    }
    if bq != queries || oq != queries || bw != 4 || ow != queries {
        return Err(LayoutError::Shape(
            "pred_boxes and order_logits disagree with logits".to_string(),
        ));
    }
    let slice = |name: &str, data: &[f32], page: usize, each: usize| {
        data.get(page * each..(page + 1) * each)
            .map(<[f32]>::to_vec)
            .ok_or_else(|| LayoutError::Shape(format!("{name} is shorter than its shape")))
    };
    (0..pages)
        .map(|page| {
            let raw = RawOutputs {
                queries,
                classes,
                logits: slice("logits", logits.1, page, queries * classes)?,
                pred_boxes: slice("pred_boxes", boxes.1, page, queries * 4)?,
                order_logits: slice("order_logits", order.1, page, queries * queries)?,
            };
            raw.validate()?;
            Ok(raw)
        })
        .collect()
}

static RUNTIME: OnceLock<Result<PathBuf, String>> = OnceLock::new();

/// Loads the ONNX Runtime library once per process; later loads must name
/// the same library.
fn init_runtime(runtime: &Path) -> Result<(), LayoutError> {
    let loaded = RUNTIME.get_or_init(|| {
        ort::init_from(runtime)
            .map(|builder| {
                builder.commit();
                runtime.to_path_buf()
            })
            .map_err(|error| format!("ONNX Runtime library failed to load: {error}"))
    });
    match loaded {
        Ok(path) if path == runtime => Ok(()),
        Ok(_) => Err(LayoutError::Backend(
            "ONNX Runtime already loaded from a different library".to_string(),
        )),
        Err(message) => Err(LayoutError::Backend(message.clone())),
    }
}

fn backend(error: impl std::fmt::Display) -> LayoutError {
    LayoutError::Backend(error.to_string())
}

fn build_session(weights: &Path, cuda: Option<i32>) -> Result<Session, LayoutError> {
    let mut builder = Session::builder().map_err(backend)?;
    if let Some(id) = cuda {
        CUDA::default()
            .with_device_id(id)
            .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
            .with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic)
            .with_conv_max_workspace(false)
            .register(&mut builder)
            .map_err(|error| LayoutError::Backend(format!("CUDA unavailable: {error}")))?;
    }
    builder.commit_from_file(weights).map_err(backend)
}

/// The batch dimension the model graph fixes for its first input, if any.
fn fixed_batch(session: &Session) -> Option<usize> {
    session
        .inputs()
        .first()
        .and_then(|input| input.dtype().tensor_shape())
        .and_then(|shape| shape.first().copied())
        .filter(|dim| *dim > 0)
        .and_then(|dim| usize::try_from(dim).ok())
}

/// Appends zero pages to the flat input `batch` of `pages` pages (`each`
/// values per page) until it holds `target` pages; returns the page count
/// to run. A batch already at or beyond `target` is left unchanged.
fn pad_pages(batch: &mut Vec<f32>, pages: usize, target: usize, each: usize) -> usize {
    if pages >= target {
        return pages;
    }
    batch.resize(target * each, 0.0);
    target
}

/// Drops the outputs of padded pages, keeping the first `pages`.
fn trim_pages(outputs: &mut Vec<RawOutputs>, pages: usize) {
    outputs.truncate(pages);
}

/// One request in flight: the input and the channel its reply returns on.
type Request<In, Out> = (In, Sender<Out>);

/// A long-lived thread that serves requests one at a time by running a
/// handler on every input. All calls, from any number of caller threads,
/// execute on this single thread. Dropping the worker ends the thread
/// after its queued requests finish and joins it.
struct InferenceThread<In, Out> {
    requests: Option<Sender<Request<In, Out>>>,
    thread: Option<JoinHandle<()>>,
}

impl<In: Send + 'static, Out: Send + 'static> InferenceThread<In, Out> {
    /// Starts the thread, moving `handler` onto it.
    fn spawn<F>(mut handler: F) -> std::io::Result<Self>
    where
        F: FnMut(In) -> Out + Send + 'static,
    {
        let (requests, inbox) = channel::<Request<In, Out>>();
        let thread = std::thread::Builder::new()
            .name("scriptor-layout-inference".to_string())
            .spawn(move || {
                while let Ok((input, reply)) = inbox.recv() {
                    let _ = reply.send(handler(input));
                }
            })?;
        Ok(Self {
            requests: Some(requests),
            thread: Some(thread),
        })
    }

    /// Runs the handler on `input` on the worker thread and returns its
    /// result; `None` when the thread is gone or the handler panicked.
    fn call(&self, input: In) -> Option<Out> {
        let (reply, response) = channel();
        self.requests.as_ref()?.send((input, reply)).ok()?;
        response.recv().ok()
    }
}

impl<In, Out> Drop for InferenceThread<In, Out> {
    fn drop(&mut self) {
        self.requests.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// A padded input batch and the page count it holds.
struct Batch {
    data: Vec<f32>,
    pages: usize,
}

/// Runs the model on `batch` and splits the outputs per page.
fn run_session(session: &mut Session, batch: Batch) -> Result<Vec<RawOutputs>, LayoutError> {
    let side = INPUT_SIZE as i64;
    let run_pages = batch.pages;
    let input =
        Tensor::from_array(([run_pages as i64, 3, side, side], batch.data)).map_err(backend)?;
    let outputs = session
        .run(ort::inputs!["pixel_values" => input])
        .map_err(backend)?;
    let tensor = |name: &str| {
        outputs
            .get(name)
            .ok_or_else(|| LayoutError::Backend(format!("model output `{name}` missing")))?
            .try_extract_tensor::<f32>()
            .map_err(backend)
    };
    let (logits_shape, logits) = tensor("logits")?;
    let (boxes_shape, boxes) = tensor("pred_boxes")?;
    let (order_shape, order) = tensor("order_logits")?;
    split_batch(
        run_pages,
        (logits_shape, logits),
        (boxes_shape, boxes),
        (order_shape, order),
    )
}

/// A layout model run through ONNX Runtime.
pub struct OnnxLayout {
    inference: InferenceThread<Batch, Result<Vec<RawOutputs>, LayoutError>>,
    device: String,
    batch_size: usize,
}

impl OnnxLayout {
    /// Loads the runtime library and the weights file. `device` is `None` or
    /// `auto` (CUDA when it registers, else CPU), `cpu`, `cuda`, or
    /// `cuda:N` (CUDA required). `batch_size` is the page count every run is
    /// padded up to, so the model sees one input shape; a fixed batch
    /// dimension in the model graph takes precedence.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::Backend`] for an unloadable runtime, a weights
    /// hash mismatch, an unsupported device, a CUDA request that cannot be
    /// satisfied, or a model that fails to load.
    pub fn load(
        runtime: &Path,
        weights: &Path,
        device: Option<&str>,
        batch_size: usize,
    ) -> Result<Self, LayoutError> {
        let choice = parse_device(device)?;
        verify_weights(weights)?;
        init_runtime(runtime)?;
        let (session, label) = match choice {
            DeviceChoice::Cpu => (build_session(weights, None)?, "cpu".to_string()),
            DeviceChoice::Cuda(id) => (build_session(weights, Some(id))?, format!("cuda:{id}")),
            DeviceChoice::Auto => match build_session(weights, Some(0)) {
                Ok(session) => (session, "cuda:0".to_string()),
                Err(_) => (build_session(weights, None)?, "cpu".to_string()),
            },
        };
        let batch_size = fixed_batch(&session).unwrap_or(batch_size).max(1);
        let mut session = session;
        let inference =
            InferenceThread::spawn(move |batch: Batch| run_session(&mut session, batch)).map_err(
                |error| {
                    LayoutError::Backend(format!(
                        "layout inference thread failed to start: {error}"
                    ))
                },
            )?;
        Ok(Self {
            inference,
            device: label,
            batch_size,
        })
    }

    /// The execution provider in use: `cpu` or `cuda:N`.
    #[must_use]
    pub fn device(&self) -> &str {
        &self.device
    }
}

impl LayoutModel for OnnxLayout {
    fn detect(&self, pages: &[RgbImage]) -> Result<Vec<RawOutputs>, LayoutError> {
        if pages.is_empty() {
            return Ok(Vec::new());
        }
        let each = 3 * INPUT_SIZE * INPUT_SIZE;
        let mut batch = Vec::with_capacity(pages.len() * each);
        for page in pages {
            let data = preprocess(page).data;
            if data.len() != each {
                return Err(LayoutError::Shape(format!(
                    "preprocessed page has {} values, expected {each}",
                    data.len()
                )));
            }
            batch.extend_from_slice(&data);
        }
        let run_pages = if self.device == "cpu" {
            pages.len()
        } else {
            pad_pages(&mut batch, pages.len(), self.batch_size, each)
        };
        let mut split = self
            .inference
            .call(Batch {
                data: batch,
                pages: run_pages,
            })
            .ok_or_else(|| {
                LayoutError::Backend("layout inference thread unavailable".to_string())
            })??;
        trim_pages(&mut split, pages.len());
        Ok(split)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_thread_serves_all_callers_on_one_thread() {
        use std::collections::HashSet;
        use std::sync::{Arc, Mutex};
        let seen = Arc::new(Mutex::new(HashSet::new()));
        let record = Arc::clone(&seen);
        let worker = InferenceThread::spawn(move |value: u32| {
            record.lock().unwrap().insert(std::thread::current().id());
            value * 2
        })
        .unwrap();
        std::thread::scope(|scope| {
            for caller in 0..16u32 {
                let worker = &worker;
                scope.spawn(move || {
                    assert_eq!(worker.call(caller), Some(caller * 2));
                });
            }
        });
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert!(!seen.contains(&std::thread::current().id()));
    }

    #[test]
    fn dropping_inference_thread_ends_it() {
        use std::sync::{Arc, Weak};
        let marker = Arc::new(());
        let weak: Weak<()> = Arc::downgrade(&marker);
        let worker = InferenceThread::spawn(move |value: u32| {
            let _held = &marker;
            value
        })
        .unwrap();
        assert_eq!(worker.call(7), Some(7));
        drop(worker);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn panicked_handler_yields_none() {
        let worker = InferenceThread::spawn(|value: u32| {
            assert!(value != 0, "boom");
            value
        })
        .unwrap();
        assert_eq!(worker.call(0), None);
        assert_eq!(worker.call(1), None);
    }

    #[test]
    fn device_strings_parse() {
        assert_eq!(parse_device(None).unwrap(), DeviceChoice::Auto);
        assert_eq!(parse_device(Some("Auto")).unwrap(), DeviceChoice::Auto);
        assert_eq!(parse_device(Some("cpu")).unwrap(), DeviceChoice::Cpu);
        assert_eq!(parse_device(Some("cuda")).unwrap(), DeviceChoice::Cuda(0));
        assert_eq!(parse_device(Some("cuda:2")).unwrap(), DeviceChoice::Cuda(2));
        assert!(parse_device(Some("cuda:x")).is_err());
        assert!(parse_device(Some("tpu")).is_err());
    }

    #[test]
    fn sha256_file_parses_and_mismatch_fails() {
        let digest = "ab".repeat(32);
        let text = format!("{digest}  model.onnx\n");
        assert_eq!(parse_sha256_file(&text).as_deref(), Some(digest.as_str()));
        assert!(parse_sha256_file("short  model.onnx").is_none());
        assert!(check_digest(&text, &digest.to_uppercase()).is_ok());
        let err = check_digest(&text, &"cd".repeat(32))
            .unwrap_err()
            .to_string();
        assert!(err.contains("hash mismatch"), "{err}");
        assert!(check_digest("garbage", &digest).is_err());
    }

    #[test]
    fn batched_outputs_split_per_page() {
        let logits: Vec<f32> = (0..2 * 3 * 2).map(|v| v as f32).collect();
        let boxes: Vec<f32> = (0..2 * 3 * 4).map(|v| v as f32).collect();
        let order: Vec<f32> = (0..2 * 3 * 3).map(|v| v as f32).collect();
        let split = split_batch(
            2,
            (&[2, 3, 2], &logits),
            (&[2, 3, 4], &boxes),
            (&[2, 3, 3], &order),
        )
        .unwrap();
        assert_eq!(split.len(), 2);
        assert_eq!((split[1].queries, split[1].classes), (3, 2));
        assert_eq!(split[1].logits, logits[6..]);
        assert_eq!(split[0].pred_boxes, boxes[..12]);
        assert_eq!(split[1].order_logits, order[9..]);
    }

    #[test]
    fn short_batches_pad_with_zero_pages_and_trim() {
        let mut batch = vec![1.0f32; 2 * 3];
        assert_eq!(pad_pages(&mut batch, 2, 4, 3), 4);
        assert_eq!(batch.len(), 12);
        assert!(batch[..6].iter().all(|v| *v == 1.0));
        assert!(batch[6..].iter().all(|v| *v == 0.0));
        let mut full = vec![1.0f32; 12];
        assert_eq!(pad_pages(&mut full, 4, 4, 3), 4);
        assert_eq!(full.len(), 12);
        let mut over = vec![1.0f32; 15];
        assert_eq!(pad_pages(&mut over, 5, 4, 3), 5);
        assert_eq!(over.len(), 15);

        let logits = vec![0.0f32; 4 * 3 * 2];
        let boxes = vec![0.0f32; 4 * 3 * 4];
        let order = vec![0.0f32; 4 * 3 * 3];
        let mut split = split_batch(
            4,
            (&[4, 3, 2], &logits),
            (&[4, 3, 4], &boxes),
            (&[4, 3, 3], &order),
        )
        .unwrap();
        trim_pages(&mut split, 2);
        assert_eq!(split.len(), 2);
    }

    #[test]
    fn mismatched_batches_are_shape_errors() {
        let data = vec![0.0f32; 64];
        let wrong_pages = split_batch(
            2,
            (&[1, 3, 2], &data[..6]),
            (&[1, 3, 4], &data[..12]),
            (&[1, 3, 3], &data[..9]),
        );
        assert!(matches!(wrong_pages, Err(LayoutError::Shape(_))));
        let short = split_batch(
            1,
            (&[1, 3, 2], &data[..5]),
            (&[1, 3, 4], &data[..12]),
            (&[1, 3, 3], &data[..9]),
        );
        assert!(matches!(short, Err(LayoutError::Shape(_))));
    }
}
