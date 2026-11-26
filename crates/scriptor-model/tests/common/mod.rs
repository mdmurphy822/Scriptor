//! Shared helpers of the real-checkpoint vision tests: snapshot lookup, weight upload and
//! synthetic text-image patches.
#![cfg(feature = "cuda")]
#![allow(dead_code)]

pub mod fixtures;
pub mod mtp;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use cudarc::driver::{CudaContext, CudaStream};
use scriptor_cuda::Device;
use scriptor_kernels::reference::conv::PATCH_ELEMS;
use scriptor_model::gpu::GpuVisionParams;
use scriptor_model::reference::TensorSource;
use scriptor_model::{Merger, VisionTower};
use scriptor_spec::{GenerationConfig, HfConfig, ModelSpec};
use scriptor_weights::SafetensorsDir;

/// Pixels of one patch edge.
pub const PATCH: usize = 14;

/// Directory of the GLM-OCR checkpoint: `$SCRIPTOR_SNAPSHOT_DIR`, else the revision `2e85a62`
/// snapshot (or any complete snapshot) of the Hugging Face cache.
pub fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SCRIPTOR_SNAPSHOT_DIR") {
        return p.into();
    }
    let hf = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".cache/huggingface")
        });
    let snaps = hf.join("hub/models--zai-org--GLM-OCR/snapshots");
    let mut dirs: Vec<_> = std::fs::read_dir(&snaps)
        .expect("snapshot cache")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("config.json").exists() && p.join("generation_config.json").exists())
        .collect();
    dirs.sort_by_key(|p| {
        !p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("2e85a62"))
    });
    dirs.into_iter().next().expect("a snapshot")
}

/// The checkpoint's model spec and tensors.
pub fn open_checkpoint() -> (ModelSpec, SafetensorsDir) {
    let dir = snapshot();
    let read = |n: &str| std::fs::read_to_string(dir.join(n)).expect(n);
    let config = HfConfig::from_json(&read("config.json")).expect("config");
    let generation = GenerationConfig::from_json(&read("generation_config.json")).ok();
    let spec = ModelSpec::from_hf(&config, generation.as_ref()).expect("spec");
    (spec, SafetensorsDir::open(&dir).expect("safetensors"))
}

/// Uploads the real vision tower and adapter as BF16.
pub fn load_vision(
    stream: &Arc<CudaStream>,
    spec: &ModelSpec,
    files: &SafetensorsDir,
) -> GpuVisionParams {
    let source = TensorSource::new(files);
    let tower = VisionTower::load(&source, &spec.vision).expect("tower");
    let merger = Merger::load(&source, &spec.vision).expect("merger");
    GpuVisionParams::from_reference(stream, &tower, &merger).expect("device weights")
}

/// What an uploaded vision weight set depends on: the checkpoint files and the vision spec.
#[derive(PartialEq, Eq)]
struct VisionKey {
    dir: PathBuf,
    shards: Vec<(String, u64, Option<SystemTime>)>,
    vision: String,
}

impl VisionKey {
    fn new(spec: &ModelSpec, files: &SafetensorsDir) -> Self {
        let root = files.dir();
        let shards = files
            .shard_names()
            .into_iter()
            .map(|name| {
                let meta = std::fs::metadata(root.join(name)).ok();
                let len = meta.as_ref().map_or(0, |m| m.len());
                let modified = meta.and_then(|m| m.modified().ok());
                (name.to_string(), len, modified)
            })
            .collect();
        Self {
            dir: root.canonicalize().unwrap_or_else(|_| root.to_path_buf()),
            shards,
            vision: format!("{:?}", spec.vision),
        }
    }
}

static VISION_CACHE: Mutex<Option<(VisionKey, Arc<GpuVisionParams>)>> = Mutex::new(None);

/// A handle on the binary-wide vision weight set. Dropping it waits for the device, so no
/// stream still reads the weights when the last reference to them is released.
pub struct SharedVision {
    params: Arc<GpuVisionParams>,
    ctx: Arc<CudaContext>,
}

impl SharedVision {
    /// The shared weights, for [`scriptor_model::gpu::GpuVision::with_params`].
    pub fn params(&self) -> Arc<GpuVisionParams> {
        self.params.clone()
    }
}

impl std::ops::Deref for SharedVision {
    type Target = GpuVisionParams;

    fn deref(&self) -> &GpuVisionParams {
        &self.params
    }
}

impl Drop for SharedVision {
    fn drop(&mut self) {
        let _ = self.ctx.synchronize();
    }
}

/// The real vision weights of `files`, uploaded once per test binary on a dedicated stream and
/// synchronised before sharing. The cache holds one weight set: a different checkpoint
/// releases the cached one before uploading. Declare the handle before any executor built
/// from it so it drops last.
pub fn shared_vision(dev: &Device, spec: &ModelSpec, files: &SafetensorsDir) -> SharedVision {
    let key = VisionKey::new(spec, files);
    let mut cache = VISION_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((cached, params)) = cache.as_ref()
        && *cached == key
    {
        return SharedVision {
            params: params.clone(),
            ctx: dev.context().clone(),
        };
    }
    if cache.take().is_some() {
        let _ = dev.context().synchronize();
    }
    let upload = dev.new_stream().expect("upload stream");
    let params = Arc::new(load_vision(&upload, spec, files));
    upload.synchronize().expect("weights upload");
    *cache = Some((key, params.clone()));
    SharedVision {
        params,
        ctx: dev.context().clone(),
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
}

/// Greyscale page of dark glyph-like strokes on a light background, `h * w` patches of the
/// grid wide, as normalised pixel values.
fn page(rng: &mut Rng, rows: usize, cols: usize) -> Vec<f32> {
    let mut px = vec![1.0f32; rows * cols];
    let line_h = 18 + (rng.next() % 6) as usize;
    let mut y = 6usize;
    while y + line_h < rows {
        let mut x = 4usize;
        while x + 8 < cols {
            let glyph_w = 5 + (rng.next() % 7) as usize;
            let glyph_h = 8 + (rng.next() % (line_h as u32 - 8)) as usize;
            let stroke = 1 + (rng.next() % 2) as usize;
            for gy in 0..glyph_h {
                for gx in 0..glyph_w {
                    let on = gx < stroke
                        || gy < stroke
                        || (gx + gy) % (glyph_w / 2 + 2) == 0
                        || (rng.next() & 7) == 0;
                    if on && y + gy < rows && x + gx < cols {
                        px[(y + gy) * cols + x + gx] = -1.0;
                    }
                }
            }
            x += glyph_w + 2 + (rng.next() % 4) as usize;
            if rng.next().is_multiple_of(9) {
                x += 8;
            }
        }
        y += line_h + 4;
    }
    px
}

/// BF16 patch rows `[t * h * w, 1176]` in merge-window order for a synthetic text image of
/// `h x w` patches (frame repeated `t` times), seeded by `seed`.
pub fn synthetic_patches(t: usize, h: usize, w: usize, seed: u64) -> Vec<u16> {
    let (rows, cols) = (h * PATCH, w * PATCH);
    let mut rng = Rng(seed | 1);
    let img = page(&mut rng, rows, cols);
    let mut out = Vec::with_capacity(t * h * w * PATCH_ELEMS);
    for _ in 0..t {
        for wr in 0..h / 2 {
            for wc in 0..w / 2 {
                for dr in 0..2 {
                    for dc in 0..2 {
                        let (py0, px0) = ((wr * 2 + dr) * PATCH, (wc * 2 + dc) * PATCH);
                        for c in 0..3 {
                            for _frame in 0..2 {
                                for py in 0..PATCH {
                                    for px in 0..PATCH {
                                        let v = img[(py0 + py) * cols + px0 + px]
                                            * (1.0 - 0.05 * c as f32);
                                        out.push(half::bf16::from_f32(v).to_bits());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}
