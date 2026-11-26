//! The repository's fixture images run through the real vision tower: each image alone, then
//! all images as one batch.

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use scriptor_cuda::{Device, DeviceArena};
use scriptor_image::{SmartResizeConfig, preprocess};
use scriptor_kernels::cublaslt::{GemmEngine, GemmOptions};
use scriptor_kernels::heuristics::TuningTable;
use scriptor_model::gpu::{GpuVision, VisionWorkspace};

use super::{load_vision, open_checkpoint};

/// Fixture images run through the tower.
pub const IMAGES: [&str; 4] = [
    "w777_h333.jpg",
    "w448_h420.jpg",
    "w140_h112.jpg",
    "modes/baseline_444.jpg",
];

/// cuBLASLt workspace of every engine the fixtures build.
pub const WORKSPACE_BYTES: usize = 64 << 20;

/// One tower input: its patch rows and grids.
pub struct Batch {
    pub label: String,
    pub patches: CudaSlice<u16>,
    pub grids: Vec<(usize, usize, usize)>,
}

impl Batch {
    /// Patch rows of the whole batch.
    pub fn rows(&self) -> usize {
        self.grids.iter().map(|&(t, h, w)| t * h * w).sum()
    }
}

/// Raw bytes of fixture image `name`.
pub fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/image")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// One preprocessed fixture: its name, BF16 patch rows and `(t, h, w)` grid.
type Image = (String, Vec<u16>, (usize, usize, usize));

/// Each image of [`IMAGES`] alone, then all of them as one batch.
pub fn batches(stream: &Arc<CudaStream>) -> Vec<Batch> {
    let images: Vec<Image> = IMAGES
        .iter()
        .map(|name| {
            let pre = preprocess(&fixture(name), &SmartResizeConfig::default()).expect(name);
            let g = pre.grid_thw;
            let bits = pre.pixels.iter().map(|v| v.to_bits()).collect();
            (
                (*name).to_owned(),
                bits,
                (g.t as usize, g.h as usize, g.w as usize),
            )
        })
        .collect();
    let mut out: Vec<Batch> = images
        .iter()
        .map(|(name, bits, grid)| Batch {
            label: name.clone(),
            patches: stream.clone_htod(bits).expect("patches"),
            grids: vec![*grid],
        })
        .collect();
    let all: Vec<u16> = images
        .iter()
        .flat_map(|(_, b, _)| b.iter().copied())
        .collect();
    out.push(Batch {
        label: "all images".to_owned(),
        patches: stream.clone_htod(&all).expect("patches"),
        grids: images.iter().map(|(_, _, g)| *g).collect(),
    });
    out
}

/// Device, stream, real vision tower, a workspace sized for every batch, and the batches.
pub struct Rig {
    pub dev: Device,
    pub stream: Arc<CudaStream>,
    pub arena: DeviceArena,
    pub vision: GpuVision,
    pub ws: VisionWorkspace,
    pub batches: Vec<Batch>,
}

impl Rig {
    /// Opens the GPU and the checkpoint and uploads the vision tower.
    pub fn new() -> Self {
        let (spec, files) = open_checkpoint();
        let dev = Device::open().expect("gpu");
        let stream = dev.new_stream().expect("stream");
        let arena = dev.arena(512 << 20, &stream).expect("arena");
        let params = load_vision(&stream, &spec, &files);
        let vision = GpuVision::new(dev.context(), params).expect("vision");
        let batches = batches(&stream);
        let budget = batches.iter().map(Batch::rows).max().expect("batches");
        let ws = VisionWorkspace::new(&stream, vision.params(), budget).expect("workspace");
        Self {
            dev,
            stream,
            arena,
            vision,
            ws,
            batches,
        }
    }

    /// A bf16 engine over `tuning` with a [`WORKSPACE_BYTES`] workspace from the rig's arena.
    pub fn engine(&mut self, tuning: TuningTable) -> GemmEngine {
        GemmEngine::new(
            self.dev.context(),
            &self.stream,
            &mut self.arena,
            tuning,
            GemmOptions {
                fp8: false,
                workspace_bytes: WORKSPACE_BYTES,
            },
        )
        .expect("gemm")
    }

    /// Runs every batch through the tower on `gemm` and returns each batch's output
    /// embeddings as BF16 bits.
    pub fn run_all(&mut self, gemm: &mut GemmEngine) -> Vec<Vec<Vec<u16>>> {
        let out_dim = self.vision.params().merger.out_dim;
        let stream = &self.stream;
        self.batches
            .iter()
            .map(|b| {
                let mut outs: Vec<CudaSlice<u16>> = b
                    .grids
                    .iter()
                    .map(|&(t, h, w)| {
                        stream
                            .alloc_zeros::<u16>(t * h * w / 4 * out_dim)
                            .expect("out")
                    })
                    .collect();
                let mut refs: Vec<&mut CudaSlice<u16>> = outs.iter_mut().collect();
                self.vision
                    .run(stream, gemm, &mut self.ws, &b.patches, &b.grids, &mut refs)
                    .expect("run");
                outs.iter()
                    .map(|o| stream.clone_dtoh(o).expect("download"))
                    .collect()
            })
            .collect()
    }
}
