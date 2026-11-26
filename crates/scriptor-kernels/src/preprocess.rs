//! Batched GPU image preprocessing: bicubic antialias resize and fused
//! normalise-and-patchify.
//!
//! The batch planners are plain host code and always compiled; the launchers
//! need the `cuda` feature.

use std::fmt;

use crate::launch::ceil_div;

/// Patch edge in pixels.
pub const PATCH: usize = 14;
/// Elements per flattened patch (`3 * 2 * 14 * 14`).
pub const PATCH_ELEMS: usize = 1176;
/// Threads per block of the patchify kernel: one per pixel of a patch.
pub const PATCHIFY_THREADS: u32 = (PATCH * PATCH) as u32;
/// Threads per block of the resize kernels.
pub const RESIZE_THREADS: u32 = 128;

/// A batch whose images cannot be planned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// The batch holds no images.
    Empty,
    /// More images than a launch grid can index.
    TooManyImages(usize),
    /// A patchify side is not a positive multiple of 28.
    BadPatchifySize {
        index: usize,
        height: usize,
        width: usize,
    },
    /// A resize dimension is zero.
    BadResizeSize { index: usize },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "empty image batch"),
            Self::TooManyImages(n) => write!(f, "{n} images exceed the launch grid"),
            Self::BadPatchifySize {
                index,
                height,
                width,
            } => write!(
                f,
                "image {index} is {height}x{width}; sides must be positive multiples of 28"
            ),
            Self::BadResizeSize { index } => write!(f, "image {index} has a zero dimension"),
        }
    }
}

impl std::error::Error for PlanError {}

const MAX_GRID_Y: usize = 65_535;

fn check_count(n: usize) -> Result<(), PlanError> {
    if n == 0 {
        Err(PlanError::Empty)
    } else if n > MAX_GRID_Y {
        Err(PlanError::TooManyImages(n))
    } else {
        Ok(())
    }
}

/// Descriptor words and extents of one batched normalise-and-patchify launch.
///
/// Source images are packed back to back as interleaved RGB u8; the output is
/// the concatenated `[total_patches, 1176]` bf16 matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchifyBatch {
    /// Four words per image: source byte offset, first output patch row,
    /// height, width.
    pub words: Vec<i64>,
    /// Packed source size in bytes.
    pub rgb_bytes: usize,
    /// Output patch rows over all images.
    pub total_patches: usize,
    /// Largest per-image patch count.
    pub max_patches: usize,
    /// First output patch row of each image.
    pub patch_offsets: Vec<usize>,
}

impl PatchifyBatch {
    /// Plans a batch from `(height, width)` pairs, each a multiple of 28.
    pub fn new(sizes: &[(usize, usize)]) -> Result<Self, PlanError> {
        check_count(sizes.len())?;
        let mut words = Vec::with_capacity(sizes.len() * 4);
        let mut patch_offsets = Vec::with_capacity(sizes.len());
        let (mut rgb_bytes, mut total, mut max) = (0usize, 0usize, 0usize);
        for (index, &(height, width)) in sizes.iter().enumerate() {
            if height == 0 || width == 0 || height % 28 != 0 || width % 28 != 0 {
                return Err(PlanError::BadPatchifySize {
                    index,
                    height,
                    width,
                });
            }
            let patches = (height / PATCH) * (width / PATCH);
            words.extend([rgb_bytes as i64, total as i64, height as i64, width as i64]);
            patch_offsets.push(total);
            rgb_bytes += height * width * 3;
            total += patches;
            max = max.max(patches);
        }
        Ok(Self {
            words,
            rgb_bytes,
            total_patches: total,
            max_patches: max,
            patch_offsets,
        })
    }

    /// Number of images.
    pub fn images(&self) -> usize {
        self.words.len() / 4
    }

    /// Output elements (bf16) of the whole batch.
    pub fn output_elems(&self) -> usize {
        self.total_patches * PATCH_ELEMS
    }
}

/// One resize request: `((in_h, in_w), (out_h, out_w))`.
pub type ResizeItem = ((usize, usize), (usize, usize));

/// Descriptor words and extents of one batched resize launch.
///
/// Sources, intermediates and destinations are each packed back to back as
/// interleaved RGB u8.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResizeBatch {
    /// Eight words per image: source, intermediate and destination byte
    /// offsets, input height and width, output height and width, reserved.
    pub words: Vec<i64>,
    /// Packed source size in bytes.
    pub src_bytes: usize,
    /// Packed intermediate size in bytes.
    pub mid_bytes: usize,
    /// Packed destination size in bytes.
    pub dst_bytes: usize,
    /// Largest per-image horizontal-pass pixel count.
    pub max_h_pixels: usize,
    /// Largest per-image vertical-pass pixel count.
    pub max_v_pixels: usize,
    /// Destination byte offset of each image.
    pub dst_offsets: Vec<usize>,
}

impl ResizeBatch {
    /// Plans a batch from `((in_h, in_w), (out_h, out_w))` pairs.
    pub fn new(items: &[ResizeItem]) -> Result<Self, PlanError> {
        check_count(items.len())?;
        let mut words = Vec::with_capacity(items.len() * 8);
        let mut dst_offsets = Vec::with_capacity(items.len());
        let (mut src, mut mid, mut dst) = (0usize, 0usize, 0usize);
        let (mut max_h, mut max_v) = (0usize, 0usize);
        for (index, &((in_h, in_w), (out_h, out_w))) in items.iter().enumerate() {
            if in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 {
                return Err(PlanError::BadResizeSize { index });
            }
            words.extend([
                src as i64,
                mid as i64,
                dst as i64,
                in_h as i64,
                in_w as i64,
                out_h as i64,
                out_w as i64,
                0,
            ]);
            dst_offsets.push(dst);
            src += in_h * in_w * 3;
            mid += in_h * out_w * 3;
            dst += out_h * out_w * 3;
            max_h = max_h.max(in_h * out_w);
            max_v = max_v.max(out_h * out_w);
        }
        Ok(Self {
            words,
            src_bytes: src,
            mid_bytes: mid,
            dst_bytes: dst,
            max_h_pixels: max_h,
            max_v_pixels: max_v,
            dst_offsets,
        })
    }

    /// Number of images.
    pub fn images(&self) -> usize {
        self.words.len() / 8
    }

    /// Blocks along x of the horizontal pass.
    pub fn h_blocks(&self) -> u32 {
        ceil_div(self.max_h_pixels, RESIZE_THREADS as usize) as u32
    }

    /// Blocks along x of the vertical pass.
    pub fn v_blocks(&self) -> u32 {
        ceil_div(self.max_v_pixels, RESIZE_THREADS as usize) as u32
    }
}

#[cfg(scriptor_cuda_enabled)]
pub use device::PreprocessKernels;

#[cfg(scriptor_cuda_enabled)]
mod device {
    use std::sync::Arc;

    use cudarc::driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    };
    use scriptor_cuda::{CudaError, KernelModule, Result};

    use super::{PATCHIFY_THREADS, PatchifyBatch, RESIZE_THREADS, ResizeBatch};
    use crate::reference::preprocess::fused_affine;

    fn load_module(ctx: &Arc<CudaContext>, stem: &str) -> Result<KernelModule> {
        let image = crate::fatbins::FATBINS
            .iter()
            .find(|(name, _)| *name == stem)
            .map(|(_, bytes)| *bytes)
            .ok_or_else(|| CudaError::Module(format!("kernel module {stem} is not embedded")))?;
        KernelModule::load(ctx, image)
    }

    /// Image preprocessing kernels (bf16 as raw `u16` bits).
    pub struct PreprocessKernels {
        patchify: CudaFunction,
        resize_h: CudaFunction,
        resize_v: CudaFunction,
    }

    impl PreprocessKernels {
        /// Loads the kernels into `ctx`.
        pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
            let patchify = load_module(ctx, "preprocess")?;
            let resize = load_module(ctx, "resize_bicubic")?;
            Ok(Self {
                patchify: patchify.kernel("scriptor_normalise_patchify_bf16")?,
                resize_h: resize.kernel("scriptor_resize_bicubic_h_u8")?,
                resize_v: resize.kernel("scriptor_resize_bicubic_v_u8")?,
            })
        }

        /// Normalises every packed image of `batch` and writes the
        /// `[total_patches, 1176]` bf16 matrix to `out`.
        ///
        /// `rgb` holds `batch.rgb_bytes` bytes, `descs` the batch words
        /// (`batch.words`), `out` at least `batch.output_elems()` elements.
        pub fn normalise_patchify(
            &self,
            stream: &Arc<CudaStream>,
            batch: &PatchifyBatch,
            rgb: &CudaSlice<u8>,
            descs: &CudaSlice<i64>,
            out: &mut CudaSlice<u16>,
        ) -> Result<()> {
            let affine = fused_affine();
            let (s0, s1, s2) = (affine[0].0, affine[1].0, affine[2].0);
            let (o0, o1, o2) = (affine[0].1, affine[1].1, affine[2].1);
            let cfg = LaunchConfig {
                grid_dim: (batch.max_patches as u32, batch.images() as u32, 1),
                block_dim: (PATCHIFY_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: argument order and types match the kernel signature; the
            // buffers are sized by `batch` as documented above.
            unsafe {
                stream
                    .launch_builder(&self.patchify)
                    .arg(rgb)
                    .arg(descs)
                    .arg(out)
                    .arg(&s0)
                    .arg(&s1)
                    .arg(&s2)
                    .arg(&o0)
                    .arg(&o1)
                    .arg(&o2)
                    .launch(cfg)?;
            }
            Ok(())
        }

        /// Resizes every packed image of `batch` with the separable bicubic
        /// antialias filter: horizontal pass into `mid`, vertical pass into
        /// `dst`.
        ///
        /// `src`, `mid` and `dst` hold `batch.src_bytes`, `batch.mid_bytes` and
        /// `batch.dst_bytes` bytes; `descs` the batch words.
        pub fn resize_bicubic(
            &self,
            stream: &Arc<CudaStream>,
            batch: &ResizeBatch,
            src: &CudaSlice<u8>,
            descs: &CudaSlice<i64>,
            mid: &mut CudaSlice<u8>,
            dst: &mut CudaSlice<u8>,
        ) -> Result<()> {
            let config = |blocks: u32| LaunchConfig {
                grid_dim: (blocks, batch.images() as u32, 1),
                block_dim: (RESIZE_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: argument order and types match the kernel signatures; the
            // buffers are sized by `batch` as documented above.
            unsafe {
                stream
                    .launch_builder(&self.resize_h)
                    .arg(src)
                    .arg(descs)
                    .arg(&mut *mid)
                    .launch(config(batch.h_blocks()))?;
                stream
                    .launch_builder(&self.resize_v)
                    .arg(&*mid)
                    .arg(descs)
                    .arg(dst)
                    .launch(config(batch.v_blocks()))?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patchify_plan_offsets() {
        let b = PatchifyBatch::new(&[(28, 28), (56, 84)]).unwrap();
        assert_eq!(b.images(), 2);
        assert_eq!(b.patch_offsets, vec![0, 4]);
        assert_eq!(b.total_patches, 4 + 24);
        assert_eq!(b.max_patches, 24);
        assert_eq!(b.rgb_bytes, 28 * 28 * 3 + 56 * 84 * 3);
        assert_eq!(b.words[4..], [(28 * 28 * 3) as i64, 4, 56, 84]);
        assert_eq!(b.output_elems(), 28 * 1176);
    }

    #[test]
    fn patchify_rejects_bad_sizes() {
        assert_eq!(PatchifyBatch::new(&[]), Err(PlanError::Empty));
        assert!(matches!(
            PatchifyBatch::new(&[(28, 30)]),
            Err(PlanError::BadPatchifySize { index: 0, .. })
        ));
        assert!(matches!(
            PatchifyBatch::new(&[(28, 28), (0, 28)]),
            Err(PlanError::BadPatchifySize { index: 1, .. })
        ));
    }

    #[test]
    fn resize_plan_extents() {
        let b = ResizeBatch::new(&[((10, 20), (5, 8)), ((4, 4), (6, 6))]).unwrap();
        assert_eq!(b.images(), 2);
        assert_eq!(b.src_bytes, (200 + 16) * 3);
        assert_eq!(b.mid_bytes, (10 * 8 + 4 * 6) * 3);
        assert_eq!(b.dst_bytes, (40 + 36) * 3);
        assert_eq!(b.dst_offsets, vec![0, 120]);
        assert_eq!(b.max_h_pixels, 80);
        assert_eq!(b.max_v_pixels, 40);
        assert_eq!(b.h_blocks(), 1);
        assert_eq!(&b.words[8..], [600, 240, 120, 4, 4, 6, 6, 0]);
    }

    #[test]
    fn resize_rejects_zero() {
        assert_eq!(
            ResizeBatch::new(&[((1, 1), (1, 0))]),
            Err(PlanError::BadResizeSize { index: 0 })
        );
    }
}
