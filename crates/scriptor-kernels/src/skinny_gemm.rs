//! Skinny-M bf16 GEMM `out[m, n] = x[m, k] . w[n, k]^T` with fp32 accumulation for the
//! decoder's six weight shapes and `m <= 128` (`kernels/skinny_gemm.cu`).
//!
//! The kernel streams each weight byte once through persistent CTAs. Shapes whose weight-row
//! tiles alone cannot fill the device split the K range; the fp32 partial tiles and the
//! per-tile completion counters live in buffers allocated once by [`SkinnyGemm::new`], so a
//! launch performs no allocation, host synchronisation or host-side state change.

use std::sync::Arc;

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use scriptor_cuda::{CudaError, KernelModule, Result};

/// Largest token count the kernel handles.
pub const MAX_M: usize = 128;

/// Weight rows each warp owns.
const WARP_ROWS: usize = 16;
/// Warps per CTA; a CTA covers `WARP_ROWS * WARPS` weight rows.
const WARPS: usize = 4;
/// Resident CTAs per SM the launch geometry targets.
const CTAS_PER_SM: usize = 1;
/// Dynamic shared-memory bytes one CTA's ring may use.
const SMEM_BUDGET: usize = 101_248;
/// Reduction elements per pipeline step.
const BLOCK_K: usize = 128;
/// Largest ring depth.
const MAX_STAGES: usize = 8;
/// Token-group counts the kernel is instantiated for.
const TOKEN_GROUPS: [usize; 5] = [1, 2, 4, 8, 16];
/// Token-group counts the fused-argmax kernels are instantiated for.
const ARGMAX_GROUPS: [usize; 4] = [1, 2, 4, 8];
/// Largest token count the narrow fused-argmax kernels handle.
pub const ARGMAX_MAX_M: usize = 64;
/// Largest token count the wide fused-argmax kernel handles.
pub const WIDE_ARGMAX_MAX_M: usize = MAX_M;
/// Token groups of the wide fused-argmax instantiation.
const WIDE_ARGMAX_GROUPS: usize = 16;
/// Counter slots (one per weight-row tile of a split-K shape).
const COUNTER_SLOTS: usize = 1024;

/// Launch plan of one weight shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapePlan {
    /// Weight rows (output columns).
    pub n: usize,
    /// Reduction length.
    pub k: usize,
    /// K-range splits whose partial tiles are summed by the completing CTA.
    pub slices: usize,
}

impl ShapePlan {
    /// Weight-row tiles.
    pub fn tiles(&self) -> usize {
        self.n / (WARP_ROWS * WARPS)
    }

    /// Work items: one per (tile, K slice).
    pub fn items(&self) -> usize {
        self.tiles() * self.slices
    }

    fn partial_bytes(&self) -> usize {
        // One fp32 accumulator tile of the widest token group per (tile, slice, warp).
        self.items() * WARPS * WARP_ROWS * MAX_M * 4
    }
}

/// The decoder weight shapes the kernel serves, each sized to fill the device.
pub const SHAPES: [ShapePlan; 6] = [
    ShapePlan {
        n: 4096,
        k: 1536,
        slices: 3,
    },
    ShapePlan {
        n: 1536,
        k: 2048,
        slices: 4,
    },
    ShapePlan {
        n: 9216,
        k: 1536,
        slices: 1,
    },
    ShapePlan {
        n: 1536,
        k: 4608,
        slices: 4,
    },
    ShapePlan {
        n: 59392,
        k: 1536,
        slices: 1,
    },
    ShapePlan {
        n: 1536,
        k: 3072,
        slices: 4,
    },
];

/// The plan for weight shape `[n, k]`, if it is one of [`SHAPES`].
pub fn plan_for(n: usize, k: usize) -> Option<ShapePlan> {
    SHAPES.iter().copied().find(|s| s.n == n && s.k == k)
}

/// Whether `[m, k] x [n, k]^T` is served by the kernel.
pub fn supports(m: usize, n: usize, k: usize) -> bool {
    (1..=MAX_M).contains(&m) && plan_for(n, k).is_some()
}

/// Whether the narrow fused-argmax kernels serve `[m, k] x [n, k]^T`: a served shape whose K
/// range is not split and at most [`ARGMAX_MAX_M`] tokens.
pub fn supports_argmax(m: usize, n: usize, k: usize) -> bool {
    (1..=ARGMAX_MAX_M).contains(&m) && plan_for(n, k).is_some_and(|p| p.slices == 1)
}

/// Which fused-argmax kernel serves a head product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgmaxRoute {
    /// The kernels that keep a per-item cross-warp reduction scratch beside the ring
    /// (`m <= ARGMAX_MAX_M`).
    Narrow,
    /// The kernel that merges after the K loop in the drained ring
    /// (`ARGMAX_MAX_M < m <= WIDE_ARGMAX_MAX_M`).
    Wide,
}

/// The fused-argmax kernel serving `[m, k] x [n, k]^T`, if any: the narrow kernels for
/// `1..=ARGMAX_MAX_M` tokens, and the wide kernel for the tokens above that only when `wide`
/// is set; the shape must be an unsplit served shape.
pub fn argmax_route(m: usize, n: usize, k: usize, wide: bool) -> Option<ArgmaxRoute> {
    if !plan_for(n, k).is_some_and(|p| p.slices == 1) {
        return None;
    }
    if (1..=ARGMAX_MAX_M).contains(&m) {
        Some(ArgmaxRoute::Narrow)
    } else if wide && (ARGMAX_MAX_M + 1..=WIDE_ARGMAX_MAX_M).contains(&m) {
        Some(ArgmaxRoute::Wide)
    } else {
        None
    }
}

/// Token groups of eight the instantiation for `m` tokens carries.
fn token_groups(m: usize) -> usize {
    TOKEN_GROUPS
        .into_iter()
        .find(|g| g * 8 >= m)
        .unwrap_or(MAX_M / 8)
}

fn stage_bytes(groups: usize) -> usize {
    (WARP_ROWS * WARPS + 8 * groups) * BLOCK_K * 2
}

/// Ring stages of the `groups` instantiation: as many as the shared-memory budget holds.
fn stages_at(groups: usize) -> usize {
    (SMEM_BUDGET / stage_bytes(groups)).min(MAX_STAGES)
}

fn dynamic_smem(groups: usize) -> u32 {
    (stages_at(groups) * stage_bytes(groups)) as u32
}

/// Dynamic shared memory of a fused-argmax launch: the ring plus a value and an index slot per
/// warp and token.
fn argmax_smem(groups: usize) -> u32 {
    dynamic_smem(groups) + (WARPS * groups * 8 * 8) as u32
}

/// Resources of the wide fused-argmax kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WideArgmaxReceipt {
    /// Registers per thread.
    pub registers: i32,
    /// Local-memory bytes per thread; non-zero means register spills or a stack frame.
    pub local_bytes: i32,
    /// Statically allocated shared bytes per CTA.
    pub static_shared_bytes: i32,
    /// Dynamic shared bytes per launch.
    pub dynamic_shared_bytes: u32,
}

/// Dynamic shared memory of a wide fused-argmax launch: the ring alone, since the warp-merge
/// scratch reuses the ring once the K loop has drained it.
fn wide_argmax_smem() -> u32 {
    dynamic_smem(WIDE_ARGMAX_GROUPS)
}

/// Loaded kernels plus the split-K scratch of one GEMM driver.
pub struct SkinnyGemm {
    stream: Arc<CudaStream>,
    kernels: Vec<(usize, CudaFunction)>,
    argmax_kernels: Vec<(usize, CudaFunction)>,
    wide_argmax: CudaFunction,
    partials: CudaSlice<u8>,
    counters: CudaSlice<i32>,
    sms: usize,
    _module: KernelModule,
    _wide_module: KernelModule,
}

impl SkinnyGemm {
    /// Loads the kernels and allocates the zeroed split-K scratch.
    pub fn new(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> Result<Self> {
        let image = crate::fatbins::FATBINS
            .iter()
            .find(|(n, _)| *n == "skinny_gemm")
            .map(|(_, b)| *b)
            .ok_or_else(|| CudaError::Module("missing fatbin skinny_gemm".into()))?;
        let module = KernelModule::load(ctx, image)?;
        let mut kernels = Vec::new();
        for groups in TOKEN_GROUPS {
            let f = module.kernel(&format!("scriptor_skinny_gemm_w{WARPS}_nb{groups}"))?;
            f.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                dynamic_smem(groups) as i32,
            )?;
            kernels.push((groups, f));
        }
        let mut argmax_kernels = Vec::new();
        for groups in ARGMAX_GROUPS {
            let f = module.kernel(&format!("scriptor_skinny_argmax_w{WARPS}_nb{groups}"))?;
            f.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                argmax_smem(groups) as i32,
            )?;
            argmax_kernels.push((groups, f));
        }
        let wide_image = crate::fatbins::FATBINS
            .iter()
            .find(|(n, _)| *n == "skinny_argmax_wide")
            .map(|(_, b)| *b)
            .ok_or_else(|| CudaError::Module("missing fatbin skinny_argmax_wide".into()))?;
        let wide_module = KernelModule::load(ctx, wide_image)?;
        let wide_argmax = wide_module.kernel(&format!(
            "scriptor_skinny_argmax_wide_w{WARPS}_nb{WIDE_ARGMAX_GROUPS}"
        ))?;
        wide_argmax.set_attribute(
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            wide_argmax_smem() as i32,
        )?;
        let partial_bytes = SHAPES
            .iter()
            .map(ShapePlan::partial_bytes)
            .max()
            .unwrap_or(0);
        let sms = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .map_or(1, |n| n.max(1) as usize);
        Ok(Self {
            stream: stream.clone(),
            kernels,
            argmax_kernels,
            wide_argmax,
            partials: stream.alloc_zeros::<u8>(partial_bytes.max(1))?,
            counters: stream.alloc_zeros::<i32>(COUNTER_SLOTS)?,
            sms,
            _module: module,
            _wide_module: wide_module,
        })
    }

    /// Resource receipt of the wide fused-argmax kernel as the driver loaded it: registers per
    /// thread, local (spill and stack) bytes per thread, static shared bytes and the dynamic
    /// shared bytes a launch requests.
    pub fn wide_argmax_receipt(&self) -> Result<WideArgmaxReceipt> {
        let attr = |a: CUfunction_attribute_enum| -> Result<i32> {
            self.wide_argmax.get_attribute(a).map_err(CudaError::from)
        };
        Ok(WideArgmaxReceipt {
            registers: attr(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS)?,
            local_bytes: attr(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?,
            static_shared_bytes: attr(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
            )?,
            dynamic_shared_bytes: wide_argmax_smem(),
        })
    }

    /// Directs later launches to `stream`. The launch geometry is unchanged.
    pub fn set_stream(&mut self, stream: &Arc<CudaStream>) {
        self.stream = stream.clone();
    }

    /// Launches `out[m, n] = bf16(x[m, k] . w[n, k]^T)` on the driver's stream; the shape must
    /// satisfy [`supports`]. `x`, `w` and `out` are device addresses of dense row-major bf16
    /// matrices.
    pub fn launch(&self, m: usize, n: usize, k: usize, x: u64, w: u64, out: u64) -> Result<()> {
        let plan = plan_for(n, k)
            .filter(|_| (1..=MAX_M).contains(&m))
            .ok_or_else(|| {
                CudaError::Driver(format!("skinny gemm: unsupported m={m} n={n} k={k}"))
            })?;
        let groups = token_groups(m);
        let f = self
            .kernels
            .iter()
            .find(|(g, _)| *g == groups)
            .map(|(_, f)| f)
            .ok_or_else(|| CudaError::Driver("skinny gemm: kernel not loaded".into()))?;
        let grid = plan.items().min(self.sms * CTAS_PER_SM).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid as u32, 1, 1),
            block_dim: ((32 * WARPS) as u32, 1, 1),
            shared_mem_bytes: dynamic_smem(groups),
        };
        let part = self.partials.device_ptr(&self.stream).0;
        let counters = self.counters.device_ptr(&self.stream).0;
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let (slices, items) = (plan.slices as i32, plan.items() as i32);
        let mut b = self.stream.launch_builder(f);
        b.arg(&x)
            .arg(&w)
            .arg(&out)
            .arg(&part)
            .arg(&counters)
            .arg(&mi)
            .arg(&ni)
            .arg(&ki)
            .arg(&slices)
            .arg(&items);
        // SAFETY: the operands cover the shape by the caller's contract; the scratch buffers
        // were sized for every plan at construction and launches on one stream are ordered.
        unsafe { b.launch(cfg) }.map_err(CudaError::from)?;
        Ok(())
    }

    /// Launches the product `[m, k] x [n, k]^T` reduced to one token per row: `tokens[r]` is the
    /// lowest column index holding the maximum of row `r` of the bf16-rounded product (the
    /// product itself is not stored). The shape must satisfy [`argmax_route`] with `wide` set;
    /// rows above [`ARGMAX_MAX_M`] run the wide kernel. `tokens` addresses `m` u32 entries.
    pub fn launch_argmax(
        &self,
        m: usize,
        n: usize,
        k: usize,
        x: u64,
        w: u64,
        tokens: u64,
    ) -> Result<()> {
        let route = argmax_route(m, n, k, true).ok_or_else(|| {
            CudaError::Driver(format!("skinny argmax: unsupported m={m} n={n} k={k}"))
        })?;
        let plan = plan_for(n, k)
            .ok_or_else(|| CudaError::Driver("skinny argmax: unknown shape".into()))?;
        let (f, smem) = match route {
            ArgmaxRoute::Narrow => {
                let groups = token_groups(m);
                let f = self
                    .argmax_kernels
                    .iter()
                    .find(|(g, _)| *g == groups)
                    .map(|(_, f)| f)
                    .ok_or_else(|| CudaError::Driver("skinny argmax: kernel not loaded".into()))?;
                (f, argmax_smem(groups))
            }
            ArgmaxRoute::Wide => (&self.wide_argmax, wide_argmax_smem()),
        };
        let grid = plan.items().min(self.sms * CTAS_PER_SM).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid as u32, 1, 1),
            block_dim: ((32 * WARPS) as u32, 1, 1),
            shared_mem_bytes: smem,
        };
        let part = self.partials.device_ptr(&self.stream).0;
        let counters = self.counters.device_ptr(&self.stream).0;
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let (slices, items) = (plan.slices as i32, plan.items() as i32);
        let mut b = self.stream.launch_builder(f);
        b.arg(&x)
            .arg(&w)
            .arg(&tokens)
            .arg(&part)
            .arg(&counters)
            .arg(&mi)
            .arg(&ni)
            .arg(&ki)
            .arg(&slices)
            .arg(&items);
        // SAFETY: the operands cover the shape by the caller's contract; the scratch buffers
        // were sized for every plan at construction, the per-CTA maxima fit the partial buffer
        // and launches on one stream are ordered.
        unsafe { b.launch(cfg) }.map_err(CudaError::from)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_plan_is_exact() {
        for s in SHAPES {
            assert_eq!(s.n % (WARP_ROWS * WARPS), 0, "{s:?}");
            assert_eq!(s.k % BLOCK_K, 0, "{s:?}");
            assert_eq!((s.k / BLOCK_K) % s.slices, 0, "{s:?}");
            if s.slices > 1 {
                assert!(s.tiles() <= COUNTER_SLOTS, "{s:?}");
            }
        }
    }

    #[test]
    fn support_covers_only_listed_shapes() {
        assert!(supports(1, 4096, 1536));
        assert!(supports(128, 59392, 1536));
        assert!(!supports(0, 4096, 1536));
        assert!(!supports(129, 4096, 1536));
        assert!(!supports(8, 4096, 1024));
    }

    #[test]
    fn argmax_support_needs_unsplit_shape_and_narrow_batch() {
        assert!(supports_argmax(1, 59392, 1536));
        assert!(supports_argmax(64, 59392, 1536));
        assert!(!supports_argmax(65, 59392, 1536));
        assert!(!supports_argmax(8, 4096, 1536));
    }

    #[test]
    fn argmax_ring_fits_shared_memory() {
        for g in ARGMAX_GROUPS {
            assert!(argmax_smem(g) as usize <= 101_376);
        }
    }

    #[test]
    fn argmax_routes_by_row_count_and_selector() {
        let (n, k) = (59392, 1536);
        for wide in [false, true] {
            assert_eq!(argmax_route(1, n, k, wide), Some(ArgmaxRoute::Narrow));
            assert_eq!(argmax_route(64, n, k, wide), Some(ArgmaxRoute::Narrow));
            assert_eq!(argmax_route(0, n, k, wide), None);
            assert_eq!(argmax_route(129, n, k, wide), None);
            assert_eq!(argmax_route(96, 4096, 1536, wide), None);
        }
        for m in [65, 96, 128] {
            assert_eq!(argmax_route(m, n, k, false), None, "m={m}");
            assert_eq!(
                argmax_route(m, n, k, true),
                Some(ArgmaxRoute::Wide),
                "m={m}"
            );
        }
    }

    #[test]
    fn wide_argmax_ring_alone_fits_and_scratch_reuses_it() {
        let ring = wide_argmax_smem() as usize;
        assert!(ring <= SMEM_BUDGET);
        assert_eq!(ring, 2 * stage_bytes(WIDE_ARGMAX_GROUPS));
        // Value and index per warp and token fit in the drained ring.
        assert!(WARPS * WIDE_ARGMAX_MAX_M * 8 <= ring);
        // Keeping the narrow layout (scratch beside the ring) would not fit.
        assert!(argmax_smem(WIDE_ARGMAX_GROUPS) as usize > SMEM_BUDGET);
    }

    #[test]
    fn token_groups_round_up() {
        assert_eq!(token_groups(1), 1);
        assert_eq!(token_groups(9), 2);
        assert_eq!(token_groups(33), 8);
        assert_eq!(token_groups(128), 16);
    }

    #[test]
    fn ring_depth_stays_in_range() {
        for g in TOKEN_GROUPS {
            assert!((2..=MAX_STAGES).contains(&stages_at(g)));
            assert!(dynamic_smem(g) as usize <= SMEM_BUDGET);
        }
    }

    /// Host model of the wide kernel's per-token reduction over the weight rows `0..values.len()`
    /// (bf16-rounded before comparison): lane-private running maxima over the tiles each CTA
    /// owns, a shuffle tree over the eight row groups, then the warp, CTA and grid merges.
    mod wide_model {
        use super::*;
        use half::bf16;

        pub const NO_INDEX: u32 = u32::MAX;

        pub fn take(best: &mut (f32, u32), v: f32, i: u32) {
            if v > best.0 || (v == best.0 && i < best.1) {
                *best = (v, i);
            }
        }

        pub fn round(v: f32) -> f32 {
            bf16::from_f32(v).to_f32()
        }

        /// Sequential definition: the lowest index of the maximum of the rounded values, NaN
        /// never winning.
        pub fn reference(values: &[f32]) -> u32 {
            let mut best = (f32::NEG_INFINITY, NO_INDEX);
            for (i, &v) in values.iter().enumerate() {
                take(&mut best, round(v), i as u32);
            }
            best.1
        }

        pub fn wide_fold(values: &[f32], ctas: usize) -> u32 {
            let rows = WARP_ROWS * WARPS;
            let tiles = values.len() / rows;
            let mut grid = (f32::NEG_INFINITY, NO_INDEX);
            for cta in 0..ctas {
                // run[warp][g]
                let mut run = [[(f32::NEG_INFINITY, NO_INDEX); 8]; WARPS];
                let mut tile = cta;
                while tile < tiles {
                    for (warp, per_warp) in run.iter_mut().enumerate() {
                        for (g, lane) in per_warp.iter_mut().enumerate() {
                            for h in 0..2 {
                                let row = tile * rows + warp * WARP_ROWS + g + 8 * h;
                                take(lane, round(values[row]), row as u32);
                            }
                        }
                    }
                    tile += ctas;
                }
                let mut cta_best = (f32::NEG_INFINITY, NO_INDEX);
                for per_warp in &run {
                    // xor 4, 8, 16 over lanes sharing a token: a tree over the eight groups.
                    let mut lanes = *per_warp;
                    let mut width = 8;
                    while width > 1 {
                        width /= 2;
                        for g in 0..width {
                            let other = lanes[g + width];
                            take(&mut lanes[g], other.0, other.1);
                        }
                    }
                    take(&mut cta_best, lanes[0].0, lanes[0].1);
                }
                take(&mut grid, cta_best.0, cta_best.1);
            }
            grid.1
        }

        pub struct Lcg(pub u64);

        impl Lcg {
            pub fn next(&mut self) -> u64 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                self.0 >> 33
            }
        }
    }

    use wide_model::*;

    const ROWS: usize = 59392;

    fn check(values: &[f32]) {
        let want = reference(values);
        for ctas in [1, 5, 48, 928] {
            assert_eq!(wide_fold(values, ctas), want, "ctas={ctas}");
        }
    }

    #[test]
    fn wide_fold_equals_sequential_argmax_on_random_values() {
        let mut rng = Lcg(7);
        let values: Vec<f32> = (0..ROWS)
            .map(|_| (rng.next() % 20001) as f32 / 1000.0 - 10.0)
            .collect();
        check(&values);
    }

    #[test]
    fn wide_fold_takes_lowest_index_of_exact_ties() {
        let mut values = vec![0.25f32; ROWS];
        for &i in &[ROWS - 1, 12345, 64, 63, 40000] {
            values[i] = 3.0;
        }
        check(&values);
        assert_eq!(reference(&values), 63);
        // A tie spread across tiles and CTAs with the lowest copy in a late tile.
        let mut values = vec![-1.0f32; ROWS];
        for &i in &[59391, 30000, 17] {
            values[i] = 2.0;
        }
        check(&values);
        assert_eq!(reference(&values), 17);
    }

    #[test]
    fn wide_fold_ties_after_bf16_rounding() {
        // Distinct fp32 values that round to the same bf16 are ties; the lowest index wins even
        // though a later one is larger before rounding.
        let mut values = vec![0.0f32; ROWS];
        values[900] = 1.0;
        values[100] = 1.0 + 1.0 / 1024.0;
        values[50] = 1.0 - 1.0 / 2048.0;
        assert_eq!(round(values[900]), round(values[100]));
        check(&values);
        assert_eq!(reference(&values), 50);
        // A step above one bf16 ulp is a strict winner regardless of position.
        values[59000] = 1.0 + 1.0 / 64.0;
        check(&values);
        assert_eq!(reference(&values), 59000);
    }

    #[test]
    fn wide_fold_special_values() {
        let all_neg_inf = vec![f32::NEG_INFINITY; ROWS];
        check(&all_neg_inf);
        assert_eq!(reference(&all_neg_inf), 0);
        let all_nan = vec![f32::NAN; ROWS];
        check(&all_nan);
        assert_eq!(reference(&all_nan), NO_INDEX);
        // NaN never wins against a finite value or an infinity.
        let mut values = vec![f32::NAN; ROWS];
        values[777] = -3.0;
        check(&values);
        assert_eq!(reference(&values), 777);
        let mut values = vec![1.0f32; ROWS];
        values[5] = f32::NAN;
        values[59000] = f32::INFINITY;
        values[4000] = f32::INFINITY;
        check(&values);
        assert_eq!(reference(&values), 4000);
        // Signed zeros compare equal: the lowest index wins.
        let mut values = vec![-1.0f32; ROWS];
        values[10] = 0.0;
        values[9] = -0.0;
        check(&values);
        assert_eq!(reference(&values), 9);
        // A value that overflows to infinity only after bf16 rounding.
        let mut values = vec![1.0f32; ROWS];
        values[100] = f32::MAX;
        assert!(round(f32::MAX).is_infinite());
        check(&values);
        assert_eq!(reference(&values), 100);
    }
}
