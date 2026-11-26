//! cuBLASLt GEMM: bf16 and fp8 e4m3 row-major `C = A * W^T (+ bias) (-> GELU)`
//! with fp32 accumulation and bf16 output.
//!
//! cuBLASLt sees the problem in the TN form `D[n x m, col-major] = W^T * X`
//! over `W [k x n, col-major]` (the row-major `[n, k]` weights) and
//! `X [k x m, col-major]` (the row-major `[m, k]` activations), whose result is
//! the row-major `[m, n]` product. The host-side fp8 codec below is always
//! available; the device driver is compiled with the `cuda` feature.

use crate::heuristics::GemmDtype;

/// Largest finite magnitude of fp8 e4m3.
pub const E4M3_MAX: f32 = 448.0;

/// Decodes an fp8 e4m3 byte (bias 7, no infinities, `0x7f`/`0xff` are NaN).
pub fn e4m3_decode(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((code >> 3) & 0x0f);
    let man = f32::from(code & 0x07);
    if exp == 0x0f && code & 0x07 == 0x07 {
        return f32::NAN;
    }
    if exp == 0 {
        sign * man / 8.0 * 2f32.powi(-6)
    } else {
        sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

/// Encodes `v` as fp8 e4m3 with round-to-nearest-even, saturating to +-448.
/// NaN encodes as `0x7f`.
pub fn e4m3_encode(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7f;
    }
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
    let mag = v.abs();
    if mag >= E4M3_MAX {
        return sign | 0x7e;
    }
    // Positive finite codes 0x00..=0x7e decode monotonically.
    let (mut lo, mut hi) = (0u8, 0x7eu8);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if e4m3_decode(mid) <= mag {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let below = lo;
    let code = if below == 0x7e {
        below
    } else {
        let (a, b) = (e4m3_decode(below), e4m3_decode(below + 1));
        let (da, db) = (mag - a, b - mag);
        if da < db || (da == db && below % 2 == 0) {
            below
        } else {
            below + 1
        }
    };
    sign | code
}

/// Whether the device path of `dtype` is available for the probe result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fp8Capability {
    /// Whether cuBLASLt accepted fp8 e4m3 operands with per-row outer-vector
    /// f32 scales and bf16 output.
    pub supported: bool,
    /// Heuristic candidates found for the probe shape (0 when unsupported).
    pub candidates: usize,
    /// The first rejection, or empty when supported.
    pub detail: String,
    /// cuBLASLt library version.
    pub library_version: usize,
}

/// Bytes of one operand matrix of `rows x cols` elements.
pub fn operand_bytes(dtype: GemmDtype, rows: usize, cols: usize) -> usize {
    dtype.element_bytes() * rows * cols
}

#[cfg(feature = "cuda")]
pub use device::*;

/// `count` row indices of `0..m` spread evenly, ascending, always including the first and
/// last row (all rows when `m <= count`).
pub fn sample_rows(m: usize, count: usize) -> Vec<usize> {
    if m <= count {
        return (0..m).collect();
    }
    (0..count)
        .map(|i| i * (m - 1) / (count - 1).max(1))
        .collect()
}

/// Relative margin a candidate must clear in paired admission timing: its
/// [`ADMIT_QUANTILE`] paired difference (candidate minus first candidate) must lie below
/// `-ADMIT_MARGIN` times the first candidate's median.
pub const ADMIT_MARGIN: f64 = 0.02;
/// Quantile of the paired differences that must clear [`ADMIT_MARGIN`].
pub const ADMIT_QUANTILE: f64 = 0.75;
/// Fewest paired samples taken per candidate during admission.
pub const ADMIT_MIN_PAIRS: usize = 16;

/// Value at quantile `q` (clamped to `[0, 1]`) of `values`, interpolated linearly between
/// adjacent order statistics; `None` when empty.
pub fn quantile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let pos = q.clamp(0.0, 1.0) * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    Some(v[lo] + (v[hi] - v[lo]) * (pos - lo as f64))
}

/// Paired timings of one candidate against the first candidate: each pair holds one sample
/// of each, taken back to back.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairedTiming {
    /// Pairs measured.
    pub pairs: usize,
    /// Median of the first candidate's samples, microseconds.
    pub first_median: f64,
    /// Median of the candidate's samples, microseconds.
    pub candidate_median: f64,
    /// Quantiles 0.10, 0.25, 0.50, 0.75 and 0.90 of the per-pair difference
    /// (candidate minus first), microseconds.
    pub diff_p10: f64,
    pub diff_p25: f64,
    pub diff_median: f64,
    pub diff_p75: f64,
    pub diff_p90: f64,
}

impl PairedTiming {
    /// Summarises `first[i]` and `candidate[i]` as pair `i`; `None` when the slices are empty
    /// or of different lengths.
    pub fn from_pairs(first: &[f64], candidate: &[f64]) -> Option<Self> {
        if first.is_empty() || first.len() != candidate.len() {
            return None;
        }
        let diffs: Vec<f64> = candidate.iter().zip(first).map(|(c, f)| c - f).collect();
        let q = |x: f64| quantile(&diffs, x).expect("non-empty");
        Some(Self {
            pairs: first.len(),
            first_median: quantile(first, 0.5)?,
            candidate_median: quantile(candidate, 0.5)?,
            diff_p10: q(0.10),
            diff_p25: q(0.25),
            diff_median: q(0.50),
            diff_p75: q(ADMIT_QUANTILE),
            diff_p90: q(0.90),
        })
    }

    /// Whether the [`ADMIT_QUANTILE`] paired difference lies below `-margin` times the first
    /// candidate's median, so the candidate is faster by more than `margin` in at least that
    /// share of pairs.
    pub fn clear_gain(&self, margin: f64) -> bool {
        self.diff_p75 < -margin * self.first_median
    }

    /// `micros` relative to the first candidate's median.
    pub fn relative(&self, micros: f64) -> f64 {
        micros / self.first_median
    }
}

/// The index of the candidate in `eligible` with the lowest median paired difference among
/// those whose [`PairedTiming::clear_gain`] holds at `margin`; 0 (the first candidate) when
/// none does.
pub fn choose_paired(eligible: &[(usize, PairedTiming)], margin: f64) -> usize {
    eligible
        .iter()
        .filter(|(_, t)| t.clear_gain(margin))
        .min_by(|a, b| a.1.diff_median.total_cmp(&b.1.diff_median))
        .map_or(0, |(idx, _)| *idx)
}

/// Percentile-bootstrap interval `(lo, hi)` of the median of `values`: the `tail` and
/// `1 - tail` quantiles of the medians of `resamples` resamples with replacement drawn from a
/// [`XorShift`] seeded with `seed`. `None` when `values` is empty or `resamples` is 0.
pub fn bootstrap_median_interval(
    values: &[f64],
    resamples: usize,
    tail: f64,
    seed: u64,
) -> Option<(f64, f64)> {
    if values.is_empty() || resamples == 0 {
        return None;
    }
    let mut rng = XorShift(seed | 1);
    let mut draw = vec![0.0; values.len()];
    let medians: Vec<f64> = (0..resamples)
        .map(|_| {
            for d in draw.iter_mut() {
                *d = values[rng.next_below(values.len())];
            }
            quantile(&draw, 0.5).expect("non-empty")
        })
        .collect();
    Some((quantile(&medians, tail)?, quantile(&medians, 1.0 - tail)?))
}

/// Order of the two samples within each pair of a paired timing.
#[derive(Debug, Clone)]
pub enum PairOrder {
    /// Even pairs run the first candidate first, odd pairs the compared candidate first.
    Alternating,
    /// Each pair picks its order with a coin flip from the generator.
    Random(XorShift),
}

impl PairOrder {
    /// Whether pair `pair` runs the compared candidate before the first candidate.
    pub fn candidate_first(&mut self, pair: usize) -> bool {
        match self {
            Self::Alternating => pair % 2 == 1,
            Self::Random(rng) => rng.next_u64() & 1 == 1,
        }
    }
}

/// Deterministic xorshift64 generator. A zero state stays zero.
#[derive(Debug, Clone)]
pub struct XorShift(pub u64);

impl XorShift {
    /// Next raw 64-bit state.
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Next value in `[-1, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// Next index in `0..n` (`n > 0`).
    pub fn next_below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[cfg(feature = "cuda")]
mod device {
    use std::collections::HashMap;
    use std::sync::Arc;

    use cudarc::cublaslt::sys as lt;
    use cudarc::driver::{
        CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
    };
    use half::bf16;
    use scriptor_cuda::{CudaError, DeviceArena, Kernel, KernelModule, Result};

    use super::{
        ADMIT_MARGIN, ADMIT_MIN_PAIRS, E4M3_MAX, Fp8Capability, PairOrder, PairedTiming, XorShift,
        choose_paired, e4m3_encode, operand_bytes, sample_rows,
    };
    use crate::heuristics::{
        AlgoBinding, AlgoIdentity, Epilogue, ExecContext, GemmDtype, GemmRoute, Selection,
        ShapeInventory, ShapeKey, TuneEntry, TuneKey, TuningTable, select_candidate,
    };

    /// A device address.
    pub type DevPtr = u64;

    /// Heuristic candidates requested per problem.
    const CANDIDATES: usize = 16;
    /// Default cuBLASLt workspace size.
    pub const DEFAULT_WORKSPACE: usize = 64 << 20;
    /// Alignment of the workspace within the arena.
    const WORKSPACE_ALIGN: usize = 256;

    /// Shape of one GEMM: `[m, k]` activations by `[n, k]` weights.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GemmProblem {
        pub m: usize,
        pub n: usize,
        pub k: usize,
        pub dtype: GemmDtype,
        pub epilogue: Epilogue,
    }

    impl GemmProblem {
        /// Report key of the problem's row bucket.
        pub fn tune_key(&self) -> TuneKey {
            TuneKey::new(self.m, self.n, self.k, self.epilogue, self.dtype)
        }

        /// Key of the problem in the tuning table: its exact shape.
        pub fn shape_key(&self) -> ShapeKey {
            ShapeKey {
                m: self.m,
                n: self.n,
                k: self.k,
                epilogue: self.epilogue,
                dtype: self.dtype,
            }
        }
    }

    /// Device operands of one GEMM. Unused pointers are 0.
    #[derive(Debug, Clone, Copy, Default)]
    pub struct GemmOperands {
        /// Activations `[m, k]`, row-major.
        pub x: DevPtr,
        /// Weights `[n, k]`, row-major.
        pub w: DevPtr,
        /// bf16 bias `[n]` for the bias epilogues.
        pub bias: DevPtr,
        /// bf16 output `[m, n]`, row-major.
        pub out: DevPtr,
        /// fp8 only: f32 per-token activation scales `[m]`.
        pub x_scale: DevPtr,
        /// fp8 only: f32 weight scale expanded to `[n]` (see
        /// [`GemmEngine::fill_f32`]).
        pub w_scale: DevPtr,
    }

    /// A cuBLASLt scratch region carved from the device arena.
    #[derive(Debug, Clone, Copy)]
    pub struct Workspace {
        pub ptr: DevPtr,
        pub len: usize,
    }

    impl Workspace {
        /// Reserves `len` bytes from `arena`.
        pub fn from_arena(
            arena: &mut DeviceArena,
            stream: &Arc<CudaStream>,
            len: usize,
        ) -> Result<Self> {
            let range = arena.alloc(len, WORKSPACE_ALIGN)?;
            let base = arena.buffer().device_ptr(stream).0;
            Ok(Self {
                ptr: base + range.offset as u64,
                len: range.len,
            })
        }
    }

    /// Engine construction options.
    #[derive(Debug, Clone, Copy)]
    pub struct GemmOptions {
        /// Enables the fp8 path; fp8 calls fail while it is off.
        pub fp8: bool,
        /// Workspace bytes reserved from the arena.
        pub workspace_bytes: usize,
    }

    impl Default for GemmOptions {
        fn default() -> Self {
            Self {
                fp8: false,
                workspace_bytes: DEFAULT_WORKSPACE,
            }
        }
    }

    fn check(op: &str, st: lt::cublasStatus_t) -> Result<()> {
        if st == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(CudaError::Driver(format!("cublasLt {op}: {st:?}")))
        }
    }

    struct Handle(lt::cublasLtHandle_t);
    // SAFETY: the handle is only used through `&mut self` entry points of its owner.
    unsafe impl Send for Handle {}
    impl Drop for Handle {
        fn drop(&mut self) {
            // SAFETY: created by cublasLtCreate and destroyed once.
            unsafe { lt::cublasLtDestroy(self.0) };
        }
    }

    struct Desc(lt::cublasLtMatmulDesc_t);
    // SAFETY: descriptors are only used through `&mut self` entry points of their owner.
    unsafe impl Send for Desc {}
    impl Drop for Desc {
        fn drop(&mut self) {
            // SAFETY: created by cublasLtMatmulDescCreate.
            unsafe { lt::cublasLtMatmulDescDestroy(self.0) };
        }
    }

    struct Layout(lt::cublasLtMatrixLayout_t);
    // SAFETY: layouts are only used through `&mut self` entry points of their owner.
    unsafe impl Send for Layout {}
    impl Layout {
        fn new(ty: lt::cudaDataType_t, rows: usize, cols: usize) -> Result<Self> {
            let mut l = std::ptr::null_mut();
            // SAFETY: out-pointer is valid; the leading dimension equals the row count.
            check("MatrixLayoutCreate", unsafe {
                lt::cublasLtMatrixLayoutCreate(&mut l, ty, rows as u64, cols as u64, rows as i64)
            })?;
            Ok(Self(l))
        }
    }
    impl Drop for Layout {
        fn drop(&mut self) {
            // SAFETY: created by cublasLtMatrixLayoutCreate.
            unsafe { lt::cublasLtMatrixLayoutDestroy(self.0) };
        }
    }

    struct Pref(lt::cublasLtMatmulPreference_t);
    impl Drop for Pref {
        fn drop(&mut self) {
            // SAFETY: created by cublasLtMatmulPreferenceCreate.
            unsafe { lt::cublasLtMatmulPreferenceDestroy(self.0) };
        }
    }

    fn set_desc<T>(desc: &Desc, attr: lt::cublasLtMatmulDescAttributes_t, v: &T) -> Result<()> {
        // SAFETY: `v` is a live value of the attribute's documented type.
        check("MatmulDescSetAttribute", unsafe {
            lt::cublasLtMatmulDescSetAttribute(
                desc.0,
                attr,
                (v as *const T).cast(),
                std::mem::size_of::<T>(),
            )
        })
    }

    fn data_type(dtype: GemmDtype) -> lt::cudaDataType_t {
        match dtype {
            GemmDtype::Bf16 => lt::cudaDataType_t::CUDA_R_16BF,
            GemmDtype::Fp8E4m3 => lt::cudaDataType_t::CUDA_R_8F_E4M3,
        }
    }

    fn epilogue_code(e: Epilogue) -> lt::cublasLtEpilogue_t {
        use lt::cublasLtEpilogue_t as E;
        match e {
            Epilogue::None => E::CUBLASLT_EPILOGUE_DEFAULT,
            Epilogue::Bias => E::CUBLASLT_EPILOGUE_BIAS,
            Epilogue::Gelu => E::CUBLASLT_EPILOGUE_GELU,
            Epilogue::BiasGelu => E::CUBLASLT_EPILOGUE_GELU_BIAS,
        }
    }

    fn build_desc(p: &GemmProblem) -> Result<Desc> {
        use lt::cublasLtMatmulDescAttributes_t as A;
        let mut d = std::ptr::null_mut();
        // SAFETY: out-pointer is valid.
        check("MatmulDescCreate", unsafe {
            lt::cublasLtMatmulDescCreate(
                &mut d,
                lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                lt::cudaDataType_t::CUDA_R_32F,
            )
        })?;
        let desc = Desc(d);
        set_desc(&desc, A::CUBLASLT_MATMUL_DESC_TRANSA, &1i32)?;
        set_desc(&desc, A::CUBLASLT_MATMUL_DESC_TRANSB, &0i32)?;
        if p.dtype == GemmDtype::Fp8E4m3 {
            let outer = lt::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
            set_desc(&desc, A::CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &(outer as i32))?;
            set_desc(&desc, A::CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &(outer as i32))?;
        }
        if p.epilogue != Epilogue::None {
            set_desc(
                &desc,
                A::CUBLASLT_MATMUL_DESC_EPILOGUE,
                &epilogue_code(p.epilogue),
            )?;
        }
        if p.epilogue.has_bias() {
            set_desc(
                &desc,
                A::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                &lt::cudaDataType_t::CUDA_R_16BF,
            )?;
        }
        Ok(desc)
    }

    /// Everything cuBLASLt needs to run one problem shape repeatedly.
    struct Plan {
        desc: Desc,
        a: Layout,
        b: Layout,
        o: Layout,
        algos: Vec<lt::cublasLtMatmulHeuristicResult_t>,
        /// Identity of each entry of `algos`, in the same order.
        ids: Vec<AlgoIdentity>,
    }

    /// Reads one algorithm configuration attribute of `size` bytes as an unsigned value.
    fn config_attr(
        algo: &lt::cublasLtMatmulAlgo_t,
        attr: lt::cublasLtMatmulAlgoConfigAttributes_t,
        size: usize,
    ) -> Result<u32> {
        let mut buf = [0u8; 4];
        let mut written = 0usize;
        // SAFETY: `algo` is a live descriptor from the heuristic query; `buf` holds `size <= 4`
        // bytes.
        check("MatmulAlgoConfigGetAttribute", unsafe {
            lt::cublasLtMatmulAlgoConfigGetAttribute(
                algo,
                attr,
                buf.as_mut_ptr().cast(),
                size,
                &mut written,
            )
        })?;
        if written != size {
            return Err(CudaError::Driver(format!(
                "cublasLt algo config {attr:?}: wrote {written} bytes, expected {size}"
            )));
        }
        Ok(match size {
            2 => u32::from(u16::from_ne_bytes([buf[0], buf[1]])),
            _ => u32::from_ne_bytes(buf),
        })
    }

    /// The executed identity of one heuristic result.
    fn algo_identity(r: &lt::cublasLtMatmulHeuristicResult_t) -> Result<AlgoIdentity> {
        use lt::cublasLtMatmulAlgoConfigAttributes_t as C;
        let a = &r.algo;
        Ok(AlgoIdentity {
            id: config_attr(a, C::CUBLASLT_ALGO_CONFIG_ID, 4)? as i32,
            tile: config_attr(a, C::CUBLASLT_ALGO_CONFIG_TILE_ID, 4)?,
            stages: config_attr(a, C::CUBLASLT_ALGO_CONFIG_STAGES_ID, 4)?,
            splitk: config_attr(a, C::CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 4)?,
            reduction: config_attr(a, C::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, 4)?,
            swizzle: config_attr(a, C::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 4)?,
            custom: config_attr(a, C::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, 4)?,
            inner_shape: config_attr(a, C::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, 2)? as u16,
            cluster_shape: config_attr(a, C::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, 2)? as u16,
            workspace_bytes: r.workspaceSize,
            opaque: a.data,
        })
    }

    /// Plan-cache key: the exact problem.
    type PlanKey = GemmProblem;

    struct QuantKernels {
        quant_token: Kernel,
        fill: Kernel,
        _module: KernelModule,
    }

    /// cuBLASLt-backed GEMM driver over one stream.
    pub struct GemmEngine {
        stream: Arc<CudaStream>,
        handle: Handle,
        quant: QuantKernels,
        skinny: Option<crate::skinny_gemm::SkinnyGemm>,
        skinny_enabled: bool,
        wide_argmax_enabled: bool,
        workspace: Workspace,
        options: GemmOptions,
        tuning: TuningTable,
        plans: HashMap<PlanKey, Plan>,
        fp8: Option<Fp8Capability>,
        library_version: usize,
        shape_log: Option<ShapeLog>,
        shape_phase: &'static str,
        operand_capture: Option<OperandCapture>,
    }

    /// Pending request of [`GemmEngine::set_operand_capture`] and its copies so far.
    struct OperandCapture {
        wanted: Vec<(ShapeKey, Vec<usize>)>,
        seen: HashMap<ShapeKey, usize>,
        taken: Vec<CapturedOperands>,
    }

    /// Device copies of the inputs of one bf16 dispatch, with an output buffer of its shape.
    pub struct CapturedOperands {
        /// Shape of the dispatch.
        pub key: ShapeKey,
        /// How many earlier eager dispatches of the same shape the engine made since the
        /// capture request (0 for the first).
        pub occurrence: usize,
        /// Activations `[m, k]`.
        pub x: CudaSlice<u8>,
        /// Weights `[n, k]`.
        pub w: CudaSlice<u8>,
        /// Bias `[n]` for the bias epilogues.
        pub bias: Option<CudaSlice<u8>>,
        /// Output `[m, n]`, zeroed at capture.
        pub out: CudaSlice<u8>,
        stream: Arc<CudaStream>,
    }

    impl CapturedOperands {
        /// The problem the dispatch ran.
        pub fn problem(&self) -> GemmProblem {
            GemmProblem {
                m: self.key.m,
                n: self.key.n,
                k: self.key.k,
                dtype: self.key.dtype,
                epilogue: self.key.epilogue,
            }
        }

        /// Operand pointers over the copies, writing to [`Self::out`].
        pub fn operands(&self) -> GemmOperands {
            let s = &self.stream;
            GemmOperands {
                x: self.x.device_ptr(s).0,
                w: self.w.device_ptr(s).0,
                bias: self.bias.as_ref().map_or(0, |b| b.device_ptr(s).0),
                out: self.out.device_ptr(s).0,
                x_scale: 0,
                w_scale: 0,
            }
        }

        /// Host copy of the whole output buffer.
        pub fn host_out(&self) -> Result<Vec<u8>> {
            self.stream.synchronize()?;
            Ok(self.stream.clone_dtoh(&self.out)?)
        }

        /// Copies `bytes` bytes from device address `src` into a new buffer on `stream`.
        fn copy_of(stream: &Arc<CudaStream>, src: DevPtr, bytes: usize) -> Result<CudaSlice<u8>> {
            let mut dst = stream.alloc_zeros::<u8>(bytes.max(1))?;
            {
                let (dst_ptr, _write) = dst.device_ptr_mut(stream);
                // SAFETY: `src` covers `bytes` bytes by the dispatch's operand contract, `dst` was
                // just allocated with at least `bytes` bytes, and the copy is ordered on `stream`
                // with the dispatch that reads `src`. The write guard records the copy on `dst`
                // until it is enqueued.
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        dst_ptr,
                        src,
                        bytes,
                        stream.cu_stream(),
                    )
                }
                .map_err(CudaError::from)?;
            }
            Ok(dst)
        }
    }

    /// Recording of the shapes [`GemmEngine::gemm`] dispatches.
    struct ShapeLog {
        timed: bool,
        inventory: ShapeInventory,
    }

    /// Phase label of dispatches recorded before any [`GemmEngine::set_shape_phase`].
    pub const UNLABELLED_PHASE: &str = "unlabelled";

    impl GemmEngine {
        /// Creates an engine on `stream`. The workspace is reserved from
        /// `arena`; `tuning` selects algorithms through [`select_candidate`] (any problem it
        /// does not bind uses the heuristic's first candidate).
        pub fn new(
            ctx: &Arc<CudaContext>,
            stream: &Arc<CudaStream>,
            arena: &mut DeviceArena,
            tuning: TuningTable,
            options: GemmOptions,
        ) -> Result<Self> {
            let mut h = std::ptr::null_mut();
            // SAFETY: out-pointer is valid.
            check("Create", unsafe { lt::cublasLtCreate(&mut h) })?;
            let handle = Handle(h);
            let image = crate::fatbins::FATBINS
                .iter()
                .find(|(n, _)| *n == "quant_act")
                .map(|(_, b)| *b)
                .ok_or_else(|| CudaError::Module("missing fatbin quant_act".into()))?;
            let module = KernelModule::load(ctx, image)?;
            let quant = QuantKernels {
                quant_token: module.kernel("scriptor_quant_act_fp8_token")?,
                fill: module.kernel("scriptor_fill_f32")?,
                _module: module,
            };
            let workspace = Workspace::from_arena(arena, stream, options.workspace_bytes.max(1))?;
            Ok(Self {
                stream: stream.clone(),
                handle,
                quant,
                skinny: Some(crate::skinny_gemm::SkinnyGemm::new(ctx, stream)?),
                skinny_enabled: true,
                wide_argmax_enabled: false,
                workspace,
                options,
                tuning,
                plans: HashMap::new(),
                fp8: None,
                // SAFETY: plain version query.
                library_version: unsafe { lt::cublasLtGetVersion() },
                shape_log: None,
                shape_phase: UNLABELLED_PHASE,
                operand_capture: None,
            })
        }

        /// Directs later launches to `stream`. Algorithms already planned are kept as chosen.
        pub fn set_stream(&mut self, stream: &Arc<CudaStream>) {
            self.stream = stream.clone();
            if let Some(skinny) = self.skinny.as_mut() {
                skinny.set_stream(stream);
            }
        }

        /// Routes bf16 products of the skinny-GEMM shapes to the dedicated kernel (the default)
        /// or, when `enabled` is false, sends every [`Self::gemm`] call to cuBLASLt.
        pub fn set_skinny_enabled(&mut self, enabled: bool) {
            self.skinny_enabled = enabled;
        }

        /// Routes fused head argmax products of more than 64 rows to the wide kernel when `enabled`;
        /// by default (and when disabled) such products are not served by [`Self::argmax_rows`].
        pub fn set_wide_argmax_enabled(&mut self, enabled: bool) {
            self.wide_argmax_enabled = enabled;
        }

        /// The tuning table in use.
        pub fn tuning(&self) -> &TuningTable {
            &self.tuning
        }

        /// The workspace region in use.
        pub fn workspace(&self) -> Workspace {
            self.workspace
        }

        fn query_algos(
            &self,
            p: &GemmProblem,
        ) -> Result<(
            Desc,
            Layout,
            Layout,
            Layout,
            Vec<lt::cublasLtMatmulHeuristicResult_t>,
        )> {
            let desc = build_desc(p)?;
            let ab = data_type(p.dtype);
            let a = Layout::new(ab, p.k, p.n)?;
            let b = Layout::new(ab, p.k, p.m)?;
            let o = Layout::new(lt::cudaDataType_t::CUDA_R_16BF, p.n, p.m)?;
            let mut pr = std::ptr::null_mut();
            // SAFETY: out-pointer is valid.
            check("PreferenceCreate", unsafe {
                lt::cublasLtMatmulPreferenceCreate(&mut pr)
            })?;
            let pref = Pref(pr);
            let ws = self.workspace.len;
            // SAFETY: `ws` is a live usize of the attribute's type.
            check("PreferenceSetAttribute", unsafe {
                lt::cublasLtMatmulPreferenceSetAttribute(
                    pref.0,
                    lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    (&ws as *const usize).cast(),
                    std::mem::size_of::<usize>(),
                )
            })?;
            // SAFETY: zeroed results are plain data; the array has CANDIDATES slots.
            let mut res: Vec<lt::cublasLtMatmulHeuristicResult_t> =
                vec![unsafe { std::mem::zeroed() }; CANDIDATES];
            let mut found = 0i32;
            // SAFETY: all handles are live and `res` holds CANDIDATES results.
            check("AlgoGetHeuristic", unsafe {
                lt::cublasLtMatmulAlgoGetHeuristic(
                    self.handle.0,
                    desc.0,
                    a.0,
                    b.0,
                    o.0,
                    o.0,
                    pref.0,
                    CANDIDATES as i32,
                    res.as_mut_ptr(),
                    &mut found,
                )
            })?;
            res.truncate(found.max(0) as usize);
            res.retain(|r| r.state == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
            Ok((desc, a, b, o, res))
        }

        /// Probes whether cuBLASLt accepts fp8 e4m3 operands with outer-vector
        /// f32 scales and bf16 output on this device. The result is cached.
        pub fn fp8_capability(&mut self) -> &Fp8Capability {
            if self.fp8.is_none() {
                let probe = GemmProblem {
                    m: 128,
                    n: 4096,
                    k: 1024,
                    dtype: GemmDtype::Fp8E4m3,
                    epilogue: Epilogue::None,
                };
                // SAFETY: plain version query.
                let library_version = unsafe { lt::cublasLtGetVersion() };
                let (supported, candidates, detail) = match self.query_algos(&probe) {
                    Ok((_, _, _, _, v)) if v.is_empty() => {
                        (false, 0, "no heuristic candidates".to_owned())
                    }
                    Ok((_, _, _, _, v)) => (true, v.len(), String::new()),
                    Err(e) => (false, 0, e.to_string()),
                };
                self.fp8 = Some(Fp8Capability {
                    supported,
                    candidates,
                    detail,
                    library_version,
                });
            }
            self.fp8.as_ref().expect("probe stored")
        }

        fn admit(&mut self, p: &GemmProblem) -> Result<()> {
            if p.m == 0 || p.n == 0 || p.k == 0 {
                return Err(CudaError::Driver("gemm: empty problem".into()));
            }
            match p.dtype {
                GemmDtype::Bf16 => {
                    if !p.k.is_multiple_of(8) || !p.n.is_multiple_of(8) {
                        return Err(CudaError::Driver(format!(
                            "gemm bf16: n={} and k={} must be multiples of 8",
                            p.n, p.k
                        )));
                    }
                }
                GemmDtype::Fp8E4m3 => {
                    if !self.options.fp8 {
                        return Err(CudaError::Driver("gemm: fp8 path is not enabled".into()));
                    }
                    if !p.k.is_multiple_of(16) || !p.n.is_multiple_of(8) {
                        return Err(CudaError::Driver(format!(
                            "gemm fp8: k={} must be a multiple of 16 and n={} of 8",
                            p.k, p.n
                        )));
                    }
                    let cap = self.fp8_capability();
                    if !cap.supported {
                        return Err(CudaError::Driver(format!(
                            "gemm fp8: unsupported by cuBLASLt ({})",
                            cap.detail
                        )));
                    }
                }
            }
            Ok(())
        }

        fn plan(&mut self, p: &GemmProblem) -> Result<&Plan> {
            if !self.plans.contains_key(p) {
                let (desc, a, b, o, algos) = self.query_algos(p)?;
                let ids = algos
                    .iter()
                    .map(algo_identity)
                    .collect::<Result<Vec<_>>>()?;
                if algos.is_empty() {
                    return Err(CudaError::Driver(format!(
                        "cublasLt: no algorithm for {} m={} n={} k={} epilogue={}",
                        p.dtype.name(),
                        p.m,
                        p.n,
                        p.k,
                        p.epilogue.name()
                    )));
                }
                self.plans.insert(
                    *p,
                    Plan {
                        desc,
                        a,
                        b,
                        o,
                        algos,
                        ids,
                    },
                );
            }
            Ok(&self.plans[p])
        }

        /// Number of heuristic candidates for the problem.
        pub fn candidate_count(&mut self, p: &GemmProblem) -> Result<usize> {
            self.admit(p)?;
            Ok(self.plan(p)?.algos.len())
        }

        /// How the tuning table binds to the problem's own heuristic candidates (see
        /// [`select_candidate`]).
        pub fn selection(&mut self, p: &GemmProblem) -> Result<Selection> {
            self.admit(p)?;
            let ctx = ExecContext {
                m: p.m,
                workspace_len: self.workspace.len,
                library_version: self.library_version,
            };
            self.plan(p)?;
            let ids = &self.plans[p].ids;
            Ok(select_candidate(self.tuning.get(&p.shape_key()), &ctx, ids))
        }

        /// Position, in the problem's own heuristic candidate list, of the candidate
        /// [`Self::gemm`] runs for it.
        pub fn tuned_index(&mut self, p: &GemmProblem) -> Result<usize> {
            Ok(self.selection(p)?.index())
        }

        /// Identity of heuristic candidate `algo_index` of the problem.
        pub fn candidate_identity(
            &mut self,
            p: &GemmProblem,
            algo_index: usize,
        ) -> Result<AlgoIdentity> {
            self.admit(p)?;
            let ids = &self.plan(p)?.ids;
            ids.get(algo_index).copied().ok_or_else(|| {
                CudaError::Driver(format!(
                    "gemm: candidate {algo_index} of {} requested",
                    ids.len()
                ))
            })
        }

        /// Whether [`Self::argmax_rows`] serves `m` rows against an `[n, k]` weight matrix.
        pub fn supports_argmax(&self, m: usize, n: usize, k: usize) -> bool {
            self.skinny.is_some()
                && self.skinny_enabled
                && crate::skinny_gemm::argmax_route(m, n, k, self.wide_argmax_enabled).is_some()
        }

        /// Reduces the bf16 product `x[m, k] . w[n, k]^T` to the lowest column index of each
        /// row's maximum, written to the `m` u32 entries at `tokens`, without storing the
        /// product. Requires [`Self::supports_argmax`].
        pub fn argmax_rows(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            x: u64,
            w: u64,
            tokens: u64,
        ) -> Result<()> {
            if !self.supports_argmax(m, n, k) {
                return Err(scriptor_cuda::CudaError::Driver(format!(
                    "argmax_rows: unsupported m={m} n={n} k={k}"
                )));
            }
            match self.skinny.as_ref().filter(|_| self.skinny_enabled) {
                Some(skinny) => skinny.launch_argmax(m, n, k, x, w, tokens),
                None => Err(scriptor_cuda::CudaError::Driver(
                    "argmax_rows: the skinny kernels are disabled".into(),
                )),
            }
        }

        /// Starts recording the shape, route and (when `timed`) GPU time of every
        /// [`Self::gemm`] dispatch, discarding earlier records; `None` stops and discards.
        /// Dispatches made while the stream is capturing a graph are counted as captured.
        /// Timed recording synchronises the stream after each dispatch and measures nothing
        /// while the stream is capturing a graph; it never changes which algorithm runs.
        pub fn set_shape_log(&mut self, timed: Option<bool>) {
            self.shape_log = timed.map(|timed| ShapeLog {
                timed,
                inventory: ShapeInventory::default(),
            });
        }

        /// Labels later recorded dispatches with `phase`.
        pub fn set_shape_phase(&mut self, phase: &'static str) {
            self.shape_phase = phase;
        }

        /// The records gathered since recording started, leaving recording on with an empty
        /// inventory.
        pub fn take_shape_log(&mut self) -> ShapeInventory {
            self.shape_log
                .as_mut()
                .map(|log| std::mem::take(&mut log.inventory))
                .unwrap_or_default()
        }

        /// Starts copying the inputs of eager bf16 [`Self::gemm`] dispatches: for each
        /// `(shape, occurrences)` in `wanted`, the dispatches of exactly that shape whose
        /// 0-based count since this call is listed. The copies are enqueued on the stream ahead
        /// of the dispatch and never change which algorithm runs or its result. `None` stops
        /// and discards. Dispatches recorded into a graph are neither copied nor counted.
        pub fn set_operand_capture(&mut self, wanted: Option<Vec<(ShapeKey, Vec<usize>)>>) {
            self.operand_capture = wanted.map(|wanted| OperandCapture {
                wanted,
                seen: HashMap::new(),
                taken: Vec::new(),
            });
        }

        /// The copies made since capture started, leaving the request in place.
        pub fn take_operand_capture(&mut self) -> Vec<CapturedOperands> {
            self.operand_capture
                .as_mut()
                .map(|c| std::mem::take(&mut c.taken))
                .unwrap_or_default()
        }

        fn capture_operands(&mut self, p: &GemmProblem, ops: &GemmOperands) -> Result<()> {
            use cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
            if p.dtype != GemmDtype::Bf16
                || self.stream.capture_status()? != CU_STREAM_CAPTURE_STATUS_NONE
            {
                return Ok(());
            }
            let stream = self.stream.clone();
            let Some(cap) = self.operand_capture.as_mut() else {
                return Ok(());
            };
            let key = p.shape_key();
            let Some((_, occurrences)) = cap.wanted.iter().find(|(k, _)| *k == key) else {
                return Ok(());
            };
            let seen = cap.seen.entry(key).or_insert(0);
            let occurrence = *seen;
            *seen += 1;
            if !occurrences.contains(&occurrence) {
                return Ok(());
            }
            let bf = GemmDtype::Bf16;
            let x = CapturedOperands::copy_of(&stream, ops.x, operand_bytes(bf, p.m, p.k))?;
            let w = CapturedOperands::copy_of(&stream, ops.w, operand_bytes(bf, p.n, p.k))?;
            let bias = if p.epilogue.has_bias() {
                Some(CapturedOperands::copy_of(
                    &stream,
                    ops.bias,
                    operand_bytes(bf, 1, p.n),
                )?)
            } else {
                None
            };
            let out = stream.alloc_zeros::<u8>(operand_bytes(bf, p.m, p.n))?;
            cap.taken.push(CapturedOperands {
                key,
                occurrence,
                x,
                w,
                bias,
                out,
                stream,
            });
            Ok(())
        }

        fn routes_to_skinny(&self, p: &GemmProblem) -> bool {
            self.skinny.is_some()
                && self.skinny_enabled
                && p.dtype == GemmDtype::Bf16
                && p.epilogue == Epilogue::None
                && crate::skinny_gemm::supports(p.m, p.n, p.k)
        }

        /// Runs the GEMM on the skinny kernel when it serves the shape, otherwise with the
        /// cuBLASLt candidate [`Self::tuned_index`] selects.
        pub fn gemm(&mut self, p: &GemmProblem, ops: &GemmOperands) -> Result<()> {
            use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
            use cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
            let route = if self.routes_to_skinny(p) {
                GemmRoute::Skinny
            } else {
                GemmRoute::CuBlasLt(self.selection(p)?)
            };
            if self.operand_capture.is_some() {
                self.capture_operands(p, ops)?;
            }
            let (captured, start) = match &self.shape_log {
                Some(log) => {
                    let captured = self.stream.capture_status()? != CU_STREAM_CAPTURE_STATUS_NONE;
                    let start = if log.timed && !captured {
                        Some(self.stream.record_event(Some(CU_EVENT_DEFAULT))?)
                    } else {
                        None
                    };
                    (captured, start)
                }
                None => (false, None),
            };
            match route {
                GemmRoute::Skinny => {
                    let skinny = self.skinny.as_ref().expect("skinny route needs the kernel");
                    skinny.launch(p.m, p.n, p.k, ops.x, ops.w, ops.out)?;
                }
                GemmRoute::CuBlasLt(sel) => self.gemm_with(p, ops, sel.index())?,
            }
            if self.shape_log.is_some() {
                let micros = match start {
                    Some(start) => {
                        let end = self.stream.record_event(Some(CU_EVENT_DEFAULT))?;
                        end.synchronize()?;
                        Some(f64::from(start.elapsed_ms(&end)?) * 1e3)
                    }
                    None => None,
                };
                let phase = self.shape_phase;
                if let Some(log) = self.shape_log.as_mut() {
                    log.inventory
                        .record(phase, p.shape_key(), route, micros, captured);
                }
            }
            Ok(())
        }

        /// Runs the GEMM with heuristic candidate `algo_index` of the problem's own candidate
        /// list; an index past the list is an error.
        pub fn gemm_with(
            &mut self,
            p: &GemmProblem,
            ops: &GemmOperands,
            algo_index: usize,
        ) -> Result<()> {
            use lt::cublasLtMatmulDescAttributes_t as A;
            self.admit(p)?;
            if p.epilogue.has_bias() && ops.bias == 0 {
                return Err(CudaError::Driver("gemm: bias epilogue without bias".into()));
            }
            if p.dtype == GemmDtype::Fp8E4m3 && (ops.x_scale == 0 || ops.w_scale == 0) {
                return Err(CudaError::Driver("gemm fp8: missing scale vectors".into()));
            }
            let ws = self.workspace;
            let stream = self.stream.clone();
            let handle = self.handle.0;
            let plan = self.plan(p)?;
            let Some(&algo) = plan.algos.get(algo_index) else {
                return Err(CudaError::Driver(format!(
                    "gemm: candidate {algo_index} of {} requested",
                    plan.algos.len()
                )));
            };
            if p.epilogue.has_bias() {
                set_desc(&plan.desc, A::CUBLASLT_MATMUL_DESC_BIAS_POINTER, &ops.bias)?;
            }
            if p.dtype == GemmDtype::Fp8E4m3 {
                set_desc(
                    &plan.desc,
                    A::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    &ops.w_scale,
                )?;
                set_desc(
                    &plan.desc,
                    A::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                    &ops.x_scale,
                )?;
            }
            let (alpha, beta) = (1.0f32, 0.0f32);
            // SAFETY: descriptors are live; the device pointers cover the
            // problem's operands by the caller's contract; the workspace was
            // reserved from the arena at construction and covers the size
            // requested at heuristic time.
            check("Matmul", unsafe {
                lt::cublasLtMatmul(
                    handle,
                    plan.desc.0,
                    (&alpha as *const f32).cast(),
                    ops.w as *const _,
                    plan.a.0,
                    ops.x as *const _,
                    plan.b.0,
                    (&beta as *const f32).cast(),
                    ops.out as *const _,
                    plan.o.0,
                    ops.out as *mut _,
                    plan.o.0,
                    &algo.algo,
                    ws.ptr as *mut _,
                    ws.len,
                    stream.cu_stream().cast(),
                )
            })
        }

        /// Quantises bf16 activations `x [m, k]` to fp8 e4m3 `xq [m, k]` with
        /// one f32 scale per row in `scales [m]`; `xq * scales[row]` recovers
        /// the activations.
        pub fn quantize_activations(
            &self,
            x: DevPtr,
            xq: DevPtr,
            scales: DevPtr,
            m: usize,
            k: usize,
        ) -> Result<()> {
            let block = crate::launch::reduction_block(k, 4, 1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (m as u32, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            let kk = k as i32;
            let mut b = self.stream.launch_builder(&self.quant.quant_token);
            b.arg(&x).arg(&xq).arg(&scales).arg(&kk);
            // SAFETY: `x` holds m*k bf16, `xq` m*k bytes and `scales` m f32 by the caller's contract.
            unsafe { b.launch(cfg) }.map_err(CudaError::from)?;
            Ok(())
        }

        /// Writes `value` to `n` consecutive f32 values at `dst` (used to
        /// expand a per-tensor weight scale to the `[n]` vector the fp8 path
        /// takes).
        pub fn fill_f32(&self, dst: DevPtr, value: f32, n: usize) -> Result<()> {
            let cfg = LaunchConfig {
                grid_dim: (n.div_ceil(256).max(1) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let count = n as i64;
            let mut b = self.stream.launch_builder(&self.quant.fill);
            b.arg(&dst).arg(&value).arg(&count);
            // SAFETY: `dst` holds n f32 by the caller's contract.
            unsafe { b.launch(cfg) }.map_err(CudaError::from)?;
            Ok(())
        }

        /// Times `iterations` launches of candidate `algo_index` after
        /// `warmup` untimed launches; returns each launch's duration in
        /// microseconds (device events).
        pub fn time_gemm(
            &mut self,
            p: &GemmProblem,
            ops: &GemmOperands,
            algo_index: usize,
            warmup: usize,
            iterations: usize,
        ) -> Result<Vec<f64>> {
            use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
            for _ in 0..warmup {
                self.gemm_with(p, ops, algo_index)?;
            }
            let mut times = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let start = self.stream.record_event(Some(CU_EVENT_DEFAULT))?;
                self.gemm_with(p, ops, algo_index)?;
                let end = self.stream.record_event(Some(CU_EVENT_DEFAULT))?;
                end.synchronize()?;
                times.push(f64::from(start.elapsed_ms(&end)?) * 1e3);
            }
            Ok(times)
        }

        /// Times one launch of candidate `algo_index` with device events. Before the start
        /// event the stream is kept busy so the event window holds only the GEMM: with
        /// `flush` it runs [`L2Flush::run`] (the GEMM then starts with its operands out of
        /// L2), otherwise it runs [`TIMED_PRIME`] untimed launches of the same candidate (the
        /// operands then start L2-resident). Synchronises on the end event.
        pub fn timed_sample(
            &mut self,
            p: &GemmProblem,
            ops: &GemmOperands,
            algo_index: usize,
            flush: Option<&mut L2Flush>,
        ) -> Result<f64> {
            use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
            match flush {
                Some(f) => f.run(&self.stream)?,
                None => {
                    for _ in 0..TIMED_PRIME {
                        self.gemm_with(p, ops, algo_index)?;
                    }
                }
            }
            let start = self.stream.record_event(Some(CU_EVENT_DEFAULT))?;
            self.gemm_with(p, ops, algo_index)?;
            let end = self.stream.record_event(Some(CU_EVENT_DEFAULT))?;
            end.synchronize()?;
            Ok(f64::from(start.elapsed_ms(&end)?) * 1e3)
        }

        /// Takes `pairs` paired samples of candidates `[first, other]` on the same operands,
        /// each pair one [`Self::timed_sample`] of each in the order `order` gives; returns the
        /// two sample lists, index `i` of both belonging to pair `i`.
        pub fn time_paired(
            &mut self,
            p: &GemmProblem,
            ops: &GemmOperands,
            [first, other]: [usize; 2],
            pairs: usize,
            mut flush: Option<&mut L2Flush>,
            order: &mut PairOrder,
        ) -> Result<(Vec<f64>, Vec<f64>)> {
            let mut a = Vec::with_capacity(pairs);
            let mut b = Vec::with_capacity(pairs);
            for i in 0..pairs {
                if order.candidate_first(i) {
                    b.push(self.timed_sample(p, ops, other, flush.as_deref_mut())?);
                    a.push(self.timed_sample(p, ops, first, flush.as_deref_mut())?);
                } else {
                    a.push(self.timed_sample(p, ops, first, flush.as_deref_mut())?);
                    b.push(self.timed_sample(p, ops, other, flush.as_deref_mut())?);
                }
            }
            Ok((a, b))
        }

        /// Admits every problem at its exact shape on synthetic operands (see
        /// [`Self::admit_candidates`]) and writes the tuning table to `path`. This is the only
        /// call that writes a tuning file.
        pub fn tune_and_save(
            &mut self,
            problems: &[GemmProblem],
            iterations: usize,
            path: &std::path::Path,
        ) -> Result<TuningTable> {
            let mut flush = L2Flush::new(&self.stream)?;
            for p in problems {
                let operands = SyntheticOperands::new(&self.stream, p, self)?;
                self.admit_candidates(p, &operands, &mut flush, TUNE_FILE_WARMUP, iterations)?;
            }
            self.tuning
                .write_file(path)
                .map_err(|e| CudaError::Driver(e.to_string()))?;
            Ok(self.tuning.clone())
        }

        /// Admits one candidate for `p`'s exact shape on `operands` (sized for at least `p`).
        ///
        /// Every candidate runs once and is eligible when its output is bit-identical to the
        /// first candidate's on [`COMPARE_ROWS`] sampled full-width output rows. After `warmup`
        /// untimed launches of both, each eligible candidate is timed against the first in
        /// `max(iterations, ADMIT_MIN_PAIRS)` alternating-order pairs, every sample starting
        /// with L2 flushed. The recorded choice is the eligible candidate with the lowest
        /// median paired difference among those whose [`PairedTiming::clear_gain`] holds at
        /// [`ADMIT_MARGIN`]; the first candidate otherwise. The entry is bound to `p.m`, the
        /// engine's workspace length, the cuBLASLt version and the identities of the first and
        /// chosen candidates. Bit equality is observed only on the sampled rows of this one
        /// synthetic operand set; it is not established for other inputs.
        fn admit_candidates(
            &mut self,
            p: &GemmProblem,
            operands: &SyntheticOperands,
            flush: &mut L2Flush,
            warmup: usize,
            iterations: usize,
        ) -> Result<Option<BootTuned>> {
            let ops = operands.operands();
            let rows = sample_rows(p.m, COMPARE_ROWS);
            let count = self.candidate_count(p)?;
            if count == 0 {
                return Ok(None);
            }
            self.gemm_with(p, &ops, 0)?;
            let reference = operands.host_out_rows(&rows, p.n)?;
            let mut eligible_idx = Vec::new();
            let mut rejected = 0;
            for idx in 1..count {
                self.gemm_with(p, &ops, idx)?;
                if operands.host_out_rows(&rows, p.n)? == reference {
                    eligible_idx.push(idx);
                } else {
                    rejected += 1;
                }
            }
            let pairs = iterations.max(ADMIT_MIN_PAIRS);
            let mut first_samples = Vec::new();
            let mut eligible: Vec<(usize, PairedTiming)> = Vec::new();
            for &idx in &eligible_idx {
                for _ in 0..warmup {
                    self.gemm_with(p, &ops, 0)?;
                    self.gemm_with(p, &ops, idx)?;
                }
                let (a, b) = self.time_paired(
                    p,
                    &ops,
                    [0, idx],
                    pairs,
                    Some(&mut *flush),
                    &mut PairOrder::Alternating,
                )?;
                first_samples.extend_from_slice(&a);
                if let Some(t) = PairedTiming::from_pairs(&a, &b) {
                    eligible.push((idx, t));
                }
            }
            if first_samples.is_empty() {
                for _ in 0..warmup {
                    self.gemm_with(p, &ops, 0)?;
                }
                for _ in 0..iterations.max(1) {
                    first_samples.push(self.timed_sample(p, &ops, 0, Some(&mut *flush))?);
                }
            }
            let first_us = crate::heuristics::median(&first_samples).unwrap_or(f64::INFINITY);
            let algo_index = choose_paired(&eligible, ADMIT_MARGIN);
            let best = eligible
                .iter()
                .min_by(|a, b| a.1.diff_median.total_cmp(&b.1.diff_median))
                .copied();
            let micros = eligible
                .iter()
                .find(|(i, _)| *i == algo_index)
                .map_or(first_us, |(_, t)| t.candidate_median);
            let baseline = self.candidate_identity(p, 0)?;
            let chosen = self.candidate_identity(p, algo_index)?;
            let binding = AlgoBinding {
                m: p.m,
                workspace_len: self.workspace.len,
                library_version: self.library_version,
                baseline,
                chosen,
            };
            self.tuning.insert(
                p.shape_key(),
                TuneEntry {
                    algo_index,
                    micros,
                    binding: Some(binding),
                },
            );
            Ok(Some(BootTuned {
                key: p.tune_key(),
                m: p.m,
                candidates: count,
                rejected,
                algo_index,
                first_micros: first_us,
                micros,
                best,
                baseline,
                chosen,
            }))
        }

        /// Runs [`Self::admit_candidates`], in memory only, for every bf16 problem at its exact
        /// shape. A tuned choice applies only to problems of exactly that shape (see
        /// [`select_candidate`]); every other shape runs its first candidate. Shapes already
        /// bound in the tuning table (entries of other row counts do not count) and shapes the
        /// skinny kernel serves are left alone.
        pub fn autotune_in_memory(
            &mut self,
            problems: &[GemmProblem],
            iterations: usize,
        ) -> Result<Vec<BootTuned>> {
            let mut groups: std::collections::BTreeMap<(usize, usize, Epilogue), Vec<usize>> =
                std::collections::BTreeMap::new();
            for p in problems.iter().filter(|p| p.dtype == GemmDtype::Bf16) {
                if !self.tuning.needs_admission(&p.shape_key()) || self.routes_to_skinny(p) {
                    continue;
                }
                let ms = groups.entry((p.n, p.k, p.epilogue)).or_default();
                if !ms.contains(&p.m) {
                    ms.push(p.m);
                }
            }
            let mut tuned = Vec::new();
            if groups.is_empty() {
                return Ok(tuned);
            }
            let mut flush = L2Flush::new(&self.stream)?;
            for ((n, k, epilogue), mut ms) in groups {
                ms.sort_unstable();
                let top = *ms.last().expect("non-empty group");
                let shape = |m: usize| GemmProblem {
                    m,
                    n,
                    k,
                    dtype: GemmDtype::Bf16,
                    epilogue,
                };
                let operands = SyntheticOperands::new(&self.stream, &shape(top), self)?;
                for m in ms {
                    let admitted = self.admit_candidates(
                        &shape(m),
                        &operands,
                        &mut flush,
                        AUTOTUNE_WARMUP,
                        iterations,
                    )?;
                    tuned.extend(admitted);
                }
            }
            Ok(tuned)
        }
    }

    /// Untimed launches of the same candidate before each warm [`GemmEngine::timed_sample`].
    pub const TIMED_PRIME: usize = 2;

    /// A device buffer at least twice the device's L2 size whose overwrite evicts earlier
    /// data from L2.
    pub struct L2Flush {
        buf: CudaSlice<u8>,
    }

    impl L2Flush {
        /// Allocates the buffer on `stream`: twice the reported L2 size, at least 8 MiB.
        pub fn new(stream: &Arc<CudaStream>) -> Result<Self> {
            use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE;
            let l2 = stream
                .context()
                .attribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
                .map_err(CudaError::from)?;
            let len = (2 * l2.max(0) as usize).max(8 << 20);
            Ok(Self {
                buf: stream.alloc_zeros::<u8>(len)?,
            })
        }

        /// Bytes overwritten per [`Self::run`].
        pub fn len(&self) -> usize {
            self.buf.len()
        }

        /// Whether the buffer is empty (never, by construction).
        pub fn is_empty(&self) -> bool {
            self.buf.len() == 0
        }

        /// Enqueues an overwrite of the whole buffer on `stream`.
        pub fn run(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
            stream.memset_zeros(&mut self.buf)?;
            Ok(())
        }
    }

    /// Untimed launches of each compared pair before its paired samples.
    const AUTOTUNE_WARMUP: usize = 1;
    /// Untimed launches of each compared pair before its paired samples in
    /// [`GemmEngine::tune_and_save`].
    const TUNE_FILE_WARMUP: usize = 2;
    /// Output rows compared bitwise between candidates.
    const COMPARE_ROWS: usize = 64;

    /// Outcome of one admission: the shape is `key` (its row bucket) with exactly `m` rows.
    #[derive(Debug, Clone, Copy)]
    pub struct BootTuned {
        pub key: TuneKey,
        /// The exact row count measured; the choice applies only to this row count.
        pub m: usize,
        /// Heuristic candidates measured.
        pub candidates: usize,
        /// Candidates dropped for differing bitwise from the first.
        pub rejected: usize,
        /// Position of the pinned candidate in the measured list (0 when no candidate beat the
        /// first).
        pub algo_index: usize,
        /// Median of the first candidate's admission samples.
        pub first_micros: f64,
        /// Median of the pinned candidate's admission samples (`first_micros` when the first
        /// candidate is pinned).
        pub micros: f64,
        /// The eligible candidate with the lowest median paired difference and its paired
        /// timing, whether or not it was pinned; `None` when no candidate was eligible.
        pub best: Option<(usize, PairedTiming)>,
        /// Identity of the first candidate.
        pub baseline: AlgoIdentity,
        /// Identity of the pinned candidate.
        pub chosen: AlgoIdentity,
    }

    /// Fills a host buffer of `count` elements of `dtype` with pseudo-random
    /// values and returns its raw bytes.
    pub fn random_operand_bytes(dtype: GemmDtype, count: usize, rng: &mut XorShift) -> Vec<u8> {
        match dtype {
            GemmDtype::Bf16 => {
                let mut bytes = Vec::with_capacity(2 * count);
                for _ in 0..count {
                    bytes
                        .extend_from_slice(&bf16::from_f32(rng.next_f32()).to_bits().to_le_bytes());
                }
                bytes
            }
            GemmDtype::Fp8E4m3 => {
                let mut bytes = Vec::with_capacity(count);
                for _ in 0..count {
                    bytes.push(e4m3_encode(rng.next_f32() * E4M3_MAX / 8.0));
                }
                bytes
            }
        }
    }

    /// Device buffers holding random operands for one problem.
    pub struct SyntheticOperands {
        x: CudaSlice<u8>,
        w: CudaSlice<u8>,
        bias: CudaSlice<u8>,
        out: CudaSlice<u8>,
        x_scale: CudaSlice<f32>,
        w_scale: CudaSlice<f32>,
        stream: Arc<CudaStream>,
    }

    impl SyntheticOperands {
        /// Allocates and fills operands for `p`.
        pub fn new(stream: &Arc<CudaStream>, p: &GemmProblem, engine: &GemmEngine) -> Result<Self> {
            let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
            let x = stream.clone_htod(&random_operand_bytes(p.dtype, p.m * p.k, &mut rng))?;
            let w = stream.clone_htod(&random_operand_bytes(p.dtype, p.n * p.k, &mut rng))?;
            let bias = stream.clone_htod(&random_operand_bytes(GemmDtype::Bf16, p.n, &mut rng))?;
            let out = stream.alloc_zeros::<u8>(operand_bytes(GemmDtype::Bf16, p.m, p.n).max(1))?;
            let x_scale = stream.alloc_zeros::<f32>(p.m.max(1))?;
            let w_scale = stream.alloc_zeros::<f32>(p.n.max(1))?;
            engine.fill_f32(x_scale.device_ptr(stream).0, 0.01, p.m)?;
            engine.fill_f32(w_scale.device_ptr(stream).0, 0.01, p.n)?;
            stream.synchronize()?;
            Ok(Self {
                x,
                w,
                bias,
                out,
                x_scale,
                w_scale,
                stream: stream.clone(),
            })
        }

        /// Host copies of the activation, weight and bias bytes.
        pub fn host_inputs(&self) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
            Ok((
                self.stream.clone_dtoh(&self.x)?,
                self.stream.clone_dtoh(&self.w)?,
                self.stream.clone_dtoh(&self.bias)?,
            ))
        }

        /// Host copy of the listed bf16 output rows (`n` columns each), concatenated.
        pub fn host_out_rows(&self, rows: &[usize], n: usize) -> Result<Vec<u8>> {
            self.stream.synchronize()?;
            let row_bytes = 2 * n;
            let mut bytes = vec![0u8; rows.len() * row_bytes];
            for (i, &r) in rows.iter().enumerate() {
                let src = self.out.slice(r * row_bytes..(r + 1) * row_bytes);
                self.stream
                    .memcpy_dtoh(&src, &mut bytes[i * row_bytes..(i + 1) * row_bytes])?;
            }
            self.stream.synchronize()?;
            Ok(bytes)
        }

        /// The operand pointers.
        pub fn operands(&self) -> GemmOperands {
            let s = &self.stream;
            GemmOperands {
                x: self.x.device_ptr(s).0,
                w: self.w.device_ptr(s).0,
                bias: self.bias.device_ptr(s).0,
                out: self.out.device_ptr(s).0,
                x_scale: self.x_scale.device_ptr(s).0,
                w_scale: self.w_scale.device_ptr(s).0,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_rows_spans_first_to_last() {
        assert_eq!(sample_rows(5, 64), vec![0, 1, 2, 3, 4]);
        let rows = sample_rows(4096, 64);
        assert_eq!(rows.len(), 64);
        assert_eq!((rows[0], rows[63]), (0, 4095));
        assert!(rows.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(rows, sample_rows(4096, 64));
    }

    #[test]
    fn quantile_interpolates_between_order_statistics() {
        assert_eq!(quantile(&[], 0.5), None);
        let v = [4.0, 1.0, 3.0, 2.0, 5.0];
        assert_eq!(quantile(&v, 0.0), Some(1.0));
        assert_eq!(quantile(&v, 0.5), Some(3.0));
        assert_eq!(quantile(&v, 1.0), Some(5.0));
        assert_eq!(quantile(&v, 0.75), Some(4.0));
        assert_eq!(quantile(&[1.0, 2.0], 0.5), Some(1.5));
        assert_eq!(quantile(&v, 2.0), Some(5.0));
    }

    fn paired(first: &[f64], cand: &[f64]) -> PairedTiming {
        PairedTiming::from_pairs(first, cand).expect("pairs")
    }

    #[test]
    fn paired_timing_rejects_mismatched_or_empty_samples() {
        assert_eq!(PairedTiming::from_pairs(&[], &[]), None);
        assert_eq!(PairedTiming::from_pairs(&[1.0], &[1.0, 2.0]), None);
    }

    #[test]
    fn paired_timing_summarises_per_pair_differences() {
        let first = [100.0, 110.0, 90.0, 100.0];
        let cand = [95.0, 104.0, 86.0, 97.0];
        let t = paired(&first, &cand);
        assert_eq!(t.pairs, 4);
        assert_eq!(t.first_median, 100.0);
        assert_eq!(t.candidate_median, 96.0);
        assert_eq!(t.diff_median, -4.5);
        assert!(t.diff_p10 <= t.diff_p25 && t.diff_p25 <= t.diff_median);
        assert!(t.diff_median <= t.diff_p75 && t.diff_p75 <= t.diff_p90);
        assert!((t.relative(t.diff_median) + 0.045).abs() < 1e-12);
    }

    #[test]
    fn clear_gain_needs_the_upper_quartile_below_the_margin() {
        let first = [100.0; 8];
        let steady: Vec<f64> = (0..8).map(|i| 95.0 + f64::from(i % 2)).collect();
        assert!(paired(&first, &steady).clear_gain(ADMIT_MARGIN));
        let mixed = [90.0, 90.0, 90.0, 90.0, 101.0, 101.0, 101.0, 101.0];
        assert!(!paired(&first, &mixed).clear_gain(ADMIT_MARGIN));
        let small = [99.0; 8];
        assert!(!paired(&first, &small).clear_gain(ADMIT_MARGIN));
    }

    #[test]
    fn choose_paired_keeps_first_without_a_clear_gain() {
        assert_eq!(choose_paired(&[], ADMIT_MARGIN), 0);
        let first = [100.0; 8];
        let slower = paired(&first, &[101.0; 8]);
        let marginal = paired(&first, &[99.0; 8]);
        assert_eq!(
            choose_paired(&[(1, slower), (2, marginal)], ADMIT_MARGIN),
            0
        );
    }

    #[test]
    fn choose_paired_takes_the_largest_clear_gain() {
        let first = [100.0; 8];
        let a = paired(&first, &[95.0; 8]);
        let b = paired(&first, &[90.0; 8]);
        let c = paired(&first, &[101.0; 8]);
        assert_eq!(choose_paired(&[(1, a), (2, b), (3, c)], ADMIT_MARGIN), 2);
    }

    #[test]
    fn alternating_order_switches_every_pair() {
        let mut order = PairOrder::Alternating;
        let got: Vec<bool> = (0..4).map(|i| order.candidate_first(i)).collect();
        assert_eq!(got, [false, true, false, true]);
    }

    #[test]
    fn random_order_is_seeded_and_mixed() {
        let draw = |seed| {
            let mut order = PairOrder::Random(XorShift(seed));
            (0..64)
                .map(|i| order.candidate_first(i))
                .collect::<Vec<_>>()
        };
        let a = draw(0x1234_5678);
        assert_eq!(a, draw(0x1234_5678));
        let firsts = a.iter().filter(|b| **b).count();
        assert!((16..=48).contains(&firsts), "{firsts}");
    }

    #[test]
    fn bootstrap_interval_brackets_the_median_and_is_deterministic() {
        assert_eq!(bootstrap_median_interval(&[], 100, 0.05, 1), None);
        let v: Vec<f64> = (0..41).map(f64::from).collect();
        let (lo, hi) = bootstrap_median_interval(&v, 400, 0.05, 7).expect("interval");
        assert!(lo <= 20.0 && 20.0 <= hi && lo < hi, "{lo} {hi}");
        assert_eq!(bootstrap_median_interval(&v, 400, 0.05, 7), Some((lo, hi)));
        assert_eq!(
            bootstrap_median_interval(&[3.0; 5], 50, 0.05, 7),
            Some((3.0, 3.0))
        );
    }

    #[test]
    fn xorshift_values_stay_in_range() {
        let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((-1.0..1.0).contains(&v));
            assert!(rng.next_below(7) < 7);
        }
    }

    #[test]
    fn e4m3_known_values() {
        assert_eq!(e4m3_decode(0x00), 0.0);
        assert_eq!(e4m3_decode(0x38), 1.0);
        assert_eq!(e4m3_decode(0x7e), 448.0);
        assert_eq!(e4m3_decode(0xb8), -1.0);
        assert_eq!(e4m3_decode(0x01), 2f32.powi(-9));
        assert!(e4m3_decode(0x7f).is_nan());
    }

    #[test]
    fn e4m3_round_trips_every_finite_code() {
        for code in 0u8..=255 {
            let v = e4m3_decode(code);
            if v.is_nan() {
                continue;
            }
            let back = e4m3_encode(v);
            if v == 0.0 {
                assert_eq!(back & 0x7f, 0);
            } else {
                assert_eq!(back, code, "code {code:#x} value {v}");
            }
        }
    }

    #[test]
    fn e4m3_rounds_to_nearest_even_and_saturates() {
        // Between 1.0 (0x38) and 1.125 (0x39): midpoint ties to the even code.
        assert_eq!(e4m3_encode(1.0625), 0x38);
        assert_eq!(e4m3_encode(1.0626), 0x39);
        // Midpoint of 1.125 (0x39) and 1.25 (0x3a) ties to 0x3a.
        assert_eq!(e4m3_encode(1.1875), 0x3a);
        assert_eq!(e4m3_encode(1000.0), 0x7e);
        assert_eq!(e4m3_encode(-1000.0), 0xfe);
        assert_eq!(e4m3_encode(f32::NAN), 0x7f);
    }

    #[test]
    fn operand_sizes() {
        assert_eq!(operand_bytes(GemmDtype::Bf16, 3, 4), 24);
        assert_eq!(operand_bytes(GemmDtype::Fp8E4m3, 3, 4), 12);
    }
}
