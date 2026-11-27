//! Varlen and paged-decode attention: geometry rules and split planning
//! (always built) and device launch wrappers (feature `cuda`).

/// Keys per split-K partial boundary; split chunks are multiples of this.
pub const SPLIT_ALIGN: usize = 64;
/// Partials written per split by one decode launch.
pub const PARTIALS_PER_SPLIT: usize = 1;
/// Partials per split written by the baseline kernel (one per warp).
#[cfg(feature = "cuda")]
const BASELINE_PARTIALS_PER_SPLIT: usize = 4;
/// Head dimension of the paged decode kernel.
pub const DECODE_HEAD_DIM: usize = 128;
/// Largest `(q heads per KV head) * q_len` one decode launch supports.
pub const DECODE_MAX_ROWS: usize = 16;
/// Query rows per thread block of the varlen kernel.
pub const VARLEN_BLOCK_Q: usize = 64;

/// Shape of one varlen attention launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarlenGeometry {
    pub q_heads: usize,
    pub kv_heads: usize,
    /// 64 or 128.
    pub head_dim: usize,
    pub causal: bool,
}

impl VarlenGeometry {
    /// Vision tower attention: `heads` heads of 64, non-causal.
    pub fn vision(heads: usize) -> Self {
        Self {
            q_heads: heads,
            kv_heads: heads,
            head_dim: 64,
            causal: false,
        }
    }

    /// Text prefill attention: head dimension 128, causal.
    pub fn prefill(q_heads: usize, kv_heads: usize) -> Self {
        Self {
            q_heads,
            kv_heads,
            head_dim: 128,
            causal: true,
        }
    }

    /// Checks the geometry against the kernel limits.
    pub fn validate(&self) -> Result<(), String> {
        if self.head_dim != 64 && self.head_dim != 128 {
            return Err(format!("head_dim {} must be 64 or 128", self.head_dim));
        }
        if self.kv_heads == 0 || !self.q_heads.is_multiple_of(self.kv_heads) {
            return Err(format!(
                "q_heads {} must be a positive multiple of kv_heads {}",
                self.q_heads, self.kv_heads
            ));
        }
        Ok(())
    }

    /// Name of the entry point that serves this geometry.
    pub fn kernel_name(&self) -> &'static str {
        match (self.head_dim, self.causal) {
            (64, false) => "scriptor_attn_varlen_d64_full",
            (64, true) => "scriptor_attn_varlen_d64_causal",
            (128, false) => "scriptor_attn_varlen_d128_full",
            _ => "scriptor_attn_varlen_d128_causal",
        }
    }

    /// Thread blocks along x for the longest sequence.
    pub fn grid_x(max_seqlen: usize) -> usize {
        max_seqlen.div_ceil(VARLEN_BLOCK_Q).max(1)
    }
}

/// Shape of one paged decode launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeGeometry {
    pub q_heads: usize,
    pub kv_heads: usize,
    /// Tokens per KV block.
    pub block_size: usize,
}

impl DecodeGeometry {
    /// Text decoder layer: 16 Q heads over 8 KV heads.
    pub fn text(block_size: usize) -> Self {
        Self {
            q_heads: 16,
            kv_heads: 8,
            block_size,
        }
    }

    /// Q heads sharing each KV head.
    pub fn group(&self) -> usize {
        self.q_heads / self.kv_heads
    }

    /// Checks the geometry and a query length against the kernel limits.
    pub fn validate(&self, q_len: usize) -> Result<(), String> {
        if self.kv_heads == 0 || !self.q_heads.is_multiple_of(self.kv_heads) {
            return Err(format!(
                "q_heads {} must be a positive multiple of kv_heads {}",
                self.q_heads, self.kv_heads
            ));
        }
        if self.block_size == 0 {
            return Err("block_size must be positive".into());
        }
        if q_len == 0 {
            return Err("q_len must be at least 1".into());
        }
        if self.group() * q_len > DECODE_MAX_ROWS {
            return Err(format!(
                "group {} * q_len {q_len} exceeds {DECODE_MAX_ROWS} rows per launch",
                self.group()
            ));
        }
        Ok(())
    }
}

/// Picks the split-K factor so `batch * kv_heads * splits` blocks cover
/// `sm_count` SMs twice over, never below one 64-key chunk per split.
pub fn choose_splits(max_ctx: usize, batch: usize, kv_heads: usize, sm_count: usize) -> usize {
    let chunks = max_ctx.div_ceil(SPLIT_ALIGN).max(1);
    let base = (batch * kv_heads).max(1);
    (2 * sm_count).div_ceil(base).max(1).min(chunks)
}

/// Keys each split covers: `ceil(max_ctx / splits)` rounded up to [`SPLIT_ALIGN`].
pub fn split_chunk(max_ctx: usize, splits: usize) -> usize {
    max_ctx.div_ceil(splits.max(1)).div_ceil(SPLIT_ALIGN).max(1) * SPLIT_ALIGN
}

/// Rows of split partials (`[seq][split][q_len][q_heads]`) for a launch.
pub fn partial_rows(batch: usize, splits: usize, q_len: usize, q_heads: usize) -> usize {
    batch * splits * PARTIALS_PER_SPLIT * q_len * q_heads
}

/// Most (sequence, split) pairs the pair-order kernel ranks in one launch.
pub const PAIR_ORDER_MAX_PAIRS: usize = 4096;
/// Pair count from which the hoisted order is used by [`PairOrderMode::Hoisted`] when selected
/// with `on`.
pub const PAIR_ORDER_DEFAULT_MIN_PAIRS: usize = 32;
/// Environment variable that selects the pair-order mode.
pub const PAIR_ORDER_ENV: &str = "SCRIPTOR_ATTN_PAIR_ORDER";

/// Length of the key range the (sequence, split) pair `pair` covers: the sequence length less
/// the keys of earlier splits, clamped to `0 ..= chunk`.
pub fn pair_key(seq_lens: &[i32], splits: usize, chunk: usize, pair: usize) -> i32 {
    let before = (pair % splits) as i64 * chunk as i64;
    let len = i64::from(seq_lens[pair / splits]) - before;
    len.clamp(0, chunk as i64) as i32
}

/// Dispatch order of the `seq_lens.len() * splits` pairs: entry `r` is the pair of rank `r`
/// when pairs are sorted by descending [`pair_key`], ties broken by ascending pair index. This
/// is the rank every decode block computes for itself in the unhoisted kernel.
pub fn pair_order_reference(seq_lens: &[i32], splits: usize, chunk: usize) -> Vec<i32> {
    let pairs = seq_lens.len() * splits;
    let mut order = vec![0i32; pairs];
    for i in 0..pairs {
        let wi = pair_key(seq_lens, splits, chunk, i);
        let rank = (0..pairs)
            .filter(|&j| {
                let wj = pair_key(seq_lens, splits, chunk, j);
                wj > wi || (wj == wi && j < i)
            })
            .count();
        order[rank] = i as i32;
    }
    order
}

/// Whether decode attention ranks its work pairs once per pass or in every block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PairOrderMode {
    /// Every block ranks the pairs itself.
    #[default]
    Off,
    /// One launch per pass ranks the pairs once when there are at least `min_pairs`.
    Hoisted { min_pairs: usize },
}

impl PairOrderMode {
    /// Mode named by the value of [`PAIR_ORDER_ENV`]: `on` hoists from
    /// [`PAIR_ORDER_DEFAULT_MIN_PAIRS`] pairs, `all` at every pair count; anything else, or no
    /// value, leaves it off.
    pub fn from_value(value: Option<&str>) -> Self {
        match value {
            Some("on") => Self::Hoisted {
                min_pairs: PAIR_ORDER_DEFAULT_MIN_PAIRS,
            },
            Some("all") => Self::Hoisted { min_pairs: 1 },
            _ => Self::Off,
        }
    }

    /// Mode selected by [`PAIR_ORDER_ENV`].
    pub fn from_env() -> Self {
        Self::from_value(std::env::var(PAIR_ORDER_ENV).ok().as_deref())
    }

    /// Whether a launch of `pairs` pairs uses the hoisted order.
    pub fn hoists(&self, pairs: usize) -> bool {
        match *self {
            Self::Off => false,
            Self::Hoisted { min_pairs } => {
                pairs >= min_pairs.max(1) && pairs <= PAIR_ORDER_MAX_PAIRS
            }
        }
    }
}

/// Environment variable that selects the resident-block count of decode attention.
pub const RESIDENT_ENV: &str = "SCRIPTOR_ATTN_RESIDENT";

/// Blocks per SM the decode attention kernel is compiled for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AttnResidency {
    /// Two resident blocks per SM.
    #[default]
    Two,
    /// Three resident blocks per SM.
    Three,
}

impl AttnResidency {
    /// Residency named by the value of [`RESIDENT_ENV`]: no value or `2` is two blocks, `3` is
    /// three; any other value, including an empty one, is an error naming the variable.
    pub fn from_value(value: Option<&str>) -> Result<Self, String> {
        match value {
            None | Some("2") => Ok(Self::Two),
            Some("3") => Ok(Self::Three),
            Some(other) => Err(format!("{RESIDENT_ENV} must be 2 or 3, got {other:?}")),
        }
    }

    /// Residency selected by [`RESIDENT_ENV`].
    pub fn from_env() -> Result<Self, String> {
        match std::env::var(RESIDENT_ENV) {
            Ok(v) => Self::from_value(Some(&v)),
            Err(std::env::VarError::NotPresent) => Ok(Self::Two),
            Err(std::env::VarError::NotUnicode(_)) => Err(format!(
                "{RESIDENT_ENV} must be 2 or 3, got a non-unicode value"
            )),
        }
    }

    /// Resident blocks per SM.
    pub const fn blocks(self) -> usize {
        match self {
            Self::Two => 2,
            Self::Three => 3,
        }
    }
}

/// Embedded module holding the decode kernel compiled for `residency`, with or without the
/// hoisted pair-order lookup.
pub const fn decode_module_name(residency: AttnResidency, ordered: bool) -> &'static str {
    match (residency, ordered) {
        (AttnResidency::Two, false) => "attn_paged_decode",
        (AttnResidency::Three, false) => "attn_paged_decode_r3",
        (AttnResidency::Two, true) => "attn_paged_decode_ordered",
        (AttnResidency::Three, true) => "attn_paged_decode_ordered_r3",
    }
}

/// Bytes of the pair-order buffer holding `pairs` entries.
pub const fn pair_order_bytes(pairs: usize) -> usize {
    pairs * std::mem::size_of::<i32>()
}

/// Pair-order entries a launch of `batch` sequences at `splits` splits needs under `mode`:
/// its `batch * splits` pairs when the mode hoists them, otherwise none.
pub fn pair_order_pairs(batch: usize, splits: usize, mode: PairOrderMode) -> usize {
    let pairs = batch * splits;
    if mode.hoists(pairs) { pairs } else { 0 }
}

/// Split-K workspace capacity: partial rows and pair-order entries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodeCapacity {
    /// Rows of split partials.
    pub rows: usize,
    /// Entries of the pair order; zero when no covered launch hoists.
    pub order_pairs: usize,
}

impl DecodeCapacity {
    /// Capacity that covers every `(batch, splits, q_len)` launch of `shapes`. The partial rows
    /// and the pair-order entries are maximised independently, because the launch with the most
    /// rows need not be the launch with the most pairs.
    pub fn covering(
        shapes: impl IntoIterator<Item = (usize, usize, usize)>,
        q_heads: usize,
        mode: PairOrderMode,
    ) -> Self {
        shapes
            .into_iter()
            .fold(Self::default(), |cap, (batch, splits, q_len)| Self {
                rows: cap.rows.max(partial_rows(batch, splits, q_len, q_heads)),
                order_pairs: cap.order_pairs.max(pair_order_pairs(batch, splits, mode)),
            })
    }

    /// Whether this capacity holds every launch `need` describes.
    pub fn holds(&self, need: &Self) -> bool {
        self.rows >= need.rows && self.order_pairs >= need.order_pairs
    }
}

/// How the decode rows of one layer reach paged attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeRoute {
    /// The decode rows are staged through separate buffers: the reference route.
    Staged,
    /// Pure decode, attention reads and writes the activation buffers in place.
    Direct,
    /// Direct pure decode with the hoisted pair order.
    DirectOrdered,
}

/// Route for a layer. A batch with prefill rows (`mixed`) is always staged, as is every batch
/// unless `direct` opts in to the in-place buffers. Under `direct` the hoisted order is used
/// when `mode` selects `pairs` and `order_capacity` holds them; otherwise the unordered kernel
/// runs.
pub fn decode_route(
    mixed: bool,
    direct: bool,
    mode: PairOrderMode,
    pairs: usize,
    order_capacity: usize,
) -> DecodeRoute {
    if mixed || !direct {
        DecodeRoute::Staged
    } else if mode.hoists(pairs) && pairs <= order_capacity {
        DecodeRoute::DirectOrdered
    } else {
        DecodeRoute::Direct
    }
}

#[cfg(feature = "cuda")]
pub use device::*;

#[cfg(feature = "cuda")]
mod device {
    use std::sync::Arc;

    use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
    use scriptor_cuda::{CudaError, Kernel, KernelModule, Result};

    use super::{
        AttnResidency, DECODE_HEAD_DIM, DecodeCapacity, DecodeGeometry, PAIR_ORDER_MAX_PAIRS,
        PARTIALS_PER_SPLIT, VarlenGeometry, choose_splits, decode_module_name, partial_rows,
        split_chunk,
    };

    fn module(ctx: &Arc<CudaContext>, name: &str) -> Result<KernelModule> {
        let image = crate::fatbins::FATBINS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, b)| *b)
            .ok_or_else(|| CudaError::Module(format!("kernel module {name} is not embedded")))?;
        KernelModule::load(ctx, image)
    }

    fn drv<T>(r: std::result::Result<T, cudarc::driver::DriverError>) -> Result<T> {
        r.map_err(CudaError::from)
    }

    /// Loaded varlen attention entry points.
    pub struct VarlenAttention {
        kernels: [Kernel; 4],
        _module: KernelModule,
    }

    impl VarlenAttention {
        /// Loads the embedded varlen attention module.
        pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
            let module = module(ctx, "attn_varlen")?;
            let load = |g: VarlenGeometry| module.kernel(g.kernel_name());
            let kernels = [
                load(VarlenGeometry::vision(1))?,
                load(VarlenGeometry {
                    causal: true,
                    ..VarlenGeometry::vision(1)
                })?,
                load(VarlenGeometry {
                    causal: false,
                    ..VarlenGeometry::prefill(1, 1)
                })?,
                load(VarlenGeometry::prefill(1, 1))?,
            ];
            Ok(Self {
                kernels,
                _module: module,
            })
        }

        /// Attention over concatenated sequences.
        ///
        /// `q` and `out` hold BF16 bits laid out `[tokens][q_heads][head_dim]`,
        /// `k` and `v` `[tokens][kv_heads][head_dim]`; `cu_seqlens` is
        /// `[num_seqs + 1]` token offsets and `max_seqlen` bounds the longest
        /// sequence. Attention never crosses a sequence boundary.
        #[allow(clippy::too_many_arguments)]
        pub fn run(
            &self,
            stream: &Arc<CudaStream>,
            geo: &VarlenGeometry,
            q: &CudaSlice<u16>,
            k: &CudaSlice<u16>,
            v: &CudaSlice<u16>,
            out: &mut CudaSlice<u16>,
            cu_seqlens: &CudaSlice<i32>,
            num_seqs: usize,
            max_seqlen: usize,
            scale: f32,
        ) -> Result<()> {
            geo.validate().map_err(CudaError::Module)?;
            if num_seqs == 0 || max_seqlen == 0 {
                return Ok(());
            }
            let which = match (geo.head_dim, geo.causal) {
                (64, false) => 0,
                (64, true) => 1,
                (128, false) => 2,
                _ => 3,
            };
            let (qh, kh) = (geo.q_heads as i32, geo.kv_heads as i32);
            let cfg = LaunchConfig {
                grid_dim: (
                    VarlenGeometry::grid_x(max_seqlen) as u32,
                    geo.q_heads as u32,
                    num_seqs as u32,
                ),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: argument types match the kernel signature; the kernel
            // bounds every access by `cu_seqlens`.
            unsafe {
                let mut b = stream.launch_builder(&self.kernels[which]);
                b.arg(q)
                    .arg(k)
                    .arg(v)
                    .arg(out)
                    .arg(cu_seqlens)
                    .arg(&qh)
                    .arg(&kh)
                    .arg(&scale);
                drv(b.launch(cfg))?;
            }
            Ok(())
        }
    }

    /// Scratch for split-K partial results.
    pub struct DecodeWorkspace {
        part_o: CudaSlice<f32>,
        part_m: CudaSlice<f32>,
        part_l: CudaSlice<f32>,
        order: CudaSlice<i32>,
        order_for: Option<(usize, usize, usize)>,
        rows: usize,
    }

    impl DecodeWorkspace {
        /// Sizes scratch for up to `batch` sequences, `splits` splits and `q_len` queries,
        /// without a pair order.
        pub fn alloc(
            stream: &Arc<CudaStream>,
            batch: usize,
            splits: usize,
            q_len: usize,
            q_heads: usize,
        ) -> Result<Self> {
            let rows = partial_rows(batch, splits, q_len, q_heads);
            Self::with_capacity(
                stream,
                DecodeCapacity {
                    rows,
                    order_pairs: 0,
                },
            )
        }

        /// Sizes scratch to `capacity`; the pair order holds `capacity.order_pairs` entries
        /// (at most [`PAIR_ORDER_MAX_PAIRS`]) and is not allocated when that is zero.
        pub fn with_capacity(stream: &Arc<CudaStream>, capacity: DecodeCapacity) -> Result<Self> {
            let rows = capacity.rows.max(1);
            let order = capacity.order_pairs.min(PAIR_ORDER_MAX_PAIRS);
            Ok(Self {
                part_o: drv(stream.alloc_zeros::<f32>(rows * DECODE_HEAD_DIM))?,
                part_m: drv(stream.alloc_zeros::<f32>(rows))?,
                part_l: drv(stream.alloc_zeros::<f32>(rows))?,
                order: drv(stream.alloc_zeros::<i32>(order))?,
                order_for: None,
                rows,
            })
        }

        /// The capacity this workspace holds.
        pub fn capacity(&self) -> DecodeCapacity {
            DecodeCapacity {
                rows: self.rows,
                order_pairs: self.order.len(),
            }
        }

        /// Downloads the first `pairs` entries of the pair order; for tests.
        #[doc(hidden)]
        pub fn read_pair_order(&self, stream: &Arc<CudaStream>, pairs: usize) -> Result<Vec<i32>> {
            let all = drv(stream.clone_dtoh(&self.order))?;
            Ok(all[..pairs.min(all.len())].to_vec())
        }

        /// Forgets the pair order, so an ordered launch fails until it is built again.
        pub fn invalidate_pair_order(&mut self) {
            self.order_for = None;
        }

        /// Whether the pair order was built for exactly `batch`, `splits` and `chunk`.
        pub fn pair_order_matches(&self, batch: usize, splits: usize, chunk: usize) -> bool {
            self.order_for == Some((batch, splits, chunk))
        }
    }

    /// Loaded kernel that ranks the work pairs of a decode pass.
    pub struct PairOrderKernel {
        kernel: Kernel,
        _module: KernelModule,
    }

    impl PairOrderKernel {
        /// Loads the embedded pair-order module.
        pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
            let module = module(ctx, "attn_pair_order")?;
            Ok(Self {
                kernel: module.kernel("scriptor_attn_pair_order")?,
                _module: module,
            })
        }

        /// Writes the dispatch order of the `batch * splits` pairs of `seq_lens` into `ws` with
        /// one launch and no host read-back. The order stays valid for exactly this `batch`,
        /// `splits` and `chunk` until the next build or [`DecodeWorkspace::invalidate_pair_order`];
        /// callers rebuild it whenever `seq_lens` is rewritten.
        pub fn build(
            &self,
            stream: &Arc<CudaStream>,
            seq_lens: &CudaSlice<i32>,
            ws: &mut DecodeWorkspace,
            batch: usize,
            splits: usize,
            chunk: usize,
        ) -> Result<()> {
            let pairs = batch * splits;
            if pairs == 0 || pairs > PAIR_ORDER_MAX_PAIRS || pairs > ws.order.len() {
                return Err(CudaError::Module(format!(
                    "{pairs} pairs outside the pair-order limits"
                )));
            }
            if seq_lens.len() < batch {
                return Err(CudaError::Module("seq_lens shorter than the batch".into()));
            }
            ws.order_for = None;
            let (sp, ch, np) = (splits as i32, chunk as i32, pairs as i32);
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: (pairs * std::mem::size_of::<i32>()) as u32,
            };
            // SAFETY: argument types match the kernel signature; it reads `batch` lengths and
            // writes exactly `pairs` entries of `ws.order`, which holds at least that many.
            unsafe {
                let mut b = stream.launch_builder(&self.kernel);
                b.arg(seq_lens)
                    .arg(&sp)
                    .arg(&ch)
                    .arg(&np)
                    .arg(&mut ws.order);
                drv(b.launch(cfg))?;
            }
            ws.order_for = Some((batch, splits, chunk));
            Ok(())
        }
    }

    /// Per-sequence block tables, row-major `[seq][max_blocks]`.
    pub struct SeqTables<'a> {
        pub block_tables: &'a CudaSlice<i32>,
        pub max_blocks: usize,
    }

    /// Loaded paged decode attention entry points.
    pub struct PagedDecodeAttention {
        decode: Kernel,
        reduce: Kernel,
        partials: usize,
        ordered: bool,
        _module: KernelModule,
    }

    impl PagedDecodeAttention {
        /// Loads the embedded paged decode module.
        pub fn load(ctx: &Arc<CudaContext>) -> Result<Self> {
            Self::load_with(ctx, AttnResidency::Two, false)
        }

        /// Loads the variant whose blocks read their work pair from the order built by
        /// [`PairOrderKernel::build`]; its arithmetic and reduction match the default kernel.
        pub fn load_ordered(ctx: &Arc<CudaContext>) -> Result<Self> {
            Self::load_with(ctx, AttnResidency::Two, true)
        }

        /// Loads the decode kernel compiled for `residency` resident blocks per SM, with the
        /// hoisted pair-order lookup when `ordered`. Every residency computes the same bits.
        pub fn load_with(
            ctx: &Arc<CudaContext>,
            residency: AttnResidency,
            ordered: bool,
        ) -> Result<Self> {
            let module = module(ctx, decode_module_name(residency, ordered))?;
            let name = if ordered {
                "scriptor_paged_attn_decode_d128_ordered"
            } else {
                "scriptor_paged_attn_decode_d128"
            };
            Ok(Self {
                decode: module.kernel(name)?,
                reduce: module.kernel("scriptor_paged_attn_reduce")?,
                partials: PARTIALS_PER_SPLIT,
                ordered,
                _module: module,
            })
        }

        /// Blocks per SM the driver reports for the decode kernel at its 128-thread launch.
        #[doc(hidden)]
        pub fn resident_blocks_per_sm(&self) -> Result<usize> {
            self.decode
                .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)
                .map(|n| n as usize)
                .map_err(CudaError::from)
        }

        /// Loads the reference kernel: the grid-order launch of the same per-item
        /// computation, which the production kernel must reproduce bit for bit.
        #[doc(hidden)]
        pub fn load_reference(ctx: &Arc<CudaContext>) -> Result<Self> {
            let module = module(ctx, "attn_paged_decode_reference")?;
            Ok(Self {
                decode: module.kernel("scriptor_paged_attn_reference_d128")?,
                reduce: module.kernel("scriptor_paged_attn_reference_reduce")?,
                partials: PARTIALS_PER_SPLIT,
                ordered: false,
                _module: module,
            })
        }

        /// Fills `bytes` of shared memory on every SM that `blocks` blocks reach with
        /// NaN so a following launch exposes any read of unwritten shared memory.
        #[doc(hidden)]
        pub fn poison_shared_memory(
            ctx: &Arc<CudaContext>,
            stream: &Arc<CudaStream>,
            blocks: u32,
            bytes: u32,
        ) -> Result<()> {
            let module = module(ctx, "attn_paged_decode_reference")?;
            let kernel = module.kernel("scriptor_poison_smem")?;
            let n = bytes as i32;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: bytes,
            };
            // SAFETY: the kernel takes one i32 and writes only `bytes` of its shared memory.
            unsafe {
                let mut b = stream.launch_builder(&kernel);
                b.arg(&n);
                drv(b.launch(cfg))?;
            }
            Ok(())
        }

        /// Loads the baseline kernel (four partials per split) that the
        /// equivalence tests and bandwidth comparisons run against. Its
        /// workspace must be allocated for four times the splits.
        #[doc(hidden)]
        pub fn load_baseline(ctx: &Arc<CudaContext>) -> Result<Self> {
            let module = module(ctx, "attn_paged_decode_baseline")?;
            Ok(Self {
                decode: module.kernel("scriptor_paged_attn_baseline_d128")?,
                reduce: module.kernel("scriptor_paged_attn_baseline_reduce")?,
                partials: super::BASELINE_PARTIALS_PER_SPLIT,
                ordered: false,
                _module: module,
            })
        }

        /// Split-K factor and chunk for the given context bound.
        pub fn plan_splits(
            geo: &DecodeGeometry,
            batch: usize,
            max_ctx: usize,
            sm_count: usize,
        ) -> (usize, usize) {
            let s = choose_splits(max_ctx, batch, geo.kv_heads, sm_count);
            (s, split_chunk(max_ctx, s))
        }

        /// Attention for `batch` sequences with `q_len` queries each.
        ///
        /// `q` and `out` hold BF16 bits laid out `[seq][q_len][q_heads][128]`;
        /// the caches are `[blocks][block_size][kv_heads][128]` BF16 bits;
        /// `seq_lens[seq]` counts cached tokens including the `q_len` new ones.
        /// `splits` and `chunk` come from [`Self::plan_splits`] and the
        /// workspace must cover at least `batch`, `splits` and `q_len`.
        #[allow(clippy::too_many_arguments)]
        pub fn run(
            &self,
            stream: &Arc<CudaStream>,
            geo: &DecodeGeometry,
            key_cache: &CudaSlice<u16>,
            value_cache: &CudaSlice<u16>,
            tables: &SeqTables<'_>,
            seq_lens: &CudaSlice<i32>,
            q: &CudaSlice<u16>,
            out: &mut CudaSlice<u16>,
            ws: &mut DecodeWorkspace,
            batch: usize,
            q_len: usize,
            splits: usize,
            chunk: usize,
            scale: f32,
        ) -> Result<()> {
            geo.validate(q_len).map_err(CudaError::Module)?;
            let parts = splits * self.partials;
            if self.ordered && !ws.pair_order_matches(batch, splits, chunk) {
                return Err(CudaError::Module(
                    "pair order not built for this batch, splits and chunk".into(),
                ));
            }
            if partial_rows(batch, parts, q_len, geo.q_heads) > ws.rows {
                return Err(CudaError::Module("decode workspace too small".into()));
            }
            let (mb, ql, qh, kh, bs, ns, ch) = (
                tables.max_blocks as i32,
                q_len as i32,
                geo.q_heads as i32,
                geo.kv_heads as i32,
                geo.block_size as i32,
                splits as i32,
                chunk as i32,
            );
            let parts = parts as i32;
            let cfg = LaunchConfig {
                grid_dim: (geo.kv_heads as u32, splits as u32, batch as u32),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let rcfg = LaunchConfig {
                grid_dim: ((q_len * geo.q_heads) as u32, batch as u32, 1),
                block_dim: (DECODE_HEAD_DIM as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: argument types match the kernel signatures and every
            // buffer is sized by the geometry validated above.
            unsafe {
                let mut b = stream.launch_builder(&self.decode);
                b.arg(q)
                    .arg(key_cache)
                    .arg(value_cache)
                    .arg(tables.block_tables)
                    .arg(&mb)
                    .arg(seq_lens)
                    .arg(&ql)
                    .arg(&qh)
                    .arg(&kh)
                    .arg(&bs)
                    .arg(&ns)
                    .arg(&ch)
                    .arg(&scale)
                    .arg(&mut ws.part_o)
                    .arg(&mut ws.part_m)
                    .arg(&mut ws.part_l);
                if self.partials == PARTIALS_PER_SPLIT {
                    b.arg(&mut *out);
                }
                if self.ordered {
                    b.arg(&ws.order);
                }
                drv(b.launch(cfg))?;
                if self.partials == PARTIALS_PER_SPLIT && splits == 1 {
                    return Ok(());
                }
                let mut b = stream.launch_builder(&self.reduce);
                b.arg(&ws.part_o)
                    .arg(&ws.part_m)
                    .arg(&ws.part_l)
                    .arg(out)
                    .arg(&ql)
                    .arg(&qh)
                    .arg(&parts);
                drv(b.launch(rcfg))?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_from_value() {
        assert_eq!(AttnResidency::from_value(None), Ok(AttnResidency::Two));
        assert_eq!(AttnResidency::from_value(Some("2")), Ok(AttnResidency::Two));
        assert_eq!(
            AttnResidency::from_value(Some("3")),
            Ok(AttnResidency::Three)
        );
        for bad in ["", "4", "0", "1", "three", "2 ", " 3", "03"] {
            let err = AttnResidency::from_value(Some(bad)).unwrap_err();
            assert!(err.contains(RESIDENT_ENV), "{bad:?}: {err}");
        }
    }

    #[test]
    fn default_residency_is_two() {
        assert_eq!(AttnResidency::default(), AttnResidency::Two);
        assert_eq!(AttnResidency::Two.blocks(), 2);
        assert_eq!(AttnResidency::Three.blocks(), 3);
    }

    #[test]
    fn resident_module_names() {
        use AttnResidency::{Three, Two};
        assert_eq!(decode_module_name(Two, false), "attn_paged_decode");
        assert_eq!(decode_module_name(Three, false), "attn_paged_decode_r3");
        assert_eq!(decode_module_name(Two, true), "attn_paged_decode_ordered");
        assert_eq!(
            decode_module_name(Three, true),
            "attn_paged_decode_ordered_r3"
        );
    }

    fn kernel_source(name: &str) -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../kernels")
            .join(name);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    #[test]
    fn resident_kernel_sources_carry_the_block_counts() {
        let base = kernel_source("attn_paged_decode.cu");
        assert!(base.contains("__launch_bounds__(kThreads, SCRIPTOR_ATTN_RESIDENT_BLOCKS)"));
        assert!(base.contains(
            "#ifndef SCRIPTOR_ATTN_RESIDENT_BLOCKS\n#define SCRIPTOR_ATTN_RESIDENT_BLOCKS 2\n#endif"
        ));
        for name in ["attn_paged_decode_r3.cu", "attn_paged_decode_ordered_r3.cu"] {
            let src = kernel_source(name);
            assert!(
                src.contains("#define SCRIPTOR_ATTN_RESIDENT_BLOCKS 3\n"),
                "{name}"
            );
            assert!(src.contains("#include \"attn_paged_decode.cu\""), "{name}");
        }
        assert!(
            kernel_source("attn_paged_decode_ordered_r3.cu")
                .contains("#define SCRIPTOR_PAGED_ATTN_ORDERED 1\n")
        );
        assert!(!kernel_source("attn_paged_decode_r3.cu").contains("SCRIPTOR_PAGED_ATTN_ORDERED"));
    }

    #[test]
    fn every_residency_module_has_a_kernel_source() {
        use AttnResidency::{Three, Two};
        for residency in [Two, Three] {
            for ordered in [false, true] {
                let name = format!("{}.cu", decode_module_name(residency, ordered));
                assert!(!kernel_source(&name).is_empty(), "{name}");
            }
        }
    }

    #[test]
    fn varlen_geometry_rules() {
        assert!(VarlenGeometry::vision(16).validate().is_ok());
        assert!(VarlenGeometry::prefill(16, 8).validate().is_ok());
        assert!(VarlenGeometry::prefill(16, 6).validate().is_err());
        let bad = VarlenGeometry {
            head_dim: 96,
            ..VarlenGeometry::vision(16)
        };
        assert!(bad.validate().is_err());
        assert_eq!(
            VarlenGeometry::vision(16).kernel_name(),
            "scriptor_attn_varlen_d64_full"
        );
        assert_eq!(
            VarlenGeometry::prefill(16, 8).kernel_name(),
            "scriptor_attn_varlen_d128_causal"
        );
        assert_eq!(VarlenGeometry::grid_x(1), 1);
        assert_eq!(VarlenGeometry::grid_x(65), 2);
    }

    #[test]
    fn decode_geometry_limits() {
        let g = DecodeGeometry::text(16);
        assert_eq!(g.group(), 2);
        assert!(g.validate(1).is_ok());
        assert!(g.validate(8).is_ok());
        assert!(g.validate(9).is_err());
        assert!(g.validate(0).is_err());
        assert!(DecodeGeometry::text(0).validate(1).is_err());
    }

    #[test]
    fn splits_cover_context() {
        for &ctx in &[1usize, 17, 300, 2048, 16384, 70000] {
            for &batch in &[1usize, 4, 32] {
                let s = choose_splits(ctx, batch, 8, 48);
                let chunk = split_chunk(ctx, s);
                assert!(s >= 1 && s <= ctx.div_ceil(SPLIT_ALIGN).max(1));
                assert_eq!(chunk % SPLIT_ALIGN, 0);
                assert!(s * chunk >= ctx, "ctx {ctx} batch {batch}");
            }
        }
        assert!(choose_splits(16384, 1, 8, 48) > choose_splits(16384, 32, 8, 48));
        assert_eq!(partial_rows(2, 3, 2, 16), 2 * 3 * 2 * 16);
    }

    /// Order computed independently with a stable sort on (descending key, index).
    fn sorted_order(seq_lens: &[i32], splits: usize, chunk: usize) -> Vec<i32> {
        let mut idx: Vec<usize> = (0..seq_lens.len() * splits).collect();
        idx.sort_by_key(|&i| std::cmp::Reverse(pair_key(seq_lens, splits, chunk, i)));
        idx.into_iter().map(|i| i as i32).collect()
    }

    #[test]
    fn pair_key_clamps_to_chunk() {
        let lens = [200, 0, 64, -5];
        assert_eq!(pair_key(&lens, 4, 64, 0), 64);
        assert_eq!(pair_key(&lens, 4, 64, 3), 8);
        assert_eq!(pair_key(&lens, 4, 64, 4), 0);
        assert_eq!(pair_key(&lens, 4, 64, 8), 64);
        assert_eq!(pair_key(&lens, 4, 64, 9), 0);
        assert_eq!(pair_key(&lens, 4, 64, 12), 0);
    }

    #[test]
    fn pair_order_is_a_bijection_with_descending_stable_keys() {
        let cases: [(&[i32], usize, usize); 6] = [
            (&[300, 17, 300, 0, 129, 64, 1, 640], 5, 128),
            (&[1], 1, 64),
            (&[0, 0, 0], 3, 64),
            (&[64; 9], 2, 64),
            (&[-3, 70, 0], 4, 64),
            (&[10_000, 3], 12, 896),
        ];
        for (lens, splits, chunk) in cases {
            let order = pair_order_reference(lens, splits, chunk);
            let pairs = lens.len() * splits;
            assert_eq!(order.len(), pairs);
            let mut seen = vec![false; pairs];
            for &p in &order {
                assert!(!std::mem::replace(&mut seen[p as usize], true), "duplicate");
            }
            for w in order.windows(2) {
                let (a, b) = (w[0] as usize, w[1] as usize);
                let (ka, kb) = (
                    pair_key(lens, splits, chunk, a),
                    pair_key(lens, splits, chunk, b),
                );
                assert!(ka > kb || (ka == kb && a < b), "{lens:?}: {a} before {b}");
            }
            assert_eq!(order, sorted_order(lens, splits, chunk));
        }
    }

    #[test]
    fn pair_order_handles_empty_and_padded_slots() {
        assert!(pair_order_reference(&[], 3, 64).is_empty());
        let order = pair_order_reference(&[130, 0, 0, 65], 2, 64);
        assert_eq!(order[..3], [0, 1, 6]);
        assert_eq!(order[3..], [7, 2, 3, 4, 5]);
    }

    #[test]
    fn pair_order_default_is_off_and_env_values_map() {
        assert_eq!(PairOrderMode::default(), PairOrderMode::Off);
        assert_eq!(PairOrderMode::from_value(None), PairOrderMode::Off);
        assert_eq!(PairOrderMode::from_value(Some("off")), PairOrderMode::Off);
        assert_eq!(PairOrderMode::from_value(Some("")), PairOrderMode::Off);
        assert_eq!(
            PairOrderMode::from_value(Some("on")),
            PairOrderMode::Hoisted {
                min_pairs: PAIR_ORDER_DEFAULT_MIN_PAIRS
            }
        );
        assert_eq!(
            PairOrderMode::from_value(Some("all")),
            PairOrderMode::Hoisted { min_pairs: 1 }
        );
    }

    #[test]
    fn pair_order_bypass_bounds() {
        let on = PairOrderMode::from_value(Some("on"));
        assert!(!PairOrderMode::Off.hoists(4096));
        assert!(!on.hoists(PAIR_ORDER_DEFAULT_MIN_PAIRS - 1));
        assert!(on.hoists(PAIR_ORDER_DEFAULT_MIN_PAIRS));
        assert!(on.hoists(PAIR_ORDER_MAX_PAIRS));
        assert!(!on.hoists(PAIR_ORDER_MAX_PAIRS + 1));
        assert!(PairOrderMode::from_value(Some("all")).hoists(1));
        assert!(!PairOrderMode::from_value(Some("all")).hoists(0));
    }

    #[test]
    fn decode_route_selection() {
        let on = PairOrderMode::from_value(Some("on"));
        let route = decode_route;
        // Staged is the default, with or without a selected order.
        assert_eq!(
            route(false, false, PairOrderMode::Off, 128, 0),
            DecodeRoute::Staged
        );
        assert_eq!(route(false, false, on, 128, 128), DecodeRoute::Staged);
        // Direct only when opted in.
        assert_eq!(
            route(false, true, PairOrderMode::Off, 128, 0),
            DecodeRoute::Direct
        );
        // A mixed batch is staged under every setting.
        assert_eq!(
            route(true, true, PairOrderMode::Off, 128, 0),
            DecodeRoute::Staged
        );
        assert_eq!(route(true, true, on, 128, 128), DecodeRoute::Staged);
        // Ordered only when selected, and it composes with Direct.
        assert_eq!(route(false, true, on, 128, 128), DecodeRoute::DirectOrdered);
        assert_eq!(route(false, true, on, 8, 128), DecodeRoute::Direct);
    }

    #[test]
    fn decode_route_falls_back_when_order_does_not_fit() {
        let all = PairOrderMode::from_value(Some("all"));
        assert_eq!(
            decode_route(false, true, all, 64, 64),
            DecodeRoute::DirectOrdered
        );
        assert_eq!(decode_route(false, true, all, 64, 63), DecodeRoute::Direct);
        assert_eq!(decode_route(false, true, all, 64, 0), DecodeRoute::Direct);
        let cap = PAIR_ORDER_MAX_PAIRS;
        assert_eq!(
            decode_route(false, true, all, cap, cap),
            DecodeRoute::DirectOrdered
        );
        assert_eq!(
            decode_route(false, true, all, cap + 1, cap + 1),
            DecodeRoute::Direct
        );
    }

    #[test]
    fn off_mode_needs_no_pair_order() {
        assert_eq!(pair_order_pairs(128, 4, PairOrderMode::Off), 0);
        let cap = DecodeCapacity::covering([(128, 4, 4), (8, 1, 1)], 16, PairOrderMode::Off);
        assert_eq!(cap.order_pairs, 0);
        assert!(cap.rows > 0);
    }

    #[test]
    fn order_pairs_are_batch_times_splits_not_partial_rows() {
        let all = PairOrderMode::from_value(Some("all"));
        // 32 sequences, 3 splits, 4 queries, 16 heads: 96 pairs against 24576 partial rows.
        assert_eq!(pair_order_pairs(32, 3, all), 96);
        assert_eq!(
            partial_rows(32, 3, 4, 16),
            32 * 3 * PARTIALS_PER_SPLIT * 4 * 16
        );
        assert_eq!(pair_order_bytes(96), 384);
        // Above the cap the launch is not hoisted and reserves nothing.
        assert_eq!(pair_order_pairs(PAIR_ORDER_MAX_PAIRS + 1, 1, all), 0);
        assert_eq!(
            pair_order_pairs(PAIR_ORDER_MAX_PAIRS, 1, all),
            PAIR_ORDER_MAX_PAIRS
        );
        // Below the `on` threshold nothing is reserved.
        let on = PairOrderMode::from_value(Some("on"));
        assert_eq!(pair_order_pairs(1, PAIR_ORDER_DEFAULT_MIN_PAIRS - 1, on), 0);
    }

    #[test]
    fn capacity_maximises_rows_and_pairs_independently() {
        let all = PairOrderMode::from_value(Some("all"));
        // (batch, splits, q_len): the first has the most partial rows, the second the most pairs.
        let wide = (4usize, 2usize, 64usize);
        let many = (64usize, 4usize, 1usize);
        let heads = 16;
        assert!(
            partial_rows(wide.0, wide.1, wide.2, heads)
                > partial_rows(many.0, many.1, many.2, heads)
        );
        assert!(many.0 * many.1 > wide.0 * wide.1);
        let cap = DecodeCapacity::covering([wide, many], heads, all);
        assert_eq!(cap.rows, partial_rows(wide.0, wide.1, wide.2, heads));
        assert_eq!(cap.order_pairs, many.0 * many.1);
        // Neither single shape's capacity covers the other.
        let only_wide = DecodeCapacity::covering([wide], heads, all);
        let only_many = DecodeCapacity::covering([many], heads, all);
        assert!(!only_wide.holds(&cap) && !only_many.holds(&cap));
        assert!(cap.holds(&only_wide) && cap.holds(&only_many));
        // Order of the shapes does not matter.
        assert_eq!(cap, DecodeCapacity::covering([many, wide], heads, all));
    }

    #[test]
    fn pure_decode_rows_are_row_major_in_place() {
        // The decode kernel addresses query row `(seq * q_len + j)` of head `h` at
        // `(seq * q_len + j) * q_heads + h` and, unsplit, writes its output at
        // `(seq * splits + split) * q_len * q_heads + j * q_heads + h` with split 0: the same
        // dense row-major index, so rows 0.. of the activation buffers need no staging copy.
        let (q_heads, q_len, splits) = (16usize, 4usize, 1usize);
        for seq in 0..5 {
            for j in 0..q_len {
                for h in 0..q_heads {
                    let read = (seq * q_len + j) * q_heads + h;
                    let write = (seq * splits) * q_len * q_heads + j * q_heads + h;
                    assert_eq!(read, write);
                }
            }
        }
    }
}
