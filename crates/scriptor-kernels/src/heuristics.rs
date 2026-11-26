//! GEMM algorithm-choice cache: keys, entries, the on-disk TOML table, and the
//! GLM-OCR shape set the tuner and the GEMM bench share.
//!
//! The table is read freely and written only by an explicit tune run to a
//! caller-supplied directory.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Operand format of a GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GemmDtype {
    /// bf16 activations and weights, fp32 accumulation, bf16 output.
    Bf16,
    /// fp8 e4m3 activations and weights, fp32 accumulation, bf16 output.
    Fp8E4m3,
}

impl GemmDtype {
    /// Stable name used in the tuning file and reports.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8E4m3 => "fp8_e4m3",
        }
    }

    /// Parses a name produced by [`GemmDtype::name`].
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "bf16" => Some(Self::Bf16),
            "fp8_e4m3" => Some(Self::Fp8E4m3),
            _ => None,
        }
    }

    /// Bytes per operand element.
    pub fn element_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::Fp8E4m3 => 1,
        }
    }
}

/// Point-wise epilogue fused into the GEMM output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Epilogue {
    /// Plain product.
    None,
    /// Adds a bf16 bias row `[n]`.
    Bias,
    /// Applies the cuBLASLt GELU to the product.
    Gelu,
    /// Adds the bias row, then applies the cuBLASLt GELU.
    BiasGelu,
}

impl Epilogue {
    /// Stable name used in the tuning file and reports.
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Bias => "bias",
            Self::Gelu => "gelu",
            Self::BiasGelu => "bias_gelu",
        }
    }

    /// Parses a name produced by [`Epilogue::name`].
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "none" => Some(Self::None),
            "bias" => Some(Self::Bias),
            "gelu" => Some(Self::Gelu),
            "bias_gelu" => Some(Self::BiasGelu),
            _ => Option::None,
        }
    }

    /// Whether the epilogue reads a bias row.
    pub fn has_bias(self) -> bool {
        matches!(self, Self::Bias | Self::BiasGelu)
    }
}

/// Activation-row bucket used as a tuning key: the next power of two at or
/// above `m` (at least 1).
pub fn m_bucket(m: usize) -> usize {
    m.max(1).next_power_of_two()
}

/// Report key of a tuning outcome: activation-row bucket, weight shape, epilogue and dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TuneKey {
    pub m_bucket: usize,
    pub n: usize,
    pub k: usize,
    pub epilogue: Epilogue,
    pub dtype: GemmDtype,
}

impl TuneKey {
    /// Key for a problem with `m` activation rows.
    pub fn new(m: usize, n: usize, k: usize, epilogue: Epilogue, dtype: GemmDtype) -> Self {
        Self {
            m_bucket: m_bucket(m),
            n,
            k,
            epilogue,
            dtype,
        }
    }
}

/// Tuning-table key: the exact activation-row count, weight shape, epilogue and dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShapeKey {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub epilogue: Epilogue,
    pub dtype: GemmDtype,
}

impl ShapeKey {
    /// The report key of the shape's row bucket.
    pub fn tune_key(&self) -> TuneKey {
        TuneKey::new(self.m, self.n, self.k, self.epilogue, self.dtype)
    }
}

/// Identity of one cuBLASLt algorithm as it is executed: its configuration attributes
/// (`cublasLtMatmulAlgoConfigAttributes_t`), the workspace bytes the heuristic says it needs,
/// and the opaque `cublasLtMatmulAlgo_t` words passed to `cublasLtMatmul`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AlgoIdentity {
    /// `CUBLASLT_ALGO_CONFIG_ID`.
    pub id: i32,
    /// `CUBLASLT_ALGO_CONFIG_TILE_ID`.
    pub tile: u32,
    /// `CUBLASLT_ALGO_CONFIG_STAGES_ID`.
    pub stages: u32,
    /// `CUBLASLT_ALGO_CONFIG_SPLITK_NUM`.
    pub splitk: u32,
    /// `CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME`.
    pub reduction: u32,
    /// `CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING`.
    pub swizzle: u32,
    /// `CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION`.
    pub custom: u32,
    /// `CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID`.
    pub inner_shape: u16,
    /// `CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID`.
    pub cluster_shape: u16,
    /// Workspace bytes the heuristic result requires.
    pub workspace_bytes: usize,
    /// The opaque algorithm descriptor words.
    pub opaque: [u64; 8],
}

impl AlgoIdentity {
    /// Colon-separated text form: the nine configuration attributes, the workspace bytes, then
    /// the eight opaque words in hex.
    pub fn encode(&self) -> String {
        let mut s = format!(
            "{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
            self.id,
            self.tile,
            self.stages,
            self.splitk,
            self.reduction,
            self.swizzle,
            self.custom,
            self.inner_shape,
            self.cluster_shape,
            self.workspace_bytes
        );
        for w in self.opaque {
            let _ = write!(s, ":{w:016x}");
        }
        s
    }

    /// Parses text produced by [`AlgoIdentity::encode`].
    pub fn decode(text: &str) -> Option<Self> {
        let parts: Vec<&str> = text.split(':').collect();
        if parts.len() != 18 {
            return None;
        }
        let mut opaque = [0u64; 8];
        for (w, p) in opaque.iter_mut().zip(&parts[10..]) {
            *w = u64::from_str_radix(p, 16).ok()?;
        }
        Some(Self {
            id: parts[0].parse().ok()?,
            tile: parts[1].parse().ok()?,
            stages: parts[2].parse().ok()?,
            splitk: parts[3].parse().ok()?,
            reduction: parts[4].parse().ok()?,
            swizzle: parts[5].parse().ok()?,
            custom: parts[6].parse().ok()?,
            inner_shape: parts[7].parse().ok()?,
            cluster_shape: parts[8].parse().ok()?,
            workspace_bytes: parts[9].parse().ok()?,
            opaque,
        })
    }

    /// Short human-readable form of the configuration attributes.
    pub fn summary(&self) -> String {
        format!(
            "algo {} tile {} stages {} splitk {} red {} swz {} opt {} inner {} cluster {} ws {}",
            self.id,
            self.tile,
            self.stages,
            self.splitk,
            self.reduction,
            self.swizzle,
            self.custom,
            self.inner_shape,
            self.cluster_shape,
            self.workspace_bytes
        )
    }
}

/// The exact conditions under which a tuned choice was admitted: the exact activation-row
/// count measured, the workspace length passed to `cublasLtMatmul`, the cuBLASLt library
/// version, the baseline (first heuristic candidate) the choice was compared against, and the
/// chosen algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlgoBinding {
    pub m: usize,
    pub workspace_len: usize,
    pub library_version: usize,
    pub baseline: AlgoIdentity,
    pub chosen: AlgoIdentity,
}

/// Chosen heuristic candidate for a key.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TuneEntry {
    /// Position of the chosen candidate in the heuristic list measured at admission; reported
    /// only, never used to select an algorithm.
    pub algo_index: usize,
    /// Measured time per call in microseconds.
    pub micros: f64,
    /// Admission conditions and algorithm identities; an entry without one selects the
    /// baseline.
    pub binding: Option<AlgoBinding>,
}

/// The executed problem's side of a selection: its exact activation-row count, the workspace
/// length the engine passes, and the loaded cuBLASLt library version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecContext {
    pub m: usize,
    pub workspace_len: usize,
    pub library_version: usize,
}

/// Why a selection ran the baseline candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineReason {
    /// No tuning entry exists for the key.
    Untuned,
    /// The entry carries no algorithm binding.
    Unbound,
    /// The executed activation-row count differs from the admitted one.
    ShapeMismatch,
    /// The executed workspace length differs from the admitted one.
    WorkspaceMismatch,
    /// The loaded cuBLASLt version differs from the admitting one.
    LibraryMismatch,
    /// The executed problem's first candidate is not the admitted baseline.
    BaselineMismatch,
    /// The executed problem's candidates do not offer the chosen algorithm within the
    /// workspace.
    NotOffered,
}

/// Outcome of binding a tuning entry to the executed problem's heuristic candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    /// Run the first candidate.
    Baseline(BaselineReason),
    /// Run the candidate at this position, whose identity equals the admitted choice.
    Tuned(usize),
}

impl Selection {
    /// Position in the executed problem's candidate list.
    pub fn index(self) -> usize {
        match self {
            Self::Baseline(_) => 0,
            Self::Tuned(i) => i,
        }
    }
}

/// Selects the candidate of `candidates` (the executed problem's heuristic list, in heuristic
/// order) that `entry` admitted. The tuned candidate runs only when the executed row count,
/// workspace length and library version equal the admitted ones, the first candidate is the
/// admitted baseline, and a candidate with the chosen identity is present and fits the
/// workspace; every other case selects the baseline.
pub fn select_candidate(
    entry: Option<&TuneEntry>,
    ctx: &ExecContext,
    candidates: &[AlgoIdentity],
) -> Selection {
    use BaselineReason as R;
    let Some(entry) = entry else {
        return Selection::Baseline(R::Untuned);
    };
    let Some(b) = entry.binding.as_ref() else {
        return Selection::Baseline(R::Unbound);
    };
    if b.m != ctx.m {
        return Selection::Baseline(R::ShapeMismatch);
    }
    if b.workspace_len != ctx.workspace_len {
        return Selection::Baseline(R::WorkspaceMismatch);
    }
    if b.library_version != ctx.library_version {
        return Selection::Baseline(R::LibraryMismatch);
    }
    if candidates.first() != Some(&b.baseline) {
        return Selection::Baseline(R::BaselineMismatch);
    }
    match candidates
        .iter()
        .position(|c| *c == b.chosen && c.workspace_bytes <= ctx.workspace_len)
    {
        Some(i) => Selection::Tuned(i),
        None => Selection::Baseline(R::NotOffered),
    }
}

/// Failure reading or writing a tuning file.
#[derive(Debug)]
pub enum TuningError {
    Io(std::io::Error),
    /// The file is not a table written by this module; the message names the line.
    Malformed(String),
}

impl std::fmt::Display for TuningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "tuning file io: {e}"),
            Self::Malformed(m) => write!(f, "malformed tuning file: {m}"),
        }
    }
}

impl std::error::Error for TuningError {}

/// Table of tuned algorithm choices.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TuningTable {
    entries: BTreeMap<ShapeKey, TuneEntry>,
}

impl TuningTable {
    /// Entry for the exact shape `key`, if tuned.
    pub fn get(&self, key: &ShapeKey) -> Option<&TuneEntry> {
        self.entries.get(key)
    }

    /// Whether the exact shape `key` lacks an entry with an algorithm binding. Entries of
    /// other row counts, bound or not, do not count.
    pub fn needs_admission(&self, key: &ShapeKey) -> bool {
        self.get(key).is_none_or(|e| e.binding.is_none())
    }

    /// Number of tuned keys.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no key is tuned.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterates entries in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&ShapeKey, &TuneEntry)> {
        self.entries.iter()
    }

    /// Records the choice for the exact shape `key`, replacing only an entry of that same
    /// shape.
    pub fn insert(&mut self, key: ShapeKey, entry: TuneEntry) {
        self.entries.insert(key, entry);
    }

    /// Serialises the table as TOML.
    pub fn to_toml(&self) -> String {
        let mut s = String::from("# scriptor GEMM tuning table\n");
        for (k, e) in &self.entries {
            let _ = writeln!(s, "\n[[entry]]");
            let _ = writeln!(s, "dtype = \"{}\"", k.dtype.name());
            let _ = writeln!(s, "epilogue = \"{}\"", k.epilogue.name());
            let _ = writeln!(s, "m = {}", k.m);
            let _ = writeln!(s, "n = {}", k.n);
            let _ = writeln!(s, "k = {}", k.k);
            let _ = writeln!(s, "algo_index = {}", e.algo_index);
            let _ = writeln!(s, "micros = {:?}", e.micros);
            if let Some(b) = &e.binding {
                let _ = writeln!(s, "bound_m = {}", b.m);
                let _ = writeln!(s, "workspace_len = {}", b.workspace_len);
                let _ = writeln!(s, "library_version = {}", b.library_version);
                let _ = writeln!(s, "baseline = \"{}\"", b.baseline.encode());
                let _ = writeln!(s, "chosen = \"{}\"", b.chosen.encode());
            }
        }
        s
    }

    /// Parses text produced by [`TuningTable::to_toml`]. An entry without an exact `m` is
    /// keyed by its `bound_m`, else by its `m_bucket`.
    pub fn from_toml(text: &str) -> Result<Self, TuningError> {
        let mut table = Self::default();
        for fields in sections(text, "[[entry]]")? {
            table.absorb(&fields)?;
        }
        Ok(table)
    }

    fn absorb(&mut self, f: &BTreeMap<String, String>) -> Result<(), TuningError> {
        let bad = |what: &str| TuningError::Malformed(format!("entry missing or invalid {what}"));
        let num = |name: &str| -> Result<usize, TuningError> {
            f.get(name)
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| bad(name))
        };
        let dtype = f
            .get("dtype")
            .and_then(|v| GemmDtype::from_name(v.trim_matches('"')))
            .ok_or_else(|| bad("dtype"))?;
        let epilogue = f
            .get("epilogue")
            .and_then(|v| Epilogue::from_name(v.trim_matches('"')))
            .ok_or_else(|| bad("epilogue"))?;
        let micros: f64 = f
            .get("micros")
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| bad("micros"))?;
        const BINDING: [&str; 5] = [
            "bound_m",
            "workspace_len",
            "library_version",
            "baseline",
            "chosen",
        ];
        let binding = if BINDING.iter().all(|name| !f.contains_key(*name)) {
            None
        } else {
            let id = |name: &str| {
                f.get(name)
                    .and_then(|v| AlgoIdentity::decode(v.trim_matches('"')))
                    .ok_or_else(|| bad(name))
            };
            Some(AlgoBinding {
                m: num("bound_m")?,
                workspace_len: num("workspace_len")?,
                library_version: num("library_version")?,
                baseline: id("baseline")?,
                chosen: id("chosen")?,
            })
        };
        let m = num("m")
            .or_else(|_| num("bound_m"))
            .or_else(|_| num("m_bucket"))?;
        self.insert(
            ShapeKey {
                m,
                n: num("n")?,
                k: num("k")?,
                epilogue,
                dtype,
            },
            TuneEntry {
                algo_index: num("algo_index")?,
                micros,
                binding,
            },
        );
        Ok(())
    }

    /// Reads a tuning file; a missing file yields an empty table.
    pub fn load(path: &Path) -> Result<Self, TuningError> {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::from_toml(&text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(TuningError::Io(e)),
        }
    }

    /// Writes the table to `path`, creating parent directories.
    pub fn write_file(&self, path: &Path) -> Result<(), TuningError> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(TuningError::Io)?;
        }
        std::fs::write(path, self.to_toml()).map_err(TuningError::Io)
    }
}

/// The `key = value` maps of every `header` section of `text`, in order. Blank lines and `#`
/// comments are skipped; any other line outside a section is malformed.
fn sections(text: &str, header: &str) -> Result<Vec<BTreeMap<String, String>>, TuningError> {
    let mut out = Vec::new();
    let mut cur: Option<BTreeMap<String, String>> = None;
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == header {
            out.extend(cur.take());
            cur = Some(BTreeMap::new());
        } else if let (Some(map), Some((k, v))) = (cur.as_mut(), line.split_once('=')) {
            map.insert(k.trim().to_owned(), v.trim().to_owned());
        } else {
            return Err(TuningError::Malformed(format!("line {}: {line}", i + 1)));
        }
    }
    out.extend(cur);
    Ok(out)
}

/// How one dispatched GEMM was served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmRoute {
    /// The dedicated skinny kernel.
    Skinny,
    /// cuBLASLt with the given selection.
    CuBlasLt(Selection),
}

impl GemmRoute {
    /// Stable name: `skinny`, `tuned:<index>` or `baseline:<reason>`.
    pub fn name(self) -> String {
        match self {
            Self::Skinny => "skinny".to_owned(),
            Self::CuBlasLt(Selection::Tuned(i)) => format!("tuned:{i}"),
            Self::CuBlasLt(Selection::Baseline(r)) => format!("baseline:{}", r.name()),
        }
    }
}

impl BaselineReason {
    /// Stable lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Untuned => "untuned",
            Self::Unbound => "unbound",
            Self::ShapeMismatch => "shape_mismatch",
            Self::WorkspaceMismatch => "workspace_mismatch",
            Self::LibraryMismatch => "library_mismatch",
            Self::BaselineMismatch => "baseline_mismatch",
            Self::NotOffered => "not_offered",
        }
    }
}

/// Dispatch statistics of one shape in one phase.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShapeStat {
    /// Dispatches recorded outside graph capture; each one executed once.
    pub calls: u64,
    /// Dispatches recorded while the stream was capturing a graph; each one executes once per
    /// replay of that graph, and the replay count is not recorded here.
    pub captured_calls: u64,
    /// Dispatches whose GPU time was measured.
    pub timed_calls: u64,
    /// Sum of the measured GPU times in microseconds. Each time is measured with the stream
    /// synchronised around that one dispatch, so it is a serial per-GEMM diagnostic and not
    /// a share of end-to-end time.
    pub gpu_micros: f64,
    /// Dispatches (eager or captured) that ran a tuned candidate other than the first one.
    pub non_baseline_calls: u64,
    /// Route of the most recent dispatch.
    pub route: String,
}

/// Dispatched GEMM shapes per caller-named phase (for example `vision`, `prefill`, `decode`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShapeInventory {
    entries: BTreeMap<(String, ShapeKey), ShapeStat>,
}

impl ShapeInventory {
    /// Counts one dispatch of `key` in `phase`, with its GPU time when measured; `captured`
    /// marks a dispatch recorded into a graph rather than executed.
    pub fn record(
        &mut self,
        phase: &str,
        key: ShapeKey,
        route: GemmRoute,
        micros: Option<f64>,
        captured: bool,
    ) {
        let stat = self.entries.entry((phase.to_owned(), key)).or_default();
        if captured {
            stat.captured_calls += 1;
        } else {
            stat.calls += 1;
        }
        if matches!(route, GemmRoute::CuBlasLt(Selection::Tuned(i)) if i > 0) {
            stat.non_baseline_calls += 1;
        }
        if let Some(us) = micros {
            stat.timed_calls += 1;
            stat.gpu_micros += us;
        }
        stat.route = route.name();
    }

    /// Adds every record of `other`.
    pub fn merge(&mut self, other: &ShapeInventory) {
        for ((phase, key), s) in &other.entries {
            let stat = self.entries.entry((phase.clone(), *key)).or_default();
            stat.calls += s.calls;
            stat.captured_calls += s.captured_calls;
            stat.non_baseline_calls += s.non_baseline_calls;
            stat.timed_calls += s.timed_calls;
            stat.gpu_micros += s.gpu_micros;
            stat.route.clone_from(&s.route);
        }
    }

    /// Number of distinct `(phase, shape)` records.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing was recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Records ordered by total measured GPU time, then eager dispatch count, both
    /// descending, then by phase and shape. Captured dispatches carry no execution count and
    /// do not raise a record's rank.
    pub fn ranked(&self) -> Vec<(&str, ShapeKey, &ShapeStat)> {
        let mut v: Vec<_> = self
            .entries
            .iter()
            .map(|((p, k), s)| (p.as_str(), *k, s))
            .collect();
        v.sort_by(|a, b| {
            b.2.gpu_micros
                .total_cmp(&a.2.gpu_micros)
                .then(b.2.calls.cmp(&a.2.calls))
                .then(a.0.cmp(b.0))
                .then(a.1.cmp(&b.1))
        });
        v
    }

    /// The distinct shapes, in [`Self::ranked`] order of first appearance, that were served by
    /// cuBLASLt rather than the skinny kernel; at most `limit` of them.
    pub fn admission_shapes(&self, limit: usize) -> Vec<ShapeKey> {
        let mut out: Vec<ShapeKey> = Vec::new();
        for (_, key, stat) in self.ranked() {
            if out.len() == limit {
                break;
            }
            if stat.route != GemmRoute::Skinny.name() && !out.contains(&key) {
                out.push(key);
            }
        }
        out
    }

    /// Records whose dispatches ran a tuned candidate other than the first one.
    pub fn non_baseline(&self) -> Vec<(&str, ShapeKey, &ShapeStat)> {
        self.ranked()
            .into_iter()
            .filter(|(_, _, s)| s.non_baseline_calls > 0)
            .collect()
    }

    /// Markdown table of the ranked records, headed by how its columns were measured.
    pub fn report(&self) -> String {
        let mut s = String::from(
            "gpu us: serial per-GEMM diagnostic (stream synchronised around each timed dispatch), not end-to-end time. captured: dispatches recorded into a graph; replay frequency unavailable, not ranked.\n\n| phase | m | n | k | dtype | epilogue | calls | captured | non-baseline | timed | gpu us | us/call | route |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
        );
        for (phase, k, st) in self.ranked() {
            let per = if st.timed_calls > 0 {
                format!("{:.1}", st.gpu_micros / st.timed_calls as f64)
            } else {
                "-".to_owned()
            };
            let _ = writeln!(
                s,
                "| {phase} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.1} | {per} | {} |",
                k.m,
                k.n,
                k.k,
                k.dtype.name(),
                k.epilogue.name(),
                st.calls,
                st.captured_calls,
                st.non_baseline_calls,
                st.timed_calls,
                st.gpu_micros,
                st.route
            );
        }
        s
    }

    /// Serialises the records as TOML `[[shape]]` sections.
    pub fn to_toml(&self) -> String {
        let mut s = String::from(
            "# scriptor dispatched GEMM shapes\n# gpu_micros: serial per-GEMM diagnostic (stream synchronised around each timed dispatch), not end-to-end time\n# captured_calls: dispatches recorded into a graph; replay frequency unavailable\n",
        );
        for ((phase, k), st) in &self.entries {
            let _ = writeln!(s, "\n[[shape]]");
            let _ = writeln!(s, "phase = \"{phase}\"");
            let _ = writeln!(s, "dtype = \"{}\"", k.dtype.name());
            let _ = writeln!(s, "epilogue = \"{}\"", k.epilogue.name());
            let _ = writeln!(s, "m = {}", k.m);
            let _ = writeln!(s, "n = {}", k.n);
            let _ = writeln!(s, "k = {}", k.k);
            let _ = writeln!(s, "calls = {}", st.calls);
            let _ = writeln!(s, "captured_calls = {}", st.captured_calls);
            let _ = writeln!(s, "non_baseline_calls = {}", st.non_baseline_calls);
            let _ = writeln!(s, "timed_calls = {}", st.timed_calls);
            let _ = writeln!(s, "gpu_micros = {:?}", st.gpu_micros);
            let _ = writeln!(s, "route = \"{}\"", st.route);
        }
        s
    }

    /// Parses text produced by [`ShapeInventory::to_toml`].
    pub fn from_toml(text: &str) -> Result<Self, TuningError> {
        let mut inv = Self::default();
        for f in sections(text, "[[shape]]")? {
            let bad =
                |what: &str| TuningError::Malformed(format!("shape missing or invalid {what}"));
            let text = |name: &str| {
                f.get(name)
                    .map(|v| v.trim_matches('"').to_owned())
                    .ok_or_else(|| bad(name))
            };
            let num = |name: &str| -> Result<u64, TuningError> {
                f.get(name)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| bad(name))
            };
            let count = |name: &str| -> Result<u64, TuningError> {
                f.get(name)
                    .map_or(Ok(0), |v| v.parse().map_err(|_| bad(name)))
            };
            let key = ShapeKey {
                m: num("m")? as usize,
                n: num("n")? as usize,
                k: num("k")? as usize,
                epilogue: Epilogue::from_name(&text("epilogue")?).ok_or_else(|| bad("epilogue"))?,
                dtype: GemmDtype::from_name(&text("dtype")?).ok_or_else(|| bad("dtype"))?,
            };
            let gpu_micros: f64 = f
                .get("gpu_micros")
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| bad("gpu_micros"))?;
            inv.entries.insert(
                (text("phase")?, key),
                ShapeStat {
                    calls: num("calls")?,
                    captured_calls: count("captured_calls")?,
                    timed_calls: num("timed_calls")?,
                    gpu_micros,
                    non_baseline_calls: count("non_baseline_calls")?,
                    route: text("route")?,
                },
            );
        }
        Ok(inv)
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect::<String>()
        .to_lowercase()
}

/// Tuning file location inside an explicit cache directory, named for the GPU
/// and CUDA version string: `<dir>/gemm-tuning-<gpu>-<cuda>.toml`.
pub fn cache_file(dir: &Path, gpu: &str, cuda: &str) -> PathBuf {
    dir.join(format!(
        "gemm-tuning-{}-{}.toml",
        sanitize(gpu),
        sanitize(cuda)
    ))
}

/// One weight-matrix shape of the GLM-OCR model: output features `n`,
/// reduction length `k`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmShape {
    /// Where the shape occurs in the model.
    pub role: &'static str,
    pub n: usize,
    pub k: usize,
    /// Whether the shape belongs to the vision tower (`true`) or the decoder.
    pub vision: bool,
}

impl GemmShape {
    /// Whether the fp8 path accepts the shape: cuBLASLt needs 16-byte aligned
    /// leading dimensions, so `k` must be a multiple of 16 and `n` of 8.
    pub fn fp8_compatible(&self) -> bool {
        self.k.is_multiple_of(16) && self.n.is_multiple_of(8)
    }
}

/// Activation row counts the GEMM probe and the tuner cover.
pub const PROBE_M: [usize; 5] = [32, 128, 1024, 16384, 65536];

/// The weight shapes of the ViT and decoder GEMMs.
pub fn model_shapes() -> Vec<GemmShape> {
    let v = |role, n, k| GemmShape {
        role,
        n,
        k,
        vision: true,
    };
    let d = |role, n, k| GemmShape {
        role,
        n,
        k,
        vision: false,
    };
    vec![
        v("vit.patch_embed", 1024, 1176),
        v("vit.qkv", 3072, 1024),
        v("vit.proj", 1024, 1024),
        v("vit.mlp_gate_up", 8192, 1024),
        v("vit.mlp_down", 1024, 4096),
        v("vit.downsample", 1536, 4096),
        v("vit.merger_proj", 1536, 1536),
        v("vit.merger_gate_up", 9216, 1536),
        v("vit.merger_down", 1536, 4608),
        d("text.qkv", 4096, 1536),
        d("text.o_proj", 1536, 2048),
        d("text.gate_up", 9216, 1536),
        d("text.down", 1536, 4608),
        d("text.lm_head", 59392, 1536),
    ]
}

/// Floating-point operations of one `[m, k] x [n, k]^T` GEMM.
pub fn gemm_flops(m: usize, n: usize, k: usize) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64
}

/// Achieved TFLOPS for one GEMM taking `micros` microseconds.
pub fn tflops(m: usize, n: usize, k: usize, micros: f64) -> f64 {
    gemm_flops(m, n, k) / (micros * 1e-6) / 1e12
}

/// Median of `values` (mean of the middle pair for an even count); `None`
/// when empty.
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    Some(if v.len() % 2 == 1 {
        v[mid]
    } else {
        (v[mid - 1] + v[mid]) / 2.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const WS: usize = 64 << 20;
    const LIB: usize = 130_400;

    /// A distinct synthetic identity per `seed`.
    fn ident(seed: u32) -> AlgoIdentity {
        AlgoIdentity {
            id: 20 + seed as i32,
            tile: seed,
            stages: 2 * seed,
            splitk: 1,
            reduction: 0,
            swizzle: seed % 2,
            custom: 0,
            inner_shape: 3,
            cluster_shape: 1,
            workspace_bytes: 1024 * seed as usize,
            opaque: [u64::from(seed); 8],
        }
    }

    fn binding(m: usize, baseline: AlgoIdentity, chosen: AlgoIdentity) -> AlgoBinding {
        AlgoBinding {
            m,
            workspace_len: WS,
            library_version: LIB,
            baseline,
            chosen,
        }
    }

    fn entry(algo_index: usize, b: Option<AlgoBinding>) -> TuneEntry {
        TuneEntry {
            algo_index,
            micros: 10.0,
            binding: b,
        }
    }

    fn shape(m: usize, n: usize, k: usize, epilogue: Epilogue, dtype: GemmDtype) -> ShapeKey {
        ShapeKey {
            m,
            n,
            k,
            epilogue,
            dtype,
        }
    }

    fn bf16(m: usize) -> ShapeKey {
        shape(m, 3072, 1024, Epilogue::Bias, GemmDtype::Bf16)
    }

    /// A table with bound entries at rows 1100 and 1900 (both in bucket 2048) choosing
    /// different algorithms over the same baseline.
    fn two_exact_entries() -> TuningTable {
        let mut t = TuningTable::default();
        t.insert(
            bf16(1100),
            entry(1, Some(binding(1100, ident(0), ident(1)))),
        );
        t.insert(
            bf16(1900),
            entry(2, Some(binding(1900, ident(0), ident(2)))),
        );
        t
    }

    fn ctx(m: usize) -> ExecContext {
        ExecContext {
            m,
            workspace_len: WS,
            library_version: LIB,
        }
    }

    #[test]
    fn selection_follows_identity_when_the_list_is_reordered() {
        let e = entry(1, Some(binding(1024, ident(0), ident(1))));
        let reordered = [ident(0), ident(2), ident(3), ident(1)];
        let sel = select_candidate(Some(&e), &ctx(1024), &reordered);
        assert_eq!(sel, Selection::Tuned(3));
        assert_eq!(sel.index(), 3);
        // Position 1 now holds a different algorithm; the ordinal must not be used.
        assert_ne!(reordered[e.algo_index], ident(1));
    }

    #[test]
    fn selection_rejects_a_reordered_baseline() {
        let e = entry(1, Some(binding(1024, ident(0), ident(1))));
        let swapped = [ident(1), ident(0), ident(2)];
        assert_eq!(
            select_candidate(Some(&e), &ctx(1024), &swapped),
            Selection::Baseline(BaselineReason::BaselineMismatch)
        );
    }

    #[test]
    fn selection_never_clamps_into_a_shorter_list() {
        let e = entry(5, Some(binding(1024, ident(0), ident(5))));
        let short = [ident(0), ident(1), ident(2)];
        let sel = select_candidate(Some(&e), &ctx(1024), &short);
        assert_eq!(sel, Selection::Baseline(BaselineReason::NotOffered));
        assert_eq!(sel.index(), 0);
    }

    #[test]
    fn selection_falls_back_on_a_mismatched_shape() {
        let e = entry(1, Some(binding(1024, ident(0), ident(1))));
        let list = [ident(0), ident(1)];
        for m in [1000, 1023, 1025, 2048] {
            assert_eq!(
                select_candidate(Some(&e), &ctx(m), &list),
                Selection::Baseline(BaselineReason::ShapeMismatch),
                "m {m}"
            );
        }
    }

    #[test]
    fn selection_falls_back_on_workspace_or_library_change() {
        let e = entry(1, Some(binding(1024, ident(0), ident(1))));
        let list = [ident(0), ident(1)];
        let ws = ExecContext {
            workspace_len: WS / 2,
            ..ctx(1024)
        };
        assert_eq!(
            select_candidate(Some(&e), &ws, &list),
            Selection::Baseline(BaselineReason::WorkspaceMismatch)
        );
        let lib = ExecContext {
            library_version: LIB + 1,
            ..ctx(1024)
        };
        assert_eq!(
            select_candidate(Some(&e), &lib, &list),
            Selection::Baseline(BaselineReason::LibraryMismatch)
        );
    }

    #[test]
    fn selection_requires_the_chosen_algorithm_to_fit_the_workspace() {
        let mut big = ident(1);
        big.workspace_bytes = WS + 1;
        let e = TuneEntry {
            binding: Some(AlgoBinding {
                chosen: big,
                ..binding(1024, ident(0), big)
            }),
            ..entry(1, None)
        };
        assert_eq!(
            select_candidate(Some(&e), &ctx(1024), &[ident(0), big]),
            Selection::Baseline(BaselineReason::NotOffered)
        );
    }

    #[test]
    fn selection_rejects_a_candidate_differing_in_one_attribute() {
        let e = entry(1, Some(binding(1024, ident(0), ident(1))));
        let mut near = ident(1);
        near.splitk = 2;
        assert_eq!(
            select_candidate(Some(&e), &ctx(1024), &[ident(0), near]),
            Selection::Baseline(BaselineReason::NotOffered)
        );
        let mut opaque = ident(1);
        opaque.opaque[7] ^= 1;
        assert_eq!(
            select_candidate(Some(&e), &ctx(1024), &[ident(0), opaque]),
            Selection::Baseline(BaselineReason::NotOffered)
        );
    }

    #[test]
    fn selection_keeps_a_valid_matching_choice() {
        let e = entry(2, Some(binding(1024, ident(0), ident(2))));
        let list = [ident(0), ident(1), ident(2), ident(3)];
        assert_eq!(
            select_candidate(Some(&e), &ctx(1024), &list),
            Selection::Tuned(2)
        );
        let same = entry(0, Some(binding(1024, ident(0), ident(0))));
        assert_eq!(
            select_candidate(Some(&same), &ctx(1024), &list),
            Selection::Tuned(0)
        );
    }

    #[test]
    fn selection_without_a_binding_is_the_baseline() {
        let list = [ident(0), ident(1)];
        assert_eq!(
            select_candidate(None, &ctx(1024), &list),
            Selection::Baseline(BaselineReason::Untuned)
        );
        assert_eq!(
            select_candidate(Some(&entry(1, None)), &ctx(1024), &list),
            Selection::Baseline(BaselineReason::Unbound)
        );
    }

    #[test]
    fn exact_rows_in_one_bucket_select_their_own_identity() {
        let t = two_exact_entries();
        assert_eq!(t.len(), 2);
        let list = [ident(0), ident(1), ident(2)];
        assert_eq!(
            select_candidate(t.get(&bf16(1100)), &ctx(1100), &list),
            Selection::Tuned(1)
        );
        assert_eq!(
            select_candidate(t.get(&bf16(1900)), &ctx(1900), &list),
            Selection::Tuned(2)
        );
    }

    #[test]
    fn unadmitted_neighbour_in_the_bucket_falls_back() {
        let t = two_exact_entries();
        let list = [ident(0), ident(1), ident(2)];
        for m in [1099, 1101, 1500, 2048] {
            assert_eq!(t.get(&bf16(m)), None, "m {m}");
            assert_eq!(
                select_candidate(t.get(&bf16(m)), &ctx(m), &list),
                Selection::Baseline(BaselineReason::Untuned),
                "m {m}"
            );
        }
    }

    #[test]
    fn reload_preserves_every_exact_row_entry() {
        let t = two_exact_entries();
        let back = TuningTable::from_toml(&t.to_toml()).unwrap();
        assert_eq!(back, t);
        assert_eq!(
            back.get(&bf16(1100)).unwrap().binding.unwrap().chosen,
            ident(1)
        );
        assert_eq!(
            back.get(&bf16(1900)).unwrap().binding.unwrap().chosen,
            ident(2)
        );
    }

    #[test]
    fn a_bound_bucket_entry_does_not_suppress_exact_admission() {
        let mut t = TuningTable::default();
        t.insert(
            bf16(2048),
            entry(1, Some(binding(2048, ident(0), ident(1)))),
        );
        assert!(!t.needs_admission(&bf16(2048)));
        assert!(t.needs_admission(&bf16(1100)));
        assert!(t.needs_admission(&bf16(1900)));
        t.insert(bf16(1024), entry(1, None));
        assert!(
            t.needs_admission(&bf16(1024)),
            "an unbound entry is re-admitted"
        );
    }

    #[test]
    fn legacy_bucket_entries_key_by_bound_or_bucket_rows() {
        let base = "[[entry]]\ndtype = \"bf16\"\nepilogue = \"bias\"\nm_bucket = 2048\nn = 3072\nk = 1024\nalgo_index = 3\nmicros = 2.0\n";
        let t = TuningTable::from_toml(base).unwrap();
        let e = t.get(&bf16(2048)).unwrap();
        assert_eq!(e.binding, None);
        assert_eq!(t.get(&bf16(1100)), None);
        assert_eq!(
            select_candidate(Some(e), &ctx(2048), &[ident(0), ident(1)]),
            Selection::Baseline(BaselineReason::Unbound)
        );
        let bound = format!(
            "{base}bound_m = 2048\nworkspace_len = {WS}\nlibrary_version = {LIB}\nbaseline = \"{}\"\nchosen = \"{}\"\n",
            ident(0).encode(),
            ident(1).encode()
        );
        let t = TuningTable::from_toml(&bound).unwrap();
        assert_eq!(
            select_candidate(t.get(&bf16(2048)), &ctx(2048), &[ident(0), ident(1)]),
            Selection::Tuned(1)
        );
        assert_eq!(t.get(&bf16(1100)), None);
    }

    #[test]
    fn inventory_counts_ranks_and_round_trips() {
        let mut inv = ShapeInventory::default();
        let tuned = GemmRoute::CuBlasLt(Selection::Tuned(1));
        let base = GemmRoute::CuBlasLt(Selection::Baseline(BaselineReason::Untuned));
        inv.record("vision", bf16(1100), base, Some(30.0), false);
        inv.record("vision", bf16(1100), tuned, Some(10.0), false);
        inv.record("prefill", bf16(700), base, Some(100.0), false);
        inv.record("decode", bf16(8), GemmRoute::Skinny, None, false);
        inv.record("decode", bf16(8), GemmRoute::Skinny, None, false);
        inv.record("vision", bf16(700), base, None, false);
        assert_eq!(inv.len(), 4);
        let ranked = inv.ranked();
        assert_eq!((ranked[0].0, ranked[0].1), ("prefill", bf16(700)));
        assert_eq!((ranked[1].0, ranked[1].1), ("vision", bf16(1100)));
        assert_eq!(ranked[1].2.calls, 2);
        assert_eq!(ranked[1].2.route, "tuned:1");
        assert_eq!(ranked[2].1, bf16(8));
        assert_eq!(inv.admission_shapes(10), vec![bf16(700), bf16(1100)]);
        assert_eq!(inv.admission_shapes(1), vec![bf16(700)]);
        let back = ShapeInventory::from_toml(&inv.to_toml()).unwrap();
        assert_eq!(back, inv);
        assert!(inv.report().contains("| vision | 1100 |"));
        assert_eq!(ranked[1].2.non_baseline_calls, 1);
        let hot: Vec<_> = inv.non_baseline().iter().map(|r| (r.0, r.1)).collect();
        assert_eq!(hot, vec![("vision", bf16(1100))]);
        let mut twice = inv.clone();
        twice.merge(&inv);
        assert_eq!(twice.ranked()[0].2.calls, 2);
        assert_eq!(twice.non_baseline()[0].2.non_baseline_calls, 2);
    }

    #[test]
    fn captured_dispatches_are_counted_apart_and_never_ranked() {
        let mut inv = ShapeInventory::default();
        let base = GemmRoute::CuBlasLt(Selection::Baseline(BaselineReason::Untuned));
        let tuned0 = GemmRoute::CuBlasLt(Selection::Tuned(0));
        for _ in 0..50 {
            inv.record("decode", bf16(16), base, None, true);
        }
        inv.record("prefill", bf16(700), tuned0, None, false);
        let ranked = inv.ranked();
        assert_eq!((ranked[0].0, ranked[0].1), ("prefill", bf16(700)));
        assert_eq!((ranked[1].2.calls, ranked[1].2.captured_calls), (0, 50));
        assert!(inv.non_baseline().is_empty(), "tuned:0 is the baseline");
        assert_eq!(inv.admission_shapes(10), vec![bf16(700), bf16(16)]);
        let back = ShapeInventory::from_toml(&inv.to_toml()).unwrap();
        assert_eq!(back, inv);
        let report = inv.report();
        assert!(report.contains("replay frequency unavailable"));
        assert!(report.contains("serial per-GEMM diagnostic"));
    }

    #[test]
    fn inventory_without_capture_counts_parses() {
        let text = "[[shape]]\nphase = \"vision\"\ndtype = \"bf16\"\nepilogue = \"bias\"\nm = 1100\nn = 3072\nk = 1024\ncalls = 3\ntimed_calls = 0\ngpu_micros = 0.0\nroute = \"skinny\"\n";
        let inv = ShapeInventory::from_toml(text).unwrap();
        let r = inv.ranked();
        assert_eq!(
            (
                r[0].2.calls,
                r[0].2.captured_calls,
                r[0].2.non_baseline_calls
            ),
            (3, 0, 0)
        );
        assert!(
            ShapeInventory::from_toml(
                &text.replace("timed_calls = 0", "timed_calls = 0\ncaptured_calls = x")
            )
            .is_err()
        );
    }

    #[test]
    fn identity_text_round_trips() {
        let mut id = ident(7);
        id.id = -3;
        id.opaque = [u64::MAX, 0, 1, 2, 3, 4, 5, 0xdead_beef];
        assert_eq!(AlgoIdentity::decode(&id.encode()), Some(id));
        assert_eq!(AlgoIdentity::decode("1:2:3"), None);
        assert_eq!(
            AlgoIdentity::decode(&id.encode().replace("deadbeef", "zeadbeef")),
            None
        );
    }

    #[test]
    fn partial_binding_in_a_file_is_rejected() {
        let text = "[[entry]]\ndtype = \"bf16\"\nepilogue = \"none\"\nm_bucket = 64\nn = 8\nk = 8\nalgo_index = 1\nmicros = 1.0\nbound_m = 64\n";
        assert!(TuningTable::from_toml(text).is_err());
        let unbound = text.replace("bound_m = 64\n", "");
        let t = TuningTable::from_toml(&unbound).unwrap();
        let key = shape(64, 8, 8, Epilogue::None, GemmDtype::Bf16);
        assert_eq!(t.get(&key).unwrap().binding, None);
    }

    #[test]
    fn buckets_are_powers_of_two() {
        assert_eq!(m_bucket(0), 1);
        assert_eq!(m_bucket(32), 32);
        assert_eq!(m_bucket(33), 64);
        assert_eq!(m_bucket(65536), 65536);
    }

    #[test]
    fn toml_round_trip() {
        let mut t = TuningTable::default();
        t.insert(
            shape(100, 3072, 1024, Epilogue::Bias, GemmDtype::Bf16),
            TuneEntry {
                algo_index: 3,
                micros: 12.5,
                binding: Some(binding(100, ident(1), ident(2))),
            },
        );
        t.insert(
            shape(65536, 9216, 1536, Epilogue::None, GemmDtype::Fp8E4m3),
            TuneEntry {
                algo_index: 0,
                micros: 8000.25,
                binding: None,
            },
        );
        let back = TuningTable::from_toml(&t.to_toml()).unwrap();
        assert_eq!(back, t);
        let key = shape(100, 3072, 1024, Epilogue::Bias, GemmDtype::Bf16);
        assert_eq!(back.get(&key).unwrap().algo_index, 3);
    }

    #[test]
    fn malformed_text_is_rejected() {
        assert!(TuningTable::from_toml("garbage").is_err());
        assert!(TuningTable::from_toml("[[entry]]\ndtype = \"bf16\"\n").is_err());
        assert!(
            TuningTable::from_toml("[[entry]]\ndtype = \"bf17\"\nepilogue = \"none\"\n").is_err()
        );
    }

    #[test]
    fn load_missing_file_is_empty_and_write_creates_dirs() {
        let dir = std::env::temp_dir().join(format!("scriptor-heur-{}", std::process::id()));
        let path = cache_file(&dir, "NVIDIA GB10", "13.0");
        assert!(path.ends_with("gemm-tuning-nvidia-gb10-13.0.toml"));
        assert!(TuningTable::load(&path).unwrap().is_empty());
        let mut t = TuningTable::default();
        t.insert(
            shape(1, 8, 16, Epilogue::Gelu, GemmDtype::Bf16),
            TuneEntry {
                algo_index: 1,
                micros: 1.0,
                binding: None,
            },
        );
        t.write_file(&path).unwrap();
        assert_eq!(TuningTable::load(&path).unwrap(), t);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn names_round_trip() {
        for e in [
            Epilogue::None,
            Epilogue::Bias,
            Epilogue::Gelu,
            Epilogue::BiasGelu,
        ] {
            assert_eq!(Epilogue::from_name(e.name()), Some(e));
        }
        for d in [GemmDtype::Bf16, GemmDtype::Fp8E4m3] {
            assert_eq!(GemmDtype::from_name(d.name()), Some(d));
        }
    }

    #[test]
    fn shapes_cover_model_and_fp8_alignment() {
        let shapes = model_shapes();
        assert!(shapes.iter().any(|s| s.n == 59392 && s.k == 1536));
        let patch = shapes.iter().find(|s| s.role == "vit.patch_embed").unwrap();
        assert!(!patch.fp8_compatible());
        assert!(
            shapes
                .iter()
                .filter(|s| s.role != "vit.patch_embed")
                .all(|s| s.fp8_compatible())
        );
    }

    #[test]
    fn throughput_math() {
        assert_eq!(gemm_flops(2, 3, 4), 48.0);
        let t = tflops(1024, 1024, 1024, 1000.0);
        assert!((t - 2.147483648).abs() < 1e-9, "{t}");
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(median(&[]), None);
    }
}
