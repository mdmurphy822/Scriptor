//! `scriptor bench gemm` and `scriptor tune`: sustained GEMM
//! throughput over the ViT and decoder shapes, and the GEMM algorithm tuner.
//!
//! The argument types and report rendering build everywhere; the device
//! runners need the `cuda` feature.

use std::fmt::Write as _;
use std::path::PathBuf;

use clap::{Args, ValueEnum};

/// Operand format selected on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DtypeArg {
    Bf16,
    Fp8,
}

/// Arguments of `bench gemm`.
#[derive(Args, Debug)]
pub struct GemmBenchArgs {
    /// Operand format to measure.
    #[arg(long, value_enum, default_value = "bf16")]
    pub dtype: DtypeArg,
    /// Writes the JSON report to this file.
    #[arg(long)]
    pub report: Option<PathBuf>,
    /// Reads the tuning table from this directory (never written by `bench`).
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
    /// Untimed launches per shape.
    #[arg(long, default_value_t = 50)]
    pub warmup: usize,
    /// Timed launches per shape; the median is reported.
    #[arg(long, default_value_t = 200)]
    pub iters: usize,
    /// Activation row counts to measure (comma separated).
    #[arg(long, value_delimiter = ',', default_values_t = scriptor_kernels::heuristics::PROBE_M)]
    pub m: Vec<usize>,
}

/// Arguments of `tune`.
#[derive(Args, Debug)]
pub struct TuneArgs {
    /// Directory the tuning table is written to (the only writer of the cache).
    #[arg(long)]
    pub cache_dir: PathBuf,
    /// Also tune the fp8 path.
    #[arg(long)]
    pub fp8: bool,
    /// Timed launches per candidate.
    #[arg(long, default_value_t = 20)]
    pub iters: usize,
    /// Activation row counts to tune (comma separated).
    #[arg(long, value_delimiter = ',', default_values_t = scriptor_kernels::heuristics::PROBE_M)]
    pub m: Vec<usize>,
    /// Tunes exactly the shapes of this dispatched-shape inventory instead of the model shapes
    /// at `--m`, ordered by total measured GPU time, then by executed dispatch count, so an
    /// untimed inventory (`serve --gemm-shape-log`) is ordered by how often each shape ran.
    #[arg(long)]
    pub shapes: Option<PathBuf>,
    /// Most shapes taken from `--shapes`.
    #[arg(long, default_value_t = 64)]
    pub max_shapes: usize,
}

/// One measured (or skipped) GEMM shape.
#[derive(Debug, Clone, PartialEq)]
pub struct GemmRow {
    pub role: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub dtype: String,
    /// Median launch time in microseconds; `None` when skipped.
    pub median_us: Option<f64>,
    pub tflops: Option<f64>,
    /// Weight bytes read per second at the median time, GB/s.
    pub weight_gbps: Option<f64>,
    /// Why the shape was not measured.
    pub skipped: Option<String>,
}

/// Run-level facts recorded beside the rows.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReportMeta {
    pub device: String,
    pub dtype: String,
    pub warmup: usize,
    pub iters: usize,
    pub fp8_probe: Option<String>,
}

impl GemmRow {
    /// A measured row from the median launch time.
    pub fn measured(
        role: &str,
        (m, n, k): (usize, usize, usize),
        dtype: &str,
        elem_bytes: usize,
        median_us: f64,
    ) -> Self {
        Self {
            role: role.to_owned(),
            m,
            n,
            k,
            dtype: dtype.to_owned(),
            median_us: Some(median_us),
            tflops: Some(scriptor_kernels::heuristics::tflops(m, n, k, median_us)),
            weight_gbps: Some((n * k * elem_bytes) as f64 / (median_us * 1e-6) / 1e9),
            skipped: None,
        }
    }

    /// A row for a shape that was not measured.
    pub fn skipped(
        role: &str,
        (m, n, k): (usize, usize, usize),
        dtype: &str,
        reason: &str,
    ) -> Self {
        Self {
            role: role.to_owned(),
            m,
            n,
            k,
            dtype: dtype.to_owned(),
            median_us: None,
            tflops: None,
            weight_gbps: None,
            skipped: Some(reason.to_owned()),
        }
    }
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_number(v: Option<f64>) -> String {
    v.map_or_else(|| "null".to_owned(), |x| format!("{x:.4}"))
}

/// Renders the report as JSON.
pub fn render_json(meta: &ReportMeta, rows: &[GemmRow]) -> String {
    let mut s = String::from("{\n");
    let _ = writeln!(s, "  \"device\": {},", json_string(&meta.device));
    let _ = writeln!(s, "  \"dtype\": {},", json_string(&meta.dtype));
    let _ = writeln!(s, "  \"warmup\": {},", meta.warmup);
    let _ = writeln!(s, "  \"iters\": {},", meta.iters);
    let probe = meta.fp8_probe.as_deref().map(json_string);
    let _ = writeln!(
        s,
        "  \"fp8_probe\": {},",
        probe.unwrap_or_else(|| "null".to_owned())
    );
    s.push_str("  \"rows\": [\n");
    for (i, r) in rows.iter().enumerate() {
        let skipped = r.skipped.as_deref().map(json_string);
        let _ = write!(
            s,
            "    {{\"role\": {}, \"m\": {}, \"n\": {}, \"k\": {}, \"dtype\": {}, \"median_us\": {}, \"tflops\": {}, \"weight_gbps\": {}, \"skipped\": {}}}",
            json_string(&r.role),
            r.m,
            r.n,
            r.k,
            json_string(&r.dtype),
            json_number(r.median_us),
            json_number(r.tflops),
            json_number(r.weight_gbps),
            skipped.unwrap_or_else(|| "null".to_owned()),
        );
        s.push_str(if i + 1 < rows.len() { ",\n" } else { "\n" });
    }
    s.push_str("  ]\n}\n");
    s
}

/// Renders the rows as a Markdown table.
pub fn render_markdown(rows: &[GemmRow]) -> String {
    let mut s = String::from(
        "| Role | M | N | K | dtype | median us | TFLOPS | weight GB/s |\n|---|---|---|---|---|---|---|---|\n",
    );
    for r in rows {
        match (&r.skipped, r.median_us, r.tflops, r.weight_gbps) {
            (None, Some(us), Some(tf), Some(gb)) => {
                let _ = writeln!(
                    s,
                    "| {} | {} | {} | {} | {} | {us:.1} | {tf:.1} | {gb:.0} |",
                    r.role, r.m, r.n, r.k, r.dtype
                );
            }
            (reason, ..) => {
                let _ = writeln!(
                    s,
                    "| {} | {} | {} | {} | {} | skipped: {} | | |",
                    r.role,
                    r.m,
                    r.n,
                    r.k,
                    r.dtype,
                    reason.as_deref().unwrap_or("unmeasured")
                );
            }
        }
    }
    s
}

/// Highest measured TFLOPS among rows with at least `min_m` activation rows;
/// `None` when no row qualifies.
pub fn sustained_tflops(rows: &[GemmRow], min_m: usize) -> Option<f64> {
    rows.iter()
        .filter(|r| r.m >= min_m)
        .filter_map(|r| r.tflops)
        .max_by(f64::total_cmp)
}

#[cfg(feature = "cuda")]
pub use device::{run_bench, run_tune};

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use scriptor_cuda::Device;
    use scriptor_kernels::cublaslt::{GemmEngine, GemmOptions, GemmProblem, SyntheticOperands};
    use scriptor_kernels::heuristics::{
        Epilogue, GemmDtype, ShapeInventory, TuningTable, cache_file, median, model_shapes,
    };

    const WORKSPACE_BYTES: usize = 64 << 20;

    fn cuda_label() -> String {
        // SAFETY: plain version query with no arguments.
        let v = unsafe { cudarc::cublaslt::sys::cublasLtGetVersion() };
        format!("lt{v}")
    }

    struct Session {
        dev: Device,
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
        engine: GemmEngine,
        _arena: scriptor_cuda::DeviceArena,
    }

    fn open(fp8: bool, cache_dir: Option<&std::path::Path>) -> Result<Session, String> {
        let dev = Device::open().map_err(|e| e.to_string())?;
        let tuning = match cache_dir {
            Some(d) => TuningTable::load(&cache_file(d, &dev.info().name, &cuda_label()))
                .map_err(|e| e.to_string())?,
            None => TuningTable::default(),
        };
        let stream = dev.new_stream().map_err(|e| e.to_string())?;
        let mut arena = dev
            .arena(WORKSPACE_BYTES * 2, &stream)
            .map_err(|e| e.to_string())?;
        let engine = GemmEngine::new(
            dev.context(),
            &stream,
            &mut arena,
            tuning,
            GemmOptions {
                fp8,
                workspace_bytes: WORKSPACE_BYTES,
            },
        )
        .map_err(|e| e.to_string())?;
        Ok(Session {
            dev,
            stream,
            engine,
            _arena: arena,
        })
    }

    /// Runs `bench gemm`; returns the process exit code.
    pub fn run_bench(args: GemmBenchArgs) -> Result<i32, String> {
        let fp8 = args.dtype == DtypeArg::Fp8;
        let dtype = if fp8 {
            GemmDtype::Fp8E4m3
        } else {
            GemmDtype::Bf16
        };
        let Session {
            dev,
            stream,
            mut engine,
            _arena,
        } = open(fp8, args.cache_dir.as_deref())?;
        let fp8_probe = fp8.then(|| format!("{:?}", engine.fp8_capability()));
        let mut rows = Vec::new();
        for shape in model_shapes() {
            for &m in &args.m {
                let dims = (m, shape.n, shape.k);
                if fp8 && !shape.fp8_compatible() {
                    rows.push(GemmRow::skipped(
                        shape.role,
                        dims,
                        dtype.name(),
                        "shape not fp8 aligned",
                    ));
                    continue;
                }
                let need = (m * shape.k + shape.n * shape.k) * dtype.element_bytes()
                    + m * shape.n * 2
                    + (m + shape.n) * 4;
                if need as f64 > dev.info().free_bytes as f64 * 0.9 {
                    rows.push(GemmRow::skipped(
                        shape.role,
                        dims,
                        dtype.name(),
                        &format!(
                            "needs {need} bytes, device has {} free",
                            dev.info().free_bytes
                        ),
                    ));
                    continue;
                }
                let p = GemmProblem {
                    m,
                    n: shape.n,
                    k: shape.k,
                    dtype,
                    epilogue: Epilogue::None,
                };
                let measured = (|| -> Result<f64, String> {
                    let operands =
                        SyntheticOperands::new(&stream, &p, &engine).map_err(|e| e.to_string())?;
                    let idx = engine.tuned_index(&p).map_err(|e| e.to_string())?;
                    let times = engine
                        .time_gemm(
                            &p,
                            &operands.operands(),
                            idx,
                            args.warmup,
                            args.iters.max(1),
                        )
                        .map_err(|e| e.to_string())?;
                    median(&times).ok_or_else(|| "no timings".to_owned())
                })();
                rows.push(match measured {
                    Ok(us) => {
                        GemmRow::measured(shape.role, dims, dtype.name(), dtype.element_bytes(), us)
                    }
                    Err(e) => GemmRow::skipped(shape.role, dims, dtype.name(), &e),
                });
            }
        }
        let meta = ReportMeta {
            device: dev.info().name.clone(),
            dtype: dtype.name().to_owned(),
            warmup: args.warmup,
            iters: args.iters,
            fp8_probe,
        };
        print!("{}", render_markdown(&rows));
        if let Some(path) = &args.report {
            std::fs::write(path, render_json(&meta, &rows)).map_err(|e| e.to_string())?;
        }
        Ok(0)
    }

    /// Runs `tune`, adding to the tuning file already in `--cache-dir`; returns the process exit
    /// code.
    pub fn run_tune(args: TuneArgs) -> Result<i32, String> {
        let dtypes: Vec<GemmDtype> = if args.fp8 {
            vec![GemmDtype::Bf16, GemmDtype::Fp8E4m3]
        } else {
            vec![GemmDtype::Bf16]
        };
        let Session {
            dev, mut engine, ..
        } = open(args.fp8, Some(args.cache_dir.as_path()))?;
        let mut problems = Vec::new();
        if let Some(file) = &args.shapes {
            let text = std::fs::read_to_string(file).map_err(|e| e.to_string())?;
            let inventory = ShapeInventory::from_toml(&text).map_err(|e| e.to_string())?;
            for key in inventory.admission_shapes(args.max_shapes) {
                if dtypes.contains(&key.dtype) {
                    problems.push(GemmProblem {
                        m: key.m,
                        n: key.n,
                        k: key.k,
                        dtype: key.dtype,
                        epilogue: key.epilogue,
                    });
                }
            }
        }
        for shape in model_shapes().into_iter().filter(|_| args.shapes.is_none()) {
            for &m in &args.m {
                for &dtype in &dtypes {
                    if dtype == GemmDtype::Fp8E4m3 && !shape.fp8_compatible() {
                        continue;
                    }
                    problems.push(GemmProblem {
                        m,
                        n: shape.n,
                        k: shape.k,
                        dtype,
                        epilogue: Epilogue::None,
                    });
                }
            }
        }
        let path = cache_file(&args.cache_dir, &dev.info().name, &cuda_label());
        let table = engine
            .tune_and_save(&problems, args.iters, &path)
            .map_err(|e| e.to_string())?;
        println!("tuned {} keys -> {}", table.len(), path.display());
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> Vec<GemmRow> {
        vec![
            GemmRow::measured("vit.qkv", (1024, 3072, 1024), "bf16", 2, 100.0),
            GemmRow::measured("vit.qkv", (65536, 3072, 1024), "bf16", 2, 4000.0),
            GemmRow::skipped(
                "vit.patch_embed",
                (32, 1024, 1176),
                "fp8_e4m3",
                "shape not fp8 aligned",
            ),
        ]
    }

    #[test]
    fn measured_row_math() {
        let r = &rows()[0];
        // 2*1024*3072*1024 = 6.442 GFLOP in 100 us = 64.4 TFLOPS.
        assert!((r.tflops.unwrap() - 64.424_509_44).abs() < 1e-6);
        // 3072*1024*2 bytes in 100 us.
        assert!((r.weight_gbps.unwrap() - 62.914_56).abs() < 1e-6);
    }

    #[test]
    fn json_has_rows_and_nulls() {
        let meta = ReportMeta {
            device: "GB10 \"test\"".into(),
            dtype: "bf16".into(),
            warmup: 50,
            iters: 200,
            fp8_probe: None,
        };
        let j = render_json(&meta, &rows());
        assert!(j.contains("\"device\": \"GB10 \\\"test\\\"\""));
        assert!(j.contains("\"fp8_probe\": null"));
        assert!(j.contains("\"skipped\": \"shape not fp8 aligned\""));
        assert_eq!(j.matches("\"role\"").count(), 3);
        assert!(j.trim_end().ends_with('}'));
    }

    #[test]
    fn markdown_lists_every_row() {
        let md = render_markdown(&rows());
        assert_eq!(md.lines().count(), 5);
        assert!(md.contains("skipped: shape not fp8 aligned"));
    }

    #[test]
    fn sustained_ignores_small_m() {
        let rs = rows();
        let big = sustained_tflops(&rs, 16384).unwrap();
        assert!((big - 2.0 * 65536.0 * 3072.0 * 1024.0 / 4000e-6 / 1e12).abs() < 1e-6);
        assert_eq!(sustained_tflops(&rs, 1_000_000), None);
    }
}
