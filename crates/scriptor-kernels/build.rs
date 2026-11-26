//! Build script: with the `cuda` feature, compiles every `kernels/*.cu` to an
//! sm_121 fatbin and generates the table that embeds them.
//!
//! Sources compile concurrently, bounded by cargo's `NUM_JOBS`. Each fatbin is
//! stored in a content-addressed cache keyed by the nvcc identity, the full
//! argument list, the source bytes and every file the source includes, so
//! identical inputs are compiled once across target directories. The cache
//! lives in `$SCRIPTOR_KERNEL_CACHE` or `$HOME/.cache/scriptor/kernels`, and
//! `SCRIPTOR_KERNEL_CACHE=off` disables it.

use sha2::{Digest, Sha256};
use std::ffi::{OsStr, OsString};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const NVCC_FLAGS: &[&str] = &[
    "-arch=sm_121",
    "-gencode",
    "arch=compute_121,code=[sm_121,compute_121]",
    "-O3",
    "-fatbin",
    "-I",
    ".",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../kernels");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=SCRIPTOR_KERNEL_CACHE");
    println!("cargo:rerun-if-env-changed=HOME");
    println!("cargo:rustc-check-cfg=cfg(scriptor_cuda_enabled)");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    println!("cargo:rustc-cfg=scriptor_cuda_enabled");

    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("out dir"));
    let kernel_dir = manifest.join("../../kernels");
    let nvcc = std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());

    let mut sources: Vec<PathBuf> = std::fs::read_dir(&kernel_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "cu"))
                .collect()
        })
        .unwrap_or_default();
    sources.sort();

    let tree = hash_tree(&kernel_dir);
    for (path, _) in &tree.files {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let ctx = Context {
        cache: cache_dir(),
        identity: nvcc_identity(&nvcc),
        nvcc,
        kernel_dir,
        out_dir: out_dir.clone(),
        tree_digest: tree.digest,
    };

    let results: Vec<Mutex<Option<BuildResult>>> =
        sources.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let workers = job_limit().min(sources.len()).max(1);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some(src) = sources.get(i) else { break };
                    let result = build_one(&ctx, src);
                    *results[i].lock().expect("result slot") = Some(result);
                }
            });
        }
    });

    let mut table = String::from("pub static FATBINS: &[(&str, &[u8])] = &[\n");
    let mut failures = Vec::new();
    for (src, slot) in sources.iter().zip(results) {
        match slot.into_inner().expect("result slot").expect("built") {
            Ok(inputs) => {
                for input in inputs {
                    println!("cargo:rerun-if-changed={}", input.display());
                }
                let stem = stem_of(src);
                let out = out_dir.join(format!("{stem}.fatbin"));
                writeln!(
                    table,
                    "    ({stem:?}, include_bytes!({:?})),",
                    out.to_str().expect("utf-8 out path")
                )
                .expect("write to string");
            }
            Err(message) => failures.push(message),
        }
    }
    if !failures.is_empty() {
        panic!(
            "{} kernel source(s) failed:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
    table.push_str("];\n");
    std::fs::write(out_dir.join("fatbins.rs"), table).expect("write fatbins.rs");
}

/// Files that trigger a rerun when changed, or the failure report for one source.
type BuildResult = Result<Vec<PathBuf>, String>;

struct Context {
    nvcc: OsString,
    kernel_dir: PathBuf,
    out_dir: PathBuf,
    cache: Option<PathBuf>,
    identity: String,
    tree_digest: String,
}

struct Tree {
    files: Vec<(PathBuf, Vec<u8>)>,
    digest: String,
}

fn stem_of(src: &Path) -> &str {
    src.file_stem()
        .and_then(|s| s.to_str())
        .expect("utf-8 kernel name")
}

/// Number of concurrent compiles: cargo's `NUM_JOBS`, else the host parallelism.
fn job_limit() -> usize {
    std::env::var("NUM_JOBS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1)
}

/// The cache directory, or `None` when caching is disabled or no location is known.
fn cache_dir() -> Option<PathBuf> {
    match std::env::var_os("SCRIPTOR_KERNEL_CACHE") {
        Some(v) if v == "off" => None,
        Some(v) if !v.is_empty() => Some(PathBuf::from(v)),
        _ => std::env::var_os("HOME")
            .filter(|h| !h.is_empty())
            .map(|h| PathBuf::from(h).join(".cache/scriptor/kernels")),
    }
}

/// The nvcc path and its `--version` output.
fn nvcc_identity(nvcc: &OsStr) -> String {
    let version = Command::new(nvcc)
        .arg("--version")
        .output()
        .unwrap_or_else(|e| panic!("failed to run nvcc: {e}"));
    format!(
        "{}\n{}{}",
        Path::new(nvcc).display(),
        String::from_utf8_lossy(&version.stdout),
        String::from_utf8_lossy(&version.stderr)
    )
}

/// Every file under `dir` (recursively, sorted) with its bytes, and a digest over names and bytes.
fn hash_tree(dir: &Path) -> Tree {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in rd.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else {
                out.push(path);
            }
        }
    }
    let mut paths = Vec::new();
    walk(dir, &mut paths);
    paths.sort();
    let mut hasher = Sha256::new();
    let mut files = Vec::new();
    for path in paths {
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let rel = path.strip_prefix(dir).unwrap_or(&path);
        hasher.update(rel.to_string_lossy().as_bytes());
        hasher.update([0]);
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
        files.push((path, bytes));
    }
    Tree {
        files,
        digest: hex(&hasher.finalize()),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// Compiles one source into `OUT_DIR`, through the cache when enabled.
/// Returns the files whose change must rerun the build.
fn build_one(ctx: &Context, src: &Path) -> BuildResult {
    let name = src.file_name().expect("source name");
    let stem = stem_of(src);
    let out = ctx.out_dir.join(format!("{stem}.fatbin"));
    let Some(cache) = &ctx.cache else {
        compile(ctx, name, &out)?;
        return Ok(Vec::new());
    };

    let deps = included_files(ctx, name)?;
    let mut hasher = Sha256::new();
    hasher.update(ctx.identity.as_bytes());
    for flag in NVCC_FLAGS {
        hasher.update([0]);
        hasher.update(flag.as_bytes());
    }
    hasher.update([0]);
    hasher.update(name.as_encoded_bytes());
    hasher.update(ctx.tree_digest.as_bytes());
    for dep in &deps {
        let bytes = std::fs::read(dep).map_err(|e| format!("read {}: {e}", dep.display()))?;
        hasher.update(dep.as_os_str().as_encoded_bytes());
        hasher.update([0]);
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    let entry = cache.join(format!("{}.fatbin", hex(&hasher.finalize())));

    if entry.is_file() && std::fs::copy(&entry, &out).is_ok() {
        return Ok(deps);
    }
    if let Err(e) = std::fs::create_dir_all(cache) {
        println!("cargo:warning=kernel cache unavailable ({e}); compiling uncached");
        compile(ctx, name, &out)?;
        return Ok(deps);
    }
    let tmp = cache.join(format!(".{stem}.{}.tmp", std::process::id()));
    compile(ctx, name, &tmp)?;
    std::fs::copy(&tmp, &out).map_err(|e| format!("copy {stem}: {e}"))?;
    if std::fs::rename(&tmp, &entry).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    Ok(deps)
}

/// The files `src` includes, as reported by `nvcc -M`.
fn included_files(ctx: &Context, name: &OsStr) -> Result<Vec<PathBuf>, String> {
    let output = Command::new(&ctx.nvcc)
        .current_dir(&ctx.kernel_dir)
        .args(NVCC_FLAGS)
        .arg("-M")
        .arg(name)
        .output()
        .map_err(|e| format!("failed to run nvcc: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "nvcc -M failed for {}:\n{}{}",
            Path::new(name).display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout).replace("\\\n", " ");
    let listed = text.split_once(':').map_or("", |(_, rest)| rest);
    let mut deps: Vec<PathBuf> = listed
        .split_whitespace()
        .map(|t| ctx.kernel_dir.join(t))
        .collect();
    deps.sort();
    deps.dedup();
    if deps.is_empty() {
        return Err(format!(
            "nvcc -M listed no inputs for {}",
            Path::new(name).display()
        ));
    }
    Ok(deps)
}

fn compile(ctx: &Context, name: &OsStr, out: &Path) -> Result<(), String> {
    let output = Command::new(&ctx.nvcc)
        .current_dir(&ctx.kernel_dir)
        .args(NVCC_FLAGS)
        .arg("-o")
        .arg(out)
        .arg(name)
        .output()
        .map_err(|e| format!("failed to run nvcc: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "nvcc failed for {}:\n{}{}",
            Path::new(name).display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}
