//! Staged boot: detect the device, plan memory (refusing early), load weights, build the KV
//! pool, load optional caches, run the graph-capture seam, start the engine, and open readiness
//! only after a self-test request completes. Persistent state is touched only through an
//! explicitly supplied [`CacheDir`].

pub mod caches;
pub mod plan;
pub mod readiness;
pub mod stages;

use std::time::Instant;

pub use caches::{BOOT_RECORD, CacheDir, CacheReport};
pub use plan::{
    BootBudget, DeviceFacts, EngineHostBytes, HostMemory, boot_budget, derive_concurrency,
    engine_host_bytes, parse_mem_available, plan_budget, plan_memory,
};
pub use stages::{BootStatus, STAGES, StageRecord, StageState};

use scriptor_spec::MemoryPlan;

/// Why a boot stopped.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BootError {
    /// The planned memory does not fit; the text carries the suggestion.
    #[error("{0}")]
    Refused(String),
    /// A stage failed.
    #[error("{stage} failed: {message}")]
    Stage {
        stage: &'static str,
        message: String,
    },
    /// The cache directory could not be used.
    #[error("cache: {0}")]
    Cache(String),
}

impl BootError {
    /// A stage failure.
    pub fn stage(stage: &'static str, message: impl ToString) -> Self {
        Self::Stage {
            stage,
            message: message.to_string(),
        }
    }
}

/// What the graph-capture seam did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphOutcome {
    /// Nothing captured; decode steps run eagerly.
    Eager,
    /// Graphs captured for this many shapes.
    Captured(usize),
}

/// The environment-specific side of a boot. Each method runs once, in stage order.
pub trait BootBackend {
    /// The running engine produced by [`BootBackend::start`].
    type Ready;

    /// Detects the device.
    fn detect(&mut self) -> Result<DeviceFacts, BootError>;

    /// Computes the memory plan; an error here stops the boot before anything is loaded.
    fn plan(&mut self, device: &DeviceFacts) -> Result<MemoryPlan, BootError>;

    /// Loads weights onto the device.
    fn load_weights(&mut self, plan: &MemoryPlan) -> Result<(), BootError>;

    /// Allocates the KV pools.
    fn build_kv_pool(&mut self, plan: &MemoryPlan) -> Result<(), BootError>;

    /// Loads derived state from the cache directory; called only when one was supplied.
    fn load_caches(&mut self, cache: &CacheDir) -> Result<CacheReport, BootError>;

    /// Captures decode graphs. The default runs eagerly.
    fn capture_graphs(&mut self) -> Result<GraphOutcome, BootError> {
        Ok(GraphOutcome::Eager)
    }

    /// Starts the scheduler over the loaded model.
    fn start(&mut self, plan: &MemoryPlan) -> Result<Self::Ready, BootError>;

    /// Runs the self-test request against the started engine.
    fn self_test(&mut self, ready: &Self::Ready) -> Result<(), BootError>;
}

/// Outcome of a completed boot.
#[derive(Debug, Clone)]
pub struct BootReport {
    pub device: DeviceFacts,
    pub plan: MemoryPlan,
    pub stages: Vec<StageRecord>,
    pub total_seconds: f64,
    pub graphs: GraphOutcome,
    pub caches: Option<CacheReport>,
}

fn timed<T>(
    status: &BootStatus,
    stage: &'static str,
    f: impl FnOnce() -> Result<T, BootError>,
    note: impl FnOnce(&T) -> String,
) -> Result<T, BootError> {
    status.begin(stage);
    let t = Instant::now();
    let value = f()?;
    status.done(stage, t.elapsed(), &note(&value));
    Ok(value)
}

/// Runs every stage in order against `backend`, recording progress in `status`. With no
/// `cache`, the cache stage is skipped and no file or directory is created.
pub fn run_boot<B: BootBackend>(
    backend: &mut B,
    cache: Option<&CacheDir>,
    status: &BootStatus,
) -> Result<(B::Ready, BootReport), BootError> {
    let started = Instant::now();
    let device = timed(status, "detect", || backend.detect(), |d| d.name.clone())?;
    let plan = timed(
        status,
        "plan",
        || backend.plan(&device),
        |p| {
            format!(
                "{} MiB planned of {} MiB; {} streams; {}",
                p.total >> 20,
                p.available >> 20,
                p.slots,
                p.capacity_summary()
            )
        },
    )?;
    timed(
        status,
        "weights",
        || backend.load_weights(&plan),
        |()| String::new(),
    )?;
    timed(
        status,
        "kv_pool",
        || backend.build_kv_pool(&plan),
        |()| format!("{} blocks", plan.kv_blocks),
    )?;
    let caches = match cache {
        Some(dir) => Some(timed(
            status,
            "caches",
            || backend.load_caches(dir),
            |r| format!("{} loaded, {} missing", r.loaded, r.missing.len()),
        )?),
        None => {
            status.skipped("caches", "no cache directory requested");
            None
        }
    };
    let graphs = timed(
        status,
        "graphs",
        || backend.capture_graphs(),
        |g| match g {
            GraphOutcome::Eager => "eager".to_owned(),
            GraphOutcome::Captured(n) => format!("{n} captured"),
        },
    )?;
    let ready = backend.start(&plan)?;
    timed(
        status,
        "self_test",
        || backend.self_test(&ready),
        |()| String::new(),
    )?;
    let report = BootReport {
        device,
        plan,
        stages: status.snapshot(),
        total_seconds: started.elapsed().as_secs_f64(),
        graphs,
        caches,
    };
    if let Some(dir) = cache {
        dir.write(BOOT_RECORD, render_record(&report).as_bytes())?;
    }
    Ok((ready, report))
}

/// JSON text of the boot record written into the cache directory.
pub fn render_record(report: &BootReport) -> String {
    let stages: Vec<serde_json::Value> = report
        .stages
        .iter()
        .map(|s| {
            serde_json::json!({
                "stage": s.name,
                "seconds": s.seconds,
                "note": s.note,
            })
        })
        .collect();
    serde_json::json!({
        "device": report.device.name,
        "total_seconds": report.total_seconds,
        "planned_bytes": report.plan.total,
        "stages": stages,
    })
    .to_string()
}
