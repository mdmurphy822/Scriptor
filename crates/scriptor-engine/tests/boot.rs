//! Boot staging over a recording mock backend: stage order, readiness only after the
//! self-test, early refusal before any load, and no files without a cache directory.

use std::path::PathBuf;

use scriptor_engine::boot::{
    BOOT_RECORD, BootBackend, BootError, BootStatus, CacheDir, CacheReport, DeviceFacts,
    GraphOutcome, HostMemory, derive_concurrency, plan_budget, plan_memory, run_boot,
};
use scriptor_spec::{ImageProcessorConfig, MemoryPlan, ModelSpec, RuntimeParams};

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/config")
        .join(name);
    std::fs::read_to_string(path).unwrap()
}

struct Mock {
    calls: Vec<&'static str>,
    params: RuntimeParams,
    total: u64,
    free: u64,
    status: BootStatus,
    ready_during_self_test: Option<bool>,
    fail_self_test: bool,
}

impl Mock {
    fn new(params: RuntimeParams, total: u64) -> Self {
        Self {
            calls: Vec::new(),
            params,
            total,
            free: total,
            status: BootStatus::new(),
            ready_during_self_test: None,
            fail_self_test: false,
        }
    }
}

impl BootBackend for Mock {
    type Ready = ();

    fn detect(&mut self) -> Result<DeviceFacts, BootError> {
        self.calls.push("detect");
        Ok(DeviceFacts {
            name: "mock".into(),
            total_bytes: self.total,
            free_bytes: self.free,
            unified_memory: false,
            host: None,
        })
    }

    fn plan(&mut self, device: &DeviceFacts) -> Result<MemoryPlan, BootError> {
        self.calls.push("plan");
        let spec = ModelSpec::from_json(&fixture("glm-ocr.config.json"), None).unwrap();
        let image = ImageProcessorConfig::from_preprocessor_json(&fixture(
            "glm-ocr.preprocessor_config.json",
        ))
        .unwrap();
        plan_memory(&spec, &image, &self.params, 2_650_000_000, device)
    }

    fn load_weights(&mut self, _: &MemoryPlan) -> Result<(), BootError> {
        self.calls.push("weights");
        Ok(())
    }

    fn build_kv_pool(&mut self, _: &MemoryPlan) -> Result<(), BootError> {
        self.calls.push("kv_pool");
        Ok(())
    }

    fn load_caches(&mut self, _: &CacheDir) -> Result<CacheReport, BootError> {
        self.calls.push("caches");
        Ok(CacheReport::default())
    }

    fn start(&mut self, _: &MemoryPlan) -> Result<(), BootError> {
        self.calls.push("start");
        Ok(())
    }

    fn self_test(&mut self, _: &()) -> Result<(), BootError> {
        self.calls.push("self_test");
        self.ready_during_self_test = Some(self.status.is_ready());
        if self.fail_self_test {
            Err(BootError::stage("self_test", "no tokens"))
        } else {
            Ok(())
        }
    }
}

fn small_params() -> RuntimeParams {
    RuntimeParams {
        max_concurrency: 4,
        max_context: 2048,
        gpu_share: 0.5,
        ..RuntimeParams::default()
    }
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("scriptor-boot-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn boot_stages_run_in_order_and_readiness_follows_the_self_test() {
    let mut mock = Mock::new(small_params(), 64 << 30);
    let status = mock.status.clone();
    let (_, report) = run_boot(&mut mock, None, &status).expect("boot");
    assert_eq!(
        mock.calls,
        ["detect", "plan", "weights", "kv_pool", "start", "self_test"]
    );
    assert_eq!(mock.ready_during_self_test, Some(false));
    assert!(status.is_ready());
    let names: Vec<_> = report.stages.iter().map(|s| s.name).collect();
    assert_eq!(names, scriptor_engine::boot::STAGES);
    assert_eq!(report.graphs, GraphOutcome::Eager);
}

#[test]
fn boot_a_failed_self_test_leaves_the_engine_not_ready() {
    let mut mock = Mock::new(small_params(), 64 << 30);
    mock.fail_self_test = true;
    let status = mock.status.clone();
    let err = run_boot(&mut mock, None, &status).expect_err("fails");
    assert!(err.to_string().contains("self_test"));
    assert!(!status.is_ready());
}

#[test]
fn boot_an_oversized_plan_is_refused_before_anything_loads() {
    let params = RuntimeParams {
        max_concurrency: 128,
        max_context: 262144,
        gpu_share: 0.15,
        ..RuntimeParams::default()
    };
    let mut mock = Mock::new(params, 120 << 30);
    let status = mock.status.clone();
    let err = run_boot(&mut mock, None, &status).expect_err("refused");
    assert_eq!(mock.calls, ["detect", "plan"]);
    let text = err.to_string();
    assert!(text.contains("short by"), "{text}");
    assert!(!status.is_ready());
}

#[test]
fn boot_low_free_memory_is_refused_early() {
    let mut mock = Mock::new(small_params(), 64 << 30);
    mock.free = 1 << 30;
    let status = mock.status.clone();
    let err = run_boot(&mut mock, None, &status).expect_err("refused");
    assert_eq!(mock.calls, ["detect", "plan"]);
    assert!(err.to_string().contains("free"));
}

#[test]
fn boot_nothing_is_written_without_a_cache_directory() {
    let dir = scratch("none");
    let mut mock = Mock::new(small_params(), 64 << 30);
    let status = mock.status.clone();
    let (_, report) = run_boot(&mut mock, None, &status).expect("boot");
    assert!(!mock.calls.contains(&"caches"));
    assert!(report.caches.is_none());
    assert!(!dir.exists());
    let cache = status.snapshot().into_iter().find(|s| s.name == "caches");
    assert_eq!(
        cache.map(|s| s.state),
        Some(scriptor_engine::boot::StageState::Skipped)
    );
}

#[test]
fn boot_an_explicit_cache_directory_receives_the_boot_record() {
    let dir = scratch("some");
    let cache = CacheDir::new(&dir);
    let mut mock = Mock::new(small_params(), 64 << 30);
    let status = mock.status.clone();
    let (_, report) = run_boot(&mut mock, Some(&cache), &status).expect("boot");
    assert!(mock.calls.contains(&"caches"));
    assert!(report.caches.is_some());
    let record = std::fs::read_to_string(dir.join(BOOT_RECORD)).expect("record written");
    assert!(record.contains("\"stages\""));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn boot_a_failed_boot_writes_no_record() {
    let dir = scratch("fail");
    let cache = CacheDir::new(&dir);
    let mut mock = Mock::new(small_params(), 64 << 30);
    mock.fail_self_test = true;
    let status = mock.status.clone();
    assert!(run_boot(&mut mock, Some(&cache), &status).is_err());
    assert!(!dir.join(BOOT_RECORD).exists());
    let _ = std::fs::remove_dir_all(&dir);
}

fn unified(total: u64, available: u64, hint: u64) -> DeviceFacts {
    DeviceFacts {
        name: "unified".into(),
        total_bytes: total,
        free_bytes: total,
        unified_memory: true,
        host: Some(HostMemory {
            total,
            available,
            watermark: 1 << 30,
            reserve_hint: hint,
        }),
    }
}

fn model() -> (ModelSpec, ImageProcessorConfig) {
    (
        ModelSpec::from_json(&fixture("glm-ocr.config.json"), None).unwrap(),
        ImageProcessorConfig::from_preprocessor_json(&fixture("glm-ocr.preprocessor_config.json"))
            .unwrap(),
    )
}

#[test]
fn unified_plan_fits_inside_the_derived_budget() {
    let (spec, image) = model();
    let params = small_params();
    let device = unified(64 << 30, 60 << 30, 0);
    let budget = plan_budget(&spec, &image, &params, &device).unwrap();
    let plan = plan_memory(&spec, &image, &params, 2_650_000_000, &device).unwrap();
    assert!(plan.total <= budget.bytes);
    assert!(budget.derivation.contains("MemAvailable"));
}

#[test]
fn unified_hint_shrinks_the_budget_and_can_refuse() {
    let (spec, image) = model();
    let mut params = small_params();
    params.gpu_share = 0.9;
    let open = plan_budget(&spec, &image, &params, &unified(64 << 30, 40 << 30, 0)).unwrap();
    let hinted = plan_budget(
        &spec,
        &image,
        &params,
        &unified(64 << 30, 40 << 30, 8 << 30),
    )
    .unwrap();
    assert!(hinted.bytes < open.bytes);
    let tight = unified(64 << 30, 20 << 30, 0);
    let err = plan_memory(&spec, &image, &params, 2_650_000_000, &tight).expect_err("refused");
    assert!(matches!(err, BootError::Refused(_)));
}

#[test]
fn derived_concurrency_fits_the_budget_it_implies() {
    let (spec, image) = model();
    let device = unified(64 << 30, 60 << 30, 0);
    let mut params = small_params();
    params.max_context = 4096;
    let n = derive_concurrency(&spec, &image, &params, 2_650_000_000, &device).unwrap();
    assert!(n >= 1);
    params.max_concurrency = n;
    plan_memory(&spec, &image, &params, 2_650_000_000, &device).expect("derived count plans");
    let reserved = unified(64 << 30, 60 << 30, 20 << 30);
    let fewer = derive_concurrency(&spec, &image, &params, 2_650_000_000, &reserved).unwrap();
    assert!(fewer <= n);
}
