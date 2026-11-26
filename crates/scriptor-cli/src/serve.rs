//! `scriptor serve`: staged boot behind an already-bound HTTP listener, and the document
//! bridge that starts once the engine is ready.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::args::SmSplitSetting;
use cudarc::driver::CudaStream;
use scriptor_cuda::{Device, SmSplit};
use scriptor_engine::boot::{
    BootBackend, BootError, BootStatus, CacheDir, CacheReport, DeviceFacts, GraphOutcome,
    HostMemory, derive_concurrency, engine_host_bytes, plan_budget, plan_memory,
    readiness::SELF_TEST_MAX_TOKENS, run_boot,
};
use scriptor_engine::pressure::{PressureMonitor, Thresholds};
use scriptor_engine::supervise::panic_message;
use scriptor_engine::{EngineEvent, EngineExit, EngineHandle, SchedulerConfig};
use scriptor_kernels::heuristics::{TuningTable, cache_file};
use scriptor_server::{AppConfig, AppState, IntakeBudget, Metrics, router};
use scriptor_spec::{MemoryPlan, RuntimeParams};

use crate::args::ServeArgs;
use crate::device_step::StepLimits;
use crate::drain::drain;
use crate::embedded_bridge::{self, RunningBridge};
use crate::exit;
use crate::gpu_executor::{GpuExecutor, GpuWeights, max_draft_depth};
use crate::seat_engine::SeatEngine;
use crate::setup::Setup;

/// What the boot thread reported.
enum BootOutcome {
    /// The engine is open and serves this many concurrent streams.
    Ready(usize),
    Failed(i32, String),
    Panicked(String),
}

/// Removes machine-specific paths from text before it is logged.
pub struct Redactor {
    rules: Vec<(String, &'static str)>,
}

impl Redactor {
    /// Builds a redactor for the home directory, the model and cache paths and `HF_HOME`.
    pub fn new(model: &std::path::Path, cache: Option<&std::path::Path>) -> Self {
        let mut rules: Vec<(String, &'static str)> = Vec::new();
        rules.push((model.display().to_string(), "<model>"));
        if let Some(c) = cache {
            rules.push((c.display().to_string(), "<cache>"));
        }
        if let Ok(v) = std::env::var("HF_HOME") {
            rules.push((v, "<hf-home>"));
        }
        if let Ok(v) = std::env::var("HOME") {
            rules.push((v, "~"));
        }
        rules.retain(|(p, _)| p.len() > 1);
        rules.sort_by_key(|(p, _)| std::cmp::Reverse(p.len()));
        Self { rules }
    }

    /// Returns `text` with every known path replaced by a placeholder.
    pub fn apply(&self, text: &str) -> String {
        let mut out = text.to_owned();
        for (path, label) in &self.rules {
            out = out.replace(path.as_str(), label);
        }
        out
    }
}

/// Threads that decode images and render prompts: one per available core, leaving one for the
/// engine thread.
fn prep_workers() -> usize {
    std::thread::available_parallelism().map_or(2, |n| n.get().saturating_sub(1).max(1))
}

/// The split used when `--sm-split` is absent.
const DEFAULT_SPLIT: SmSplit = SmSplit {
    vision: 40,
    decode: 8,
};

/// The SM count [`DEFAULT_SPLIT`] is sized for.
const DEFAULT_SPLIT_DEVICE_SMS: u32 = 48;

/// The default split when `device` has exactly the SM count it is sized for and the driver
/// accepts it; otherwise `None` with one log line naming the reason.
fn default_split(device: &Device) -> Option<SmSplit> {
    let sms = match scriptor_cuda::green::device_sms(device.context()) {
        Ok(found) => found.sms,
        Err(e) => {
            eprintln!("scriptor: SM partitions unavailable ({e}); using the full device");
            return None;
        }
    };
    if sms != DEFAULT_SPLIT_DEVICE_SMS {
        eprintln!(
            "scriptor: default SM split {DEFAULT_SPLIT} is sized for {DEFAULT_SPLIT_DEVICE_SMS} SMs \
             but the device has {sms}; using the full device"
        );
        return None;
    }
    if let Err(e) = scriptor_cuda::green::validate_split(device.context(), &DEFAULT_SPLIT) {
        eprintln!(
            "scriptor: default SM split {DEFAULT_SPLIT} refused ({e}); using the full device"
        );
        return None;
    }
    Some(DEFAULT_SPLIT)
}

struct GpuBackend {
    setup: Setup,
    params: RuntimeParams,
    derive_concurrency: bool,
    reserve_hint: u64,
    pressure: Option<Thresholds>,
    vision_overlap: bool,
    embedding_release_after_prefill: bool,
    sm_setting: Option<SmSplitSetting>,
    sm_split: Option<SmSplit>,
    cache: Option<CacheDir>,
    engine: SeatEngine,
    exit: tokio::sync::mpsc::UnboundedSender<EngineExit>,
    device: Option<Device>,
    stream: Option<Arc<CudaStream>>,
    weights: Option<GpuWeights>,
    executor: Option<GpuExecutor>,
    tuning: TuningTable,
    tuning_found: bool,
    gemm_shape_log: Option<PathBuf>,
}

fn stage_err(stage: &'static str) -> impl Fn(scriptor_engine::EngineError) -> BootError {
    move |e| BootError::stage(stage, e)
}

/// Refuses boot when an L1 counter switch holds an invalid value.
fn check_l1_switches(
    counters: Option<&str>,
    step_lines: Option<&str>,
    pass_lines: Option<&str>,
) -> Result<(), BootError> {
    scriptor_engine::l1_counters::parse_all_switches(counters, step_lines, pass_lines)
        .map(|_| ())
        .map_err(BootError::Refused)
}

impl BootBackend for GpuBackend {
    type Ready = EngineHandle;

    fn detect(&mut self) -> Result<DeviceFacts, BootError> {
        check_l1_switches(
            std::env::var(scriptor_engine::l1_counters::L1_COUNTERS_ENV)
                .ok()
                .as_deref(),
            std::env::var(scriptor_engine::l1_counters::L1_STEP_LINES_ENV)
                .ok()
                .as_deref(),
            std::env::var(scriptor_engine::l1_counters::L1_PASS_LINES_ENV)
                .ok()
                .as_deref(),
        )?;
        let device = Device::open().map_err(|e| BootError::stage("detect", e))?;
        let info = device.info().clone();
        self.sm_split = match self.sm_setting {
            Some(SmSplitSetting::Off) => None,
            Some(SmSplitSetting::Split(split)) => {
                scriptor_cuda::green::validate_split(device.context(), &split)
                    .map_err(|e| BootError::Refused(format!("--sm-split {split}: {e}")))?;
                Some(split)
            }
            None => default_split(&device),
        };
        if let Some(dir) = &self.cache {
            let file = cache_file(dir.path(), &info.name, &cuda_label());
            if file.exists() {
                self.tuning = TuningTable::load(&file)
                    .map_err(|e| BootError::Cache(format!("tuning file: {e}")))?;
                self.tuning_found = true;
            }
        }
        self.device = Some(device);
        Ok(DeviceFacts {
            name: info.name,
            total_bytes: info.total_bytes as u64,
            free_bytes: info.free_bytes as u64,
            unified_memory: info.unified_memory,
            host: HostMemory::read(self.reserve_hint),
        })
    }

    fn plan(&mut self, device: &DeviceFacts) -> Result<MemoryPlan, BootError> {
        let text = &self.setup.spec.text;
        let limit = max_draft_depth(text.num_heads, text.num_kv_heads);
        if self.params.speculative_tokens > limit {
            return Err(BootError::Refused(format!(
                "--num-speculative-tokens {} exceeds the supported maximum of {limit} for one \
                 verification launch",
                self.params.speculative_tokens
            )));
        }
        if self.params.speculative_tokens > 0 && text.num_mtp_layers == 0 {
            return Err(BootError::Refused(
                "the checkpoint has no MTP layer; use --num-speculative-tokens 0".into(),
            ));
        }
        if self.derive_concurrency {
            self.params.max_concurrency = derive_concurrency(
                &self.setup.spec,
                &self.setup.image,
                &self.params,
                self.setup.weight_bytes,
                device,
            )?;
            self.derive_concurrency = false;
            eprintln!(
                "scriptor: serving {} concurrent streams (derived from the memory share)",
                self.params.max_concurrency
            );
        }
        let budget = plan_budget(&self.setup.spec, &self.setup.image, &self.params, device)?;
        eprintln!("scriptor: {}", budget.derivation);
        if let Some(host) = &device.host {
            self.pressure = Some(Thresholds::derive(
                host,
                &engine_host_bytes(&self.setup.spec, &self.setup.image, &self.params),
            ));
        }
        let plan = plan_memory(
            &self.setup.spec,
            &self.setup.image,
            &self.params,
            self.setup.weight_bytes,
            device,
        )?;
        eprintln!("scriptor: KV cache size: {} tokens", plan.kv_tokens());
        eprintln!(
            "scriptor: Maximum concurrency for {} tokens per request: {:.2}x",
            plan.max_context,
            plan.concurrency_at(plan.max_context)
        );
        eprintln!("scriptor: streams: {}", plan.slots);
        Ok(plan)
    }

    fn load_weights(&mut self, _plan: &MemoryPlan) -> Result<(), BootError> {
        let device = self.device.as_ref().expect("detected");
        let greatest = cudarc::driver::result::stream::get_priority_range().map_or(0, |(_, g)| g);
        let stream = device
            .context()
            .new_stream_with_priority(greatest)
            .map_err(|e| BootError::stage("weights", e))?;
        let weights = GpuWeights::load(
            &stream,
            &self.setup.dir,
            &self.setup.spec,
            self.params.speculative_tokens > 0,
        )
        .map_err(stage_err("weights"))?;
        self.stream = Some(stream);
        self.weights = Some(weights);
        Ok(())
    }

    fn build_kv_pool(&mut self, plan: &MemoryPlan) -> Result<(), BootError> {
        let device = self.device.as_ref().expect("detected");
        let stream = self.stream.as_ref().expect("stream");
        let weights = self.weights.take().expect("weights");
        let mut executor = GpuExecutor::new(
            device,
            stream,
            weights,
            &self.setup.spec,
            plan,
            self.params.speculative_tokens,
            std::mem::take(&mut self.tuning),
            StepLimits {
                max_seqs: self.params.max_concurrency,
                max_context: self.params.max_context,
            },
        )
        .map_err(stage_err("kv_pool"))?;
        if let Some(path) = self.gemm_shape_log.take() {
            executor.record_gemm_shapes(path);
        }
        self.executor = Some(executor);
        Ok(())
    }

    fn capture_graphs(&mut self) -> Result<GraphOutcome, BootError> {
        let executor = self.executor.as_mut().expect("executor");
        if let Some(split) = self.sm_split {
            match executor.partition(split) {
                Ok(()) => eprintln!(
                    "scriptor: SM partitions {split}: vision batches and decode graphs run on \
                     their own partitions; eager prefill stays on the full device"
                ),
                Err(e) if self.sm_setting.is_none() => {
                    eprintln!(
                        "scriptor: default SM split {split} not granted ({e}); using the full device"
                    );
                    self.sm_split = None;
                }
                Err(e) => return Err(stage_err("partition")(e)),
            }
        }
        let report = executor.capture_graphs().map_err(stage_err("graphs"))?;
        eprintln!(
            "scriptor: captured {} decode graphs in {:.2} s ({:?} capture, {:?}; selector scratch {} MiB requested, not counted in the capture figure)",
            report.graphs,
            report.seconds,
            report.scope,
            report.memory,
            report.split_scratch_bytes >> 20
        );
        if report.split_graphs > 0 {
            eprintln!(
                "scriptor: {} of those graphs use the two-stage target selector ({} buckets served, {} fall back to the single-block kernel; scratch rows are verification rows, {} of them): {:?} observed free-memory change over its warm-up and capture, {:.2} s of the capture; no budget-completeness claim",
                report.split_graphs,
                report.split_served_buckets,
                report.fallback_buckets,
                report.split_scratch_rows,
                report.split_memory,
                report.split_seconds
            );
        }
        Ok(GraphOutcome::Captured(report.graphs))
    }

    fn load_caches(&mut self, _cache: &CacheDir) -> Result<CacheReport, BootError> {
        let mut report = CacheReport::default();
        if self.tuning_found {
            report.loaded += 1;
        } else {
            report.missing.push("gemm tuning".into());
        }
        Ok(report)
    }

    fn start(&mut self, plan: &MemoryPlan) -> Result<EngineHandle, BootError> {
        scriptor_engine::spec_log::enabled_from_env().map_err(BootError::Refused)?;
        let spec = &self.setup.spec;
        let executor = self.executor.take().expect("executor");
        let config = SchedulerConfig {
            max_concurrency: self.params.max_concurrency,
            max_context: self.params.max_context,
            vision_batch_patches: plan.vision_batch_patches,
            vision_overlap: self.vision_overlap || self.sm_split.is_some(),
            spatial_merge: self.setup.image.merge_size,
            vocab_size: spec.text.vocab_size,
            eos_token_ids: spec.text.eos_token_ids.clone(),
            image_token_id: spec.image_token_id,
            embedding_cache_bytes: plan.embedding_cache as usize,
            embedding_hidden: spec.text.hidden_size,
            embedding_elem_bytes: 2,
            fallback_seed: 0x5eed,
            embedding_release_after_prefill: self.embedding_release_after_prefill,
        };
        let layout = scriptor_cache::KvLayout::from_plan(plan)
            .map_err(|e| BootError::stage("kv_pool", e))?;
        let exit = self.exit.clone();
        Ok(EngineHandle::spawn_with_policy(
            config,
            layout,
            executor,
            self.params.kv_policy,
            self.engine.fatal_slot(),
            move |e| {
                let _ = exit.send(e);
            },
        ))
    }

    fn self_test(&mut self, ready: &EngineHandle) -> Result<(), BootError> {
        let prepared = self
            .engine
            .self_test_request(SELF_TEST_MAX_TOKENS)
            .map_err(|e| BootError::stage("self_test", format!("{e:?}")))?;
        let mut stream = ready
            .submit(prepared.request)
            .map_err(stage_err("self_test"))?;
        let mut tokens = 0usize;
        loop {
            match stream.events.blocking_recv() {
                Some(EngineEvent::Token(_)) => tokens += 1,
                Some(EngineEvent::Finished(_)) => break,
                Some(EngineEvent::Failed(e)) => return Err(BootError::stage("self_test", e)),
                #[cfg(feature = "kv-demand-diagnostic")]
                Some(EngineEvent::Restarted) => {
                    return Err(BootError::stage("self_test", "the request was restarted"));
                }
                Some(EngineEvent::Aborted | EngineEvent::Cancelled(_)) | None => {
                    return Err(BootError::stage("self_test", "the request ended early"));
                }
            }
        }
        if tokens == 0 {
            return Err(BootError::stage(
                "self_test",
                "the request produced no tokens",
            ));
        }
        Ok(())
    }
}

/// Runs the server until a shutdown signal or a failed boot; returns the exit code.
pub fn run(args: &ServeArgs) -> i32 {
    let redact = Redactor::new(&args.model, args.cache_dir.as_deref());
    let log = |line: String| eprintln!("scriptor: {}", redact.apply(&line));
    if args.engine_file.is_some() {
        log("--engine-file is ignored; weights load from --model".into());
    }
    let mut setup = match Setup::load(&args.model) {
        Ok(s) => s,
        Err(e) => {
            log(format!("cannot read the checkpoint: {e}"));
            return exit::REFUSED;
        }
    };
    if let Some(max_pixels) = args.max_pixels {
        setup.image = match setup.image.with_max_pixels(max_pixels) {
            Ok(image) => image,
            Err(e) => {
                log(format!("--max-pixels: {e}"));
                return 1;
            }
        };
    }
    let mut bridge_config = match embedded_bridge::config(args) {
        Ok(config) => config,
        Err(e) => {
            log(format!("document bridge: {e}"));
            return exit::REFUSED;
        }
    };
    let bridge_link = embedded_bridge::link(args);
    let params = args.runtime_params();
    let derive_concurrency = args.max_num_seqs.is_none();
    let reserve_hint = args.reserve_hint();
    let status = Arc::new(BootStatus::new());
    let engine = match SeatEngine::new(
        &args.model,
        &setup.spec,
        &setup.image,
        prep_workers(),
        Arc::clone(&status),
        args.served_model_name.clone(),
        args.max_model_len as u64,
    ) {
        Ok(e) => e,
        Err(e) => {
            log(e);
            return exit::REFUSED;
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            log(format!("runtime: {e}"));
            return exit::INTERNAL;
        }
    };
    let address = args.listen_address();
    let listener = match runtime.block_on(tokio::net::TcpListener::bind(&address)) {
        Ok(l) => l,
        Err(e) => {
            log(format!("cannot listen on {address}: {e}"));
            return exit::TRANSIENT;
        }
    };
    log(format!("listening on {address}; booting"));
    let mut app_config = AppConfig {
        sampling_defaults: setup.spec.sampling_defaults,
        min_pixels: setup.image.min_pixels as u64,
        max_pixels: setup.image.max_pixels as u64,
        ..AppConfig::default()
    };
    let host_paths = scriptor_engine::host_mem::HostPaths::from_env();
    let Some(mem_available) =
        scriptor_engine::host_mem::read_meminfo(&host_paths).map(|m| m.available)
    else {
        log("cannot read host MemAvailable to derive the request intake budget".into());
        return exit::REFUSED;
    };
    let intake = IntakeBudget::derive(mem_available, app_config.max_request_bytes());
    log(intake.describe());
    app_config.intake_bytes = Some(intake.bytes);
    let state = AppState::new(
        Arc::new(engine.clone()),
        Arc::new(Metrics::new()),
        app_config,
    );
    let (done_tx, mut done_rx) = tokio::sync::mpsc::unbounded_channel::<BootOutcome>();
    let (exit_tx, mut exit_rx) = tokio::sync::mpsc::unbounded_channel::<EngineExit>();
    let (protect_tx, mut protect_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let monitor_slot: Arc<std::sync::Mutex<Option<scriptor_engine::pressure::MonitorGuard>>> =
        Arc::new(std::sync::Mutex::new(None));
    let boot_monitor = Arc::clone(&monitor_slot);
    let cache_dir: Option<PathBuf> = args.cache_dir.clone();
    let gemm_shape_log = args.gemm_shape_log.clone();
    let vision_overlap = args.vision_overlap;
    let embedding_release_after_prefill = args.embedding_release_after_prefill;
    let sm_setting = args.sm_split;
    let boot_engine = engine.clone();
    let boot_status = Arc::clone(&status);
    let started = Instant::now();
    let drain_bound = args.drain_seconds.map(Duration::from_secs_f64);
    let booter = std::thread::Builder::new()
        .name("scriptor-boot".into())
        .spawn(move || {
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                let cache = cache_dir.map(CacheDir::new);
                let mut backend = GpuBackend {
                    setup,
                    params,
                    derive_concurrency,
                    reserve_hint,
                    pressure: None,
                    vision_overlap,
                    embedding_release_after_prefill,
                    sm_setting,
                    sm_split: None,
                    cache: cache.clone(),
                    engine: boot_engine.clone(),
                    exit: exit_tx,
                    device: None,
                    stream: None,
                    weights: None,
                    executor: None,
                    tuning: TuningTable::default(),
                    tuning_found: false,
                    gemm_shape_log,
                };
                match run_boot(&mut backend, cache.as_ref(), &boot_status) {
                    Ok((handle, report)) => {
                        if let Some(thresholds) = backend.pressure {
                            let monitor = PressureMonitor::new(
                                thresholds,
                                scriptor_engine::host_mem::HostPaths::from_env(),
                                handle.pressure(),
                            );
                            let guard = monitor.spawn(handle.clone(), move || {
                                let _ = protect_tx.send(());
                            });
                            *boot_monitor.lock().unwrap_or_else(|p| p.into_inner()) = Some(guard);
                        }
                        boot_engine.open(handle);
                        let mut lines = vec![format!(
                            "ready in {:.1}s on {} ({})",
                            report.total_seconds,
                            report.device.name,
                            match &report.graphs {
                                scriptor_engine::boot::GraphOutcome::Eager =>
                                    "eager decode".to_owned(),
                                scriptor_engine::boot::GraphOutcome::Captured(n) =>
                                    format!("{n} graphs"),
                            }
                        )];
                        for s in &report.stages {
                            lines.push(format!("stage {} {:.2}s {}", s.name, s.seconds, s.note));
                        }
                        eprintln!("{}", lines.join("\n"));
                        BootOutcome::Ready(backend.params.max_concurrency)
                    }
                    Err(e) => BootOutcome::Failed(exit::boot_exit_code(&e), e.to_string()),
                }
            }))
            .unwrap_or_else(|p| BootOutcome::Panicked(panic_message(p.as_ref())));
            let _ = done_tx.send(outcome);
        });
    if let Err(e) = booter {
        log(format!("cannot start the boot thread: {e}"));
        return exit::INTERNAL;
    }
    let fatal_engine = engine.clone();
    let (signal_tx, mut signal_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let mut bridge: Option<RunningBridge> = None;
    let code = runtime.block_on(async {
        let (mut term, mut int) = match (
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()),
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()),
        ) {
            (Ok(t), Ok(i)) => (t, i),
            (Err(e), _) | (_, Err(e)) => {
                log(format!("cannot install signal handlers: {e}"));
                return 1;
            }
        };
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = term.recv() => {}
                    _ = int.recv() => {}
                }
                if signal_tx.send(()).is_err() {
                    return;
                }
            }
        });
        let app = router(state);
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
        let mut server = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = stop_rx.await;
                })
                .await
        });
        let mut code = exit::CLEAN;
        let mut booting = true;
        let mut fatal = false;
        let mut signalled = false;
        loop {
            tokio::select! {
                _ = signal_rx.recv() => { signalled = true; break },
                _ = protect_rx.recv() => {
                    log("host memory fell below the kernel watermark; all requests failed, exiting for a restart".into());
                    code = exit::TRANSIENT;
                    break;
                }
                msg = done_rx.recv(), if booting => match msg {
                    Some(BootOutcome::Ready(streams)) => {
                        booting = false;
                        log(format!("boot finished in {:.1}s", started.elapsed().as_secs_f64()));
                        if let Some(config) = bridge_config.take() {
                            let mut link = bridge_link.clone();
                            link.max_num_seqs.get_or_insert(streams);
                            let started_bridge = tokio::task::spawn_blocking(move || {
                                RunningBridge::start(config, &link)
                            })
                            .await
                            .unwrap_or_else(|e| Err(format!("bridge start panicked: {e}")));
                            match started_bridge {
                                Ok(running) => bridge = Some(running),
                                Err(e) => {
                                    log(format!("cannot start the document bridge: {e}"));
                                    code = exit::TRANSIENT;
                                    break;
                                }
                            }
                        }
                    }
                    Some(BootOutcome::Failed(c, e)) => { log(format!("boot failed: {e}")); code = c; break; }
                    Some(BootOutcome::Panicked(e)) => {
                        log(format!("boot failed: panic: {e}"));
                        code = exit::INTERNAL;
                        break;
                    }
                    None => {
                        log("boot failed: the boot thread ended without a result".into());
                        code = exit::INTERNAL;
                        break;
                    }
                },
                err = fatal_engine.wait_fatal() => {
                    log(format!("engine stopped: {err}; exiting for a restart"));
                    code = exit::TRANSIENT;
                    fatal = true;
                    break;
                }
                Some(EngineExit::Panicked(e)) = exit_rx.recv() => {
                    log(format!("engine failed: panic: {e}"));
                    code = exit::INTERNAL;
                    break;
                }
            }
        }
        if let Some(running) = &bridge {
            running.stop();
        }
        engine.begin_drain();
        let _ = stop_tx.send(());
        if fatal {
            let _ = tokio::time::timeout(fatal_flush_window(), server).await;
        } else if signalled {
            let deadline = engine.drain_deadline(drain_bound);
            log(format!(
                "draining: listener closed, health draining, deadline {:.1}s{}",
                deadline.as_secs_f64(),
                drain_bound.map_or(String::new(), |b| format!(
                    " (bounded by --drain-seconds {:.1})",
                    b.as_secs_f64()
                ))
            ));
            let end = drain(&mut server, &mut signal_rx, deadline, || {
                log("drain: cancelling the requests still in flight".into());
                engine.interrupt();
            })
            .await;
            log(format!("drain ended: {end:?}"));
        } else {
            engine.interrupt();
            let _ = drain(&mut server, &mut signal_rx, Duration::ZERO, || {}).await;
        }
        code
    });
    if let Some(guard) = monitor_slot
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .take()
    {
        guard.stop();
    }
    let clean = engine.shutdown();
    if let Some(running) = bridge
        && !running.wait(BRIDGE_JOIN_GRACE)
    {
        log("document bridge still finishing requests; exiting".into());
    }
    if code == 0 && !clean {
        exit::INTERNAL
    } else {
        code
    }
}

/// Time the process waits for the document bridge's in-flight documents after the engine has
/// stopped; they fail fast once it has.
const BRIDGE_JOIN_GRACE: Duration = Duration::from_secs(5);

/// Time the listener keeps flushing the failure of in-flight responses after a sticky device
/// fault before the process exits without draining. Defaults to one second; `SCRIPTOR_FATAL_FLUSH_MS`
/// overrides it.
fn fatal_flush_window() -> std::time::Duration {
    let ms = std::env::var("SCRIPTOR_FATAL_FLUSH_MS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(1000);
    std::time::Duration::from_millis(ms)
}

/// Label of the cuBLASLt version that names tuning files.
fn cuda_label() -> String {
    // SAFETY: plain version query with no arguments.
    let v = unsafe { cudarc::cublaslt::sys::cublasLtGetVersion() };
    format!("lt{v}")
}

#[cfg(test)]
mod l1_switch_tests {
    use super::*;

    #[test]
    fn l1_switch_error_refuses_boot_naming_the_variable() {
        assert!(check_l1_switches(None, None, None).is_ok());
        assert!(check_l1_switches(Some("1"), Some("1"), Some("1")).is_ok());
        let BootError::Refused(text) = check_l1_switches(Some("yes"), None, None).unwrap_err()
        else {
            panic!("expected a refusal");
        };
        assert!(text.contains("SCRIPTOR_L1_COUNTERS"), "{text}");
        let BootError::Refused(text) = check_l1_switches(Some("1"), Some("2"), None).unwrap_err()
        else {
            panic!("expected a refusal");
        };
        assert!(text.contains("SCRIPTOR_L1_STEP_LINES"), "{text}");
        assert!(check_l1_switches(None, Some("1"), None).is_err());
        let BootError::Refused(text) = check_l1_switches(Some("1"), None, Some("x")).unwrap_err()
        else {
            panic!("expected a refusal");
        };
        assert!(text.contains("SCRIPTOR_L1_PASS_LINES"), "{text}");
        assert!(check_l1_switches(None, None, Some("1")).is_err());
    }
}
