//! Host-pressure monitor: samples host memory and memory-stall pressure and moves the engine
//! through the tiers Normal, Watch, Shed and Protect, only ever freeing memory in response.
//!
//! Tier thresholds derive from the boot reserve (see [`Thresholds::derive`]):
//! - Watch: available memory below the host reserve plus the host footprint one more stream
//!   adds; new requests are refused with a retry delay.
//! - Shed: available memory below the host reserve, or below the Watch level with full memory
//!   stall sustained for one kernel pressure window; the youngest running request is cancelled once per decode step,
//!   the embedding cache is dropped and the device memory pool is trimmed.
//! - Protect: available memory below the kernel watermark; every request is failed and the
//!   owner is told to drain and exit.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::boot::{EngineHostBytes, HostMemory};
use crate::handle::EngineHandle;
use crate::host_mem::{self, HostPaths};

/// Seconds the kernel averages `avg10` over; full stall sustained this long, with available
/// memory below the Watch level, counts as Shed.
pub const PSI_WINDOW_SECS: f64 = 10.0;
/// Decode steps between two samples while no shedding is under way: the declared ratio that
/// ties the sampling interval to the measured step time.
pub const SAMPLE_STEPS: f64 = 8.0;
/// Shortest sampling interval in seconds, the granularity of the monitor thread's sleep.
pub const MIN_INTERVAL_SECS: f64 = 0.05;
/// Interval in seconds used until a decode step has been measured.
pub const UNMEASURED_INTERVAL_SECS: f64 = 1.0;
/// Environment variable overriding the sampling interval, in milliseconds.
pub const INTERVAL_ENV: &str = "SCRIPTOR_PRESSURE_INTERVAL_MS";

/// Pressure tier, ordered by severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Normal = 0,
    Watch = 1,
    Shed = 2,
    Protect = 3,
}

impl Tier {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Tier::Watch,
            2 => Tier::Shed,
            3 => Tier::Protect,
            _ => Tier::Normal,
        }
    }

    /// Lowercase name for logs and metric labels.
    pub fn name(self) -> &'static str {
        match self {
            Tier::Normal => "normal",
            Tier::Watch => "watch",
            Tier::Shed => "shed",
            Tier::Protect => "protect",
        }
    }
}

/// `MemAvailable` levels, in bytes, below which each tier applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Thresholds {
    pub watch_below: u64,
    pub shed_below: u64,
    pub protect_below: u64,
}

impl Thresholds {
    /// Protect below the kernel watermark, Shed below the host reserve (bytes in use by the
    /// rest of the machine at boot, the watermark and the host's hint), Watch below the reserve
    /// plus the host footprint of one stream. The engine's host footprint for all streams is
    /// already resident, and so already absent from `MemAvailable`, when a saturated engine
    /// is sampled; Watch therefore spans only what admitting one more stream can add.
    pub fn derive(host: &HostMemory, engine_host: &EngineHostBytes) -> Self {
        let reserve = host.base_reserve();
        Self {
            watch_below: reserve.saturating_add(engine_host.per_stream()),
            shed_below: reserve,
            protect_below: host.watermark,
        }
    }

    /// The tier for `available` bytes; `full_stall_sustained` raises Watch to Shed and leaves
    /// Normal unchanged.
    pub fn classify(&self, available: u64, full_stall_sustained: bool) -> Tier {
        if available < self.protect_below {
            Tier::Protect
        } else if available < self.shed_below
            || (available < self.watch_below && full_stall_sustained)
        {
            Tier::Shed
        } else if available < self.watch_below {
            Tier::Watch
        } else {
            Tier::Normal
        }
    }
}

/// Pressure figures shared between the monitor, the engine handle and the metrics endpoint.
#[derive(Debug, Default)]
pub struct PressureState {
    tier: std::sync::atomic::AtomicU8,
    available: AtomicU64,
    psi_full_milli: AtomicU64,
    retry_after_secs: AtomicU64,
    shed_actions: AtomicU64,
    protect_actions: AtomicU64,
    rejected: AtomicU64,
}

/// A consistent-enough copy of [`PressureState`] for metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureSnapshot {
    pub tier: Tier,
    pub host_mem_available_bytes: u64,
    pub psi_full_avg10: f64,
    pub shed_actions: u64,
    pub protect_actions: u64,
    pub admission_rejected: u64,
}

impl PressureState {
    /// The current tier.
    pub fn tier(&self) -> Tier {
        Tier::from_u8(self.tier.load(Ordering::SeqCst))
    }

    /// Seconds a refused client is told to wait.
    pub fn retry_after_secs(&self) -> u64 {
        self.retry_after_secs.load(Ordering::SeqCst).max(1)
    }

    /// Counts a request refused because of pressure.
    pub fn record_rejected(&self) {
        self.rejected.fetch_add(1, Ordering::SeqCst);
    }

    /// Current values.
    pub fn snapshot(&self) -> PressureSnapshot {
        PressureSnapshot {
            tier: self.tier(),
            host_mem_available_bytes: self.available.load(Ordering::SeqCst),
            psi_full_avg10: self.psi_full_milli.load(Ordering::SeqCst) as f64 / 1000.0,
            shed_actions: self.shed_actions.load(Ordering::SeqCst),
            protect_actions: self.protect_actions.load(Ordering::SeqCst),
            admission_rejected: self.rejected.load(Ordering::SeqCst),
        }
    }
}

/// Samples the host and tracks the tier; the thread driver is [`PressureMonitor::spawn`].
pub struct PressureMonitor {
    thresholds: Thresholds,
    paths: HostPaths,
    state: Arc<PressureState>,
    psi_stalled_secs: f64,
}

impl PressureMonitor {
    /// A monitor reading `paths` and publishing into `state`.
    pub fn new(thresholds: Thresholds, paths: HostPaths, state: Arc<PressureState>) -> Self {
        Self {
            thresholds,
            paths,
            state,
            psi_stalled_secs: 0.0,
        }
    }

    /// Takes one sample, `elapsed` seconds after the previous one, and publishes it. An
    /// unreadable meminfo keeps the previous tier; an unreadable pressure file counts as no
    /// stall.
    pub fn sample(&mut self, elapsed: f64) -> Tier {
        let Some(info) = host_mem::read_meminfo(&self.paths) else {
            return self.state.tier();
        };
        let psi = host_mem::read_psi(&self.paths).map_or(0.0, |p| p.full_avg10);
        if psi > 0.0 {
            self.psi_stalled_secs += elapsed;
        } else {
            self.psi_stalled_secs = 0.0;
        }
        let sustained = self.psi_stalled_secs >= PSI_WINDOW_SECS;
        let tier = self.thresholds.classify(info.available, sustained);
        let s = &self.state;
        s.available.store(info.available, Ordering::SeqCst);
        s.psi_full_milli
            .store((psi * 1000.0).round() as u64, Ordering::SeqCst);
        s.tier.store(tier as u8, Ordering::SeqCst);
        tier
    }

    /// Seconds between samples for a mean decode step of `step_secs` (zero when unmeasured):
    /// [`SAMPLE_STEPS`] steps, or one step while shedding so a request is cancelled per step.
    pub fn interval_secs(tier: Tier, step_secs: f64) -> f64 {
        if let Some(ms) = std::env::var(INTERVAL_ENV)
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
        {
            return (ms / 1000.0).max(MIN_INTERVAL_SECS / 10.0);
        }
        if step_secs <= 0.0 {
            return UNMEASURED_INTERVAL_SECS;
        }
        let steps = if tier >= Tier::Shed {
            1.0
        } else {
            SAMPLE_STEPS
        };
        (step_secs * steps).max(MIN_INTERVAL_SECS)
    }

    /// Runs the monitor on its own thread until stopped or Protect is reached. Shed samples
    /// send a shed command to the engine; Protect sends the protect command, calls
    /// `on_protect` once and ends the thread. Each tier change is logged with its inputs.
    pub fn spawn(
        mut self,
        engine: EngineHandle,
        on_protect: impl FnOnce() + Send + 'static,
    ) -> MonitorGuard {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let state = Arc::clone(&self.state);
        let thresholds = self.thresholds;
        let thread = std::thread::Builder::new()
            .name("scriptor-pressure".into())
            .spawn(move || {
                let mut last = Tier::Normal;
                let mut interval = MonitorGuard::first_interval();
                let mut on_protect = Some(on_protect);
                while !flag.load(Ordering::SeqCst) {
                    let tier = self.sample(interval);
                    let snap = self.state.snapshot();
                    if tier != last {
                        eprintln!(
                            "scriptor: host memory pressure {} -> {}: MemAvailable {} MiB, \
                             full stall avg10 {:.2}, thresholds watch<{} shed<{} protect<{} MiB",
                            last.name(),
                            tier.name(),
                            snap.host_mem_available_bytes >> 20,
                            snap.psi_full_avg10,
                            thresholds.watch_below >> 20,
                            thresholds.shed_below >> 20,
                            thresholds.protect_below >> 20
                        );
                        last = tier;
                    }
                    let metrics = engine.metrics();
                    let step_secs = if metrics.decode_steps == 0 {
                        0.0
                    } else {
                        metrics.decode_seconds / metrics.decode_steps as f64
                    };
                    interval = Self::interval_secs(tier, step_secs);
                    state
                        .retry_after_secs
                        .store(interval.ceil() as u64, Ordering::SeqCst);
                    match tier {
                        Tier::Shed => {
                            state.shed_actions.fetch_add(1, Ordering::SeqCst);
                            engine.shed();
                        }
                        Tier::Protect => {
                            state.protect_actions.fetch_add(1, Ordering::SeqCst);
                            engine.protect();
                            if let Some(f) = on_protect.take() {
                                f();
                            }
                            break;
                        }
                        _ => {}
                    }
                    std::thread::park_timeout(Duration::from_secs_f64(interval));
                }
            })
            .ok();
        MonitorGuard {
            stop,
            thread: Mutex::new(thread),
        }
    }
}

/// Owner of the monitor thread; dropping it stops the thread.
pub struct MonitorGuard {
    stop: Arc<AtomicBool>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl MonitorGuard {
    fn first_interval() -> f64 {
        PressureMonitor::interval_secs(Tier::Normal, 0.0)
    }

    /// Stops the thread and waits for it.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        let t = self.thread.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(t) = t {
            t.thread().unpark();
            let _ = t.join();
        }
    }
}

impl Drop for MonitorGuard {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thresholds() -> Thresholds {
        Thresholds {
            watch_below: 300,
            shed_below: 200,
            protect_below: 50,
        }
    }

    #[test]
    fn classifies_by_available_memory() {
        let t = thresholds();
        assert_eq!(t.classify(1000, false), Tier::Normal);
        assert_eq!(t.classify(250, false), Tier::Watch);
        assert_eq!(t.classify(150, false), Tier::Shed);
        assert_eq!(t.classify(10, false), Tier::Protect);
        assert_eq!(t.classify(1000, true), Tier::Normal);
        assert_eq!(t.classify(250, true), Tier::Shed);
        assert_eq!(t.classify(10, true), Tier::Protect);
    }

    #[test]
    fn derives_thresholds_from_the_reserve() {
        let host = HostMemory {
            total: 1000,
            available: 800,
            watermark: 20,
            reserve_hint: 30,
        };
        let engine = EngineHostBytes {
            logits_readback: 4,
            image_payloads: 6,
            streams: 2,
        };
        let t = Thresholds::derive(&host, &engine);
        assert_eq!(t.protect_below, 20);
        assert_eq!(t.shed_below, 200 + 20 + 30);
        assert_eq!(t.watch_below, 250 + 5);
        assert!(t.protect_below <= t.shed_below && t.shed_below <= t.watch_below);
    }

    const MIB: u64 = 1 << 20;

    /// The host of the observed run: 121.69 GiB total, 109.25 GiB available at boot.
    fn observed() -> Thresholds {
        let host = HostMemory {
            total: 124_611 * MIB,
            available: 111_872 * MIB,
            watermark: 297 * MIB,
            reserve_hint: 0,
        };
        let engine = EngineHostBytes {
            logits_readback: 6_000 * MIB,
            image_payloads: 1_096 * MIB,
            streams: 32,
        };
        Thresholds::derive(&host, &engine)
    }

    #[test]
    fn saturated_steady_state_with_resident_host_footprint_is_normal() {
        let t = observed();
        assert_eq!(t.protect_below, 297 * MIB);
        assert_eq!(t.shed_below, (12_739 + 297) * MIB);
        assert_eq!(t.watch_below, t.shed_below + (7_096 * MIB).div_ceil(32));
        for available in [19_607, 20_119, 20_460] {
            assert_eq!(t.classify(available * MIB, false), Tier::Normal);
        }
        assert!(t.watch_below < 19_607 * MIB - 5_000 * MIB);
    }

    #[test]
    fn observed_host_still_sheds_and_protects_on_real_pressure() {
        let t = observed();
        assert_eq!(t.classify(t.watch_below - 1, false), Tier::Watch);
        assert_eq!(t.classify(t.shed_below - 1, false), Tier::Shed);
        assert_eq!(t.classify(t.watch_below - 1, true), Tier::Shed);
        assert_eq!(t.classify(20_000 * MIB, true), Tier::Normal);
        assert_eq!(t.classify(t.protect_below - 1, false), Tier::Protect);
    }

    #[test]
    fn interval_follows_step_time() {
        assert_eq!(
            PressureMonitor::interval_secs(Tier::Normal, 0.0),
            UNMEASURED_INTERVAL_SECS
        );
        assert_eq!(
            PressureMonitor::interval_secs(Tier::Normal, 0.1),
            0.1 * SAMPLE_STEPS
        );
        assert_eq!(PressureMonitor::interval_secs(Tier::Shed, 0.1), 0.1);
        assert_eq!(
            PressureMonitor::interval_secs(Tier::Shed, 0.0001),
            MIN_INTERVAL_SECS
        );
    }

    fn write(dir: &std::path::Path, name: &str, text: &str) {
        std::fs::write(dir.join(name), text).unwrap();
    }

    #[test]
    fn walks_tiers_through_fake_files() {
        let dir = std::env::temp_dir().join(format!("scriptor-pressure-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let paths = HostPaths {
            meminfo: dir.join("meminfo"),
            psi: dir.join("psi"),
            vm_dir: dir.clone(),
        };
        let mem = |kb: u64| format!("MemTotal: 10000 kB\nMemAvailable: {kb} kB\n");
        let stall = |v: f64| {
            format!(
                "some avg10={v:.2} avg60=0 avg300=0 total=0\nfull avg10={v:.2} avg60=0 avg300=0 total=0\n"
            )
        };
        let t = Thresholds {
            watch_below: 300 * 1024,
            shed_below: 200 * 1024,
            protect_below: 50 * 1024,
        };
        let state = Arc::new(PressureState::default());
        let mut m = PressureMonitor::new(t, paths, Arc::clone(&state));
        write(&dir, "psi", &stall(0.0));
        write(&dir, "meminfo", &mem(5000));
        assert_eq!(m.sample(1.0), Tier::Normal);
        write(&dir, "meminfo", &mem(250));
        assert_eq!(m.sample(1.0), Tier::Watch);
        write(&dir, "meminfo", &mem(150));
        assert_eq!(m.sample(1.0), Tier::Shed);
        assert_eq!(state.snapshot().host_mem_available_bytes, 150 * 1024);
        write(&dir, "meminfo", &mem(10));
        assert_eq!(m.sample(1.0), Tier::Protect);
        assert_eq!(state.tier(), Tier::Protect);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sustained_full_stall_sheds_only_below_the_watch_level() {
        let dir =
            std::env::temp_dir().join(format!("scriptor-pressure-psi-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let paths = HostPaths {
            meminfo: dir.join("meminfo"),
            psi: dir.join("psi"),
            vm_dir: dir.clone(),
        };
        write(
            &dir,
            "meminfo",
            "MemTotal: 10000 kB\nMemAvailable: 9000 kB\n",
        );
        write(
            &dir,
            "psi",
            "some avg10=2 avg60=0 avg300=0 total=0\nfull avg10=0.50 avg60=0 avg300=0 total=0\n",
        );
        let t = Thresholds {
            watch_below: 300 * 1024,
            shed_below: 200 * 1024,
            protect_below: 50 * 1024,
        };
        let mut m = PressureMonitor::new(t, paths, Arc::new(PressureState::default()));
        assert_eq!(m.sample(PSI_WINDOW_SECS / 2.0), Tier::Normal);
        assert_eq!(m.sample(PSI_WINDOW_SECS / 2.0), Tier::Normal);
        write(
            &dir,
            "meminfo",
            "MemTotal: 10000 kB\nMemAvailable: 250 kB\n",
        );
        assert_eq!(m.sample(1.0), Tier::Shed);
        write(
            &dir,
            "psi",
            "some avg10=0 avg60=0 avg300=0 total=0\nfull avg10=0.00 avg60=0 avg300=0 total=0\n",
        );
        assert_eq!(m.sample(1.0), Tier::Watch);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unreadable_meminfo_keeps_the_tier() {
        let paths = HostPaths {
            meminfo: "/nonexistent/scriptor-meminfo".into(),
            psi: "/nonexistent/scriptor-psi".into(),
            vm_dir: "/nonexistent".into(),
        };
        let state = Arc::new(PressureState::default());
        let mut m = PressureMonitor::new(thresholds(), paths, state);
        assert_eq!(m.sample(1.0), Tier::Normal);
    }
}
