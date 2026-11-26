//! Boot stage names, their order, and the shared status record that readiness and metrics read.

use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Boot stages in execution order.
pub const STAGES: [&str; 7] = [
    "detect",
    "plan",
    "weights",
    "kv_pool",
    "caches",
    "graphs",
    "self_test",
];

/// Progress of one stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageState {
    /// Not started.
    Pending,
    /// Running now.
    Running,
    /// Completed.
    Done,
    /// Deliberately not run; counts as complete for readiness.
    Skipped,
}

/// One stage with its state, wall time and a short note.
#[derive(Debug, Clone, PartialEq)]
pub struct StageRecord {
    pub name: &'static str,
    pub state: StageState,
    pub seconds: f64,
    pub note: String,
}

/// Cloneable record of boot progress shared between the boot thread, the health endpoint and
/// the metrics endpoint.
#[derive(Debug, Clone)]
pub struct BootStatus {
    inner: Arc<Mutex<Vec<StageRecord>>>,
}

impl Default for BootStatus {
    fn default() -> Self {
        Self::new()
    }
}

impl BootStatus {
    /// A status with every stage pending.
    pub fn new() -> Self {
        let records = STAGES
            .iter()
            .map(|&name| StageRecord {
                name,
                state: StageState::Pending,
                seconds: 0.0,
                note: String::new(),
            })
            .collect();
        Self {
            inner: Arc::new(Mutex::new(records)),
        }
    }

    fn update(&self, name: &str, f: impl FnOnce(&mut StageRecord)) {
        let mut guard = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(r) = guard.iter_mut().find(|r| r.name == name) {
            f(r);
        }
    }

    /// Marks a stage running.
    pub fn begin(&self, name: &str) {
        self.update(name, |r| r.state = StageState::Running);
    }

    /// Marks a stage complete with its wall time.
    pub fn done(&self, name: &str, took: Duration, note: &str) {
        self.update(name, |r| {
            r.state = StageState::Done;
            r.seconds = took.as_secs_f64();
            r.note = note.to_owned();
        });
    }

    /// Marks a stage skipped.
    pub fn skipped(&self, name: &str, note: &str) {
        self.update(name, |r| {
            r.state = StageState::Skipped;
            r.note = note.to_owned();
        });
    }

    /// Copy of every record in stage order.
    pub fn snapshot(&self) -> Vec<StageRecord> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Whether every stage is done or skipped.
    pub fn is_ready(&self) -> bool {
        self.snapshot()
            .iter()
            .all(|r| matches!(r.state, StageState::Done | StageState::Skipped))
    }
}
