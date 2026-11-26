//! Opt-in telemetry of the sequences each device-resident decode step serves: exact histograms
//! of the active-sequence count and of the executed batch rows, and a bounded ring of per-step
//! samples against a monotonic clock whose wallclock origin is recorded. Sample timestamps are
//! the host time the step was submitted (launched), not its completion or device duration.

use std::fmt::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Environment variable that turns the telemetry on when set to `1`.
pub const ACTIVE_TELEMETRY_ENV: &str = "SCRIPTOR_ACTIVE_TELEMETRY";

/// Samples the ring holds before it overwrites its oldest.
pub const RING_CAPACITY: usize = 1 << 16;

/// Counts the histograms keep exactly (zero through `COUNT_BINS - 1`); larger counts are
/// tallied separately as overflow.
pub const COUNT_BINS: usize = 1025;

/// One decode step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepSample {
    /// Nanoseconds since the telemetry origin when the host submitted (launched) the step.
    pub nanos: u64,
    /// Distinct live sequences the step served.
    pub active: u32,
    /// Batch rows the step executed.
    pub rows: u32,
}

/// Histograms, extrema and the sample ring of the recorded steps.
#[derive(Debug, Clone)]
pub struct ActiveTelemetry {
    active_hist: Vec<u64>,
    rows_hist: Vec<u64>,
    active_overflow: u64,
    rows_overflow: u64,
    retained: usize,
    steps: u64,
    min_active: u64,
    max_active: u64,
    min_rows: u64,
    max_rows: u64,
    ring: Vec<StepSample>,
    capacity: usize,
    next: usize,
    dropped: u64,
    origin: Instant,
    origin_unix_nanos: u64,
}

impl ActiveTelemetry {
    /// An empty telemetry whose ring holds `capacity` samples (at least one); the monotonic
    /// origin and its wallclock reading are taken now.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let origin_unix_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Self {
            active_hist: vec![0; COUNT_BINS],
            rows_hist: vec![0; COUNT_BINS],
            active_overflow: 0,
            rows_overflow: 0,
            retained: 0,
            steps: 0,
            min_active: u64::MAX,
            max_active: 0,
            min_rows: u64::MAX,
            max_rows: 0,
            ring: Vec::with_capacity(capacity),
            capacity,
            next: 0,
            dropped: 0,
            origin: Instant::now(),
            origin_unix_nanos,
        }
    }

    /// Records one step now.
    pub fn record(&mut self, active: usize, rows: usize) {
        let nanos = self.origin.elapsed().as_nanos() as u64;
        self.record_at(nanos, active, rows);
    }

    /// Records one step at `nanos` after the origin.
    pub fn record_at(&mut self, nanos: u64, active: usize, rows: usize) {
        self.steps += 1;
        match self.active_hist.get_mut(active) {
            Some(n) => *n += 1,
            None => self.active_overflow += 1,
        }
        match self.rows_hist.get_mut(rows) {
            Some(n) => *n += 1,
            None => self.rows_overflow += 1,
        }
        let (a, r) = (active as u64, rows as u64);
        self.min_active = self.min_active.min(a);
        self.max_active = self.max_active.max(a);
        self.min_rows = self.min_rows.min(r);
        self.max_rows = self.max_rows.max(r);
        let sample = StepSample {
            nanos,
            active: active.min(u32::MAX as usize) as u32,
            rows: rows.min(u32::MAX as usize) as u32,
        };
        if self.ring.len() < self.capacity {
            self.ring.push(sample);
            self.retained = self.ring.len();
        } else {
            self.ring[self.next] = sample;
            self.dropped += 1;
        }
        self.next = (self.next + 1) % self.capacity;
    }

    /// Steps recorded.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Samples overwritten because the ring was full.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Steps that served exactly `count` sequences; zero for counts beyond the exact bins.
    pub fn active_steps(&self, count: usize) -> u64 {
        self.active_hist.get(count).copied().unwrap_or(0)
    }

    /// Steps that executed exactly `rows` batch rows; zero for counts beyond the exact bins.
    pub fn row_steps(&self, rows: usize) -> u64 {
        self.rows_hist.get(rows).copied().unwrap_or(0)
    }

    /// Steps that served more than `COUNT_BINS - 1` sequences.
    pub fn active_overflow(&self) -> u64 {
        self.active_overflow
    }

    /// Steps that executed more than `COUNT_BINS - 1` rows.
    pub fn row_overflow(&self) -> u64 {
        self.rows_overflow
    }

    /// Copy of the counters, histograms and clock origin without the sample ring.
    pub fn counters(&self) -> Self {
        Self {
            active_hist: self.active_hist.clone(),
            rows_hist: self.rows_hist.clone(),
            active_overflow: self.active_overflow,
            rows_overflow: self.rows_overflow,
            retained: self.retained,
            steps: self.steps,
            min_active: self.min_active,
            max_active: self.max_active,
            min_rows: self.min_rows,
            max_rows: self.max_rows,
            ring: Vec::new(),
            capacity: self.capacity,
            next: self.next,
            dropped: self.dropped,
            origin: self.origin,
            origin_unix_nanos: self.origin_unix_nanos,
        }
    }

    /// Smallest and largest active count seen; `None` before the first step.
    pub fn active_range(&self) -> Option<(u64, u64)> {
        (self.steps > 0).then_some((self.min_active, self.max_active))
    }

    /// Smallest and largest executed row count seen; `None` before the first step.
    pub fn row_range(&self) -> Option<(u64, u64)> {
        (self.steps > 0).then_some((self.min_rows, self.max_rows))
    }

    /// The retained samples, oldest first.
    pub fn samples(&self) -> Vec<StepSample> {
        if self.ring.len() < self.capacity {
            return self.ring.clone();
        }
        let (newer, older) = self.ring.split_at(self.next);
        older.iter().chain(newer).copied().collect()
    }

    /// Prometheus text: the two histograms as per-value step counters (non-empty values
    /// only, with steps beyond the exact bins under the label value `overflow`), the extrema,
    /// and the ring's sample and dropped counts, under `labels`. Sample timestamps are host
    /// submission times; the counters carry no timing.
    pub fn render_metrics(&self, out: &mut String, labels: &str) {
        let hist = |out: &mut String, name: &str, label: &str, bins: &[u64], overflow: u64| {
            let _ = writeln!(out, "# TYPE {name} counter");
            for (value, &n) in bins.iter().enumerate().filter(|(_, n)| **n > 0) {
                let _ = writeln!(out, "{name}{{{labels},{label}=\"{value}\"}} {n}");
            }
            if overflow > 0 {
                let _ = writeln!(out, "{name}{{{labels},{label}=\"overflow\"}} {overflow}");
            }
        };
        hist(
            out,
            "scriptor_decode_active_count_steps_total",
            "active",
            &self.active_hist,
            self.active_overflow,
        );
        hist(
            out,
            "scriptor_decode_executed_rows_steps_total",
            "rows",
            &self.rows_hist,
            self.rows_overflow,
        );
        let (amin, amax) = self.active_range().unwrap_or((0, 0));
        let (rmin, rmax) = self.row_range().unwrap_or((0, 0));
        for (name, kind, value) in [
            ("scriptor_decode_active_count_min", "gauge", amin),
            ("scriptor_decode_active_count_max", "gauge", amax),
            ("scriptor_decode_executed_rows_min", "gauge", rmin),
            ("scriptor_decode_executed_rows_max", "gauge", rmax),
            (
                "scriptor_decode_step_ring_samples",
                "gauge",
                self.retained as u64,
            ),
            (
                "scriptor_decode_step_ring_dropped_total",
                "counter",
                self.dropped,
            ),
        ] {
            let _ = writeln!(out, "# TYPE {name} {kind}");
            let _ = writeln!(out, "{name}{{{labels}}} {value}");
        }
    }

    /// The full state as JSON: clock origin, ring bounds, both histograms keyed by value, and
    /// the retained samples oldest first as `[nanos, active, rows]`. `first_index` is the
    /// zero-based step number of the oldest retained sample.
    pub fn dump_json(&self) -> String {
        let mut out = String::new();
        let _ = write!(
            out,
            "{{\"clock\":\"monotonic_ns_since_origin_at_host_submission\",\"origin_unix_ns\":{},\
             \"capacity\":{},\"steps\":{},\"dropped\":{},\"first_index\":{},",
            self.origin_unix_nanos, self.capacity, self.steps, self.dropped, self.dropped
        );
        let (amin, amax) = self.active_range().unwrap_or((0, 0));
        let (rmin, rmax) = self.row_range().unwrap_or((0, 0));
        let _ = write!(
            out,
            "\"active_min\":{amin},\"active_max\":{amax},\"rows_min\":{rmin},\"rows_max\":{rmax},"
        );
        for (key, bins, overflow) in [
            ("active_hist", &self.active_hist, self.active_overflow),
            ("rows_hist", &self.rows_hist, self.rows_overflow),
        ] {
            let _ = write!(out, "\"{key}\":{{");
            let mut first = true;
            for (value, &n) in bins.iter().enumerate().filter(|(_, n)| **n > 0) {
                let _ = write!(out, "{}\"{value}\":{n}", if first { "" } else { "," });
                first = false;
            }
            if overflow > 0 {
                let _ = write!(
                    out,
                    "{}\"overflow\":{overflow}",
                    if first { "" } else { "," }
                );
            }
            out.push_str("},");
        }
        out.push_str("\"samples\":[");
        for (i, s) in self.samples().iter().enumerate() {
            let _ = write!(
                out,
                "{}[{},{},{}]",
                if i == 0 { "" } else { "," },
                s.nanos,
                s.active,
                s.rows
            );
        }
        out.push_str("]}");
        out
    }
}

static GLOBAL: OnceLock<Mutex<ActiveTelemetry>> = OnceLock::new();

/// Whether the environment asks for the telemetry; creates the process-wide instance when it
/// does and returns whether recording is on.
pub fn enable_from_env() -> bool {
    if std::env::var(ACTIVE_TELEMETRY_ENV).as_deref() == Ok("1") {
        GLOBAL.get_or_init(|| Mutex::new(ActiveTelemetry::new(RING_CAPACITY)));
        true
    } else {
        false
    }
}

/// Records one step into the process-wide instance; does nothing while it is off.
pub fn record_step(active: usize, rows: usize) {
    if let Some(t) = GLOBAL.get() {
        t.lock()
            .unwrap_or_else(|e| e.into_inner())
            .record(active, rows);
    }
}

/// A copy of the process-wide counters without the sample ring, for regular scrapes; `None`
/// while the telemetry is off.
pub fn counters_snapshot() -> Option<ActiveTelemetry> {
    GLOBAL
        .get()
        .map(|t| t.lock().unwrap_or_else(|e| e.into_inner()).counters())
}

/// A copy of the process-wide telemetry including the sample ring, taken under the lock for
/// serialising outside it; `None` while it is off.
pub fn full_snapshot() -> Option<ActiveTelemetry> {
    GLOBAL
        .get()
        .map(|t| t.lock().unwrap_or_else(|e| e.into_inner()).clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histograms_count_exact_values_and_extrema() {
        let mut t = ActiveTelemetry::new(8);
        assert_eq!(t.active_range(), None);
        for (a, r) in [(3, 4), (3, 4), (1, 1), (7, 8)] {
            t.record_at(0, a, r);
        }
        assert_eq!(t.steps(), 4);
        assert_eq!(
            (t.active_steps(3), t.active_steps(1), t.active_steps(7)),
            (2, 1, 1)
        );
        assert_eq!((t.row_steps(4), t.row_steps(8), t.row_steps(2)), (2, 1, 0));
        assert_eq!(t.active_range(), Some((1, 7)));
        assert_eq!(t.row_range(), Some((1, 8)));
    }

    #[test]
    fn counts_beyond_the_bins_export_as_overflow_never_as_exact() {
        let mut t = ActiveTelemetry::new(2);
        t.record_at(0, 5000, 5000);
        t.record_at(0, COUNT_BINS - 1, COUNT_BINS - 1);
        assert_eq!(t.active_steps(COUNT_BINS - 1), 1);
        assert_eq!((t.active_overflow(), t.row_overflow()), (1, 1));
        assert_eq!(t.active_range(), Some((COUNT_BINS as u64 - 1, 5000)));
        let mut out = String::new();
        t.render_metrics(&mut out, "m=\"x\"");
        assert!(out.contains("active=\"overflow\"} 1\n"));
        assert!(out.contains("rows=\"overflow\"} 1\n"));
        assert!(out.contains("active=\"1024\"} 1\n"));
        let json = t.dump_json();
        assert!(json.contains("\"overflow\":1"));
        assert!(json.contains("\"1024\":1"));
    }

    #[test]
    fn counters_copy_drops_the_ring_but_keeps_every_count() {
        let mut t = ActiveTelemetry::new(2);
        for i in 0..5 {
            t.record_at(i, 3, 4);
        }
        let c = t.counters();
        assert!(c.samples().is_empty());
        assert_eq!((c.steps(), c.dropped(), c.active_steps(3)), (5, 3, 5));
        let (mut a, mut b) = (String::new(), String::new());
        t.render_metrics(&mut a, "m=\"x\"");
        c.render_metrics(&mut b, "m=\"x\"");
        assert_eq!(a, b);
        assert!(b.contains("scriptor_decode_step_ring_samples{m=\"x\"} 2\n"));
    }

    #[test]
    fn full_ring_overwrites_oldest_and_counts_drops() {
        let mut t = ActiveTelemetry::new(3);
        for i in 0..5u64 {
            t.record_at(i * 10, i as usize, i as usize + 1);
        }
        assert_eq!(t.dropped(), 2);
        let s = t.samples();
        assert_eq!(
            s.iter().map(|x| x.nanos).collect::<Vec<_>>(),
            vec![20, 30, 40]
        );
        assert_eq!(s[0].active, 2);
        assert_eq!(t.steps(), 5);
        assert!(t.dump_json().contains("\"first_index\":2"));
    }

    #[test]
    fn partial_ring_keeps_insertion_order_without_drops() {
        let mut t = ActiveTelemetry::new(4);
        t.record_at(1, 1, 1);
        t.record_at(2, 2, 2);
        assert_eq!(t.dropped(), 0);
        assert_eq!(t.samples().len(), 2);
        assert_eq!(t.samples()[1].nanos, 2);
    }

    #[test]
    fn metrics_and_dump_render_the_recorded_values() {
        let mut t = ActiveTelemetry::new(4);
        t.record_at(5, 2, 4);
        t.record_at(9, 2, 4);
        let mut out = String::new();
        t.render_metrics(&mut out, "m=\"x\"");
        assert!(out.contains("scriptor_decode_active_count_steps_total{m=\"x\",active=\"2\"} 2\n"));
        assert!(out.contains("scriptor_decode_executed_rows_steps_total{m=\"x\",rows=\"4\"} 2\n"));
        assert!(out.contains("scriptor_decode_active_count_max{m=\"x\"} 2\n"));
        assert!(out.contains("scriptor_decode_step_ring_dropped_total{m=\"x\"} 0\n"));
        let json = t.dump_json();
        assert!(json.contains("\"active_hist\":{\"2\":2}"));
        assert!(json.contains("\"samples\":[[5,2,4],[9,2,4]]"));
        assert!(json.contains("\"origin_unix_ns\":"));
    }

    #[test]
    fn off_path_records_nothing_and_has_no_snapshot() {
        // The process-wide instance is only created by `enable_from_env`, which this binary
        // never calls with the variable set.
        record_step(3, 4);
        assert!(counters_snapshot().is_none() && full_snapshot().is_none());
    }
}
