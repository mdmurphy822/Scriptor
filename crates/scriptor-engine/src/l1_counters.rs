//! Host-span and vision-flight counters of the engine loop, off by default.
//!
//! The ledger records, per device decode step, the host spans between launch return and the
//! next launch return (overlap work, the stream synchronize, the collect tail and the
//! turnaround with prefill and blocking vision waits carved out), and per vision flight the
//! launch, start wait, copy, device, poll lag and idle-gap terms. It only reads clocks and
//! writes counters; no scheduling decision depends on it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;
use std::time::Instant;

use crate::scheduler::VisionJob;

/// Environment variable that turns the counters on with `1`.
pub const L1_COUNTERS_ENV: &str = "SCRIPTOR_L1_COUNTERS";

/// Environment variable that adds one stderr line per decode step with `1`; it requires
/// [`L1_COUNTERS_ENV`].
pub const L1_STEP_LINES_ENV: &str = "SCRIPTOR_L1_STEP_LINES";

/// Environment variable that adds one stderr line per prefill pass with `1`; it requires
/// [`L1_COUNTERS_ENV`].
pub const L1_PASS_LINES_ENV: &str = "SCRIPTOR_L1_PASS_LINES";

/// Upper bounds in nanoseconds of the host-only turnaround histogram, ascending; a further
/// entry counts everything above.
pub const TURNAROUND_BOUNDS_NS: [u64; 8] = [
    100_000, 250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000,
];

/// Upper bounds in nanoseconds of the vision poll-lag histogram, ascending; a further entry
/// counts everything above.
pub const POLL_LAG_BOUNDS_NS: [u64; 8] = [
    1_000_000,
    5_000_000,
    10_000_000,
    20_000_000,
    40_000_000,
    80_000_000,
    160_000_000,
    320_000_000,
];

/// Which counter outputs are on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct L1Switches {
    /// Cumulative counters, flight lines, device events and host callbacks.
    pub counters: bool,
    /// One stderr line per decode step.
    pub step_lines: bool,
    /// One stderr line per prefill pass.
    pub pass_lines: bool,
}

fn parse_flag(var: &str, value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{var}={other:?}: expected 0 or 1")),
    }
}

/// Parses the counters and step-lines switches; see [`parse_all_switches`].
pub fn parse_switches(
    counters: Option<&str>,
    step_lines: Option<&str>,
) -> Result<L1Switches, String> {
    parse_all_switches(counters, step_lines, None)
}

/// Parses the three switches: `1` enables, `0`, empty or unset disables, anything else is an
/// error naming its variable. Checks the counters value, then the step-lines value, then the
/// pass-lines value, then that step lines and pass lines have the counters on; the first error
/// is returned.
pub fn parse_all_switches(
    counters: Option<&str>,
    step_lines: Option<&str>,
    pass_lines: Option<&str>,
) -> Result<L1Switches, String> {
    let counters = parse_flag(L1_COUNTERS_ENV, counters)?;
    let step_lines = parse_flag(L1_STEP_LINES_ENV, step_lines)?;
    let pass_lines = parse_flag(L1_PASS_LINES_ENV, pass_lines)?;
    if step_lines && !counters {
        return Err(format!(
            "{L1_STEP_LINES_ENV}=1 requires {L1_COUNTERS_ENV}=1"
        ));
    }
    if pass_lines && !counters {
        return Err(format!(
            "{L1_PASS_LINES_ENV}=1 requires {L1_COUNTERS_ENV}=1"
        ));
    }
    Ok(L1Switches {
        counters,
        step_lines,
        pass_lines,
    })
}

/// The switches from the process environment.
pub fn switches_from_env() -> Result<L1Switches, String> {
    parse_all_switches(
        std::env::var(L1_COUNTERS_ENV).ok().as_deref(),
        std::env::var(L1_STEP_LINES_ENV).ok().as_deref(),
        std::env::var(L1_PASS_LINES_ENV).ok().as_deref(),
    )
}

static EPOCH: OnceLock<Instant> = OnceLock::new();

/// Nanoseconds since the first call in this process; the zero shared by every L1 timestamp.
pub fn epoch_ns() -> u64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

/// Source of timestamps in nanoseconds.
pub trait L1Clock: Send {
    /// The current time.
    fn now_ns(&self) -> u64;
}

/// The process-epoch clock.
#[derive(Debug, Clone, Copy, Default)]
pub struct EpochClock;

impl L1Clock for EpochClock {
    fn now_ns(&self) -> u64 {
        epoch_ns()
    }
}

/// Host-side sample of one device decode step taken by the executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepHostSample {
    /// Wall nanoseconds of the stream synchronize.
    pub sync_wall_ns: u64,
    /// Thread CPU nanoseconds spent inside the stream synchronize.
    pub sync_cpu_ns: u64,
    /// CPU the engine thread ran on at launch.
    pub cpu: u32,
}

/// Device and host timing of one vision flight reported by the executor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VisionTiming {
    /// Device nanoseconds of the host-to-device copy.
    pub copy_ns: u64,
    /// Device nanoseconds of the copy plus the tower.
    pub device_ns: u64,
    /// Epoch time the flight started on the device, when host callbacks are available.
    pub start_host_ns: Option<u64>,
    /// Epoch time the flight finished on the device, when host callbacks are available.
    pub ready_host_ns: Option<u64>,
}

/// Names a flight in the ledger.
pub type FlightId = u64;

/// Counts of a bounded histogram: `counts[i]` holds values up to bound `i`, the last entry the
/// rest.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NsHistogram {
    /// Entries per bucket, the last one unbounded.
    pub counts: [u64; 9],
    /// Summed nanoseconds.
    pub sum_ns: u64,
}

impl NsHistogram {
    fn record(&mut self, bounds: &[u64; 8], ns: u64) {
        let at = bounds.iter().position(|&b| ns <= b).unwrap_or(bounds.len());
        self.counts[at] += 1;
        self.sum_ns += ns;
    }

    /// Values recorded.
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }
}

/// Cumulative L1 counters; durations are nanoseconds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct L1Stats {
    /// Device decode steps seen.
    pub steps: u64,
    /// Steps that closed a launch-to-launch window.
    pub windows: u64,
    /// Host work between launch return and collect entry.
    pub overlap_ns: u64,
    /// Stream synchronize wall time.
    pub sync_ns: u64,
    /// Thread CPU time inside the stream synchronize.
    pub sync_cpu_ns: u64,
    /// Collect time outside the synchronize.
    pub collect_tail_ns: u64,
    /// Collect return to the next launch return.
    pub turnaround_ns: u64,
    /// Prefill wall time inside turnaround windows.
    pub turnaround_prefill_ns: u64,
    /// Blocking vision wait inside turnaround windows.
    pub turnaround_vision_wait_ns: u64,
    /// Turnaround minus prefill and blocking vision wait.
    pub turnaround_host_ns: u64,
    /// Distribution of the host-only turnaround.
    pub turnaround_host: NsHistogram,
    /// Steps by CPU the engine thread launched on.
    pub engine_cpu_steps: BTreeMap<u32, u64>,
    /// Launches on a different CPU than the previous one.
    pub engine_cpu_migrations: u64,
    /// Steps that launched with a vision flight outstanding.
    pub steps_vision_busy: u64,
    /// Steps that launched with no vision flight outstanding.
    pub steps_vision_idle: u64,
    /// Synchronize wall time of steps with a vision flight outstanding.
    pub sync_vision_busy_ns: u64,
    /// Synchronize wall time of steps with no vision flight outstanding.
    pub sync_vision_idle_ns: u64,
    /// Vision flights observed.
    pub flights: u64,
    /// Patches over observed flights.
    pub flight_patches: u64,
    /// Sum over images of patches squared.
    pub flight_sum_n2: u64,
    /// Executor launch time.
    pub vision_launch_ns: u64,
    /// Launch return to the start on the device.
    pub vision_start_wait_ns: u64,
    /// Device copy time.
    pub vision_copy_ns: u64,
    /// Device copy plus compute time.
    pub vision_device_ns: u64,
    /// Device completion to the scheduler observing it.
    pub vision_poll_lag_ns: u64,
    /// Vision-stream idle between flights while images were pending.
    pub vision_gap_pending_ns: u64,
    /// Vision-stream idle between flights with nothing pending.
    pub vision_gap_starved_ns: u64,
    /// Distribution of the poll lag.
    pub vision_poll_lag: NsHistogram,
    /// Readiness polls that found the flight finished.
    pub polls_ready: u64,
    /// Readiness polls that found it running.
    pub polls_not_ready: u64,
    /// Flights observed before their device completion stamp.
    pub neg_lag: u64,
    /// Flights without host callback stamps.
    pub callbacks_missing: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct StepCur {
    n: u64,
    launch_ret: u64,
    rows: usize,
    flight: bool,
    collect_enter: u64,
    collect_ret: u64,
    sync_wall: u64,
    sync_cpu: u64,
    cpu: Option<u32>,
    sched_step: u64,
}

#[derive(Debug, Clone)]
struct FlightRec {
    begin: u64,
    returned: u64,
    images: usize,
    patches: u64,
    sum_n2: u64,
    gap: Option<u64>,
    polls: u64,
    polls_not_ready: u64,
}

#[derive(Debug)]
struct State {
    switches: L1Switches,
    stats: L1Stats,
    lines: Vec<String>,
    cur: Option<StepCur>,
    win_prefill: u64,
    win_vblock: u64,
    sched_steps: u64,
    flights: BTreeMap<FlightId, FlightRec>,
    next_flight: FlightId,
    last_ready: Option<(u64, usize)>,
    last_cpu: Option<u32>,
    passes: u64,
    seen_shapes: BTreeSet<(usize, usize)>,
}

/// The L1 ledger: every method is a no-op that touches no state and reads no clock when the
/// counters are off.
pub struct L1Ledger {
    state: Option<Box<State>>,
    clock: Box<dyn L1Clock>,
}

impl Default for L1Ledger {
    fn default() -> Self {
        Self::new(L1Switches::default())
    }
}

impl std::fmt::Debug for L1Ledger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("L1Ledger")
            .field("enabled", &self.enabled())
            .finish()
    }
}

fn us(ns: u64) -> u64 {
    ns / 1_000
}

fn ms3(ns: u64) -> String {
    format!("{}.{:03}", ns / 1_000_000, (ns / 1_000) % 1_000)
}

fn opt_us(v: Option<u64>) -> String {
    v.map_or_else(|| "-".to_string(), |ns| us(ns).to_string())
}

/// `a - b` in signed nanoseconds, saturating at the `i64` range.
fn signed_diff_ns(a: u64, b: u64) -> i64 {
    i64::try_from(i128::from(a) - i128::from(b)).unwrap_or(if a >= b { i64::MAX } else { i64::MIN })
}

/// Signed nanoseconds as truncated microseconds; `-` when absent.
fn opt_signed_us(v: Option<i64>) -> String {
    v.map_or_else(|| "-".to_string(), |ns| (ns / 1_000).to_string())
}

impl L1Ledger {
    /// A ledger recording under `switches` against the process-epoch clock.
    pub fn new(switches: L1Switches) -> Self {
        Self::with_clock(switches, Box::new(EpochClock))
    }

    /// A ledger recording under `switches` against `clock`.
    pub fn with_clock(switches: L1Switches, clock: Box<dyn L1Clock>) -> Self {
        let state = switches.counters.then(|| {
            Box::new(State {
                switches,
                stats: L1Stats::default(),
                lines: Vec::new(),
                cur: None,
                win_prefill: 0,
                win_vblock: 0,
                sched_steps: 0,
                flights: BTreeMap::new(),
                next_flight: 0,
                last_ready: None,
                last_cpu: None,
                passes: 0,
                seen_shapes: BTreeSet::new(),
            })
        });
        Self { state, clock }
    }

    /// Whether the counters are on.
    pub fn enabled(&self) -> bool {
        self.state.is_some()
    }

    /// The current time, or zero without reading the clock when the counters are off.
    pub fn now(&self) -> u64 {
        if self.state.is_some() {
            self.clock.now_ns()
        } else {
            0
        }
    }

    /// Counts one scheduler step.
    pub fn on_scheduler_step(&mut self) {
        if let Some(s) = self.state.as_mut() {
            s.sched_steps += 1;
        }
    }

    /// Records the launch return of a device decode step of `rows` sequences; closes the
    /// previous step's turnaround window.
    pub fn on_launch_return(&mut self, now: u64, rows: usize, vision_in_flight: bool) {
        let Some(s) = self.state.as_mut() else {
            return;
        };
        if let Some(prev) = s.cur.take()
            && (prev.collect_ret > 0 || prev.collect_enter > 0)
        {
            let turn = now.saturating_sub(prev.collect_ret);
            let host = turn
                .saturating_sub(s.win_prefill)
                .saturating_sub(s.win_vblock);
            let between = s.sched_steps.saturating_sub(prev.sched_step + 1);
            let st = &mut s.stats;
            st.windows += 1;
            st.turnaround_ns += turn;
            st.turnaround_prefill_ns += s.win_prefill;
            st.turnaround_vision_wait_ns += s.win_vblock;
            st.turnaround_host_ns += host;
            st.turnaround_host.record(&TURNAROUND_BOUNDS_NS, host);
            if s.switches.step_lines {
                s.lines.push(format!(
                        "scriptor: l1-step n={} t_ms={} rows={} a_us={} sync_us={} sync_cpu_us={} c_us={} turn_us={} turn_prefill_us={} turn_vblock_us={} turn_host_us={} between={} flight={} cpu={}",
                        prev.n,
                        ms3(prev.launch_ret),
                        prev.rows,
                        us(prev.collect_enter.saturating_sub(prev.launch_ret)),
                        us(prev.sync_wall),
                        us(prev.sync_cpu),
                        us(prev
                            .collect_ret
                            .saturating_sub(prev.collect_enter)
                            .saturating_sub(prev.sync_wall)),
                        us(turn),
                        us(s.win_prefill),
                        us(s.win_vblock),
                        us(host),
                        between,
                        u8::from(prev.flight),
                        prev.cpu.map_or_else(|| "-".to_string(), |c| c.to_string()),
                    ));
            }
        }
        s.win_prefill = 0;
        s.win_vblock = 0;
        s.stats.steps += 1;
        if vision_in_flight {
            s.stats.steps_vision_busy += 1;
        } else {
            s.stats.steps_vision_idle += 1;
        }
        s.cur = Some(StepCur {
            n: s.stats.steps,
            launch_ret: now,
            rows,
            flight: vision_in_flight,
            sched_step: s.sched_steps,
            ..StepCur::default()
        });
    }

    /// Records the instant the engine starts collecting the step.
    pub fn on_collect_enter(&mut self, now: u64) {
        if let Some(cur) = self.state.as_mut().and_then(|s| s.cur.as_mut()) {
            cur.collect_enter = now;
        }
    }

    /// Records the instant collecting returned, with the executor's host sample when it has
    /// one.
    pub fn on_collect_return(&mut self, now: u64, sample: Option<StepHostSample>) {
        let Some(s) = self.state.as_mut() else {
            return;
        };
        let Some(cur) = s.cur.as_mut() else {
            return;
        };
        cur.collect_ret = now;
        if let Some(h) = sample {
            cur.sync_wall = h.sync_wall_ns;
            cur.sync_cpu = h.sync_cpu_ns;
            cur.cpu = Some(h.cpu);
        }
        let overlap = cur.collect_enter.saturating_sub(cur.launch_ret);
        let tail = now
            .saturating_sub(cur.collect_enter)
            .saturating_sub(cur.sync_wall);
        let (sync, sync_cpu, busy, cpu) = (cur.sync_wall, cur.sync_cpu, cur.flight, cur.cpu);
        let st = &mut s.stats;
        st.overlap_ns += overlap;
        st.sync_ns += sync;
        st.sync_cpu_ns += sync_cpu;
        st.collect_tail_ns += tail;
        if busy {
            st.sync_vision_busy_ns += sync;
        } else {
            st.sync_vision_idle_ns += sync;
        }
        if let Some(cpu) = cpu {
            *st.engine_cpu_steps.entry(cpu).or_insert(0) += 1;
            if s.last_cpu.is_some_and(|prev| prev != cpu) {
                st.engine_cpu_migrations += 1;
            }
            s.last_cpu = Some(cpu);
        }
    }

    /// Records one prefill pass of `tokens` tokens over `rows` rows that took `wall_ns`. With
    /// the pass-lines switch on it queues one `l1-pass` line whose `shape` key is
    /// `<tokens>-<rows>` and whose `first_seen` is 1 the first time that key occurs in the
    /// ledger; otherwise it does nothing and reads no clock.
    pub fn on_prefill_pass(&mut self, wall_ns: u64, tokens: usize, rows: usize) {
        if !self.state.as_ref().is_some_and(|s| s.switches.pass_lines) {
            return;
        }
        let t = self.clock.now_ns();
        let Some(s) = self.state.as_mut() else {
            return;
        };
        s.passes += 1;
        let first = s.seen_shapes.insert((tokens, rows));
        s.lines.push(format!(
            "scriptor: l1-pass n={} t_ms={} wall_us={} tokens={} rows={} shape={}-{} first_seen={}",
            s.passes,
            ms3(t),
            us(wall_ns),
            tokens,
            rows,
            tokens,
            rows,
            u8::from(first),
        ));
    }

    /// Adds prefill wall time spent inside the current turnaround window.
    pub fn exclude_prefill(&mut self, ns: u64) {
        if let Some(s) = self.state.as_mut() {
            s.win_prefill += ns;
        }
    }

    /// Adds blocking vision wait time spent inside the current turnaround window.
    pub fn exclude_vision_block(&mut self, ns: u64) {
        if let Some(s) = self.state.as_mut() {
            s.win_vblock += ns;
        }
    }

    /// Records a flight launched over `jobs`, `begin` before and `returned` after the executor
    /// call, and returns its id.
    pub fn on_flight_begin(&mut self, begin: u64, returned: u64, jobs: &[VisionJob]) -> FlightId {
        let Some(s) = self.state.as_mut() else {
            return 0;
        };
        let id = s.next_flight;
        s.next_flight += 1;
        let patches: u64 = jobs.iter().map(|j| j.patches as u64).sum();
        let sum_n2: u64 = jobs.iter().map(|j| (j.patches as u64).pow(2)).sum();
        let gap = match s.last_ready {
            Some((ready, pending)) => {
                let gap = begin.saturating_sub(ready);
                if pending > 0 {
                    s.stats.vision_gap_pending_ns += gap;
                } else {
                    s.stats.vision_gap_starved_ns += gap;
                }
                Some(gap)
            }
            None => None,
        };
        s.stats.vision_launch_ns += returned.saturating_sub(begin);
        s.flights.insert(
            id,
            FlightRec {
                begin,
                returned,
                images: jobs.len(),
                patches,
                sum_n2,
                gap,
                polls: 0,
                polls_not_ready: 0,
            },
        );
        id
    }

    /// Records one readiness poll of flight `id`.
    pub fn on_poll(&mut self, id: FlightId, ready: bool) {
        let Some(s) = self.state.as_mut() else {
            return;
        };
        if let Some(f) = s.flights.get_mut(&id) {
            f.polls += 1;
            if ready {
                s.stats.polls_ready += 1;
            } else {
                f.polls_not_ready += 1;
                s.stats.polls_not_ready += 1;
            }
        }
    }

    /// Records flight `id` observed finished at `now` with the executor's `timing`;
    /// `pending_now` is the number of images still unresolved.
    pub fn on_flight_observed(
        &mut self,
        id: FlightId,
        now: u64,
        timing: Option<VisionTiming>,
        pending_now: usize,
    ) {
        let Some(s) = self.state.as_mut() else {
            return;
        };
        let Some(f) = s.flights.remove(&id) else {
            return;
        };
        let counter = now.saturating_sub(f.returned);
        let t = timing.unwrap_or_default();
        let stamps = match (timing, t.start_host_ns, t.ready_host_ns) {
            (Some(_), Some(start), Some(ready)) => Some((start, ready)),
            _ => None,
        };
        let (start_wait, lag, estimated) = match stamps {
            Some((start, ready)) => {
                if now < ready {
                    s.stats.neg_lag += 1;
                }
                (
                    Some(start.saturating_sub(f.returned)),
                    now.saturating_sub(ready),
                    false,
                )
            }
            None => {
                s.stats.callbacks_missing += 1;
                (None, counter.saturating_sub(t.device_ns), true)
            }
        };
        let start_rel = stamps.map(|(start, _)| signed_diff_ns(start, f.returned));
        let cb_lat = stamps.map(|(start, ready)| {
            signed_diff_ns(ready, start)
                .saturating_sub(i64::try_from(t.device_ns).unwrap_or(i64::MAX))
        });
        let st = &mut s.stats;
        st.flights += 1;
        st.flight_patches += f.patches;
        st.flight_sum_n2 += f.sum_n2;
        st.vision_start_wait_ns += start_wait.unwrap_or(0);
        st.vision_copy_ns += t.copy_ns;
        st.vision_device_ns += t.device_ns;
        st.vision_poll_lag_ns += lag;
        st.vision_poll_lag.record(&POLL_LAG_BOUNDS_NS, lag);
        s.last_ready = stamps.map(|(_, ready)| (ready, pending_now));
        s.lines.push(format!(
            "scriptor: l1-flight id={} t_begin_ms={} images={} frames={} patches={} sum_n2={} launch_us={} start_wait_us={} copy_us={} device_us={} lag_us={} counter_us={} polls={} polls_not_ready={} gap_us={} pending_after={} cb={} start_rel_us={} cb_lat_us={}{}",
            id,
            ms3(f.begin),
            f.images,
            f.images,
            f.patches,
            f.sum_n2,
            us(f.returned.saturating_sub(f.begin)),
            opt_us(start_wait),
            us(t.copy_ns),
            us(t.device_ns),
            us(lag),
            us(counter),
            f.polls,
            f.polls_not_ready,
            opt_us(f.gap),
            pending_now,
            u8::from(!estimated),
            opt_signed_us(start_rel),
            opt_signed_us(cb_lat),
            if estimated { " lag=est" } else { "" },
        ));
    }

    /// The cumulative counters; `None` when off.
    pub fn stats(&self) -> Option<L1Stats> {
        self.state.as_ref().map(|s| s.stats.clone())
    }

    /// Takes the lines produced since the last call.
    pub fn drain_lines(&mut self) -> Vec<String> {
        match self.state.as_mut() {
            Some(s) => std::mem::take(&mut s.lines),
            None => Vec::new(),
        }
    }

    #[cfg(test)]
    fn capacity_probe(&self) -> usize {
        self.state.as_ref().map_or(0, |s| s.lines.capacity())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    struct PanicClock;

    impl L1Clock for PanicClock {
        fn now_ns(&self) -> u64 {
            panic!("a disabled ledger must not read the clock");
        }
    }

    fn on() -> L1Ledger {
        L1Ledger::new(L1Switches {
            counters: true,
            step_lines: true,
            pass_lines: false,
        })
    }

    fn jobs(patches: &[usize]) -> Vec<VisionJob> {
        patches
            .iter()
            .enumerate()
            .map(|(i, &p)| VisionJob {
                seq: 1,
                image: i,
                key: scriptor_cache::EmbeddingKey([0; 32]),
                grid: scriptor_cache::ImageGrid { t: 1, h: 1, w: 1 },
                patches: p,
                payload: Arc::new(Vec::new()),
            })
            .collect()
    }

    #[test]
    fn switches_default_off_and_accept_zero_one_empty() {
        assert_eq!(parse_switches(None, None), Ok(L1Switches::default()));
        assert_eq!(
            parse_switches(Some(""), Some("")),
            Ok(L1Switches::default())
        );
        assert_eq!(
            parse_switches(Some("0"), Some("0")),
            Ok(L1Switches::default())
        );
        let on = parse_switches(Some("1"), None).unwrap();
        assert!(on.counters && !on.step_lines);
        let both = parse_switches(Some("1"), Some("1")).unwrap();
        assert!(both.counters && both.step_lines);
    }

    #[test]
    fn switches_reject_other_values_naming_the_variable() {
        for bad in ["yes", "true", "2", " 1", "01"] {
            let e = parse_switches(Some(bad), None).unwrap_err();
            assert!(e.contains(L1_COUNTERS_ENV), "{e}");
            let e = parse_switches(Some("1"), Some(bad)).unwrap_err();
            assert!(e.contains(L1_STEP_LINES_ENV), "{e}");
        }
    }

    #[test]
    fn step_lines_require_the_master_switch() {
        let e = parse_switches(None, Some("1")).unwrap_err();
        assert!(
            e.contains(L1_STEP_LINES_ENV) && e.contains("requires"),
            "{e}"
        );
        assert!(parse_switches(Some("0"), Some("1")).is_err());
        assert!(parse_switches(None, Some("0")).is_ok());
    }

    #[test]
    fn golden_flight_line() {
        let mut l = on();
        let j = jobs(&[1200, 640, 2300, 40, 9448, 624]);
        let id = l.on_flight_begin(1_234_567_000, 1_234_879_000, &j);
        l.on_poll(id, false);
        l.on_poll(id, true);
        let timing = VisionTiming {
            copy_ns: 3_100_000,
            device_ns: 262_400_000,
            start_host_ns: Some(1_276_079_000),
            ready_host_ns: Some(1_538_479_000),
        };
        l.on_flight_observed(id, 1_566_579_000, Some(timing), 3);
        assert_eq!(
            l.drain_lines(),
            vec![
                "scriptor: l1-flight id=0 t_begin_ms=1234.567 images=6 frames=6 patches=14252 sum_n2=96795280 launch_us=312 start_wait_us=41200 copy_us=3100 device_us=262400 lag_us=28100 counter_us=331700 polls=2 polls_not_ready=1 gap_us=- pending_after=3 cb=1 start_rel_us=41200 cb_lat_us=0"
            ]
        );
    }

    fn three_steps(l: &mut L1Ledger) {
        l.on_launch_return(1_000_000, 24, true);
        l.on_collect_enter(3_100_000);
        l.on_collect_return(
            48_200_000,
            Some(StepHostSample {
                sync_wall_ns: 45_010_000,
                sync_cpu_ns: 44_980_000,
                cpu: 17,
            }),
        );
        l.exclude_prefill(500_000);
        l.exclude_vision_block(200_000);
        l.on_launch_return(50_000_000, 24, false);
        l.on_collect_enter(52_000_000);
        l.on_collect_return(
            90_000_000,
            Some(StepHostSample {
                sync_wall_ns: 37_000_000,
                sync_cpu_ns: 36_000_000,
                cpu: 3,
            }),
        );
        l.on_launch_return(91_500_000, 23, false);
    }

    #[test]
    fn golden_step_line() {
        let mut l = on();
        three_steps(&mut l);
        let lines = l.drain_lines();
        assert_eq!(
            lines[0],
            "scriptor: l1-step n=1 t_ms=1.000 rows=24 a_us=2100 sync_us=45010 sync_cpu_us=44980 c_us=90 turn_us=1800 turn_prefill_us=500 turn_vblock_us=200 turn_host_us=1100 between=0 flight=1 cpu=17"
        );
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn step_spans_partition_launch_to_launch() {
        let mut l = on();
        three_steps(&mut l);
        let st = l.stats().unwrap();
        assert_eq!(st.steps, 3);
        assert_eq!(st.windows, 2);
        let total = st.overlap_ns + st.sync_ns + st.collect_tail_ns + st.turnaround_ns;
        assert_eq!(total, 91_500_000 - 1_000_000);
        assert_eq!(
            st.turnaround_host_ns,
            st.turnaround_ns - st.turnaround_prefill_ns - st.turnaround_vision_wait_ns
        );
        assert_eq!(st.turnaround_host.total(), 2);
        assert_eq!(st.engine_cpu_steps.get(&17), Some(&1));
        assert_eq!(st.engine_cpu_migrations, 1);
        assert_eq!((st.steps_vision_busy, st.steps_vision_idle), (1, 2));
        assert_eq!(st.sync_vision_busy_ns, 45_010_000);
    }

    #[test]
    fn flight_decomposition_closes_and_flags_negative_lag() {
        let mut l = on();
        let j = jobs(&[100]);
        let id = l.on_flight_begin(0, 1_000_000, &j);
        let timing = VisionTiming {
            copy_ns: 1_000_000,
            device_ns: 9_000_000,
            start_host_ns: Some(5_000_000),
            ready_host_ns: Some(14_000_000),
        };
        l.on_flight_observed(id, 20_000_000, Some(timing), 0);
        let st = l.stats().unwrap();
        assert_eq!(st.vision_start_wait_ns, 4_000_000);
        assert_eq!(st.vision_poll_lag_ns, 6_000_000);
        assert_eq!(
            st.vision_start_wait_ns + 9_000_000 + st.vision_poll_lag_ns,
            20_000_000 - 1_000_000
        );
        assert_eq!(st.neg_lag, 0);
        let id = l.on_flight_begin(30_000_000, 31_000_000, &j);
        let early = VisionTiming {
            ready_host_ns: Some(50_000_000),
            start_host_ns: Some(32_000_000),
            ..VisionTiming::default()
        };
        l.on_flight_observed(id, 40_000_000, Some(early), 0);
        let st = l.stats().unwrap();
        assert_eq!(st.neg_lag, 1);
        assert_eq!(st.vision_poll_lag_ns, 6_000_000);
    }

    fn stamped(start: u64, ready: u64, device_ns: u64) -> VisionTiming {
        VisionTiming {
            copy_ns: 0,
            device_ns,
            start_host_ns: Some(start),
            ready_host_ns: Some(ready),
        }
    }

    fn field(line: &str, key: &str) -> i64 {
        line.split_whitespace()
            .find_map(|t| t.strip_prefix(key))
            .unwrap_or_else(|| panic!("{key} missing in {line}"))
            .parse()
            .unwrap()
    }

    #[test]
    fn cb_lat_keeps_the_sign_when_start_precedes_return() {
        let mut l = on();
        let j = jobs(&[10]);
        let id = l.on_flight_begin(0, 1_000_000, &j);
        l.on_flight_observed(
            id,
            6_000_000,
            Some(stamped(200_000, 5_200_000, 5_000_000)),
            0,
        );
        let line = l.drain_lines().remove(0);
        assert_eq!(field(&line, "start_rel_us="), -800);
        assert_eq!(field(&line, "cb_lat_us="), 0);
        assert_eq!(field(&line, "start_wait_us="), 0);
        assert_eq!(
            field(&line, "counter_us="),
            field(&line, "start_rel_us=")
                + field(&line, "device_us=")
                + field(&line, "cb_lat_us=")
                + field(&line, "lag_us=")
        );
    }

    #[test]
    fn cb_lat_negative_when_device_exceeds_host_span() {
        let mut l = on();
        let j = jobs(&[10]);
        let id = l.on_flight_begin(0, 1_000_000, &j);
        l.on_flight_observed(
            id,
            6_000_000,
            Some(stamped(200_000, 5_200_000, 5_300_000)),
            0,
        );
        let line = l.drain_lines().remove(0);
        assert_eq!(field(&line, "cb_lat_us="), -300);
        assert_eq!(field(&line, "start_rel_us="), -800);
        assert_eq!(field(&line, "start_wait_us="), 0);
    }

    #[test]
    fn gap_is_attributed_by_pending_after() {
        let mut l = on();
        let j = jobs(&[10]);
        let stamps = |start, ready| VisionTiming {
            start_host_ns: Some(start),
            ready_host_ns: Some(ready),
            ..VisionTiming::default()
        };
        let a = l.on_flight_begin(0, 1, &j);
        l.on_flight_observed(a, 100, Some(stamps(2, 90)), 4);
        let b = l.on_flight_begin(130, 131, &j);
        l.on_flight_observed(b, 300, Some(stamps(132, 250)), 0);
        let c = l.on_flight_begin(280, 281, &j);
        l.on_flight_observed(c, 400, Some(stamps(282, 390)), 0);
        let st = l.stats().unwrap();
        assert_eq!(st.vision_gap_pending_ns, 40);
        assert_eq!(st.vision_gap_starved_ns, 30);
    }

    #[test]
    fn callbacks_missing_uses_the_lower_bound_lag() {
        let mut l = on();
        let j = jobs(&[10]);
        let id = l.on_flight_begin(0, 1_000_000, &j);
        let timing = VisionTiming {
            copy_ns: 0,
            device_ns: 6_000_000,
            ..VisionTiming::default()
        };
        l.on_flight_observed(id, 10_000_000, Some(timing), 0);
        let st = l.stats().unwrap();
        assert_eq!(st.callbacks_missing, 1);
        assert_eq!(st.vision_poll_lag_ns, 3_000_000);
        let line = l.drain_lines().remove(0);
        assert!(
            line.contains("start_wait_us=- ")
                && line.ends_with("cb=0 start_rel_us=- cb_lat_us=- lag=est"),
            "{line}"
        );
    }

    #[test]
    fn a_disabled_ledger_is_zero_cost() {
        let mut l = L1Ledger::with_clock(L1Switches::default(), Box::new(PanicClock));
        assert_eq!(l.now(), 0);
        l.on_scheduler_step();
        l.on_launch_return(1, 8, true);
        l.on_collect_enter(2);
        l.on_collect_return(3, None);
        l.exclude_prefill(4);
        l.exclude_vision_block(5);
        let id = l.on_flight_begin(0, 1, &jobs(&[1]));
        l.on_poll(id, true);
        l.on_flight_observed(id, 9, None, 0);
        assert!(!l.enabled());
        assert!(l.stats().is_none());
        assert!(l.drain_lines().is_empty());
        assert_eq!(l.capacity_probe(), 0);
    }

    struct FixedClock(u64);

    impl L1Clock for FixedClock {
        fn now_ns(&self) -> u64 {
            self.0
        }
    }

    fn pass_ledger(counters: bool, pass_lines: bool) -> L1Ledger {
        L1Ledger::with_clock(
            L1Switches {
                counters,
                step_lines: false,
                pass_lines,
            },
            Box::new(FixedClock(12_345_678)),
        )
    }

    #[test]
    fn golden_pass_line() {
        let mut l = pass_ledger(true, true);
        l.on_prefill_pass(2_345_678, 1536, 3);
        assert_eq!(
            l.drain_lines(),
            vec![
                "scriptor: l1-pass n=1 t_ms=12.345 wall_us=2345 tokens=1536 rows=3 shape=1536-3 first_seen=1"
                    .to_string()
            ]
        );
    }

    #[test]
    fn pass_first_seen_flips_once_per_shape_key() {
        let mut l = pass_ledger(true, true);
        l.on_prefill_pass(1_000, 512, 2);
        l.on_prefill_pass(1_000, 512, 2);
        l.on_prefill_pass(1_000, 512, 3);
        l.on_prefill_pass(1_000, 513, 2);
        l.on_prefill_pass(1_000, 512, 2);
        let flags: Vec<&str> = l
            .drain_lines()
            .iter()
            .map(|line| line.rsplit(' ').next().unwrap().to_string())
            .map(|f| if f == "first_seen=1" { "1" } else { "0" })
            .collect();
        assert_eq!(flags, ["1", "0", "1", "1", "0"]);
    }

    #[test]
    fn pass_lines_are_off_by_default() {
        let parsed = parse_all_switches(Some("1"), None, None).unwrap();
        assert!(!parsed.pass_lines);
        assert_eq!(
            parse_all_switches(None, None, None),
            Ok(L1Switches::default())
        );
        let mut l = pass_ledger(true, false);
        l.on_prefill_pass(1_000, 512, 2);
        assert!(l.drain_lines().is_empty());
    }

    #[test]
    fn pass_lines_require_the_master_switch() {
        let e = parse_all_switches(None, None, Some("1")).unwrap_err();
        assert!(
            e.contains(L1_PASS_LINES_ENV) && e.contains("requires"),
            "{e}"
        );
        assert!(parse_all_switches(Some("0"), None, Some("1")).is_err());
        assert!(parse_all_switches(None, None, Some("0")).is_ok());
        let e = parse_all_switches(Some("1"), None, Some("yes")).unwrap_err();
        assert!(e.contains(L1_PASS_LINES_ENV), "{e}");
        let mut l = pass_ledger(false, true);
        l.on_prefill_pass(1_000, 512, 2);
        assert!(l.drain_lines().is_empty());
    }

    #[test]
    fn histograms_use_the_stated_bounds() {
        let mut h = NsHistogram::default();
        h.record(&POLL_LAG_BOUNDS_NS, 1_000_000);
        h.record(&POLL_LAG_BOUNDS_NS, 1_000_001);
        h.record(&POLL_LAG_BOUNDS_NS, 320_000_001);
        assert_eq!(h.counts[0], 1);
        assert_eq!(h.counts[1], 1);
        assert_eq!(h.counts[8], 1);
        let mut t = NsHistogram::default();
        t.record(&TURNAROUND_BOUNDS_NS, 100_000);
        assert_eq!(t.counts[0], 1);
    }
}
