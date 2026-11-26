//! How requests to the recognition seat are paced: the jitter of a retry
//! wait, the stop signal that cuts a wait short, and the per-seat limit on
//! concurrent requests that shrinks when the seat answers 429 and regrows on
//! success.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The longest a backoff wait goes without re-reading its stop flag when no
/// stopper signals it.
const STOP_RECHECK: Duration = Duration::from_secs(1);

/// A source of numbers in `[0, 1)` for backoff jitter.
pub trait Jitter {
    fn unit(&mut self) -> f64;
}

/// Jitter from a xorshift generator seeded from the clock and the thread.
pub struct ThreadJitter(u64);

impl ThreadJitter {
    pub fn new() -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::thread::current().id().hash(&mut hasher);
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos())
            .hash(&mut hasher);
        Self(hasher.finish() | 1)
    }
}

impl Default for ThreadJitter {
    fn default() -> Self {
        Self::new()
    }
}

impl Jitter for ThreadJitter {
    fn unit(&mut self) -> f64 {
        let mut state = self.0;
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        self.0 = state;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A wake-up signal for waits that end early on a stop request. A stopper
/// sets its stop flag and then calls [`StopWake::notify`]; every wait in
/// progress wakes at once and re-reads its flag, so a stop is not delayed by
/// timer coalescing or by a polling interval.
#[derive(Default)]
pub struct StopWake {
    epoch: Mutex<u64>,
    changed: Condvar,
}

impl StopWake {
    /// An unsignalled wake-up signal.
    pub const fn new() -> Self {
        Self {
            epoch: Mutex::new(0),
            changed: Condvar::new(),
        }
    }

    /// Wakes every wait in progress on this signal.
    pub fn notify(&self) {
        let mut epoch = self
            .epoch
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        *epoch = epoch.wrapping_add(1);
        drop(epoch);
        self.changed.notify_all();
    }

    /// Waits `total`; returns `false` as soon as `stopped` holds. The flag is
    /// re-read after every [`StopWake::notify`], and at least every
    /// [`STOP_RECHECK`] for a stopper that only sets its flag.
    pub fn sleep_unless(&self, total: Duration, stopped: &dyn Fn() -> bool) -> bool {
        let started = Instant::now();
        let mut epoch = self
            .epoch
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        loop {
            if stopped() {
                return false;
            }
            let left = total.saturating_sub(started.elapsed());
            if left.is_zero() {
                return true;
            }
            let seen = *epoch;
            epoch = self
                .changed
                .wait_timeout_while(epoch, left.min(STOP_RECHECK), |now| *now == seen)
                .unwrap_or_else(|poison| poison.into_inner())
                .0;
        }
    }
}

/// The signal every pipeline wait listens on.
static STOP_WAKE: StopWake = StopWake::new();

/// Wakes every pipeline wait so it re-reads its stop flag; call it after
/// setting a stop flag handed to the pipeline.
pub fn notify_stop() {
    STOP_WAKE.notify();
    let gates: Vec<Arc<SeatGate>> = seat_gates()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .values()
        .cloned()
        .collect();
    for gate in gates {
        gate.wake();
    }
}

/// Waits `total`; returns `false` as soon as `stopped` holds after a
/// [`notify_stop`].
pub fn sleep_unless(total: Duration, stopped: &dyn Fn() -> bool) -> bool {
    STOP_WAKE.sleep_unless(total, stopped)
}

#[derive(Default)]
struct GateState {
    in_flight: usize,
    /// The most requests ever in flight at once.
    peak: usize,
    /// The concurrent requests allowed while the seat is busy; `None` when it
    /// has not pushed back.
    allowed: Option<usize>,
    /// Successful requests since the allowance last changed.
    streak: usize,
    /// The most requests the seat serves at once across every holder in the
    /// process; `None` when none has been recorded.
    ceiling: Option<usize>,
    /// The ticket handed to the next arrival.
    next_ticket: u64,
    /// The tickets of the lanes waiting for a slot, oldest first.
    queue: std::collections::VecDeque<u64>,
}

impl GateState {
    /// The requests admitted at once: the allowance and the ceiling, whichever
    /// is lower; `None` when neither is set.
    fn limit(&self) -> Option<usize> {
        match (self.allowed, self.ceiling) {
            (Some(allowed), Some(ceiling)) => Some(allowed.min(ceiling)),
            (allowed, ceiling) => allowed.or(ceiling),
        }
    }
}

/// The concurrent requests one seat is allowed: halved when the seat answers
/// 429, regrown by one after as many successes as the current allowance.
#[derive(Default)]
pub struct SeatGate {
    state: Mutex<GateState>,
    released: Condvar,
}

/// A request slot on a seat, released on drop.
pub struct Permit<'a> {
    gate: &'a SeatGate,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut state = self.gate.lock();
        state.in_flight = state.in_flight.saturating_sub(1);
        drop(state);
        self.gate.released.notify_all();
    }
}

impl SeatGate {
    fn lock(&self) -> std::sync::MutexGuard<'_, GateState> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Wakes every lane waiting for a slot so it re-reads its stop flag.
    pub fn wake(&self) {
        drop(self.lock());
        self.released.notify_all();
    }

    /// Waits for a free slot; `None` when `stopped` holds first. Slots go to
    /// waiters in arrival order. The flag is re-read after [`notify_stop`]
    /// and at least every [`STOP_RECHECK`].
    pub fn acquire(&self, stopped: &dyn Fn() -> bool) -> Option<Permit<'_>> {
        self.acquire_noting(stopped, &mut || {})
    }

    /// Like [`SeatGate::acquire`], calling `waiting` once when the request
    /// cannot take a slot at once and so queues behind earlier waiters or
    /// full slots.
    pub fn acquire_noting(
        &self,
        stopped: &dyn Fn() -> bool,
        waiting: &mut dyn FnMut(),
    ) -> Option<Permit<'_>> {
        let ticket = self.reserve();
        self.acquire_ticket(ticket, stopped, waiting)
    }

    /// Takes the next place in the queue now; the place is used by
    /// [`SeatGate::acquire_ticket`] or dropped when its wait is stopped.
    pub fn reserve(&self) -> u64 {
        let mut state = self.lock();
        let ticket = state.next_ticket;
        state.next_ticket += 1;
        state.queue.push_back(ticket);
        ticket
    }

    /// Waits until `ticket` is first in the queue and a slot is free; `None`
    /// when `stopped` holds first, in which case the place is given up.
    /// `waiting` is called once when the place cannot be taken at once.
    pub fn acquire_ticket(
        &self,
        ticket: u64,
        stopped: &dyn Fn() -> bool,
        waiting: &mut dyn FnMut(),
    ) -> Option<Permit<'_>> {
        let mut state = self.lock();
        let mut announced = false;
        loop {
            if stopped() {
                state.queue.retain(|queued| *queued != ticket);
                drop(state);
                self.released.notify_all();
                return None;
            }
            let first = state.queue.front() == Some(&ticket);
            if first && state.limit().is_none_or(|limit| state.in_flight < limit) {
                state.queue.pop_front();
                state.in_flight += 1;
                state.peak = state.peak.max(state.in_flight);
                drop(state);
                self.released.notify_all();
                return Some(Permit { gate: self });
            }
            if !announced {
                announced = true;
                drop(state);
                waiting();
                state = self.lock();
                continue;
            }
            state = self
                .released
                .wait_timeout(state, STOP_RECHECK)
                .unwrap_or_else(|poison| poison.into_inner())
                .0;
        }
    }

    /// Records the most requests the seat serves at once (at least one) and
    /// wakes waiters that the new ceiling admits.
    pub fn set_ceiling(&self, ceiling: usize) {
        self.lock().ceiling = Some(ceiling.max(1));
        self.released.notify_all();
    }

    /// Halves the allowance, never below one.
    pub fn throttled(&self) {
        let mut state = self.lock();
        let current = state.allowed.unwrap_or(state.peak.max(1));
        state.allowed = Some((current / 2).max(1));
        state.streak = 0;
    }

    /// Counts a successful request toward regrowing the allowance.
    pub fn succeeded(&self) {
        let mut state = self.lock();
        let Some(allowed) = state.allowed else {
            return;
        };
        state.streak += 1;
        if state.streak >= allowed {
            state.streak = 0;
            state.allowed = (allowed + 1 < state.peak).then_some(allowed + 1);
            drop(state);
            self.released.notify_all();
        }
    }

    /// The current allowance; `None` when the seat has not pushed back.
    #[cfg(test)]
    pub(crate) fn allowed(&self) -> Option<usize> {
        self.lock().allowed
    }
}

/// The gate shared by every request to the seat at `url`.
pub fn seat_gate(url: &str) -> Arc<SeatGate> {
    seat_gates()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .entry(url.to_string())
        .or_default()
        .clone()
}

/// Every seat gate handed out by [`seat_gate`], keyed by seat URL.
fn seat_gates() -> &'static Mutex<HashMap<String, Arc<SeatGate>>> {
    static GATES: std::sync::OnceLock<Mutex<HashMap<String, Arc<SeatGate>>>> =
        std::sync::OnceLock::new();
    GATES.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Runs `holders` threads that each take a permit from `gate`, hold it
    /// briefly, and release it; returns the most held at once.
    fn peak_holders(gate: &Arc<SeatGate>, holders: usize) -> usize {
        let live = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let threads: Vec<_> = (0..holders)
            .map(|_| {
                let (gate, live, peak) = (gate.clone(), live.clone(), peak.clone());
                std::thread::spawn(move || {
                    let permit = gate.acquire(&|| false).expect("admitted");
                    let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(30));
                    live.fetch_sub(1, Ordering::SeqCst);
                    drop(permit);
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        peak.load(Ordering::SeqCst)
    }

    #[test]
    fn a_ceiling_of_k_admits_at_most_k_concurrent_holders() {
        let gate = Arc::new(SeatGate::default());
        gate.set_ceiling(3);
        let peak = peak_holders(&gate, 9);
        assert!((1..=3).contains(&peak), "peak {peak}");
    }

    #[test]
    fn a_gate_without_a_ceiling_admits_every_holder() {
        let gate = Arc::new(SeatGate::default());
        let held: Vec<_> = (0..16)
            .map(|_| gate.acquire(&|| false).expect("admitted"))
            .collect();
        assert_eq!(held.len(), 16);
        assert_eq!(gate.allowed(), None);
    }

    #[test]
    fn halving_stays_below_the_ceiling() {
        let gate = SeatGate::default();
        gate.set_ceiling(8);
        let held: Vec<_> = (0..8)
            .map(|_| gate.acquire(&|| false).expect("admitted"))
            .collect();
        gate.throttled();
        assert_eq!(gate.allowed(), Some(4));
        drop(held);
        let state = gate.lock();
        assert_eq!(state.limit(), Some(4));
    }

    #[test]
    fn waiters_are_served_in_arrival_order() {
        let gate = Arc::new(SeatGate::default());
        gate.set_ceiling(1);
        let held = gate.acquire(&|| false).expect("admitted");
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut threads = Vec::new();
        for id in 0..5usize {
            let (gate_for, order) = (gate.clone(), order.clone());
            threads.push(std::thread::spawn(move || {
                let permit = gate_for.acquire(&|| false).expect("admitted");
                order.lock().unwrap().push(id);
                drop(permit);
            }));
            while gate.lock().queue.len() < id + 1 {
                std::thread::yield_now();
            }
        }
        drop(held);
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn a_free_slot_is_not_taken_ahead_of_a_waiter() {
        let gate = Arc::new(SeatGate::default());
        gate.set_ceiling(1);
        let held = gate.acquire(&|| false).expect("admitted");
        let order = Arc::new(Mutex::new(Vec::new()));
        let early = {
            let (gate, order) = (gate.clone(), order.clone());
            std::thread::spawn(move || {
                let permit = gate.acquire(&|| false).expect("admitted");
                order.lock().unwrap().push("early");
                drop(permit);
            })
        };
        while gate.lock().queue.is_empty() {
            std::thread::yield_now();
        }
        drop(held);
        let permit = gate.acquire(&|| false).expect("admitted");
        order.lock().unwrap().push("late");
        drop(permit);
        early.join().unwrap();
        assert_eq!(*order.lock().unwrap(), vec!["early", "late"]);
    }

    #[test]
    fn waiting_is_announced_only_when_a_slot_is_not_free() {
        let gate = Arc::new(SeatGate::default());
        gate.set_ceiling(1);
        let mut free = 0;
        let held = gate
            .acquire_noting(&|| false, &mut || free += 1)
            .expect("admitted");
        assert_eq!(free, 0);
        let stop = std::sync::atomic::AtomicBool::new(false);
        let blocked = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| {
                gate.acquire_noting(&|| stop.load(Ordering::SeqCst), &mut || {
                    blocked.fetch_add(1, Ordering::SeqCst);
                })
                .is_none()
            });
            while blocked.load(Ordering::SeqCst) == 0 {
                std::thread::yield_now();
            }
            stop.store(true, Ordering::SeqCst);
            gate.wake();
            assert!(waiter.join().unwrap());
        });
        assert_eq!(blocked.load(Ordering::SeqCst), 1);
        assert!(gate.lock().queue.is_empty());
        drop(held);
    }

    #[test]
    fn a_full_gate_returns_none_once_stopped() {
        let gate = SeatGate::default();
        gate.set_ceiling(1);
        let _held = gate.acquire(&|| false).expect("admitted");
        assert!(gate.acquire(&|| true).is_none());
    }

    #[test]
    fn reserved_places_admit_in_reservation_order() {
        let gate = Arc::new(SeatGate::default());
        gate.set_ceiling(1);
        let tickets: Vec<u64> = (0..3).map(|_| gate.reserve()).collect();
        let order = Arc::new(Mutex::new(Vec::new()));
        let waiting = Arc::new(AtomicUsize::new(0));
        let spawn = |at: usize| {
            let (gate, order, waiting) = (gate.clone(), order.clone(), waiting.clone());
            let ticket = tickets[at];
            std::thread::spawn(move || {
                let permit = gate
                    .acquire_ticket(ticket, &|| false, &mut || {
                        waiting.fetch_add(1, Ordering::SeqCst);
                    })
                    .expect("admitted");
                order.lock().unwrap().push(at);
                drop(permit);
            })
        };
        let last = spawn(2);
        while waiting.load(Ordering::SeqCst) < 1 {
            std::thread::yield_now();
        }
        let middle = spawn(1);
        while waiting.load(Ordering::SeqCst) < 2 {
            std::thread::yield_now();
        }
        assert!(order.lock().unwrap().is_empty());
        let first = spawn(0);
        for handle in [first, middle, last] {
            handle.join().unwrap();
        }
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn a_stopped_waiter_gives_up_its_place_and_a_fresh_ticket_queues_last() {
        let gate = Arc::new(SeatGate::default());
        gate.set_ceiling(1);
        let held = gate.reserve();
        let stopped = Arc::new(AtomicBool::new(false));
        let ahead = gate.reserve();
        let waiting = Arc::new(AtomicUsize::new(0));
        let paused = {
            let (gate, stopped, waiting) = (gate.clone(), stopped.clone(), waiting.clone());
            std::thread::spawn(move || {
                gate.acquire_ticket(ahead, &|| stopped.load(Ordering::SeqCst), &mut || {
                    waiting.fetch_add(1, Ordering::SeqCst);
                })
                .is_none()
            })
        };
        while waiting.load(Ordering::SeqCst) < 1 {
            std::thread::yield_now();
        }
        let behind = gate.reserve();
        stopped.store(true, Ordering::SeqCst);
        gate.wake();
        assert!(paused.join().unwrap());
        let fresh = gate.reserve();
        assert_eq!(gate.lock().queue, VecDeque::from([held, behind, fresh]));
    }

    #[test]
    fn a_stop_signal_ends_a_long_wait_without_waiting_it_out() {
        let wake = StopWake::new();
        let flag = std::sync::atomic::AtomicBool::new(false);
        let stopped = || flag.load(std::sync::atomic::Ordering::SeqCst);
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| wake.sleep_unless(Duration::from_secs(3600), &stopped));
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
            wake.notify();
            assert!(!waiter.join().unwrap());
        });
    }

    #[test]
    fn an_unstopped_wait_runs_to_its_end() {
        let wake = StopWake::new();
        assert!(wake.sleep_unless(Duration::ZERO, &|| false));
    }

    #[test]
    fn a_busy_seat_halves_its_lanes_regrows_on_success_and_leaves_other_seats_alone() {
        let busy = SeatGate::default();
        let never = || false;
        let permits = (0..8)
            .map(|_| busy.acquire(&never).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(busy.allowed(), None);
        busy.throttled();
        assert_eq!(busy.allowed(), Some(4));
        busy.throttled();
        busy.throttled();
        busy.throttled();
        assert_eq!(busy.allowed(), Some(1));
        drop(permits);
        busy.succeeded();
        assert_eq!(busy.allowed(), Some(2));
        busy.succeeded();
        busy.succeeded();
        assert_eq!(busy.allowed(), Some(3));
        let other = SeatGate::default();
        assert_eq!(other.allowed(), None);
        assert!(!Arc::ptr_eq(
            &seat_gate("http://a/v1"),
            &seat_gate("http://b/v1")
        ));
        assert!(Arc::ptr_eq(
            &seat_gate("http://a/v1"),
            &seat_gate("http://a/v1")
        ));
    }

    #[test]
    fn a_lane_waits_for_a_slot_and_a_stop_ends_the_wait() {
        let gate = SeatGate::default();
        let never = || false;
        let held = gate.acquire(&never).unwrap();
        gate.throttled();
        assert_eq!(gate.allowed(), Some(1));
        let started = Instant::now();
        let flag = std::sync::atomic::AtomicBool::new(false);
        let stopped = || flag.load(std::sync::atomic::Ordering::SeqCst);
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| gate.acquire(&stopped).is_none());
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
            gate.wake();
            assert!(waiter.join().unwrap());
        });
        assert!(started.elapsed() < Duration::from_secs(5));
        drop(held);
        assert!(gate.acquire(&never).is_some());
    }
}
