//! Thread-owning handle through which the server submits, cancels and observes requests.

use crate::drain::drain_deadline;
use crate::error::EngineError;
use crate::pressure::{PressureState, Tier};
use crate::scheduler::{
    EngineMetrics, ModelExecutor, Scheduler, SchedulerConfig, validate_request,
};
use crate::sequence::{CancelReason, EngineEvent, EngineRequest, RequestId};
use crate::supervise::panic_message;
use scriptor_cache::KvLayout;
use scriptor_spec::KvPolicy;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio::sync::watch;

/// Shared slot through which the engine thread publishes the sticky device fault that stopped it.
pub type FatalSlot = Arc<watch::Sender<Option<EngineError>>>;

/// Creates an empty [`FatalSlot`].
pub fn fatal_slot() -> FatalSlot {
    Arc::new(watch::channel(None).0)
}

/// Requests queued for admission, as the handle can see them without a round trip to the
/// engine thread.
#[cfg(feature = "kv-demand-diagnostic")]
#[derive(Default)]
struct QueueGate {
    /// Submissions sent to the engine thread and not yet applied.
    inbox: std::sync::atomic::AtomicUsize,
    /// Never-admitted requests waiting in the scheduler as of its last update.
    waiting_new: std::sync::atomic::AtomicUsize,
}

enum Command {
    Submit {
        id: RequestId,
        request: EngineRequest,
        tx: UnboundedSender<EngineEvent>,
        #[cfg(feature = "kv-demand-diagnostic")]
        restartable: bool,
    },
    Cancel(RequestId),
    Shed,
    Protect,
    Shutdown,
}

/// How the engine thread ended other than by an orderly shutdown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineExit {
    /// The thread panicked; the text is the panic message. Request and device state are
    /// unknown, so the process should exit rather than serve on.
    Panicked(String),
}

/// A reading of the engine thread's progress counter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Heartbeat {
    /// Loop iterations completed since the thread started.
    pub beats: u64,
    /// Whether the thread has work (a request is waiting or running).
    pub busy: bool,
    /// Time since the last completed iteration while busy; zero when idle.
    pub age: Duration,
}

/// Progress counter the engine thread advances once per loop iteration.
struct Beat {
    origin: Instant,
    beats: AtomicU64,
    last_micros: AtomicU64,
    busy: AtomicBool,
}

impl Beat {
    fn new() -> Self {
        Self {
            origin: Instant::now(),
            beats: AtomicU64::new(0),
            last_micros: AtomicU64::new(0),
            busy: AtomicBool::new(false),
        }
    }

    fn tick(&self, busy: bool) {
        self.last_micros
            .store(self.origin.elapsed().as_micros() as u64, Ordering::Release);
        self.busy.store(busy, Ordering::Release);
        self.beats.fetch_add(1, Ordering::AcqRel);
    }

    fn read(&self) -> Heartbeat {
        let busy = self.busy.load(Ordering::Acquire);
        let last = Duration::from_micros(self.last_micros.load(Ordering::Acquire));
        Heartbeat {
            beats: self.beats.load(Ordering::Acquire),
            busy,
            age: if busy {
                self.origin.elapsed().saturating_sub(last)
            } else {
                Duration::ZERO
            },
        }
    }
}

/// Receiving end of one request's events.
pub struct RequestStream {
    /// Id to pass to [`EngineHandle::cancel`].
    pub id: RequestId,
    /// Token events followed by exactly one terminal event.
    pub events: UnboundedReceiver<EngineEvent>,
}

/// Cloneable front of the engine thread. Dropping a request's event receiver cancels the
/// request on the next token it would have received.
#[derive(Clone)]
pub struct EngineHandle {
    commands: UnboundedSender<Command>,
    metrics: Arc<Mutex<EngineMetrics>>,
    next_id: Arc<AtomicU64>,
    config: Arc<SchedulerConfig>,
    layout: KvLayout,
    pressure: Arc<PressureState>,
    thread: Arc<Mutex<Option<JoinHandle<()>>>>,
    fatal: FatalSlot,
    draining: Arc<AtomicBool>,
    interrupt: Arc<AtomicBool>,
    beat: Arc<Beat>,
    dead: Arc<Mutex<Option<String>>>,
    rejected_queue: Arc<AtomicU64>,
    #[cfg(feature = "kv-demand-diagnostic")]
    queue: Arc<QueueGate>,
    #[cfg(feature = "kv-demand-diagnostic")]
    policy: KvPolicy,
}

impl EngineHandle {
    /// Starts the engine thread running the scheduler over `executor`.
    pub fn spawn<E: ModelExecutor + 'static>(
        config: SchedulerConfig,
        layout: KvLayout,
        executor: E,
    ) -> Self {
        Self::spawn_with(config, layout, executor, fatal_slot(), |_| {})
    }

    /// As [`EngineHandle::spawn`], publishing a sticky device fault to `fatal` before the engine
    /// thread stops.
    pub fn spawn_with_fatal<E: ModelExecutor + 'static>(
        config: SchedulerConfig,
        layout: KvLayout,
        executor: E,
        fatal: FatalSlot,
    ) -> Self {
        Self::spawn_with(config, layout, executor, fatal, |_| {})
    }

    /// As [`EngineHandle::spawn`], also calling `on_exit` from the engine thread if the thread
    /// panics.
    pub fn spawn_supervised<E: ModelExecutor + 'static>(
        config: SchedulerConfig,
        layout: KvLayout,
        executor: E,
        on_exit: impl FnOnce(EngineExit) + Send + 'static,
    ) -> Self {
        Self::spawn_with(config, layout, executor, fatal_slot(), on_exit)
    }

    /// Starts the engine thread with both death channels. Each step runs under `catch_unwind`.
    /// A panic, or a sticky device fault the scheduler records (published to `fatal`), fails
    /// every waiting, running and queued request, marks the engine dead (see
    /// [`EngineHandle::dead_reason`]), leaks the executor so no device call runs on unknown
    /// state, and ends the thread. A panic additionally calls `on_exit`.
    pub fn spawn_with<E: ModelExecutor + 'static>(
        config: SchedulerConfig,
        layout: KvLayout,
        executor: E,
        fatal: FatalSlot,
        on_exit: impl FnOnce(EngineExit) + Send + 'static,
    ) -> Self {
        Self::spawn_with_policy(
            config,
            layout,
            executor,
            KvPolicy::default(),
            fatal,
            on_exit,
        )
    }

    /// As [`EngineHandle::spawn_with`], with the scheduler committing KV blocks under `policy`.
    pub fn spawn_with_policy<E: ModelExecutor + 'static>(
        config: SchedulerConfig,
        layout: KvLayout,
        executor: E,
        policy: KvPolicy,
        fatal: FatalSlot,
        on_exit: impl FnOnce(EngineExit) + Send + 'static,
    ) -> Self {
        let thread_fatal = Arc::clone(&fatal);
        let (commands, mut inbox) = unbounded_channel::<Command>();
        let metrics = Arc::new(Mutex::new(EngineMetrics::default()));
        let shared = Arc::clone(&metrics);
        let cfg = Arc::new(config.clone());
        let interrupt = Arc::new(AtomicBool::new(false));
        let beat = Arc::new(Beat::new());
        let dead = Arc::new(Mutex::new(None));
        #[cfg(feature = "kv-demand-diagnostic")]
        let queue = Arc::new(QueueGate::default());
        #[cfg(feature = "kv-demand-diagnostic")]
        let t_queue = Arc::clone(&queue);
        let (t_interrupt, t_beat, t_dead) =
            (Arc::clone(&interrupt), Arc::clone(&beat), Arc::clone(&dead));
        let thread = std::thread::Builder::new()
            .name("scriptor-engine".into())
            .spawn(move || {
                let mut scheduler = Scheduler::new(config, layout, executor);
                scheduler.set_kv_policy(policy);
                scheduler.set_interrupt_flag(Arc::clone(&t_interrupt));
                scheduler.set_fatal_sink(thread_fatal);
                let ran = catch_unwind(AssertUnwindSafe(|| {
                    engine_loop(
                        &mut scheduler,
                        &mut inbox,
                        &shared,
                        &t_beat,
                        &t_interrupt,
                        #[cfg(feature = "kv-demand-diagnostic")]
                        &t_queue,
                    );
                }));
                match ran {
                    Ok(()) => {
                        *shared.lock().unwrap_or_else(|p| p.into_inner()) = scheduler.metrics();
                        if let Some(err) = scheduler.fatal().cloned() {
                            bury(scheduler, &mut inbox, &t_dead, &err);
                        }
                    }
                    Err(payload) => {
                        let text = panic_message(payload.as_ref());
                        let err =
                            EngineError::Executor(format!("the engine thread panicked: {text}"));
                        bury(scheduler, &mut inbox, &t_dead, &err);
                        on_exit(EngineExit::Panicked(text));
                    }
                }
            })
            .expect("engine thread spawns");
        Self {
            commands,
            metrics,
            next_id: Arc::new(AtomicU64::new(0)),
            config: cfg,
            layout,
            pressure: Arc::new(PressureState::default()),
            thread: Arc::new(Mutex::new(Some(thread))),
            fatal,
            draining: Arc::new(AtomicBool::new(false)),
            interrupt,
            beat,
            dead,
            rejected_queue: Arc::new(AtomicU64::new(0)),
            #[cfg(feature = "kv-demand-diagnostic")]
            queue,
            #[cfg(feature = "kv-demand-diagnostic")]
            policy,
        }
    }

    /// The sticky device fault that stopped the engine thread, if any.
    pub fn fatal(&self) -> Option<EngineError> {
        self.fatal.borrow().clone()
    }

    /// Validates the request and queues it; refusals carry the user-facing message.
    pub fn submit(&self, request: EngineRequest) -> Result<RequestStream, EngineError> {
        self.submit_as(
            request,
            #[cfg(feature = "kv-demand-diagnostic")]
            false,
        )
    }

    /// As [`EngineHandle::submit`], marking the request restartable: under the demand policy an
    /// eviction may discard its output and run it again from the prompt, announced by
    /// [`EngineEvent::Restarted`].
    #[cfg(feature = "kv-demand-diagnostic")]
    pub fn submit_restartable(&self, request: EngineRequest) -> Result<RequestStream, EngineError> {
        self.submit_as(request, true)
    }

    fn submit_as(
        &self,
        request: EngineRequest,
        #[cfg(feature = "kv-demand-diagnostic")] restartable: bool,
    ) -> Result<RequestStream, EngineError> {
        if let Some(err) = self.fatal() {
            return Err(err);
        }
        if self.draining.load(Ordering::Acquire) {
            return Err(EngineError::Draining);
        }
        validate_request(&self.config, &self.layout, &request)?;
        if self.pressure.tier() >= Tier::Watch {
            self.pressure.record_rejected();
            return Err(EngineError::Pressure {
                retry_after_secs: self.pressure.retry_after_secs(),
            });
        }
        #[cfg(feature = "kv-demand-diagnostic")]
        if self.policy == KvPolicy::DemandDiagnostic {
            let queued = self.queue.inbox.load(Ordering::Acquire)
                + self.queue.waiting_new.load(Ordering::Acquire);
            if queued >= 2 * self.config.max_concurrency {
                self.rejected_queue.fetch_add(1, Ordering::Relaxed);
                return Err(EngineError::QueueFull {
                    retry_after_secs: 1,
                });
            }
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, events) = unbounded_channel();
        #[cfg(feature = "kv-demand-diagnostic")]
        self.queue.inbox.fetch_add(1, Ordering::AcqRel);
        self.commands
            .send(Command::Submit {
                id,
                request,
                tx,
                #[cfg(feature = "kv-demand-diagnostic")]
                restartable,
            })
            .map_err(|_| EngineError::Closed)?;
        Ok(RequestStream { id, events })
    }

    /// The policy under which the scheduler commits KV blocks.
    #[cfg(feature = "kv-demand-diagnostic")]
    pub fn kv_policy(&self) -> KvPolicy {
        self.policy
    }

    /// Cancels a request; its stream ends with [`EngineEvent::Aborted`] unless it already finished.
    pub fn cancel(&self, id: RequestId) {
        let _ = self.commands.send(Command::Cancel(id));
    }

    /// Pressure figures the host-pressure monitor publishes and admission consults.
    pub fn pressure(&self) -> Arc<PressureState> {
        Arc::clone(&self.pressure)
    }

    /// Cancels the youngest running request and releases cached and pooled memory.
    pub fn shed(&self) {
        let _ = self.commands.send(Command::Shed);
    }

    /// Fails every request; the owner is expected to stop the engine afterwards.
    pub fn protect(&self) {
        let _ = self.commands.send(Command::Protect);
    }

    /// Latest metrics published by the engine thread.
    pub fn metrics(&self) -> EngineMetrics {
        let mut m = self
            .metrics
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        m.rejected_queue += self.rejected_queue.load(Ordering::Relaxed);
        m
    }

    /// Makes [`EngineHandle::submit`] refuse new requests; running ones continue.
    pub fn begin_drain(&self) {
        self.draining.store(true, Ordering::Release);
    }

    /// Whether [`EngineHandle::begin_drain`] was called.
    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::Acquire)
    }

    /// Ends all work now: the engine thread stops after the stage it is in, cancels every
    /// waiting and running request with [`CancelReason::Drain`], and goes idle. New requests
    /// are refused as by [`EngineHandle::begin_drain`].
    pub fn interrupt(&self) {
        self.begin_drain();
        self.interrupt.store(true, Ordering::Release);
    }

    /// Why the engine thread died, when it did.
    pub fn dead_reason(&self) -> Option<String> {
        self.dead.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// The engine thread's progress counter.
    pub fn heartbeat(&self) -> Heartbeat {
        self.beat.read()
    }

    /// Time the engine may keep serving during a drain: see [`drain_deadline`]. The decode
    /// rate and mean prefill pass come from the metrics the engine has published.
    pub fn drain_deadline(&self, bound: Option<Duration>) -> Duration {
        let m = self.metrics();
        let rate = if m.decode_seconds > 0.0 {
            m.generation_tokens as f64 / m.decode_seconds
        } else {
            0.0
        };
        let prefill = if m.prefill_passes > 0 {
            m.prefill_seconds / m.prefill_passes as f64
        } else {
            0.0
        };
        drain_deadline(m.tokens_in_flight, rate, prefill, bound)
    }

    /// Stops the engine thread after the current step and waits for it; returns whether the
    /// thread ended without a panic.
    pub fn shutdown(&self) -> bool {
        let _ = self.commands.send(Command::Shutdown);
        let handle = self.thread.lock().unwrap_or_else(|p| p.into_inner()).take();
        handle.is_none_or(|h| h.join().is_ok()) && self.dead_reason().is_none()
    }
}

/// Records `err` as the reason the engine died, fails every request the scheduler holds and
/// every submission still queued, refuses later submissions, and leaks the scheduler so no
/// device call or teardown runs on state that can no longer be trusted.
fn bury<E: ModelExecutor>(
    mut scheduler: Scheduler<E>,
    inbox: &mut UnboundedReceiver<Command>,
    dead: &Mutex<Option<String>>,
    err: &EngineError,
) {
    *dead.lock().unwrap_or_else(|p| p.into_inner()) = Some(err.to_string());
    let _ = catch_unwind(AssertUnwindSafe(|| scheduler.abandon_all(err)));
    while let Ok(cmd) = inbox.try_recv() {
        if let Command::Submit { tx, .. } = cmd {
            let _ = tx.send(EngineEvent::Failed(err.clone()));
        }
    }
    inbox.close();
    std::mem::forget(scheduler);
}

/// Runs the scheduler until shutdown or until every handle is dropped.
fn engine_loop<E: ModelExecutor>(
    scheduler: &mut Scheduler<E>,
    inbox: &mut UnboundedReceiver<Command>,
    shared: &Mutex<EngineMetrics>,
    beat: &Beat,
    interrupt: &AtomicBool,
    #[cfg(feature = "kv-demand-diagnostic")] queue: &QueueGate,
) {
    'run: loop {
        if scheduler.is_idle() {
            beat.tick(false);
            match inbox.blocking_recv() {
                Some(cmd) => {
                    if apply(
                        scheduler,
                        cmd,
                        #[cfg(feature = "kv-demand-diagnostic")]
                        queue,
                    ) {
                        break 'run;
                    }
                }
                None => break 'run,
            }
        }
        while let Ok(cmd) = inbox.try_recv() {
            if apply(
                scheduler,
                cmd,
                #[cfg(feature = "kv-demand-diagnostic")]
                queue,
            ) {
                break 'run;
            }
        }
        if interrupt.load(Ordering::Acquire) {
            scheduler.cancel_all(CancelReason::Drain);
        }
        scheduler.step();
        #[cfg(feature = "kv-demand-diagnostic")]
        queue
            .waiting_new
            .store(scheduler.waiting_new(), Ordering::Release);
        *shared.lock().unwrap_or_else(|p| p.into_inner()) = scheduler.metrics();
        if scheduler.fatal().is_some() {
            break 'run;
        }
        beat.tick(!scheduler.is_idle());
    }
    *shared.lock().unwrap_or_else(|p| p.into_inner()) = scheduler.metrics();
}

/// Applies a command; returns whether the loop must stop.
fn apply<E: ModelExecutor>(
    scheduler: &mut Scheduler<E>,
    cmd: Command,
    #[cfg(feature = "kv-demand-diagnostic")] queue: &QueueGate,
) -> bool {
    match cmd {
        Command::Submit {
            id,
            request,
            tx,
            #[cfg(feature = "kv-demand-diagnostic")]
            restartable,
        } => {
            #[cfg(feature = "kv-demand-diagnostic")]
            queue.inbox.fetch_sub(1, Ordering::AcqRel);
            #[cfg(feature = "kv-demand-diagnostic")]
            let submitted =
                scheduler.submit_restartable_with_id(id, request, tx.clone(), restartable);
            #[cfg(not(feature = "kv-demand-diagnostic"))]
            let submitted = scheduler.submit_with_id(id, request, tx.clone());
            if let Err(e) = submitted {
                let _ = tx.send(EngineEvent::Failed(e));
            }
            #[cfg(feature = "kv-demand-diagnostic")]
            queue
                .waiting_new
                .store(scheduler.waiting_new(), Ordering::Release);
            false
        }
        Command::Cancel(id) => {
            scheduler.cancel(id);
            false
        }
        Command::Shed => {
            scheduler.shed_youngest();
            scheduler.clear_embedding_cache();
            scheduler.executor_mut().trim_memory();
            false
        }
        Command::Protect => {
            scheduler.shed_all();
            scheduler.clear_embedding_cache();
            false
        }
        Command::Shutdown => true,
    }
}
