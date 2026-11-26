//! Prometheus text exposition: vLLM-aliased series plus `scriptor_*` engine-native series.

use std::fmt::Write as _;
use std::sync::Mutex;

use crate::app::EngineStats;

const LATENCY_BOUNDS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];
const TOKEN_BOUNDS: &[f64] = &[
    1.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0,
];
const PATCH_BOUNDS: &[f64] = &[
    64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0, 131072.0,
];
const BATCH_BOUNDS: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];

#[derive(Debug, Clone)]
struct Histogram {
    bounds: &'static [f64],
    buckets: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Self {
        Self {
            bounds,
            buckets: vec![0; bounds.len()],
            sum: 0.0,
            count: 0,
        }
    }

    fn observe(&mut self, value: f64) {
        for (i, bound) in self.bounds.iter().enumerate() {
            if value <= *bound {
                self.buckets[i] += 1;
            }
        }
        self.sum += value;
        self.count += 1;
    }

    fn render(&self, out: &mut String, name: &str, labels: &str) {
        for (bound, n) in self.bounds.iter().zip(&self.buckets) {
            let _ = writeln!(out, "{name}_bucket{{{labels},le=\"{bound}\"}} {n}");
        }
        let _ = writeln!(out, "{name}_bucket{{{labels},le=\"+Inf\"}} {}", self.count);
        let _ = writeln!(out, "{name}_sum{{{labels}}} {}", self.sum);
        let _ = writeln!(out, "{name}_count{{{labels}}} {}", self.count);
    }
}

#[derive(Debug)]
struct Inner {
    prompt_tokens: u64,
    generation_tokens: u64,
    success_stop: u64,
    success_length: u64,
    rejected_busy: u64,
    cancelled: std::collections::BTreeMap<String, u64>,
    iteration_tokens: Histogram,
    prefill_time: Histogram,
    decode_time: Histogram,
    e2e_time: Histogram,
    step_time: Histogram,
    first_token_time: Histogram,
    inter_token_time: Histogram,
    queue_time: Histogram,
}

/// Thread-safe metric registry shared by the HTTP handlers and the engine.
#[derive(Debug)]
pub struct Metrics {
    inner: Mutex<Inner>,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                prompt_tokens: 0,
                generation_tokens: 0,
                success_stop: 0,
                success_length: 0,
                rejected_busy: 0,
                cancelled: std::collections::BTreeMap::new(),
                iteration_tokens: Histogram::new(TOKEN_BOUNDS),
                prefill_time: Histogram::new(LATENCY_BOUNDS),
                decode_time: Histogram::new(LATENCY_BOUNDS),
                e2e_time: Histogram::new(LATENCY_BOUNDS),
                step_time: Histogram::new(LATENCY_BOUNDS),
                first_token_time: Histogram::new(LATENCY_BOUNDS),
                inter_token_time: Histogram::new(LATENCY_BOUNDS),
                queue_time: Histogram::new(LATENCY_BOUNDS),
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Records one finished request: token counts, finish reason and phase durations in seconds.
    pub fn record_request(
        &self,
        prompt_tokens: u64,
        completion_tokens: u64,
        finish_reason: &str,
        prefill_seconds: f64,
        decode_seconds: f64,
    ) {
        let mut m = self.lock();
        m.prompt_tokens += prompt_tokens;
        m.generation_tokens += completion_tokens;
        if finish_reason == "length" {
            m.success_length += 1;
        } else {
            m.success_stop += 1;
        }
        m.prefill_time.observe(prefill_seconds);
        m.decode_time.observe(decode_seconds);
        m.e2e_time.observe(prefill_seconds + decode_seconds);
    }

    /// Records one request the engine cancelled, by reason.
    pub fn record_cancelled(&self, reason: &str) {
        *self.lock().cancelled.entry(reason.to_owned()).or_insert(0) += 1;
    }

    /// Records the seconds from request start to the first delivered text.
    pub fn observe_first_token(&self, seconds: f64) {
        self.lock().first_token_time.observe(seconds);
    }

    /// Records the seconds between two consecutive text deliveries of one request.
    pub fn observe_inter_token(&self, seconds: f64) {
        self.lock().inter_token_time.observe(seconds);
    }

    /// Records the seconds a request waited in the engine before admission.
    pub fn observe_queue_time(&self, seconds: f64) {
        self.lock().queue_time.observe(seconds);
    }

    /// Counts a request answered with 429.
    pub fn record_rejected_busy(&self) {
        self.lock().rejected_busy += 1;
    }

    /// Renders the registry and an engine snapshot as Prometheus text.
    pub fn render(&self, model: &str, stats: &EngineStats) -> String {
        let m = self.lock();
        let labels = format!("engine=\"0\",model_name=\"{}\"", escape(model));
        let mut out = String::new();
        let gauge = |out: &mut String, name: &str, kind: &str, value: f64| {
            let _ = writeln!(out, "# TYPE {name} {kind}");
            let _ = writeln!(out, "{name}{{{labels}}} {value}");
        };
        gauge(
            &mut out,
            "vllm:num_requests_running",
            "gauge",
            stats.running as f64,
        );
        gauge(
            &mut out,
            "vllm:num_requests_waiting",
            "gauge",
            stats.waiting as f64,
        );
        gauge(
            &mut out,
            "vllm:kv_cache_usage_perc",
            "gauge",
            stats.kv_cache_usage,
        );
        let native_labels = format!("model_name=\"{}\"", escape(model));
        let _ = writeln!(out, "# TYPE scriptor_engine_state gauge");
        for state in ["running", "draining", "dead"] {
            let on = u8::from(stats.lifecycle.label() == state);
            let _ = writeln!(
                out,
                "scriptor_engine_state{{{native_labels},state=\"{state}\"}} {on}"
            );
        }
        let _ = writeln!(out, "# TYPE scriptor_engine_heartbeat_total counter");
        let _ = writeln!(
            out,
            "scriptor_engine_heartbeat_total{{{native_labels}}} {}",
            stats.heartbeat_beats
        );
        let _ = writeln!(out, "# TYPE scriptor_engine_heartbeat_age_seconds gauge");
        let _ = writeln!(
            out,
            "scriptor_engine_heartbeat_age_seconds{{{native_labels}}} {}",
            stats.heartbeat_age_seconds
        );
        let _ = writeln!(out, "# TYPE scriptor_requests_cancelled_total counter");
        for (reason, n) in &m.cancelled {
            let _ = writeln!(
                out,
                "scriptor_requests_cancelled_total{{{native_labels},reason=\"{}\"}} {n}",
                escape(reason)
            );
        }
        for (name, value) in [
            ("scriptor_max_num_seqs", stats.slots as f64),
            ("scriptor_kv_cache_reserved_perc", stats.kv_reserved_perc),
            ("scriptor_kv_cache_tokens", stats.kv_cache_tokens as f64),
        ] {
            let _ = writeln!(out, "# TYPE {name} gauge");
            let _ = writeln!(out, "{name}{{{native_labels}}} {value}");
        }
        gauge(
            &mut out,
            "vllm:prompt_tokens_total",
            "counter",
            m.prompt_tokens as f64,
        );
        gauge(
            &mut out,
            "vllm:generation_tokens_total",
            "counter",
            m.generation_tokens as f64,
        );
        let _ = writeln!(out, "# TYPE vllm:request_success_total counter");
        let _ = writeln!(
            out,
            "vllm:request_success_total{{{labels},finished_reason=\"stop\"}} {}",
            m.success_stop
        );
        let _ = writeln!(
            out,
            "vllm:request_success_total{{{labels},finished_reason=\"length\"}} {}",
            m.success_length
        );
        for (name, hist) in [
            ("vllm:iteration_tokens_total", &m.iteration_tokens),
            ("vllm:request_prefill_time_seconds", &m.prefill_time),
            ("vllm:request_decode_time_seconds", &m.decode_time),
            ("vllm:e2e_request_latency_seconds", &m.e2e_time),
            ("vllm:time_to_first_token_seconds", &m.first_token_time),
            ("vllm:inter_token_latency_seconds", &m.inter_token_time),
            ("vllm:request_queue_time_seconds", &m.queue_time),
        ] {
            let _ = writeln!(out, "# TYPE {name} histogram");
            hist.render(&mut out, name, &labels);
        }
        gauge(
            &mut out,
            "vllm:num_preemptions_total",
            "counter",
            stats.preemptions as f64,
        );
        gauge(
            &mut out,
            "vllm:spec_decode_num_drafts_total",
            "counter",
            stats.spec_drafts as f64,
        );
        gauge(
            &mut out,
            "vllm:spec_decode_num_draft_tokens_total",
            "counter",
            stats.spec_draft_tokens as f64,
        );
        gauge(
            &mut out,
            "vllm:spec_decode_num_accepted_tokens_total",
            "counter",
            stats.spec_accepted_tokens as f64,
        );
        let _ = writeln!(
            out,
            "# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter"
        );
        let per_pos = if stats.spec_accepted_per_pos.is_empty() {
            &[0u64][..]
        } else {
            &stats.spec_accepted_per_pos[..]
        };
        for (position, n) in per_pos.iter().enumerate() {
            let _ = writeln!(
                out,
                "vllm:spec_decode_num_accepted_tokens_per_pos_total{{{labels},position=\"{position}\"}} {n}"
            );
        }
        let native = format!("model_name=\"{}\"", escape(model));
        let _ = writeln!(out, "# TYPE scriptor_vision_batch_size histogram");
        let mut vision_batch = Histogram::new(BATCH_BOUNDS);
        for &(images, batches) in &stats.vision_batch_sizes {
            for _ in 0..batches {
                vision_batch.observe(images as f64);
            }
        }
        vision_batch.render(&mut out, "scriptor_vision_batch_size", &native);
        let _ = writeln!(out, "# TYPE scriptor_vision_batch_patches histogram");
        let mut patch_batch = Histogram::new(PATCH_BOUNDS);
        for &(upper, batches) in &stats.vision_batch_patch_sizes {
            for _ in 0..batches {
                patch_batch.observe(upper as f64);
            }
        }
        patch_batch.sum = stats.vision_patches as f64;
        patch_batch.render(&mut out, "scriptor_vision_batch_patches", &native);
        for (name, value) in [
            ("scriptor_vision_patches_total", stats.vision_patches),
            ("scriptor_vision_images_total", stats.vision_images),
        ] {
            let _ = writeln!(out, "# TYPE {name} counter");
            let _ = writeln!(out, "{name}{{{native}}} {value}");
        }
        render_image_series(&mut out, &native, stats);
        render_prefill_passes(&mut out, &native, stats);
        if let Some(l1) = &stats.l1 {
            render_l1(&mut out, &native, l1);
        }
        let _ = writeln!(out, "# TYPE scriptor_step_time_seconds histogram");
        m.step_time
            .render(&mut out, "scriptor_step_time_seconds", &native);
        let _ = writeln!(out, "# TYPE scriptor_embedding_cache_hits_total counter");
        let _ = writeln!(
            out,
            "scriptor_embedding_cache_hits_total{{{native}}} {}",
            stats.embedding_cache_hits
        );
        for (name, kind, value) in [
            ("scriptor_graphs_captured", "gauge", stats.graphs_captured),
            (
                "scriptor_graph_replays_total",
                "counter",
                stats.graph_replays,
            ),
            (
                "scriptor_graph_split_replays_total",
                "counter",
                stats.graph_split_replays,
            ),
            (
                "scriptor_graph_fallbacks_total",
                "counter",
                stats.graph_fallbacks,
            ),
        ] {
            let _ = writeln!(out, "# TYPE {name} {kind}");
            let _ = writeln!(out, "{name}{{{native}}} {value}");
        }
        crate::occupancy::render(&mut out, &native, &stats.slot_occupancy);
        if let Some(t) = scriptor_engine::active_telemetry::counters_snapshot() {
            t.render_metrics(&mut out, &native);
        }
        let _ = writeln!(out, "# TYPE scriptor_preemptions_total counter");
        let _ = writeln!(
            out,
            "scriptor_preemptions_total{{{native}}} {}",
            stats.preemptions
        );
        let _ = writeln!(
            out,
            "# HELP scriptor_preemptions_by_action_total Sequences taken off the device, by action: \
             restart and recompute sum to scriptor_preemptions_total; yield (a vision or prefill \
             sequence returned to the queue) is not part of that sum."
        );
        let _ = writeln!(out, "# TYPE scriptor_preemptions_by_action_total counter");
        for (action, n) in [
            ("restart", stats.preemptions_restart),
            ("recompute", stats.preemptions_recompute),
            ("yield", stats.preemptions_yield),
        ] {
            let _ = writeln!(
                out,
                "scriptor_preemptions_by_action_total{{{native},action=\"{action}\"}} {n}"
            );
        }
        render_admission(&mut out, &native, stats);
        let _ = writeln!(out, "# TYPE scriptor_boot_stage_seconds gauge");
        for (stage, seconds) in &stats.boot_stage_seconds {
            let _ = writeln!(
                out,
                "scriptor_boot_stage_seconds{{{native},stage=\"{}\"}} {seconds}",
                escape(stage)
            );
        }
        for (name, seconds) in [
            ("scriptor_vision_seconds_total", stats.vision_seconds),
            (
                "scriptor_vision_host_seconds_total",
                stats.vision_host_seconds,
            ),
            ("scriptor_prefill_seconds_total", stats.prefill_seconds),
            ("scriptor_decode_seconds_total", stats.decode_seconds),
            // Reservation plus row assembly only; executor input preparation stays in
            // `scriptor_decode_seconds_total`, which this counter is not part of.
            (
                "scriptor_decode_prep_seconds_total",
                stats.decode_prep_seconds,
            ),
        ] {
            let _ = writeln!(out, "# TYPE {name} counter");
            let _ = writeln!(out, "{name}{{{native}}} {seconds}");
        }
        for (name, kind, value) in [
            (
                "scriptor_pressure_tier",
                "gauge",
                stats.pressure_tier as f64,
            ),
            (
                "scriptor_host_mem_available_bytes",
                "gauge",
                stats.host_mem_available_bytes as f64,
            ),
            (
                "scriptor_host_psi_memory_full_avg10",
                "gauge",
                stats.host_psi_memory_full_avg10,
            ),
            (
                "scriptor_requests_shed_total",
                "counter",
                stats.requests_shed_total as f64,
            ),
        ] {
            let _ = writeln!(out, "# TYPE {name} {kind}");
            let _ = writeln!(out, "{name}{{{native}}} {value}");
        }
        let _ = writeln!(out, "# TYPE scriptor_load_shed_total counter");
        for (tier, n) in [
            ("shed", stats.load_shed_total),
            ("protect", stats.load_protect_total),
        ] {
            let _ = writeln!(
                out,
                "scriptor_load_shed_total{{{native},tier=\"{tier}\"}} {n}"
            );
        }
        let _ = writeln!(out, "# TYPE scriptor_admission_rejected_total counter");
        let _ = writeln!(
            out,
            "scriptor_admission_rejected_total{{{native},reason=\"pressure\"}} {}",
            stats.admission_rejected_pressure
        );
        let _ = writeln!(out, "# TYPE scriptor_requests_rejected_busy_total counter");
        let _ = writeln!(
            out,
            "scriptor_requests_rejected_busy_total{{{native}}} {}",
            m.rejected_busy
        );
        out
    }
}

/// Renders the KV-policy, admission and retained-embedding series. Every series is present
/// under every policy; the demand-only ones read zero under the reserve policy.
fn render_admission(out: &mut String, native: &str, stats: &EngineStats) {
    let _ = writeln!(out, "# TYPE scriptor_kv_policy gauge");
    for (policy, on) in [
        ("reserve", !stats.kv_demand),
        ("demand-diagnostic", stats.kv_demand),
    ] {
        let _ = writeln!(
            out,
            "scriptor_kv_policy{{{native},policy=\"{policy}\"}} {}",
            u8::from(on)
        );
    }
    for (name, kind, value) in [
        ("scriptor_kv_used_blocks", "gauge", stats.kv_used_blocks),
        (
            "scriptor_kv_step_growth_blocks",
            "gauge",
            stats.kv_step_growth_blocks,
        ),
        (
            "scriptor_accepted_requests",
            "gauge",
            stats.accepted_requests,
        ),
        (
            "scriptor_retained_embedding_bytes",
            "gauge",
            stats.retained_embedding_bytes,
        ),
        (
            "scriptor_pinned_embedding_bytes",
            "gauge",
            stats.pinned_embedding_bytes,
        ),
        (
            "scriptor_restart_discarded_tokens_total",
            "counter",
            stats.restart_discarded_tokens,
        ),
        (
            "scriptor_requests_rejected_queue_total",
            "counter",
            stats.requests_rejected_queue,
        ),
    ] {
        let _ = writeln!(out, "# TYPE {name} {kind}");
        let _ = writeln!(out, "{name}{{{native}}} {value}");
    }
    let _ = writeln!(out, "# TYPE scriptor_admission_blocked_steps_total counter");
    for (reason, n) in [
        "slots",
        "watermark",
        "hysteresis",
        "prompt_blocks",
        "accepted",
        "embedding",
    ]
    .into_iter()
    .zip(stats.admission_blocked)
    {
        let _ = writeln!(
            out,
            "scriptor_admission_blocked_steps_total{{{native},reason=\"{reason}\"}} {n}"
        );
    }
}

/// Renders the queue-depth and phase gauges, the embedding-release counters, the per-image patch
/// histogram and the exact-key repeat counters.
fn render_image_series(out: &mut String, native: &str, stats: &EngineStats) {
    let _ = writeln!(out, "# TYPE scriptor_waiting_requests gauge");
    let _ = writeln!(
        out,
        "scriptor_waiting_requests{{{native}}} {}",
        stats.waiting
    );
    let _ = writeln!(out, "# TYPE scriptor_active_by_phase gauge");
    for (phase, n) in [
        ("vision", stats.active_vision),
        ("prefill", stats.active_prefill),
        ("running", stats.active_running),
    ] {
        let _ = writeln!(
            out,
            "scriptor_active_by_phase{{{native},phase=\"{phase}\"}} {n}"
        );
    }
    let _ = writeln!(out, "# TYPE scriptor_image_patches histogram");
    let mut per_image = Histogram::new(PATCH_BOUNDS);
    for &(upper, images) in &stats.image_patch_sizes {
        for _ in 0..images {
            per_image.observe(upper as f64);
        }
    }
    per_image.sum = stats.image_patches_submitted as f64;
    per_image.render(out, "scriptor_image_patches", native);
    for (name, value) in [
        ("scriptor_images_submitted_total", stats.images_submitted),
        ("scriptor_image_key_unique_total", stats.image_key_unique),
        ("scriptor_image_key_repeat_total", stats.image_key_repeat),
        (
            "scriptor_image_key_repeat_patches_total",
            stats.image_key_repeat_patches,
        ),
        (
            "scriptor_image_key_untracked_total",
            stats.image_key_untracked,
        ),
        (
            "scriptor_embedding_released_total",
            stats.embedding_released,
        ),
        (
            "scriptor_embedding_released_bytes_total",
            stats.embedding_released_bytes,
        ),
    ] {
        let _ = writeln!(out, "# TYPE {name} counter");
        let _ = writeln!(out, "{name}{{{native}}} {value}");
    }
}

/// Renders the prefill pass size histogram and the below-ridge pass counters.
fn render_prefill_passes(out: &mut String, native: &str, stats: &EngineStats) {
    let _ = writeln!(out, "# TYPE scriptor_prefill_pass_tokens histogram");
    let mut per_pass = Histogram::new(TOKEN_BOUNDS);
    for &(upper, passes) in &stats.prefill_pass_token_sizes {
        for _ in 0..passes {
            per_pass.observe(upper as f64);
        }
    }
    per_pass.sum = stats.prefill_pass_tokens as f64;
    per_pass.render(out, "scriptor_prefill_pass_tokens", native);
    for (name, value) in [
        (
            "scriptor_prefill_passes_below_ridge_total",
            stats.prefill_passes_below_ridge,
        ),
        (
            "scriptor_prefill_pass_tokens_below_ridge_total",
            stats.prefill_pass_tokens_below_ridge,
        ),
    ] {
        let _ = writeln!(out, "# TYPE {name} counter");
        let _ = writeln!(out, "{name}{{{native}}} {value}");
    }
}

/// Renders the nanosecond histogram `hist` with `bounds` as a seconds histogram `name`.
fn render_ns_histogram(
    out: &mut String,
    name: &str,
    native: &str,
    bounds: &[u64; 8],
    hist: &scriptor_engine::l1_counters::NsHistogram,
) {
    let _ = writeln!(out, "# TYPE {name} histogram");
    let mut cumulative = 0u64;
    for (bound, n) in bounds.iter().zip(&hist.counts) {
        cumulative += n;
        let _ = writeln!(
            out,
            "{name}_bucket{{{native},le=\"{}\"}} {cumulative}",
            *bound as f64 / 1e9
        );
    }
    let total = hist.total();
    let _ = writeln!(out, "{name}_bucket{{{native},le=\"+Inf\"}} {total}");
    let _ = writeln!(out, "{name}_sum{{{native}}} {}", hist.sum_ns as f64 / 1e9);
    let _ = writeln!(out, "{name}_count{{{native}}} {total}");
}

/// Renders the L1 host-span and vision-flight series.
/// One labelled metric family: name, help text and (label, value) samples.
type LabelledSeries<'a> = (&'a str, &'a str, Vec<(&'a str, f64)>);

fn render_l1(out: &mut String, native: &str, l1: &scriptor_engine::l1_counters::L1Stats) {
    use scriptor_engine::l1_counters::{POLL_LAG_BOUNDS_NS, TURNAROUND_BOUNDS_NS};
    let secs = |ns: u64| ns as f64 / 1e9;
    for (name, value) in [
        ("scriptor_l1_steps_total", l1.steps),
        ("scriptor_l1_step_windows_total", l1.windows),
        (
            "scriptor_l1_engine_cpu_migrations_total",
            l1.engine_cpu_migrations,
        ),
        ("scriptor_l1_vision_flights_total", l1.flights),
        ("scriptor_l1_vision_flight_patches_total", l1.flight_patches),
        ("scriptor_l1_vision_flight_sum_n2_total", l1.flight_sum_n2),
        ("scriptor_l1_vision_neg_lag_total", l1.neg_lag),
        (
            "scriptor_l1_vision_callbacks_missing_total",
            l1.callbacks_missing,
        ),
    ] {
        let _ = writeln!(out, "# TYPE {name} counter");
        let _ = writeln!(out, "{name}{{{native}}} {value}");
    }
    let labelled: [LabelledSeries; 5] = [
        (
            "scriptor_l1_step_span_seconds_total",
            "span",
            vec![
                ("overlap", secs(l1.overlap_ns)),
                ("sync", secs(l1.sync_ns)),
                ("sync_cpu", secs(l1.sync_cpu_ns)),
                ("collect_tail", secs(l1.collect_tail_ns)),
                ("turnaround", secs(l1.turnaround_ns)),
                ("turnaround_prefill", secs(l1.turnaround_prefill_ns)),
                ("turnaround_vision_wait", secs(l1.turnaround_vision_wait_ns)),
                ("turnaround_host", secs(l1.turnaround_host_ns)),
            ],
        ),
        (
            "scriptor_l1_steps_by_vision_total",
            "flight",
            vec![
                ("busy", l1.steps_vision_busy as f64),
                ("idle", l1.steps_vision_idle as f64),
            ],
        ),
        (
            "scriptor_l1_sync_by_vision_seconds_total",
            "flight",
            vec![
                ("busy", secs(l1.sync_vision_busy_ns)),
                ("idle", secs(l1.sync_vision_idle_ns)),
            ],
        ),
        (
            "scriptor_l1_vision_seconds_total",
            "part",
            vec![
                ("launch", secs(l1.vision_launch_ns)),
                ("start_wait", secs(l1.vision_start_wait_ns)),
                ("copy", secs(l1.vision_copy_ns)),
                ("device", secs(l1.vision_device_ns)),
                ("poll_lag", secs(l1.vision_poll_lag_ns)),
                ("gap_pending", secs(l1.vision_gap_pending_ns)),
                ("gap_starved", secs(l1.vision_gap_starved_ns)),
            ],
        ),
        (
            "scriptor_l1_vision_polls_total",
            "result",
            vec![
                ("ready", l1.polls_ready as f64),
                ("not_ready", l1.polls_not_ready as f64),
            ],
        ),
    ];
    for (name, label, series) in &labelled {
        let _ = writeln!(out, "# TYPE {name} counter");
        for (value_label, value) in series {
            let _ = writeln!(out, "{name}{{{native},{label}=\"{value_label}\"}} {value}");
        }
    }
    let _ = writeln!(out, "# TYPE scriptor_l1_engine_cpu_steps_total counter");
    for (cpu, steps) in l1.engine_cpu_steps.iter().filter(|(_, n)| **n > 0) {
        let _ = writeln!(
            out,
            "scriptor_l1_engine_cpu_steps_total{{{native},cpu=\"{cpu}\"}} {steps}"
        );
    }
    render_ns_histogram(
        out,
        "scriptor_l1_turnaround_host_seconds",
        native,
        &TURNAROUND_BOUNDS_NS,
        &l1.turnaround_host,
    );
    render_ns_histogram(
        out,
        "scriptor_l1_vision_poll_lag_seconds",
        native,
        &POLL_LAG_BOUNDS_NS,
        &l1.vision_poll_lag,
    );
}

fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rendered(stats: &EngineStats) -> String {
        Metrics::new().render("m", stats)
    }

    fn l1_sample() -> scriptor_engine::l1_counters::L1Stats {
        let mut l1 = scriptor_engine::l1_counters::L1Stats {
            steps: 3,
            windows: 2,
            sync_ns: 1_500_000_000,
            flights: 4,
            vision_start_wait_ns: 250_000_000,
            polls_not_ready: 7,
            engine_cpu_migrations: 2,
            ..Default::default()
        };
        l1.engine_cpu_steps.insert(17, 5);
        l1.engine_cpu_steps.insert(3, 0);
        l1.turnaround_host.counts[0] = 1;
        l1.turnaround_host.counts[8] = 1;
        l1.turnaround_host.sum_ns = 20_000_000;
        l1.vision_poll_lag.counts[1] = 2;
        l1.vision_poll_lag.sum_ns = 12_000_000;
        l1
    }

    #[test]
    fn l1_series_render_with_their_values() {
        let stats = EngineStats {
            l1: Some(l1_sample()),
            ..EngineStats::default()
        };
        let t = rendered(&stats);
        assert!(t.contains("scriptor_l1_steps_total{model_name=\"m\"} 3\n"));
        assert!(
            t.contains("scriptor_l1_step_span_seconds_total{model_name=\"m\",span=\"sync\"} 1.5\n")
        );
        assert!(t.contains(
            "scriptor_l1_vision_seconds_total{model_name=\"m\",part=\"start_wait\"} 0.25\n"
        ));
        assert!(
            t.contains("scriptor_l1_vision_polls_total{model_name=\"m\",result=\"not_ready\"} 7\n")
        );
        assert!(t.contains("scriptor_l1_engine_cpu_migrations_total{model_name=\"m\"} 2\n"));
        assert!(t.contains(
            "scriptor_l1_turnaround_host_seconds_bucket{model_name=\"m\",le=\"0.0001\"} 1\n"
        ));
        assert!(t.contains(
            "scriptor_l1_turnaround_host_seconds_bucket{model_name=\"m\",le=\"+Inf\"} 2\n"
        ));
        assert!(t.contains("scriptor_l1_turnaround_host_seconds_sum{model_name=\"m\"} 0.02\n"));
        assert!(t.contains("scriptor_l1_turnaround_host_seconds_count{model_name=\"m\"} 2\n"));
        assert!(t.contains(
            "scriptor_l1_vision_poll_lag_seconds_bucket{model_name=\"m\",le=\"0.005\"} 2\n"
        ));
    }

    #[test]
    fn l1_series_absent_when_off() {
        let t = rendered(&EngineStats::default());
        assert!(!t.contains("scriptor_l1_"));
    }

    #[test]
    fn l1_cpu_series_only_for_nonzero_cpus() {
        let stats = EngineStats {
            l1: Some(l1_sample()),
            ..EngineStats::default()
        };
        let t = rendered(&stats);
        assert!(t.contains("scriptor_l1_engine_cpu_steps_total{model_name=\"m\",cpu=\"17\"} 5\n"));
        assert!(!t.contains("cpu=\"3\""));
    }

    #[test]
    fn vision_patch_counters_render() {
        let stats = EngineStats {
            vision_patches: 5000,
            vision_images: 7,
            ..EngineStats::default()
        };
        let text = rendered(&stats);
        assert!(text.contains("scriptor_vision_patches_total{model_name=\"m\"} 5000\n"));
        assert!(text.contains("scriptor_vision_images_total{model_name=\"m\"} 7\n"));
    }

    #[test]
    fn vision_batch_patch_histogram_is_cumulative() {
        let stats = EngineStats {
            vision_patches: 3000,
            vision_batch_patch_sizes: vec![(1024, 2), (4096, 1)],
            ..EngineStats::default()
        };
        let text = rendered(&stats);
        let line = |le: &str| {
            format!("scriptor_vision_batch_patches_bucket{{model_name=\"m\",le=\"{le}\"}}")
        };
        assert!(text.contains(&format!("{} 0\n", line("512"))));
        assert!(text.contains(&format!("{} 2\n", line("1024"))));
        assert!(text.contains(&format!("{} 2\n", line("2048"))));
        assert!(text.contains(&format!("{} 3\n", line("4096"))));
        assert!(text.contains(&format!("{} 3\n", line("+Inf"))));
        assert!(text.contains("scriptor_vision_batch_patches_sum{model_name=\"m\"} 3000\n"));
        assert!(text.contains("scriptor_vision_batch_patches_count{model_name=\"m\"} 3\n"));
    }

    fn series_value(text: &str, series: &str) -> Option<u64> {
        text.lines()
            .find_map(|l| l.strip_prefix(series))
            .and_then(|v| v.trim().parse().ok())
    }

    #[test]
    fn admission_series_render_with_their_values() {
        let stats = EngineStats {
            preemptions: 5,
            preemptions_recompute: 2,
            preemptions_restart: 3,
            preemptions_yield: 9,
            restart_discarded_tokens: 41,
            kv_demand: true,
            kv_used_blocks: 12,
            kv_step_growth_blocks: 4,
            accepted_requests: 6,
            pinned_embedding_bytes: 100,
            retained_embedding_bytes: 300,
            admission_blocked: [1, 2, 3, 4, 5, 6],
            requests_rejected_queue: 7,
            ..EngineStats::default()
        };
        let text = rendered(&stats);
        for (series, want) in [
            (
                "scriptor_kv_policy{model_name=\"m\",policy=\"demand-diagnostic\"} ",
                1,
            ),
            (
                "scriptor_kv_policy{model_name=\"m\",policy=\"reserve\"} ",
                0,
            ),
            ("scriptor_kv_used_blocks{model_name=\"m\"} ", 12),
            ("scriptor_kv_step_growth_blocks{model_name=\"m\"} ", 4),
            ("scriptor_accepted_requests{model_name=\"m\"} ", 6),
            ("scriptor_retained_embedding_bytes{model_name=\"m\"} ", 300),
            ("scriptor_pinned_embedding_bytes{model_name=\"m\"} ", 100),
            (
                "scriptor_restart_discarded_tokens_total{model_name=\"m\"} ",
                41,
            ),
            (
                "scriptor_requests_rejected_queue_total{model_name=\"m\"} ",
                7,
            ),
            (
                "scriptor_preemptions_by_action_total{model_name=\"m\",action=\"restart\"} ",
                3,
            ),
            (
                "scriptor_preemptions_by_action_total{model_name=\"m\",action=\"recompute\"} ",
                2,
            ),
            (
                "scriptor_preemptions_by_action_total{model_name=\"m\",action=\"yield\"} ",
                9,
            ),
            ("scriptor_preemptions_total{model_name=\"m\"} ", 5),
            (
                "vllm:num_preemptions_total{engine=\"0\",model_name=\"m\"} ",
                5,
            ),
        ] {
            assert_eq!(series_value(&text, series), Some(want), "{series}");
        }
        for (i, reason) in [
            "slots",
            "watermark",
            "hysteresis",
            "prompt_blocks",
            "accepted",
            "embedding",
        ]
        .iter()
        .enumerate()
        {
            let series = format!(
                "scriptor_admission_blocked_steps_total{{model_name=\"m\",reason=\"{reason}\"}} "
            );
            assert_eq!(series_value(&text, &series), Some(i as u64 + 1), "{reason}");
        }
    }

    #[test]
    fn admission_series_are_present_and_zero_under_the_reserve_policy() {
        let text = rendered(&EngineStats::default());
        assert!(!text.contains("scriptor_preemptions_total{model_name=\"m\",action"));
        for series in [
            "scriptor_kv_policy{model_name=\"m\",policy=\"demand-diagnostic\"} ",
            "scriptor_kv_used_blocks{model_name=\"m\"} ",
            "scriptor_kv_step_growth_blocks{model_name=\"m\"} ",
            "scriptor_accepted_requests{model_name=\"m\"} ",
            "scriptor_retained_embedding_bytes{model_name=\"m\"} ",
            "scriptor_restart_discarded_tokens_total{model_name=\"m\"} ",
            "scriptor_requests_rejected_queue_total{model_name=\"m\"} ",
            "scriptor_preemptions_by_action_total{model_name=\"m\",action=\"restart\"} ",
            "scriptor_preemptions_by_action_total{model_name=\"m\",action=\"yield\"} ",
            "scriptor_admission_blocked_steps_total{model_name=\"m\",reason=\"watermark\"} ",
        ] {
            assert_eq!(series_value(&text, series), Some(0), "{series}");
        }
        assert_eq!(
            series_value(
                &text,
                "scriptor_kv_policy{model_name=\"m\",policy=\"reserve\"} "
            ),
            Some(1)
        );
    }

    #[test]
    fn phase_second_counters_render_and_keep_names() {
        let stats = EngineStats {
            vision_seconds: 1.5,
            prefill_seconds: 2.5,
            decode_seconds: 3.5,
            decode_prep_seconds: 0.125,
            ..EngineStats::default()
        };
        let text = rendered(&stats);
        for line in [
            "scriptor_vision_seconds_total{model_name=\"m\"} 1.5",
            "scriptor_prefill_seconds_total{model_name=\"m\"} 2.5",
            "scriptor_decode_seconds_total{model_name=\"m\"} 3.5",
            "# TYPE scriptor_decode_prep_seconds_total counter",
            "scriptor_decode_prep_seconds_total{model_name=\"m\"} 0.125",
            "# TYPE scriptor_vision_batch_size histogram",
        ] {
            assert!(text.contains(line), "missing {line}");
        }
    }

    #[test]
    fn image_and_phase_series_render_with_their_values() {
        let stats = EngineStats {
            waiting: 9,
            active_vision: 3,
            active_prefill: 2,
            active_running: 11,
            embedding_released: 5,
            embedding_released_bytes: 4096,
            images_submitted: 6,
            image_patches_submitted: 700,
            image_patch_sizes: vec![(64, 4), (512, 2)],
            image_key_unique: 4,
            image_key_repeat: 2,
            image_key_repeat_patches: 128,
            ..EngineStats::default()
        };
        let text = rendered(&stats);
        for (series, want) in [
            ("scriptor_waiting_requests{model_name=\"m\"} ", 9),
            (
                "scriptor_active_by_phase{model_name=\"m\",phase=\"vision\"} ",
                3,
            ),
            (
                "scriptor_active_by_phase{model_name=\"m\",phase=\"prefill\"} ",
                2,
            ),
            (
                "scriptor_active_by_phase{model_name=\"m\",phase=\"running\"} ",
                11,
            ),
            ("scriptor_embedding_released_total{model_name=\"m\"} ", 5),
            (
                "scriptor_embedding_released_bytes_total{model_name=\"m\"} ",
                4096,
            ),
            ("scriptor_images_submitted_total{model_name=\"m\"} ", 6),
            ("scriptor_image_key_unique_total{model_name=\"m\"} ", 4),
            ("scriptor_image_key_repeat_total{model_name=\"m\"} ", 2),
            (
                "scriptor_image_key_repeat_patches_total{model_name=\"m\"} ",
                128,
            ),
            ("scriptor_image_key_untracked_total{model_name=\"m\"} ", 0),
            (
                "scriptor_image_patches_bucket{model_name=\"m\",le=\"64\"} ",
                4,
            ),
            (
                "scriptor_image_patches_bucket{model_name=\"m\",le=\"512\"} ",
                6,
            ),
            ("scriptor_image_patches_count{model_name=\"m\"} ", 6),
            ("scriptor_image_patches_sum{model_name=\"m\"} ", 700),
        ] {
            assert_eq!(series_value(&text, series), Some(want), "{series}");
        }
    }

    #[test]
    fn prefill_pass_series_render_with_their_values() {
        let stats = EngineStats {
            prefill_pass_token_sizes: vec![(512, 3), (2048, 1)],
            prefill_pass_tokens: 2400,
            prefill_passes_below_ridge: 2,
            prefill_pass_tokens_below_ridge: 700,
            ..EngineStats::default()
        };
        let text = rendered(&stats);
        for (series, want) in [
            (
                "scriptor_prefill_pass_tokens_bucket{model_name=\"m\",le=\"256\"} ",
                0,
            ),
            (
                "scriptor_prefill_pass_tokens_bucket{model_name=\"m\",le=\"512\"} ",
                3,
            ),
            (
                "scriptor_prefill_pass_tokens_bucket{model_name=\"m\",le=\"2048\"} ",
                4,
            ),
            ("scriptor_prefill_pass_tokens_count{model_name=\"m\"} ", 4),
            ("scriptor_prefill_pass_tokens_sum{model_name=\"m\"} ", 2400),
            (
                "scriptor_prefill_passes_below_ridge_total{model_name=\"m\"} ",
                2,
            ),
            (
                "scriptor_prefill_pass_tokens_below_ridge_total{model_name=\"m\"} ",
                700,
            ),
        ] {
            assert_eq!(series_value(&text, series), Some(want), "{series}");
        }
    }
}
