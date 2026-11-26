//! Concurrency-ladder throughput measurement.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::openai_client::{ChatEndpoint, request_body};
use crate::regions::Case;

/// The concurrency levels of the standard ladder.
pub const DEFAULT_LEVELS: [usize; 4] = [32, 64, 128, 256];

/// Measurements at one concurrency level.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelResult {
    pub concurrency: usize,
    pub requests: usize,
    pub errors: usize,
    pub wall_seconds: f64,
    pub requests_per_second: f64,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub prompt_tokens_per_second: f64,
    pub generation_tokens_per_second: f64,
}

/// The `q` quantile (0..=1) of `sorted` ascending values by nearest rank; 0 when empty.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Sends `requests` cases (cycled from `cases`) with `concurrency` in flight.
pub fn run_level(
    endpoint: &dyn ChatEndpoint,
    model: &str,
    cases: &[Case],
    concurrency: usize,
    requests: usize,
) -> LevelResult {
    let next = AtomicUsize::new(0);
    let samples: Mutex<Vec<(f64, u64, u64)>> = Mutex::new(Vec::new());
    let errors = AtomicUsize::new(0);
    let start = Instant::now();
    if !cases.is_empty() {
        std::thread::scope(|scope| {
            for _ in 0..concurrency.max(1) {
                scope.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= requests {
                            break;
                        }
                        let case = &cases[i % cases.len()];
                        let body = request_body(model, case.task, &case.data_url);
                        match endpoint.chat(&body) {
                            Ok(r) => samples.lock().unwrap().push((
                                r.latency.as_secs_f64() * 1000.0,
                                r.prompt_tokens,
                                r.completion_tokens,
                            )),
                            Err(_) => {
                                errors.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                });
            }
        });
    }
    let wall = start.elapsed().as_secs_f64();
    let samples = samples.into_inner().unwrap();
    let mut lat: Vec<f64> = samples.iter().map(|s| s.0).collect();
    lat.sort_by(f64::total_cmp);
    let prompt: u64 = samples.iter().map(|s| s.1).sum();
    let gen_tokens: u64 = samples.iter().map(|s| s.2).sum();
    let per_s = |v: f64| if wall > 0.0 { v / wall } else { 0.0 };
    LevelResult {
        concurrency,
        requests: samples.len() + errors.load(Ordering::Relaxed),
        errors: errors.load(Ordering::Relaxed),
        wall_seconds: wall,
        requests_per_second: per_s(samples.len() as f64),
        p50_ms: percentile(&lat, 0.50),
        p99_ms: percentile(&lat, 0.99),
        prompt_tokens_per_second: per_s(prompt as f64),
        generation_tokens_per_second: per_s(gen_tokens as f64),
    }
}

/// Runs every level; each level sends `rounds` requests per concurrent slot.
pub fn run_ladder(
    endpoint: &dyn ChatEndpoint,
    model: &str,
    cases: &[Case],
    levels: &[usize],
    rounds: usize,
) -> Vec<LevelResult> {
    levels
        .iter()
        .map(|&c| run_level(endpoint, model, cases, c, c * rounds.max(1)))
        .collect()
}
