//! Cost measurements of the two-stage greedy target selector against the single-block kernel.
//! Three views are printed, none of them asserted against a speed target:
//!
//! * `HOST_PLUS_DEVICE_US`: wall-clock time of eager Rust launches plus a final synchronize, so
//!   it includes host launch cost and the second launch of the split selector;
//! * `EVENT_EAGER_US`: CUDA-event span of the same eager launches, which excludes the host's
//!   launch loop only when the device is the bottleneck;
//! * `GRAPH_US`: CUDA-event span of replays of one graph that captured many launches of the
//!   selection segment, the shape production runs in. Old and new alternate which goes first
//!   in every repeat, and every repeat is printed.
//!
//! Every allocation, upload and graph capture happens outside the timed spans. The effective
//! GB/s lines count the logits bytes only (`2 * rows * vocab / elapsed_ns`); the inputs are small
//! and repeated, so they may be cache-resident and the figure is not a memory-bandwidth claim.
//! The split selector's extra scratch traffic is reported on its own `SCRATCH_*` lines.
//!
//! Run with
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda --profile gpu-test --test sampling_split_timing_gpu -- --ignored --nocapture --test-threads=1`.
#![cfg(feature = "cuda")]

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::*;
use cudarc::driver::sys::{CUdevice_attribute, CUevent_flags};
use cudarc::driver::{CudaContext, CudaEvent, CudaStream};
use scriptor_cuda::{CapturedGraph, CudaError};
use scriptor_kernels::sampling::{SamplingKernels, SplitScratch, SplitSelect, TOP2_WORDS};

const CONFIGS: [(usize, usize); 5] = [(1, 0), (4, 0), (8, 0), (1, 3), (2, 3)];
/// Launches of the selection segment captured into one graph.
const LAUNCHES_PER_GRAPH: usize = 32;
/// Graph replays per timed span.
const REPLAYS: usize = 16;
/// Repeats of each old/new comparison.
const REPEATS: usize = 4;
/// Longest a single test may run.
const BUDGET: Duration = Duration::from_secs(30);

fn events(ctx: &Arc<CudaContext>) -> [CudaEvent; 2] {
    let make = || {
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event")
    };
    [make(), make()]
}

/// Microseconds the stream takes to run `f`'s enqueued work, between two recorded events.
fn event_us(s: &Arc<CudaStream>, ev: &[CudaEvent; 2], f: &mut dyn FnMut()) -> f64 {
    ev[0].record(s).expect("record");
    f();
    ev[1].record(s).expect("record");
    ev[1].synchronize().expect("sync");
    f64::from(ev[0].elapsed_ms(&ev[1]).expect("elapsed")) * 1e3
}

fn sm_geometry(ctx: &Arc<CudaContext>) -> (usize, usize) {
    let attr = |a| ctx.attribute(a).expect("attribute") as usize;
    (
        attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR),
    )
}

/// Logits-only effective GB/s of `rows` bf16 rows of `vocab` read once per `us` microseconds.
fn logits_gbps(rows: usize, vocab: usize, us: f64) -> f64 {
    (2 * rows * vocab) as f64 / (us * 1e3)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "needs GPU"]
fn split_selection_host_plus_device_wall_clock_prints_microseconds_per_call() {
    let Some((dev, s)) = setup() else { return };
    let started = Instant::now();
    let kern = SamplingKernels::load(dev.context()).expect("load");
    let alloc = Instant::now();
    let mut scratch = kern.alloc_split_scratch(&s, 8).expect("scratch");
    s.synchronize().expect("sync");
    eprintln!(
        "SPLIT_SCRATCH_ALLOC_US {:.1}",
        alloc.elapsed().as_secs_f64() * 1e6
    );
    let iters = 300;
    for (seqs, k) in CONFIGS {
        let b = text_batch(seqs, k, VOCAB, 11);
        let mut d = upload(&s, &b);
        let rows = seqs * (k + 1);
        let time = |f: &mut dyn FnMut()| {
            for _ in 0..20 {
                f();
            }
            s.synchronize().expect("sync");
            let t = Instant::now();
            for _ in 0..iters {
                f();
            }
            s.synchronize().expect("sync");
            t.elapsed().as_secs_f64() * 1e6 / f64::from(iters)
        };
        let base = time(&mut || original(&kern, &s, &b, &mut d));
        eprintln!("HOST_PLUS_DEVICE_US rows={rows} original {base:.1}");
        for (block, ctas) in SELECTS {
            let sel = select(block, ctas);
            let us = time(&mut || {
                split(&kern, &s, &sel, &mut scratch, &b, &mut d);
            });
            eprintln!(
                "HOST_PLUS_DEVICE_US rows={rows} {block}x{ctas} {us:.1} ({:.2}x)",
                base / us
            );
        }
    }
    assert!(started.elapsed() < BUDGET, "test exceeded its time budget");
}

#[test]
#[ignore = "needs GPU"]
fn split_selection_cuda_event_timing_of_eager_launches_prints_microseconds_per_call() {
    let Some((dev, s)) = setup() else { return };
    let started = Instant::now();
    let kern = SamplingKernels::load(dev.context()).expect("load");
    let ev = events(dev.context());
    let mut scratch = kern.alloc_split_scratch(&s, 8).expect("scratch");
    let iters = 200;
    for (seqs, k) in CONFIGS {
        let b = text_batch(seqs, k, VOCAB, 11);
        let mut d = upload(&s, &b);
        let rows = seqs * (k + 1);
        for _ in 0..20 {
            original(&kern, &s, &b, &mut d);
        }
        s.synchronize().expect("sync");
        let base = event_us(&s, &ev, &mut || {
            for _ in 0..iters {
                original(&kern, &s, &b, &mut d);
            }
        }) / iters as f64;
        eprintln!(
            "EVENT_EAGER_US rows={rows} original {base:.1} logits_only_effective_gbps {:.1}",
            logits_gbps(rows, VOCAB, base)
        );
        for (block, ctas) in SELECTS {
            let sel = select(block, ctas);
            let us = event_us(&s, &ev, &mut || {
                for _ in 0..iters {
                    split(&kern, &s, &sel, &mut scratch, &b, &mut d);
                }
            }) / iters as f64;
            eprintln!(
                "EVENT_EAGER_US rows={rows} {block}x{ctas} {us:.1} ({:.2}x) \
                 logits_only_effective_gbps {:.1}",
                base / us,
                logits_gbps(rows, VOCAB, us)
            );
        }
    }
    assert!(started.elapsed() < BUDGET, "test exceeded its time budget");
}

/// Captures `LAUNCHES_PER_GRAPH` launches of `launch` into a graph on `s`.
fn capture(s: &Arc<CudaStream>, mut launch: impl FnMut() -> Result<(), String>) -> CapturedGraph {
    CapturedGraph::capture(s, |_| {
        for _ in 0..LAUNCHES_PER_GRAPH {
            launch().map_err(CudaError::Graph)?;
        }
        Ok(())
    })
    .expect("capture")
}

/// Per-call microseconds of `REPLAYS` replays of `graph`, between two events.
fn graph_us(s: &Arc<CudaStream>, ev: &[CudaEvent; 2], graph: &CapturedGraph) -> f64 {
    event_us(s, ev, &mut || {
        for _ in 0..REPLAYS {
            graph.launch().expect("replay");
        }
    }) / (REPLAYS * LAUNCHES_PER_GRAPH) as f64
}

#[test]
#[ignore = "needs GPU"]
fn split_selection_captured_graph_old_versus_new_alternating_order() {
    let Some((dev, s)) = setup() else { return };
    let started = Instant::now();
    let kern = SamplingKernels::load(dev.context()).expect("load");
    let ev = events(dev.context());
    let (sms, threads) = sm_geometry(dev.context());
    let mut scratch: SplitScratch = kern.alloc_split_scratch(&s, 8).expect("scratch");
    for (seqs, k) in CONFIGS {
        let rows = seqs * (k + 1);
        let b = text_batch(seqs, k, VOCAB, 11);
        let mut d = upload(&s, &b);
        original(&kern, &s, &b, &mut d);
        let want = result(&s, &d);
        let old = capture(&s, || {
            kern.choose_spec_bf16(
                &s,
                &d.logits,
                b.seqs,
                b.k,
                b.vocab,
                &d.params,
                &d.prompt,
                &d.output,
                &d.counts,
                b.masks.words(),
                &d.drafts,
                &mut d.tokens,
                &mut d.margins,
            )
            .map_err(|e| e.to_string())
        });
        s.synchronize().expect("sync");
        for (block, ctas) in SELECTS {
            let sel: SplitSelect = select(block, ctas);
            let Some(geometry) = sel.geometry(rows, sms, threads) else {
                continue;
            };
            let new = capture(&s, || {
                kern.choose_spec_bf16_split(
                    &s,
                    &sel,
                    &mut scratch,
                    &d.logits,
                    b.seqs,
                    b.k,
                    b.vocab,
                    &d.params,
                    &d.prompt,
                    &d.output,
                    &d.counts,
                    b.masks.words(),
                    &d.drafts,
                    &mut d.tokens,
                    &mut d.margins,
                )
                .map_err(|e| e.to_string())
                .and_then(|ran| {
                    ran.then_some(())
                        .ok_or_else(|| "split declined".to_string())
                })
            });
            s.synchronize().expect("sync");
            for graph in [&old, &new] {
                for _ in 0..4 {
                    graph.launch().expect("warm");
                }
            }
            s.synchronize().expect("sync");
            let (mut olds, mut news) = (Vec::new(), Vec::new());
            for repeat in 0..REPEATS {
                let (o, n);
                if repeat % 2 == 0 {
                    o = graph_us(&s, &ev, &old);
                    n = graph_us(&s, &ev, &new);
                } else {
                    n = graph_us(&s, &ev, &new);
                    o = graph_us(&s, &ev, &old);
                }
                eprintln!(
                    "GRAPH_US rows={rows} {block}x{ctas} repeat {repeat} order {} old {o:.1} new {n:.1}",
                    if repeat % 2 == 0 {
                        "old-first"
                    } else {
                        "new-first"
                    }
                );
                olds.push(o);
                news.push(n);
            }
            let (o, n) = (median(&mut olds), median(&mut news));
            let scratch_bytes = geometry.partial_words * 4;
            eprintln!(
                "GRAPH_US_MEDIAN rows={rows} {block}x{ctas} old {o:.1} new {n:.1} ({:.2}x) \
                 logits_only_effective_gbps old {:.1} new {:.1} \
                 (cache-resident inputs; not LPDDR bandwidth)",
                o / n,
                logits_gbps(rows, VOCAB, o),
                logits_gbps(rows, VOCAB, n)
            );
            eprintln!(
                "SCRATCH_TRAFFIC rows={rows} {block}x{ctas} written_bytes {scratch_bytes} \
                 read_bytes {scratch_bytes} words_per_partial {TOP2_WORDS} \
                 extra_gbps_at_new {:.2}",
                (2 * scratch_bytes) as f64 / (n * 1e3)
            );
            assert_eq!(
                result(&s, &d),
                want,
                "graph replay of the split selector diverged"
            );
        }
        assert!(started.elapsed() < BUDGET, "test exceeded its time budget");
    }
}
