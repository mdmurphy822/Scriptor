//! GPU tests of the L1 counters: vision events and host callbacks leave embeddings bitwise
//! unchanged, host callback stamps bracket the device work, and the step host sample is
//! consistent.
//!
//! The vision tests load the checkpoint named by `SCRIPTOR_MODEL_DIR`. Run with
//! `SCRIPTOR_GPU_TESTS=1 SCRIPTOR_MODEL_DIR=<snapshot> HF_HUB_OFFLINE=1 cargo test -p scriptor-cli --features cuda --profile gpu-test l1_ -- --ignored --nocapture --test-threads=1`.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use scriptor_cache::{ImageGrid, embedding_key};
use scriptor_cuda::Device;
use scriptor_engine::l1_counters::{L1Ledger, L1Switches, VisionTiming, epoch_ns};
use scriptor_engine::{DecodeRow, DeviceRow, ModelExecutor, VisionJob};
use scriptor_image::PATCH_ELEMS;
use scriptor_kernels::heuristics::TuningTable;
use scriptor_spec::{MemoryPlan, ModelSpec};
use scriptor_weights::SafetensorsDir;

use super::{BLOCK, Rng, Seq, defer, executor, fixture, gpu_enabled, sequences};
use crate::device_step::StepLimits;
use crate::gpu_executor::{GpuExecutor, GpuWeights};

/// Image grids of the vision batch, in patch units.
const GRIDS: [(u32, u32, u32); 2] = [(1, 16, 16), (1, 8, 12)];
/// Patches the vision workspace holds; covers both grids.
const VISION_BATCH_PATCHES: usize = 2048;
/// Image grids of the stamp-sign flights, in patch units: 64, 4096 and 47104 patches.
const SIGN_GRIDS: [(u32, u32, u32); 3] = [(1, 8, 8), (1, 64, 64), (1, 184, 256)];
/// Upper bound in nanoseconds on the host callback latency of a flight on a quiet GPU.
const CB_LAT_MAX_NS: i64 = 5_000_000;
/// Device steps of the sync-span test.
const STEPS: usize = 50;
/// Slack in nanoseconds between the device-event span and the host stamp span.
const STAMP_SLACK_NS: u64 = 1_000_000;

/// Prints a flushed phase marker with the seconds elapsed since `since`.
fn phase(name: &str, since: Instant) {
    println!(
        "phase {name} elapsed_s={:.3}",
        since.elapsed().as_secs_f64()
    );
    let _ = std::io::stdout().flush();
}

/// An executor over the real checkpoint named by `SCRIPTOR_MODEL_DIR`.
fn real_executor() -> GpuExecutor {
    real_executor_with(VISION_BATCH_PATCHES)
}

/// An executor over the real checkpoint whose vision workspace holds `batch_patches` patches.
fn real_executor_with(batch_patches: usize) -> GpuExecutor {
    assert!(gpu_enabled(), "set SCRIPTOR_GPU_TESTS=1 to run GPU tests");
    let dir =
        PathBuf::from(std::env::var("SCRIPTOR_MODEL_DIR").expect("SCRIPTOR_MODEL_DIR must be set"));
    let config = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let generation = std::fs::read_to_string(dir.join("generation_config.json")).ok();
    let spec = ModelSpec::from_json(&config, generation.as_deref()).expect("model spec");
    let device = Device::open().expect("gpu");
    let stream = device.new_stream().expect("stream");
    let files = SafetensorsDir::open(&dir).expect("safetensors");
    let weights = GpuWeights::load(&stream, &files, &spec, true).expect("weights");
    let plan = MemoryPlan {
        weights: 0,
        kv_cache: 0,
        mtp_kv_cache: 0,
        embedding_cache: 0,
        vision_workspace: 0,
        logits_workspace: 0,
        step_workspace: 0,
        graph_memory: 0,
        total: 0,
        available: 0,
        kv_blocks: 10,
        blocks_per_seq: 4,
        full_context_seqs: 2,
        max_context: 0,
        slots: 2,
        kv_block_tokens: BLOCK,
        vision_batch_patches: batch_patches,
    };
    GpuExecutor::new(
        &device,
        &stream,
        weights,
        &spec,
        &plan,
        0,
        TuningTable::default(),
        StepLimits {
            max_seqs: 2,
            max_context: 64,
        },
    )
    .expect("executor")
}

/// Two seeded image jobs of different grids with deterministic bf16 patch payloads.
fn vision_jobs() -> Vec<VisionJob> {
    jobs_for(&GRIDS)
}

/// One seeded image job per grid with deterministic bf16 patch payloads.
fn jobs_for(grids: &[(u32, u32, u32)]) -> Vec<VisionJob> {
    let mut rng = Rng(0x51ab_1e5c_0ffe_e123);
    grids
        .iter()
        .enumerate()
        .map(|(i, &(t, h, w))| {
            let patches = (t * h * w) as usize;
            let mut bytes = Vec::with_capacity(patches * PATCH_ELEMS * 2);
            for _ in 0..patches * PATCH_ELEMS {
                let bits = half::bf16::from_f32(rng.next() * 0.5).to_bits();
                bytes.extend_from_slice(&bits.to_le_bytes());
            }
            let grid = ImageGrid { t, h, w };
            VisionJob {
                seq: 1 + i as u64,
                image: 0,
                key: embedding_key(&bytes, grid),
                grid,
                patches,
                payload: Arc::new(bytes),
            }
        })
        .collect()
}

/// Runs the jobs through the executor and returns each embedding's bf16 words.
fn embed(exec: &mut GpuExecutor, jobs: &[VisionJob]) -> Vec<Vec<u16>> {
    let outs = exec.vision(jobs).expect("vision");
    outs.iter()
        .map(|e| exec.embedding_bits(e).expect("download"))
        .collect()
}

#[test]
#[ignore = "needs GPU"]
fn l1_flight_events_leave_embeddings_bitwise_equal() {
    let t0 = Instant::now();
    let mut exec = real_executor();
    phase("loaded", t0);
    let jobs = vision_jobs();
    exec.set_l1_probes(false);
    let off = embed(&mut exec, &jobs);
    phase("off", t0);
    exec.set_l1_probes(true);
    let on = embed(&mut exec, &jobs);
    phase("on", t0);
    exec.set_l1_probes(false);
    let off_again = embed(&mut exec, &jobs);
    phase("off_again", t0);
    assert!(off.iter().all(|e| !e.is_empty()));
    assert_eq!(off, on, "L1 events changed an embedding");
    assert_eq!(off, off_again, "the embeddings are not reproducible");
}

#[test]
#[ignore = "needs GPU"]
fn l1_host_callback_stamps_precede_the_done_event() {
    let t0 = Instant::now();
    let mut exec = real_executor();
    phase("loaded", t0);
    let jobs = vision_jobs();
    exec.set_l1_probes(true);
    let begun = epoch_ns();
    let token = exec.vision_begin(&jobs).expect("vision_begin");
    let outs = exec.vision_finish(token).expect("vision_finish");
    let finished = epoch_ns();
    phase("flight", t0);
    assert_eq!(outs.len(), jobs.len());
    let VisionTiming {
        copy_ns,
        device_ns,
        start_host_ns,
        ready_host_ns,
    } = exec.take_vision_timing(token).expect("timing recorded");
    let start = start_host_ns.expect("start stamp set before the done event completed");
    let ready = ready_host_ns.expect("ready stamp set before the done event completed");
    assert!(begun <= start, "start stamp precedes the launch");
    assert!(start <= ready, "ready stamp precedes the start stamp");
    assert!(ready <= finished, "ready stamp follows the finish");
    assert!(copy_ns <= device_ns);
    assert!(device_ns > 0);
    assert!(
        device_ns <= ready - start + STAMP_SLACK_NS,
        "device span {device_ns} ns exceeds the stamp span {} ns",
        ready - start
    );
    assert!(
        exec.take_vision_timing(token).is_none(),
        "timing is taken once"
    );
}

/// Value of the `key` field of a flight line.
fn line_field(line: &str, key: &str) -> i64 {
    line.split_whitespace()
        .find_map(|t| t.strip_prefix(key))
        .unwrap_or_else(|| panic!("{key} missing in {line}"))
        .parse()
        .unwrap_or_else(|_| panic!("{key} is not an integer in {line}"))
}

#[test]
#[ignore = "needs GPU"]
fn l1_cb_latency_sign_matches_raw_stamps() {
    let t0 = Instant::now();
    let mut exec = real_executor_with(47_104);
    phase("loaded", t0);
    exec.set_l1_probes(true);
    let mut ledger = L1Ledger::new(L1Switches {
        counters: true,
        step_lines: false,
        pass_lines: false,
    });
    for (i, &grid) in SIGN_GRIDS.iter().enumerate() {
        let jobs = jobs_for(&[grid]);
        let begun = epoch_ns();
        let token = exec.vision_begin(&jobs).expect("vision_begin");
        let returned = epoch_ns();
        let outs = exec.vision_finish(token).expect("vision_finish");
        let finished = epoch_ns();
        assert_eq!(outs.len(), 1);
        let timing = exec.take_vision_timing(token).expect("timing recorded");
        let start = timing.start_host_ns.expect("start stamp");
        let ready = timing.ready_host_ns.expect("ready stamp");
        let id = ledger.on_flight_begin(begun, returned, &jobs);
        ledger.on_flight_observed(id, finished, Some(timing), 0);
        let line = ledger.drain_lines().remove(0);
        println!("{line}");
        let cb_lat = (ready as i64 - start as i64) - timing.device_ns as i64;
        let start_rel = start as i64 - returned as i64;
        assert!(
            (line_field(&line, "cb_lat_us=") - cb_lat / 1_000).abs() <= 1,
            "cb_lat_us disagrees with the raw stamps ({cb_lat} ns): {line}"
        );
        assert!(
            (line_field(&line, "start_rel_us=") - start_rel / 1_000).abs() <= 1,
            "start_rel_us disagrees with the raw stamps ({start_rel} ns): {line}"
        );
        assert!(
            cb_lat >= -(STAMP_SLACK_NS as i64),
            "flight {i}: cb_lat {cb_lat} ns below -1 ms"
        );
        assert!(
            cb_lat <= CB_LAT_MAX_NS,
            "flight {i}: cb_lat {cb_lat} ns above 5 ms"
        );
        if i == SIGN_GRIDS.len() - 1 {
            assert!(
                start < returned,
                "start stamp {start} does not precede the launch return {returned}"
            );
            assert!(line_field(&line, "start_rel_us=") < 0, "{line}");
        }
        phase("flight", t0);
    }
}

/// Largest cpu index the machine has.
fn max_cpu_index() -> u32 {
    std::fs::read_dir("/sys/devices/system/cpu")
        .expect("cpu directory")
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            name.strip_prefix("cpu")?.parse::<u32>().ok()
        })
        .max()
        .expect("at least one cpu")
}

#[test]
#[ignore = "needs GPU"]
fn l1_sync_span_cpu_time_is_bounded_by_wall() {
    let t0 = Instant::now();
    let fx = fixture("l1-sync");
    let mut exec = executor(&fx, 0);
    exec.set_l1_probes(true);
    phase("loaded", t0);
    let mut seqs: Vec<Seq> = sequences(3);
    defer(&mut exec, &mut seqs);
    let max_cpu = max_cpu_index();
    for step in 0..STEPS {
        let rows: Vec<DecodeRow> = seqs.iter().map(Seq::row).collect();
        let results = {
            let device_rows: Vec<DeviceRow<'_>> = seqs
                .iter()
                .zip(&rows)
                .map(|(s, row)| DeviceRow {
                    row,
                    params: s.params(),
                    masks: &s.masks,
                })
                .collect();
            assert!(exec.decode_device_begin(&device_rows).expect("begin"));
            exec.decode_device_finish().expect("finish")
        };
        let sample = exec.take_host_sample().expect("host sample on every step");
        assert!(
            sample.sync_cpu_ns <= sample.sync_wall_ns,
            "step {step}: sync cpu {} ns exceeds wall {} ns",
            sample.sync_cpu_ns,
            sample.sync_wall_ns
        );
        assert!(sample.cpu <= max_cpu, "step {step}: cpu {}", sample.cpu);
        assert!(exec.take_host_sample().is_none(), "sample is taken once");
        for (s, r) in seqs.iter_mut().zip(&results) {
            for &t in &r.tokens {
                s.push(t);
            }
        }
    }
    phase("steps", t0);
    exec.set_l1_probes(false);
}
