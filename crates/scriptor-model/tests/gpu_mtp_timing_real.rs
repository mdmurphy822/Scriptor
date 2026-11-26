//! Old/new timing of the MTP prologue and of the whole resident MTP pass on the real checkpoint.
//!
//! Every scratch and candidate buffer is allocated before timing, and each timed enqueue is
//! exactly the route's launches: the reference is gather, norm, copy, norm, copy; the candidate
//! is the one fused kernel. Three figures are reported per route and unit, each the median of
//! twelve repetitions after three warm-up runs:
//! - `ev`: CUDA-event GPU time between two events around one eager enqueue (host launch gaps
//!   included);
//! - `wall`: synchronised wall latency of one eager enqueue through completion;
//! - `graph`: CUDA-event GPU time of one replay of the captured graph, the resident serving
//!   case.
//!
//! The binary has a 30 s budget including checkpoint load; each test prints the rows it covered.
//! Run `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda --profile gpu-test
//! --test gpu_mtp_timing_real -- --ignored --nocapture --test-threads 1`.
#![cfg(feature = "cuda")]

mod common;

use common::mtp::*;

fn timing(rows: usize) {
    with_rig(|rig| {
        let stream = rig.stream.clone();
        let mut scratch = Scratch::new(rig, rows);
        let ids = stream.clone_htod(&rig.valid_ids(rows, 0)).expect("ids");
        let hidden = rig.hidden_rows(rows, 0);
        enqueue_reference(rig, &mut scratch, &ids, &hidden);
        enqueue_fused(rig, &mut scratch, &ids, &hidden);
        stream.synchronize().expect("sync");
        let ref_graph = capture(&stream, || {
            enqueue_reference(rig, &mut scratch, &ids, &hidden);
            Ok(())
        });
        let fused_graph = capture(&stream, || {
            enqueue_fused(rig, &mut scratch, &ids, &hidden);
            Ok(())
        });
        let mut prologue = [[0.0f64; 3]; 2];
        for (i, graph) in [&ref_graph, &fused_graph].into_iter().enumerate() {
            let mut enqueue = || {
                if i == 0 {
                    enqueue_reference(rig, &mut scratch, &ids, &hidden);
                } else {
                    enqueue_fused(rig, &mut scratch, &ids, &hidden);
                }
            };
            prologue[i][0] = median_of(&mut || gpu_us(&stream, &mut enqueue));
            prologue[i][1] = median_of(&mut || wall_us(&stream, &mut enqueue));
            prologue[i][2] =
                median_of(&mut || gpu_us(&stream, &mut || graph.launch().expect("replay")));
        }

        let mut mtp_us = [[0.0f64; 3]; 2];
        let start_ids = rig.valid_ids(rows, 0);
        for (i, fused) in [false, true].into_iter().enumerate() {
            let mut route = Route::new(rig, rows, fused);
            route.prepare(rig, 0, &start_ids);
            rig.exec.set_fused_mtp_prologue(fused);
            let mut out = stream
                .alloc_zeros::<u16>(rows * rig.geometry.hidden)
                .expect("carry");
            let mut pass = |rig: &mut MtpRig| {
                run_mtp(
                    &mut rig.exec,
                    &rig.mtp,
                    &mut route.kv,
                    &route.plan,
                    &hidden,
                    &mut route.logits,
                    &mut out,
                )
                .expect("mtp pass");
            };
            pass(rig);
            stream.synchronize().expect("sync");
            let graph = capture(&stream, || {
                pass(rig);
                Ok(())
            });
            mtp_us[i][0] = median_of(&mut || gpu_us(&stream, &mut || pass(rig)));
            mtp_us[i][1] = median_of(&mut || wall_us(&stream, &mut || pass(rig)));
            mtp_us[i][2] =
                median_of(&mut || gpu_us(&stream, &mut || graph.launch().expect("replay")));
        }
        println!(
            "| {rows} | prologue ref {:.1}/{:.1}/{:.1} | fused {:.1}/{:.1}/{:.1} | MTP pass ref {:.1}/{:.1}/{:.1} | fused {:.1}/{:.1}/{:.1} | (us: ev/wall/graph)",
            prologue[0][0],
            prologue[0][1],
            prologue[0][2],
            prologue[1][0],
            prologue[1][1],
            prologue[1][2],
            mtp_us[0][0],
            mtp_us[0][1],
            mtp_us[0][2],
            mtp_us[1][0],
            mtp_us[1][1],
            mtp_us[1][2],
        );
    });
}

/// Timing at rows 1, 8, 16.
#[test]
#[ignore = "needs GPU and the checkpoint"]
fn timing_rows_1_8_16() {
    for rows in [1, 8, 16] {
        timing(rows);
    }
    check_budget("timing_rows_1_8_16");
}

/// Timing at rows 32, 64, 128.
#[test]
#[ignore = "needs GPU and the checkpoint"]
fn timing_rows_32_64_128() {
    for rows in [32, 64, 128] {
        timing(rows);
    }
    check_budget("timing_rows_32_64_128");
}
