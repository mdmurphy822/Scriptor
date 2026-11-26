//! Fused MTP prologue against the five-launch reference on the real checkpoint. The hidden rows
//! are the real decoder's per-row hidden states over synthetic token ids.
//!
//! - `prologue_cat_bits_*`: exact bf16 bits of the `eh_proj` input at 1, 8, 16, 32, 64 and 128
//!   rows, with ids that include zero, negative and out-of-vocabulary values (isolated prologue
//!   only).
//! - `prologue_graph_replay_*`: both prologue routes captured once into graphs at stable buffer
//!   addresses, replayed with changed ids and hidden rows, and compared with eager reference
//!   output.
//! - `carried_chain_*`: four consecutive resident MTP passes per route, each fed the previous
//!   pass's carry and its own greedy draft ids (all valid), with each route keeping its own MTP
//!   key/value pool; logits, carry and cache bits are compared after every step.
//!
//! The whole binary has a 30 s budget including checkpoint load, printed by every test. Set
//! `SCRIPTOR_SNAPSHOT_DIR` to the checkpoint directory (or leave it unset to use the Hugging
//! Face cache) and run
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda --profile gpu-test --test gpu_mtp_prologue_real -- --ignored --nocapture --test-threads 1`.
#![cfg(feature = "cuda")]

mod common;

use common::mtp::*;

const SHIFTS: [usize; 2] = [0, 37];

fn cat_bits(rows: usize) {
    with_rig(|rig| {
        let mut scratch = Scratch::new(rig, rows);
        for (pass, shift) in SHIFTS.into_iter().enumerate() {
            let ids = rig.stream.clone_htod(&rig.ids(rows, pass)).expect("ids");
            let hidden = rig.hidden_rows(rows, shift);
            enqueue_reference(rig, &mut scratch, &ids, &hidden);
            enqueue_fused(rig, &mut scratch, &ids, &hidden);
            rig.stream.synchronize().expect("sync");
            let want = rig.stream.clone_dtoh(&scratch.cat_ref).expect("dtoh");
            let got = rig.stream.clone_dtoh(&scratch.cat_fused).expect("dtoh");
            assert_eq!(
                first_diff(&want, &got),
                None,
                "rows={rows} pass={pass}: first differing element"
            );
        }
        println!("rows={rows}: cat bits identical (first and rotated-hidden passes)");
    });
}

/// The fused kernel's `eh_proj` input equals the five-op pipeline's bit for bit, rows 1, 8, 16.
#[test]
#[ignore = "needs GPU and the checkpoint"]
fn prologue_cat_bits_rows_1_8_16() {
    for rows in [1, 8, 16] {
        cat_bits(rows);
    }
    check_budget("prologue_cat_bits_rows_1_8_16");
}

/// The fused kernel's `eh_proj` input equals the five-op pipeline's bit for bit, rows 32, 64, 128.
#[test]
#[ignore = "needs GPU and the checkpoint"]
fn prologue_cat_bits_rows_32_64_128() {
    for rows in [32, 64, 128] {
        cat_bits(rows);
    }
    check_budget("prologue_cat_bits_rows_32_64_128");
}

fn graph_replay(rows: usize) {
    with_rig(|rig| {
        let mut graph_scratch = Scratch::new(rig, rows);
        let mut eager = Scratch::new(rig, rows);
        let mut ids = rig.stream.clone_htod(&rig.ids(rows, 0)).expect("ids");
        let mut hidden = rig.hidden_rows(rows, 0);
        let stream = rig.stream.clone();
        let reference = capture(&stream, || {
            enqueue_reference(rig, &mut graph_scratch, &ids, &hidden);
            Ok(())
        });
        let fused = capture(&stream, || {
            enqueue_fused(rig, &mut graph_scratch, &ids, &hidden);
            Ok(())
        });
        for round in 0..3usize {
            // New ids and hidden rows are written into the buffers the graphs were captured on.
            rig.stream
                .memcpy_htod(&rig.ids(rows, round + 3), &mut ids)
                .expect("ids");
            let fresh = rig.hidden_rows(rows, 11 * round + 5);
            rig.stream.memcpy_dtod(&fresh, &mut hidden).expect("hidden");
            reference.launch().expect("replay reference");
            fused.launch().expect("replay fused");
            enqueue_reference(rig, &mut eager, &ids, &hidden);
            rig.stream.synchronize().expect("sync");
            let want = rig.stream.clone_dtoh(&eager.cat_ref).expect("dtoh");
            for (name, cat) in [
                ("reference", &graph_scratch.cat_ref),
                ("fused", &graph_scratch.cat_fused),
            ] {
                let got = rig.stream.clone_dtoh(cat).expect("dtoh");
                assert_eq!(
                    first_diff(&want, &got),
                    None,
                    "rows={rows} round={round}: replayed {name} prologue vs eager reference"
                );
            }
        }
        println!(
            "rows={rows}: replayed reference and fused prologue match eager over 3 id/hidden updates"
        );
    });
}

/// Graph-replayed prologues at stable addresses match eager reference output, rows 1, 8, 16.
#[test]
#[ignore = "needs GPU and the checkpoint"]
fn prologue_graph_replay_rows_1_8_16() {
    for rows in [1, 8, 16] {
        graph_replay(rows);
    }
    check_budget("prologue_graph_replay_rows_1_8_16");
}

/// Graph-replayed prologues at stable addresses match eager reference output, rows 32, 64, 128.
#[test]
#[ignore = "needs GPU and the checkpoint"]
fn prologue_graph_replay_rows_32_64_128() {
    for rows in [32, 64, 128] {
        graph_replay(rows);
    }
    check_budget("prologue_graph_replay_rows_32_64_128");
}

fn carried_chain(rows: usize) {
    with_rig(|rig| {
        let h0 = rig.hidden_rows(rows, 0);
        let mut routes = [Route::new(rig, rows, false), Route::new(rig, rows, true)];
        let first = rig.valid_ids(rows, 0);
        let mut ids = [first.clone(), first];
        let mut seen: Vec<Vec<i32>> = Vec::new();
        for step in 0..CHAIN_STEPS {
            let mut logits = Vec::new();
            let mut carries = Vec::new();
            let mut caches = Vec::new();
            for (i, route) in routes.iter_mut().enumerate() {
                chain_step(rig, route, step, &ids[i], &h0);
                rig.stream.synchronize().expect("sync");
                logits.push(rig.stream.clone_dtoh(&route.logits).expect("logits"));
                carries.push(
                    rig.stream
                        .clone_dtoh(&route.carry[step % 2])
                        .expect("carry"),
                );
                caches.push(route.kv_bits(rig));
            }
            assert!(
                logits[0].iter().any(|v| *v != 0.0),
                "rows={rows} step={step}: logits are all zero"
            );
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
            assert_eq!(
                first_diff(&bits(&logits[0]), &bits(&logits[1])),
                None,
                "rows={rows} step={step}: logits bits"
            );
            assert_eq!(
                first_diff(&carries[0], &carries[1]),
                None,
                "rows={rows} step={step}: carry bits"
            );
            assert_eq!(
                first_diff(&caches[0].0, &caches[1].0),
                None,
                "rows={rows} step={step}: MTP key cache bits"
            );
            assert_eq!(
                first_diff(&caches[0].1, &caches[1].1),
                None,
                "rows={rows} step={step}: MTP value cache bits"
            );
            let next: Vec<Vec<i32>> = logits.iter().map(|l| rig.greedy(l, rows)).collect();
            assert_eq!(next[0], next[1], "rows={rows} step={step}: draft ids");
            assert!(
                next[0]
                    .iter()
                    .all(|&t| t >= 0 && (t as usize) < rig.geometry.vocab),
                "rows={rows} step={step}: draft ids are valid"
            );
            seen.push(next[0].clone());
            ids = [next[0].clone(), next[1].clone()];
        }
        let distinct = seen.windows(2).filter(|w| w[0] != w[1]).count();
        println!(
            "rows={rows}: {CHAIN_STEPS}-step carried chain identical (logits, carry, cache); \
             {distinct}/{} draft-id sets changed between steps",
            CHAIN_STEPS - 1
        );
    });
}

/// Four carried passes per route at rows 1, 8, 16.
#[test]
#[ignore = "needs GPU and the checkpoint"]
fn carried_chain_rows_1_8_16() {
    for rows in [1, 8, 16] {
        carried_chain(rows);
    }
    check_budget("carried_chain_rows_1_8_16");
}

/// Four carried passes per route at rows 32 and 64.
#[test]
#[ignore = "needs GPU and the checkpoint"]
fn carried_chain_rows_32_64() {
    for rows in [32, 64] {
        carried_chain(rows);
    }
    check_budget("carried_chain_rows_32_64");
}

/// Four carried passes per route at 128 rows.
#[test]
#[ignore = "needs GPU and the checkpoint"]
fn carried_chain_rows_128() {
    carried_chain(128);
    check_budget("carried_chain_rows_128");
}
