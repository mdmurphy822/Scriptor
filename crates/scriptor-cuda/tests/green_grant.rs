//! Boot-only check of which vision/decode SM splits the driver grants as green contexts: no
//! weights and no engine. The control split 40/8 must always be granted; any other split must
//! be granted when `SCRIPTOR_REQUIRE_SPLIT` names it and is otherwise reported as skipped.
//!
//! Run with
//! `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-cuda --features cuda --profile gpu-test --test green_grant -- --ignored --nocapture --test-threads=1`.
#![cfg(feature = "cuda")]

use std::time::Instant;

use scriptor_cuda::green::device_sms;
use scriptor_cuda::{Device, GreenPartitions, REQUIRE_SPLIT_ENV, SmSplit, required_split};

/// The splits checked, as `(vision, decode)` SMs, one test each; the first is the control.
const GRANT_SPLITS: [(u32, u32); 3] = [(40, 8), (36, 12), (32, 16)];

fn check_grant(split: (u32, u32)) {
    let required = required_split(std::env::var_os(REQUIRE_SPLIT_ENV).as_deref())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        std::env::var("SCRIPTOR_GPU_TESTS").as_deref(),
        Ok("1"),
        "set SCRIPTOR_GPU_TESTS=1 to run GPU tests"
    );
    let (vision, decode) = split;
    let must_grant = split == GRANT_SPLITS[0] || required == Some(split);
    let device = Device::open().expect("gpu");
    let ctx = device.context();
    let sms = device_sms(ctx).expect("device sms");
    println!(
        "grant: device SMs {} minSmPartitionSize {} smCoscheduledAlignment {}",
        sms.sms, sms.granularity.min_partition, sms.granularity.alignment
    );
    let requested = SmSplit { vision, decode };
    let greatest = cudarc::driver::result::stream::get_priority_range().map_or(0, |(_, g)| g);
    let started = Instant::now();
    let created = GreenPartitions::create(ctx, requested, (0, greatest));
    let create_ms = started.elapsed().as_secs_f64() * 1e3;
    let partitions = match created {
        Ok(p) => p,
        Err(e) if must_grant => panic!("required split {vision}/{decode} was not granted: {e}"),
        Err(e) => {
            println!("SKIPPED: split {vision}/{decode} cannot be formed: {e}");
            return;
        }
    };
    for line in partitions.grant_log() {
        println!("grant: request {line}");
    }
    println!(
        "grant: vision context {:#x} sms={vision}, decode context {:#x} sms={decode}",
        partitions.vision_context(),
        partitions.decode_context()
    );
    assert_ne!(
        partitions.vision_context(),
        partitions.decode_context(),
        "the partitions share a context handle"
    );
    let remaining = sms.sms.saturating_sub(vision + decode);
    println!("grant: SMs outside both partitions: {remaining}");
    if split == GRANT_SPLITS[1] {
        assert_eq!(remaining, 0, "36/12 uses the whole device");
    }
    println!("grant: create_ms={create_ms:.1}");
    if must_grant {
        println!("REQUIRED split {vision}/{decode} granted");
    }
}

#[test]
#[ignore = "needs GPU"]
fn partition_grants_40_8_control() {
    check_grant(GRANT_SPLITS[0]);
}

#[test]
#[ignore = "needs GPU"]
fn partition_grants_36_12() {
    check_grant(GRANT_SPLITS[1]);
}

#[test]
#[ignore = "needs GPU"]
fn partition_grants_32_16() {
    check_grant(GRANT_SPLITS[2]);
}

#[test]
fn grant_cases_are_the_three_splits_once() {
    let mut all = GRANT_SPLITS.to_vec();
    all.sort();
    all.dedup();
    assert_eq!(all.len(), 3);
    assert_eq!(GRANT_SPLITS[0], (40, 8));
    for (vision, decode) in GRANT_SPLITS {
        assert!(SmSplit { vision, decode }.check_fits(48).is_ok());
    }
}
