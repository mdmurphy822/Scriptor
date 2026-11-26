//! `scriptor doctor`: device, memory plan and cache diagnostics.

use scriptor_cuda::Device;
use scriptor_engine::boot::{
    BootError, CacheDir, DeviceFacts, HostMemory, derive_concurrency, plan_budget, plan_memory,
};

use crate::args::DoctorArgs;
use crate::setup::Setup;

/// Prints diagnostics and returns the process exit code (non-zero when the plan is refused).
pub fn run(args: &DoctorArgs) -> i32 {
    let device = match args.device_memory_gib {
        Some(gib) => {
            let bytes = gib << 30;
            println!("device      none opened; planning against {gib} GiB, all free");
            DeviceFacts {
                name: "stated".into(),
                total_bytes: bytes,
                free_bytes: bytes,
                unified_memory: false,
                host: None,
            }
        }
        None => match Device::open() {
            Ok(d) => {
                let info = d.info();
                println!(
                    "device      {} sm_{}{} {} MiB free of {} MiB",
                    info.name,
                    info.sm.0,
                    info.sm.1,
                    info.free_bytes >> 20,
                    info.total_bytes >> 20
                );
                DeviceFacts {
                    name: info.name.clone(),
                    total_bytes: info.total_bytes as u64,
                    free_bytes: info.free_bytes as u64,
                    unified_memory: info.unified_memory,
                    host: HostMemory::read(args.reserve_hint()),
                }
            }
            Err(e) => {
                println!("device      none ({e})");
                return 1;
            }
        },
    };
    match &args.cache_dir {
        Some(dir) => {
            let cache = CacheDir::new(dir);
            println!(
                "cache       explicit directory; boot record {}",
                if cache.contains(scriptor_engine::boot::BOOT_RECORD) {
                    "present"
                } else {
                    "absent"
                }
            );
        }
        None => println!("cache       none requested; nothing is persisted"),
    }
    let Some(model) = &args.model else {
        println!("model       not given; plan skipped");
        return 0;
    };
    let setup = match Setup::load(model) {
        Ok(s) => s,
        Err(e) => {
            println!("model       unreadable ({e})");
            return 1;
        }
    };
    println!(
        "model       {} tensors, {} MiB of weights, mtp layers {}",
        setup.dir.tensors().len(),
        setup.weight_bytes >> 20,
        setup.spec.text.num_mtp_layers
    );
    let facts = device;
    if let Some(h) = &facts.host {
        println!(
            "host        MemTotal {} MiB, MemAvailable {} MiB, in use {} MiB, kernel watermark {} MiB, host hint {} MiB",
            h.total >> 20,
            h.available >> 20,
            h.in_use() >> 20,
            h.watermark >> 20,
            h.reserve_hint >> 20
        );
    }
    let mut params = args.runtime_params();
    if args.max_num_seqs.is_none() {
        match derive_concurrency(
            &setup.spec,
            &setup.image,
            &params,
            setup.weight_bytes,
            &facts,
        ) {
            Ok(n) => {
                params.max_concurrency = n;
                println!("concurrency {n} streams (derived from the memory share)");
            }
            Err(e) => {
                println!("plan        refused: {e}");
                return refusal_code(&e);
            }
        }
    }
    match plan_budget(&setup.spec, &setup.image, &params, &facts) {
        Ok(b) => println!("budget      {}", b.derivation),
        Err(e) => {
            println!("budget      refused: {e}");
            return refusal_code(&e);
        }
    }
    match plan_memory(
        &setup.spec,
        &setup.image,
        &params,
        setup.weight_bytes,
        &facts,
    ) {
        Ok(p) => {
            println!(
                "plan        fits: {} MiB planned of {} MiB; {} streams",
                p.total >> 20,
                p.available >> 20,
                p.slots
            );
            println!("plan        {}", p.capacity_summary());
            0
        }
        Err(e) => {
            println!("plan        refused: {e}");
            refusal_code(&e)
        }
    }
}

/// Exit code of a refused plan: 78 for a configuration refusal, 1 otherwise.
fn refusal_code(e: &BootError) -> i32 {
    match e {
        BootError::Refused(_) => crate::exit::REFUSED,
        _ => 1,
    }
}
