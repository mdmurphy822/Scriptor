//! Memory planning for a boot: the early refusal point.

use scriptor_spec::{ImageProcessorConfig, MemoryPlan, MemoryPlanInputs, ModelSpec, RuntimeParams};

use super::BootError;
use crate::host_mem::{self, HostPaths};

pub use crate::host_mem::parse_mem_available;

/// Host memory facts the boot budget derives from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemory {
    /// `MemTotal` in bytes.
    pub total: u64,
    /// `MemAvailable` in bytes at boot.
    pub available: u64,
    /// The kernel's high reclaim watermark in bytes.
    pub watermark: u64,
    /// Bytes the host asked the engine to leave for other tenants (zero when not given).
    pub reserve_hint: u64,
}

impl HostMemory {
    /// Reads the host facts from procfs (or the `SCRIPTOR_FAKE_*` overrides); `None` when meminfo or
    /// the watermark is unreadable.
    pub fn read(reserve_hint: u64) -> Option<Self> {
        Self::read_from(&HostPaths::from_env(), reserve_hint)
    }

    /// As [`HostMemory::read`] with explicit paths.
    pub fn read_from(paths: &HostPaths, reserve_hint: u64) -> Option<Self> {
        let info = host_mem::read_meminfo(paths)?;
        Some(Self {
            total: info.total,
            available: info.available,
            watermark: host_mem::read_watermark(paths, info.total)?,
            reserve_hint,
        })
    }

    /// Bytes the rest of the machine holds at boot: `MemTotal - MemAvailable`.
    pub fn in_use(&self) -> u64 {
        self.total.saturating_sub(self.available)
    }

    /// Host reserve without the engine's own host footprint: bytes in use by everything else,
    /// the kernel watermark and the host's hint.
    pub fn base_reserve(&self) -> u64 {
        self.in_use()
            .saturating_add(self.watermark)
            .saturating_add(self.reserve_hint)
    }
}

/// Device facts the plan depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceFacts {
    /// Device name for logs.
    pub name: String,
    /// Total device memory in bytes.
    pub total_bytes: u64,
    /// Device memory currently free in bytes; a log figure on a unified-memory device.
    pub free_bytes: u64,
    /// Whether the device shares memory with the host.
    pub unified_memory: bool,
    /// Host memory facts, when they could be read.
    pub host: Option<HostMemory>,
}

/// The engine's own host-side footprint, itemised in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineHostBytes {
    /// Host copy of the per-step logits rows.
    pub logits_readback: u64,
    /// Decoded image patch payloads held for as many requests as there are streams.
    pub image_payloads: u64,
    /// Concurrent streams the footprint is sized for.
    pub streams: u64,
}

impl EngineHostBytes {
    /// Sum of the items.
    pub fn total(&self) -> u64 {
        self.logits_readback.saturating_add(self.image_payloads)
    }

    /// The footprint one stream adds: `total / streams`, rounded up (the whole total when
    /// no stream count is set).
    pub fn per_stream(&self) -> u64 {
        self.total().div_ceil(self.streams.max(1))
    }
}

/// Itemises the engine's host footprint for `params`: logits read-back of `rows * vocab` f32
/// values and one decoded f32 patch payload of the largest accepted image per stream.
pub fn engine_host_bytes(
    spec: &ModelSpec,
    image: &ImageProcessorConfig,
    params: &RuntimeParams,
) -> EngineHostBytes {
    let image = match params.max_pixels {
        Some(p) => image.with_max_pixels(p).unwrap_or_else(|_| image.clone()),
        None => image.clone(),
    };
    let streams = params.max_concurrency as u64;
    let rows = streams * (1 + params.speculative_tokens as u64);
    let patches = image.max_image_tokens() as u64 * spec.vision.merge_factor() as u64;
    EngineHostBytes {
        logits_readback: rows * spec.text.vocab_size as u64 * 4,
        image_payloads: streams * patches * spec.vision.patch_dim() as u64 * 4,
        streams,
    }
}

/// The memory the plan may fill and how it was derived.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootBudget {
    /// Bytes the plan may total.
    pub bytes: u64,
    /// One line stating the budget, its formula and each input.
    pub derivation: String,
}

fn gib(bytes: u64) -> String {
    format!("{:.2}", bytes as f64 / (1u64 << 30) as f64)
}

/// Derives the boot budget. On a unified-memory device,
/// `min(gpu_share * MemTotal, MemAvailable - host_reserve)` where `host_reserve` is the bytes
/// the rest of the machine already uses, the kernel watermark, the engine's own host footprint
/// and the host's hint; device-reported free memory is never an input there. On a discrete
/// device, `min(gpu_share * device total, device free)`. Refuses when host memory cannot be
/// read on a unified-memory device or the reserve consumes all available memory.
pub fn boot_budget(
    device: &DeviceFacts,
    params: &RuntimeParams,
    engine_host: &EngineHostBytes,
) -> Result<BootBudget, BootError> {
    if !device.unified_memory {
        let share = (device.total_bytes as f64 * params.gpu_share) as u64;
        let bytes = share.min(device.free_bytes);
        return Ok(BootBudget {
            bytes,
            derivation: format!(
                "budget {} GiB = min({:.2}x{} GiB device total, {} GiB device free)",
                gib(bytes),
                params.gpu_share,
                gib(device.total_bytes),
                gib(device.free_bytes)
            ),
        });
    }
    let host = device.host.ok_or_else(|| {
        BootError::Refused(
            "host memory (MemTotal, MemAvailable, vm watermark) is unreadable, so the budget on              a unified-memory device cannot be derived"
                .into(),
        )
    })?;
    let share = (host.total as f64 * params.gpu_share) as u64;
    let reserve = host.base_reserve().saturating_add(engine_host.total());
    let headroom = host.available.saturating_sub(reserve);
    let bytes = share.min(headroom);
    let derivation = format!(
        "budget {} GiB = min({:.2}x{} GiB, {} GiB - ({} + {} + {} + {}) GiB); share, MemAvailable,          in use, kernel watermark, engine host, host hint (device free {} GiB not used)",
        gib(bytes),
        params.gpu_share,
        gib(host.total),
        gib(host.available),
        gib(host.in_use()),
        gib(host.watermark),
        gib(engine_host.total()),
        gib(host.reserve_hint),
        gib(device.free_bytes)
    );
    if headroom == 0 {
        return Err(BootError::Refused(format!(
            "the host reserve leaves nothing for the engine: {derivation}"
        )));
    }
    Ok(BootBudget { bytes, derivation })
}

/// The budget for `params` with the engine host footprint derived from them.
pub fn plan_budget(
    spec: &ModelSpec,
    image: &ImageProcessorConfig,
    params: &RuntimeParams,
    device: &DeviceFacts,
) -> Result<BootBudget, BootError> {
    boot_budget(device, params, &engine_host_bytes(spec, image, params))
}

/// The concurrency that fills the memory share: the largest count for which the plan fits
/// within the boot budget, the budget being re-derived for the count (the engine's host
/// footprint grows with it) until the count and the budget agree.
pub fn derive_concurrency(
    spec: &ModelSpec,
    image: &ImageProcessorConfig,
    params: &RuntimeParams,
    weight_bytes: u64,
    device: &DeviceFacts,
) -> Result<usize, BootError> {
    let derive = |budget: &BootBudget| {
        MemoryPlan::derive_concurrency(
            spec,
            image,
            params,
            MemoryPlanInputs {
                weight_bytes,
                device_total_bytes: device.total_bytes,
                free_bytes: Some(budget.bytes),
            },
        )
        .map_err(|e| BootError::Refused(format!("{e}; {}", budget.derivation)))
    };
    let mut at = params.clone();
    at.max_concurrency = 1;
    let mut n = derive(&plan_budget(spec, image, &at, device)?)?;
    while n > 1 {
        at.max_concurrency = n;
        let next = derive(&plan_budget(spec, image, &at, device)?)?;
        if next >= n {
            break;
        }
        n = next;
    }
    Ok(n)
}

/// Computes the plan within the boot budget; refuses, with the budget derivation and the
/// plan's suggestion, when not even one full-context sequence fits.
pub fn plan_memory(
    spec: &ModelSpec,
    image: &ImageProcessorConfig,
    params: &RuntimeParams,
    weight_bytes: u64,
    device: &DeviceFacts,
) -> Result<MemoryPlan, BootError> {
    let budget = plan_budget(spec, image, params, device)?;
    let plan = MemoryPlan::compute(
        spec,
        image,
        params,
        MemoryPlanInputs {
            weight_bytes,
            device_total_bytes: device.total_bytes,
            free_bytes: Some(budget.bytes),
        },
    )
    .map_err(|e| BootError::Refused(format!("{e}; {}", budget.derivation)))?;
    if plan.total > budget.bytes {
        return Err(BootError::Refused(format!(
            "the plan needs {} MiB but the budget is {} MiB; lower --max-num-seqs or \
             --max-model-len, or free memory held by other processes ({})",
            plan.total >> 20,
            budget.bytes >> 20,
            budget.derivation
        )));
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(total: u64, available: u64, watermark: u64, hint: u64) -> HostMemory {
        HostMemory {
            total,
            available,
            watermark,
            reserve_hint: hint,
        }
    }

    fn facts(free: u64, unified: bool, host: Option<HostMemory>) -> DeviceFacts {
        DeviceFacts {
            name: "d".into(),
            total_bytes: 1000,
            free_bytes: free,
            unified_memory: unified,
            host,
        }
    }

    fn params(share: f64) -> RuntimeParams {
        RuntimeParams {
            gpu_share: share,
            ..RuntimeParams::default()
        }
    }

    const NO_HOST_BYTES: EngineHostBytes = EngineHostBytes {
        logits_readback: 0,
        image_payloads: 0,
        streams: 1,
    };

    #[test]
    fn unified_budget_is_share_when_headroom_is_larger() {
        let h = host(1000, 900, 10, 0);
        let b = boot_budget(&facts(1, true, Some(h)), &params(0.5), &NO_HOST_BYTES).unwrap();
        assert_eq!(b.bytes, 500);
    }

    #[test]
    fn unified_budget_subtracts_the_reserve_from_available() {
        let h = host(1000, 600, 50, 30);
        let engine = EngineHostBytes {
            logits_readback: 10,
            image_payloads: 5,
            streams: 3,
        };
        let b = boot_budget(&facts(9_999, true, Some(h)), &params(0.9), &engine).unwrap();
        assert_eq!(b.bytes, 600 - (400 + 50 + 15 + 30));
        assert!(b.derivation.contains("device free"));
    }

    #[test]
    fn unified_budget_ignores_device_free() {
        let h = host(1000, 900, 0, 0);
        let low = boot_budget(&facts(1, true, Some(h)), &params(0.5), &NO_HOST_BYTES).unwrap();
        let high = boot_budget(&facts(999, true, Some(h)), &params(0.5), &NO_HOST_BYTES).unwrap();
        assert_eq!(low.bytes, high.bytes);
    }

    #[test]
    fn reserve_is_monotone() {
        let base = host(1000, 800, 10, 0);
        let more = host(1000, 800, 10, 100);
        let a = boot_budget(&facts(0, true, Some(base)), &params(1.0), &NO_HOST_BYTES).unwrap();
        let b = boot_budget(&facts(0, true, Some(more)), &params(1.0), &NO_HOST_BYTES).unwrap();
        assert!(b.bytes < a.bytes);
    }

    #[test]
    fn unified_without_host_facts_refuses() {
        let e = boot_budget(&facts(100, true, None), &params(0.5), &NO_HOST_BYTES).unwrap_err();
        assert!(matches!(e, BootError::Refused(_)));
    }

    #[test]
    fn exhausted_headroom_refuses_with_derivation() {
        let h = host(1000, 300, 10, 0);
        let e = boot_budget(&facts(0, true, Some(h)), &params(0.5), &NO_HOST_BYTES).unwrap_err();
        let BootError::Refused(m) = e else {
            panic!("expected a refusal")
        };
        assert!(m.contains("budget"));
    }

    #[test]
    fn discrete_budget_is_min_of_share_and_device_free() {
        let b = boot_budget(&facts(300, false, None), &params(0.5), &NO_HOST_BYTES).unwrap();
        assert_eq!(b.bytes, 300);
        let b = boot_budget(&facts(900, false, None), &params(0.5), &NO_HOST_BYTES).unwrap();
        assert_eq!(b.bytes, 500);
    }

    #[test]
    fn host_memory_reads_fake_files() {
        let dir = std::env::temp_dir().join(format!("scriptor-plan-host-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("meminfo"),
            "MemTotal: 1000 kB\nMemAvailable: 600 kB\n",
        )
        .unwrap();
        std::fs::write(dir.join("min_free_kbytes"), "4\n").unwrap();
        let paths = HostPaths {
            meminfo: dir.join("meminfo"),
            psi: dir.join("psi"),
            vm_dir: dir.clone(),
        };
        let h = HostMemory::read_from(&paths, 7).unwrap();
        assert_eq!(h.total, 1000 * 1024);
        assert_eq!(h.in_use(), 400 * 1024);
        assert_eq!(h.reserve_hint, 7);
        assert_eq!(h.watermark, host_mem::watermark_bytes(4, 0, 0));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
