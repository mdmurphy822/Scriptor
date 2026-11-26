//! Host memory facts read from procfs: `/proc/meminfo`, `/proc/pressure/memory` and the
//! kernel's reclaim watermark. Each reader honours an environment override naming a file (or,
//! for the watermark, a directory) so tests and fault injection can stand in for the kernel.

use std::path::PathBuf;

/// Environment variable naming a file read in place of `/proc/meminfo`.
pub const FAKE_MEMINFO_ENV: &str = "SCRIPTOR_FAKE_MEMINFO";
/// Environment variable naming a file read in place of `/proc/pressure/memory`.
pub const FAKE_PSI_ENV: &str = "SCRIPTOR_FAKE_PSI";
/// Environment variable naming a directory read in place of `/proc/sys/vm`.
pub const FAKE_VM_DIR_ENV: &str = "SCRIPTOR_FAKE_VM_DIR";

/// `MemTotal` and `MemAvailable` in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemInfo {
    pub total: u64,
    pub available: u64,
}

/// Memory-stall averages in percent over the kernel's ten-second window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Psi {
    /// Share of time at least one task stalled on memory.
    pub some_avg10: f64,
    /// Share of time every non-idle task stalled on memory.
    pub full_avg10: f64,
}

/// Paths the readers consult.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostPaths {
    pub meminfo: PathBuf,
    pub psi: PathBuf,
    pub vm_dir: PathBuf,
}

impl HostPaths {
    /// The procfs paths, each replaced by its environment override when that is set.
    pub fn from_env() -> Self {
        let pick = |var: &str, default: &str| {
            std::env::var_os(var).map_or_else(|| PathBuf::from(default), PathBuf::from)
        };
        Self {
            meminfo: pick(FAKE_MEMINFO_ENV, "/proc/meminfo"),
            psi: pick(FAKE_PSI_ENV, "/proc/pressure/memory"),
            vm_dir: pick(FAKE_VM_DIR_ENV, "/proc/sys/vm"),
        }
    }
}

/// Bytes of one `meminfo` field, `None` when absent or malformed.
fn field_bytes(meminfo: &str, key: &str) -> Option<u64> {
    let line = meminfo.lines().find_map(|l| l.strip_prefix(key))?;
    let line = line.strip_prefix(':')?;
    let mut parts = line.split_whitespace();
    let value: u64 = parts.next()?.parse().ok()?;
    match parts.next() {
        Some("kB") => value.checked_mul(1024),
        None => Some(value),
        Some(_) => None,
    }
}

/// `MemTotal` and `MemAvailable` from the text of a meminfo file.
pub fn parse_meminfo(text: &str) -> Option<MemInfo> {
    Some(MemInfo {
        total: field_bytes(text, "MemTotal")?,
        available: field_bytes(text, "MemAvailable")?,
    })
}

/// `MemAvailable` in bytes from the text of a meminfo file.
pub fn parse_mem_available(text: &str) -> Option<u64> {
    field_bytes(text, "MemAvailable")
}

/// The `some` and `full` `avg10` figures from the text of a pressure file.
pub fn parse_psi(text: &str) -> Option<Psi> {
    let avg10 = |kind: &str| {
        let line = text.lines().find(|l| l.starts_with(kind))?;
        line.split_whitespace()
            .find_map(|f| f.strip_prefix("avg10="))?
            .parse::<f64>()
            .ok()
    };
    Some(Psi {
        some_avg10: avg10("some")?,
        full_avg10: avg10("full")?,
    })
}

/// Reads the meminfo file; `None` when unreadable or malformed.
pub fn read_meminfo(paths: &HostPaths) -> Option<MemInfo> {
    parse_meminfo(&std::fs::read_to_string(&paths.meminfo).ok()?)
}

/// Reads the pressure file; `None` when unreadable or malformed.
pub fn read_psi(paths: &HostPaths) -> Option<Psi> {
    parse_psi(&std::fs::read_to_string(&paths.psi).ok()?)
}

/// The kernel's high reclaim watermark in bytes for `total` bytes of memory: the minimum free
/// reserve plus two gaps, where a gap is the larger of a quarter of the minimum and
/// `watermark_scale_factor` ten-thousandths of the total (the kernel's per-zone rule applied to
/// the whole machine).
pub fn watermark_bytes(min_free_kbytes: u64, scale_factor: u64, total: u64) -> u64 {
    let min = min_free_kbytes.saturating_mul(1024);
    let gap = (min / 4).max(total / 10_000 * scale_factor);
    min.saturating_add(gap.saturating_mul(2))
}

/// The watermark from the vm sysctl files; `None` when `min_free_kbytes` is unreadable. A
/// missing `watermark_scale_factor` counts as zero.
pub fn read_watermark(paths: &HostPaths, total: u64) -> Option<u64> {
    let read = |name: &str| -> Option<u64> {
        std::fs::read_to_string(paths.vm_dir.join(name))
            .ok()?
            .trim()
            .parse()
            .ok()
    };
    let min = read("min_free_kbytes")?;
    Some(watermark_bytes(
        min,
        read("watermark_scale_factor").unwrap_or(0),
        total,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MEMINFO: &str =
        "MemTotal:       1000 kB\nMemFree:         10 kB\nMemAvailable:    2048 kB\n";

    #[test]
    fn parses_total_and_available() {
        assert_eq!(
            parse_meminfo(MEMINFO),
            Some(MemInfo {
                total: 1000 * 1024,
                available: 2048 * 1024
            })
        );
        assert_eq!(parse_mem_available("MemAvailable: 7\n"), Some(7));
    }

    #[test]
    fn rejects_malformed_meminfo() {
        assert_eq!(parse_meminfo(""), None);
        assert_eq!(parse_meminfo("MemTotal: 1 kB\n"), None);
        assert_eq!(parse_mem_available("MemAvailable: x kB\n"), None);
        assert_eq!(parse_mem_available("MemAvailable: 1 MB\n"), None);
        assert_eq!(parse_mem_available("MemAvailableX: 1 kB\n"), None);
    }

    #[test]
    fn parses_psi() {
        let text = "some avg10=1.50 avg60=0.00 avg300=0.00 total=5\nfull avg10=0.25 avg60=0.00 avg300=0.00 total=2\n";
        assert_eq!(
            parse_psi(text),
            Some(Psi {
                some_avg10: 1.5,
                full_avg10: 0.25
            })
        );
        assert_eq!(parse_psi("some avg10=1.0\n"), None);
        assert_eq!(parse_psi("garbage"), None);
    }

    #[test]
    fn watermark_uses_the_larger_gap() {
        assert_eq!(watermark_bytes(1024, 0, 0), 1024 * 1024 + 2 * 256 * 1024);
        let total = 100_000_000;
        assert_eq!(
            watermark_bytes(1, 10, total),
            1024 + 2 * (total / 10_000 * 10)
        );
    }

    #[test]
    fn readers_follow_the_given_paths() {
        let dir = std::env::temp_dir().join(format!("scriptor-hostmem-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("meminfo"), MEMINFO).unwrap();
        std::fs::write(dir.join("min_free_kbytes"), "4\n").unwrap();
        let paths = HostPaths {
            meminfo: dir.join("meminfo"),
            psi: dir.join("missing"),
            vm_dir: dir.clone(),
        };
        assert_eq!(read_meminfo(&paths).unwrap().available, 2048 * 1024);
        assert_eq!(read_psi(&paths), None);
        assert_eq!(read_watermark(&paths, 0), Some(watermark_bytes(4, 0, 0)));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
