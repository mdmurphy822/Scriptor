//! Environment-gated fault injection and a substitute device total.
//!
//! Every hook is off unless its variable is set, is read once, and is logged when armed:
//!
//! - `SCRIPTOR_FAULT` holds comma-separated rules: `alloc:<site>:<nth>[:oom|fatal]` makes the
//!   `nth` (1-based) allocation checked at `<site>` fail with a synthetic out-of-memory
//!   (default) or illegal-address error; `sync:<site>:hang` makes the wait at `<site>` report
//!   not-ready forever; `panic:<stage>` makes the check at `<stage>` panic.
//! - `SCRIPTOR_FAKE_DEVICE_TOTAL` is a byte count replacing the detected device total.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::error::{CUDA_ERROR_OUT_OF_MEMORY, CudaError, Result};

/// Variable holding the fault rules.
pub const FAULT_ENV: &str = "SCRIPTOR_FAULT";
/// Variable holding the substitute device total in bytes.
pub const FAKE_DEVICE_TOTAL_ENV: &str = "SCRIPTOR_FAKE_DEVICE_TOTAL";

/// `CUresult` of the injected sticky fault (`CUDA_ERROR_ILLEGAL_ADDRESS`).
const INJECTED_FATAL_CODE: i32 = 700;

/// What an armed allocation rule raises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocFault {
    /// A recoverable out-of-memory failure.
    Oom,
    /// A sticky device fault.
    Fatal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Rule {
    Alloc {
        site: String,
        nth: u64,
        kind: AllocFault,
    },
    SyncHang {
        site: String,
    },
    Panic {
        stage: String,
    },
}

/// A parsed set of fault rules with per-site allocation counters.
#[derive(Debug, Default)]
pub struct Faults {
    rules: Vec<Rule>,
    counts: Mutex<HashMap<String, u64>>,
}

impl Faults {
    /// Parses the value of `SCRIPTOR_FAULT`; the error names the offending rule.
    pub fn parse(text: &str) -> std::result::Result<Self, String> {
        let mut rules = Vec::new();
        for raw in text.split(',').map(str::trim).filter(|r| !r.is_empty()) {
            let parts: Vec<&str> = raw.split(':').collect();
            let rule = match parts.as_slice() {
                ["alloc", site, nth, rest @ ..] if rest.len() <= 1 => {
                    let nth: u64 = nth
                        .parse()
                        .ok()
                        .filter(|n| *n >= 1)
                        .ok_or_else(|| format!("{raw}: <nth> must be a positive integer"))?;
                    let kind = match rest.first().copied() {
                        None | Some("oom") => AllocFault::Oom,
                        Some("fatal") => AllocFault::Fatal,
                        Some(_) => return Err(format!("{raw}: kind must be oom or fatal")),
                    };
                    Rule::Alloc {
                        site: (*site).to_owned(),
                        nth,
                        kind,
                    }
                }
                ["sync", site, "hang"] => Rule::SyncHang {
                    site: (*site).to_owned(),
                },
                ["panic", stage] => Rule::Panic {
                    stage: (*stage).to_owned(),
                },
                _ => return Err(format!("{raw}: not a fault rule")),
            };
            rules.push(rule);
        }
        Ok(Self {
            rules,
            counts: Mutex::new(HashMap::new()),
        })
    }

    /// Whether any rule is armed.
    pub fn is_armed(&self) -> bool {
        !self.rules.is_empty()
    }

    /// Counts one allocation at `site` and fails it when an armed rule names this count.
    pub fn check_alloc(&self, site: &str) -> Result<()> {
        if !self
            .rules
            .iter()
            .any(|r| matches!(r, Rule::Alloc { site: s, .. } if s == site))
        {
            return Ok(());
        }
        let n = {
            let mut counts = self.counts.lock().unwrap_or_else(|p| p.into_inner());
            let n = counts.entry(site.to_owned()).or_insert(0);
            *n += 1;
            *n
        };
        for rule in &self.rules {
            if let Rule::Alloc { site: s, nth, kind } = rule
                && s == site
                && *nth == n
            {
                return Err(match kind {
                    AllocFault::Oom => CudaError::Cuda {
                        code: CUDA_ERROR_OUT_OF_MEMORY,
                        message: format!("injected out of memory at {site}"),
                    },
                    AllocFault::Fatal => CudaError::Cuda {
                        code: INJECTED_FATAL_CODE,
                        message: format!("injected illegal address at {site}"),
                    },
                });
            }
        }
        Ok(())
    }

    /// Whether the wait at `site` must report not-ready forever.
    pub fn sync_hangs(&self, site: &str) -> bool {
        self.rules
            .iter()
            .any(|r| matches!(r, Rule::SyncHang { site: s } if s == site))
    }

    /// Panics when a rule names `stage`.
    pub fn check_panic(&self, stage: &str) {
        if self
            .rules
            .iter()
            .any(|r| matches!(r, Rule::Panic { stage: s } if s == stage))
        {
            panic!("injected panic at {stage}");
        }
    }

    fn describe(&self) -> String {
        self.rules
            .iter()
            .map(|r| match r {
                Rule::Alloc { site, nth, kind } => format!("alloc:{site}:{nth}:{kind:?}"),
                Rule::SyncHang { site } => format!("sync:{site}:hang"),
                Rule::Panic { stage } => format!("panic:{stage}"),
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn announce(what: &str, detail: &str) {
    eprintln!("scriptor: fault injection armed: {what} {detail}");
}

/// The process-wide rules from `SCRIPTOR_FAULT`, parsed on first use. An unparsable value is logged
/// and leaves every rule off.
pub fn faults() -> &'static Faults {
    static FAULTS: OnceLock<Faults> = OnceLock::new();
    FAULTS.get_or_init(|| {
        let Ok(text) = std::env::var(FAULT_ENV) else {
            return Faults::default();
        };
        match Faults::parse(&text) {
            Ok(f) => {
                if f.is_armed() {
                    announce(FAULT_ENV, &f.describe());
                }
                f
            }
            Err(e) => {
                eprintln!("scriptor: {FAULT_ENV} ignored: {e}");
                Faults::default()
            }
        }
    })
}

/// Counts one allocation at `site` against the process-wide rules.
pub fn check_alloc(site: &str) -> Result<()> {
    faults().check_alloc(site)
}

/// Whether the wait at `site` hangs under the process-wide rules.
pub fn sync_hangs(site: &str) -> bool {
    faults().sync_hangs(site)
}

/// Panics when the process-wide rules name `stage`.
pub fn check_panic(stage: &str) {
    faults().check_panic(stage);
}

/// The device total replacing the detected one, when `SCRIPTOR_FAKE_DEVICE_TOTAL` is set and valid.
pub fn fake_device_total() -> Option<u64> {
    let raw = std::env::var(FAKE_DEVICE_TOTAL_ENV).ok()?;
    match raw.trim().parse::<u64>() {
        Ok(v) => {
            static ONCE: OnceLock<()> = OnceLock::new();
            ONCE.get_or_init(|| announce(FAKE_DEVICE_TOTAL_ENV, &format!("= {v} bytes")));
            Some(v)
        }
        Err(_) => {
            eprintln!("scriptor: {FAKE_DEVICE_TOTAL_ENV} ignored: not a byte count");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_off() {
        let f = Faults::parse("").unwrap();
        assert!(!f.is_armed());
        assert!(f.check_alloc("kv_pool").is_ok());
        assert!(!f.sync_hangs("decode"));
        f.check_panic("step");
    }

    #[test]
    fn alloc_fires_on_the_nth_call_only() {
        let f = Faults::parse("alloc:work:2").unwrap();
        assert!(f.check_alloc("work").is_ok());
        let e = f.check_alloc("work").unwrap_err();
        assert!(e.is_out_of_memory() && !e.is_fatal());
        assert!(f.check_alloc("work").is_ok());
        assert!(f.check_alloc("other").is_ok());
    }

    #[test]
    fn fatal_kind_raises_a_sticky_code() {
        let f = Faults::parse("alloc:kv_pool:1:fatal").unwrap();
        let e = f.check_alloc("kv_pool").unwrap_err();
        assert!(e.is_fatal());
        assert_eq!(e.code(), Some(700));
    }

    #[test]
    fn sync_and_panic_rules() {
        let f = Faults::parse("sync:decode:hang, panic:step").unwrap();
        assert!(f.sync_hangs("decode"));
        assert!(!f.sync_hangs("prefill"));
        f.check_panic("boot");
        assert!(std::panic::catch_unwind(|| f.check_panic("step")).is_err());
    }

    #[test]
    fn malformed_rules_are_rejected() {
        for bad in [
            "alloc:x",
            "alloc:x:0",
            "alloc:x:1:weird",
            "sync:x:go",
            "nope",
        ] {
            assert!(Faults::parse(bad).is_err(), "{bad}");
        }
    }
}
