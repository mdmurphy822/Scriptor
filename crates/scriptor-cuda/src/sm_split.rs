//! SM partition sizes for the vision and decode green contexts: parsing of the
//! `vision=<sms>,decode=<sms>` form, the split requests to try, and the division of a driver
//! grant between the two partitions.

use std::ffi::OsStr;
use std::fmt;
use std::str::FromStr;

/// Streaming multiprocessors given to each of the two partitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmSplit {
    pub vision: u32,
    pub decode: u32,
}

/// The partition granularity a device reports for green-context SM splits: the smallest
/// partition and the alignment every partition's SM count must be a multiple of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmGranularity {
    pub min_partition: u32,
    pub alignment: u32,
}

/// What one driver split call granted: `groups` homogeneous groups of `group_sms` SMs and a
/// remaining resource of `remaining_sms` SMs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmGrant {
    pub group_sms: u32,
    pub groups: u32,
    pub remaining_sms: u32,
}

/// One split request: the per-group minimum and whether the SM co-scheduling alignment is
/// ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmAttempt {
    pub min_count: u32,
    pub ignore_coscheduling: bool,
}

/// How a grant is divided between the partitions: the leading groups of each, and whether the
/// remaining resource joins it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmAssignment {
    pub vision_groups: u32,
    pub vision_remaining: bool,
    pub decode_groups: u32,
    pub decode_remaining: bool,
}

/// Why a split text or a split against a device is refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmSplitError {
    /// The text is not `vision=N,decode=N`.
    Syntax(String),
    /// A partition has no SMs.
    Empty(&'static str),
    /// A partition is not a multiple of two SMs, the finest partition the driver forms.
    Odd { partition: &'static str, sms: u32 },
    /// The partitions together exceed the device's SMs.
    Oversubscribed { requested: u32, device: u32 },
}

impl fmt::Display for SmSplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax(text) => {
                write!(f, "invalid SM split {text:?}: expected vision=N,decode=N")
            }
            Self::Empty(name) => write!(f, "the {name} partition needs at least one SM"),
            Self::Odd { partition, sms } => {
                write!(
                    f,
                    "the {partition} partition of {sms} SMs is not a multiple of 2"
                )
            }
            Self::Oversubscribed { requested, device } => write!(
                f,
                "the partitions request {requested} SMs but the device has {device}"
            ),
        }
    }
}

impl std::error::Error for SmSplitError {}

impl FromStr for SmSplit {
    type Err = SmSplitError;

    /// Parses `vision=N,decode=N` (either order, each key exactly once).
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let bad = || SmSplitError::Syntax(text.to_string());
        let mut vision = None;
        let mut decode = None;
        for part in text.split(',') {
            let (key, value) = part.split_once('=').ok_or_else(bad)?;
            let sms: u32 = value.trim().parse().map_err(|_| bad())?;
            let slot = match key.trim() {
                "vision" => &mut vision,
                "decode" => &mut decode,
                _ => return Err(bad()),
            };
            if slot.replace(sms).is_some() {
                return Err(bad());
            }
        }
        let split = Self {
            vision: vision.ok_or_else(bad)?,
            decode: decode.ok_or_else(bad)?,
        };
        if split.vision == 0 {
            return Err(SmSplitError::Empty("vision"));
        }
        if split.decode == 0 {
            return Err(SmSplitError::Empty("decode"));
        }
        Ok(split)
    }
}

impl fmt::Display for SmSplit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vision={},decode={}", self.vision, self.decode)
    }
}

/// Environment variable naming the one `<vision>,<decode>` split a GPU gate must be granted.
pub const REQUIRE_SPLIT_ENV: &str = "SCRIPTOR_REQUIRE_SPLIT";

/// Why the text of [`REQUIRE_SPLIT_ENV`] is refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequireSplitError(pub String);

impl fmt::Display for RequireSplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{REQUIRE_SPLIT_ENV}: {}", self.0)
    }
}

impl std::error::Error for RequireSplitError {}

fn require_token(name: &str, token: &str) -> Result<u32, RequireSplitError> {
    let well_formed =
        !token.is_empty() && token.bytes().all(|b| b.is_ascii_digit()) && !token.starts_with('0');
    if !well_formed {
        return Err(RequireSplitError(format!(
            "{name} {token:?} is not a positive decimal number without sign, leading zero or separators"
        )));
    }
    let value: u32 = token
        .parse()
        .map_err(|_| RequireSplitError(format!("{name} {token:?} does not fit 32 bits")))?;
    if !value.is_multiple_of(2) {
        return Err(RequireSplitError(format!(
            "{name} {value} is not a multiple of two SMs"
        )));
    }
    Ok(value)
}

/// The `(vision, decode)` split a GPU gate must be granted, from the value of
/// [`REQUIRE_SPLIT_ENV`]: `None` when the variable is unset, otherwise exactly
/// `<vision>,<decode>` as two positive even decimal numbers with no whitespace. Any other
/// value, including an empty or non-UTF-8 one, is an error.
pub fn required_split(env: Option<&OsStr>) -> Result<Option<(u32, u32)>, RequireSplitError> {
    let Some(raw) = env else {
        return Ok(None);
    };
    let text = raw
        .to_str()
        .ok_or_else(|| RequireSplitError("value is not valid UTF-8".into()))?;
    let mut parts = text.split(',');
    let (Some(vision), Some(decode), None) = (parts.next(), parts.next(), parts.next()) else {
        return Err(RequireSplitError(format!(
            "{text:?} is not exactly <vision>,<decode>"
        )));
    };
    Ok(Some((
        require_token("vision", vision)?,
        require_token("decode", decode)?,
    )))
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

impl SmSplit {
    /// Checks that each partition is a non-empty multiple of two SMs and that the two fit a
    /// device of `device_sms` SMs.
    pub fn check_fits(&self, device_sms: u32) -> Result<(), SmSplitError> {
        for (partition, sms) in [("vision", self.vision), ("decode", self.decode)] {
            if sms == 0 {
                return Err(SmSplitError::Empty(partition));
            }
            if sms % 2 != 0 {
                return Err(SmSplitError::Odd { partition, sms });
            }
        }
        let requested = self.vision + self.decode;
        if requested > device_sms {
            return Err(SmSplitError::Oversubscribed {
                requested,
                device: device_sms,
            });
        }
        Ok(())
    }

    /// The split requests worth trying, most conservative first: the default alignment with
    /// the device's own granularity and with sizes derived from the split, then the same
    /// sizes with the co-scheduling alignment ignored.
    pub fn attempts(&self, granularity: SmGranularity) -> Vec<SmAttempt> {
        let (min, align) = (
            granularity.min_partition.max(1),
            granularity.alignment.max(1),
        );
        let unit = min / gcd(min, align) * align;
        let mut sizes = vec![
            unit,
            gcd(self.vision, self.decode),
            self.vision,
            self.decode,
        ];
        sizes.extend([8, 4, 2]);
        let mut seen = Vec::new();
        for ignore_coscheduling in [false, true] {
            for &min_count in &sizes {
                let attempt = SmAttempt {
                    min_count,
                    ignore_coscheduling,
                };
                if min_count > 0 && !seen.contains(&attempt) {
                    seen.push(attempt);
                }
            }
        }
        seen
    }

    /// Divides `grant` into the two partitions: each takes leading groups, optionally plus
    /// the remaining resource (given to at most one), so that the SM counts equal the split.
    /// `None` when no division matches.
    pub fn assign(&self, grant: &SmGrant) -> Option<SmAssignment> {
        let size = |groups: u32, remaining: bool| {
            groups * grant.group_sms + if remaining { grant.remaining_sms } else { 0 }
        };
        for (vision_remaining, decode_remaining) in [(false, false), (false, true), (true, false)] {
            for vision_groups in 0..=grant.groups {
                for decode_groups in 0..=(grant.groups - vision_groups) {
                    if size(vision_groups, vision_remaining) == self.vision
                        && size(decode_groups, decode_remaining) == self.decode
                    {
                        return Some(SmAssignment {
                            vision_groups,
                            vision_remaining,
                            decode_groups,
                            decode_remaining,
                        });
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn required(text: &str) -> Result<Option<(u32, u32)>, RequireSplitError> {
        required_split(Some(OsStr::new(text)))
    }

    #[test]
    fn require_split_unset_is_none() {
        assert_eq!(required_split(None), Ok(None));
    }

    #[test]
    fn require_split_accepts_36_12() {
        assert_eq!(required("36,12"), Ok(Some((36, 12))));
        assert_eq!(required("32,16"), Ok(Some((32, 16))));
        assert_eq!(required("40,8"), Ok(Some((40, 8))));
        assert_eq!(required("24,24"), Ok(Some((24, 24))));
    }

    #[test]
    fn require_split_rejects_malformed_text() {
        for bad in [
            "",
            "36",
            "36,",
            ",12",
            "36,12,0",
            " 36,12",
            "36, 12",
            "36,12 ",
            "36,12\n",
            "36;12",
            "36:12",
            "36/12",
            "+36,12",
            "-36,12",
            "036,12",
            "36,012",
            "0,12",
            "36,0",
            "36.0,12",
            "0x24,12",
            "36_0,12",
            "99999999999,12",
            "37,12",
            "36,13",
            "vision=36,decode=12",
            "36,12;32,16",
        ] {
            let e = required(bad).expect_err(bad);
            assert!(e.to_string().contains(REQUIRE_SPLIT_ENV), "{e}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn require_split_rejects_non_utf8() {
        use std::os::unix::ffi::OsStrExt;
        let raw = OsStr::from_bytes(b"36,1\xff");
        assert!(required_split(Some(raw)).is_err());
    }

    const GB10: SmGranularity = SmGranularity {
        min_partition: 8,
        alignment: 8,
    };

    fn split(vision: u32, decode: u32) -> SmSplit {
        SmSplit { vision, decode }
    }

    #[test]
    fn parses_either_order() {
        let a: SmSplit = "vision=32,decode=16".parse().unwrap();
        let b: SmSplit = "decode=16, vision=32".parse().unwrap();
        assert_eq!(a, split(32, 16));
        assert_eq!(a, b);
        assert_eq!(a.to_string(), "vision=32,decode=16");
    }

    #[test]
    fn rejects_malformed_text() {
        for text in [
            "",
            "vision=32",
            "decode=16",
            "vision=32,decode",
            "vision=32,decode=x",
            "vision=32,vision=16",
            "vision=32,decode=16,decode=8",
            "vision=32,other=16",
            "vision=-1,decode=16",
            "vision=32;decode=16",
        ] {
            assert!(
                matches!(text.parse::<SmSplit>(), Err(SmSplitError::Syntax(_))),
                "{text:?} was accepted"
            );
        }
    }

    #[test]
    fn rejects_empty_partitions() {
        assert_eq!(
            "vision=0,decode=16".parse::<SmSplit>(),
            Err(SmSplitError::Empty("vision"))
        );
        assert_eq!(
            "vision=16,decode=0".parse::<SmSplit>(),
            Err(SmSplitError::Empty("decode"))
        );
    }

    #[test]
    fn accepts_splits_that_fit() {
        for (v, d) in [(32, 16), (40, 8), (24, 24), (24, 16)] {
            assert_eq!(split(v, d).check_fits(48), Ok(()));
        }
    }

    #[test]
    fn rejects_odd_and_oversubscribed_partitions() {
        assert_eq!(
            split(31, 16).check_fits(48),
            Err(SmSplitError::Odd {
                partition: "vision",
                sms: 31
            })
        );
        assert_eq!(
            split(40, 16).check_fits(48),
            Err(SmSplitError::Oversubscribed {
                requested: 56,
                device: 48
            })
        );
    }

    #[test]
    fn attempts_start_with_the_device_granularity_and_never_repeat() {
        let attempts = split(32, 16).attempts(GB10);
        assert_eq!(
            attempts[0],
            SmAttempt {
                min_count: 8,
                ignore_coscheduling: false
            }
        );
        assert!(attempts.iter().any(|a| a.ignore_coscheduling));
        for (i, a) in attempts.iter().enumerate() {
            assert!(a.min_count > 0);
            assert!(!attempts[..i].contains(a), "{a:?} repeats");
        }
        let lcm = split(24, 12).attempts(SmGranularity {
            min_partition: 4,
            alignment: 6,
        });
        assert_eq!(lcm[0].min_count, 12);
    }

    #[test]
    fn assigns_groups_and_the_remaining_resource() {
        // The GB10 grant seen for minCount 8: four groups of 8 and 16 remaining.
        let grant = SmGrant {
            group_sms: 8,
            groups: 4,
            remaining_sms: 16,
        };
        assert_eq!(
            split(32, 16).assign(&grant),
            Some(SmAssignment {
                vision_groups: 4,
                vision_remaining: false,
                decode_groups: 0,
                decode_remaining: true
            })
        );
        assert_eq!(
            split(24, 24).assign(&grant),
            Some(SmAssignment {
                vision_groups: 3,
                vision_remaining: false,
                decode_groups: 1,
                decode_remaining: true
            })
        );
        assert_eq!(
            split(40, 8).assign(&grant),
            Some(SmAssignment {
                vision_groups: 3,
                vision_remaining: true,
                decode_groups: 1,
                decode_remaining: false
            })
        );
        assert_eq!(split(20, 28).assign(&grant), None);
    }

    #[test]
    fn assigns_homogeneous_groups_without_a_remainder() {
        let grant = SmGrant {
            group_sms: 8,
            groups: 6,
            remaining_sms: 0,
        };
        let a = split(40, 8).assign(&grant).unwrap();
        assert_eq!((a.vision_groups, a.decode_groups), (5, 1));
        assert!(!a.vision_remaining && !a.decode_remaining);
        assert_eq!(split(32, 24).assign(&grant), None);
    }

    #[test]
    fn never_assigns_a_group_to_both_partitions() {
        let grant = SmGrant {
            group_sms: 8,
            groups: 4,
            remaining_sms: 0,
        };
        assert_eq!(split(24, 16).assign(&grant), None);
        assert!(split(16, 16).assign(&grant).is_some());
    }
}
