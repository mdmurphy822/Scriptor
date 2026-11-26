//! Per-request speculative acceptance log: one line per finished request, off by default.

use std::collections::BTreeMap;

use crate::sequence::RequestId;

/// Environment variable that turns the log on with `1` (`0` or unset leaves it off).
pub const SPEC_REQUEST_LOG_ENV: &str = "SCRIPTOR_SPEC_REQUEST_LOG";

const PRUNE_ABOVE: usize = 4096;

#[derive(Debug, Clone, Copy)]
struct Shape {
    prompt_tokens: usize,
    images: usize,
    completion_tokens: usize,
}

#[derive(Debug, Default, Clone)]
struct RequestSpec {
    closed: Option<Shape>,
    verifications: u64,
    drafted: u64,
    accepted: u64,
    per_pos: Vec<u64>,
}

/// Speculative verification outcomes accumulated per live request.
#[derive(Debug, Default, Clone)]
pub struct SpecLog {
    enabled: bool,
    deferring: bool,
    requests: BTreeMap<RequestId, RequestSpec>,
}

/// Parses the log switch: `1` enables, `0`, empty or unset disables, anything else is an error
/// naming the variable.
pub fn parse_enabled(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{SPEC_REQUEST_LOG_ENV}={other:?}: expected 0 or 1")),
    }
}

/// The log switch from the process environment.
pub fn enabled_from_env() -> Result<bool, String> {
    parse_enabled(std::env::var(SPEC_REQUEST_LOG_ENV).ok().as_deref())
}

impl SpecLog {
    /// A log that records when `enabled`.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            deferring: false,
            requests: BTreeMap::new(),
        }
    }

    /// A log enabled by the environment; a malformed value leaves it off.
    pub fn from_env() -> Self {
        Self::new(enabled_from_env().unwrap_or(false))
    }

    /// Adds one verification of `drafted` tokens, `accepted` of which matched, to request `id`.
    /// Entries of requests that never finish are dropped once `live` no longer lists them.
    pub fn record(
        &mut self,
        id: RequestId,
        drafted: usize,
        accepted: usize,
        live: impl Fn(RequestId) -> bool,
    ) {
        if !self.enabled {
            return;
        }
        if self.requests.len() > PRUNE_ABOVE {
            self.requests.retain(|&k, _| live(k));
        }
        let r = self.requests.entry(id).or_default();
        r.verifications += 1;
        r.drafted += drafted as u64;
        r.accepted += accepted as u64;
        if r.per_pos.len() < drafted {
            r.per_pos.resize(drafted, 0);
        }
        for slot in r.per_pos.iter_mut().take(accepted) {
            *slot += 1;
        }
    }

    /// Starts a span in which requests may finish before their last verification is recorded;
    /// their lines wait for [`SpecLog::end_span`].
    pub fn begin_span(&mut self) {
        self.deferring = true;
    }

    /// Ends the span and returns the lines of the requests that finished inside it.
    pub fn end_span(&mut self) -> Vec<String> {
        self.deferring = false;
        let done: Vec<RequestId> = self
            .requests
            .iter()
            .filter(|(_, r)| r.closed.is_some())
            .map(|(&id, _)| id)
            .collect();
        done.into_iter()
            .filter_map(|id| self.take_line(id))
            .collect()
    }

    /// The log line of finished request `id`, or `None` when disabled or inside a span (the
    /// line then comes from [`SpecLog::end_span`]). The line carries the request's shape so a
    /// run can be joined to its task type offline.
    pub fn finish(
        &mut self,
        id: RequestId,
        prompt_tokens: usize,
        images: usize,
        completion_tokens: usize,
    ) -> Option<String> {
        if !self.enabled {
            return None;
        }
        let shape = Shape {
            prompt_tokens,
            images,
            completion_tokens,
        };
        self.requests.entry(id).or_default().closed = Some(shape);
        if self.deferring {
            return None;
        }
        self.take_line(id)
    }

    fn take_line(&mut self, id: RequestId) -> Option<String> {
        let r = self.requests.remove(&id)?;
        let s = r.closed?;
        let per_pos: Vec<String> = r.per_pos.iter().map(u64::to_string).collect();
        Some(format!(
            "scriptor: spec-request id={id} prompt_tokens={} images={} completion_tokens={} \
             verifications={} drafted={} accepted={} accepted_per_pos={}",
            s.prompt_tokens,
            s.images,
            s.completion_tokens,
            r.verifications,
            r.drafted,
            r.accepted,
            per_pos.join(",")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_parses_strictly() {
        assert_eq!(parse_enabled(None), Ok(false));
        assert_eq!(parse_enabled(Some("")), Ok(false));
        assert_eq!(parse_enabled(Some("0")), Ok(false));
        assert_eq!(parse_enabled(Some("1")), Ok(true));
        let err = parse_enabled(Some("yes")).unwrap_err();
        assert!(err.contains("SCRIPTOR_SPEC_REQUEST_LOG"));
    }

    #[test]
    fn a_disabled_log_records_and_prints_nothing() {
        let mut l = SpecLog::new(false);
        l.record(1, 3, 2, |_| true);
        assert_eq!(l.finish(1, 10, 1, 5), None);
    }

    #[test]
    fn a_request_line_sums_its_verifications_and_positions() {
        let mut l = SpecLog::new(true);
        l.record(7, 3, 3, |_| true);
        l.record(7, 3, 1, |_| true);
        l.record(8, 3, 0, |_| true);
        assert_eq!(
            l.finish(7, 120, 2, 9).as_deref(),
            Some(
                "scriptor: spec-request id=7 prompt_tokens=120 images=2 completion_tokens=9 \
                 verifications=2 drafted=6 accepted=4 accepted_per_pos=2,1,1"
            )
        );
        assert_eq!(
            l.finish(8, 50, 0, 3).as_deref(),
            Some(
                "scriptor: spec-request id=8 prompt_tokens=50 images=0 completion_tokens=3 \
                 verifications=1 drafted=3 accepted=0 accepted_per_pos=0,0,0"
            )
        );
        assert!(l.finish(7, 1, 0, 0).unwrap().contains("verifications=0"));
    }

    #[test]
    fn a_request_finished_inside_a_span_logs_its_last_verification() {
        let mut l = SpecLog::new(true);
        l.begin_span();
        assert_eq!(l.finish(5, 10, 0, 4), None);
        l.record(5, 3, 2, |_| true);
        l.finish(6, 10, 0, 1);
        let lines = l.end_span();
        assert_eq!(lines.len(), 2);
        assert!(
            lines[0].contains("id=5 ") && lines[0].contains("accepted=2 accepted_per_pos=1,1,0")
        );
        assert!(lines[1].contains("id=6 ") && lines[1].contains("verifications=0"));
        assert!(l.end_span().is_empty());
    }

    #[test]
    fn pruning_drops_requests_that_are_no_longer_live() {
        let mut l = SpecLog::new(true);
        for id in 0..=PRUNE_ABOVE as u64 {
            l.record(id, 1, 1, |_| true);
        }
        l.record(99_999, 1, 1, |id| id == 3);
        assert_eq!(l.requests.len(), 2);
    }
}
