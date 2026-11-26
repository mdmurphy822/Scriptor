//! The shutdown drain deadline derived from the work in flight.

use std::time::Duration;

/// Time the engine may keep serving after a shutdown signal: the output tokens still owed to
/// running and waiting requests divided by the measured decode rate, plus one prefill pass,
/// capped by `bound` when the host supplies one. Without a measured decode rate (no token has
/// been decoded) there is no basis for waiting, so the deadline is zero; with nothing in
/// flight it is zero as well.
pub fn drain_deadline(
    tokens_in_flight: u64,
    decode_tokens_per_second: f64,
    prefill_pass_seconds: f64,
    bound: Option<Duration>,
) -> Duration {
    if tokens_in_flight == 0
        || !decode_tokens_per_second.is_finite()
        || decode_tokens_per_second <= 0.0
    {
        return Duration::ZERO;
    }
    let seconds =
        tokens_in_flight as f64 / decode_tokens_per_second + prefill_pass_seconds.max(0.0);
    let derived = Duration::try_from_secs_f64(seconds).unwrap_or(Duration::MAX);
    bound.map_or(derived, |b| derived.min(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadline_is_tokens_over_rate_plus_one_prefill() {
        let d = drain_deadline(200, 50.0, 0.5, None);
        assert_eq!(d, Duration::from_secs_f64(4.5));
    }

    #[test]
    fn nothing_in_flight_or_no_rate_gives_zero() {
        assert_eq!(drain_deadline(0, 50.0, 1.0, None), Duration::ZERO);
        assert_eq!(drain_deadline(10, 0.0, 1.0, None), Duration::ZERO);
        assert_eq!(drain_deadline(10, f64::NAN, 1.0, None), Duration::ZERO);
    }

    #[test]
    fn host_bound_caps_the_derived_deadline() {
        let d = drain_deadline(10_000, 10.0, 1.0, Some(Duration::from_secs(30)));
        assert_eq!(d, Duration::from_secs(30));
        let d = drain_deadline(100, 10.0, 0.0, Some(Duration::from_secs(30)));
        assert_eq!(d, Duration::from_secs(10));
    }
}
