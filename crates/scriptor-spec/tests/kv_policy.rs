//! The KV policy selector: reserve by default, demand-diagnostic only in feature builds.

use scriptor_spec::{KvPolicy, RuntimeParams};

#[test]
fn the_default_policy_is_reserve() {
    assert_eq!(KvPolicy::default(), KvPolicy::Reserve);
    assert_eq!(RuntimeParams::default().kv_policy, KvPolicy::Reserve);
    assert_eq!("reserve".parse::<KvPolicy>(), Ok(KvPolicy::Reserve));
}

#[test]
fn an_unknown_policy_is_refused() {
    assert!("greedy".parse::<KvPolicy>().is_err());
}

#[cfg(not(feature = "kv-demand-diagnostic"))]
#[test]
fn demand_diagnostic_is_refused_without_the_feature() {
    let err = "demand-diagnostic".parse::<KvPolicy>().unwrap_err();
    assert!(err.contains("kv-demand-diagnostic"), "{err}");
}

#[cfg(feature = "kv-demand-diagnostic")]
#[test]
fn demand_diagnostic_parses_with_the_feature() {
    let policy = "demand-diagnostic".parse::<KvPolicy>().unwrap();
    assert_eq!(policy.to_string(), "demand-diagnostic");
}
