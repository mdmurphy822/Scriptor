//! CPU checks of the cost-case measurement schedule.
#[allow(dead_code)]
#[path = "rope_schedule/mod.rs"]
mod rope_schedule;

use rope_schedule::{median, plan_rounds, round_order, schedule};

#[test]
fn every_round_is_a_permutation() {
    for r in 0..24 {
        let mut o = round_order(4, r);
        o.sort_unstable();
        assert_eq!(o, vec![0, 1, 2, 3], "round {r}");
    }
}

#[test]
fn full_cycle_places_each_variant_in_each_position_equally() {
    for n in [2usize, 3, 4, 5] {
        let rounds = schedule(n, 2 * n);
        let mut count = vec![vec![0usize; n]; n];
        for order in &rounds {
            for (pos, &v) in order.iter().enumerate() {
                count[v][pos] += 1;
            }
        }
        for row in &count {
            assert!(row.iter().all(|&c| c == 2), "n={n}: {count:?}");
        }
    }
}

#[test]
fn first_rotation_pass_alone_is_position_balanced() {
    let rounds = schedule(4, 4);
    for pos in 0..4 {
        let mut seen: Vec<usize> = rounds.iter().map(|o| o[pos]).collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3]);
    }
}

#[test]
fn schedule_retains_every_attempt() {
    let rounds = schedule(4, 12);
    assert_eq!(rounds.len(), 12);
    assert_eq!(rounds.iter().map(Vec::len).sum::<usize>(), 48);
}

#[test]
fn plan_respects_budget_and_balance() {
    let (rounds, total) = plan_rounds(4, 1.3, 30.0, 64);
    assert_eq!(rounds % 4, 0);
    assert!(total <= 30.0, "{total}");
    assert_eq!(rounds, 20);
}

#[test]
fn plan_keeps_at_least_one_round_per_variant_and_caps() {
    assert_eq!(plan_rounds(4, 50.0, 30.0, 64).0, 4);
    assert_eq!(plan_rounds(4, 0.001, 30.0, 16).0, 16);
    assert_eq!(plan_rounds(4, 0.0, 30.0, 8).0, 8);
}

#[test]
fn median_handles_odd_and_even() {
    assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
    assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
}
