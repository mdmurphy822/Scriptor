//! Counterbalanced measurement schedule for the RoPE + KV-append cost cases.

/// Measurement order of `variants` variants in `round`: a cyclic rotation by `round`, reversed on
/// every second pass over the rotations, so a cycle of `2 * variants` rounds places each variant
/// in each position equally often and in each neighbour relation both ways.
pub fn round_order(variants: usize, round: usize) -> Vec<usize> {
    let cycle = round % (2 * variants);
    let (shift, reversed) = (cycle % variants, cycle >= variants);
    let mut order: Vec<usize> = (0..variants).map(|p| (p + shift) % variants).collect();
    if reversed {
        order.reverse();
    }
    order
}

/// Orders for `rounds` consecutive rounds.
pub fn schedule(variants: usize, rounds: usize) -> Vec<Vec<usize>> {
    (0..rounds).map(|r| round_order(variants, r)).collect()
}

/// Number of rounds that fit `budget_secs` when one round takes `round_secs`: a multiple of
/// `variants` (so every variant sees every position equally), at least `variants`, at most
/// `max_rounds`. Returns the rounds and the total seconds they take.
pub fn plan_rounds(
    variants: usize,
    round_secs: f64,
    budget_secs: f64,
    max_rounds: usize,
) -> (usize, f64) {
    let fit = if round_secs > 0.0 {
        (budget_secs / round_secs).floor() as usize
    } else {
        max_rounds
    };
    let capped = fit.min(max_rounds);
    let rounds = (capped / variants).max(1) * variants;
    (rounds, rounds as f64 * round_secs)
}

/// Median of `values`; the mean of the two middle values for an even count.
pub fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}
