//! Penalised greedy sampling reference.

/// Result of one greedy step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GreedyPick {
    /// Chosen token id (lowest index among ties).
    pub token: u32,
    /// Best logit minus second-best logit after penalties.
    pub margin: f32,
}

/// Penalty settings applied to a logits row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Penalties {
    /// Repetition penalty; `1.0` disables.
    pub repetition: f32,
    /// Presence penalty; `0.0` disables.
    pub presence: f32,
    /// Frequency penalty; `0.0` disables.
    pub frequency: f32,
}

impl Default for Penalties {
    fn default() -> Self {
        Self {
            repetition: 1.0,
            presence: 0.0,
            frequency: 0.0,
        }
    }
}

/// Applies penalties to `logits` in place with vLLM semantics: the repetition
/// penalty touches every id present in the prompt or the output (positive
/// logits divided, negative multiplied); then `frequency * output_count` and
/// `presence * (output_count > 0)` are subtracted. Ids at or beyond the
/// vocabulary are ignored.
pub fn apply_penalties(logits: &mut [f32], prompt: &[u32], output: &[u32], p: Penalties) {
    let vocab = logits.len();
    let mut counts = vec![0u32; vocab];
    for &t in output {
        if (t as usize) < vocab {
            counts[t as usize] += 1;
        }
    }
    let mut in_prompt = vec![false; vocab];
    for &t in prompt {
        if (t as usize) < vocab {
            in_prompt[t as usize] = true;
        }
    }
    for (i, logit) in logits.iter_mut().enumerate() {
        if p.repetition != 1.0 && (in_prompt[i] || counts[i] > 0) {
            *logit = if *logit > 0.0 {
                *logit / p.repetition
            } else {
                *logit * p.repetition
            };
        }
        *logit -= p.frequency * counts[i] as f32;
        if counts[i] > 0 {
            *logit -= p.presence;
        }
    }
}

/// Argmax with the top-2 margin; ties resolve to the lowest index.
pub fn argmax_with_margin(logits: &[f32]) -> GreedyPick {
    assert!(!logits.is_empty());
    let (mut best, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut token = 0usize;
    for (i, &l) in logits.iter().enumerate() {
        if l > best {
            second = best;
            best = l;
            token = i;
        } else if l > second {
            second = l;
        }
    }
    GreedyPick {
        token: token as u32,
        margin: best - second,
    }
}

/// Penalties followed by greedy argmax on a copy of `logits`.
pub fn penalty_argmax(logits: &[f32], prompt: &[u32], output: &[u32], p: Penalties) -> GreedyPick {
    let mut row = logits.to_vec();
    apply_penalties(&mut row, prompt, output, p);
    argmax_with_margin(&row)
}

#[cfg(test)]
mod tests {
    use super::*;

    const REP: Penalties = Penalties {
        repetition: 2.0,
        presence: 0.0,
        frequency: 0.0,
    };

    #[test]
    fn positive_divided_negative_multiplied() {
        let mut l = [4.0, -4.0, 4.0, -4.0];
        apply_penalties(&mut l, &[0, 1], &[], REP);
        assert_eq!(l, [2.0, -8.0, 4.0, -4.0]);
    }

    #[test]
    fn output_ids_penalised_once() {
        let mut l = [4.0, 4.0];
        apply_penalties(&mut l, &[], &[0, 0, 0], REP);
        assert_eq!(l, [2.0, 4.0]);
    }

    #[test]
    fn presence_and_frequency_use_output_only() {
        let mut l = [1.0, 1.0, 1.0];
        let p = Penalties {
            repetition: 1.0,
            presence: 0.5,
            frequency: 0.25,
        };
        apply_penalties(&mut l, &[0], &[1, 1], p);
        assert_eq!(l, [1.0, 1.0 - 0.5 - 0.5, 1.0]);
    }

    #[test]
    fn penalty_changes_argmax() {
        let l = [3.0, 2.0, 0.0];
        let pick = penalty_argmax(&l, &[0], &[], REP);
        assert_eq!(pick.token, 1);
        assert!((pick.margin - 0.5).abs() < 1e-6);
    }

    #[test]
    fn tie_resolves_lowest_index() {
        assert_eq!(argmax_with_margin(&[1.0, 5.0, 5.0]).token, 1);
        assert_eq!(argmax_with_margin(&[1.0, 5.0, 5.0]).margin, 0.0);
    }

    #[test]
    fn out_of_vocab_ids_ignored() {
        let mut l = [1.0];
        apply_penalties(&mut l, &[9], &[9], REP);
        assert_eq!(l, [1.0]);
    }
}
