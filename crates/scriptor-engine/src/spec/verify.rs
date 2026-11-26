//! Verification side of the speculative loop: per-sequence sampling state and
//! the acceptance of a draft chain against the verifier's logits.

use crate::request::SamplingParams;
use crate::sampling::FinishReason;
use scriptor_kernels::reference::sampling::{Penalties, apply_penalties, argmax_with_margin};
use scriptor_kernels::sampling::uniform;
use scriptor_kernels::verify::{
    AcceptOutcome, accept_greedy, accept_rejection, target_distribution,
};

/// Salt separating the correction-draw stream from the acceptance stream.
const RESAMPLE_SALT: u64 = 0xA5A5_5A5A_C3C3_3C3C;

/// Sampling state of one sequence: parameters, prompt and accepted output.
#[derive(Debug, Clone)]
pub struct SeqSampling {
    params: SamplingParams,
    seed: u64,
    prompt: Vec<u32>,
    output: Vec<u32>,
}

impl SeqSampling {
    /// Starts a sequence; every prompt id, including image placeholders, is
    /// visible to the repetition penalty.
    pub fn new(prompt: &[u32], params: SamplingParams, fallback_seed: u64) -> Self {
        Self {
            seed: params.seed.unwrap_or(fallback_seed),
            params,
            prompt: prompt.to_vec(),
            output: Vec::new(),
        }
    }

    /// The sequence parameters.
    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Tokens accepted so far.
    pub fn generated(&self) -> usize {
        self.output.len()
    }

    /// Whether the sequence is decoded greedily.
    pub fn is_greedy(&self) -> bool {
        self.params.temperature <= 0.0
    }

    /// Records an accepted output token.
    pub fn accept(&mut self, token: u32) {
        self.output.push(token);
    }

    /// Finish state after the last accepted token.
    pub fn finish(&self, last: u32, eos: &[u32]) -> Option<FinishReason> {
        if eos.contains(&last) {
            Some(FinishReason::Stop)
        } else if self
            .params
            .max_tokens
            .is_some_and(|m| self.output.len() >= m as usize)
        {
            Some(FinishReason::Length)
        } else {
            None
        }
    }

    /// Output tokens still allowed by `max_tokens`, `usize::MAX` when unbounded.
    pub fn remaining(&self) -> usize {
        self.params.max_tokens.map_or(usize::MAX, |m| {
            (m as usize).saturating_sub(self.output.len())
        })
    }

    fn penalised(&self, logits: &[f32], drafts: &[u32]) -> Vec<f32> {
        let mut row = logits.to_vec();
        let mut output = self.output.clone();
        output.extend_from_slice(drafts);
        apply_penalties(
            &mut row,
            &self.prompt,
            &output,
            Penalties {
                repetition: self.params.repetition_penalty,
                presence: self.params.presence_penalty,
                frequency: self.params.frequency_penalty,
            },
        );
        row
    }

    /// Next token from raw `logits` with `drafts` appended to the output the
    /// penalties see; argmax when greedy, otherwise a seeded draw at `step`.
    pub fn next_token(&self, logits: &[f32], drafts: &[u32]) -> u32 {
        let row = self.penalised(logits, drafts);
        if self.is_greedy() {
            argmax_with_margin(&row).token
        } else {
            let dist = self.distribution_of(&row);
            let u = uniform(self.seed, (self.output.len() + drafts.len()) as u32);
            scriptor_kernels::verify::sample_distribution(&dist, u)
        }
    }

    fn distribution_of(&self, penalised: &[f32]) -> Vec<f32> {
        target_distribution(
            penalised,
            self.params.temperature,
            self.params.top_k,
            self.params.top_p,
        )
    }

    /// Verifies `drafts` against the verifier's raw `logits` (one row per
    /// verified position, `drafts.len() + 1` rows) and returns the emitted
    /// tokens. Greedy sequences accept by exact match with the penalised
    /// argmax, reproducing non-speculative decoding exactly; sampled
    /// sequences use rejection sampling against the penalised, temperature
    /// and top-k / top-p filtered distribution.
    pub fn verify(&self, drafts: &[u32], logits: &[Vec<f32>]) -> AcceptOutcome {
        assert_eq!(logits.len(), drafts.len() + 1);
        if self.is_greedy() {
            let target: Vec<u32> = (0..=drafts.len())
                .map(|j| {
                    let row = self.penalised(&logits[j], &drafts[..j]);
                    argmax_with_margin(&row).token
                })
                .collect();
            return accept_greedy(drafts, &target);
        }
        let dists: Vec<Vec<f32>> = (0..=drafts.len())
            .map(|j| self.distribution_of(&self.penalised(&logits[j], &drafts[..j])))
            .collect();
        let step = self.output.len() as u32;
        let accept_u: Vec<f32> = (0..drafts.len())
            .map(|j| uniform(self.seed, step + j as u32))
            .collect();
        let resample_u: Vec<f32> = (0..=drafts.len())
            .map(|j| uniform(self.seed ^ RESAMPLE_SALT, step + j as u32))
            .collect();
        accept_rejection(drafts, &dists, &accept_u, &resample_u)
    }
}
