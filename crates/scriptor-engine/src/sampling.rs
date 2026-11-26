//! Per-sequence token selection: penalties from prompt/output bitsets, greedy
//! or seeded sampling, top-2 margin, log-probabilities, and finish checks.
//! The CPU path here is the reference the device kernel is held to.

use crate::request::SamplingParams;
use scriptor_kernels::sampling::{RowParams, TokenMasks, process_row};

/// One selected token.
#[derive(Debug, Clone, PartialEq)]
pub struct Sampled {
    /// Chosen token id.
    pub token: u32,
    /// Best minus second-best penalised logit.
    pub margin: f32,
    /// Log-probability of the chosen token under the raw logits, when asked.
    pub logprob: Option<f32>,
    /// Highest-probability alternatives under the raw logits, descending.
    pub top_logprobs: Vec<(u32, f32)>,
}

/// Why a sequence stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// An end-of-sequence id or a stop string was produced.
    Stop,
    /// The generation limit was reached.
    Length,
}

/// Sampling state of one sequence: parameters, bitsets and step counter.
#[derive(Debug, Clone)]
pub struct SequenceSampler {
    params: SamplingParams,
    seed: u64,
    masks: TokenMasks,
    generated: u32,
}

impl SequenceSampler {
    /// Starts a sequence over a `vocab`-sized id space. Every prompt id,
    /// including image placeholders, is marked for the repetition penalty.
    /// `fallback_seed` is used when the request carries no seed.
    pub fn new(vocab: usize, prompt: &[u32], params: SamplingParams, fallback_seed: u64) -> Self {
        let mut masks = TokenMasks::new(1, vocab);
        masks.reset_slot(0, prompt);
        Self {
            seed: params.seed.unwrap_or(fallback_seed),
            params,
            masks,
            generated: 0,
        }
    }

    /// The sequence parameters.
    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Tokens accepted so far.
    pub fn generated(&self) -> u32 {
        self.generated
    }

    /// The sequence's prompt and output bitsets and counts, in the layout the device reads.
    pub fn masks(&self) -> &TokenMasks {
        &self.masks
    }

    /// Device row parameters for the next token of this sequence.
    pub fn row_params(&self, slot: u32) -> RowParams {
        RowParams {
            seed: self.seed,
            repetition: self.params.repetition_penalty,
            presence: self.params.presence_penalty,
            frequency: self.params.frequency_penalty,
            temperature: self.params.temperature,
            top_p: self.params.top_p,
            top_k: self.params.top_k,
            step: self.generated,
            slot,
        }
    }

    /// Selects the next token from a raw fp32 logits row without accepting it.
    pub fn sample(&self, logits: &[f32]) -> Sampled {
        let pick = process_row(logits, &self.masks, &self.row_params(0));
        let (logprob, top_logprobs) = match self.params.logprobs {
            Some(n) => log_probs(logits, pick.token, n as usize),
            None => (None, Vec::new()),
        };
        Sampled {
            token: pick.token,
            margin: pick.margin,
            logprob,
            top_logprobs,
        }
    }

    /// Records an accepted token in the output bitset and advances the step.
    pub fn accept(&mut self, token: u32) {
        self.masks.push_output(0, token);
        self.generated += 1;
    }

    /// Finish state after the last accepted token: `Stop` when it is an
    /// end-of-sequence id, `Length` when the generation limit is reached.
    pub fn finish_after(&self, token: u32, eos: &[u32]) -> Option<FinishReason> {
        if eos.contains(&token) {
            Some(FinishReason::Stop)
        } else if self.params.max_tokens.is_some_and(|m| self.generated >= m) {
            Some(FinishReason::Length)
        } else {
            None
        }
    }
}

/// Log-probability of `chosen` and the `top` most probable ids (descending,
/// ties by lowest id) under a softmax of the raw logits.
pub fn log_probs(logits: &[f32], chosen: u32, top: usize) -> (Option<f32>, Vec<(u32, f32)>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&l| f64::from(l - max).exp()).sum();
    let lse = f64::from(max) + sum.ln();
    let lp = |l: f32| (f64::from(l) - lse) as f32;
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    ids.sort_by(|&a, &b| {
        logits[b as usize]
            .total_cmp(&logits[a as usize])
            .then(a.cmp(&b))
    });
    let alts = ids
        .into_iter()
        .take(top)
        .map(|i| (i, lp(logits[i as usize])))
        .collect();
    (logits.get(chosen as usize).map(|&l| lp(l)), alts)
}

/// Byte offset of the earliest occurrence of any stop string in `text`,
/// searching only matches that end inside `text[from..]` extended backwards by
/// the longest stop string, so a stop spanning a delta boundary is found.
pub fn find_stop(text: &str, from: usize, stops: &[String]) -> Option<usize> {
    let longest = stops.iter().map(|s| s.len()).max()?;
    let mut start = from.saturating_sub(longest.saturating_sub(1));
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    stops
        .iter()
        .filter_map(|s| text[start..].find(s.as_str()).map(|i| start + i))
        .min()
}

/// Shortest repeated unit, in characters, that counts as a repetition loop.
pub const REPEAT_MIN_UNIT: usize = 10;
/// Consecutive copies of a unit that make a repetition loop.
pub const REPEAT_MIN_COUNT: usize = 10;
/// Characters of text below which a repetition is left to the consumer.
pub const REPEAT_MIN_TEXT: usize = 2048;

fn is_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

fn is_collapsible(c: char) -> bool {
    matches!(c, '.' | '\u{b7}' | '_' | '\\')
}

/// Incremental detector of a repetition loop in generated text: some unit of
/// at least [`REPEAT_MIN_UNIT`] characters occurs [`REPEAT_MIN_COUNT`] times
/// in a row and the text, less collapsible punctuation runs, has at least
/// [`REPEAT_MIN_TEXT`] characters. This is the condition under which a
/// consumer that truncates a repeated unit to one copy removes everything
/// after the first copy. Text containing a run of four or more identical
/// dots, middle dots or underscores (or `\_` pairs) is never reported, since
/// such runs are shortened before the repetition is judged.
#[derive(Debug, Default)]
pub struct RepeatTail {
    chars: Vec<char>,
    /// `runs[u - REPEAT_MIN_UNIT]`: trailing positions `i` with `chars[i] == chars[i - u]`.
    runs: Vec<usize>,
    leading_space: usize,
    all_space: bool,
    effective: usize,
    same_run: usize,
    disabled: bool,
}

impl RepeatTail {
    /// An empty detector.
    pub fn new() -> Self {
        Self {
            all_space: true,
            ..Self::default()
        }
    }

    /// Appends text; returns whether the text so far is a repetition loop.
    pub fn push_str(&mut self, piece: &str) -> bool {
        let mut hit = false;
        for c in piece.chars() {
            hit |= self.push_char(c);
        }
        hit
    }

    fn push_char(&mut self, c: char) -> bool {
        if self.disabled {
            return false;
        }
        let n = self.chars.len();
        self.same_run = if n > 0 && self.chars[n - 1] == c {
            self.same_run + 1
        } else {
            1
        };
        if self.all_space && is_space(c) {
            self.leading_space += 1;
        } else {
            self.all_space = false;
        }
        if is_collapsible(c) {
            if self.same_run >= 4 && c != '\\' {
                self.disabled = true;
            }
        } else if !(self.all_space) {
            self.effective += 1;
        }
        self.chars.push(c);
        if self.chars.len() >= 8
            && self
                .chars
                .iter()
                .rev()
                .take(8)
                .copied()
                .eq("_\\_\\_\\_\\".chars())
        {
            self.disabled = true;
        }
        if self.disabled {
            self.chars = Vec::new();
            self.runs = Vec::new();
            return false;
        }
        let length = n + 1;
        let max_unit = length / REPEAT_MIN_UNIT;
        let mut hit = false;
        let judged = self.effective >= REPEAT_MIN_TEXT && !is_space(c);
        for (offset, run) in self.runs.iter_mut().enumerate() {
            let unit = REPEAT_MIN_UNIT + offset;
            *run = if self.chars[n] == self.chars[n - unit] {
                *run + 1
            } else {
                0
            };
            if judged
                && *run >= unit * (REPEAT_MIN_COUNT - 1)
                && length - unit * REPEAT_MIN_COUNT >= self.leading_space
            {
                hit = true;
            }
        }
        while REPEAT_MIN_UNIT + self.runs.len() <= max_unit {
            let unit = REPEAT_MIN_UNIT + self.runs.len();
            let mut run = 0;
            while run + unit <= n && self.chars[n - run] == self.chars[n - run - unit] {
                run += 1;
            }
            if judged
                && run >= unit * (REPEAT_MIN_COUNT - 1)
                && length - unit * REPEAT_MIN_COUNT >= self.leading_space
            {
                hit = true;
            }
            self.runs.push(run);
        }
        hit
    }
}
