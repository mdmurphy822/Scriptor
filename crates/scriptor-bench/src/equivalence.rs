//! Token-for-token comparison of two chat endpoints over a case list.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use scriptor_chat::Tokenizer;

use crate::openai_client::{ChatEndpoint, ChatReply, request_body, with_logprobs};
use crate::regions::Case;

/// Where two replies first differ.
#[derive(Debug, Clone, PartialEq)]
pub struct Divergence {
    /// Index of the first differing token (the shorter length when one reply is a
    /// prefix of the other).
    pub index: usize,
    pub token_a: Option<u32>,
    pub token_b: Option<u32>,
    /// Gap between the two most likely candidates at that position on each side.
    pub margin_a: Option<f64>,
    pub margin_b: Option<f64>,
}

/// The outcome of one (region, task) case.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseResult {
    pub region: String,
    pub task: String,
    pub identical: bool,
    pub tokens_a: usize,
    pub tokens_b: usize,
    pub completion_tokens_a: u64,
    pub completion_tokens_b: u64,
    pub completion_tokens_match: bool,
    pub divergence: Option<Divergence>,
    pub error: Option<String>,
}

/// The outcome of a comparison run.
#[derive(Debug, Clone, PartialEq)]
pub struct EquivalenceReport {
    pub model: String,
    pub endpoint_a: String,
    pub endpoint_b: String,
    pub margins_available: bool,
    pub cases: Vec<CaseResult>,
}

impl EquivalenceReport {
    /// Cases whose token streams are identical.
    pub fn identical(&self) -> usize {
        self.cases.iter().filter(|c| c.identical).count()
    }

    /// Identical cases as a percentage of all cases.
    pub fn identity_percent(&self) -> f64 {
        if self.cases.is_empty() {
            return 100.0;
        }
        100.0 * self.identical() as f64 / self.cases.len() as f64
    }

    /// Whether every case is identical.
    pub fn all_identical(&self) -> bool {
        !self.cases.is_empty() && self.identical() == self.cases.len()
    }
}

/// The first position where `a` and `b` differ.
pub fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    let common = a.len().min(b.len());
    (0..common)
        .find(|&i| a[i] != b[i])
        .or_else(|| (a.len() != b.len()).then_some(common))
}

fn margin_at(reply: &ChatReply, index: usize) -> Option<f64> {
    reply
        .margins
        .as_ref()
        .and_then(|m| m.get(index))
        .copied()
        .filter(|v| v.is_finite())
}

/// Compares one reply pair.
pub fn compare_replies(
    case: &Case,
    a: &ChatReply,
    b: &ChatReply,
    tokenizer: &Tokenizer,
) -> Result<CaseResult, String> {
    let ta = tokenizer.encode(&a.content).map_err(|e| e.to_string())?;
    let tb = tokenizer.encode(&b.content).map_err(|e| e.to_string())?;
    let at = first_divergence(&ta, &tb);
    Ok(CaseResult {
        region: case.region.clone(),
        task: case.task.name().to_owned(),
        identical: at.is_none(),
        tokens_a: ta.len(),
        tokens_b: tb.len(),
        completion_tokens_a: a.completion_tokens,
        completion_tokens_b: b.completion_tokens,
        completion_tokens_match: a.completion_tokens == b.completion_tokens,
        divergence: at.map(|i| Divergence {
            index: i,
            token_a: ta.get(i).copied(),
            token_b: tb.get(i).copied(),
            margin_a: margin_at(a, i),
            margin_b: margin_at(b, i),
        }),
        error: None,
    })
}

fn failed(case: &Case, error: String) -> CaseResult {
    CaseResult {
        region: case.region.clone(),
        task: case.task.name().to_owned(),
        identical: false,
        tokens_a: 0,
        tokens_b: 0,
        completion_tokens_a: 0,
        completion_tokens_b: 0,
        completion_tokens_match: false,
        divergence: None,
        error: Some(error),
    }
}

/// Posts every case to both endpoints with `workers` concurrent cases and
/// compares the replies. Logprobs are requested when `logprobs` is set and are
/// dropped per endpoint when it rejects them.
#[allow(clippy::too_many_arguments)]
pub fn compare(
    a: &dyn ChatEndpoint,
    b: &dyn ChatEndpoint,
    labels: (&str, &str),
    model: &str,
    tokenizer: &Tokenizer,
    cases: &[Case],
    workers: usize,
    logprobs: bool,
) -> EquivalenceReport {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, CaseResult)>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..workers.max(1) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(case) = cases.get(i) else { break };
                    let mut body = request_body(model, case.task, &case.data_url);
                    if logprobs {
                        body = with_logprobs(body);
                    }
                    let outcome = match (a.chat(&body), b.chat(&body)) {
                        (Ok(ra), Ok(rb)) => compare_replies(case, &ra, &rb, tokenizer)
                            .unwrap_or_else(|e| failed(case, e)),
                        (Err(e), _) => failed(case, format!("endpoint a: {e}")),
                        (_, Err(e)) => failed(case, format!("endpoint b: {e}")),
                    };
                    results.lock().unwrap().push((i, outcome));
                }
            });
        }
    });
    let mut rows = results.into_inner().unwrap();
    rows.sort_by_key(|(i, _)| *i);
    EquivalenceReport {
        model: model.to_owned(),
        endpoint_a: labels.0.to_owned(),
        endpoint_b: labels.1.to_owned(),
        margins_available: logprobs && a.logprobs_supported() && b.logprobs_supported(),
        cases: rows.into_iter().map(|(_, r)| r).collect(),
    }
}
