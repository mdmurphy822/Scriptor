//! Markdown and JSON reports of the benchmark runs.

use std::fmt::Write as _;

use serde_json::{Value, json};

use crate::equivalence::EquivalenceReport;
use crate::ladder::LevelResult;

fn opt(v: Option<f64>) -> Value {
    v.map_or(Value::Null, |x| json!(x))
}

/// The equivalence report as JSON.
pub fn equivalence_json(r: &EquivalenceReport) -> Value {
    json!({
        "kind": "equivalence",
        "model": r.model,
        "endpoint_a": r.endpoint_a,
        "endpoint_b": r.endpoint_b,
        "cases": r.cases.len(),
        "identical": r.identical(),
        "identity_percent": r.identity_percent(),
        "margins_available": r.margins_available,
        "results": r.cases.iter().map(|c| json!({
            "region": c.region,
            "task": c.task,
            "identical": c.identical,
            "tokens_a": c.tokens_a,
            "tokens_b": c.tokens_b,
            "completion_tokens_a": c.completion_tokens_a,
            "completion_tokens_b": c.completion_tokens_b,
            "completion_tokens_match": c.completion_tokens_match,
            "error": c.error,
            "divergence": c.divergence.as_ref().map(|d| json!({
                "index": d.index,
                "token_a": d.token_a,
                "token_b": d.token_b,
                "margin_a": opt(d.margin_a),
                "margin_b": opt(d.margin_b),
            })),
        })).collect::<Vec<_>>(),
    })
}

/// The equivalence report as Markdown.
pub fn equivalence_markdown(r: &EquivalenceReport) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# Equivalence: {} vs {}\n", r.endpoint_a, r.endpoint_b);
    let _ = writeln!(s, "| Item | Value |\n|---|---|");
    let _ = writeln!(s, "| Model | {} |", r.model);
    let _ = writeln!(s, "| Cases | {} |", r.cases.len());
    let _ = writeln!(
        s,
        "| Token-identical | {} ({:.2} %) |",
        r.identical(),
        r.identity_percent()
    );
    let _ = writeln!(
        s,
        "| Logit margins | {} |\n",
        if r.margins_available {
            "recorded"
        } else {
            "unavailable"
        }
    );
    let bad: Vec<_> = r.cases.iter().filter(|c| !c.identical).collect();
    if bad.is_empty() {
        s.push_str("All cases are token-identical.\n");
        return s;
    }
    let _ = writeln!(
        s,
        "| Region | Task | First divergence | Token A | Token B | Margin A | Margin B | Completion tokens A/B | Error |\n|---|---|---|---|---|---|---|---|---|"
    );
    let num = |v: Option<f64>| v.map_or("-".to_owned(), |x| format!("{x:.4}"));
    let tok = |v: Option<u32>| v.map_or("-".to_owned(), |x| x.to_string());
    for c in bad {
        let d = c.divergence.as_ref();
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} | {} | {}/{} | {} |",
            c.region,
            c.task,
            d.map_or("-".to_owned(), |d| d.index.to_string()),
            tok(d.and_then(|d| d.token_a)),
            tok(d.and_then(|d| d.token_b)),
            num(d.and_then(|d| d.margin_a)),
            num(d.and_then(|d| d.margin_b)),
            c.completion_tokens_a,
            c.completion_tokens_b,
            c.error.as_deref().unwrap_or("-"),
        );
    }
    s
}

/// The ladder report as JSON.
pub fn ladder_json(endpoint: &str, model: &str, levels: &[LevelResult]) -> Value {
    json!({
        "kind": "ladder",
        "endpoint": endpoint,
        "model": model,
        "levels": levels.iter().map(|l| json!({
            "concurrency": l.concurrency,
            "requests": l.requests,
            "errors": l.errors,
            "wall_seconds": l.wall_seconds,
            "requests_per_second": l.requests_per_second,
            "p50_ms": l.p50_ms,
            "p99_ms": l.p99_ms,
            "prompt_tokens_per_second": l.prompt_tokens_per_second,
            "generation_tokens_per_second": l.generation_tokens_per_second,
        })).collect::<Vec<_>>(),
    })
}

/// The ladder report as Markdown.
pub fn ladder_markdown(endpoint: &str, model: &str, levels: &[LevelResult]) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# Throughput ladder: {endpoint}\n");
    let _ = writeln!(s, "Model `{model}`.\n");
    let _ = writeln!(
        s,
        "| Concurrency | Requests | Errors | Requests/s | p50 ms | p99 ms | Prompt tok/s | Gen tok/s |\n|---|---|---|---|---|---|---|---|"
    );
    for l in levels {
        let _ = writeln!(
            s,
            "| {} | {} | {} | {:.2} | {:.1} | {:.1} | {:.0} | {:.0} |",
            l.concurrency,
            l.requests,
            l.errors,
            l.requests_per_second,
            l.p50_ms,
            l.p99_ms,
            l.prompt_tokens_per_second,
            l.generation_tokens_per_second,
        );
    }
    s
}
