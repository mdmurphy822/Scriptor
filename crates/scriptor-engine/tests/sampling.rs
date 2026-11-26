use scriptor_engine::request::{SamplingParams, SamplingRequest, StopSpec};
use scriptor_engine::sampling::{FinishReason, SequenceSampler, find_stop, log_probs};
use serde_json::json;

fn greedy(rep: f32) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        repetition_penalty: rep,
        ..SamplingParams::default()
    }
}

#[test]
fn positive_logits_divided_negative_multiplied() {
    // Prompt ids 0 and 1 are penalised; 2 is not.
    let s = SequenceSampler::new(4, &[0, 1], greedy(2.0), 0);
    // Raw [3, -1, 1.9, -3]: after penalty [1.5, -2, 1.9, -3] -> id 2 wins.
    let r = s.sample(&[3.0, -1.0, 1.9, -3.0]);
    assert_eq!(r.token, 2);
    assert!((r.margin - 0.4).abs() < 1e-6);
}

#[test]
fn prompt_only_ids_are_penalised() {
    let s = SequenceSampler::new(3, &[0], greedy(1.1), 0);
    // 5.0 / 1.1 = 4.545 < 4.6
    assert_eq!(s.sample(&[5.0, 4.6, 0.0]).token, 1);
}

#[test]
fn image_placeholder_id_is_in_the_prompt_mask() {
    let placeholder = 5u32;
    let prompt = [3, 4, placeholder, placeholder, placeholder, 2];
    let s = SequenceSampler::new(8, &prompt, greedy(2.0), 0);
    assert_eq!(s.sample(&[0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 3.5]).token, 7);
}

#[test]
fn penalty_off_when_one() {
    let s = SequenceSampler::new(3, &[0], greedy(1.0), 0);
    assert_eq!(s.sample(&[5.0, 4.6, 0.0]).token, 0);
}

#[test]
fn output_tokens_are_penalised_after_accept() {
    let mut s = SequenceSampler::new(3, &[], greedy(2.0), 0);
    let logits = [5.0, 4.0, 0.0];
    assert_eq!(s.sample(&logits).token, 0);
    s.accept(0);
    assert_eq!(s.sample(&logits).token, 1);
    assert_eq!(s.generated(), 1);
}

#[test]
fn presence_and_frequency_count_output_only() {
    let params = SamplingParams {
        temperature: 0.0,
        presence_penalty: 0.5,
        frequency_penalty: 0.25,
        ..SamplingParams::default()
    };
    let mut s = SequenceSampler::new(3, &[1], params, 0);
    s.accept(0);
    s.accept(0);
    // id 0: 2.0 - 0.5 - 0.5 = 1.0; id 1 (prompt only): 1.2
    assert_eq!(s.sample(&[2.0, 1.2, 0.0]).token, 1);
}

#[test]
fn greedy_ignores_top_k_and_top_p() {
    let req = SamplingRequest {
        temperature: Some(0.0),
        top_p: Some(1e-5),
        top_k: Some(1),
        repetition_penalty: Some(1.1),
        max_tokens: Some(8192),
        ..Default::default()
    };
    let p = SamplingParams::from_request(&req).unwrap();
    assert!(p.is_greedy());
    assert_eq!((p.top_p, p.top_k), (1.0, 0));
    assert_eq!(p.max_tokens, Some(8192));
}

#[test]
fn seeded_sampling_is_reproducible_and_seed_sensitive() {
    let params = SamplingParams {
        temperature: 1.0,
        seed: Some(11),
        ..SamplingParams::default()
    };
    let logits: Vec<f32> = (0..64).map(|i| ((i * 5) % 7) as f32 * 0.2).collect();
    let run = |p: &SamplingParams| {
        let mut s = SequenceSampler::new(64, &[], p.clone(), 0);
        (0..16)
            .map(|_| {
                let t = s.sample(&logits).token;
                s.accept(t);
                t
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(run(&params), run(&params));
    let other = SamplingParams {
        seed: Some(12),
        ..params.clone()
    };
    assert_ne!(run(&params), run(&other));
}

#[test]
fn finish_reasons() {
    let params = SamplingParams {
        max_tokens: Some(2),
        ..greedy(1.0)
    };
    let mut s = SequenceSampler::new(4, &[], params, 0);
    s.accept(1);
    assert_eq!(s.finish_after(1, &[3]), None);
    s.accept(2);
    assert_eq!(s.finish_after(2, &[3]), Some(FinishReason::Length));
    assert_eq!(s.finish_after(3, &[3]), Some(FinishReason::Stop));
}

#[test]
fn logprobs_sum_and_order() {
    let (lp, top) = log_probs(&[0.0, 0.0, (2.0f32).ln()], 2, 2);
    assert!((lp.unwrap() - (0.5f32).ln()).abs() < 1e-6);
    assert_eq!(top[0].0, 2);
    assert_eq!(top[1].0, 0);
    let total: f32 = log_probs(&[1.0, 2.0, 3.0], 0, 3)
        .1
        .iter()
        .map(|x| x.1.exp())
        .sum();
    assert!((total - 1.0).abs() < 1e-5);
}

#[test]
fn sampled_logprobs_are_returned_when_requested() {
    let params = SamplingParams {
        logprobs: Some(2),
        ..greedy(1.0)
    };
    let s = SequenceSampler::new(3, &[], params, 0);
    let r = s.sample(&[1.0, 3.0, 2.0]);
    assert_eq!(r.token, 1);
    assert_eq!(r.top_logprobs.len(), 2);
    assert!(r.logprob.is_some());
}

#[test]
fn stop_search_spans_delta_boundary() {
    let stops = vec!["END".to_string()];
    assert_eq!(find_stop("abcEN", 0, &stops), None);
    assert_eq!(find_stop("abcEND", 5, &stops), Some(3));
    assert_eq!(find_stop("héllo END", 4, &stops), Some(7));
}

#[test]
fn request_parsing_and_vllm_errors() {
    let p = SamplingParams::from_json(&json!({
        "temperature": 0.0, "top_p": 1e-5, "top_k": 1,
        "repetition_penalty": 1.1, "max_tokens": 8192,
        "stop": "x", "seed": 5, "logprobs": true, "top_logprobs": 3
    }))
    .unwrap();
    assert_eq!(p.stop, vec!["x"]);
    assert_eq!(p.logprobs, Some(3));
    assert_eq!(p.seed, Some(5));
    let err = |v: serde_json::Value| SamplingParams::from_json(&v).unwrap_err().message;
    assert_eq!(
        err(json!({"temperature": -1})),
        "temperature must be non-negative, got -1."
    );
    assert_eq!(err(json!({"top_p": 0})), "top_p must be in (0, 1], got 0.");
    assert_eq!(
        err(json!({"repetition_penalty": 0})),
        "repetition_penalty must be greater than zero, got 0."
    );
    assert_eq!(
        err(json!({"max_tokens": 0})),
        "max_tokens must be at least 1, got 0."
    );
    assert_eq!(
        err(json!({"presence_penalty": 3})),
        "presence_penalty must be in [-2, 2], got 3."
    );
    let defaults = SamplingParams::from_json(&json!({})).unwrap();
    assert_eq!(defaults, SamplingParams::default());
    let many = SamplingRequest {
        stop: Some(StopSpec::Many(vec!["a".into(), String::new()])),
        ..Default::default()
    };
    assert_eq!(SamplingParams::from_request(&many).unwrap().stop, vec!["a"]);
}

fn feed(text: &str) -> bool {
    use scriptor_engine::sampling::RepeatTail;
    let mut tail = RepeatTail::new();
    let mut hit = false;
    for piece in text.as_bytes().chunks(3) {
        hit |= tail.push_str(&String::from_utf8_lossy(piece));
    }
    hit
}

fn prose(len: usize) -> String {
    let mut x: u64 = 7;
    let mut s = String::new();
    while s.len() < len {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s.push((b'a' + ((x >> 33) % 26) as u8) as char);
        if (x >> 40).is_multiple_of(6) {
            s.push(' ');
        }
    }
    s
}

#[test]
fn repeat_tail_fires_on_ten_copies_of_a_long_unit() {
    let unit = "| cell one | cell two |\n";
    assert!(feed(&format!("{}{}|", prose(2100), unit.repeat(10))));
    assert!(!feed(&format!("{}{}|", prose(2100), unit.repeat(9))));
}

#[test]
fn repeat_tail_unit_boundary_is_ten_characters() {
    let ten = "abcdefghi ";
    assert!(feed(&format!("{}{}a", prose(2100), ten.repeat(10))));
    assert!(!feed(&format!("{}{}a", prose(2100), ten.repeat(9))));
    // A short period counts through its multiples of at least ten characters.
    assert!(feed(&format!("{}{}", prose(2100), "ab c".repeat(30) + "a")));
}

#[test]
fn repeat_tail_ignores_near_periodic_text() {
    let mut tail = String::new();
    for i in 0..40 {
        tail.push_str(&format!("| row {i} | value |\n"));
    }
    assert!(!feed(&format!("{}{}", prose(2100), tail)));
    let mut broken = String::new();
    for i in 0..30 {
        broken.push_str("abcdefghij klmno");
        if i % 9 == 8 {
            broken.push('!');
        }
    }
    assert!(!feed(&format!("{}{}", prose(2100), broken)));
}

#[test]
fn repeat_tail_waits_for_the_consumer_length_gate() {
    assert!(!feed(&"| cell one | cell two |\n".repeat(20)));
    assert!(feed(&"| cell one | cell two |\n".repeat(100)));
}

#[test]
fn repeat_tail_leaves_collapsible_runs_to_the_consumer() {
    let text = format!("{}{}{}", prose(1500), ".....", prose(1500));
    assert!(!feed(&format!(
        "{text}{}",
        "| cell one | cell two |\n".repeat(20)
    )));
}

#[test]
fn repeat_tail_requires_a_non_space_ending() {
    let unit = "| cell one | cell two |\n";
    let mut tail = scriptor_engine::sampling::RepeatTail::new();
    assert!(!tail.push_str(&prose(2100)));
    assert!(!tail.push_str(&unit.repeat(10).trim_end().to_string()[..unit.len() * 10 - 2]));
}

#[test]
fn stop_on_repetition_is_opt_in_and_parses() {
    use scriptor_engine::request::{SamplingParams, SamplingRequest};
    let off = SamplingParams::from_request(&SamplingRequest::default()).unwrap();
    assert!(!off.stop_on_repetition);
    let on = SamplingParams::from_request(&SamplingRequest {
        stop_on_repetition: Some(true),
        ..SamplingRequest::default()
    })
    .unwrap();
    assert!(on.stop_on_repetition);
}
