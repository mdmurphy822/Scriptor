use scriptor_engine::request::SamplingParams;
use scriptor_engine::sampling::{FinishReason, SequenceSampler};
use scriptor_engine::spec::mtp::{MtpBatch, MtpOutput, MtpRows, shifted_ids};
use scriptor_engine::spec::policy::PolicyConfig;
use scriptor_engine::spec::{
    SeqEmit, SpecConfig, SpecEngine, SpecModel, VerifyBatch, VerifyOutput, VerifyRows,
};
use scriptor_engine::{EngineError, RequestId};
use std::collections::HashMap;

const VOCAB: usize = 48;

fn mix(a: u64, b: u64) -> u64 {
    let mut z = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn extend(hash: u64, token: u32) -> u64 {
    mix(hash, u64::from(token) + 1)
}

/// Deterministic next-token logits for a context hash; the peak never lands on
/// the last vocabulary id.
fn logits_of(hash: u64) -> Vec<f32> {
    let peak = (mix(hash, 0xFFFF) % (VOCAB as u64 - 1)) as usize;
    (0..VOCAB)
        .map(|v| {
            let noise = (mix(hash, v as u64) % 1000) as f32 / 1000.0;
            if v == peak { noise + 3.0 } else { noise }
        })
        .collect()
}

#[derive(Default)]
struct Seq {
    hashes: Vec<u64>,
    mtp_len: usize,
}

struct Tiny {
    /// Percentage of draft positions whose prediction is corrupted.
    noise_pct: u64,
    verify_base: f64,
    verify_per_row: f64,
    draft_per_pass: f64,
    seqs: HashMap<RequestId, Seq>,
    mtp_calls: usize,
    first_mtp_start: HashMap<RequestId, usize>,
}

impl Tiny {
    fn new(noise_pct: u64) -> Self {
        Self {
            noise_pct,
            verify_base: 0.010,
            verify_per_row: 0.0001,
            draft_per_pass: 0.0005,
            seqs: HashMap::new(),
            mtp_calls: 0,
            first_mtp_start: HashMap::new(),
        }
    }

    fn prefill(&mut self, id: RequestId, prompt: &[u32]) -> (Vec<f32>, Vec<u64>) {
        let mut h = 0u64;
        let hashes: Vec<u64> = prompt
            .iter()
            .map(|&t| {
                h = extend(h, t);
                h
            })
            .collect();
        let logits = logits_of(*hashes.last().unwrap());
        self.seqs.insert(
            id,
            Seq {
                hashes: hashes.clone(),
                mtp_len: 0,
            },
        );
        (logits, hashes)
    }
}

impl SpecModel for Tiny {
    type Hidden = u64;

    fn verify(&mut self, rows: &[VerifyRows]) -> Result<VerifyBatch<u64>, EngineError> {
        let mut outputs = Vec::new();
        let mut total = 0usize;
        for r in rows {
            let seq = self.seqs.get_mut(&r.seq).expect("prefilled sequence");
            assert!(
                seq.hashes.len() == r.start_position || seq.hashes.len() == r.start_position + 1,
                "main KV length"
            );
            seq.hashes.truncate(r.start_position);
            let mut h = seq.hashes.last().copied().unwrap_or(0);
            let mut logits = Vec::new();
            let mut hiddens = Vec::new();
            for &t in &r.tokens {
                h = extend(h, t);
                seq.hashes.push(h);
                logits.push(logits_of(h));
                hiddens.push(h);
            }
            total += r.tokens.len();
            outputs.push(VerifyOutput { logits, hiddens });
        }
        Ok(VerifyBatch {
            outputs,
            seconds: self.verify_base + self.verify_per_row * total as f64,
        })
    }

    fn mtp_forward(&mut self, groups: &[MtpRows<u64>]) -> Result<MtpBatch<u64>, EngineError> {
        self.mtp_calls += 1;
        let mut outputs = Vec::new();
        for g in groups {
            let seq = self.seqs.get_mut(&g.seq).expect("prefilled sequence");
            assert!(g.start_position <= seq.mtp_len, "no gap in the MTP KV");
            self.first_mtp_start
                .entry(g.seq)
                .or_insert(g.start_position);
            assert!(!g.rows.is_empty());
            let mut last = None;
            for row in &g.rows {
                let next = extend(row.hidden, row.token);
                let predicted = if mix(next, 77) % 100 < self.noise_pct {
                    mix(next, 5)
                } else {
                    next
                };
                last = Some(MtpOutput {
                    logits: logits_of(predicted),
                    hidden: next,
                });
            }
            seq.mtp_len = g.start_position + g.rows.len();
            outputs.push(last.unwrap());
        }
        Ok(MtpBatch {
            outputs,
            seconds: self.draft_per_pass,
        })
    }

    fn rollback(&mut self, seq: RequestId, kv_len: usize) {
        self.seqs.get_mut(&seq).unwrap().hashes.truncate(kv_len);
    }

    fn release(&mut self, seq: RequestId) {
        self.seqs.remove(&seq);
    }
}

fn greedy_params(max_tokens: u32) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        repetition_penalty: 1.1,
        max_tokens: Some(max_tokens),
        ..SamplingParams::default()
    }
}

fn config(k_max: usize, eos: Vec<u32>) -> SpecConfig {
    SpecConfig {
        eos_token_ids: eos,
        max_context: 4096,
        fallback_seed: 7,
        policy: PolicyConfig {
            k_max,
            ..PolicyConfig::default()
        },
    }
}

fn prompt_for(i: usize) -> Vec<u32> {
    let len = 3 + (i * 7) % 29;
    (0..len)
        .map(|j| (mix(i as u64, j as u64) % VOCAB as u64) as u32)
        .collect()
}

/// Non-speculative decode of one prompt through the scheduler's sampler.
fn reference(prompt: &[u32], params: &SamplingParams, eos: &[u32]) -> Vec<u32> {
    let mut h = 0u64;
    for &t in prompt {
        h = extend(h, t);
    }
    let mut sampler = SequenceSampler::new(VOCAB, prompt, params.clone(), 0);
    let mut out = Vec::new();
    loop {
        let tok = sampler.sample(&logits_of(h)).token;
        sampler.accept(tok);
        out.push(tok);
        h = extend(h, tok);
        if sampler.finish_after(tok, eos).is_some() {
            return out;
        }
    }
}

fn drive(
    engine: &mut SpecEngine<Tiny>,
    prompts: &[Vec<u32>],
    params: &SamplingParams,
) -> (Vec<Vec<u32>>, Vec<FinishReason>) {
    let mut outs: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    let mut reasons = vec![FinishReason::Length; prompts.len()];
    let apply = |emit: SeqEmit, outs: &mut Vec<Vec<u32>>, reasons: &mut Vec<FinishReason>| {
        let i = emit.id as usize;
        outs[i].extend(&emit.tokens);
        if let Some(r) = emit.finish {
            reasons[i] = r;
        }
    };
    for (i, p) in prompts.iter().enumerate() {
        let (logits, hiddens) = engine.model_mut().prefill(i as u64, p);
        let emit = engine
            .start(i as u64, p, params.clone(), &logits, hiddens)
            .unwrap();
        apply(emit, &mut outs, &mut reasons);
    }
    while engine.live() > 0 {
        for emit in engine.step().unwrap() {
            apply(emit, &mut outs, &mut reasons);
        }
    }
    (outs, reasons)
}

fn spec_matches_reference(
    k_max: usize,
    noise: u64,
    eos: Vec<u32>,
    max_tokens: u32,
) -> Vec<FinishReason> {
    let prompts: Vec<Vec<u32>> = (0..50).map(prompt_for).collect();
    let params = greedy_params(max_tokens);
    let mut engine = SpecEngine::new(Tiny::new(noise), config(k_max, eos.clone()));
    let (got, reasons) = drive(&mut engine, &prompts, &params);
    for (i, p) in prompts.iter().enumerate() {
        assert_eq!(got[i], reference(p, &params, &eos), "prompt {i}");
    }
    reasons
}

#[test]
fn spec_greedy_output_equals_non_spec_for_50_prompts() {
    spec_matches_reference(3, 30, vec![VOCAB as u32 - 1], 40);
}

#[test]
fn spec_output_equals_non_spec_with_perfect_drafter_and_long_chain() {
    spec_matches_reference(5, 0, vec![VOCAB as u32 - 1], 37);
}

#[test]
fn spec_output_equals_non_spec_with_useless_drafter() {
    spec_matches_reference(4, 100, vec![VOCAB as u32 - 1], 40);
}

#[test]
fn deferred_first_token_matches_non_spec_and_the_immediate_start() {
    let eos = vec![5];
    for (k_max, noise, max_tokens) in [(3, 30, 40), (4, 0, 1), (3, 100, 2), (0, 0, 25)] {
        let prompts: Vec<Vec<u32>> = (0..30).map(prompt_for).collect();
        let params = greedy_params(max_tokens);
        let mut engine = SpecEngine::new(Tiny::new(noise), config(k_max, eos.clone()));
        let mut outs: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
        for (i, p) in prompts.iter().enumerate() {
            let (_, hiddens) = engine.model_mut().prefill(i as u64, p);
            engine
                .start_deferred(i as u64, p, params.clone(), hiddens)
                .unwrap();
        }
        let mut first_step = true;
        while engine.live() > 0 {
            for emit in engine.step().unwrap() {
                if first_step {
                    assert_eq!(emit.tokens.len(), 1, "the first step verifies no drafts");
                }
                outs[emit.id as usize].extend(&emit.tokens);
            }
            first_step = false;
        }
        for (i, p) in prompts.iter().enumerate() {
            assert_eq!(
                outs[i],
                reference(p, &params, &eos),
                "prompt {i}, k {k_max}"
            );
        }
    }
}

#[test]
fn spec_disabled_equals_non_spec() {
    spec_matches_reference(0, 0, vec![VOCAB as u32 - 1], 25);
}

#[test]
fn spec_output_stops_at_eos_inside_an_accepted_chain() {
    // Token 5 occurs among the generated ids, so some chains end mid-accept.
    let reasons = spec_matches_reference(4, 10, vec![5], 200);
    assert!(reasons.contains(&FinishReason::Stop));
}

#[test]
fn perfect_drafter_emits_k_plus_one_tokens_per_step() {
    let prompts: Vec<Vec<u32>> = (0..4).map(prompt_for).collect();
    // 1 token from the prompt pass plus 5 full steps of k + 1 = 4.
    let params = greedy_params(21);
    let mut engine = SpecEngine::new(Tiny::new(0), config(3, vec![VOCAB as u32 - 1]));
    let mut steps = 0;
    for (i, p) in prompts.iter().enumerate() {
        let (l, h) = engine.model_mut().prefill(i as u64, p);
        engine.start(i as u64, p, params.clone(), &l, h).unwrap();
    }
    while engine.live() > 0 {
        let emits = engine.step().unwrap();
        steps += 1;
        for e in &emits {
            assert_eq!(e.tokens.len(), 4, "step {steps}");
        }
    }
    assert_eq!(steps, 5);
    let stats = engine.stats();
    assert_eq!(stats.tokens_per_step(), 4.0);
    assert_eq!(stats.accepted_per_position, stats.proposed_per_position);
    assert_eq!(stats.proposed_per_position, vec![20, 20, 20]);
    let text = stats.prometheus();
    assert!(text.contains("vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"} 20"));
    assert!(text.contains("vllm:spec_decode_num_draft_tokens_total 60"));
}

#[test]
fn noisy_drafter_records_decreasing_acceptance_by_position() {
    let prompts: Vec<Vec<u32>> = (0..20).map(prompt_for).collect();
    let mut engine = SpecEngine::new(Tiny::new(35), config(3, vec![VOCAB as u32 - 1]));
    drive(&mut engine, &prompts, &greedy_params(60));
    let s = engine.stats();
    assert!(s.drafted > 0 && s.accepted > 0 && s.accepted < s.drafted);
    let rate = |j: usize| s.accepted_per_position[j] as f64 / s.proposed_per_position[j] as f64;
    assert!(rate(0) >= rate(1) && rate(1) >= rate(2));
}

#[test]
fn mtp_prefill_feeds_every_prompt_position_from_zero() {
    let prompts: Vec<Vec<u32>> = vec![prompt_for(3)];
    let mut engine = SpecEngine::new(Tiny::new(0), config(2, vec![VOCAB as u32 - 1]));
    drive(&mut engine, &prompts, &greedy_params(10));
    assert_eq!(engine.model().first_mtp_start[&0], 0);
}

#[test]
fn shifted_ids_keep_image_placeholders() {
    let placeholder = 9;
    let prompt = [1, 2, placeholder, placeholder, placeholder, 3];
    assert_eq!(
        shifted_ids(&prompt, 40),
        vec![2, placeholder, placeholder, placeholder, 3, 40]
    );
}

#[test]
fn drafts_that_cost_more_than_they_save_are_disabled() {
    let prompts: Vec<Vec<u32>> = (0..8).map(prompt_for).collect();
    let mut model = Tiny::new(100);
    model.draft_per_pass = 5.0;
    let eos = vec![VOCAB as u32 - 1];
    let mut engine = SpecEngine::new(model, config(3, eos.clone()));
    let params = greedy_params(120);
    let (got, _) = drive(&mut engine, &prompts, &params);
    // Output is still exact, and after the policy has measured the costs it
    // runs without drafts for most steps.
    for (i, p) in prompts.iter().enumerate() {
        assert_eq!(got[i], reference(p, &params, &eos));
    }
    let s = engine.stats();
    assert!(
        s.drafted < s.sequence_steps * 3 / 2,
        "drafted {} over {} sequence-steps",
        s.drafted,
        s.sequence_steps
    );
    assert_eq!(engine.last_k(), 0);
}

#[test]
fn sampled_decoding_is_reproducible_and_in_vocabulary() {
    let prompts: Vec<Vec<u32>> = (0..10).map(prompt_for).collect();
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 6,
        top_p: 0.95,
        repetition_penalty: 1.1,
        max_tokens: Some(50),
        seed: Some(1234),
        ..SamplingParams::default()
    };
    let run = || {
        let mut engine = SpecEngine::new(Tiny::new(20), config(3, vec![VOCAB as u32 - 1]));
        drive(&mut engine, &prompts, &params).0
    };
    let (a, b) = (run(), run());
    assert_eq!(a, b);
    assert!(a.iter().flatten().all(|&t| (t as usize) < VOCAB));
    assert!(a.iter().all(|o| !o.is_empty()));
}
