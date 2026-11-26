//! Bench harness tests against in-process mock OpenAI servers.

use std::net::SocketAddr;
use std::path::PathBuf;

use scriptor_bench::equivalence::{compare, first_divergence};
use scriptor_bench::ladder::{percentile, run_ladder};
use scriptor_bench::mock_seat::{MockConfig, RunningSeat};
use scriptor_bench::openai_client::{ChatEndpoint, Client, RetryPolicy, request_body};
use scriptor_bench::regions::{
    SYNTHETIC_MANIFEST, Task, approx_image_tokens, bridge_mix, cases, load_font, load_region_dir,
    parse_synthetic_manifest, synthetic_regions, write_region_dir,
};
use scriptor_bench::report;
use scriptor_chat::Tokenizer;

fn tokenizer() -> Tokenizer {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/chat/tiny-tokenizer.json");
    Tokenizer::from_file(p).unwrap()
}

fn seat(reply: &str) -> RunningSeat {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    RunningSeat::start(addr, MockConfig::new(reply)).unwrap()
}

fn small_regions() -> Vec<scriptor_bench::regions::Region> {
    let specs = parse_synthetic_manifest(SYNTHETIC_MANIFEST).unwrap();
    let font = load_font(None).unwrap();
    let small: Vec<_> = specs
        .into_iter()
        .filter(|s| s.width * s.height < 200_000)
        .take(4)
        .collect();
    synthetic_regions(&small, &font).unwrap()
}

#[test]
fn synthetic_manifest_covers_sizes() {
    let specs = parse_synthetic_manifest(SYNTHETIC_MANIFEST).unwrap();
    assert_eq!(specs.len(), 20);
    assert!(
        specs
            .iter()
            .any(|s| s.width % 28 != 0 || s.height % 28 != 0)
    );
    assert!(
        specs
            .iter()
            .any(|s| approx_image_tokens(s.width, s.height) == 6000)
    );
    for t in [Task::Text, Task::Table, Task::Formula] {
        assert!(specs.iter().any(|s| s.task == t));
    }
}

#[test]
fn rendering_is_deterministic_and_round_trips() {
    let a = small_regions();
    let b = small_regions();
    assert_eq!(a, b);
    assert!(a.iter().all(|r| r.jpeg.starts_with(&[0xFF, 0xD8])));
    let dir = tempfile::tempdir().unwrap();
    write_region_dir(dir.path(), &a).unwrap();
    assert_eq!(load_region_dir(dir.path()).unwrap(), a);
}

#[test]
fn request_body_matches_the_bridge_golden() {
    let body = request_body("glm-ocr", Task::Table, "data:image/jpeg;base64,AAAA");
    let golden = serde_json::json!({
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
            {"type": "text", "text": "Table Recognition:"},
        ]}],
        "max_tokens": 8192, "temperature": 0.0, "top_p": 0.00001, "top_k": 1,
        "repetition_penalty": 1.1, "model": "glm-ocr",
    });
    assert_eq!(body, golden);
}

#[test]
fn mock_seat_answers_the_body_and_lists_the_model() {
    let s = seat("hello world");
    let c = Client::new(&s.url(), None, RetryPolicy::default());
    assert_eq!(c.models().unwrap(), vec!["glm-ocr".to_owned()]);
    let r = c
        .chat(&request_body(
            "glm-ocr",
            Task::Text,
            "data:image/jpeg;base64,AAAA",
        ))
        .unwrap();
    assert_eq!(r.content, "hello world");
    assert_eq!(r.finish_reason, "stop");
    assert_eq!(r.prompt_tokens, 247);
    assert!(r.completion_tokens > 0);
    assert_eq!(s.served(), 1);
}

#[test]
fn identical_endpoints_are_fully_equivalent() {
    let (a, b) = (seat("alpha beta gamma"), seat("alpha beta gamma"));
    let (ca, cb) = (
        Client::new(&a.url(), None, RetryPolicy::default()),
        Client::new(&b.url(), None, RetryPolicy::default()),
    );
    let list = cases(&small_regions(), Some(&[Task::Text, Task::Formula]));
    assert_eq!(list.len(), 8);
    let r = compare(
        &ca,
        &cb,
        ("a", "b"),
        "glm-ocr",
        &tokenizer(),
        &list,
        3,
        true,
    );
    assert!(r.all_identical());
    assert_eq!(r.identity_percent(), 100.0);
    assert!(!r.margins_available || r.cases.iter().all(|c| c.divergence.is_none()));
    assert_eq!(a.served(), 8);
}

#[test]
fn divergent_endpoints_report_the_first_differing_token() {
    let (a, b) = (seat("alpha beta gamma"), seat("alpha beta delta"));
    let (ca, cb) = (
        Client::new(&a.url(), None, RetryPolicy::default()),
        Client::new(&b.url(), None, RetryPolicy::default()),
    );
    let list = cases(&small_regions(), Some(&[Task::Text]));
    let r = compare(
        &ca,
        &cb,
        ("a", "b"),
        "glm-ocr",
        &tokenizer(),
        &list,
        2,
        false,
    );
    assert!(!r.all_identical());
    assert_eq!(r.identical(), 0);
    let tk = tokenizer();
    let (ta, tb) = (
        tk.encode("alpha beta gamma").unwrap(),
        tk.encode("alpha beta delta").unwrap(),
    );
    let expected = first_divergence(&ta, &tb).unwrap();
    for c in &r.cases {
        assert_eq!(c.divergence.as_ref().unwrap().index, expected);
    }
    let md = report::equivalence_markdown(&r);
    assert!(md.contains("First divergence"));
    let j = report::equivalence_json(&r);
    assert_eq!(j["identical"], 0);
}

#[test]
fn first_divergence_handles_prefixes() {
    assert_eq!(first_divergence(&[1, 2, 3], &[1, 2, 3]), None);
    assert_eq!(first_divergence(&[1, 2, 3], &[1, 2]), Some(2));
    assert_eq!(first_divergence(&[1, 9], &[1, 2, 3]), Some(1));
}

#[test]
fn ladder_measures_every_level() {
    let s = seat("some recognised text");
    let c = Client::new(&s.url(), None, RetryPolicy::default());
    let list = bridge_mix(&small_regions(), 16);
    let levels = run_ladder(&c, "glm-ocr", &list, &[2, 4], 3);
    assert_eq!(levels.len(), 2);
    assert_eq!(levels[0].requests, 6);
    assert_eq!(levels[1].requests, 12);
    assert!(
        levels
            .iter()
            .all(|l| l.errors == 0 && l.requests_per_second > 0.0)
    );
    assert!(levels[0].prompt_tokens_per_second > 0.0);
    let md = report::ladder_markdown(&s.url(), "glm-ocr", &levels);
    assert!(md.contains("| 4 |"));
    let j = report::ladder_json(&s.url(), "glm-ocr", &levels);
    assert_eq!(j["levels"].as_array().unwrap().len(), 2);
    assert_eq!(s.served(), 18);
}

#[test]
fn percentile_uses_nearest_rank() {
    let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    assert_eq!(percentile(&v, 0.5), 5.0);
    assert_eq!(percentile(&v, 0.99), 10.0);
    assert_eq!(percentile(&[], 0.5), 0.0);
}

#[test]
fn bridge_mix_is_deterministic_and_mostly_text() {
    let regs = small_regions();
    let a = bridge_mix(&regs, 2000);
    assert_eq!(a, bridge_mix(&regs, 2000));
    let text = a.iter().filter(|c| c.task == Task::Text).count();
    assert!(text > 1800);
}
