//! Wire-level tests of the HTTP surface over a scripted engine.

use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use scriptor_server::{AppConfig, AppState, EngineError, Metrics, ScriptedEngine, Stage, router};
use serde_json::{Value, json};
use tower::ServiceExt;

fn fixture(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/server")
        .join(name);
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn app_with(engine: Arc<ScriptedEngine>, config: AppConfig) -> Router {
    router(AppState::new(engine, Arc::new(Metrics::new()), config))
}

fn engine() -> Arc<ScriptedEngine> {
    Arc::new(ScriptedEngine::new(
        "glm-ocr",
        16384,
        &["Hello", ", ", "world"],
    ))
}

async fn send(
    app: &Router,
    request: Request<Body>,
) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    (status, headers, body.to_vec())
}

fn post(body: &Value) -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn get(path: &str) -> Request<Body> {
    Request::get(path).body(Body::empty()).unwrap()
}

fn json_of(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).unwrap()
}

#[tokio::test]
async fn semantik_golden_body_is_answered_in_the_client_shape() {
    let eng = engine();
    let app = app_with(eng.clone(), AppConfig::default());
    let (status, _, body) = send(&app, post(&fixture("semantik_request.json"))).await;
    assert_eq!(status, StatusCode::OK);
    let reply = json_of(&body);
    for path in fixture("semantik_response_shape.json")["required_paths"]
        .as_array()
        .unwrap()
    {
        assert!(
            reply.pointer(path.as_str().unwrap()).is_some(),
            "missing {path}"
        );
    }
    assert_eq!(reply["choices"][0]["message"]["content"], "Hello, world");
    assert_eq!(reply["choices"][0]["finish_reason"], "stop");
    assert_eq!(reply["usage"]["prompt_tokens"], 12 + 235);
    assert_eq!(reply["usage"]["completion_tokens"], 3);
    assert_eq!(reply["usage"]["total_tokens"], 12 + 235 + 3);

    let seen = eng.received();
    assert_eq!(seen.len(), 1);
    let s = &seen[0].sampling;
    assert!(s.is_greedy());
    assert_eq!(s.repetition_penalty, 1.1);
    assert_eq!(s.max_tokens, Some(8192));
    assert!(seen[0].render.add_generation_prompt);
}

#[tokio::test]
async fn finish_reason_length_at_max_tokens() {
    let app = app_with(engine(), AppConfig::default());
    let mut body = fixture("semantik_request.json");
    body["max_tokens"] = json!(2);
    let (status, _, bytes) = send(&app, post(&body)).await;
    assert_eq!(status, StatusCode::OK);
    let reply = json_of(&bytes);
    assert_eq!(reply["choices"][0]["finish_reason"], "length");
    assert_eq!(reply["choices"][0]["message"]["content"], "Hello, ");
    assert_eq!(reply["usage"]["completion_tokens"], 2);
}

#[tokio::test]
async fn models_is_503_until_ready_then_200() {
    let eng = engine();
    eng.set_ready(false);
    let app = app_with(eng.clone(), AppConfig::default());
    let (status, _, body) = send(&app, get("/v1/models")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(json_of(&body)["error"]["message"].is_string());
    let (status, _, _) = send(&app, post(&fixture("semantik_request.json"))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);

    eng.set_ready(true);
    let (status, _, body) = send(&app, get("/v1/models")).await;
    assert_eq!(status, StatusCode::OK);
    let listing = json_of(&body);
    assert_eq!(listing["data"][0]["id"], "glm-ocr");
    assert_eq!(listing["data"][0]["max_model_len"], 16384);
}

#[tokio::test]
async fn health_is_503_dead_after_a_device_fault() {
    let eng = engine();
    let app = app_with(eng.clone(), AppConfig::default());
    let (status, _, _) = send(&app, get("/health")).await;
    assert_eq!(status, StatusCode::OK);
    eng.set_dead(scriptor_server::Dead {
        cuda: Some(700),
        message: "CUDA_ERROR_ILLEGAL_ADDRESS".into(),
    });
    let (status, _, body) = send(&app, get("/health")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let report = json_of(&body);
    assert_eq!(report["state"], "dead");
    assert_eq!(report["cuda"], "700");
}

#[tokio::test]
async fn health_reports_stages() {
    let eng = engine();
    eng.set_stages(vec![
        Stage::new("weights", true),
        Stage::new("graphs", false),
    ]);
    let app = app_with(eng.clone(), AppConfig::default());
    let (status, _, body) = send(&app, get("/health")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let report = json_of(&body);
    assert_eq!(report["status"], "starting");
    assert_eq!(report["stages"][1]["name"], "graphs");
    assert_eq!(report["stages"][1]["done"], false);

    eng.set_stages(vec![
        Stage::new("weights", true),
        Stage::new("graphs", true),
    ]);
    let (status, _, body) = send(&app, get("/health")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json_of(&body)["status"], "ready");
}

#[tokio::test]
async fn metrics_expose_vllm_aliases_and_native_series() {
    let app = app_with(engine(), AppConfig::default());
    send(&app, post(&fixture("semantik_request.json"))).await;
    let (status, headers, body) = send(&app, get("/metrics")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        headers[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/plain")
    );
    let text = String::from_utf8(body).unwrap();
    for name in fixture("semantik_response_shape.json")["required_metrics"]
        .as_array()
        .unwrap()
    {
        let name = name.as_str().unwrap();
        assert!(
            text.lines().any(|l| l.starts_with(name)),
            "metric {name} missing"
        );
    }
    assert!(text.contains("vllm:prompt_tokens_total{engine=\"0\",model_name=\"glm-ocr\"} 247"));
    assert!(text.contains("vllm:generation_tokens_total{engine=\"0\",model_name=\"glm-ocr\"} 3"));
}

#[tokio::test]
async fn context_overflow_is_a_vllm_worded_400() {
    let app = app_with(engine(), AppConfig::default());
    let mut body = fixture("semantik_request.json");
    body["max_tokens"] = json!(16384);
    let (status, _, bytes) = send(&app, post(&body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let message = json_of(&bytes)["error"]["message"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(message.contains("input tokens and requested 16384 output tokens"));
    assert!(message.contains("context length is only 16384"));
}

#[tokio::test]
async fn too_many_images_and_unsupported_media_are_400() {
    let app = app_with(engine(), AppConfig::default());
    let mut two = fixture("semantik_request.json");
    let image = two["messages"][0]["content"][0].clone();
    two["messages"][0]["content"]
        .as_array_mut()
        .unwrap()
        .push(image);
    let (status, _, bytes) = send(&app, post(&two)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        json_of(&bytes)["error"]["message"],
        "At most 1 image(s) may be provided in one prompt."
    );

    let mut remote = fixture("semantik_request.json");
    remote["messages"][0]["content"][0]["image_url"]["url"] =
        json!("https://example.invalid/a.jpg");
    let (status, _, bytes) = send(&app, post(&remote)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        json_of(&bytes)["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Unsupported image URL")
    );

    let mut audio = fixture("semantik_request.json");
    audio["messages"][0]["content"][0] = json!({"type": "input_audio", "input_audio": {}});
    let (status, _, bytes) = send(&app, post(&audio)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        json_of(&bytes)["error"]["message"],
        "At most 0 audio(s) may be provided in one prompt."
    );
}

#[tokio::test]
async fn engine_unsupported_media_error_maps_to_400() {
    let eng = engine();
    eng.fail_with(EngineError::UnsupportedMedia(
        "Failed to decode the image.".into(),
    ));
    let app = app_with(eng, AppConfig::default());
    let (status, _, bytes) = send(&app, post(&fixture("semantik_request.json"))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        json_of(&bytes)["error"]["message"],
        "Failed to decode the image."
    );
}

#[tokio::test]
async fn tool_choice_and_template_kwargs_are_accepted() {
    let eng = engine();
    let app = app_with(eng.clone(), AppConfig::default());
    let mut body = fixture("semantik_request.json");
    body["tool_choice"] = json!("none");
    body["chat_template_kwargs"] = json!({"enable_thinking": false});
    let (status, _, _) = send(&app, post(&body)).await;
    assert_eq!(status, StatusCode::OK);
    let seen = eng.received();
    assert_eq!(seen[0].render.enable_thinking, Some(false));
    assert!(seen[0].render.add_generation_prompt);
}

async fn rejected_param(app: &Router, field: &str, value: Value, param: &str) {
    for stream in [false, true] {
        let mut body = fixture("semantik_request.json");
        body[field] = value.clone();
        body["stream"] = json!(stream);
        let (status, _, bytes) = send(app, post(&body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{field} stream={stream}");
        let error = &json_of(&bytes)["error"];
        assert_eq!(error["param"], param);
        assert_eq!(error["type"], "BadRequestError");
        assert_eq!(error["code"], 400);
        assert!(error["message"].as_str().unwrap().contains("not supported"));
    }
}

#[tokio::test]
async fn logprobs_true_is_a_400() {
    rejected_param(
        &app_with(engine(), AppConfig::default()),
        "logprobs",
        json!(true),
        "logprobs",
    )
    .await;
}

#[tokio::test]
async fn top_logprobs_is_a_400() {
    let app = app_with(engine(), AppConfig::default());
    rejected_param(&app, "top_logprobs", json!(2), "top_logprobs").await;
}

#[tokio::test]
async fn json_response_format_is_a_400() {
    let app = app_with(engine(), AppConfig::default());
    rejected_param(
        &app,
        "response_format",
        json!({"type": "json_object"}),
        "response_format",
    )
    .await;
}

#[tokio::test]
async fn logit_bias_with_entries_is_a_400() {
    let app = app_with(engine(), AppConfig::default());
    rejected_param(&app, "logit_bias", json!({"50": 5}), "logit_bias").await;
}

#[tokio::test]
async fn inert_values_of_unsupported_features_are_accepted() {
    let eng = engine();
    let app = app_with(eng.clone(), AppConfig::default());
    let mut body = fixture("semantik_request.json");
    body["logprobs"] = json!(false);
    body["top_logprobs"] = json!(0);
    body["response_format"] = json!({"type": "text"});
    body["logit_bias"] = json!({});
    let (status, _, bytes) = send(&app, post(&body)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json_of(&bytes)["choices"][0]["message"]["content"],
        "Hello, world"
    );
}

#[tokio::test]
async fn unknown_model_is_404_and_bad_body_is_400() {
    let app = app_with(engine(), AppConfig::default());
    let mut body = fixture("semantik_request.json");
    body["model"] = json!("other");
    let (status, _, _) = send(&app, post(&body)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let request = Request::post("/v1/chat/completions")
        .body(Body::from("{not json"))
        .unwrap();
    let (status, _, _) = send(&app, request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn sse_streams_deltas_usage_and_done() {
    let app = app_with(engine(), AppConfig::default());
    let mut body = fixture("semantik_request.json");
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage": true});
    let (status, headers, bytes) = send(&app, post(&body)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        headers[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream")
    );
    let text = String::from_utf8(bytes).unwrap();
    let frames: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .map(str::trim)
        .collect();
    assert_eq!(frames.last(), Some(&"[DONE]"));
    let chunks: Vec<Value> = frames[..frames.len() - 1]
        .iter()
        .map(|f| serde_json::from_str(f).unwrap())
        .collect();
    let content: String = chunks
        .iter()
        .filter_map(|c| {
            c.pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
        })
        .collect();
    assert_eq!(content, "Hello, world");
    let finish = chunks.iter().find_map(|c| {
        c.pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
    });
    assert_eq!(finish, Some("stop"));
    let last = chunks.last().unwrap();
    assert_eq!(last["usage"]["completion_tokens"], 3);
    assert_eq!(last["usage"]["prompt_tokens"], 247);
}

fn with_image_payload(base64_len: usize) -> Value {
    let mut body = fixture("semantik_request.json");
    body["messages"][0]["content"][0]["image_url"]["url"] =
        json!(format!("data:image/png;base64,{}", "A".repeat(base64_len)));
    body
}

#[tokio::test]
async fn image_bodies_above_two_mebibytes_are_accepted() {
    let app = app_with(engine(), AppConfig::default());
    let (status, _, _) = send(&app, post(&with_image_payload(8 << 20))).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn bodies_above_the_derived_limit_are_413() {
    let config = AppConfig::default();
    let over = config.max_body_bytes() + 1;
    let app = app_with(engine(), config);
    let (status, _, _) = send(&app, post(&with_image_payload(over))).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
}

fn minimal_chat() -> Value {
    json!({"model": "glm-ocr", "messages": [{"role": "user", "content": "hi"}]})
}

fn greedy_defaults() -> AppConfig {
    AppConfig {
        sampling_defaults: scriptor_spec::SamplingDefaults {
            temperature: Some(0.0),
            ..Default::default()
        },
        ..AppConfig::default()
    }
}

#[tokio::test]
async fn omitted_temperature_takes_the_served_default() {
    let eng = engine();
    let app = app_with(eng.clone(), greedy_defaults());
    let (status, _, _) = send(&app, post(&minimal_chat())).await;
    assert_eq!(status, StatusCode::OK);
    assert!(eng.received()[0].sampling.is_greedy());
}

#[tokio::test]
async fn explicit_temperature_overrides_the_served_default() {
    let eng = engine();
    let app = app_with(eng.clone(), greedy_defaults());
    let mut body = minimal_chat();
    body["temperature"] = json!(0.7);
    let (status, _, _) = send(&app, post(&body)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(eng.received()[0].sampling.temperature, 0.7);
}

#[tokio::test]
async fn without_served_defaults_the_builtin_sampling_applies() {
    let eng = engine();
    let app = app_with(eng.clone(), AppConfig::default());
    let (status, _, _) = send(&app, post(&minimal_chat())).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(eng.received()[0].sampling.temperature, 1.0);
}

#[tokio::test]
async fn latency_histograms_and_preemptions_are_exposed() {
    for stream in [false, true] {
        let app = app_with(engine(), AppConfig::default());
        let mut body = minimal_chat();
        body["stream"] = json!(stream);
        send(&app, post(&body)).await;
        let (_, _, bytes) = send(&app, get("/metrics")).await;
        let text = String::from_utf8(bytes).unwrap();
        for name in [
            "vllm:time_to_first_token_seconds_count",
            "vllm:inter_token_latency_seconds_count",
            "vllm:request_queue_time_seconds_count",
            "vllm:num_preemptions_total",
        ] {
            assert!(text.contains(name), "metric {name} missing");
        }
        assert!(text.contains(
            "vllm:time_to_first_token_seconds_count{engine=\"0\",model_name=\"glm-ocr\"} 1"
        ));
        assert!(text.contains(
            "vllm:inter_token_latency_seconds_count{engine=\"0\",model_name=\"glm-ocr\"} 2"
        ));
        assert!(text.contains(
            "vllm:request_queue_time_seconds_count{engine=\"0\",model_name=\"glm-ocr\"} 1"
        ));
    }
}

#[tokio::test]
async fn oversized_body_is_413_naming_the_limit() {
    let config = AppConfig::default();
    let limit = config.max_body_bytes();
    let app = app_with(engine(), config);
    let padding = "x".repeat(limit + 1);
    let mut body = fixture("semantik_request.json");
    body["padding"] = json!(padding);
    let (status, _, bytes) = send(&app, post(&body)).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    let message = json_of(&bytes)["error"]["message"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(message.contains(&limit.to_string()), "{message}");
}

#[tokio::test]
async fn request_beyond_the_intake_budget_is_413_and_a_full_intake_is_503() {
    let body = fixture("semantik_request.json");
    let resize = AppConfig::default().resize_config();
    let images: u64 = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["content"].as_array())
        .flatten()
        .filter_map(|p| p["image_url"]["url"].as_str())
        .map(|url| scriptor_server::intake::image_charge(url, &resize))
        .sum();
    let cost = body.to_string().len() as u64 + images;
    let tiny = AppConfig {
        intake_bytes: Some(cost - 1),
        ..AppConfig::default()
    };
    let (status, _, bytes) = send(&app_with(engine(), tiny), post(&body)).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert!(
        json_of(&bytes)["error"]["message"]
            .as_str()
            .unwrap()
            .contains("intake budget")
    );

    let config = AppConfig {
        intake_bytes: Some(cost),
        retry_after_secs: 4,
        ..AppConfig::default()
    };
    let state = AppState::new(engine(), Arc::new(Metrics::new()), config);
    let held = state.intake.as_ref().unwrap().reserve(1, 1, 4).unwrap();
    let app = router(state);
    let (status, headers, bytes) = send(&app, post(&body)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(headers[header::RETRY_AFTER], "4");
    let (_, _, metrics) = send(&app, get("/metrics")).await;
    assert!(
        String::from_utf8(metrics)
            .unwrap()
            .contains("scriptor_requests_rejected_busy_total{model_name=\"glm-ocr\"} 1")
    );
    assert!(
        json_of(&bytes)["error"]["message"]
            .as_str()
            .unwrap()
            .contains("intake budget")
    );
    drop(held);
    let (status, _, _) = send(&app, post(&body)).await;
    assert_eq!(status, StatusCode::OK);
}

struct PanickingEngine(ScriptedEngine);

impl scriptor_server::Engine for PanickingEngine {
    fn served_name(&self) -> &str {
        self.0.served_name()
    }
    fn max_model_len(&self) -> u64 {
        self.0.max_model_len()
    }
    fn health(&self) -> Vec<Stage> {
        self.0.health()
    }
    fn stats(&self) -> scriptor_server::EngineStats {
        self.0.stats()
    }
    fn submit(
        &self,
        _: scriptor_server::GenerateRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<scriptor_server::GenEvent>, EngineError> {
        panic!("injected handler panic");
    }
}

#[tokio::test]
async fn handler_panic_is_a_500_and_the_server_keeps_serving() {
    let inner = ScriptedEngine::new("glm-ocr", 16384, &["a"]);
    let state = AppState::new(
        Arc::new(PanickingEngine(inner)),
        Arc::new(Metrics::new()),
        AppConfig::default(),
    );
    let app = router(state);
    let (status, _, bytes) = send(&app, post(&fixture("semantik_request.json"))).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(json_of(&bytes)["error"]["type"], "InternalServerError");
    let (status, _, _) = send(&app, get("/health")).await;
    assert_eq!(status, StatusCode::OK);
}

fn sse_chunks(bytes: Vec<u8>) -> Vec<Value> {
    let text = String::from_utf8(bytes).unwrap();
    text.lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .map(str::trim)
        .filter(|f| *f != "[DONE]")
        .map(|f| serde_json::from_str(f).unwrap())
        .collect()
}

#[tokio::test]
async fn eos_stop_has_a_null_stop_reason_in_json_and_none_in_sse() {
    let app = app_with(engine(), AppConfig::default());
    let (_, _, bytes) = send(&app, post(&fixture("semantik_request.json"))).await;
    let reply = json_of(&bytes);
    assert_eq!(reply["choices"][0]["finish_reason"], "stop");
    assert!(reply["choices"][0]["stop_reason"].is_null());

    let mut body = fixture("semantik_request.json");
    body["stream"] = json!(true);
    let (_, _, bytes) = send(&app, post(&body)).await;
    for c in sse_chunks(bytes) {
        assert!(c["choices"][0].get("stop_reason").is_none());
    }
}

#[tokio::test]
async fn repetition_stop_reports_stop_reason_in_json() {
    let eng =
        Arc::new(ScriptedEngine::new("glm-ocr", 16384, &["a", "b"]).with_stop_reason("repetition"));
    let app = app_with(eng, AppConfig::default());
    let (status, _, bytes) = send(&app, post(&fixture("semantik_request.json"))).await;
    assert_eq!(status, StatusCode::OK);
    let reply = json_of(&bytes);
    assert_eq!(reply["choices"][0]["finish_reason"], "stop");
    assert_eq!(reply["choices"][0]["stop_reason"], "repetition");
}

#[tokio::test]
async fn repetition_stop_reports_stop_reason_on_the_final_sse_chunk_only() {
    let eng =
        Arc::new(ScriptedEngine::new("glm-ocr", 16384, &["a", "b"]).with_stop_reason("repetition"));
    let app = app_with(eng, AppConfig::default());
    let mut body = fixture("semantik_request.json");
    body["stream"] = json!(true);
    let (_, _, bytes) = send(&app, post(&body)).await;
    let chunks = sse_chunks(bytes);
    let finals: Vec<&Value> = chunks
        .iter()
        .filter(|c| {
            c.pointer("/choices/0/finish_reason")
                .is_some_and(|f| f == "stop")
        })
        .collect();
    assert_eq!(finals.len(), 1);
    assert_eq!(finals[0]["choices"][0]["stop_reason"], "repetition");
    let tagged = chunks
        .iter()
        .filter(|c| c.pointer("/choices/0/stop_reason").is_some())
        .count();
    assert_eq!(tagged, 1);
}
