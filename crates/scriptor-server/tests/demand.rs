//! HTTP behaviour under the demand KV policy: full queue, restarts and the streaming refusal.
#![cfg(feature = "kv-demand-diagnostic")]

use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use scriptor_server::{AppConfig, AppState, EngineError, Metrics, ScriptedEngine, router};
use serde_json::{Value, json};
use tower::ServiceExt;

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/server/semantik_request.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn app(engine: Arc<ScriptedEngine>) -> Router {
    router(AppState::new(
        engine,
        Arc::new(Metrics::new()),
        AppConfig::default(),
    ))
}

fn engine() -> Arc<ScriptedEngine> {
    Arc::new(ScriptedEngine::new(
        "glm-ocr",
        16384,
        &["Hello", ", ", "world"],
    ))
}

async fn send(app: &Router, body: &Value) -> (StatusCode, axum::http::HeaderMap, Value) {
    let request = Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, headers, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn a_full_admission_queue_is_a_429_with_retry_after() {
    let eng = engine();
    eng.fail_with(EngineError::QueueFull { retry_after: 3 });
    let (status, headers, _) = send(&app(eng), &fixture()).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(headers[header::RETRY_AFTER], "3");
}

#[tokio::test]
async fn a_restart_discards_partial_output_and_the_reply_holds_the_final_text() {
    let eng = engine();
    eng.restart_after(2);
    let (status, _, reply) = send(&app(eng.clone()), &fixture()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reply["choices"][0]["message"]["content"], "Hello, world");
    assert!(eng.received()[0].restartable);
}

#[tokio::test]
async fn streaming_is_refused_with_a_400_and_never_reaches_the_engine() {
    let eng = engine();
    eng.refuse_streaming();
    let mut body = fixture();
    body["stream"] = json!(true);
    let (status, _, reply) = send(&app(eng.clone()), &body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        reply["error"]["message"],
        "streaming is unsupported under kv_policy=demand-diagnostic"
    );
    assert!(eng.received().is_empty());
}

#[tokio::test]
async fn non_streaming_requests_are_marked_restartable() {
    let eng = engine();
    eng.refuse_streaming();
    let (status, _, _) = send(&app(eng.clone()), &fixture()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(eng.received()[0].restartable);
}

#[tokio::test]
async fn a_request_refused_by_engine_intake_after_the_first_check_is_a_429_with_retry_after() {
    let eng = engine();
    eng.refuse_at_intake(5);
    let (status, headers, reply) = send(&app(eng.clone()), &fixture()).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(headers[header::RETRY_AFTER], "5");
    assert_eq!(reply["error"]["type"], "TooManyRequests");
    assert_eq!(eng.received().len(), 1, "the request passed submit");
}

#[tokio::test]
async fn usage_after_a_restart_counts_only_the_final_attempt() {
    let eng = engine();
    eng.restart_after(1);
    let (status, _, reply) = send(&app(eng), &fixture()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reply["choices"][0]["message"]["content"], "Hello, world");
    assert_eq!(reply["usage"]["completion_tokens"], 3);
}
