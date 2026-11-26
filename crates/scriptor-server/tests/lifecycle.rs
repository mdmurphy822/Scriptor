//! Health states, drain refusals and distinct cancelled finishes over the scripted engine.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use scriptor_server::{AppConfig, AppState, Lifecycle, Metrics, ScriptedEngine, router};
use serde_json::{Value, json};
use tower::ServiceExt;

fn app(engine: Arc<ScriptedEngine>) -> Router {
    router(AppState::new(
        engine,
        Arc::new(Metrics::new()),
        AppConfig::default(),
    ))
}

fn engine() -> Arc<ScriptedEngine> {
    Arc::new(ScriptedEngine::new("glm-ocr", 16384, &["a", "b", "c"]))
}

async fn send(app: &Router, request: Request<Body>) -> (StatusCode, Vec<u8>) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    (status, body.to_vec())
}

fn chat(stream: bool) -> Request<Body> {
    let body = json!({
        "model": "glm-ocr",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
    });
    Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn get(path: &str) -> Request<Body> {
    Request::get(path).body(Body::empty()).unwrap()
}

#[tokio::test]
async fn health_is_503_draining_at_once_and_dead_names_the_cause() {
    let eng = engine();
    let app = app(eng.clone());
    let (status, _) = send(&app, get("/health")).await;
    assert_eq!(status, StatusCode::OK);

    eng.set_lifecycle(Lifecycle::Draining);
    let (status, body) = send(&app, get("/health")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let report: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(report["status"], "draining");

    eng.set_lifecycle(Lifecycle::Dead("the engine thread panicked: boom".into()));
    let (status, body) = send(&app, get("/health")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let report: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(report["state"], "dead");
    assert!(report["reason"].as_str().unwrap().contains("boom"));
}

#[tokio::test]
async fn a_draining_or_dead_engine_refuses_requests_with_503() {
    let eng = engine();
    let app = app(eng.clone());
    eng.set_lifecycle(Lifecycle::Draining);
    let (status, body) = send(&app, chat(false)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(String::from_utf8_lossy(&body).contains("draining"));
    eng.set_lifecycle(Lifecycle::Dead("x".into()));
    let (status, _) = send(&app, chat(false)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn a_cancelled_stream_ends_with_a_distinct_finish_reason() {
    let eng = engine();
    eng.cancel_with("drain");
    let app = app(eng);
    let (status, body) = send(&app, chat(true)).await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(body).unwrap();
    let frames: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .map(str::trim)
        .collect();
    assert_eq!(frames.last(), Some(&"[DONE]"));
    let last: Value = serde_json::from_str(frames[frames.len() - 2]).unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "cancelled");
    assert_eq!(last["cancel_reason"], "drain");
}

#[tokio::test]
async fn a_cancelled_json_request_is_a_503_not_a_partial_success() {
    let eng = engine();
    eng.cancel_with("drain");
    let app = app(eng);
    let (status, body) = send(&app, chat(false)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let report: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(report["error"]["type"], "RequestCancelled");
    let (_, metrics) = send(&app, get("/metrics")).await;
    let metrics = String::from_utf8(metrics).unwrap();
    assert!(
        metrics.contains(
            "scriptor_requests_cancelled_total{model_name=\"glm-ocr\",reason=\"drain\"} 1"
        )
    );
    assert!(metrics.contains("scriptor_engine_state{model_name=\"glm-ocr\",state=\"running\"} 1"));
}
