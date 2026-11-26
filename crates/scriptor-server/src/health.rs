//! Staged readiness reporting for `GET /health`.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::json;

use crate::app::{AppState, Lifecycle};

/// One startup stage and whether it has completed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Stage {
    /// Stage name, such as `weights` or `graphs`.
    pub name: String,
    /// Whether the stage has completed.
    pub done: bool,
}

impl Stage {
    /// Builds a stage record.
    pub fn new(name: impl Into<String>, done: bool) -> Self {
        Self {
            name: name.into(),
            done,
        }
    }
}

/// Why an engine stopped serving for good.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Dead {
    /// Driver result code of the sticky device fault, when there is one.
    pub cuda: Option<i32>,
    /// What stopped the engine.
    pub message: String,
}

/// Whether every stage has completed.
pub fn all_done(stages: &[Stage]) -> bool {
    stages.iter().all(|s| s.done)
}

/// `GET /health`: 200 once every stage is done, 503 with the stage list before. A draining or
/// dead engine answers 503 with `status` `draining` or `dead` regardless of its stages; a dead
/// engine names its cause as the driver `cuda` code and `message` of a device fault, or as the
/// `reason` of any other death.
pub async fn health(State(state): State<AppState>) -> Response {
    if let Some(dead) = state.engine.dead() {
        let body = json!({
            "status": "dead",
            "state": "dead",
            "cuda": dead.cuda.map(|c| c.to_string()),
            "message": dead.message,
            "stages": state.engine.health(),
        });
        return (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response();
    }
    let stages = state.engine.health();
    match state.engine.lifecycle() {
        Lifecycle::Running => {}
        Lifecycle::Draining => {
            let body = json!({"status": "draining", "state": "draining", "stages": stages});
            return (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response();
        }
        Lifecycle::Dead(reason) => {
            let body =
                json!({"status": "dead", "state": "dead", "reason": reason, "stages": stages});
            return (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response();
        }
    }
    let ready = all_done(&stages);
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    let body = json!({"status": if ready { "ready" } else { "starting" }, "stages": stages});
    (status, Json(body)).into_response()
}
