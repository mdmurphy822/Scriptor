//! `GET /v1/models`: the served model and its context length, available once the engine is ready.

use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::app::AppState;
use crate::errors::ApiError;
use crate::health::all_done;

/// Lists the served model; 503 until every startup stage is done.
pub async fn list_models(State(state): State<AppState>) -> Response {
    if !all_done(&state.engine.health()) {
        return ApiError::unavailable("The engine is still starting.").into_response();
    }
    let id = state.engine.served_name();
    let stats = state.engine.stats();
    Json(json!({
        "object": "list",
        "data": [{
            "id": id,
            "object": "model",
            "created": state.started_unix,
            "owned_by": "oe",
            "root": id,
            "parent": null,
            "max_model_len": state.engine.max_model_len(),
            "max_num_seqs": stats.slots,
            "kv_cache_tokens": stats.kv_cache_tokens,
        }],
    }))
    .into_response()
}
