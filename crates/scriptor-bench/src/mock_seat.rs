//! A mock seat: answers the Semantik chat body instantly with canned text.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};

use crate::{BenchError, Result};

/// Configuration of a mock seat.
#[derive(Debug, Clone)]
pub struct MockConfig {
    /// The served model name listed by `/v1/models`.
    pub model: String,
    /// The text every reply carries.
    pub reply: String,
    /// The `prompt_tokens` every reply reports.
    pub prompt_tokens: u64,
    /// The `finish_reason` every reply reports.
    pub finish_reason: String,
}

impl MockConfig {
    /// A config with the served name `glm-ocr`, 247 prompt tokens and `stop`.
    pub fn new(reply: impl Into<String>) -> Self {
        Self {
            model: "glm-ocr".to_owned(),
            reply: reply.into(),
            prompt_tokens: 247,
            finish_reason: "stop".to_owned(),
        }
    }
}

struct Shared {
    config: MockConfig,
    served: AtomicU64,
}

fn completion_tokens(text: &str) -> u64 {
    ((text.chars().count() as f64 / 3.5).ceil() as u64).max(1)
}

async fn models(State(s): State<Arc<Shared>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": s.config.model, "object": "model", "owned_by": "mock-seat"}],
    }))
}

async fn chat(State(s): State<Arc<Shared>>, Json(_body): Json<Value>) -> Json<Value> {
    let n = s.served.fetch_add(1, Ordering::Relaxed) + 1;
    Json(json!({
        "id": format!("chatcmpl-mock-{n}"),
        "object": "chat.completion",
        "created": 0,
        "model": s.config.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": s.config.reply},
            "logprobs": null,
            "finish_reason": s.config.finish_reason,
        }],
        "usage": {
            "prompt_tokens": s.config.prompt_tokens,
            "completion_tokens": completion_tokens(&s.config.reply),
            "total_tokens": s.config.prompt_tokens + completion_tokens(&s.config.reply),
        },
    }))
}

fn router(shared: Arc<Shared>) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .with_state(shared)
}

fn runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(BenchError::from)
}

/// Serves until the process is stopped; returns only on failure.
pub fn serve_forever(bind: SocketAddr, config: MockConfig) -> Result<()> {
    let rt = runtime()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(bind).await?;
        let shared = Arc::new(Shared {
            config,
            served: AtomicU64::new(0),
        });
        axum::serve(listener, router(shared)).await?;
        Ok(())
    })
}

/// A mock seat running on a background runtime until dropped.
pub struct RunningSeat {
    addr: SocketAddr,
    shared: Arc<Shared>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    rt: Option<tokio::runtime::Runtime>,
}

impl RunningSeat {
    /// Binds `bind` (port 0 picks a free port) and starts serving.
    pub fn start(bind: SocketAddr, config: MockConfig) -> Result<Self> {
        let rt = runtime()?;
        let shared = Arc::new(Shared {
            config,
            served: AtomicU64::new(0),
        });
        let listener = rt.block_on(tokio::net::TcpListener::bind(bind))?;
        let addr = listener.local_addr()?;
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let app = router(shared.clone());
        rt.spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await;
        });
        Ok(Self {
            addr,
            shared,
            stop: Some(tx),
            rt: Some(rt),
        })
    }

    /// The base URL of the seat.
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Chat requests answered so far.
    pub fn served(&self) -> u64 {
        self.shared.served.load(Ordering::Relaxed)
    }
}

impl Drop for RunningSeat {
    fn drop(&mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        if let Some(rt) = self.rt.take() {
            rt.shutdown_background();
        }
    }
}
