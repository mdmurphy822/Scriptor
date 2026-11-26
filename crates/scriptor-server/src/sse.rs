//! Server-sent-event streaming of chat completions.

use std::convert::Infallible;
use std::time::Instant;

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::app::{AppState, GenEvent, unix_now};
use crate::chat::{DeliveryClock, record_finished};
use crate::usage::Usage;

fn chunk(
    id: &str,
    model: &str,
    created: u64,
    delta: Value,
    finish: Option<&str>,
    stop_reason: Option<&str>,
) -> Event {
    let mut body = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "logprobs": null,
            "finish_reason": finish,
        }],
    });
    if let Some(reason) = stop_reason {
        body["choices"][0]["stop_reason"] = json!(reason);
    }
    Event::default().data(body.to_string())
}

/// Streams the engine events as `chat.completion.chunk` SSE frames ending in `[DONE]`.
/// A dropped client connection ends the task and releases the engine stream.
pub fn stream_response(
    state: AppState,
    mut rx: mpsc::Receiver<GenEvent>,
    id: String,
    model: String,
    include_usage: bool,
    started: Instant,
    reservation: Option<crate::intake::Reservation>,
) -> Response {
    let (tx, out) = mpsc::channel::<Result<Event, Infallible>>(64);
    tokio::spawn(async move {
        let _reservation = reservation;
        let created = unix_now();
        let mut clock = DeliveryClock::new(started);
        if tx
            .send(Ok(chunk(
                &id,
                &model,
                created,
                json!({"role": "assistant", "content": ""}),
                None,
                None,
            )))
            .await
            .is_err()
        {
            return;
        }
        while let Some(event) = rx.recv().await {
            match event {
                GenEvent::Delta(piece) => {
                    clock.delivered(&state.metrics);
                    let frame = chunk(&id, &model, created, json!({"content": piece}), None, None);
                    if tx.send(Ok(frame)).await.is_err() {
                        return;
                    }
                }
                GenEvent::Queued(seconds) => state.metrics.observe_queue_time(seconds),
                GenEvent::Finished {
                    finish_reason,
                    stop_reason,
                    prompt_tokens,
                    completion_tokens,
                } => {
                    let usage = Usage::new(prompt_tokens, completion_tokens);
                    record_finished(&state, &usage, &finish_reason, started, clock.first());
                    let last = chunk(
                        &id,
                        &model,
                        created,
                        json!({}),
                        Some(&finish_reason),
                        stop_reason.as_deref(),
                    );
                    if tx.send(Ok(last)).await.is_err() {
                        return;
                    }
                    if include_usage {
                        let body = json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": usage,
                        });
                        let frame = Event::default().data(body.to_string());
                        if tx.send(Ok(frame)).await.is_err() {
                            return;
                        }
                    }
                    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                    return;
                }
                GenEvent::Cancelled { reason } => {
                    state.metrics.record_cancelled(&reason);
                    let mut last = json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "logprobs": null,
                            "finish_reason": "cancelled",
                        }],
                    });
                    last["cancel_reason"] = json!(reason);
                    let frame = Event::default().data(last.to_string());
                    if tx.send(Ok(frame)).await.is_err() {
                        return;
                    }
                    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                    return;
                }
                #[cfg(feature = "kv-demand-diagnostic")]
                GenEvent::Restarted => {
                    let body = json!({"error": {
                        "message": "a streamed request was restarted by the engine",
                        "type": "InternalServerError",
                        "param": null,
                        "code": 500,
                    }});
                    let _ = tx.send(Ok(Event::default().data(body.to_string()))).await;
                    return;
                }
                GenEvent::QueueFull { retry_after } => {
                    let body = json!({"error": {
                        "message": "The admission queue is full; retry after the indicated delay.",
                        "type": "TooManyRequests",
                        "param": null,
                        "code": 429,
                        "retry_after": retry_after,
                    }});
                    let _ = tx.send(Ok(Event::default().data(body.to_string()))).await;
                    return;
                }
                GenEvent::Failed(message) => {
                    let body = json!({"error": {
                        "message": message,
                        "type": "InternalServerError",
                        "param": null,
                        "code": 500,
                    }});
                    let _ = tx.send(Ok(Event::default().data(body.to_string()))).await;
                    return;
                }
            }
        }
    });
    Sse::new(ReceiverStream::new(out))
        .keep_alive(KeepAlive::default())
        .into_response()
}
