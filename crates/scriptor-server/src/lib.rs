//! OpenAI-compatible HTTP surface: model listing, chat completions (JSON and SSE),
//! Prometheus metrics and staged health, served over an engine trait.

pub mod app;
pub mod chat;
pub mod errors;
pub mod health;
pub mod intake;
pub mod metrics;
pub mod models;
pub mod occupancy;
pub mod relay;
pub mod sse;
pub mod usage;

pub use app::{
    AppConfig, AppState, Engine, EngineStats, GenEvent, GenerateRequest, Lifecycle, ScriptedEngine,
    router,
};
pub use errors::{ApiError, EngineError};
pub use health::{Dead, Stage};
pub use intake::{Intake, IntakeBudget};
pub use metrics::Metrics;
pub use relay::{Next, next_event};
pub use usage::Usage;
