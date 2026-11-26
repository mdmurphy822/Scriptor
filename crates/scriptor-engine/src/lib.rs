//! Scheduler, sampler, speculative decoding loop and readiness.

pub mod active_telemetry;
pub mod boot;
pub mod decode_step;
#[cfg(feature = "kv-demand-diagnostic")]
mod demand_evict;
#[cfg(all(test, feature = "kv-demand-diagnostic"))]
mod demand_evict_tests;
#[cfg(all(test, feature = "kv-demand-diagnostic"))]
mod demand_tests;
pub mod drain;
pub mod embedding_budget;
#[cfg(test)]
mod embedding_release_tests;
pub mod embedding_reuse;
pub mod error;
pub mod handle;
pub mod host_mem;
pub mod image_ledger;
pub mod l1_counters;
pub mod pass_ledger;
pub mod prefill;
pub mod pressure;
#[cfg(test)]
mod rank_tests;
pub mod request;
#[cfg(test)]
mod requeue_order_tests;
pub mod sampling;
pub mod scheduler;
pub mod sequence;
pub mod slot_occupancy;
pub mod spec;
pub mod spec_log;
pub mod supervise;
pub mod vision_queue;

pub use drain::drain_deadline;
pub use error::EngineError;
pub use handle::{EngineExit, EngineHandle, FatalSlot, Heartbeat, RequestStream, fatal_slot};
pub use scheduler::{
    AdmissionBlocked, DecodeRow, DeviceRow, EngineMetrics, GraphStats, ModelExecutor, PrefillRow,
    Scheduler, SchedulerConfig, SpecRowOutput, StepReport, StepRowResult, VisionJob, VisionToken,
    validate_request,
};
pub use sequence::{
    CancelReason, EngineEvent, EngineRequest, FinishedEvent, ImageInput, RequestId, RequestTimings,
    TokenEvent,
};
pub use slot_occupancy::{
    BucketOccupancy, OCCUPANCY_BUCKETS, STEP_HISTOGRAM_LEN, SlotOccupancy, StepHistogram,
};
