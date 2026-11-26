//! Request, event and per-sequence state types shared by the scheduler stages.

use crate::error::EngineError;
use crate::request::SamplingParams;
use crate::sampling::{FinishReason, SequenceSampler};
use scriptor_cache::{EmbeddingKey, ImageGrid};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::UnboundedSender;

/// Identifier of a request; also its sequence id in the block pool.
pub type RequestId = u64;

/// One decoded and resized image of a request.
#[derive(Debug, Clone)]
pub struct ImageInput {
    /// Patch grid after resizing.
    pub grid: ImageGrid,
    /// Content key of the decoded pixels plus grid.
    pub key: EmbeddingKey,
    /// Executor-defined pixel payload (patchified or decoded bytes).
    pub payload: Arc<Vec<u8>>,
}

impl ImageInput {
    /// Patches the vision tower processes for this image.
    pub fn patches(&self) -> usize {
        self.grid.t as usize * self.grid.h as usize * self.grid.w as usize
    }

    /// Embedding rows the image contributes after merging `merge`×`merge` patches.
    pub fn tokens(&self, merge: usize) -> usize {
        self.patches() / (merge * merge)
    }
}

/// A fully tokenised request: placeholder-expanded prompt ids, images in prompt order, sampling.
#[derive(Debug, Clone)]
pub struct EngineRequest {
    /// Prompt token ids with every image placeholder expanded.
    pub prompt: Vec<u32>,
    /// Images in the order their placeholder runs appear in the prompt.
    pub images: Vec<ImageInput>,
    /// Validated sampling parameters.
    pub params: SamplingParams,
}

/// Wall-clock time a request spent in each stage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RequestTimings {
    /// From submission to the first stage that started it (admission).
    pub queue: Duration,
    /// Vision batches the request took part in.
    pub vision: Duration,
    /// Prefill passes the request took part in.
    pub prefill: Duration,
    /// Decode steps the request took part in.
    pub decode: Duration,
}

/// One generated token delivered to the consumer.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenEvent {
    pub id: RequestId,
    /// Position among the generated tokens, from zero.
    pub index: u32,
    pub token: u32,
    /// Best minus second-best penalised logit.
    pub margin: f32,
    pub logprob: Option<f32>,
    pub top_logprobs: Vec<(u32, f32)>,
}

/// Terminal state of a completed request.
#[derive(Debug, Clone, PartialEq)]
pub struct FinishedEvent {
    pub id: RequestId,
    pub reason: FinishReason,
    pub prompt_tokens: usize,
    /// Generated tokens including a terminating end-of-sequence id.
    pub completion_tokens: usize,
    pub timings: RequestTimings,
    /// Times the sequence was evicted and recomputed.
    pub preemptions: u32,
}

/// Why the engine cancelled a request on its own initiative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelReason {
    /// The engine is shutting down and its drain deadline passed or a second signal arrived.
    Drain,
}

impl CancelReason {
    /// Stable lowercase label used on the wire and in metrics.
    pub fn label(self) -> &'static str {
        match self {
            CancelReason::Drain => "drain",
        }
    }
}

/// Messages on a request's event channel; exactly one terminal event ends the stream.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    /// A generated token (end-of-sequence ids are not delivered as tokens).
    Token(TokenEvent),
    /// The request completed.
    Finished(FinishedEvent),
    #[cfg(feature = "kv-demand-diagnostic")]
    /// The request was evicted and starts over from its prompt: every token delivered so far
    /// is void and the full output follows again.
    Restarted,
    /// The request was cancelled at its consumer's request.
    Aborted,
    /// The engine cancelled the request itself.
    Cancelled(CancelReason),
    /// The request failed.
    Failed(EngineError),
}

/// Where a sequence is in the step pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// Holds blocks, waits for image embeddings.
    Vision,
    /// Embeddings ready, waits for a prefill pass.
    Prefill,
    /// Decoding.
    Running,
}

pub(crate) struct Sequence<B> {
    pub(crate) id: RequestId,
    /// Submission sequence number, assigned once and kept across requeues; lower is older.
    pub(crate) rank: u64,
    pub(crate) prompt: Vec<u32>,
    pub(crate) images: Vec<ImageInput>,
    pub(crate) embeds: Vec<Option<Arc<B>>>,
    pub(crate) sampler: SequenceSampler,
    pub(crate) generated: Vec<u32>,
    pub(crate) tx: UnboundedSender<EngineEvent>,
    pub(crate) timings: RequestTimings,
    pub(crate) submitted: Instant,
    pub(crate) admitted_once: bool,
    pub(crate) preemptions: u32,
    pub(crate) phase: Phase,
    /// Set after a prefill that left the next token to the device step; cleared when that
    /// step delivers it.
    pub(crate) device_pending: bool,
    /// Whether an eviction may discard the output and run the request again from its prompt.
    #[cfg(feature = "kv-demand-diagnostic")]
    pub(crate) restartable: bool,
}

impl<B> Sequence<B> {
    /// Tokens the sequence currently occupies: prompt plus generated.
    pub(crate) fn len(&self) -> usize {
        self.prompt.len() + self.generated.len()
    }

    /// Prompt followed by generated ids: the tokens a recompute prefill replays.
    pub(crate) fn context(&self) -> Vec<u32> {
        let mut t = Vec::with_capacity(self.len());
        t.extend_from_slice(&self.prompt);
        t.extend_from_slice(&self.generated);
        t
    }

    /// The token a decode step feeds: the last generated id, or the last prompt id before
    /// anything is generated.
    pub(crate) fn last_token(&self) -> u32 {
        *self
            .generated
            .last()
            .or_else(|| self.prompt.last())
            .expect("a sequence holds at least one token")
    }

    pub(crate) fn embeddings_ready(&self) -> bool {
        self.embeds.iter().all(Option::is_some)
    }
}
