//! Command-line argument types of `bench` and `tune`.

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{Args, Subcommand, ValueEnum};
use scriptor_bench::ladder::DEFAULT_LEVELS;

/// Equivalence and throughput subcommands of `bench`.
#[derive(Subcommand, Debug)]
pub enum BenchCommand {
    /// Posts every (region, task) to two endpoints and compares replies token for token.
    Compare(CompareArgs),
    /// Measures throughput at rising concurrency with the bridge-shaped request mix.
    Ladder(LadderArgs),
    /// Answers the Semantik chat body instantly with canned text.
    MockSeat(MockSeatArgs),
    /// Writes a region set to a directory.
    Regions(RegionsArgs),
    /// Measures sustained GEMM throughput over the ViT and decoder shapes.
    Gemm(GemmBenchArgs),
    /// Runs the f32 CPU reference on a checkpoint and writes the greedy tokens with their top-2
    /// logit margins.
    ReferenceDump(ReferenceDumpArgs),
}

/// Arguments shared by the endpoint-driving subcommands.
#[derive(Args, Debug)]
pub struct EndpointArgs {
    /// Directory holding `manifest.json` and the crop JPEGs.
    #[arg(long)]
    pub regions: PathBuf,
    /// Served model name sent in the request body.
    #[arg(long, default_value = "glm-ocr")]
    pub model: String,
    /// Bearer token sent to the endpoints, if any.
    #[arg(long)]
    pub bearer: Option<String>,
    /// Retries on 429 and 5xx replies.
    #[arg(long, default_value_t = 5)]
    pub retries: usize,
    /// Writes the Markdown report here.
    #[arg(long)]
    pub report_md: Option<PathBuf>,
    /// Writes the JSON report here.
    #[arg(long)]
    pub report_json: Option<PathBuf>,
}

/// Arguments of `bench compare`.
#[derive(Args, Debug)]
pub struct CompareArgs {
    #[command(flatten)]
    pub common: EndpointArgs,
    /// Reference endpoint (base URL or chat-completions URL).
    #[arg(long)]
    pub a: String,
    /// Endpoint under test.
    #[arg(long)]
    pub b: String,
    /// `tokenizer.json` used to compare replies token for token.
    #[arg(long)]
    pub tokenizer: PathBuf,
    /// Tasks to post for every region instead of the manifest's (comma separated).
    #[arg(long, value_delimiter = ',')]
    pub tasks: Vec<String>,
    /// Cases in flight at once.
    #[arg(long, default_value_t = 8)]
    pub workers: usize,
    /// Request logprobs to report top-2 logit margins at the first divergence.
    #[arg(long)]
    pub logprobs: bool,
}

/// Arguments of `bench ladder`.
#[derive(Args, Debug)]
pub struct LadderArgs {
    #[command(flatten)]
    pub common: EndpointArgs,
    /// Endpoint to load (base URL or chat-completions URL).
    #[arg(long)]
    pub url: String,
    /// Concurrency levels (comma separated).
    #[arg(long, value_delimiter = ',', default_values_t = DEFAULT_LEVELS)]
    pub levels: Vec<usize>,
    /// Requests sent per concurrent slot at each level.
    #[arg(long, default_value_t = 4)]
    pub rounds: usize,
}

/// Arguments of `bench mock-seat`.
#[derive(Args, Debug)]
pub struct MockSeatArgs {
    /// Address to listen on.
    #[arg(long)]
    pub bind: SocketAddr,
    /// File whose text every reply carries.
    #[arg(long)]
    pub reply_file: PathBuf,
    /// Served model name listed by `/v1/models`.
    #[arg(long, default_value = "glm-ocr")]
    pub model: String,
    /// `prompt_tokens` every reply reports.
    #[arg(long, default_value_t = 247)]
    pub prompt_tokens: u64,
}

/// Arguments of `bench regions`.
#[derive(Args, Debug)]
pub struct RegionsArgs {
    /// Directory the region set is written to.
    #[arg(long)]
    pub out: PathBuf,
    /// Font file overriding the bundled font.
    #[arg(long)]
    pub font: Option<PathBuf>,
    /// Manifest overriding the bundled synthetic manifest.
    #[arg(long)]
    pub manifest: Option<PathBuf>,
}

/// Operand format selected on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DtypeArg {
    Bf16,
    Fp8,
}

/// Arguments of `bench gemm`.
#[derive(Args, Debug)]
pub struct GemmBenchArgs {
    /// Operand format to measure.
    #[arg(long, value_enum, default_value = "bf16")]
    pub dtype: DtypeArg,
    /// Writes the JSON report to this file.
    #[arg(long)]
    pub report: Option<PathBuf>,
    /// Reads the tuning table from this directory (never written by `bench`).
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
    /// Untimed launches per shape.
    #[arg(long, default_value_t = 50)]
    pub warmup: usize,
    /// Timed launches per shape; the median is reported.
    #[arg(long, default_value_t = 200)]
    pub iters: usize,
    /// Activation row counts to measure (comma separated).
    #[arg(long, value_delimiter = ',', default_values_t = scriptor_kernels::heuristics::PROBE_M)]
    pub m: Vec<usize>,
}

/// Arguments of `tune`.
#[derive(Args, Debug)]
pub struct TuneArgs {
    /// Directory the tuning table is written to (the only writer of the cache).
    #[arg(long)]
    pub cache_dir: PathBuf,
    /// Also tune the fp8 path.
    #[arg(long)]
    pub fp8: bool,
    /// Timed launches per candidate.
    #[arg(long, default_value_t = 20)]
    pub iters: usize,
    /// Activation row counts to tune (comma separated).
    #[arg(long, value_delimiter = ',', default_values_t = scriptor_kernels::heuristics::PROBE_M)]
    pub m: Vec<usize>,
    /// Tunes exactly the shapes of this dispatched-shape inventory instead of the model shapes
    /// at `--m`, ordered by total measured GPU time, then by executed dispatch count, so an
    /// untimed inventory (`serve --gemm-shape-log`) is ordered by how often each shape ran.
    #[arg(long)]
    pub shapes: Option<PathBuf>,
    /// Most shapes taken from `--shapes`.
    #[arg(long, default_value_t = 64)]
    pub max_shapes: usize,
}

/// Arguments of `bench reference-dump`.
#[derive(Debug, Args)]
pub struct ReferenceDumpArgs {
    /// Checkpoint snapshot directory (config, tokenizer, chat template and safetensors).
    #[arg(long)]
    pub model_dir: PathBuf,
    /// Image file (JPEG, PNG or WebP).
    #[arg(long)]
    pub image: PathBuf,
    /// Text part of the user message.
    #[arg(long)]
    pub prompt: String,
    /// Maximum number of tokens to generate.
    #[arg(long, default_value_t = 32)]
    pub steps: usize,
    /// Repetition penalty applied before the greedy pick.
    #[arg(long, default_value_t = 1.1)]
    pub repetition_penalty: f32,
    /// Write the JSON dump here instead of standard output.
    #[arg(long)]
    pub output: Option<PathBuf>,
}
