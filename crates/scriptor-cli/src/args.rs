//! Command-line argument definitions for `scriptor`.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use scriptor_bridge::cli::OcrBridgeArgs;
use scriptor_bridge::serve::DEFAULT_PORT as DEFAULT_BRIDGE_PORT;

/// The `scriptor` command line.
#[derive(Parser, Debug)]
#[command(
    name = "scriptor",
    version,
    about = "Scriptor, a GLM-OCR inference engine"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// The engine subcommands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Serve the OpenAI-compatible endpoint and the document OCR bridge.
    Serve(Box<ServeArgs>),
    /// Serve the document OCR bridge against an engine running elsewhere.
    Bridge(OcrBridgeArgs),
    /// Convert a checkpoint to the engine-native format.
    Convert(ConvertArgs),
    /// Inspect a checkpoint directory or engine file.
    Inspect(InspectArgs),
    /// Report device, memory-plan and cache diagnostics.
    Doctor(DoctorArgs),
    /// Run benchmarks and equivalence checks.
    Bench,
    /// Tune GEMM kernels for the detected device.
    Tune,
}

/// Arguments of `serve`. Every path is explicit; none has a default, and nothing is written to
/// disk unless `--cache-dir` is given.
#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Checkpoint snapshot directory holding config.json, the tokenizer files and the
    /// safetensors weights.
    #[arg(long, visible_alias = "model-dir")]
    pub model: PathBuf,
    /// Directory for tuning results and boot records; without it no derived state is read or
    /// written.
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
    /// Engine-native weights file; accepted for seat compatibility, weights load from `--model`.
    #[arg(long, hide = true)]
    pub engine_file: Option<PathBuf>,
    /// Address to listen on.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// Port to listen on.
    #[arg(long, default_value_t = 8002)]
    pub port: u16,
    /// Listen address as `host:port`; overrides `--host` and `--port`.
    #[arg(long, hide = true)]
    pub bind: Option<String>,
    /// Model name exposed by /v1/models and accepted in requests.
    #[arg(long, default_value = "glm-ocr")]
    pub served_model_name: String,
    /// Fraction of device memory the engine may use.
    #[arg(long, visible_alias = "gpu-share", default_value_t = 0.60)]
    pub gpu_memory_utilization: f64,
    /// Concurrent sequences; absent derives the largest count whose slots can each hold a
    /// full context within the memory share.
    #[arg(long)]
    pub max_num_seqs: Option<usize>,
    /// Context length in tokens.
    #[arg(long, default_value_t = 16384)]
    pub max_model_len: usize,
    /// Image pixel ceiling; the model default applies when absent.
    #[arg(long)]
    pub max_pixels: Option<usize>,
    /// MTP draft tokens per step; 0 disables speculation.
    #[arg(long, visible_alias = "speculative-tokens", default_value_t = 0)]
    pub num_speculative_tokens: usize,
    /// KV commitment policy: `reserve` (default) or `demand-diagnostic`, which exists only in
    /// builds with the `kv-demand-diagnostic` feature.
    #[arg(long, env = "SCRIPTOR_KV_POLICY", default_value = "reserve")]
    pub kv_policy: scriptor_spec::runtime_params::KvPolicy,
    /// Run vision batches concurrently on their own stream instead of on the main stream
    /// one after another with decode.
    #[arg(long)]
    pub vision_overlap: bool,
    /// Release a sequence's image embeddings and their embedding-budget charge once its
    /// prefill completes; a sequence that is recomputed re-resolves them from the embedding
    /// cache or the vision tower. Off keeps them charged until the sequence ends.
    #[arg(long, env = "SCRIPTOR_EMBEDDING_RELEASE_AFTER_PREFILL")]
    pub embedding_release_after_prefill: bool,
    /// Split the device's SMs into a vision partition and a decode partition, as
    /// `vision=<sms>,decode=<sms>` (for example `vision=32,decode=16`), or `off` for the full
    /// device. Vision batches run on the vision partition and the decode graphs on the decode
    /// partition, which implies `--vision-overlap`. Absent, `vision=40,decode=8` is used on a
    /// 48-SM device that can grant it and the full device otherwise; an explicit split the
    /// device cannot grant fails boot.
    #[arg(long, value_name = "SPLIT|off")]
    pub sm_split: Option<SmSplitSetting>,
    /// Upper bound in seconds on the shutdown drain, for hosts that bound stopping (systemd
    /// `TimeoutStopSec`); absent leaves the derived deadline unbounded.
    #[arg(long)]
    pub drain_seconds: Option<f64>,
    /// Bytes of host memory the engine leaves for other tenants of the machine, added to the
    /// derived host reserve; absent derives the reserve from the host alone.
    #[arg(long)]
    pub host_reserve_bytes: Option<u64>,
    /// File the dispatched GEMM shapes are written to at shutdown, as the inventory
    /// `scriptor tune --shapes` reads. Recording counts each dispatch's shape and route,
    /// measures no time and never changes which algorithm runs.
    #[arg(long, value_name = "FILE")]
    pub gemm_shape_log: Option<PathBuf>,
    /// Port of the document OCR bridge that runs beside the engine.
    #[arg(long, default_value_t = DEFAULT_BRIDGE_PORT, conflicts_with = "no_bridge")]
    pub bridge_port: u16,
    /// Bind address of the bridge (default loopback; any other address needs a bearer token
    /// and TLS).
    #[arg(long, conflicts_with = "no_bridge")]
    pub bridge_listen: Option<String>,
    /// TLS certificate chain (PEM) of the bridge.
    #[arg(long, conflicts_with = "no_bridge")]
    pub bridge_tls_cert: Option<PathBuf>,
    /// TLS private key (PEM) of the bridge.
    #[arg(long, conflicts_with = "no_bridge")]
    pub bridge_tls_key: Option<PathBuf>,
    /// Environment variable holding the bridge's bearer token (default SCRIPTOR_BRIDGE_TOKEN,
    /// then SEMANTIK_BRIDGE_TOKEN, for a non-loopback bind).
    #[arg(long, conflicts_with = "no_bridge")]
    pub bridge_token_env: Option<String>,
    /// Serve the engine alone, without the document OCR bridge.
    #[arg(long)]
    pub no_bridge: bool,
}

impl ServeArgs {
    /// The options of the bridge that runs beside the engine, or `None` under `--no-bridge`.
    pub fn bridge_args(&self) -> Option<OcrBridgeArgs> {
        (!self.no_bridge).then(|| OcrBridgeArgs {
            serve: true,
            port: Some(self.bridge_port),
            listen: self.bridge_listen.clone(),
            tls_cert: self.bridge_tls_cert.clone(),
            tls_key: self.bridge_tls_key.clone(),
            token_env: self.bridge_token_env.clone(),
        })
    }

    /// The chat-completions URL the bridge reaches this engine at, over loopback: the
    /// listen address, with a wildcard host replaced by the loopback address of its family.
    pub fn bridge_engine_url(&self) -> String {
        let (host, port) = match &self.bind {
            Some(bind) => match bind.rsplit_once(':') {
                Some((host, port)) => (host.to_owned(), port.to_owned()),
                None => (bind.clone(), self.port.to_string()),
            },
            None => (self.host.clone(), self.port.to_string()),
        };
        let host = host.trim_start_matches('[').trim_end_matches(']');
        let host = match host.parse::<std::net::IpAddr>() {
            Ok(address) if address.is_unspecified() && address.is_ipv4() => "127.0.0.1".to_owned(),
            Ok(address) if address.is_unspecified() => "[::1]".to_owned(),
            Ok(address) if address.is_ipv6() => format!("[{address}]"),
            _ => host.to_owned(),
        };
        format!("http://{host}:{port}/v1/chat/completions")
    }

    /// The port the engine listens on.
    pub fn engine_port(&self) -> u16 {
        self.bind
            .as_deref()
            .and_then(|bind| bind.rsplit_once(':'))
            .and_then(|(_, port)| port.parse().ok())
            .unwrap_or(self.port)
    }
}

impl ServeArgs {
    /// The `host:port` string to bind.
    pub fn listen_address(&self) -> String {
        match &self.bind {
            Some(b) => b.clone(),
            None if self.host.contains(':') => format!("[{}]:{}", self.host, self.port),
            None => format!("{}:{}", self.host, self.port),
        }
    }
}

/// An explicit `--sm-split` value: a partition request or the full device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmSplitSetting {
    /// Use the full device.
    Off,
    /// Partition the device as requested.
    Split(scriptor_cuda::SmSplit),
}

impl std::str::FromStr for SmSplitSetting {
    type Err = scriptor_cuda::SmSplitError;

    /// Parses `off` or `vision=N,decode=N`.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text.trim() == "off" {
            Ok(Self::Off)
        } else {
            text.parse().map(Self::Split)
        }
    }
}

/// Arguments of `convert`.
#[derive(Args, Debug)]
pub struct ConvertArgs {
    /// Checkpoint snapshot directory holding the `*.safetensors` shards.
    #[arg(long)]
    pub model_dir: PathBuf,
    /// Engine file to write; replaced atomically.
    #[arg(long)]
    pub out: PathBuf,
}

/// Arguments of `inspect`.
#[derive(Args, Debug)]
pub struct InspectArgs {
    /// Checkpoint snapshot directory or engine file.
    #[arg(long)]
    pub path: PathBuf,
}

/// Arguments of `doctor`.
#[derive(Args, Debug)]
pub struct DoctorArgs {
    /// Checkpoint snapshot directory; enables the memory plan and checkpoint checks.
    #[arg(long, visible_alias = "model-dir")]
    pub model: Option<PathBuf>,
    /// Cache directory to report on.
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
    /// Concurrent sequences to plan for; absent plans for the derived count.
    #[arg(long)]
    pub max_num_seqs: Option<usize>,
    /// Context length to plan for.
    #[arg(long, default_value_t = 16384)]
    pub max_model_len: usize,
    /// Fraction of device memory to plan for.
    #[arg(long, visible_alias = "gpu-share", default_value_t = 0.60)]
    pub gpu_memory_utilization: f64,
    /// Draft tokens per step to plan for.
    #[arg(long, visible_alias = "speculative-tokens", default_value_t = 0)]
    pub num_speculative_tokens: usize,
    /// Plan against a device of this many GiB of memory, all free, without opening a device.
    #[arg(long)]
    pub device_memory_gib: Option<u64>,
    /// Bytes of host memory to leave for other tenants, added to the derived host reserve.
    #[arg(long)]
    pub host_reserve_bytes: Option<u64>,
}
impl ServeArgs {
    /// The host reserve hint in bytes: the flag, else `SCRIPTOR_HOST_RESERVE_BYTES`, else zero.
    pub fn reserve_hint(&self) -> u64 {
        reserve_hint(self.host_reserve_bytes)
    }
}

impl DoctorArgs {
    /// The host reserve hint in bytes: the flag, else `SCRIPTOR_HOST_RESERVE_BYTES`, else zero.
    pub fn reserve_hint(&self) -> u64 {
        reserve_hint(self.host_reserve_bytes)
    }
}

fn reserve_hint(flag: Option<u64>) -> u64 {
    flag.or_else(|| {
        std::env::var("SCRIPTOR_HOST_RESERVE_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
    })
    .unwrap_or(0)
}
