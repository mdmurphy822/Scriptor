//! `scriptor bench compare|ladder|mock-seat|regions|gemm|reference-dump`: endpoint
//! equivalence, throughput ladder, instant mock seat, region-set generation, sustained GEMM
//! throughput and the CPU reference dump.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use clap::{Args, Subcommand};
use scriptor_bench::equivalence::compare;
use scriptor_bench::ladder::{DEFAULT_LEVELS, run_ladder};
use scriptor_bench::mock_seat::{MockConfig, serve_forever};
use scriptor_bench::openai_client::{Client, RetryPolicy};
use scriptor_bench::regions::{
    SYNTHETIC_MANIFEST, Task, bridge_mix, cases, load_font, load_region_dir,
    parse_synthetic_manifest, synthetic_regions, write_region_dir,
};
use scriptor_bench::report;
use scriptor_chat::Tokenizer;

use crate::bench_gemm::{GemmBenchArgs, run_bench};
use crate::reference_dump::{self, ReferenceDumpArgs};

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

fn parse_tasks(names: &[String]) -> Result<Option<Vec<Task>>, String> {
    if names.is_empty() {
        return Ok(None);
    }
    names
        .iter()
        .map(|n| Task::from_name(n).ok_or_else(|| format!("unknown task `{n}`")))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn write_reports(
    common: &EndpointArgs,
    markdown: &str,
    json: &serde_json::Value,
) -> Result<(), String> {
    print!("{markdown}");
    if let Some(p) = &common.report_md {
        std::fs::write(p, markdown).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    if let Some(p) = &common.report_json {
        let text = serde_json::to_string_pretty(json).map_err(|e| e.to_string())?;
        std::fs::write(p, text + "\n").map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(())
}

fn client(common: &EndpointArgs, url: &str) -> Client {
    Client::new(
        url,
        common.bearer.clone(),
        RetryPolicy {
            attempts: common.retries,
            ..RetryPolicy::default()
        },
    )
}

fn run_compare(args: &CompareArgs) -> Result<i32, String> {
    let regions = load_region_dir(&args.common.regions).map_err(|e| e.to_string())?;
    let tasks = parse_tasks(&args.tasks)?;
    let list = cases(&regions, tasks.as_deref());
    let tokenizer = Tokenizer::from_file(&args.tokenizer).map_err(|e| e.to_string())?;
    let (a, b) = (client(&args.common, &args.a), client(&args.common, &args.b));
    let r = compare(
        &a,
        &b,
        (&args.a, &args.b),
        &args.common.model,
        &tokenizer,
        &list,
        args.workers,
        args.logprobs,
    );
    write_reports(
        &args.common,
        &report::equivalence_markdown(&r),
        &report::equivalence_json(&r),
    )?;
    Ok(if r.all_identical() { 0 } else { 1 })
}

fn run_ladder_cmd(args: &LadderArgs) -> Result<i32, String> {
    let regions = load_region_dir(&args.common.regions).map_err(|e| e.to_string())?;
    let c = client(&args.common, &args.url);
    let peak = args.levels.iter().copied().max().unwrap_or(1);
    let list = bridge_mix(&regions, peak * args.rounds.max(1));
    let levels = run_ladder(&c, &args.common.model, &list, &args.levels, args.rounds);
    write_reports(
        &args.common,
        &report::ladder_markdown(&args.url, &args.common.model, &levels),
        &report::ladder_json(&args.url, &args.common.model, &levels),
    )?;
    Ok(if levels.iter().any(|l| l.errors > 0) {
        1
    } else {
        0
    })
}

fn run_mock_seat(args: &MockSeatArgs) -> Result<i32, String> {
    let reply = std::fs::read_to_string(&args.reply_file)
        .map_err(|e| format!("{}: {e}", args.reply_file.display()))?;
    let mut config = MockConfig::new(reply.trim_end());
    config.model = args.model.clone();
    config.prompt_tokens = args.prompt_tokens;
    serve_forever(args.bind, config).map_err(|e| e.to_string())?;
    Ok(0)
}

fn run_regions(args: &RegionsArgs) -> Result<i32, String> {
    let manifest = match &args.manifest {
        Some(p) => std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?,
        None => SYNTHETIC_MANIFEST.to_owned(),
    };
    let specs = parse_synthetic_manifest(&manifest).map_err(|e| e.to_string())?;
    let font = load_font(args.font.as_deref()).map_err(|e| e.to_string())?;
    let regions = synthetic_regions(&specs, &font).map_err(|e| e.to_string())?;
    write_region_dir(Path::new(&args.out), &regions).map_err(|e| e.to_string())?;
    println!("wrote {} regions to {}", regions.len(), args.out.display());
    Ok(0)
}

/// Runs a `bench` subcommand; returns the process exit code.
pub fn run(command: BenchCommand) -> i32 {
    let outcome = match command {
        BenchCommand::Compare(a) => run_compare(&a),
        BenchCommand::Ladder(a) => run_ladder_cmd(&a),
        BenchCommand::MockSeat(a) => run_mock_seat(&a),
        BenchCommand::Regions(a) => run_regions(&a),
        BenchCommand::Gemm(a) => run_bench(a),
        BenchCommand::ReferenceDump(a) => reference_dump::run(&a)
            .map(|()| 0)
            .map_err(|e| e.to_string()),
    };
    match outcome {
        Ok(code) => code,
        Err(e) => {
            eprintln!("scriptor bench: {e}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{Cli, Command};
    use clap::{CommandFactory, Parser};

    fn bench(argv: &[&str]) -> BenchCommand {
        let mut full = vec!["scriptor", "bench"];
        full.extend_from_slice(argv);
        match Cli::try_parse_from(full).unwrap().command {
            Command::Bench { command } => command,
            _ => panic!("expected bench"),
        }
    }

    #[test]
    fn the_command_tree_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn bench_gemm_parses_its_dtype_and_row_counts() {
        match bench(&["gemm", "--dtype", "fp8", "--m", "32,128"]) {
            BenchCommand::Gemm(a) => {
                assert_eq!(a.dtype, crate::bench_gemm::DtypeArg::Fp8);
                assert_eq!(a.m, vec![32, 128]);
            }
            other => panic!("expected gemm, got {other:?}"),
        }
    }

    #[test]
    fn bench_regions_and_reference_dump_parse() {
        assert!(matches!(
            bench(&["regions", "--out", "d"]),
            BenchCommand::Regions(_)
        ));
        assert!(matches!(
            bench(&[
                "reference-dump",
                "--model-dir",
                "m",
                "--image",
                "i.png",
                "--prompt",
                "Text Recognition:"
            ]),
            BenchCommand::ReferenceDump(_)
        ));
    }

    #[test]
    fn tune_requires_a_cache_dir_and_reads_it() {
        assert!(Cli::try_parse_from(["scriptor", "tune"]).is_err());
        match Cli::try_parse_from(["scriptor", "tune", "--cache-dir", "c", "--fp8"])
            .unwrap()
            .command
        {
            Command::Tune(a) => {
                assert_eq!(a.cache_dir, std::path::PathBuf::from("c"));
                assert!(a.fp8);
            }
            _ => panic!("expected tune"),
        }
    }
}
