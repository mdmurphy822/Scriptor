//! `scriptor bench compare|ladder|mock-seat|regions|gemm|reference-dump`: endpoint
//! equivalence, throughput ladder, instant mock seat, region-set generation, sustained GEMM
//! throughput and the CPU reference dump.

use std::path::Path;

use scriptor_bench::equivalence::compare;
use scriptor_bench::ladder::run_ladder;
use scriptor_bench::mock_seat::{MockConfig, serve_forever};
use scriptor_bench::openai_client::{Client, RetryPolicy};
use scriptor_bench::regions::{
    SYNTHETIC_MANIFEST, Task, bridge_mix, cases, load_font, load_region_dir,
    parse_synthetic_manifest, synthetic_regions, write_region_dir,
};
use scriptor_bench::report;
use scriptor_chat::Tokenizer;

pub use crate::args::bench_args::{
    BenchCommand, CompareArgs, EndpointArgs, LadderArgs, MockSeatArgs, RegionsArgs,
};
use crate::bench_gemm::run_bench;
use crate::reference_dump;

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
