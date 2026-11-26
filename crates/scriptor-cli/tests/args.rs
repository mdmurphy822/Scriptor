//! Parsing of the `serve` and `convert` command lines.

#[path = "../src/args.rs"]
#[allow(dead_code)]
mod args;

use args::{Cli, Command, SmSplitSetting};
use clap::{CommandFactory, Parser};

fn parse(argv: &[&str]) -> Result<Cli, clap::Error> {
    Cli::try_parse_from(std::iter::once("scriptor").chain(argv.iter().copied()))
}

#[test]
fn serve_parses_every_flag() {
    let cli = parse(&[
        "serve",
        "--model",
        "m",
        "--cache-dir",
        "c",
        "--host",
        "0.0.0.0",
        "--port",
        "9000",
        "--served-model-name",
        "x",
        "--gpu-memory-utilization",
        "0.15",
        "--max-num-seqs",
        "16",
        "--max-model-len",
        "4096",
        "--max-pixels",
        "1000",
        "--num-speculative-tokens",
        "3",
    ])
    .unwrap();
    let Command::Serve(serve) = cli.command else {
        panic!("expected serve");
    };
    assert_eq!(serve.listen_address(), "0.0.0.0:9000");
    assert_eq!(serve.max_num_seqs, Some(16));
    assert_eq!(serve.max_pixels, Some(1000));
    assert_eq!(serve.num_speculative_tokens, 3);
    assert_eq!(serve.served_model_name, "x");
    assert_eq!(serve.cache_dir, Some(std::path::PathBuf::from("c")));
}

#[test]
fn serve_defaults_persist_nothing() {
    let cli = parse(&["serve", "--model", "m"]).unwrap();
    let Command::Serve(serve) = cli.command else {
        panic!("expected serve");
    };
    assert_eq!(serve.served_model_name, "glm-ocr");
    assert_eq!(serve.listen_address(), "127.0.0.1:8002");
    assert_eq!(serve.max_num_seqs, None);
    assert_eq!(serve.max_model_len, 16384);
    assert_eq!(serve.num_speculative_tokens, 0);
    assert!(serve.cache_dir.is_none());
    assert!(serve.engine_file.is_none());
    assert!(serve.max_pixels.is_none());
    assert!(serve.sm_split.is_none());
}

#[test]
fn serve_sm_split_off_selects_the_full_device() {
    let cli = parse(&["serve", "--model", "m", "--sm-split", "off"]).unwrap();
    let Command::Serve(serve) = cli.command else {
        panic!("expected serve");
    };
    assert_eq!(serve.sm_split, Some(SmSplitSetting::Off));
}

#[test]
fn serve_accepts_any_speculative_depth() {
    for k in ["1", "7", "12"] {
        let cli = parse(&["serve", "--model", "m", "--num-speculative-tokens", k]).unwrap();
        let Command::Serve(serve) = cli.command else {
            panic!("expected serve");
        };
        assert_eq!(serve.num_speculative_tokens.to_string(), k);
    }
}

#[test]
fn inspect_and_doctor_parse() {
    assert!(matches!(
        parse(&["inspect", "--path", "p"]).unwrap().command,
        Command::Inspect(_)
    ));
    assert!(matches!(
        parse(&["doctor", "--model", "m", "--num-speculative-tokens", "3"])
            .unwrap()
            .command,
        Command::Doctor(_)
    ));
    assert!(matches!(
        parse(&["doctor"]).unwrap().command,
        Command::Doctor(_)
    ));
}

#[test]
fn convert_parses() {
    let cli = parse(&["convert", "--model-dir", "m", "--out", "o"]).unwrap();
    assert!(matches!(cli.command, Command::Convert(_)));
}

#[test]
fn paths_are_required() {
    assert!(parse(&["serve"]).is_err());
    assert!(parse(&["serve", "--port", "1"]).is_err());
    assert!(parse(&["convert", "--model-dir", "m"]).is_err());
}

#[test]
fn no_default_embeds_a_path() {
    let command = Cli::command();
    for sub in command.get_subcommands() {
        for arg in sub.get_arguments() {
            for value in arg.get_default_values() {
                let text = value.to_string_lossy();
                assert!(
                    !text.starts_with('/') && !text.contains("/home/"),
                    "default of --{} embeds a path: {text}",
                    arg.get_id()
                );
            }
        }
    }
}

#[test]
fn serve_parses_the_sm_split() {
    let cli = parse(&["serve", "--model", "m", "--sm-split", "vision=32,decode=16"]).unwrap();
    let Command::Serve(serve) = cli.command else {
        panic!("expected serve");
    };
    let Some(SmSplitSetting::Split(split)) = serve.sm_split else {
        panic!("expected a split");
    };
    assert_eq!((split.vision, split.decode), (32, 16));
    let cli = parse(&["serve", "--model", "m", "--sm-split", "decode=8,vision=40"]).unwrap();
    let Command::Serve(serve) = cli.command else {
        panic!("expected serve");
    };
    let Some(SmSplitSetting::Split(split)) = serve.sm_split else {
        panic!("expected a split");
    };
    assert_eq!((split.vision, split.decode), (40, 8));
}

#[test]
fn serve_rejects_a_malformed_sm_split() {
    for text in [
        "32,16",
        "vision=32",
        "vision=32,decode=0",
        "vision=a,decode=16",
        "vision=32,decode=16,vision=8",
        "OFF",
        "off,vision=32",
        "",
    ] {
        assert!(
            parse(&["serve", "--model", "m", "--sm-split", text]).is_err(),
            "{text:?} was accepted"
        );
    }
}

fn serve_of(argv: &[&str]) -> args::ServeArgs {
    let Command::Serve(serve) = parse(argv).unwrap().command else {
        panic!("expected serve");
    };
    *serve
}

#[test]
fn serve_runs_the_bridge_on_the_default_port_by_default() {
    let serve = serve_of(&["serve", "--model", "m"]);
    let bridge = serve.bridge_args().expect("the bridge is on by default");
    assert!(bridge.serve);
    assert_eq!(bridge.port, Some(8765));
    assert!(bridge.listen.is_none() && bridge.tls_cert.is_none() && bridge.tls_key.is_none());
    assert!(bridge.token_env.is_none());
}

#[test]
fn serve_parses_every_bridge_flag() {
    let serve = serve_of(&[
        "serve",
        "--model",
        "m",
        "--bridge-port",
        "18002",
        "--bridge-listen",
        "0.0.0.0",
        "--bridge-tls-cert",
        "cert.pem",
        "--bridge-tls-key",
        "key.pem",
        "--bridge-token-env",
        "BRIDGE_TOKEN",
    ]);
    let bridge = serve.bridge_args().unwrap();
    assert_eq!(bridge.port, Some(18002));
    assert_eq!(bridge.listen.as_deref(), Some("0.0.0.0"));
    assert_eq!(bridge.tls_cert, Some(std::path::PathBuf::from("cert.pem")));
    assert_eq!(bridge.tls_key, Some(std::path::PathBuf::from("key.pem")));
    assert_eq!(bridge.token_env.as_deref(), Some("BRIDGE_TOKEN"));
}

#[test]
fn no_bridge_turns_the_bridge_off_and_excludes_its_options() {
    assert!(
        serve_of(&["serve", "--model", "m", "--no-bridge"])
            .bridge_args()
            .is_none()
    );
    for option in [
        ["--bridge-port", "9000"],
        ["--bridge-listen", "0.0.0.0"],
        ["--bridge-tls-cert", "c"],
        ["--bridge-tls-key", "k"],
        ["--bridge-token-env", "T"],
    ] {
        assert!(
            parse(&["serve", "--model", "m", "--no-bridge", option[0], option[1]]).is_err(),
            "{} was accepted with --no-bridge",
            option[0]
        );
    }
    assert!(parse(&["serve", "--model", "m", "--bridge-port", "70000"]).is_err());
}

#[test]
fn bridge_subcommand_parses_the_standalone_bridge_options() {
    let Command::Bridge(bridge) = parse(&[
        "bridge",
        "--serve",
        "--port",
        "18002",
        "--listen",
        "0.0.0.0",
        "--tls-cert",
        "cert.pem",
        "--tls-key",
        "key.pem",
        "--token-env",
        "BRIDGE_TOKEN",
    ])
    .unwrap()
    .command
    else {
        panic!("expected bridge");
    };
    assert!(bridge.serve);
    assert_eq!(bridge.port, Some(18002));
    assert_eq!(bridge.listen.as_deref(), Some("0.0.0.0"));
    assert_eq!(bridge.token_env.as_deref(), Some("BRIDGE_TOKEN"));
    assert!(parse(&["bridge", "--unknown"]).is_err());
    assert!(parse(&["bridge", "--port", "70000"]).is_err());
}

#[test]
fn the_bridge_reaches_the_engine_at_the_serve_host_and_port() {
    let url = |argv: &[&str]| {
        let mut full = vec!["serve", "--model", "m"];
        full.extend_from_slice(argv);
        serve_of(&full).bridge_engine_url()
    };
    assert_eq!(url(&[]), "http://127.0.0.1:8002/v1/chat/completions");
    assert_eq!(
        url(&["--port", "9000"]),
        "http://127.0.0.1:9000/v1/chat/completions"
    );
    assert_eq!(
        url(&["--host", "localhost", "--port", "9000"]),
        "http://localhost:9000/v1/chat/completions"
    );
    assert_eq!(
        url(&["--host", "0.0.0.0", "--port", "9000"]),
        "http://127.0.0.1:9000/v1/chat/completions"
    );
    assert_eq!(
        url(&["--host", "::", "--port", "9000"]),
        "http://[::1]:9000/v1/chat/completions"
    );
    assert_eq!(
        url(&["--host", "::1"]),
        "http://[::1]:8002/v1/chat/completions"
    );
    assert_eq!(
        url(&["--bind", "127.0.0.1:8123", "--port", "9000"]),
        "http://127.0.0.1:8123/v1/chat/completions"
    );
    assert_eq!(
        url(&["--bind", "[::]:8123"]),
        "http://[::1]:8123/v1/chat/completions"
    );
}

#[test]
fn the_engine_port_follows_bind_over_port() {
    assert_eq!(serve_of(&["serve", "--model", "m"]).engine_port(), 8002);
    assert_eq!(
        serve_of(&["serve", "--model", "m", "--bind", "0.0.0.0:8123"]).engine_port(),
        8123
    );
}
