//! The native OCR bridge HTTP server: the options of `scriptor bridge`, and
//! the building blocks `scriptor serve` uses to run the bridge beside the
//! engine.

use crate::document::{
    DocumentConfig, MODEL_ENV, NativeDocumentOcr, SharedLayout, URL_ENV, with_legacy_fallback,
};
use crate::serve::{DocumentOcr, MAX_NUM_SEQS_ENV, OcrFailure, OcrServer, OcrTask, ServeConfig};
use anyhow::{Result, bail};
use clap::Args;
use scriptor_layout::OnnxLayout;
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The options of the bridge server.
#[derive(Args, Debug)]
pub struct OcrBridgeArgs {
    /// Serve the OCR contract persistently (loopback unless `--listen` says otherwise).
    #[arg(long)]
    pub serve: bool,
    /// HTTP port (requires --serve); 8765 when absent.
    #[arg(long)]
    pub port: Option<u16>,
    /// Bind address (default loopback; any other address needs a bearer token and TLS).
    #[arg(long)]
    pub listen: Option<String>,
    /// TLS certificate chain (PEM).
    #[arg(long)]
    pub tls_cert: Option<PathBuf>,
    /// TLS private key (PEM).
    #[arg(long)]
    pub tls_key: Option<PathBuf>,
    /// Environment variable holding the bearer token (default
    /// SCRIPTOR_BRIDGE_TOKEN, then SEMANTIK_BRIDGE_TOKEN, for a non-loopback bind).
    #[arg(long)]
    pub token_env: Option<String>,
}

/// The conversion served when the native pipeline cannot be built: every
/// document is answered as unavailable with the diagnostics naming why.
struct FailClosed {
    diagnostics: Vec<(String, u64)>,
}

impl DocumentOcr for FailClosed {
    fn ocr(
        &self,
        _source: &[u8],
        _task: OcrTask,
        _progress: &(dyn Fn(usize, usize) + Sync),
        _cancel: &(dyn Fn() -> bool + Sync),
        _diagnostic: &(dyn Fn(&str, u64) + Sync),
    ) -> Result<Value, OcrFailure> {
        Err(OcrFailure::Unavailable {
            diagnostics: self.diagnostics.clone(),
        })
    }
}

/// The layout model of the given runtime and weights files.
fn load_layout(
    runtime: &Path,
    weights: &Path,
    device: Option<&str>,
    batch_size: usize,
) -> Result<SharedLayout, (String, u64)> {
    match OnnxLayout::load(runtime, weights, device, batch_size) {
        Ok(layout) => {
            let _ = writeln!(
                std::io::stderr(),
                "scriptor-bridge: layout model on {}",
                layout.device()
            );
            Ok(Arc::new(layout))
        }
        Err(error) => {
            let _ = writeln!(std::io::stderr(), "scriptor-bridge: {error}");
            Err(("layout_backend_unavailable".to_string(), 1))
        }
    }
}

fn build_pipeline(
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<Arc<dyn DocumentOcr>, (String, u64)> {
    let config = DocumentConfig::from_env(env).map_err(|error| error.diagnostic())?;
    let (runtime, weights) = config.layout_files().map_err(|error| error.diagnostic())?;
    let layout = load_layout(
        runtime,
        weights,
        config.layout_device.as_deref(),
        config.layout_batch_size,
    )?;
    Ok(Arc::new(NativeDocumentOcr::new(config, layout)))
}

/// The conversion behind the server: the native pipeline when its settings
/// and layout model are present, else one that answers every document as
/// unavailable and names the missing setting in its diagnostics (and in one
/// startup line).
fn native_pipeline(env: &dyn Fn(&str) -> Option<String>) -> Arc<dyn DocumentOcr> {
    match build_pipeline(&with_legacy_fallback(env)) {
        Ok(pipeline) => pipeline,
        Err(diagnostic) => {
            let _ = writeln!(
                std::io::stderr(),
                "scriptor-bridge: unavailable ({}); every document will be answered as unavailable",
                diagnostic.0
            );
            Arc::new(FailClosed {
                diagnostics: vec![diagnostic],
            })
        }
    }
}

/// The validated server settings of `args` and `env`, where each
/// `SCRIPTOR_` setting falls back to its legacy `SEMANTIK_` name.
///
/// # Errors
///
/// Returns the first invalid or missing option or setting.
pub fn serve_config(
    args: &OcrBridgeArgs,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<ServeConfig> {
    if !args.serve {
        let network_option = args.port.is_some()
            || args.listen.is_some()
            || args.tls_cert.is_some()
            || args.tls_key.is_some()
            || args.token_env.is_some();
        if network_option {
            bail!("--port and network options require --serve");
        }
        bail!("the native OCR bridge runs only with --serve");
    }
    ServeConfig::from_options(
        args.port,
        args.listen.as_deref(),
        args.tls_cert.as_deref(),
        args.tls_key.as_deref(),
        args.token_env.as_deref(),
        &with_legacy_fallback(env),
    )
}

/// The engine a bridge started beside it recognises through.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineLink {
    /// The engine's chat-completions URL.
    pub url: String,
    /// The served model name sent in every request.
    pub model: String,
    /// The engine's concurrent sequences, when fixed; the bridge sizes its
    /// workers and concurrent documents from it unless a setting says otherwise.
    pub max_num_seqs: Option<usize>,
}

/// `env` with the engine's address, model and sequence count laid over it:
/// the link names the engine, so no setting can point the bridge elsewhere.
fn linked_env<'a>(
    link: &'a EngineLink,
    env: &'a dyn Fn(&str) -> Option<String>,
) -> impl Fn(&str) -> Option<String> + 'a {
    let resolved = with_legacy_fallback(env);
    move |name: &str| match name {
        URL_ENV => Some(link.url.clone()),
        MODEL_ENV => Some(link.model.clone()),
        MAX_NUM_SEQS_ENV => resolved(name).or_else(|| link.max_num_seqs.map(|n| n.to_string())),
        _ => resolved(name),
    }
}

/// Binds the server of `config` over the native pipeline built from `env`.
///
/// # Errors
///
/// Returns the bind failure; a missing layout model is not one.
pub fn bind(config: ServeConfig, env: &dyn Fn(&str) -> Option<String>) -> Result<OcrServer> {
    OcrServer::bind(config, native_pipeline(env))
}

/// Binds the server of `config` over the native pipeline for the engine
/// named by `link`.
///
/// # Errors
///
/// Returns the bind failure; a missing layout model is not one.
pub fn bind_for_engine(
    config: ServeConfig,
    link: &EngineLink,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<OcrServer> {
    OcrServer::bind(config, native_pipeline(&linked_env(link, env)))
}

/// Writes the line that says `server` is accepting connections.
pub fn announce_ready(server: &OcrServer) {
    let _ = writeln!(
        std::io::stderr(),
        "scriptor-bridge: ready on {}",
        server.local_addr()
    );
}

/// Runs the server until the process is stopped.
///
/// # Errors
///
/// Returns an invalid option or setting, or the bind failure.
pub fn run(args: OcrBridgeArgs) -> Result<()> {
    let env = |name: &str| std::env::var(name).ok();
    let config = serve_config(&args, &env)?;
    let server = bind(config, &env)?;
    announce_ready(&server);
    server.serve();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct Cli {
        #[command(flatten)]
        args: OcrBridgeArgs,
    }

    fn parse(arguments: &[&str]) -> Result<OcrBridgeArgs, clap::Error> {
        let mut argv = vec!["bridge"];
        argv.extend_from_slice(arguments);
        Ok(Cli::try_parse_from(argv)?.args)
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn the_flags_of_the_python_bridge_parse() {
        let args = parse(&[
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
        .unwrap();
        assert!(args.serve);
        assert_eq!(args.port, Some(18002));
        assert_eq!(args.listen.as_deref(), Some("0.0.0.0"));
        assert_eq!(args.tls_cert, Some(PathBuf::from("cert.pem")));
        assert_eq!(args.tls_key, Some(PathBuf::from("key.pem")));
        assert_eq!(args.token_env.as_deref(), Some("BRIDGE_TOKEN"));
        assert!(parse(&["--port", "70000"]).is_err());
        assert!(parse(&["--unknown"]).is_err());
    }

    #[test]
    fn serving_defaults_to_loopback_on_the_default_port() {
        let config = serve_config(&parse(&["--serve"]).unwrap(), &no_env).unwrap();
        assert!(config.address.is_loopback());
        assert_eq!(config.port, 8765);
        let config =
            serve_config(&parse(&["--serve", "--port", "18002"]).unwrap(), &no_env).unwrap();
        assert_eq!(config.port, 18002);
    }

    #[test]
    fn network_options_require_serve_and_a_public_bind_requires_tls() {
        let message = |arguments: &[&str]| {
            serve_config(&parse(arguments).unwrap(), &no_env)
                .err()
                .unwrap()
                .to_string()
        };
        assert_eq!(
            message(&["--port", "9000"]),
            "--port and network options require --serve"
        );
        assert_eq!(message(&[]), "the native OCR bridge runs only with --serve");
        assert_eq!(
            message(&["--serve", "--listen", "0.0.0.0"]),
            "a non-loopback bind address requires --tls-cert and --tls-key"
        );
    }

    #[test]
    fn missing_settings_fail_closed_naming_the_setting() {
        let outcome = native_pipeline(&no_env).ocr(
            b"%PDF-",
            OcrTask::Layout,
            &|_, _| {},
            &|| false,
            &|_, _| {},
        );
        assert_eq!(
            outcome,
            Err(OcrFailure::Unavailable {
                diagnostics: vec![("setting_scriptor_bridge_engine_url".to_string(), 1)]
            })
        );
        let env = |name: &str| match name {
            "SEMANTIK_GLM_OCR_URL" => Some("http://127.0.0.1:8000/v1/chat/completions".into()),
            "SEMANTIK_GLM_OCR_MODEL" => Some("glm-ocr".into()),
            _ => None,
        };
        let outcome =
            native_pipeline(&env).ocr(b"%PDF-", OcrTask::Text, &|_, _| {}, &|| false, &|_, _| {});
        assert_eq!(
            outcome,
            Err(OcrFailure::Unavailable {
                diagnostics: vec![("setting_scriptor_bridge_layout_runtime".to_string(), 1)]
            })
        );
    }

    #[test]
    fn legacy_settings_still_configure_the_pipeline() {
        let env = |name: &str| match name {
            "SEMANTIK_GLM_OCR_URL" => Some("http://127.0.0.1:8000/v1/chat/completions".into()),
            "SEMANTIK_GLM_OCR_MODEL" => Some("glm-ocr".into()),
            _ => None,
        };
        let outcome =
            native_pipeline(&env).ocr(b"%PDF-", OcrTask::Text, &|_, _| {}, &|| false, &|_, _| {});
        assert_eq!(
            outcome,
            Err(OcrFailure::Unavailable {
                diagnostics: vec![("setting_scriptor_bridge_layout_runtime".to_string(), 1)]
            })
        );
    }

    #[test]
    fn the_bearer_token_follows_scriptor_then_legacy_precedence() {
        let args = parse(&["--serve", "--token-env", "SCRIPTOR_BRIDGE_TOKEN"]).unwrap();
        let token =
            |env: &dyn Fn(&str) -> Option<String>| serve_config(&args, env).unwrap().token.unwrap();
        let legacy_only = |name: &str| (name == "SEMANTIK_BRIDGE_TOKEN").then(|| "old".to_string());
        assert_eq!(token(&legacy_only), "old");
        let both = |name: &str| match name {
            "SCRIPTOR_BRIDGE_TOKEN" => Some("new".to_string()),
            "SEMANTIK_BRIDGE_TOKEN" => Some("old".to_string()),
            _ => None,
        };
        assert_eq!(token(&both), "new");
    }

    #[test]
    fn the_engine_link_overrides_the_endpoint_and_model() {
        let link = EngineLink {
            url: "http://127.0.0.1:8002/v1/chat/completions".into(),
            model: "glm-ocr".into(),
            max_num_seqs: Some(16),
        };
        let raw = |name: &str| match name {
            "SEMANTIK_GLM_OCR_URL" => Some("http://127.0.0.1:1/v1/chat/completions".into()),
            "SCRIPTOR_BRIDGE_MODEL" => Some("other".into()),
            _ => None,
        };
        let env = linked_env(&link, &raw);
        assert_eq!(env(URL_ENV).as_deref(), Some(link.url.as_str()));
        assert_eq!(env(MODEL_ENV).as_deref(), Some("glm-ocr"));
        assert_eq!(env(MAX_NUM_SEQS_ENV).as_deref(), Some("16"));
    }

    #[test]
    fn a_configured_sequence_count_wins_over_the_engines() {
        let link = EngineLink {
            url: "http://127.0.0.1:8002/v1/chat/completions".into(),
            model: "glm-ocr".into(),
            max_num_seqs: Some(16),
        };
        let raw = |name: &str| (name == "SEMANTIK_GLM_OCR_MAX_NUM_SEQS").then(|| "8".to_string());
        assert_eq!(
            linked_env(&link, &raw)(MAX_NUM_SEQS_ENV).as_deref(),
            Some("8")
        );
        let none = |_: &str| None;
        let derived = EngineLink {
            max_num_seqs: None,
            ..link
        };
        assert_eq!(linked_env(&derived, &none)(MAX_NUM_SEQS_ENV), None);
    }
}
