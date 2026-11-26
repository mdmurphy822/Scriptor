//! The document OCR bridge `scriptor serve` runs in the same process as the engine.

use std::sync::mpsc;
use std::time::Duration;

use scriptor_bridge::cli::EngineLink;
use scriptor_bridge::serve::{OcrServer, ServeConfig, ShutdownHandle};

use crate::args::ServeArgs;

/// The settings of the bridge that runs beside the engine, validated before the engine boots;
/// `None` under `--no-bridge`.
///
/// # Errors
///
/// Returns the invalid option or setting, or a bridge port equal to the engine's.
pub fn config(args: &ServeArgs) -> Result<Option<ServeConfig>, String> {
    let Some(bridge) = args.bridge_args() else {
        return Ok(None);
    };
    let config = scriptor_bridge::cli::serve_config(&bridge, &|name| std::env::var(name).ok())
        .map_err(|e| format!("{e:#}"))?;
    if config.port == args.engine_port() {
        return Err(format!(
            "--bridge-port {} is the engine's own port",
            config.port
        ));
    }
    Ok(Some(config))
}

/// The engine the bridge sends regions to: the serve address over loopback, the served
/// model name and the configured sequence count (the derived count fills it at boot when unset).
pub fn link(args: &ServeArgs) -> EngineLink {
    EngineLink {
        url: args.bridge_engine_url(),
        model: args.served_model_name.clone(),
        max_num_seqs: args.max_num_seqs,
    }
}

/// A bridge serving in its own thread.
pub struct RunningBridge {
    stop: ShutdownHandle,
    done: mpsc::Receiver<()>,
}

impl RunningBridge {
    /// Binds the bridge of `config` for the engine named by `link` and serves it in a thread.
    /// A missing layout model is not an error: the bridge then answers every document as
    /// unavailable.
    ///
    /// # Errors
    ///
    /// Returns the bind failure or the thread that could not start.
    pub fn start(config: ServeConfig, link: &EngineLink) -> Result<Self, String> {
        let server: OcrServer =
            scriptor_bridge::cli::bind_for_engine(config, link, &|name| std::env::var(name).ok())
                .map_err(|e| format!("{e:#}"))?;
        scriptor_bridge::cli::announce_ready(&server);
        let stop = server.shutdown_handle();
        let (done_tx, done) = mpsc::channel();
        std::thread::Builder::new()
            .name("scriptor-bridge".into())
            .spawn(move || {
                server.serve();
                let _ = done_tx.send(());
            })
            .map_err(|e| format!("cannot start the bridge thread: {e}"))?;
        Ok(Self { stop, done })
    }

    /// Stops accepting documents; those in flight run on.
    pub fn stop(&self) {
        self.stop.stop();
    }

    /// Waits up to `grace` for the in-flight documents to finish; whether they did.
    pub fn wait(self, grace: Duration) -> bool {
        self.done.recv_timeout(grace).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::Cli;
    use clap::Parser;

    fn serve_args(extra: &[&str]) -> ServeArgs {
        let mut argv = vec!["scriptor", "serve", "--model", "m"];
        argv.extend_from_slice(extra);
        match Cli::try_parse_from(argv).unwrap().command {
            crate::args::Command::Serve(serve) => *serve,
            _ => panic!("expected serve"),
        }
    }

    #[test]
    fn the_link_names_the_serve_address_model_and_sequences() {
        let link = link(&serve_args(&[
            "--port",
            "9000",
            "--served-model-name",
            "x",
            "--max-num-seqs",
            "16",
        ]));
        assert_eq!(link.url, "http://127.0.0.1:9000/v1/chat/completions");
        assert_eq!(link.model, "x");
        assert_eq!(link.max_num_seqs, Some(16));
    }

    #[test]
    fn no_bridge_has_no_config() {
        assert!(config(&serve_args(&["--no-bridge"])).unwrap().is_none());
    }

    #[test]
    fn the_bridge_cannot_share_the_engine_port() {
        let error = config(&serve_args(&["--port", "8765"])).err().unwrap();
        assert!(error.contains("8765"), "{error}");
    }

    #[test]
    fn a_public_bridge_bind_needs_tls() {
        let error = config(&serve_args(&["--bridge-listen", "0.0.0.0"]))
            .err()
            .unwrap();
        assert!(error.contains("--tls-cert"), "{error}");
    }
}
