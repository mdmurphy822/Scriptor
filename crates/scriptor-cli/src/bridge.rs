//! `scriptor bridge`: the document OCR bridge against an engine running elsewhere.

use scriptor_bridge::cli::OcrBridgeArgs;

use crate::exit;

/// Runs the bridge of `args` against an engine elsewhere until the process is stopped; returns
/// the exit code.
pub fn run_standalone(args: OcrBridgeArgs) -> i32 {
    match scriptor_bridge::cli::run(args) {
        Ok(()) => exit::CLEAN,
        Err(e) => {
            eprintln!("scriptor bridge: {e:#}");
            exit::REFUSED
        }
    }
}
