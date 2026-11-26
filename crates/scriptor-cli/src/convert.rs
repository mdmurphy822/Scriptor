//! `scriptor convert`: the only writer of engine files.

use crate::args::ConvertArgs;

/// Converts the checkpoint and prints a one-line summary. Returns the process
/// exit code.
pub fn run(args: &ConvertArgs) -> i32 {
    match scriptor_weights::convert(&args.model_dir, &args.out) {
        Ok(r) => {
            println!(
                "converted revision {} -> {}: {} tensors, {} source bytes, {} engine bytes ({:.4}x)",
                r.revision,
                r.out_path.display(),
                r.tensors,
                r.source_bytes,
                r.out_bytes,
                r.out_bytes as f64 / r.source_bytes.max(1) as f64,
            );
            0
        }
        Err(e) => {
            eprintln!("scriptor convert: {e}");
            1
        }
    }
}
