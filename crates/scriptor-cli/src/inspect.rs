//! `scriptor inspect`: summarises a checkpoint directory or an engine file.

use crate::args::InspectArgs;
use crate::setup::Setup;

/// Prints a summary of the path and returns the process exit code.
pub fn run(args: &InspectArgs) -> i32 {
    if args.path.is_dir() {
        match Setup::load(&args.path) {
            Ok(s) => {
                let t = &s.spec.text;
                println!("checkpoint");
                println!("  tensors        {}", s.dir.tensors().len());
                println!("  weight bytes   {}", s.weight_bytes);
                println!(
                    "  text           {} layers, hidden {}, heads {}/{}, vocab {}",
                    t.num_layers, t.hidden_size, t.num_heads, t.num_kv_heads, t.vocab_size
                );
                println!("  mtp layers     {}", t.num_mtp_layers);
                println!(
                    "  vision         {} blocks, hidden {}",
                    s.spec.vision.depth, s.spec.vision.hidden_size
                );
                println!("  max pixels     {}", s.image.max_pixels);
                0
            }
            Err(e) => {
                eprintln!("scriptor inspect: {e}");
                1
            }
        }
    } else {
        match scriptor_weights::EngineFile::open(&args.path) {
            Ok(f) => {
                let h = f.header();
                println!("engine file");
                println!("  layout version {}", h.layout_version);
                println!("  revision       {}", h.revision);
                println!("  text layers    {}", h.text_layers);
                println!("  vision blocks  {}", h.vision_blocks);
                println!("  tensors        {}", h.tensors.len());
                println!("  file bytes     {}", h.file_len);
                0
            }
            Err(e) => {
                eprintln!("scriptor inspect: {e}");
                1
            }
        }
    }
}
