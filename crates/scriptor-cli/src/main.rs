//! The `scriptor` command-line interface.

mod args;
mod bridge;
mod convert;
#[cfg(feature = "cuda")]
mod device_step;
#[cfg(all(test, feature = "cuda"))]
mod device_step_tests;
#[cfg(feature = "cuda")]
mod doctor;
#[cfg(feature = "cuda")]
mod drain;
#[cfg(feature = "cuda")]
mod embedded_bridge;
mod exit;
#[cfg(feature = "cuda")]
mod gpu_executor;
#[cfg(all(test, feature = "cuda"))]
mod image_feed_tests;
mod inspect;
#[cfg(feature = "cuda")]
mod seat_engine;
#[cfg(feature = "cuda")]
mod serve;
mod setup;
#[cfg(feature = "cuda")]
mod slot_book;
#[cfg(all(test, feature = "cuda"))]
mod slot_book_tests;
#[cfg(feature = "cuda")]
mod state_scratch;
#[cfg(feature = "cuda")]
mod table_mirror;
#[cfg(all(test, feature = "cuda"))]
mod table_mirror_tests;

use args::{Cli, Command};

use clap::Parser;

#[cfg(feature = "cuda")]
fn serve(a: &args::ServeArgs) -> i32 {
    serve::run(a)
}

#[cfg(not(feature = "cuda"))]
fn serve(_: &args::ServeArgs) -> i32 {
    eprintln!("scriptor serve: this build has no CUDA support");
    exit::REFUSED
}

#[cfg(feature = "cuda")]
fn doctor(a: &args::DoctorArgs) -> i32 {
    doctor::run(a)
}

#[cfg(not(feature = "cuda"))]
fn doctor(_: &args::DoctorArgs) -> i32 {
    eprintln!("scriptor doctor: this build has no CUDA support");
    exit::REFUSED
}

fn main() {
    let cli = Cli::parse();
    let code = match cli.command {
        Command::Serve(a) => serve(&a),
        Command::Bridge(a) => bridge::run_standalone(a),
        Command::Convert(a) => convert::run(&a),
        Command::Inspect(a) => inspect::run(&a),
        Command::Doctor(a) => doctor(&a),
        Command::Bench | Command::Tune => {
            eprintln!("scriptor: this subcommand is not implemented");
            2
        }
    };
    std::process::exit(code);
}
