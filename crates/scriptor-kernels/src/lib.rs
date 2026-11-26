//! CUDA kernels, launch wrappers and CPU reference implementations.
//!
//! The `reference` module holds readable f32 implementations that define the
//! contract of every device kernel; `launch` holds the launch-geometry helpers
//! shared by the wrappers.

#[cfg(feature = "cuda")]
pub mod activation;
pub mod attention;
pub mod cublaslt;
#[cfg(feature = "cuda")]
pub mod fused;
pub mod heuristics;
pub mod kv;
pub mod launch;
pub mod logits_fixture;
#[cfg(feature = "cuda")]
pub mod norm;
#[cfg(feature = "nvjpeg")]
pub mod nvjpeg_ffi;
pub mod preprocess;
pub mod reference;
#[cfg(feature = "cuda")]
pub mod rope;
pub mod rope_kv;
pub mod sampling;
#[cfg(feature = "cuda")]
pub mod scatter;
#[cfg(feature = "cuda")]
pub mod skinny_gemm;
#[cfg(feature = "cuda")]
pub mod step;
pub mod verify;
#[cfg(feature = "cuda")]
pub mod vision;

/// Embedded sm_121 fatbins, one per `kernels/*.cu` source, keyed by file stem.
#[cfg(scriptor_cuda_enabled)]
pub mod fatbins {
    include!(concat!(env!("OUT_DIR"), "/fatbins.rs"));
}
