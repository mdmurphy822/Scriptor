//! f32 CPU references that define the contract of each device kernel.
//!
//! All tensors are dense row-major `&[f32]` slices with the shape named in each
//! function's documentation.

pub mod activation;
pub mod attention;
pub mod conv;
pub mod gemm;
pub mod norm;
pub mod preprocess;
pub mod rope;
pub mod sampling;
pub mod scatter;
