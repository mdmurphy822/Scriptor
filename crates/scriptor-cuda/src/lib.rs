//! CUDA device, stream, memory pool and graph wrappers.
//!
//! Device discovery and the arena bookkeeping are available on every build;
//! the driver-backed wrappers (`device`, `graph`, `module`) are compiled with
//! the `cuda` feature.

pub mod detect;
pub mod error;
pub mod fault;
pub mod pool;
pub mod sm_split;

#[cfg(feature = "cuda")]
pub mod device;
#[cfg(feature = "cuda")]
pub mod graph;
#[cfg(feature = "cuda")]
pub mod green;
#[cfg(feature = "cuda")]
pub mod module;

pub use detect::{Detection, DeviceInfo, detect, select_ordinal};
pub use error::{CudaError, Result, fatal_code_name, is_fatal_code};
pub use pool::{Arena, ArenaRange};
pub use sm_split::{
    REQUIRE_SPLIT_ENV, RequireSplitError, SmAssignment, SmAttempt, SmGrant, SmGranularity, SmSplit,
    SmSplitError, required_split,
};

#[cfg(feature = "cuda")]
pub use device::{Device, DeviceArena};
#[cfg(feature = "cuda")]
pub use graph::{CapturedGraph, KernelArgs};
#[cfg(feature = "cuda")]
pub use green::{DeviceSms, GreenPartitions};
#[cfg(feature = "cuda")]
pub use module::{Kernel, KernelModule};
