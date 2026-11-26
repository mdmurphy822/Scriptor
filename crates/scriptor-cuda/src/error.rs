//! Error type shared by every `scriptor-cuda` operation.

use std::fmt;

/// Failure raised by device discovery, allocation, graph or module operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaError {
    /// No usable CUDA driver or device is present (or the build has no CUDA support).
    DeviceUnavailable(String),
    /// `SCRIPTOR_CUDA_DEVICE` or an explicit ordinal does not name an enumerated device.
    InvalidDevice(String),
    /// A CUDA driver call failed without a driver result code.
    Driver(String),
    /// A CUDA driver call returned the `CUresult` `code`; `message` is the driver's description.
    Cuda { code: i32, message: String },
    /// A runtime allocation at `site` asks for more than the boot plan reserved for it.
    PlanExceeded {
        site: String,
        requested: usize,
        planned: usize,
    },
    /// An arena allocation does not fit in the remaining capacity.
    PoolExhausted {
        requested: usize,
        available: usize,
        capacity: usize,
    },
    /// Graph capture, update or launch failed.
    Graph(String),
    /// Module or kernel loading failed.
    Module(String),
}

/// `CUresult` of an allocation that found no memory.
pub const CUDA_ERROR_OUT_OF_MEMORY: i32 = 2;

/// Whether the `CUresult` `code` leaves the context unusable: the process must be relaunched
/// because the fault is sticky and no further CUDA call can be trusted.
pub fn is_fatal_code(code: i32) -> bool {
    matches!(
        code,
        214 | 700 | 702 | 709 | 710 | 714 | 715 | 716 | 717 | 718 | 719 | 999
    )
}

/// The driver's name for the fatal `CUresult` codes, `None` for any other code.
pub fn fatal_code_name(code: i32) -> Option<&'static str> {
    Some(match code {
        214 => "CUDA_ERROR_ECC_UNCORRECTABLE",
        700 => "CUDA_ERROR_ILLEGAL_ADDRESS",
        702 => "CUDA_ERROR_LAUNCH_TIMEOUT",
        709 => "CUDA_ERROR_CONTEXT_IS_DESTROYED",
        710 => "CUDA_ERROR_ASSERT",
        714 => "CUDA_ERROR_HARDWARE_STACK_ERROR",
        715 => "CUDA_ERROR_ILLEGAL_INSTRUCTION",
        716 => "CUDA_ERROR_MISALIGNED_ADDRESS",
        717 => "CUDA_ERROR_INVALID_ADDRESS_SPACE",
        718 => "CUDA_ERROR_INVALID_PC",
        719 => "CUDA_ERROR_LAUNCH_FAILED",
        999 => "CUDA_ERROR_UNKNOWN",
        _ => return None,
    })
}

impl CudaError {
    /// The driver result code, when the failure carries one.
    pub fn code(&self) -> Option<i32> {
        match self {
            Self::Cuda { code, .. } => Some(*code),
            _ => None,
        }
    }

    /// Whether the failure leaves the CUDA context unusable (see [`is_fatal_code`]).
    pub fn is_fatal(&self) -> bool {
        self.code().is_some_and(is_fatal_code)
    }

    /// Whether the failure is a request-scoped lack of memory that leaves the context usable.
    pub fn is_out_of_memory(&self) -> bool {
        matches!(self, Self::PlanExceeded { .. } | Self::PoolExhausted { .. })
            || self.code() == Some(CUDA_ERROR_OUT_OF_MEMORY)
    }
}

/// Result alias for `scriptor-cuda` operations.
pub type Result<T> = std::result::Result<T, CudaError>;

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceUnavailable(m) => write!(f, "cuda device unavailable: {m}"),
            Self::InvalidDevice(m) => write!(f, "invalid cuda device: {m}"),
            Self::Driver(m) => write!(f, "cuda driver error: {m}"),
            Self::PoolExhausted {
                requested,
                available,
                capacity,
            } => write!(
                f,
                "arena exhausted: requested {requested} bytes, {available} of {capacity} available"
            ),
            Self::Cuda { code, message } => {
                write!(f, "cuda driver error {code}: {message}")
            }
            Self::PlanExceeded {
                site,
                requested,
                planned,
            } => write!(
                f,
                "allocation at {site} needs {requested} bytes but the plan reserved {planned}"
            ),
            Self::Graph(m) => write!(f, "cuda graph error: {m}"),
            Self::Module(m) => write!(f, "cuda module error: {m}"),
        }
    }
}

impl std::error::Error for CudaError {}

#[cfg(feature = "cuda")]
impl From<cudarc::driver::DriverError> for CudaError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self::Cuda {
            code: e.0 as i32,
            message: e.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fatal_codes_are_classified() {
        for code in [214, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719, 999] {
            let e = CudaError::Cuda {
                code,
                message: String::new(),
            };
            assert!(e.is_fatal(), "{code}");
            assert!(fatal_code_name(code).is_some());
        }
    }

    #[test]
    fn recoverable_errors_are_not_fatal() {
        let oom = CudaError::Cuda {
            code: CUDA_ERROR_OUT_OF_MEMORY,
            message: "out of memory".into(),
        };
        assert!(!oom.is_fatal());
        assert!(oom.is_out_of_memory());
        let invalid = CudaError::Cuda {
            code: 1,
            message: String::new(),
        };
        assert!(!invalid.is_fatal() && !invalid.is_out_of_memory());
        let plan = CudaError::PlanExceeded {
            site: "work".into(),
            requested: 8,
            planned: 4,
        };
        assert!(!plan.is_fatal() && plan.is_out_of_memory());
        assert!(!CudaError::Driver("x".into()).is_fatal());
        assert_eq!(plan.code(), None);
    }

    #[test]
    fn display_names_the_plan_site() {
        let plan = CudaError::PlanExceeded {
            site: "work".into(),
            requested: 8,
            planned: 4,
        };
        assert!(plan.to_string().contains("work"));
    }
}
