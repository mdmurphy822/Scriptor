//! CUDA device enumeration and `SCRIPTOR_CUDA_DEVICE` selection.

use crate::error::{CudaError, Result};

/// Environment variable naming the device ordinal to use.
pub const DEVICE_ENV: &str = "SCRIPTOR_CUDA_DEVICE";

/// Properties of one enumerated CUDA device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    /// Compute capability as `(major, minor)`.
    pub sm: (i32, i32),
    pub free_bytes: usize,
    pub total_bytes: usize,
    /// Whether the device shares memory with the host (`CU_DEVICE_ATTRIBUTE_INTEGRATED`).
    pub unified_memory: bool,
}

/// Enumerated devices and the ordinal selected for use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Detection {
    pub devices: Vec<DeviceInfo>,
    pub selected: usize,
}

impl Detection {
    /// The selected device's properties.
    pub fn selected_device(&self) -> &DeviceInfo {
        &self.devices[self.selected]
    }
}

/// Resolves the ordinal to use from the `SCRIPTOR_CUDA_DEVICE` value and the
/// device count; an unset (or blank) value selects device 0.
pub fn select_ordinal(env_value: Option<&str>, count: usize) -> Result<usize> {
    if count == 0 {
        return Err(CudaError::DeviceUnavailable("no cuda devices".into()));
    }
    let ordinal = match env_value.map(str::trim).filter(|v| !v.is_empty()) {
        None => 0,
        Some(v) => v.parse::<usize>().map_err(|_| {
            CudaError::InvalidDevice(format!("{DEVICE_ENV}={v} is not a device ordinal"))
        })?,
    };
    if ordinal >= count {
        return Err(CudaError::InvalidDevice(format!(
            "{DEVICE_ENV}={ordinal} but only {count} device(s) are present"
        )));
    }
    Ok(ordinal)
}

/// Enumerates CUDA devices and applies `SCRIPTOR_CUDA_DEVICE`. Returns
/// `DeviceUnavailable` when the driver or a device is absent, without panicking.
pub fn detect() -> Result<Detection> {
    detect_impl()
}

#[cfg(feature = "cuda")]
fn detect_impl() -> Result<Detection> {
    use cudarc::driver::CudaContext;

    let guarded = std::panic::catch_unwind(CudaContext::device_count);
    let count = match guarded {
        Ok(Ok(n)) => n.max(0) as usize,
        Ok(Err(e)) => return Err(CudaError::DeviceUnavailable(e.to_string())),
        Err(_) => {
            return Err(CudaError::DeviceUnavailable(
                "cuda driver library could not be loaded".into(),
            ));
        }
    };
    let env = std::env::var(DEVICE_ENV).ok();
    let selected = select_ordinal(env.as_deref(), count)?;
    let mut devices = Vec::with_capacity(count);
    for ordinal in 0..count {
        let ctx = CudaContext::new(ordinal)?;
        let (free_bytes, mut total_bytes) = ctx.mem_get_info()?;
        if let Some(fake) = crate::fault::fake_device_total() {
            total_bytes = fake as usize;
        }
        devices.push(DeviceInfo {
            ordinal,
            name: ctx.name()?,
            sm: ctx.compute_capability()?,
            free_bytes,
            total_bytes,
            unified_memory: ctx.attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED,
            )? != 0,
        });
    }
    Ok(Detection { devices, selected })
}

#[cfg(not(feature = "cuda"))]
fn detect_impl() -> Result<Detection> {
    Err(CudaError::DeviceUnavailable(
        "built without the `cuda` feature".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinal_selection() {
        assert_eq!(select_ordinal(None, 2).unwrap(), 0);
        assert_eq!(select_ordinal(Some(" "), 2).unwrap(), 0);
        assert_eq!(select_ordinal(Some("1"), 2).unwrap(), 1);
        assert!(matches!(
            select_ordinal(Some("2"), 2),
            Err(CudaError::InvalidDevice(_))
        ));
        assert!(matches!(
            select_ordinal(Some("x"), 2),
            Err(CudaError::InvalidDevice(_))
        ));
        assert!(matches!(
            select_ordinal(None, 0),
            Err(CudaError::DeviceUnavailable(_))
        ));
    }
}
