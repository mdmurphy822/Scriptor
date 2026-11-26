//! Device detection never panics and degrades to `DeviceUnavailable`.

use scriptor_cuda::{CudaError, detect};

#[test]
fn detect_returns_cleanly_without_a_gpu() {
    if std::env::var("SCRIPTOR_GPU_TESTS").as_deref() == Ok("1") {
        let d = detect().expect("gpu expected");
        assert!(!d.devices.is_empty());
        assert!(d.selected_device().total_bytes > 0);
        return;
    }
    match detect() {
        Ok(d) => assert!(d.selected < d.devices.len()),
        Err(CudaError::DeviceUnavailable(_) | CudaError::InvalidDevice(_)) => {}
        Err(other) => panic!("unexpected error: {other}"),
    }
}
