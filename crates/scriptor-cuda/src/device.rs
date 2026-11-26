//! Device handle, stream creation and device-resident arena.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use crate::detect::{DeviceInfo, detect};
use crate::error::{CudaError, Result};
use crate::pool::{Arena, ArenaRange};

/// An opened CUDA device with its primary context.
pub struct Device {
    ctx: Arc<CudaContext>,
    info: DeviceInfo,
}

impl Device {
    /// Opens the device selected by `SCRIPTOR_CUDA_DEVICE` (default 0).
    pub fn open() -> Result<Self> {
        let detection = detect()?;
        Self::open_ordinal(detection.selected)
    }

    /// Opens the device with the given ordinal.
    pub fn open_ordinal(ordinal: usize) -> Result<Self> {
        let detection = detect()?;
        let info = detection
            .devices
            .into_iter()
            .find(|d| d.ordinal == ordinal)
            .ok_or_else(|| CudaError::InvalidDevice(format!("no device with ordinal {ordinal}")))?;
        let ctx = CudaContext::new(ordinal)?;
        let device = Self { ctx, info };
        device.set_pool_release_threshold(device.info.total_bytes)?;
        Ok(device)
    }

    /// Sets how many bytes the device's stream-ordered allocator keeps cached across
    /// synchronisation points instead of returning them to the operating system.
    pub fn set_pool_release_threshold(&self, bytes: usize) -> Result<()> {
        use cudarc::driver::sys;
        self.ctx.bind_to_thread()?;
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        let mut value = bytes as u64;
        // SAFETY: `pool` and `value` outlive the calls, and the attribute takes a `cuuint64_t`.
        unsafe {
            sys::cuDeviceGetDefaultMemPool(&mut pool, self.ctx.cu_device()).result()?;
            sys::cuMemPoolSetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut value as *mut u64).cast(),
            )
            .result()?;
        }
        Ok(())
    }

    /// Returns unused pooled memory to the operating system, keeping at least `keep_bytes`
    /// reserved. Pending stream-ordered frees must have completed.
    pub fn trim_pool(&self, keep_bytes: usize) -> Result<()> {
        trim_context_pool(&self.ctx, keep_bytes)
    }

    /// The release threshold currently set on the device's stream-ordered allocator.
    pub fn pool_release_threshold(&self) -> Result<usize> {
        self.pool_attribute(
            cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
        )
    }

    /// Bytes the device's stream-ordered allocator currently holds from the driver.
    pub fn pool_reserved_bytes(&self) -> Result<usize> {
        self.pool_attribute(
            cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
        )
    }

    fn pool_attribute(&self, attribute: cudarc::driver::sys::CUmemPool_attribute) -> Result<usize> {
        use cudarc::driver::sys;
        self.ctx.bind_to_thread()?;
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        let mut value = 0u64;
        // SAFETY: `pool` and `value` outlive the calls, and pool attributes yield a `cuuint64_t`.
        unsafe {
            sys::cuDeviceGetDefaultMemPool(&mut pool, self.ctx.cu_device()).result()?;
            sys::cuMemPoolGetAttribute(pool, attribute, (&mut value as *mut u64).cast())
                .result()?;
        }
        Ok(value as usize)
    }

    /// Properties captured when the device was opened.
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// The underlying driver context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Creates a new non-blocking stream on this device.
    pub fn new_stream(&self) -> Result<Arc<CudaStream>> {
        Ok(self.ctx.new_stream()?)
    }

    /// Blocks until all work on the device completes.
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.ctx.synchronize()?)
    }

    /// Allocates a device arena of exactly `capacity` bytes.
    pub fn arena(&self, capacity: usize, stream: &Arc<CudaStream>) -> Result<DeviceArena> {
        DeviceArena::new(stream, capacity)
    }
}

/// Trims the stream-ordered allocator of `ctx`'s device to `keep_bytes` reserved.
pub fn trim_context_pool(ctx: &Arc<CudaContext>, keep_bytes: usize) -> Result<()> {
    use cudarc::driver::sys;
    ctx.bind_to_thread()?;
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    // SAFETY: `pool` outlives the calls and the trim takes a pool handle and a byte count.
    unsafe {
        sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()).result()?;
        sys::cuMemPoolTrimTo(pool, keep_bytes).result()?;
    }
    Ok(())
}

/// A single device allocation carved up by an [`Arena`].
pub struct DeviceArena {
    buffer: CudaSlice<u8>,
    arena: Arena,
}

impl DeviceArena {
    /// Allocates `capacity` zeroed bytes on the stream's device.
    pub fn new(stream: &Arc<CudaStream>, capacity: usize) -> Result<Self> {
        crate::fault::check_alloc("arena")?;
        Ok(Self {
            buffer: stream.alloc_zeros::<u8>(capacity)?,
            arena: Arena::new(capacity),
        })
    }

    /// Reserves an aligned range within the arena.
    pub fn alloc(&mut self, len: usize, align: usize) -> Result<ArenaRange> {
        self.arena.alloc(len, align)
    }

    /// Releases all ranges for reuse.
    pub fn reset(&mut self) {
        self.arena.reset();
    }

    /// Offset bookkeeping for this arena.
    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    /// The backing device buffer.
    pub fn buffer(&self) -> &CudaSlice<u8> {
        &self.buffer
    }

    /// The backing device buffer, mutably.
    pub fn buffer_mut(&mut self) -> &mut CudaSlice<u8> {
        &mut self.buffer
    }
}
