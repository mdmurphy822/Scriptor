//! Green-context SM partitions of one device: a vision partition and a decode partition, each
//! with a stream whose kernels (and, when captured from it, graph kernel nodes) run only on
//! that partition's SMs.

use std::sync::Arc;

use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaStream};

use crate::error::{CudaError, Result};
use crate::sm_split::{SmAttempt, SmGrant, SmGranularity, SmSplit, SmSplitError};

/// The SM count and partition granularity of a device's full SM resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceSms {
    pub sms: u32,
    pub granularity: SmGranularity,
}

fn zeroed_resource() -> sys::CUdevResource {
    // SAFETY: `CUdevResource` is a plain C struct; all-zero is its documented empty value.
    unsafe { std::mem::zeroed() }
}

fn sm_view(resource: &sys::CUdevResource) -> sys::CUdevSmResource {
    // SAFETY: every resource handled here was filled by the driver as an SM-type resource.
    unsafe { resource.__bindgen_anon_1.sm }
}

fn refusal(error: SmSplitError) -> CudaError {
    CudaError::Driver(error.to_string())
}

/// Reads the SM count and partition granularity the driver reports for `ctx`'s device.
pub fn device_sms(ctx: &Arc<CudaContext>) -> Result<DeviceSms> {
    ctx.bind_to_thread()?;
    let mut resource = zeroed_resource();
    // SAFETY: `resource` is a valid out-parameter for the SM resource type.
    unsafe {
        sys::cuDeviceGetDevResource(
            ctx.cu_device(),
            &mut resource,
            sys::CUdevResourceType::CU_DEV_RESOURCE_TYPE_SM,
        )
        .result()?;
    }
    let sm = sm_view(&resource);
    Ok(DeviceSms {
        sms: sm.smCount,
        granularity: SmGranularity {
            min_partition: sm.minSmPartitionSize,
            alignment: sm.smCoscheduledAlignment,
        },
    })
}

/// Checks `split` against `ctx`'s device without creating any context.
pub fn validate_split(ctx: &Arc<CudaContext>, split: &SmSplit) -> Result<()> {
    let device = device_sms(ctx)?;
    split.check_fits(device.sms).map_err(refusal)
}

/// What one split request produced.
struct Granted {
    groups: Vec<sys::CUdevResource>,
    remaining: sys::CUdevResource,
    grant: SmGrant,
}

fn run_attempt(
    full: &sys::CUdevResource,
    device_sms: u32,
    attempt: SmAttempt,
) -> std::result::Result<Granted, String> {
    let wanted = (device_sms / attempt.min_count).max(1);
    let mut groups: Vec<sys::CUdevResource> = (0..wanted).map(|_| zeroed_resource()).collect();
    let mut remaining = zeroed_resource();
    let mut count = wanted;
    let flags = if attempt.ignore_coscheduling {
        sys::CUdevSmResourceSplit_flags::CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING as u32
    } else {
        0
    };
    // SAFETY: `groups` has room for `count` resources; `full` is the device's SM resource.
    let code = unsafe {
        sys::cuDevSmResourceSplitByCount(
            groups.as_mut_ptr(),
            &mut count,
            full,
            &mut remaining,
            flags,
            attempt.min_count,
        )
    };
    if code != sys::cudaError_enum::CUDA_SUCCESS {
        return Err(format!("{code:?}"));
    }
    groups.truncate(count as usize);
    let grant = SmGrant {
        group_sms: groups.first().map_or(0, |g| sm_view(g).smCount),
        groups: count,
        remaining_sms: sm_view(&remaining).smCount,
    };
    Ok(Granted {
        groups,
        remaining,
        grant,
    })
}

struct Partition {
    green: sys::CUgreenCtx,
    raw: sys::CUstream,
    stream: Arc<CudaStream>,
    context: usize,
    sms: u32,
}

// SAFETY: the handles are driver identifiers; the driver serialises operations on them.
unsafe impl Send for Partition {}
// SAFETY: as above; the wrapper stream is `Send + Sync`.
unsafe impl Sync for Partition {}

impl Partition {
    fn create(
        ctx: &Arc<CudaContext>,
        resources: &mut [sys::CUdevResource],
        priority: i32,
    ) -> Result<Self> {
        let mut desc: sys::CUdevResourceDesc = std::ptr::null_mut();
        // SAFETY: `resources` holds `len` driver-filled SM resources from one split call.
        unsafe {
            sys::cuDevResourceGenerateDesc(
                &mut desc,
                resources.as_mut_ptr(),
                resources.len() as u32,
            )
            .result()?;
        }
        let mut green: sys::CUgreenCtx = std::ptr::null_mut();
        // SAFETY: `desc` was generated above; the flag requests the context's default stream.
        unsafe {
            sys::cuGreenCtxCreate(
                &mut green,
                desc,
                ctx.cu_device(),
                sys::CUgreenCtxCreate_flags::CU_GREEN_CTX_DEFAULT_STREAM as u32,
            )
            .result()?;
        }
        let mut owned = Self {
            green,
            raw: std::ptr::null_mut(),
            stream: ctx.default_stream(),
            context: 0,
            sms: 0,
        };
        // SAFETY: `green` is the live context created above.
        unsafe {
            sys::cuGreenCtxStreamCreate(
                &mut owned.raw,
                green,
                sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                priority,
            )
            .result()?;
        }
        let mut granted = zeroed_resource();
        // SAFETY: `granted` is a valid out-parameter for the SM resource type.
        unsafe {
            sys::cuGreenCtxGetDevResource(
                green,
                &mut granted,
                sys::CUdevResourceType::CU_DEV_RESOURCE_TYPE_SM,
            )
            .result()?;
        }
        owned.sms = sm_view(&granted).smCount;
        let mut primary: sys::CUcontext = std::ptr::null_mut();
        // SAFETY: `green` is live and `primary` is a valid out-parameter.
        unsafe { sys::cuCtxFromGreenCtx(&mut primary, green).result()? };
        owned.context = primary as usize;
        // SAFETY: `raw` is a stream of the device's primary-context family; `owned` destroys it
        // after every user of the wrapper is gone.
        owned.stream = unsafe { ctx.wrap_borrowed_stream(owned.raw) };
        Ok(owned)
    }
}

impl Drop for Partition {
    fn drop(&mut self) {
        // SAFETY: the stream and context were created by `create` and are destroyed once.
        unsafe {
            if !self.raw.is_null() {
                let _ = sys::cuStreamSynchronize(self.raw);
                let _ = sys::cuStreamDestroy_v2(self.raw);
            }
            if !self.green.is_null() {
                let _ = sys::cuGreenCtxDestroy(self.green);
            }
        }
    }
}

/// The vision and decode green contexts of one device, created from disjoint SM groups of a
/// single split of the device's SM resource, where a partition may be the split's remaining
/// resource. The wrapped streams must not be used after this value is dropped.
pub struct GreenPartitions {
    vision: Partition,
    decode: Partition,
    split: SmSplit,
    log: Vec<String>,
}

impl GreenPartitions {
    /// Creates the two partitions on `ctx`'s device. `priorities` are the vision and decode
    /// stream priorities (lower numbers are higher priority). Split requests are tried in
    /// turn until one grant can be divided into the requested sizes; on failure the error
    /// lists what every request was granted.
    pub fn create(ctx: &Arc<CudaContext>, split: SmSplit, priorities: (i32, i32)) -> Result<Self> {
        let device = device_sms(ctx)?;
        split.check_fits(device.sms).map_err(refusal)?;
        let mut full = zeroed_resource();
        // SAFETY: `full` is a valid out-parameter for the SM resource type.
        unsafe {
            sys::cuDeviceGetDevResource(
                ctx.cu_device(),
                &mut full,
                sys::CUdevResourceType::CU_DEV_RESOURCE_TYPE_SM,
            )
            .result()?;
        }
        let mut log = Vec::new();
        let mut chosen = None;
        for attempt in split.attempts(device.granularity) {
            let label = format!(
                "minCount={} ignoreCoscheduling={}",
                attempt.min_count, attempt.ignore_coscheduling
            );
            match run_attempt(&full, device.sms, attempt) {
                Ok(granted) => {
                    let g = granted.grant;
                    log.push(format!(
                        "{label}: {} groups of {} SMs, remaining {} SMs",
                        g.groups, g.group_sms, g.remaining_sms
                    ));
                    if let Some(assignment) = split.assign(&g) {
                        chosen = Some((granted, assignment));
                        break;
                    }
                }
                Err(code) => log.push(format!("{label}: refused ({code})")),
            }
        }
        let Some((granted, a)) = chosen else {
            return Err(CudaError::Driver(format!(
                "the device cannot form {split}; the driver granted: {}",
                log.join("; ")
            )));
        };
        let take = |first: u32, groups: u32, remaining: bool| {
            let mut set: Vec<sys::CUdevResource> =
                granted.groups[first as usize..(first + groups) as usize].to_vec();
            if remaining {
                set.push(granted.remaining);
            }
            set
        };
        let mut vision_set = take(0, a.vision_groups, a.vision_remaining);
        let mut decode_set = take(a.vision_groups, a.decode_groups, a.decode_remaining);
        let vision = Partition::create(ctx, &mut vision_set, priorities.0)?;
        let decode = Partition::create(ctx, &mut decode_set, priorities.1)?;
        for (name, partition, want) in [
            ("vision", &vision, split.vision),
            ("decode", &decode, split.decode),
        ] {
            if partition.sms != want {
                return Err(CudaError::Driver(format!(
                    "the {name} green context holds {} SMs instead of {want}",
                    partition.sms
                )));
            }
        }
        Ok(Self {
            vision,
            decode,
            split,
            log,
        })
    }

    /// What each split request tried while forming the partitions was granted.
    pub fn grant_log(&self) -> &[String] {
        &self.log
    }

    /// The requested split.
    pub fn split(&self) -> SmSplit {
        self.split
    }

    /// The stream whose work runs on the vision partition.
    pub fn vision_stream(&self) -> &Arc<CudaStream> {
        &self.vision.stream
    }

    /// The stream whose work runs on the decode partition.
    pub fn decode_stream(&self) -> &Arc<CudaStream> {
        &self.decode.stream
    }

    /// The driver's context handle of the vision partition, as a graph kernel node reports it.
    pub fn vision_context(&self) -> usize {
        self.vision.context
    }

    /// The driver's context handle of the decode partition, as a graph kernel node reports it.
    pub fn decode_context(&self) -> usize {
        self.decode.context
    }
}
