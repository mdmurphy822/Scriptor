//! Runtime-loaded binding to `libnvjpeg` and a batched JPEG-to-device-RGB decoder.
//!
//! The library is resolved by name when [`NvjpegLib::load`] runs; a missing
//! library or symbol is an error, never a fallback to a different decoder. The
//! device decoder writes interleaved 8-bit RGB (`HWC`), the layout the
//! preprocessing kernels read, with images packed back to back.

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::ffi::{c_int, c_uint, c_void};
use std::fmt;
use std::sync::Arc;

use libloading::Library;

/// Library names tried in order when resolving nvJPEG.
pub const LIBRARY_NAMES: &[&str] = &["libnvjpeg.so.13", "libnvjpeg.so"];

/// Failure raised by the nvJPEG binding.
#[derive(Debug)]
pub enum NvjpegError {
    /// The shared library or one of its symbols could not be resolved.
    Load(String),
    /// An nvJPEG call returned a nonzero status.
    Status {
        /// The call that failed.
        call: &'static str,
        /// The `nvjpegStatus_t` value.
        code: i32,
    },
    /// A CUDA driver call failed.
    Driver(String),
    /// The input cannot be decoded as planned.
    Input(String),
}

impl fmt::Display for NvjpegError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(m) => write!(f, "nvjpeg unavailable: {m}"),
            Self::Status { call, code } => write!(f, "{call} failed: {}", status_name(*code)),
            Self::Driver(m) => write!(f, "cuda driver error: {m}"),
            Self::Input(m) => write!(f, "nvjpeg input error: {m}"),
        }
    }
}

impl std::error::Error for NvjpegError {}

/// Names an `nvjpegStatus_t` value.
pub fn status_name(code: i32) -> String {
    match code {
        0 => "success".into(),
        1 => "not initialized".into(),
        2 => "invalid parameter".into(),
        3 => "bad jpeg".into(),
        4 => "jpeg not supported".into(),
        5 => "allocator failure".into(),
        6 => "execution failed".into(),
        7 => "architecture mismatch".into(),
        8 => "internal error".into(),
        9 => "implementation not supported".into(),
        10 => "incomplete bitstream".into(),
        other => format!("status {other}"),
    }
}

type Handle = *mut c_void;
type State = *mut c_void;

const OUTPUT_RGBI: c_int = 5;
const FLAG_UPSAMPLING_WITH_INTERPOLATION: c_uint = 1 << 5;

#[repr(C)]
struct NvjpegImage {
    channel: [*mut u8; 4],
    pitch: [usize; 4],
}

type CreateEx = unsafe extern "C" fn(c_int, *mut c_void, *mut c_void, c_uint, *mut Handle) -> c_int;
type Destroy = unsafe extern "C" fn(Handle) -> c_int;
type StateCreate = unsafe extern "C" fn(Handle, *mut State) -> c_int;
type StateDestroy = unsafe extern "C" fn(State) -> c_int;
type GetImageInfo = unsafe extern "C" fn(
    Handle,
    *const u8,
    usize,
    *mut c_int,
    *mut c_int,
    *mut c_int,
    *mut c_int,
) -> c_int;
type Decode = unsafe extern "C" fn(
    Handle,
    State,
    *const u8,
    usize,
    c_int,
    *mut NvjpegImage,
    *mut c_void,
) -> c_int;
type BatchedInit = unsafe extern "C" fn(Handle, State, c_int, c_int, c_int) -> c_int;
type DecodeBatched = unsafe extern "C" fn(
    Handle,
    State,
    *const *const u8,
    *const usize,
    *mut NvjpegImage,
    *mut c_void,
) -> c_int;

/// The resolved nvJPEG entry points; keeps the shared library mapped.
pub struct NvjpegLib {
    create_ex: CreateEx,
    destroy: Destroy,
    state_create: StateCreate,
    state_destroy: StateDestroy,
    get_image_info: GetImageInfo,
    decode: Decode,
    batched_init: BatchedInit,
    decode_batched: DecodeBatched,
    _library: Library,
}

impl fmt::Debug for NvjpegLib {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NvjpegLib")
    }
}

fn check(call: &'static str, code: c_int) -> Result<(), NvjpegError> {
    if code == 0 {
        Ok(())
    } else {
        Err(NvjpegError::Status { call, code })
    }
}

impl NvjpegLib {
    /// Resolves `libnvjpeg` by its soname through the dynamic loader's search path.
    pub fn load() -> Result<Arc<Self>, NvjpegError> {
        Self::load_named(LIBRARY_NAMES)
    }

    /// Resolves the first loadable library in `names` and all required symbols.
    pub fn load_named(names: &[&str]) -> Result<Arc<Self>, NvjpegError> {
        let mut last = String::from("no library names given");
        for name in names {
            // SAFETY: loading runs the library's initialisers; nvJPEG's are benign.
            match unsafe { Library::new(*name) } {
                Ok(library) => return Self::bind(library).map(Arc::new),
                Err(e) => last = format!("{name}: {e}"),
            }
        }
        Err(NvjpegError::Load(last))
    }

    fn bind(library: Library) -> Result<Self, NvjpegError> {
        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                // SAFETY: the symbol is looked up by its documented name and the
                // function pointer type matches its declaration in nvjpeg.h.
                let s: libloading::Symbol<'_, $ty> = unsafe { library.get($name) }
                    .map_err(|e| NvjpegError::Load(format!("{}: {e}", stringify!($name))))?;
                *s
            }};
        }
        Ok(Self {
            create_ex: sym!(b"nvjpegCreateEx\0", CreateEx),
            destroy: sym!(b"nvjpegDestroy\0", Destroy),
            state_create: sym!(b"nvjpegJpegStateCreate\0", StateCreate),
            state_destroy: sym!(b"nvjpegJpegStateDestroy\0", StateDestroy),
            get_image_info: sym!(b"nvjpegGetImageInfo\0", GetImageInfo),
            decode: sym!(b"nvjpegDecode\0", Decode),
            batched_init: sym!(b"nvjpegDecodeBatchedInitialize\0", BatchedInit),
            decode_batched: sym!(b"nvjpegDecodeBatched\0", DecodeBatched),
            _library: library,
        })
    }
}

/// Decoder options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecoderOptions {
    /// Upsample chroma with interpolation instead of replication.
    pub interpolate_chroma: bool,
    /// The `nvjpegBackend_t` value: 0 default, 1 hybrid, 2 GPU hybrid, 3 hardware.
    pub backend: i32,
    /// Host threads nvJPEG may use for batched Huffman decoding.
    pub max_cpu_threads: i32,
}

impl Default for DecoderOptions {
    fn default() -> Self {
        Self {
            interpolate_chroma: true,
            backend: 0,
            max_cpu_threads: 1,
        }
    }
}

/// Dimensions of one JPEG as reported by nvJPEG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageInfo {
    /// Colour components in the stream.
    pub components: i32,
    /// nvJPEG's chroma-subsampling code.
    pub subsampling: i32,
    /// Width of the first component.
    pub width: u32,
    /// Height of the first component.
    pub height: u32,
}

/// One decoded image's placement inside a [`DeviceBatch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent {
    /// Height in pixels.
    pub height: u32,
    /// Width in pixels.
    pub width: u32,
    /// Byte offset of the image in the packed buffer.
    pub offset: usize,
    /// Whether the GPU decoded this image (false: libjpeg-turbo on the host).
    pub on_gpu: bool,
}

#[cfg(feature = "cuda")]
pub use device::{DeviceBatch, NvjpegDecoder};

#[cfg(feature = "cuda")]
mod device {
    use cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut};
    use scriptor_image::nvjpeg::{GpuJpegPolicy, Route, parse_header};

    use super::*;

    /// Decoded images packed back to back as `HWC` `u8` RGB in device memory.
    pub struct DeviceBatch {
        /// The packed pixels.
        pub rgb: CudaSlice<u8>,
        /// One extent per input image, in input order.
        pub extents: Vec<Extent>,
    }

    /// An nvJPEG handle bound to a stream, decoding batches to device RGB.
    pub struct NvjpegDecoder {
        lib: Arc<NvjpegLib>,
        handle: Handle,
        state: State,
        stream: Arc<CudaStream>,
        initialised_batch: i32,
        options: DecoderOptions,
    }

    // SAFETY: the handle and state are owned exclusively by this value and every
    // call that uses them takes `&mut self` or `&self` without shared mutation.
    unsafe impl Send for NvjpegDecoder {}

    fn driver(e: impl fmt::Display) -> NvjpegError {
        NvjpegError::Driver(e.to_string())
    }

    impl NvjpegDecoder {
        /// Creates a decoder whose work is ordered on `stream`.
        pub fn new(
            lib: Arc<NvjpegLib>,
            stream: Arc<CudaStream>,
            options: DecoderOptions,
        ) -> Result<Self, NvjpegError> {
            stream.context().bind_to_thread().map_err(driver)?;
            let flags = if options.interpolate_chroma {
                FLAG_UPSAMPLING_WITH_INTERPOLATION
            } else {
                0
            };
            let mut handle: Handle = std::ptr::null_mut();
            // SAFETY: null allocators select the library defaults; `handle` is a valid out slot.
            check("nvjpegCreateEx", unsafe {
                (lib.create_ex)(
                    options.backend,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    flags,
                    &mut handle,
                )
            })?;
            let mut state: State = std::ptr::null_mut();
            // SAFETY: `handle` was just created; `state` is a valid out slot.
            let status = unsafe { (lib.state_create)(handle, &mut state) };
            if let Err(e) = check("nvjpegJpegStateCreate", status) {
                // SAFETY: `handle` is live and unused afterwards.
                unsafe { (lib.destroy)(handle) };
                return Err(e);
            }
            Ok(Self {
                lib,
                handle,
                state,
                stream,
                initialised_batch: 0,
                options,
            })
        }

        /// Reads the stream parameters nvJPEG derives from the JPEG header.
        pub fn image_info(&self, jpeg: &[u8]) -> Result<ImageInfo, NvjpegError> {
            let (mut n, mut css) = (0 as c_int, 0 as c_int);
            let (mut w, mut h) = ([0 as c_int; 4], [0 as c_int; 4]);
            // SAFETY: `w` and `h` hold the four entries the call writes; `jpeg` is read for its length.
            check("nvjpegGetImageInfo", unsafe {
                (self.lib.get_image_info)(
                    self.handle,
                    jpeg.as_ptr(),
                    jpeg.len(),
                    &mut n,
                    &mut css,
                    w.as_mut_ptr(),
                    h.as_mut_ptr(),
                )
            })?;
            Ok(ImageInfo {
                components: n,
                subsampling: css,
                width: w[0] as u32,
                height: h[0] as u32,
            })
        }

        /// Decodes every JPEG on the device in one batched call, regardless of policy.
        pub fn decode_batch(&mut self, jpegs: &[&[u8]]) -> Result<DeviceBatch, NvjpegError> {
            let mut dims = Vec::with_capacity(jpegs.len());
            for jpeg in jpegs {
                let info = self.image_info(jpeg)?;
                dims.push((info.height, info.width));
            }
            let (mut extents, total) = pack(&dims);
            let mut rgb = self
                .stream
                .alloc_zeros::<u8>(total.max(1))
                .map_err(driver)?;
            let gpu: Vec<usize> = (0..jpegs.len()).collect();
            self.decode_gpu(jpegs, &extents, &gpu, &mut rgb)?;
            for e in &mut extents {
                e.on_gpu = true;
            }
            Ok(DeviceBatch { rgb, extents })
        }

        /// Decodes a batch, sending each image to the device or to libjpeg-turbo as
        /// `policy` routes it, and returns all of them packed in one device buffer.
        pub fn decode_routed(
            &mut self,
            jpegs: &[&[u8]],
            policy: &GpuJpegPolicy,
        ) -> Result<DeviceBatch, NvjpegError> {
            let mut dims = Vec::with_capacity(jpegs.len());
            let mut routes = Vec::with_capacity(jpegs.len());
            for jpeg in jpegs {
                let header = parse_header(jpeg).map_err(|e| NvjpegError::Input(e.to_string()))?;
                dims.push((header.height, header.width));
                routes.push(policy.route(&header));
            }
            let mut gpu: Vec<usize> = (0..jpegs.len())
                .filter(|&i| routes[i] == Route::Gpu)
                .collect();
            let (mut extents, total) = pack(&dims);
            let mut rgb = self
                .stream
                .alloc_zeros::<u8>(total.max(1))
                .map_err(driver)?;
            if !gpu.is_empty() {
                let gpu_jpegs: Vec<&[u8]> = gpu.iter().map(|&i| jpegs[i]).collect();
                let gpu_extents: Vec<Extent> = gpu.iter().map(|&i| extents[i]).collect();
                let all: Vec<usize> = (0..gpu.len()).collect();
                if self
                    .decode_gpu(&gpu_jpegs, &gpu_extents, &all, &mut rgb)
                    .is_ok()
                {
                    for &i in &gpu {
                        extents[i].on_gpu = true;
                    }
                    gpu.clear();
                }
            }
            for i in 0..jpegs.len() {
                if extents[i].on_gpu {
                    continue;
                }
                let image = scriptor_image::decode_jpeg_libjpeg(jpegs[i])
                    .map_err(|e| NvjpegError::Input(e.to_string()))?;
                if (image.height, image.width) != dims[i] {
                    return Err(NvjpegError::Input("header and decoded size differ".into()));
                }
                let e = extents[i];
                let mut dst = rgb.slice_mut(e.offset..e.offset + image.data.len());
                self.stream
                    .memcpy_htod(&image.data, &mut dst)
                    .map_err(driver)?;
            }
            self.stream.synchronize().map_err(driver)?;
            Ok(DeviceBatch { rgb, extents })
        }

        fn decode_gpu(
            &mut self,
            jpegs: &[&[u8]],
            extents: &[Extent],
            indices: &[usize],
            rgb: &mut CudaSlice<u8>,
        ) -> Result<(), NvjpegError> {
            let n = indices.len() as i32;
            if n == 0 {
                return Ok(());
            }
            self.stream.context().bind_to_thread().map_err(driver)?;
            if self.initialised_batch != n {
                // SAFETY: handle and state are live; RGBI is a valid output format.
                check("nvjpegDecodeBatchedInitialize", unsafe {
                    (self.lib.batched_init)(
                        self.handle,
                        self.state,
                        n,
                        self.options.max_cpu_threads,
                        OUTPUT_RGBI,
                    )
                })?;
                self.initialised_batch = n;
            }
            let base = {
                let (ptr, _guard) = rgb.device_ptr_mut(&self.stream);
                ptr as usize
            };
            let mut data = Vec::with_capacity(indices.len());
            let mut lengths = Vec::with_capacity(indices.len());
            let mut dests = Vec::with_capacity(indices.len());
            for &i in indices {
                let e = extents[i];
                data.push(jpegs[i].as_ptr());
                lengths.push(jpegs[i].len());
                dests.push(NvjpegImage {
                    channel: [
                        (base + e.offset) as *mut u8,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                    ],
                    pitch: [e.width as usize * 3, 0, 0, 0],
                });
            }
            // SAFETY: `data`, `lengths` and `dests` each hold `n` entries; every destination
            // is a region of `rgb` sized `height * width * 3`; the inputs outlive the
            // synchronisation below.
            let status = unsafe {
                (self.lib.decode_batched)(
                    self.handle,
                    self.state,
                    data.as_ptr(),
                    lengths.as_ptr(),
                    dests.as_mut_ptr(),
                    self.stream.cu_stream() as *mut c_void,
                )
            };
            let synced = self.stream.synchronize().map_err(driver);
            check("nvjpegDecodeBatched", status)?;
            synced
        }

        /// Decodes one JPEG with the non-batched entry point to a host RGB image.
        pub fn decode_single_to_host(
            &mut self,
            jpeg: &[u8],
        ) -> Result<scriptor_image::RgbImage, NvjpegError> {
            let info = self.image_info(jpeg)?;
            let len = info.width as usize * info.height as usize * 3;
            let mut rgb = self.stream.alloc_zeros::<u8>(len).map_err(driver)?;
            let base = {
                let (ptr, _guard) = rgb.device_ptr_mut(&self.stream);
                ptr as usize
            };
            let mut dest = NvjpegImage {
                channel: [
                    base as *mut u8,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ],
                pitch: [info.width as usize * 3, 0, 0, 0],
            };
            self.stream.context().bind_to_thread().map_err(driver)?;
            // SAFETY: `dest` points at a `len`-byte device region; `jpeg` is valid for its length.
            let status = unsafe {
                (self.lib.decode)(
                    self.handle,
                    self.state,
                    jpeg.as_ptr(),
                    jpeg.len(),
                    OUTPUT_RGBI,
                    &mut dest,
                    self.stream.cu_stream() as *mut c_void,
                )
            };
            let synced = self.stream.synchronize().map_err(driver);
            check("nvjpegDecode", status)?;
            synced?;
            let data = self.stream.clone_dtoh(&rgb).map_err(driver)?;
            Ok(scriptor_image::RgbImage {
                width: info.width,
                height: info.height,
                data,
            })
        }

        /// Copies a batch back to host images, one per extent.
        pub fn download(
            &self,
            batch: &DeviceBatch,
        ) -> Result<Vec<scriptor_image::RgbImage>, NvjpegError> {
            let all = self.stream.clone_dtoh(&batch.rgb).map_err(driver)?;
            Ok(batch
                .extents
                .iter()
                .map(|e| scriptor_image::RgbImage {
                    width: e.width,
                    height: e.height,
                    data: all[e.offset..e.offset + e.width as usize * e.height as usize * 3]
                        .to_vec(),
                })
                .collect())
        }
    }

    impl Drop for NvjpegDecoder {
        fn drop(&mut self) {
            // SAFETY: state and handle are live, owned here, and not used afterwards.
            unsafe {
                (self.lib.state_destroy)(self.state);
                (self.lib.destroy)(self.handle);
            }
        }
    }
}

/// Lays `(height, width)` images out back to back as RGB, all marked as host-decoded.
fn pack(dims: &[(u32, u32)]) -> (Vec<Extent>, usize) {
    let mut offset = 0usize;
    let extents = dims
        .iter()
        .map(|&(height, width)| {
            let e = Extent {
                height,
                width,
                offset,
                on_gpu: false,
            };
            offset += height as usize * width as usize * 3;
            e
        })
        .collect();
    (extents, offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_library_fails_closed() {
        let err = NvjpegLib::load_named(&["libnvjpeg_does_not_exist.so.0"]).unwrap_err();
        assert!(matches!(err, NvjpegError::Load(_)));
        assert!(err.to_string().contains("nvjpeg unavailable"));
    }

    #[test]
    fn empty_name_list_fails_closed() {
        assert!(matches!(
            NvjpegLib::load_named(&[]),
            Err(NvjpegError::Load(_))
        ));
    }

    #[test]
    fn library_without_nvjpeg_symbols_fails_closed() {
        let err = NvjpegLib::load_named(&["libc.so.6"]).unwrap_err();
        assert!(matches!(err, NvjpegError::Load(m) if m.contains("nvjpegCreateEx")));
    }

    #[test]
    fn pack_lays_images_back_to_back() {
        let (extents, total) = pack(&[(2, 3), (4, 5)]);
        assert_eq!(extents[0].offset, 0);
        assert_eq!(extents[1].offset, 18);
        assert_eq!(total, 18 + 60);
    }

    #[test]
    fn status_names_cover_known_codes() {
        assert_eq!(status_name(3), "bad jpeg");
        assert_eq!(status_name(99), "status 99");
    }
}
