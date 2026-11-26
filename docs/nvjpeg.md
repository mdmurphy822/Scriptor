# nvJPEG GPU decode (T55)

Status: **off, and kept off.** nvJPEG output is not pixel-identical to the
libjpeg-turbo path in any JPEG mode, so the enabled policy routes every image
to the CPU decoder. The code is retained behind a cargo feature for
measurement only.

## What exists

| Piece | Location |
| --- | --- |
| Dynamic binding to `libnvjpeg` (resolved by soname `libnvjpeg.so.13`, then `libnvjpeg.so`; absent library or symbol is an error) | `crates/scriptor-kernels/src/nvjpeg_ffi.rs`, cargo feature `scriptor-kernels/nvjpeg` |
| Batched decode to device RGB (`u8`, HWC, images packed back to back: the layout `scriptor_normalise_patchify_bf16` reads) | `NvjpegDecoder::decode_batch`, `decode_routed` |
| Header inspection and per-image routing policy | `crates/scriptor-image/src/nvjpeg.rs` |
| Runtime switch | environment `SCRIPTOR_NVJPEG`: unset or any other value = off; `1`/`true` = on with the measured eligibility rules; `unverified` = route everything decodable to the device (measurement only) |
| Fixtures | `tests/fixtures/image/modes/*.jpg`, written by `tools/fixtures/gen_jpeg_modes.py` (synthetic, PIL encoder) |
| Tests | `crates/scriptor-kernels/tests/nvjpeg_gpu.rs` (GPU, `#[ignore]`), unit tests in both source files |

The feature is not implied by any other feature and is not in any default set.
With the switch off, or with the feature absent, decoding is exactly
`decode_jpeg_libjpeg`. Each image is routed individually; an image the policy
rejects (arithmetic coding, non-8-bit, component count other than 1 or 3, or a
mode listed in `CPU_ONLY_SUBSAMPLING` / `PROGRESSIVE_ON_CPU`) is decoded on the
host and uploaded into the same packed device buffer. A failed device batch
falls back to the host for all its images.

## Pixel-identity gate result

Compared with `decode_jpeg_libjpeg` (islow IDCT, fancy upsampling, JCS_RGB) on
the synthetic fixtures; CUDA 13.0 nvJPEG 13.0.1, GB10. "Differing bytes" counts
bytes of the RGB output that are not equal.

| Mode (fixture) | Pixel-identical | Max abs diff (chroma interpolation on) | Differing bytes |
| --- | --- | --- | --- |
| baseline 4:4:4 (96x64, 67x45) | no | 2 | 668 of 18432, 360 of 9045 |
| baseline 4:2:2 (96x64, 67x45) | no | 2, 3 | 631, 324 |
| baseline 4:2:0 (96x64, 67x45, 259x17) | no | 3 | 569, 256, 478 |
| progressive 4:4:4 / 4:2:0 | no | 3 | 674, 586 |
| grayscale (96x64, 67x45) | no | 1 | 297, 147 |
| 3x5 4:2:0 | no | 42 | 33 |
| 4:2:0 with EXIF orientation 6 | no | 2 | 577 |

Findings:

- No mode is identical, including 4:4:4 and grayscale where there is no
  upsampling: the IDCT and YCbCr-to-RGB rounding differ by up to 1-3 levels.
  The default, hybrid and GPU-hybrid backends give the same bytes; the hardware
  backend does not exist on this device (`architecture mismatch`).
- With `NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION` off, subsampled modes are
  far worse (max diff 55-79 on 4:2:0 and 4:2:2: nearest-neighbour chroma);
  the option is therefore on by default in `DecoderOptions`.
- 4:4:0 and 4:1:1 fixtures could not be produced with the installed Pillow;
  they are classified by the header parser and route to the CPU by policy.
- EXIF orientation: neither decoder applies it. Both return the stored raster
  (96x64 for the orientation-6 fixture); the test asserts that.

Consequence: all subsampling modes and progressive frames are in the
CPU-only lists, so `SCRIPTOR_NVJPEG=1` decodes on the CPU. `SCRIPTOR_NVJPEG=unverified`
exists only to measure the device path. The switch stays off unless a future
nvJPEG release closes the gap; re-run
`SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features nvjpeg,cuda --test nvjpeg_gpu -- --ignored --test-threads=1`
and the tests that assert "now byte-identical" will say so.

## Timing

Synthetic page-sized 4:2:0 baseline JPEGs (`gen_jpeg_modes.py --page`), mean of
8 runs after one warm-up, other workers running on the machine (noisy,
indicative only). CPU = libjpeg-turbo to a host RGB buffer, single thread.
nvJPEG = allocate, decode, synchronise (output stays on the device; no
download counted).

| Page | File size | CPU (ms) | nvJPEG default backend, 1 thread (ms) | nvJPEG GPU-hybrid, 1 thread (ms) | nvJPEG GPU-hybrid, batch of 4 (ms/image) |
| --- | --- | --- | --- | --- | --- |
| 1700x2200 | 1.4 MB | 14-15 | 20.1 | 13.3 | 13.2 |
| 2550x3300 | 3.2 MB | 34-36 | 37.5 | 29.2 | 29.4 |

nvJPEG is at best about 1.2x faster than one libjpeg-turbo thread and at worst
slower; Huffman decoding dominates in both. It would save at most a few
milliseconds per page against seconds of generation per page, and the CPU
decode runs concurrently with GPU work, so it is not on the critical path.
Device-resident output does skip the host-to-device copy of the RGB raster
(about 11-25 MB per page), which the table does not include.

## What the bridge sends

`tests/fixtures/server/semantik_request.json` shows the request shape:
`image_url` parts carrying `data:image/jpeg;base64,...` URLs, so the engine
decodes JPEG page crops. Real-document JPEG parameters (subsampling, size) are
not in the repository fixtures.

## Reproducing

```
python3 -I tools/fixtures/gen_jpeg_modes.py <dir> --page 1700x2200 2550x3300
SCRIPTOR_NVJPEG_BENCH_DIR=<dir> SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels \
  --features nvjpeg,cuda --test nvjpeg_gpu -- --ignored --test-threads=1 --nocapture decode_timing
```

`SCRIPTOR_NVJPEG_BACKEND` (0, 1, 2) and `SCRIPTOR_NVJPEG_THREADS` select the backend and
host thread count for the timing test.
