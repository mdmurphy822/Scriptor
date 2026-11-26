# GEMM

Row-major `C[m, n] = A[m, k] * W[n, k]^T (+ bias[n]) (-> GELU)` on cuBLASLt with
fp32 accumulation and bf16 output. Code: `crates/scriptor-kernels/src/cublaslt.rs`
(driver), `crates/scriptor-kernels/src/heuristics.rs` (tuning cache, shape set,
throughput math), `kernels/quant_act.cu` (fp8 activation quantiser),
`crates/scriptor-cli/src/bench_gemm.rs` (`bench gemm`, `tune`).

## Formats

| dtype | A (activations) / W (weights) | scales | output |
|---|---|---|---|
| `bf16` | bf16 / bf16 | none | bf16 |
| `fp8_e4m3` | e4m3 / e4m3 | per-tensor weight scale (expanded to an `[n]` f32 vector), per-token activation scale `[m]` f32 | bf16 |

Constraints: bf16 needs `n` and `k` multiples of 8; fp8 needs `k` a multiple of 16
and `n` of 8 (the ViT patch embedding, `k = 1176`, therefore stays bf16).

cuBLASLt sees `D[n x m, col-major] = W^T * X`, which is the row-major `[m, n]`
product, so no transposes or copies are made.

## Epilogues

`none`, `bias` (bf16 bias `[n]`), `gelu`, `bias_gelu` (bias first). The GELU is the
cuBLASLt epilogue; it is the tanh-form approximation, not the erf GELU the merger
needs, so the merger's GELU stays a separate elementwise kernel unless the
equivalence set accepts the difference. The GPU test checks it against the tanh form.

## Workspace

`GemmEngine::new` reserves the cuBLASLt workspace from the caller's `DeviceArena`
(256-byte aligned; default 64 MiB). Nothing is allocated per call.

## fp8 path

Behind `GemmOptions::fp8` (CLI: `--dtype fp8`, `tune --fp8`). Calls fail while the
flag is off. Activations are quantised per token by
`GemmEngine::quantize_activations` (`scriptor_quant_act_fp8_token`: `scale = amax/448`,
`1` for an all-zero row). The weight scale is expanded once with
`GemmEngine::fill_f32`. cuBLASLt receives both scales in `OUTER_VEC_32F` mode
(A scale length `n`, B scale length `m`). If the library rejects that combination
the probe reports it and fp8 calls fail closed; there is no silent bf16 fallback.

### Capability probe

`GemmEngine::fp8_capability()` asks cuBLASLt for heuristic candidates of an
`m=128, n=4096, k=1024` fp8 problem with `OUTER_VEC_32F` scales and bf16 output,
and returns `{supported, candidates, detail, library_version}`.
`bench gemm --dtype fp8` prints it in the report's `fp8_probe` field.

Recorded result on the GB10: not yet measured (the probe needs the GPU). Fill in
after the first `bench gemm --dtype fp8` run:

| Field | Value |
|---|---|
| supported | |
| candidates | |
| detail | |
| cuBLASLt version | |

## Tuning cache

Key: `(m_bucket, n, k, epilogue, dtype)` with `m_bucket` the next power of two at
or above `m`. Value: index into the heuristic candidate list and the measured
microseconds. File: `<cache-dir>/gemm-tuning-<gpu>-<cublaslt version>.toml`.

- Running GEMMs and `bench gemm --cache-dir` only read the table.
- `scriptor tune --cache-dir <dir>` is the only writer
  (`GemmEngine::tune_and_save`); the directory is always explicit and never defaulted.
- An untuned key uses the heuristic's first candidate.

## `bench gemm`

```
scriptor bench gemm --dtype bf16 --report gemm-bf16.json
scriptor bench gemm --dtype fp8  --report gemm-fp8.json
```

Shapes: every weight shape in `heuristics::model_shapes()` (ViT patch embed, qkv,
proj, mlp gate+up and down, downsample, merger; decoder qkv, o_proj, gate+up, down,
lm_head) at `M` in {32, 128, 1024, 16384, 65536}. Each shape runs `--warmup`
(default 50) untimed and `--iters` (default 200) timed launches; the median device
event time gives TFLOPS (`2*M*N*K / t`) and weight GB/s. Shapes whose operands do not
fit the free device memory are reported as skipped with the byte count. The seat must
be down while it runs.

Sustained numbers on the GB10: not yet measured (needs the GPU). Paste the Markdown
table the command prints here, with `nvidia-smi -q -d CLOCK,POWER` before and after:

| Role | M | N | K | dtype | median us | TFLOPS | weight GB/s |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

The highest large-M (`M >= 16384`) bf16 TFLOPS replaces the assumed ~80 TFLOPS in
`docs/roofline.md`, and the small-M weight GB/s replaces the assumed 240 GB/s.

## Verification

- CPU: `cargo test -p scriptor-kernels` (fp8 codec, tuning table round trip, shape set,
  throughput math).
- Build: `cargo build -p scriptor-kernels --features cuda`.
- GPU (deferred): `SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-kernels --features cuda cublaslt -- --ignored --nocapture`
  compares bf16, bias, GELU and fp8 products with `reference::gemm::gemm_bias`
  and checks that only `tune` writes a cache file.
