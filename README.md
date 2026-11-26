<div align="center">
<pre>
███████╗ ██████╗██████╗ ██╗██████╗ ████████╗ ██████╗ ██████╗ 
██╔════╝██╔════╝██╔══██╗██║██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
███████╗██║     ██████╔╝██║██████╔╝   ██║   ██║   ██║██████╔╝
╚════██║██║     ██╔══██╗██║██╔═══╝    ██║   ██║   ██║██╔══██╗
███████║╚██████╗██║  ██║██║██║        ██║   ╚██████╔╝██║  ██║
╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝
</pre>
</div>

<div align="center">

*A native GLM-OCR inference server for the NVIDIA GB10: scans in, recognised text out, over an OpenAI-compatible API, with no saved warm state.*

[![License](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20Commercial-blue)](https://chironcourseworks.com)
![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)
![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20NVIDIA%20GB10-lightgrey)

[Getting Started](#getting-started) • [Install](#install) • [Performance](#performance) • [OCR bridge](#ocr-bridge) • [Reproduce](#reproducing-the-benchmark) • [Documentation](#documentation) • [Contributing](#contributing) • [License](#license)

</div>

## Introduction

Scriptor (Latin for "scribe") is a throughput-optimised, local-only inference engine for the GLM-OCR model, written in Rust with hand-written CUDA kernels and built for the NVIDIA GB10 in the DGX Spark (ARM64). It serves GLM-OCR through an OpenAI-compatible chat-completions endpoint, accepts page images in the request, and exposes vLLM-compatible `vllm:*` Prometheus metrics, so a client or dashboard built for a vLLM seat works unchanged. It is local and fail-closed: the model directory and bind address are explicit, nothing is downloaded, nothing is discovered, and a missing, stale or mismatched input stops the start instead of being guessed around.

Highlights:

- OpenAI-compatible `POST /v1/chat/completions` (streaming and non-streaming), `GET /v1/models`, `GET /health` and `GET /metrics`, with the `vllm:*` series a vLLM seat reports plus `scriptor_*` engine-native series.
- CUDA graphs and on-device sampling and drafting, captured at every boot and held in device memory only. Nothing is persisted: no compile caches, tuning tables or warm-boot snapshots.
- Speculative decoding with the model's multi-token-prediction (MTP) head, `--num-speculative-tokens` draft tokens per step, with draft acceptance reported per position.
- Fused vision attention and down-projection, batched vision encoding across waiting requests, and an optional `--vision-overlap` mode that runs vision on its own stream.
- Crash safety: the engine thread is supervised, request intake is bounded by bytes, host memory is watched through four pressure tiers, and device faults end the process with an exit code a supervisor can act on.
- Staged boot (detect, plan, weights, kv_pool, caches, graphs, self_test) reported live on `/health` and timed per stage on `/metrics`.
- Weights, vision and KV cache stay in BF16.

### Crash safety

- **Memory plan.** The plan stage refuses a boot whose worst-case plan exceeds the memory free, naming the source it measured. On a unified-memory device such as the GB10 the figure is the larger of device free memory and host `MemAvailable`.
- **Bounded intake.** Every in-flight request holds a reservation sized by its body and the decoded payload of its images, capped at one sixteenth of host `MemAvailable`. Load the intake cannot hold is answered with 429 or 503 and a `Retry-After` header.
- **Pressure tiers.** A monitor samples host memory and memory-stall pressure and moves the engine through Normal, Watch, Shed and Protect, only ever freeing memory in response. Watch refuses new requests with a retry delay; Shed cancels the youngest running request once per decode step, drops the embedding cache and trims the device memory pool; Protect fails every request and tells the owner to drain and exit.
- **Device faults.** On a sticky device fault (illegal address, launch failure, ECC error, destroyed context) the engine fails every in-flight request with a message naming the CUDA result code, answers `/health` with 503 and `{"state":"dead","cuda":"<code>"}`, stops listening and exits 75 without draining. Out-of-memory and plan-exceeded failures are per-request errors and do not stop the engine.
- **Supervised workers.** A panic in a worker is contained: the failed job is dropped, its caller sees a disconnect, the worker state is rebuilt, and one bad request never shrinks the pool.

| Exit code | Meaning | Restart |
| --------- | ------- | ------- |
| 0 | Clean stop after a drain | no |
| 78 | Refused configuration: the memory plan does not fit, or the checkpoint, tokenizer or cache directory is unusable | never |
| 75 | Transient failure: a boot stage failed, or the device entered a sticky fault | yes |
| 70 | Internal failure: a panic, a lost thread or a runtime that could not start | yes |

`SIGTERM` stops admitting requests, finishes those in flight and exits 0.

## Install

The [releases page](https://github.com/mdmurphy822/Scriptor/releases) has a prebuilt aarch64 Linux CUDA build for the NVIDIA GB10 (DGX Spark). Building from source needs Rust 1.85 or newer (the workspace uses edition 2024; `rust-toolchain.toml` selects the stable channel through rustup), the CUDA 13 toolkit with `nvcc` (kernels are compiled for `sm_121`), and a GLM-OCR checkpoint snapshot directory you supply.

```bash
cargo build --release --locked --workspace
# one binary, target/release/scriptor, runs the engine and the OCR bridge
```

Kernel fatbins compile in parallel and are cached by content in `$SCRIPTOR_KERNEL_CACHE` (default `~/.cache/scriptor/kernels`, `off` disables); `cargo test --profile gpu-test` skips LTO for faster GPU-test builds.

## Getting Started

Check the device and, with a checkpoint, the memory plan before serving. `doctor` prints the itemised plan and exits non-zero when it is refused:

```bash
scriptor doctor --model /path/to/glm-ocr \
  --max-model-len 16384 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

Serve the endpoint. Every path is explicit; nothing is written to disk unless `--cache-dir` is given:

```bash
scriptor serve --model /path/to/glm-ocr \
  --host 127.0.0.1 --port 8002 \
  --max-model-len 16384 --max-num-seqs 32 \
  --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

`scriptor serve` also runs the document OCR bridge in the same process, on port 8765 by default (see [OCR bridge](#ocr-bridge)); add `--no-bridge` to serve the engine alone.

The engine is ready when `GET /health` answers 200 (it answers 503 with the per-stage boot record until then) and `GET /v1/models` lists the served name, `glm-ocr` by default. Send a page image as a chat message with an image data URL.

`scriptor serve` flags, with defaults:

| Flag | Meaning |
| ---- | ------- |
| `--model PATH` (alias `--model-dir`) | Checkpoint snapshot directory: `config.json`, tokenizer files, safetensors weights. Required. |
| `--cache-dir PATH` | Directory for tuning results and boot records; without it no derived state is read or written. |
| `--host ADDR`, `--port N` | Listen address (`127.0.0.1`) and port (`8002`). |
| `--served-model-name NAME` | Model name exposed by `/v1/models` and accepted in requests (`glm-ocr`). |
| `--gpu-memory-utilization F` (alias `--gpu-share`) | Fraction of device memory the engine may use (`0.60`). |
| `--max-num-seqs N` | Concurrent sequences; absent derives the largest count whose slots can each hold a full context within the memory share. |
| `--max-model-len N` | Context length in tokens (`16384`). |
| `--max-pixels N` | Image pixel ceiling; the model default applies when absent. |
| `--num-speculative-tokens N` (alias `--speculative-tokens`) | MTP draft tokens per step; `0` disables speculation (`0`). |
| `--vision-overlap` | Run vision batches concurrently on their own stream (off by default). |
| `--embedding-release-after-prefill` | Release a sequence's image embeddings and their embedding-budget charge once its prefill completes (off by default; env `SCRIPTOR_EMBEDDING_RELEASE_AFTER_PREFILL`). A recomputed sequence re-resolves them from the embedding cache or the vision tower. |
| `--sm-split SPLIT\|off` | SM partitions as `vision=N,decode=N` (implies `--vision-overlap`), or `off` for the full device. Absent, `vision=40,decode=8` is used on a 48-SM device whose driver grants it, and the full device otherwise (one `scriptor:` log line says which); an explicit split the device cannot grant fails boot. |
| `--drain-seconds F` | Upper bound on the shutdown drain, for hosts that bound stopping, such as systemd `TimeoutStopSec`. |
| `--bridge-port N` | Port of the document OCR bridge that runs beside the engine (`8765`). The bridge starts once the engine is ready. |
| `--bridge-listen ADDR`, `--bridge-tls-cert PATH`, `--bridge-tls-key PATH`, `--bridge-token-env NAME` | Bridge bind address (loopback by default); any other address requires a bearer token and TLS. See [OCR bridge](#ocr-bridge). |
| `--no-bridge` | Serve the engine alone, without the bridge. |
| `--host-reserve-bytes N` | Host memory left for other tenants of the machine, added to the derived reserve; `SCRIPTOR_HOST_RESERVE_BYTES` supplies it when the flag is absent. |

Requests that use an unimplemented feature are refused with a 400 in the OpenAI error body, naming the parameter: `logprobs: true`, `top_logprobs` above 0, `response_format` other than `{"type": "text"}`, `logit_bias` with entries, `n` other than 1, and image URLs that are not base64 data URLs.

A request may set `stop_on_repetition: true` to end generation with a `stop` once the text repeats a unit of at least ten characters ten times in a row; it is off unless set.

`scriptor convert --model-dir DIR --out FILE` converts a checkpoint to the engine-native file, and `scriptor inspect --path PATH` reads a checkpoint directory or engine file. The `seat/` folder holds a launch-script template and a systemd unit example for running Scriptor as a detached seat.

> [!IMPORTANT]
> Scriptor is local and explicit. It binds the address you give it, reads the model directory you give it, downloads nothing and discovers nothing, and it fails closed on missing, stale or mismatched state.

## Performance

Scriptor and vLLM were run head to head on the same model, the same documents, the same OCR bridge protocol and the same GPU, in 10 paired runs per engine with a fresh engine boot for every run.

### Setup

| Item | Value |
| ---- | ----- |
| Hardware | NVIDIA DGX Spark, GB10 (Cortex-X925 CPU, 121 GiB unified memory), driver 580.178.04, Linux 7.0.0-1025-nvidia |
| Model | zai-org/GLM-OCR, snapshot 2e85a62840ccac27daa451df36c736c4636b8628; BF16 weights, vision and KV cache |
| Scriptor | commit 3f6b50f35f9a7761d998f1504c0c0cf4b45158bd, default build, `--num-speculative-tokens 4`, `--max-num-seqs 32`, `--max-model-len 16384`, GPU memory share 0.60 |
| vLLM | `vllm/vllm-openai:v0.33.0`, MTP speculative decoding with 3 draft tokens (`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`), `--max-num-seqs 32`, `--max-model-len 16384`, `--gpu-memory-utilization 0.60`, `--limit-mm-per-prompt '{"image":1}'`, `--trust-remote-code`, `--allowed-local-media-path /` |
| Workload | 15 example documents, 368 pages per run (3,680 pages per engine), sent through an OCR bridge as PNG region crops with 32 requests in flight; about 46,500 requests per engine (46,570 for Scriptor, 46,563 for vLLM) |
| Runs | 10 completed pairs of the 10 planned, so 20 engine boots; order alternated (Scriptor first, then vLLM first, and so on); one engine resident at a time; no warm-up |
| Statistics | Ratios are Scriptor / vLLM, with 95% paired bootstrap confidence intervals (10,000 resamples, seed 20261125) |

Verdict rule: for a lower-is-better metric the advantage is the vLLM value divided by the Scriptor value (the reverse for throughput). A metric is a WIN when the whole confidence interval of the advantage is above 1.05, a LOSS when it is entirely below 0.952, and a TIE when the interval overlaps the plus or minus 5% band.

Each engine ran its own shipped configuration, so this compares two configurations, not identical settings. The engines differ in speculative depth (4 draft tokens for Scriptor, 3 for vLLM).

The published run drove both engines through a separate, compatible OCR bridge from the owner's document pipeline. A follow-up check ran the same 15 documents three times through `scriptor serve` with its own built-in bridge (see [OCR bridge](#ocr-bridge)).

### Results

p50 and p99 are taken over runs (wall time, throughput, boot), over 150 per-document observations (per-document latency), or over request histograms (bucket-interpolated estimates; the true value lies inside the quoted bucket).

| Metric | Scriptor p50 | vLLM p50 | p50 ratio [95% CI] | Scriptor p99 | vLLM p99 | p99 ratio [95% CI] | Verdict |
| ------ | -----------: | -------: | ------------------ | -----------: | -------: | ------------------ | ------- |
| Wall time, 368 pages (s) | 236.7 | 454.8 | 0.520 [0.518, 0.526] | 240.6 | 457.1 | 0.526 [0.519, 0.528] | WIN |
| Throughput (pages/hour) | 5,598 | 2,913 | 1.922 [1.902, 1.930] | 5,658 | 2,932 | 1.930 [1.915, 1.934] | WIN |
| Per-document latency (s) | 106.4 | 149.1 | 0.714 [0.650, 0.752] | 239.9 | 456.0 | 0.526 [0.519, 0.530] | WIN |
| Time to first token (s) | 5.070 | 9.167 | 0.553 [0.540, 0.577] | 29.54 | 39.48 | 0.748 [0.748, 0.749] | WIN |
| End-to-end request (s) | 8.339 | 13.29 | 0.627 [0.623, 0.632] | 29.69 | 41.86 | 0.709 [0.697, 0.726] | WIN |
| Request decode time (s) | 0.578 | 1.085 | 0.533 [0.527, 0.539] | 7.054 | 16.18 | 0.436 [0.425, 0.446] | WIN |
| Boot to ready, after the first boot (s) | 14.25 | 154.3 | 0.092 [0.092, 0.096] | 14.69 | 154.4 | 0.095 [0.092, 0.095] | WIN |

All 14 headline comparisons shown are WINs (0 TIE, 0 LOSS). Two more metrics carry no verdict because the engines define them differently: inter-token latency (per token for Scriptor, per speculative step for vLLM) and request prefill time (Scriptor's includes queueing, vLLM's does not).

Run-to-run separation, minimum to maximum over the 10 runs per engine: wall time 234.0 to 240.7 s for Scriptor and 451.8 to 457.1 s for vLLM; pages per hour 5,503 to 5,662 against 2,898 to 2,933; boot to ready (9 boots each) 14.2 to 14.7 s against 148.2 to 154.4 s. For these three measures every Scriptor run beat every vLLM run. Time to first token, end-to-end and decode time come from pooled histograms, so there is no per-run range; their verdicts rest on the bootstrap intervals above.

Per-document latency is completion time inside a saturated 15-document batch, and time to first token includes queueing; neither is an isolated single-request latency. Boot to ready excludes the first boot of each engine, which found a cold page cache.

### Accuracy, runaways and memory

| Measure | Scriptor | vLLM | Verdict |
| ------- | -------: | ---: | ------- |
| Character error rate against the PDF text layer (mean) | 0.006 | 0.006 | TIE (ratio 0.995 [0.995, 0.997]) |
| Word recovery against the PDF text layer (mean, %) | 97.69 | 97.67 | TIE (ratio 1.000 [1.000, 1.000]) |
| Same-engine region agreement between runs (mean, %) | 99.83 | 98.24 | no verdict |
| Scriptor against vLLM region agreement, same pair index (mean, %) | 96.29 | | no verdict |
| Length-limit finishes ("runaways") | 0 | 90 | no verdict |
| Memory used while running (MemAvailable drop, p50 MiB) | 80,105 | 78,415 | TIE (ratio 1.022 [1.020, 1.025]) |

Accuracy is scored against the text layer embedded in the example PDFs (958 scorable regions, text-layer documents only). The accuracy verdict is capped at TIE by design, so a WIN cannot be read as a quality claim. Scriptor and vLLM do not produce byte-identical text: 96.3% of regions agree between the engines, and Scriptor agrees with itself across runs more often (99.83%) than vLLM does (98.24%). The 0 and 90 length-limit finishes are out of about 46,500 requests per engine.

### Limitations

- One machine, one GPU, one session. The results do not transfer to other hardware.
- The 15 documents are a fixed set, not a random sample. Per-document intervals reflect run-to-run variation only.
- With 10 runs per engine, the p99 of a per-run metric is close to the maximum and its interval is wide.
- Percentiles of per-request metrics are bucket-interpolated and as coarse as the bucket widths; each engine reports its own histograms with its own clock.
- The first boot of each engine had a cold page cache and its run is kept in the latency statistics. A page-cold boot was not measured.
- The ten-run comparison used a separate compatible bridge. Scriptor's built-in bridge was checked with three runs only, against Scriptor's own ten runs, not against vLLM.
- Throughput in pages per hour is derived from wall time for 368 pages.

## OCR bridge

`crates/scriptor-bridge` is the document OCR bridge, built into the `scriptor` binary. It turns whole documents into recognised text by rendering pages, finding layout regions, cropping each region, sending the crops to an OpenAI-compatible recognition server (Scriptor or vLLM) and assembling the replies.

`scriptor serve` runs the engine and the bridge as one process from one command. The bridge starts after the engine reports ready, recognises through the engine's own address over loopback, and stops with it on SIGINT, SIGTERM or an engine failure. If the bridge cannot bind its port, the launch fails with that error.

```bash
scriptor serve --model /path/to/glm-ocr --max-num-seqs 32 --num-speculative-tokens 4
# engine on 127.0.0.1:8002, bridge on 127.0.0.1:8765
```

- **Inputs.** A PDF or a PNG, JPEG or WebP image, posted to `POST /ocr` as raw bytes with the matching `Content-Type`, or named by a JSON `path` request. `GET /health`, `GET /progress/<sha256>` and `POST /cancel/<sha256>` report and cancel work. Text-layer PDF pages can be read from the text layer instead of recognised.
- **Layout model.** Layout detection runs a PP-DocLayoutV3 ONNX model through onnxruntime, loaded dynamically at start. You supply both files; nothing is downloaded. Set `SCRIPTOR_BRIDGE_LAYOUT_RUNTIME` to the onnxruntime shared library and `SCRIPTOR_BRIDGE_LAYOUT_WEIGHTS` to the layout ONNX file. If either is missing, the bridge starts and prints one `scriptor-bridge: unavailable (...)` line naming the setting, then answers every document as unavailable.
- **Bridge flags of `scriptor serve`.** `--bridge-port N` (default 8765), `--bridge-listen ADDR` (default loopback), `--bridge-tls-cert`, `--bridge-tls-key`, `--bridge-token-env NAME` and `--no-bridge`. A bind address other than loopback requires a bearer token and TLS; the token is read from the environment variable named by `--bridge-token-env`, `SCRIPTOR_BRIDGE_TOKEN` by default.
- **Engine elsewhere.** `scriptor bridge --serve [--port N] [--listen ADDR] [--tls-cert PATH --tls-key PATH] [--token-env NAME]` runs the bridge alone against an engine you name with `SCRIPTOR_BRIDGE_ENGINE_URL` (a loopback chat-completions URL) and `SCRIPTOR_BRIDGE_MODEL`, which suits a vLLM seat.
- **Settings.** Every setting is an environment variable; the `SCRIPTOR_BRIDGE_` name wins and the older `SEMANTIK_` name is read when it is unset. Under `scriptor serve` the engine URL and model are set from the serve address and `--served-model-name`, and `--max-num-seqs` sets the sequence count unless a setting overrides it.

| Setting | Legacy name | Meaning |
| ------- | ----------- | ------- |
| `SCRIPTOR_BRIDGE_ENGINE_URL` | `SEMANTIK_GLM_OCR_URL` | Chat-completions URL of the recognition engine (loopback). Set automatically by `scriptor serve`. |
| `SCRIPTOR_BRIDGE_MODEL` | `SEMANTIK_GLM_OCR_MODEL` | Served model name. Set automatically by `scriptor serve`. |
| `SCRIPTOR_BRIDGE_API_KEY` | `SEMANTIK_GLM_OCR_API_KEY` | Bearer token sent to the engine. |
| `SCRIPTOR_BRIDGE_LAYOUT_RUNTIME` | `SEMANTIK_LAYOUT_RUNTIME` | onnxruntime shared library. |
| `SCRIPTOR_BRIDGE_LAYOUT_WEIGHTS` | `SEMANTIK_LAYOUT_WEIGHTS` | Layout ONNX file. |
| `SCRIPTOR_BRIDGE_LAYOUT_DEVICE` | `SEMANTIK_GLM_OCR_LAYOUT_DEVICE` | Layout compute device. |
| `SCRIPTOR_BRIDGE_LAYOUT_BATCH_SIZE` | `SEMANTIK_GLM_OCR_LAYOUT_BATCH_SIZE` | Layout batch size (default 4, at most 64). |
| `SCRIPTOR_BRIDGE_WORKERS` | `SEMANTIK_GLM_OCR_WORKERS` | Recognition concurrency (default 32, at most 128; twice the engine's sequences when `SCRIPTOR_BRIDGE_MAX_NUM_SEQS` is set and this is not). |
| `SCRIPTOR_BRIDGE_MAX_NUM_SEQS` | `SEMANTIK_GLM_OCR_MAX_NUM_SEQS` | The engine's concurrent sequences; sizes the workers and the documents served at once. |
| `SCRIPTOR_BRIDGE_HTTP_CONCURRENCY` | `SEMANTIK_GLM_OCR_HTTP_CONCURRENCY` | Documents converted at once (default 2, at most 32). |
| `SCRIPTOR_BRIDGE_FIGURE_DPI` | `SEMANTIK_FIGURE_DPI` | Page render resolution (default 300, at most 600). |
| `SCRIPTOR_BRIDGE_CHECKPOINT_DIR` | `SEMANTIK_GLM_OCR_CHECKPOINT_DIR` | Absolute directory for per-document page checkpoints (not the model checkpoint). |
| `SCRIPTOR_BRIDGE_CONNECT_TIMEOUT_SECS` | `SEMANTIK_GLM_OCR_CONNECT_TIMEOUT_SECS` | Connect timeout (default 30, at most 300). |
| `SCRIPTOR_BRIDGE_REQUEST_TIMEOUT_SECS` | `SEMANTIK_GLM_OCR_REQUEST_TIMEOUT_SECS` | Total request deadline (unset imposes none; at most 1200). |
| `SCRIPTOR_BRIDGE_TOKEN` | `SEMANTIK_BRIDGE_TOKEN` | Bearer token a non-loopback bridge requires. |

The published benchmark above drove the engines through a separate compatible bridge from the owner's document pipeline. Scriptor's built-in bridge was then checked against it: three fresh `scriptor serve` runs over the same 15 documents, compared with the ten published Scriptor runs.

- Speed: median wall time 228.0 s against 236.7 s (3.7% faster, inside the 5% tie band), and every run inside the published range widened by 5%. Pages per hour: 5811 against 5598.
- Text: 34 of the 45 document outputs equal, region by region, an output the published runs also produced. The other 11 differ from their closest published output in one or two regions by one to a few characters (a space, a punctuation mark, a digit, one LaTeX token). That is the same kind of run-to-run variation the published runs show among themselves: ten of the fifteen documents produced between two and seven distinct outputs across those ten runs.
- Layout: one region box on one document (`other`) differs by 2.6 pixels in one coordinate in every run, with identical text. The built-in bridge's layout step is not bit-identical to the separate bridge's.

## Reproducing the benchmark

This repository does not yet contain everything needed to regenerate the published tables. What it contains is the engine, the bridge (both in the one `scriptor` binary) and the driver that records per-document timings and engine metrics. The report generator that pairs runs and computes the ratios and confidence intervals is not yet included here.

Inputs you must supply (none are bundled or downloaded):

| Input | Used for |
| ----- | -------- |
| GLM-OCR snapshot `2e85a62840ccac27daa451df36c736c4636b8628` of `zai-org/GLM-OCR` | The model, served by both engines |
| PP-DocLayoutV3 layout model in ONNX form | Layout detection in the bridge |
| An onnxruntime shared library | Running the layout model |
| 15 example PDF documents, 368 pages in total, and a text file listing them one per line | The workload. The published set is not part of this repository; any set of your own gives a different result. |
| `vllm/vllm-openai:v0.33.0` and the vLLM flags in the setup table | The comparison engine |

Steps:

1. Build Scriptor, which includes the bridge: `cargo build --release --locked --workspace`.
2. Start one engine at a time at a 0.60 GPU memory share. For Scriptor, use the `scriptor serve` command in [Getting Started](#getting-started) with `--num-speculative-tokens 4 --max-num-seqs 32`. For vLLM, use the flags in the setup table.
3. With the engine listening, drive the documents through the bridge and record per-document seconds and the engine metrics. `--bridge-bin` runs `scriptor bridge --serve` against the engine at `--url`:

   ```bash
   tools/e2e/compare_bridge_runs.sh --bridge-bin target/release/scriptor \
     --example-dir <dir> --docs-file <list> --out-dir <out> \
     --layout-runtime <libonnxruntime.so> --layout-weights <layout.onnx> \
     --url http://127.0.0.1:8002/v1/chat/completions --seqs 32 --tag run
   ```

   To run Scriptor's engine and bridge as one process instead, add `--engine-model <checkpoint dir>` and let the script start `scriptor serve` on the host and port of `--url`; pass extra `scriptor serve` arguments with repeated `--serve-arg`. The script also accepts `--semantik-bin <path>` in place of `--bridge-bin` to drive the compatible bridge the published run used.
4. Stop the engine, start the other one, and repeat. Alternate which engine goes first across 10 pairs, with a fresh boot for every run.
5. Compare replies between engines with `python3 -I tools/e2e/diff_replies.py <out> <tagA> <tagB> --show`. The ratios, bootstrap intervals and verdicts come from a report generator that is not yet included in this repository.

## Model and dependency licenses

Scriptor does not bundle or download models or runtime libraries; each keeps its own license, which applies to your use of it.

| Component | Where it comes from | License |
| --------- | ------------------- | ------- |
| GLM-OCR model weights | `zai-org/GLM-OCR` | See the upstream model card |
| PP-DocLayoutV3 layout model (ONNX) | The PP-DocLayoutV3 release you convert or download | See the upstream model card |
| onnxruntime shared library | The onnxruntime release you install | See the upstream onnxruntime repository |

The license of the weights you use is not changed by this repository's AGPL-3.0-only or commercial license, which covers Scriptor's own code.

## Documentation

- [docs/boot.md](docs/boot.md) — the boot memory plan, boot stages and how boot time is measured
- [docs/drafting.md](docs/drafting.md) — MTP speculative decoding, acceptance policy and its metrics
- [docs/attention.md](docs/attention.md) — attention kernels
- [docs/gemm.md](docs/gemm.md) — GEMM kernels
- [docs/equivalence.md](docs/equivalence.md) and [docs/e2e-equivalence.md](docs/e2e-equivalence.md) — kernel and end-to-end equivalence checks
- [docs/engine-file.md](docs/engine-file.md) — the engine-native weights file
- [docs/model-facts.md](docs/model-facts.md) — the GLM-OCR facts the engine matches
- [docs/vision-overlap.md](docs/vision-overlap.md) and [docs/nvjpeg.md](docs/nvjpeg.md) — vision overlap and GPU JPEG decode
- [seat/README.md](seat/README.md) — the seat template, readiness and drain contracts, exit codes and fault injection
- [docs/phase0-protocol.md](docs/phase0-protocol.md) — the reference measurement protocol
- [docs/third-party-notices.md](docs/third-party-notices.md) — third-party notices

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) first: Conventional Commit subjects, the format, clippy and test gate, synthetic fixtures only. Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

---

## License

Dual-licensed under AGPL-3.0-only or a commercial license (see
[LICENSE](LICENSE) and [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Commercial licenses are issued through
[Chiron Courseworks](https://chironcourseworks.com); inquiries:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Third-party notices are in [docs/third-party-notices.md](docs/third-party-notices.md). Models and runtime libraries are not bundled and keep their own licenses (see [Model and dependency licenses](#model-and-dependency-licenses)).
