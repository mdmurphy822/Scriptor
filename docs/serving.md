# Serving

Running `scriptor serve`, its flags, request limits and the crash-safety contract. Install first: [install.md](install.md). The document bridge is covered in [bridge.md](bridge.md).

## Features

Highlights:

- OpenAI-compatible `POST /v1/chat/completions` (streaming and non-streaming), `GET /v1/models`, `GET /health` and `GET /metrics`, with the `vllm:*` series a vLLM seat reports plus `scriptor_*` engine-native series.
- CUDA graphs and on-device sampling and drafting, captured at every boot and held in device memory only. Nothing is persisted: no compile caches, tuning tables or warm-boot snapshots.
- Speculative decoding with the model's multi-token-prediction (MTP) head, `--num-speculative-tokens` draft tokens per step, with draft acceptance reported per position.
- Fused vision attention and down-projection, batched vision encoding across waiting requests, and an optional `--vision-overlap` mode that runs vision on its own stream.
- Crash safety: the engine thread is supervised, request intake is bounded by bytes, host memory is watched through four pressure tiers, and device faults end the process with an exit code a supervisor can act on.
- Staged boot (detect, plan, weights, kv_pool, caches, graphs, self_test) reported live on `/health` and timed per stage on `/metrics`.
- Weights, vision and KV cache stay in BF16.

## Crash safety

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

`scriptor serve` also runs the document OCR bridge in the same process, on port 8765 by default (see [OCR bridge](bridge.md)); add `--no-bridge` to serve the engine alone.

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
| `--bridge-listen ADDR`, `--bridge-tls-cert PATH`, `--bridge-tls-key PATH`, `--bridge-token-env NAME` | Bridge bind address (loopback by default); any other address requires a bearer token and TLS. See [OCR bridge](bridge.md). |
| `--no-bridge` | Serve the engine alone, without the bridge. |
| `--host-reserve-bytes N` | Host memory left for other tenants of the machine, added to the derived reserve; `SCRIPTOR_HOST_RESERVE_BYTES` supplies it when the flag is absent. |

Requests that use an unimplemented feature are refused with a 400 in the OpenAI error body, naming the parameter: `logprobs: true`, `top_logprobs` above 0, `response_format` other than `{"type": "text"}`, `logit_bias` with entries, `n` other than 1, and image URLs that are not base64 data URLs.

A request may set `stop_on_repetition: true` to end generation with a `stop` once the text repeats a unit of at least ten characters ten times in a row; it is off unless set.

`scriptor convert --model-dir DIR --out FILE` converts a checkpoint to the engine-native file, and `scriptor inspect --path PATH` reads a checkpoint directory or engine file. The `seat/` folder holds a launch-script template and a systemd unit example for running Scriptor as a detached seat.

> [!IMPORTANT]
> Scriptor is local and explicit. It binds the address you give it, reads the model directory you give it, downloads nothing and discovers nothing, and it fails closed on missing, stale or mismatched state.
