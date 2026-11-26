# OCR bridge

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

The [published benchmark](benchmark.md) drove the engines through a separate compatible bridge from the owner's document pipeline. Scriptor's built-in bridge was then checked against it: three fresh `scriptor serve` runs over the same 15 documents, compared with the ten published Scriptor runs.

- Speed: median wall time 228.0 s against 236.7 s (3.7% faster, inside the 5% tie band), and every run inside the published range widened by 5%. Pages per hour: 5811 against 5598.
- Text: 34 of the 45 document outputs equal, region by region, an output the published runs also produced. The other 11 differ from their closest published output in one or two regions by one to a few characters (a space, a punctuation mark, a digit, one LaTeX token). That is the same kind of run-to-run variation the published runs show among themselves: ten of the fifteen documents produced between two and seven distinct outputs across those ten runs.
- Layout: one region box on one document (`other`) differs by 2.6 pixels in one coordinate in every run, with identical text. The built-in bridge's layout step is not bit-identical to the separate bridge's.

## Layout dependencies

`./install.sh layout` and `./install.sh onnxruntime` fetch the layout weights and an onnxruntime library, and `./install.sh env` prints the two environment lines the bridge needs. See [install.md](install.md).

## Request headers

A raw-bytes `POST /ocr` carries two headers: `Semantik-Source-SHA256`, the lowercase hex sha256 of the body, and `Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1`. A request without a matching digest and contract is rejected with 422. `tools/e2e/compare_bridge_runs.sh` sends both.
