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

**[Project page](https://chironcourseworks.com/scriptor)** • [Quick start](#quick-start) • [Install](docs/install.md) • [OCR bridge](docs/bridge.md) • [Benchmark](docs/benchmark.md) • [Documentation](#documentation) • [Contributing](#contributing) • [License](#license) • [Languages](docs/languages.md)

</div>

## Introduction

Scriptor (Latin for "scribe") is a throughput-optimised, local-only inference engine for the GLM-OCR model, written in Rust with hand-written CUDA kernels and built for the NVIDIA GB10 in the DGX Spark (ARM64). It serves GLM-OCR through an OpenAI-compatible chat-completions endpoint, accepts page images in the request, and exposes vLLM-compatible `vllm:*` Prometheus metrics, so a client or dashboard built for a vLLM seat works unchanged. It is local and fail-closed: the model directory and bind address are explicit, nothing is downloaded, nothing is discovered, and a missing, stale or mismatched input stops the start instead of being guessed around. See [docs/serving.md](docs/serving.md) for the feature list and crash-safety contract.

## Headline results

Scriptor against vLLM, same model, documents, OCR bridge protocol and GPU: 10 paired runs per engine, a fresh boot for every run, 368 pages per run, ratios with 95% paired bootstrap confidence intervals.

| Metric | Scriptor p50 | vLLM p50 | Verdict |
| ------ | -----------: | -------: | ------- |
| Wall time, 368 pages (s) | 236.7 | 454.8 | WIN (0.520x) |
| Throughput (pages/hour) | 5,598 | 2,913 | WIN (1.922x) |
| Boot to ready, after the first boot (s) | 14.25 | 154.3 | WIN (0.092x) |
| Time to first token (s) | 5.070 | 9.167 | WIN (0.553x) |
| End-to-end request (s) | 8.339 | 13.29 | WIN (0.627x) |

One machine, one GPU, one session, a fixed 15-document set; each engine ran its own shipped configuration. Setup, all metrics, accuracy, limitations and how to reproduce: [docs/benchmark.md](docs/benchmark.md).

## Quick start

```bash
git clone https://github.com/mdmurphy822/Scriptor && cd Scriptor
./install.sh all                          # binary, GLM-OCR checkpoint, layout model, onnxruntime
eval "$(./install.sh env | grep '^export')"  # bridge settings for the installed files
scriptor serve --model ~/.local/share/scriptor/models/glm-ocr \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.60 --num-speculative-tokens 4
```

With those settings exported, the engine listens on 127.0.0.1:8002 and the OCR bridge on 127.0.0.1:8765. Send a page image to the engine as an image data URL, and a whole document to the bridge:

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' -d @request.json
curl -s http://127.0.0.1:8765/ocr -H 'Content-Type: application/pdf' \
  -H "Semantik-Source-SHA256: $(sha256sum document.pdf | cut -d' ' -f1)" \
  -H 'Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1' \
  --data-binary @document.pdf
```

`request.json` is a chat-completions body whose user message carries a base64 `image_url` data URL. The server downloads nothing; `install.sh` is the only networked step.

## Install

`./install.sh` installs the release binary, the GLM-OCR checkpoint, the layout model and a CUDA build of onnxruntime for the GB10, each step opt-in and checksum-verified; `./install.sh check` verifies the result. You can also build from source. See [docs/install.md](docs/install.md).

## OCR bridge

`scriptor serve` also runs a document OCR bridge: PDFs and images in, recognised text out, with layout detection through onnxruntime. Flags, settings and the built-in bridge check are in [docs/bridge.md](docs/bridge.md).

## Benchmark

Full tables, accuracy, limitations and reproduction steps: [docs/benchmark.md](docs/benchmark.md).

## Crash safety

Bounded intake, memory pressure tiers, supervised workers and exit codes a supervisor can act on: [docs/serving.md](docs/serving.md#crash-safety).

## Documentation

- [docs/install.md](docs/install.md) — install script, prebuilt release, build from source, model and dependency licenses
- [docs/serving.md](docs/serving.md) — `scriptor serve` flags, request limits, crash safety
- [docs/bridge.md](docs/bridge.md) — the OCR bridge, its flags and settings
- [docs/benchmark.md](docs/benchmark.md) — results, accuracy, limitations, reproducing the benchmark
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
- [docs/languages.md](docs/languages.md) — this README in 38 languages

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) first: Conventional Commit subjects, the format, clippy and test gate, synthetic fixtures only. Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

---

## License

Dual-licensed under AGPL-3.0-only or a commercial license (see
[LICENSE](LICENSE) and [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)).
Copyright 2026 Matthew D. Murphy. Commercial licenses are issued through
[Chiron Courseworks](https://chironcourseworks.com); inquiries:
[contact@chironcourseworks.com](mailto:contact@chironcourseworks.com).

Third-party notices are in [docs/third-party-notices.md](docs/third-party-notices.md). Model weights and runtime libraries are separate downloads and keep their own licenses (see [Model and dependency licenses](docs/install.md#model-and-dependency-licenses)).
