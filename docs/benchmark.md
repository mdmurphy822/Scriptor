# Benchmark

Scriptor against vLLM on the same model, documents, bridge protocol and GPU, plus how to reproduce the run. Back to the [README](../README.md).


Scriptor and vLLM were run head to head on the same model, the same documents, the same OCR bridge protocol and the same GPU, in 10 paired runs per engine with a fresh engine boot for every run.

## Setup

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

The published run drove both engines through a separate, compatible OCR bridge from the owner's document pipeline. A follow-up check ran the same 15 documents three times through `scriptor serve` with its own built-in bridge (see [OCR bridge](bridge.md)).

## Results

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

## Accuracy, runaways and memory

| Measure | Scriptor | vLLM | Verdict |
| ------- | -------: | ---: | ------- |
| Character error rate against the PDF text layer (mean) | 0.006 | 0.006 | TIE (ratio 0.995 [0.995, 0.997]) |
| Word recovery against the PDF text layer (mean, %) | 97.69 | 97.67 | TIE (ratio 1.000 [1.000, 1.000]) |
| Same-engine region agreement between runs (mean, %) | 99.83 | 98.24 | no verdict |
| Scriptor against vLLM region agreement, same pair index (mean, %) | 96.29 | | no verdict |
| Length-limit finishes ("runaways") | 0 | 90 | no verdict |
| Memory used while running (MemAvailable drop, p50 MiB) | 80,105 | 78,415 | TIE (ratio 1.022 [1.020, 1.025]) |

Accuracy is scored against the text layer embedded in the example PDFs (958 scorable regions, text-layer documents only). The accuracy verdict is capped at TIE by design, so a WIN cannot be read as a quality claim. Scriptor and vLLM do not produce byte-identical text: 96.3% of regions agree between the engines, and Scriptor agrees with itself across runs more often (99.83%) than vLLM does (98.24%). The 0 and 90 length-limit finishes are out of about 46,500 requests per engine.

## Limitations

- One machine, one GPU, one session. The results do not transfer to other hardware.
- The 15 documents are a fixed set, not a random sample. Per-document intervals reflect run-to-run variation only.
- With 10 runs per engine, the p99 of a per-run metric is close to the maximum and its interval is wide.
- Percentiles of per-request metrics are bucket-interpolated and as coarse as the bucket widths; each engine reports its own histograms with its own clock.
- The first boot of each engine had a cold page cache and its run is kept in the latency statistics. A page-cold boot was not measured.
- The ten-run comparison used a separate compatible bridge. Scriptor's built-in bridge was checked with three runs only, against Scriptor's own ten runs, not against vLLM.
- Throughput in pages per hour is derived from wall time for 368 pages.
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
2. Start one engine at a time at a 0.60 GPU memory share. For Scriptor, use the `scriptor serve` command in [Getting Started](serving.md#getting-started) with `--num-speculative-tokens 4 --max-num-seqs 32`. For vLLM, use the flags in the setup table.
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
