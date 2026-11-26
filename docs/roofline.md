# GLM-OCR workload and roofline

Measured workload, per-page roofline and throughput targets for Scriptor (the GLM-OCR engine) on GB10 (sm_121). Figures scale with the sustained tensor-core rate measured by the GEMM probe.

## Workload, roofline and targets

### 1 Hardware and toolchain

GB10 (`sm_121`), 121.7 GiB unified LPDDR5X (273 GB/s nominal; ~240 GB/s sustained assumed), 20 ARM cores, CUDA 13.0.88 (`nvcc`, cuBLASLt 13, NPP, nvJPEG headers present; **no cuDNN, no CUTLASS** on the host), Rust 1.97.1. Local registry: `cudarc 0.19.10`, `axum 0.8`, `tokio`, `hyper 1`, `tokenizers 0.22`, `image 0.25` + `zune-jpeg 0.5`, `sha2`, `base64`, `rayon`, `memmap2`, `half`, `bytemuck`, `prometheus`, `reqwest`; `minijinja` and a libjpeg-turbo binding are not cached (pinned build-time fetch, as the sister plan allows). Host Python has `jinja2 3.1.2` and `PIL 10.2` (no numpy/torch). `pdftoppm`/`pdfinfo` present. Dense BF16 tensor-core rate of GB10 is **assumed ~125 TFLOPS nominal, ~80 TFLOPS sustained** (FP4-sparse 1 PFLOP marketing ÷ 8); T02 measures it with a cuBLASLt GEMM probe — every compute figure below scales with that number. The probe is `scriptor bench gemm` (`docs/gemm.md`); its measured large-M TFLOPS replaces the assumption once recorded there.

### 2 The workload (measured, 15 example documents, 368 pages)

From `bsweep` replies (`c32-*.reply.json`) and bridge logs: **4150 regions + 717 recovery-band re-reads ≈ 4870 seat requests = 13.2 requests/page**; prompt ≈ 1.2 M tokens → **247 tokens/request (≈ 235 image tokens ≈ 940 patches, a crop of roughly 420×450 px)**; generated ≈ 252 k → **52 tokens/request** (872 k characters, ~3.5 chars/token); 62 % paragraphs, 16 % page furniture, 10 % headings, 3.4 % formulas. The shared text prefix is ~6 tokens (`[gMASK]<sop><|user|>\n<|begin_of_image|>`), so **prompt prefix caching is worth ≈ 0**. Request lifetime ≈ one vision pass + one 247-token prefill + ~52 decode steps.

| vLLM v0.33.0, share 0.60 | pages/h | notes |
|---|---|---|
| max_num_seqs 32 | **2348** | seat full 29–32/32; KV ≤ 4.7 %; GPU util ≈ 95 % (shared with the bridge's PP-DocLayout-V3 ONNX model on the same GPU) |
| max_num_seqs 64 | 2222 | full 64/64; KV ≤ 6.6 % |
| max_num_seqs 128 | 2234 | full ~125/128; KV ≤ 11.7 % |
| max_num_seqs 256 | stopped by the user as pointless (seat ready 131 s; per-document completions visibly slower than s32 — first doc 124 s vs 66 s); sweep ended at s128 | |
| NGC vLLM 0.21 image, 32 | 2420 | |
| instrumented baseline `basec32` | in flight at writing (12/15 documents done at 281 s, same pace as s32); no `mstat`/`loadstat` line yet | |
| boot to `/v1/models` | **116–136 s** (compile caches warm; 45–65 s in an earlier sweep with the NGC image) | |

Bridge-side accounting from the `ocr-bridge: schedule` lines (45 lines = 3 runs × 15 documents, 368 pages each): per run **render ≈ 44 s, layout (PP-DocLayout ONNX on the same GPU) ≈ 174 s ≈ 0.47 s/page of GPU co-tenant time**, cumulative `queue_wait` ≈ 1160 s and `worker_idle` ≈ 7950 s across workers — the bridge's workers spend most of their time waiting on the seat, so at 32–128 the run is **seat-bound, not bridge-bound**; the layout model is nevertheless a real GPU co-tenant that caps the engine's share of the device (and is why the "95 % GPU util" reading overstates the seat's own utilisation).

Throughput is **flat from 32 to 128 concurrent sequences** (and worse at 256) while KV stays under 12 %: the seat is neither KV- nor bandwidth-bound. Doubling the batch should halve decode weight traffic per page; that it does not means the step cost grows with the batch — the signature of per-step host work proportional to B (vLLM's penalty path rebuilds prompt/output token tensors from Python lists every step when `repetition_penalty` is set; scheduler and input-preparation Python; multimodal preprocessing in the API-server process; encoder-budget scheduling) plus vision-encoder efficiency. The instrumented baseline (step count, prefill/decode seconds, bridge CPU) was not in `results-c.txt` at writing; T02 captures it before any kernel work is judged.

### 3 Roofline per page (13.2 requests, 3260 prompt tokens, 685 generated tokens)

Compute (vision dominates):

| Crop (patches → tokens) | ViT linear | ViT attention (N²) | other | total @80 TFLOPS |
|---|---|---|---|---|
| 400 → 100 | 322 GF | 16 GF | 7 GF | 4.3 ms |
| **920 → 230 (typical)** | 741 GF | 83 GF | 16 GF | **10.5 ms** |
| 2400 → 600 | 1.93 TF | 0.57 TF | 42 GF | 32 ms |
| 8000 → 2000 | 6.4 TF | 6.3 TF | 0.14 TF | 161 ms |
| 24000 → 6000 (Semantik's cap) | 19.3 TF | 56.6 TF | 0.4 TF | 955 ms |

Per page: vision **10.9 TF**, text prefill 3.2 TF (2·491 M·3260), decode 0.8 TF → **≈ 186 ms/page at 80 TFLOPS** (≈ 120 ms if the ViT GEMMs run FP8). Attention is 10 % of ViT work at typical crops and dominates only above ~5000 patches.

Bandwidth (decode): weights 1.164 GB/step → 4.8 ms/step floor regardless of batch (BF16 stays memory-bound until B ≈ 400); KV read per step = B·ctx·64 KB.

| Batch B | decode steps/page | weight bytes/page | KV bytes/page (ctx ≈ 300) | decode floor/page |
|---|---|---|---|---|
| 32 | 21.4 | 24.9 GB → 104 ms | 13.5 GB → 56 ms | 160 ms |
| 64 | 10.7 | 12.5 GB → 52 ms | 56 ms | 108 ms |
| 128 | 5.3 | 6.2 GB → 26 ms | 56 ms | 82 ms |
| 256 | 2.7 | 3.1 GB → 13 ms | 56 ms | 69 ms |

MTP with acceptance a per position and k drafts divides the step count by (1 + accepted tokens/step) and adds 0.254 GB per draft step; at B ≥ 128 the decode floor is already < 30 % of the page budget, so **MTP's throughput value is bounded at ~15–20 % of total** except in the low-concurrency tail (last ~50 s of the 564 s run ran at 1–6 sequences, where it is worth ~1.5–2×).

Per-step host overhead: a Rust scheduler step (block-table and token writes into pinned buffers, one graph launch, one async D2H of sampled ids) is budgeted at **≤ 0.3 ms**; at B = 128 the GPU step is ~8–12 ms, so host overhead is ≤ 3 %. vLLM's per-step Python cost is the quantity T02 measures (`wall/steps`).

**Where the time goes (honest):** vLLM spends **1533 ms of GPU-wall per page** at B = 32 against a roofline of **≈ 280–350 ms/page** (compute ≈ 190 + decode bandwidth 70–160 with no overlap) → ~20–23 % of roofline. The gap is not weight bandwidth (never the limiter here) and not KV capacity; it is step-count × per-step overhead and vision efficiency. Perfect overlap of vision/prefill with decode would push the ceiling to ~190 ms/page → **~19,000 pages/h**, but the bridge itself (page render, layout model sharing the GPU, 717 recovery re-reads, Python-free but 32–128 blocking workers) has its own ceiling that is **unmeasured**; T02 measures it with a mock seat.

### 4 Targets

| Milestone | pages/h (15-doc set, share 0.60, bridge unchanged) | basis |
|---|---|---|
| vLLM baseline | 2348 (32) / 2234 (128) | measured |
| M3 correct engine, batched vision, cuBLASLt GEMMs, Rust scheduler, no graphs | ≥ 3000 (1.3×) | removes Python per-step cost at B = 128–256; eager launches ≈ 1–2 ms/step |
| M5 CUDA-graph decode, fused norms/rope/sampler, GPU preprocessing, vision/prefill overlapped with decode | 4500–6000 | decode floor 70–80 ms + vision 190 ms partly hidden; bridge ceiling permitting |
| M6 MTP (gated) + FP8 ViT GEMMs (gated) | 6000–8000 | −15 % decode steps at B ≥ 128; vision 190 → ~120 ms |
| Roofline | ~10,000–13,000 (no overlap) / ~19,000 (full overlap) | §2.3 |
| Boot to ready (warm page cache / cold NVMe) | **≤ 5 s / ≤ 15 s** vs 125–136 s | 2.65 GB engine file, ~10 captured graphs, cuBLASLt heuristic cache |

Everything above 3000 depends on the bridge ceiling and on the T02 attribution; the plan measures both before M5 kernels are judged.

