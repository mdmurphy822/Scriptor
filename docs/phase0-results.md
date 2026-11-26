# Phase-0 results

Known numbers come from the earlier concurrency sweep (15 documents, 368 pages, share
0.60, bridge unchanged). Rows marked pending are filled in by the user from
`docs/phase0-protocol.md`.

## Known: seat sweep (vLLM v0.33.0)

| Configuration | pages/h | wall (s) | seat ready (s) | peak KV | notes |
|---|---|---|---|---|---|
| max_num_seqs 32 | 2348 | 564 | 136 | 4.7 % | seat full 29-32/32 |
| max_num_seqs 64 | 2222 | 596 | 126 | 6.6 % | |
| max_num_seqs 128 | 2234 | 593 | 125 | 11.7 % | |
| max_num_seqs 256 | stopped | | 131 | | stopped by the user as pointless; per-document completions slower than 32 |
| instrumented baseline, 32 workers 32 seqs | 2462 | 538 | 116 | 4.6 % | seat full 80 %, queue non-empty 83 %, mean running 26, bridge CPU median 44 % / max 314 % |
| NGC vLLM 0.21 image, 32 | 2420 | | | | |

Workload: 4150 regions plus 717 recovery re-reads, about 13.2 requests per page, about
247 prompt tokens and 52 generated tokens per request.

Bridge-side accounting per run: render about 44 s, layout about 174 s (about 0.47 s per
page on the shared GPU), cumulative queue wait about 1160 s, cumulative worker idle
about 7950 s.

## 1. Metrics delta (pending)

Paste the `table` output for each run.

| Run | wall / step (ms) | steps | mean scheduled tokens / step | mean request prefill (ms) | mean request decode (ms) | accept rate by position |
|---|---|---|---|---|---|---|
| seqs 32, no spec | | | | | | |
| seqs 128, no spec | | | | | | |
| seqs 128, MTP k=1 | | | | | | |
| seqs 128, MTP k=2 | | | | | | |
| seqs 128, MTP k=3 | | | | | | |
| seqs 128, MTP k=4 | | | | | | |

Concurrency summary (`series` output) per run: peak running, mean running, share of time
at 90 % of the ceiling.

## 2. Boot time (pending)

| Start state | boot_seconds | notes |
|---|---|---|
| cold | | |
| warm | | |

Earlier sweeps: 116-136 s with compile caches warm; 45-65 s in an earlier sweep with the
NGC image.

## 3. Bridge ceiling (pending)

| Workers / max seqs | wall (s) | pages/h | render_ms | layout_ms | queue_wait_ms | worker_idle_ms |
|---|---|---|---|---|---|---|
| 32 / 32 | | | | | | |
| 128 / 128 | | | | | | |

## 4. GEMM probe (pending)

| Shape role | M | dtype | median ms | TFLOPS | GB/s |
|---|---|---|---|---|---|
| | | | | | |

Sustained BF16 TFLOPS (large ViT shapes): pending. Sustained FP8 TFLOPS: pending.
Sustained decode-shape bandwidth: pending.
