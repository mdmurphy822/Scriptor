# GEMM probe recipe

Purpose: measure the sustained dense tensor-core throughput of the GB10 so every
compute figure in `docs/roofline.md` can be scaled by a measured number instead of
the assumed ~80 TFLOPS (BF16) and the FP8 estimate.

Command (available once T21 lands; runs on the GPU, so the seat must be down):

```
scriptor bench gemm --dtype bf16 --report gemm-bf16.json
scriptor bench gemm --dtype fp8  --report gemm-fp8.json
```

Probe shapes (M x N x K, row-major, cuBLASLt, heuristic cache warmed, 50 warm-up and
200 timed iterations per shape, median reported):

| Role | M | N | K |
|---|---|---|---|
| ViT qkv | 8192, 32768, 98304 | 3072 | 1024 |
| ViT mlp up | 8192, 32768, 98304 | 4096 | 1024 |
| ViT mlp down | 8192, 32768, 98304 | 1024 | 4096 |
| Text gate_up (prefill) | 4096, 16384 | 9216 | 1536 |
| Text down (prefill) | 4096, 16384 | 1536 | 4608 |
| Decode gate_up | 32, 128, 256, 1024 | 9216 | 1536 |
| lm_head | 32, 128, 256, 1024 | 59392 | 1536 |

Record for every shape: median milliseconds, achieved TFLOPS (`2*M*N*K / t`), and the
achieved bandwidth for the decode-sized shapes (`bytes of B / t`). Also record the
clock and power state (`nvidia-smi -q -d CLOCK,POWER` before and after) and whether any
other process held the GPU.

Reading the result: the sustained TFLOPS of the large ViT shapes replace the assumed
80 TFLOPS in `docs/roofline.md` section 2.3; the decode-shape bandwidth replaces the
assumed 240 GB/s. Paste the table into `docs/phase0-results.md`.
