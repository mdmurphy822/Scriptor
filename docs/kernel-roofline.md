# Decoder kernel roofline

Per-stage decode timing of the text decoder on GB10 (sm_121), produced by the in-engine
profiler (`TextExecutor::set_profile` / `take_profile`: CUDA events between stages, planned
bytes divided by measured time). Reproduce with

```
SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda --release \
  --test gpu_decode_profile_real -- --ignored --nocapture
```

The test reads the real GLM-OCR checkpoint (revision `2e85a62`, resolved from `$SCRIPTOR_SNAPSHOT_DIR`,
`$HF_HOME` or `~/.cache/huggingface`) and runs 128 decode sequences of 300 cached tokens, one row
each, 16-token blocks, 20 timed steps after 3 warm-up steps. Stage times include event
overhead; planned bytes are the ideal traffic of each stage, so GB/s is a lower bound on achieved
bandwidth for stages that re-read operands from cache.

## Fused kernels

| Fusion | Replaces | Launches |
|---|---|---|
| `fused_rope_kv.cu` | M-RoPE + KV append | 2 -> 1 |
| `fused_norm_residual.cu` (attention join) | post_self_attn norm + residual add + post_attention norm | 2 -> 1 |
| `fused_norm_residual.cu` (MLP join) | post_mlp norm + residual add + next layer input norm | 2 -> 1 |

Non-GEMM launches per layer besides attention: 7 before (rope, kv_append, rms, add_rms,
silu_mul, rms, add_rms), 4 after (fused_rope_kv, norm_add_norm, silu_mul, norm_add_norm). Each fused kernel is bit-identical to the unfused
sequence (`crates/scriptor-kernels/tests/fused_gpu.rs`, `gpu_text_fused_layers_equal_unfused_layers`).
Selection is `TextExecutor::set_fusion(Fusion)`; `Fusion::Fused` is the default.

## Decode step at B = 128, context 300 (shared GPU; ranges over 5 runs)

| Quantity | Value |
|---|---|
| Weight floor (given) | 4.8 ms |
| KV traffic per step (16 layers, planned) | 2.54 GB, 10.6 ms at 240 GB/s |
| Weight floor + KV time | ~15.4 ms |
| Whole `forward` step, unfused | 21.8 - 24.0 ms |
| Whole `forward` step, fused | 21.5 - 24.4 ms |
| Ratio to weight floor + KV time | 1.4x - 1.6x |

Fused versus unfused step times are within run-to-run noise; the measured stage-time saving is
about 0.35 ms of ~22 ms (RoPE + KV append 0.27 -> 0.16 ms, each join 0.20 -> 0.13 ms, summed over
16 layers). The fusions pay off mainly as fewer launches, which matters once the step is
device-resident and graph-captured.

## Per-stage table, fused layers (one run, ms summed over 16 layers per step)

| stage | launches/layer | ms/step | planned MB/step | GB/s |
|---|---|---|---|---|
| qkv_gemm | 3 | 2.76 | 237.0 | 86 |
| rope_kv_append | 1 | 0.16 | 37.7 | 230 |
| attention | 1 | 16.4 | 2541.7 | 155 |
| o_gemm | 1 | 0.57 | 115.3 | 201 |
| attn_join | 1 | 0.13 | 25.3 | 191 |
| gate_up_gemm | 1 | 2.72 | 497.0 | 183 |
| silu_mul | 1 | 0.14 | 56.6 | 394 |
| down_gemm | 1 | 1.22 | 251.7 | 206 |
| mlp_join | 1 | 0.13 | 25.3 | 193 |

Findings: paged decode attention is ~65-70 % of the step at 155-175 GB/s against ~240 GB/s
sustained; the K/V/Q projections run at ~86 GB/s (three small GEMMs with M = 128), the largest
non-attention gap; the fused element-wise stages are small (<= 0.16 ms each).

## Attention rewrite and fused QKV

| Stage | Before | Now |
|---|---|---|
| Paged decode attention | 141-188 GB/s (16.4 ms/step) | 194-205 GB/s (12.8-13.8 ms/step) |
| QKV projection | 3 GEMMs, ~86 GB/s | 1 GEMM, ~90 GB/s |
| Device copy, same runs | - | 218-220 GB/s |

Paged decode attention launches once when `splits == 1`. The fused QKV GEMM (128 x 4096 x 1024)
is latency-bound and is the next target.

Step profile at B = 128 (shared GPU during measurement):

| Context | Attention | Whole step | Whole step vs 4.8 ms weight floor |
|---|---|---|---|
| 300 | 12.8 - 13.8 ms | 20.1 - 22.0 ms | 4.2x - 4.6x |
| 2000 | 81.8 - 86.6 ms | 89 - 94 ms | - |

`SCRIPTOR_PROFILE_CONTEXT` selects the cached context length in `gpu_decode_profile_real`.

## Skinny GEMM and decode attention occupancy

Back-to-back launches over distinct weight copies (no L2 reuse), effective weight GB/s, one CTA
of 4 warps per SM with 128-element K steps against the previous two CTAs per SM with 32-64
element steps; outputs are bit-identical (same K-reduction order per output element).

| Weight shape (n x k) | M = 128 before | M = 128 after | M = 32 before | M = 32 after |
|---|---|---|---|---|
| 59392 x 1536 (LM head) | 166 | 199 | 206 | 214 |
| 9216 x 1536 (gate/up) | 174 | 191 | 209 | 218 |
| 4096 x 1536 (qkv) | 133 | 160 | 199 | 207 |
| 1536 x 4608 (down) | 185 | 192 | 209 | 210 |
| 1536 x 2048 (o proj) | 160 | 160 | 189 | 193 |

Each weight row streams as 256-byte contiguous segments per ring stage; shorter segments cost
10-20 % at M = 128 where the activation tile leaves room for only a 2-3 stage ring. Row groups
sharing a weight stream (two 16-row tiles per warp), token-half CTAs and wider CTAs measured no
better than the 4-warp CTA.

Paged decode attention (`b = 32`, q_len 4): the kernel holds two blocks per SM (230 registers)
instead of three (168). Uniform contexts are unchanged (about 235-245 GB/s); batches with mixed
lengths gain 5 % (224 -> 237 GB/s) and batches with one long sequence 10-25 %, because fewer
long items run concurrently in the tail. Deeper K prefetch (two tiles) measured no gain.

## Vision tower

Per-stage timing of the 24-block vision tower plus adapter on GB10 (sm_121), produced by
`GpuVision::set_profile` / `take_profile` (CUDA events between stages; planned FLOPs and bytes
divided by measured time). Reproduce with

```
SCRIPTOR_GPU_TESTS=1 cargo test -p scriptor-model --features cuda --release \
  --test gpu_vision_profile_real vision_stage -- --ignored --nocapture
```

The test reads the real checkpoint (revision `2e85a62`, found by name under `$SCRIPTOR_SNAPSHOT_DIR`,
`$HF_HOME` or `~/.cache/huggingface`) and runs synthetic text images (dark glyph strokes on a
light page, patchified in merge-window order, one frame repeated twice as Semantik's preprocessing
does): a full page of 120 x 84 patches (10080 patches, 2520 tokens), a half column of 56 x 84, a
small line of 4 x 100 and a batch of 12 mixed crops (4948 patches). Two warm-up batches, five
timed batches; the GPU was held exclusively by a lock during the runs. `vision_gemm_candidate_sweep`
lists the cuBLASLt candidates per tower GEMM.

Ceilings measured in the same run: device copy 217 GB/s (read + write, 256 MiB `memcpy_dtod`); BF16
dense peak 96.7 TFLOP/s, the best cuBLASLt heuristic candidate at 8192 x 8192 x 8192 (best of 5
launches per candidate). That is the BF16 tensor figure assumed below; a roofline against the
vendor's headline number would not be reachable through cuBLASLt on this part.

### Before: full page, unfused (477.8 ms for the batch)

| stage | calls | ms | TFLOP/s | % of BF16 peak | GB/s | % of copy BW |
|---|---|---|---|---|---|---|
| patch_embed | 1 | 0.345 | 70.3 | 73% | 135 | 63% |
| norm | 1 | 0.174 | - | - | 237 | 110% |
| qkv_gemm | 24 | 19.834 | 76.7 | 79% | 108 | 50% |
| qk_norm_rope | 24 | 9.056 | - | - | 219 | 101% |
| qkv_split | 24 | 13.675 | - | - | 217 | 100% |
| attention | 24 | 175.607 | 56.9 | 59% | 11 | 5% |
| proj_gemm | 24 | 6.631 | 76.5 | 79% | 157 | 72% |
| join_attn | 24 | 9.220 | - | - | 215 | 99% |
| gate_up_gemm | 24 | 52.198 | 77.8 | 80% | 93 | 43% |
| silu_mul | 24 | 26.412 | - | - | 225 | 104% |
| down_gemm | 24 | 153.813 | 13.2 | 14% | 17 | 8% |
| join_mlp | 24 | 8.601 | - | - | 230 | 106% |
| merger (6 stages) | 1 | 2.272 | 60-90 | - | - | - |

### After: full page, fused (290.4 ms for the batch)

| stage | calls | ms | TFLOP/s | % of BF16 peak | GB/s | % of copy BW |
|---|---|---|---|---|---|---|
| patch_embed | 1 | 0.337 | 72.0 | 75% | 139 | 64% |
| norm | 1 | 0.174 | - | - | 238 | 110% |
| qkv_gemm | 24 | 19.983 | 76.2 | 79% | 107 | 49% |
| qk_norm_rope | 24 | 9.298 | - | - | 213 | 98% |
| attention (reads `qkv` in place) | 24 | 130.454 | 76.6 | 79% | 15 | 7% |
| proj_gemm | 24 | 6.748 | 75.2 | 78% | 154 | 71% |
| join_attn | 24 | 9.127 | - | - | 217 | 100% |
| gate_up_gemm | 24 | 53.493 | 75.9 | 78% | 91 | 42% |
| silu_mul | 24 | 27.138 | - | - | 219 | 101% |
| down_gemm (no bias epilogue) | 24 | 22.386 | 90.7 | 94% | 120 | 55% |
| join_mlp (bias folded in) | 24 | 8.859 | - | - | 224 | 103% |
| merger (6 stages) | 1 | 2.324 | 65-93 | - | - | - |

Bandwidth percentages use planned bytes (each operand once), so the GEMM and attention rows are
lower bounds; every element-wise stage runs at the 217 GB/s copy ceiling.

### Whole-tower time per batch (ms, five-batch mean)

| Batch | Patches | Unfused | Fused | Speedup |
|---|---|---|---|---|
| full page 120 x 84 | 10080 | 485.0 | 287.1 | 1.69x |
| half column 56 x 84 | 4704 | 121.3 | 103.0 | 1.18x |
| small line 4 x 100 | 400 | 14.1 | 9.5 | 1.49x |
| 12 mixed crops | 4948 | 93.4 | 80.0 | 1.17x |

### Findings and changes

- The down-projection GEMM (K = 4096, N = 1024) with the cuBLASLt bias epilogue ran at 13 TFLOP/s
  (14% of peak) at row counts of roughly 2000 - 13000: every heuristic candidate for that
  epilogue is 5 - 7x slower than the same GEMM without it (5.7 ms against 0.83 ms at 10080 rows;
  workspace size and alignment preferences make no difference). The fused path runs that GEMM
  without the epilogue and applies the bias in the residual-join kernel (`vision_join.cu`:
  `bf16(bf16(x + bias) + residual)`, then RMSNorm, one vectorised pass). The other tower GEMMs keep
  their bias epilogue, which is as fast as none.
- Vision attention was a flash-style `mma.sync` kernel at 57 TFLOP/s whose V fragments were built
  from 16-bit shared-memory loads and whose q / k / v had to be split out of the `qkv` GEMM output
  by three 2-D copies (13.7 ms). `attn_vision.cu` reads q, k and v in place from the packed
  `qkv` rows, uses 128-row blocks, a two-stage `cp.async` K / V pipeline and `ldmatrix`
  (`.trans` for V), reaching 77 TFLOP/s (79% of the measured peak). Its output is bit-identical to
  the previous kernel's (`vision_attention_equals_varlen_on_split_inputs`).
- The remaining non-GEMM stages (rope, joins, SiLU-and-multiply) sit at the copy ceiling. SiLU
  cannot move into a cuBLASLt epilogue (no SWISH epilogue exists, and the gate / up product needs
  both halves), so it stays a separate pass: 9% of the tower.
- `Fusion::Unfused` (and `SCRIPTOR_VISION_FUSION=unfused` for a new executor) keeps the previous kernel
  sequence.

### Equality of fused and unfused

Attention is bit-identical; the fused path differs only in the bias rounding of the down
projection (the GEMM output is rounded to BF16 before the bias is added, where the epilogue added
it in fp32) and in the order of the norm's row reduction. On the tiny fixture the largest
difference is 2.0e-3 with cosine 0.99995. On the real checkpoint the embeddings of the five test
images have cosine 0.9984 - 0.9995 to the unfused ones (largest single-value difference 5 - 30% of
the image's largest magnitude, concentrated in a few outlier channels), and both sets sit equally
far from the f32 reference tower (cosine 0.9975 unfused, 0.9973 fused on a 96-patch image), i.e.
the difference is BF16 noise. End to end through the served engine, four rendered text images
decode to identical greedy text fused and unfused.
