# kernels

CUDA sources compiled by `crates/scriptor-kernels/build.rs` (feature `cuda`) to sm_121
fatbins, one per `*.cu` file, each including `common.cuh`.

Every kernel is specified by an f32 CPU reference in
`crates/scriptor-kernels/src/reference/`; a kernel is correct when it matches its
reference within the bf16 tolerance stated by its test.

| Reference | Contract |
|---|---|
| `norm::rms_norm`, `add_rms_norm`, `head_rms_norm` | plain RMSNorm; fused `sum = x + residual` returning both; per-head over `head_dim` |
| `norm::layer_norm_gelu` | LayerNorm (weight, bias) then erf-GELU |
| `activation::silu_mul`, `silu_mul_fused` | `silu(gate) * up`; fused layout is `[rows, gate \| up]` |
| `rope::mrope_gptj` | interleaved-pair M-RoPE, 3-D positions, sections `[16, 24, 24]` over 64 pairs |
| `rope::rope_vision_2d` | NeoX pairs `(j, j + 32)` over 64 dims; pairs `0..16` rotate by patch row, `16..32` by patch column, per-axis frequencies `theta^(-2k/32)` |
| `attention::varlen_attention` | `cu_seqlens`, causal or not, GQA, explicit scale |
| `attention::paged_decode_attention` | paged KV, `q_len` queries per sequence with in-step causal mask |
| `gemm::gemm_bias` | `C = A * W^T + bias`, row-major |
| `conv::patch_embed`, `downsample_2x2` | Conv3d / Conv2d expressed as GEMMs over the patchify order |
| `preprocess::normalise_patchify` | u8 RGB to fused-affine normalised `[n, 1176]` patches in `(h/28, w/28, 2, 2)` order, frame repeated along `t` |
| `sampling::penalty_argmax` | repetition / presence / frequency penalties, then argmax with top-2 margin |
| `scatter::embedding_gather`, `image_scatter` | token embedding gather; overwrite placeholder rows with image embeddings |
| `scatter::mtp_accept` | longest matching draft prefix plus the verifier's next token |
