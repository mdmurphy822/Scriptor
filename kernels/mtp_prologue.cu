// MTP prologue: writes both halves of the `[rows, 2 * n]` eh_proj input directly.
//   cat[r, 0..n)   = rmsnorm(embedding(ids[r])) * enorm
//   cat[r, n..2n)  = rmsnorm(hidden[r])         * hnorm
// One block per (row, half) in `blockIdx.x` / `blockIdx.y`. Each block runs the reduction of
// `scriptor_rmsnorm_bf16` unchanged: the same thread-strided fp32 accumulation, the same
// `scriptor_block_sum` tree, `rsqrtf(total / n + eps)`, the same product order and a single bf16
// rounding per output. The embedding half reads the bf16 table row (ids outside `[0, vocab)`
// read as +0 exactly as `scriptor_embedding_gather_bf16` stores them) and normalises it like
// any other row. Launch with the block width of `row_launch(rows, n, 4, 1024)`.
#include "common.cuh"

extern "C" __global__ void scriptor_mtp_prologue_bf16(
    const __nv_bfloat16* table, const int* ids, const __nv_bfloat16* hidden,
    const __nv_bfloat16* enorm, const __nv_bfloat16* hnorm, __nv_bfloat16* cat, int n, int vocab,
    float eps) {
    __shared__ float scratch[32];
    const long row = blockIdx.x;
    const bool embed_half = blockIdx.y == 0;
    const __nv_bfloat16* w = embed_half ? enorm : hnorm;
    const __nv_bfloat16* src = hidden + row * n;
    bool valid = true;
    if (embed_half) {
        const int id = ids[row];
        valid = id >= 0 && id < vocab;
        src = table + (long)(valid ? id : 0) * n;
    }
    __nv_bfloat16* dst = cat + row * 2 * (long)n + (embed_half ? 0 : n);
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = valid ? scriptor_to_f32(src[i]) : scriptor_to_f32(scriptor_to_bf16(0.0f));
        acc += v * v;
    }
    const float total = scriptor_block_sum(acc, scratch);
    const float inv = rsqrtf(total / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = valid ? scriptor_to_f32(src[i]) : scriptor_to_f32(scriptor_to_bf16(0.0f));
        dst[i] = scriptor_to_bf16(v * inv * scriptor_to_f32(w[i]));
    }
}
