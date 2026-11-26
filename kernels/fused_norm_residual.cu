// Sandwich-norm residual join: post-sublayer RMSNorm, residual add and the next RMSNorm in
// one launch. bf16 activations, fp32 math; every intermediate is rounded to bf16 exactly where
// the separate rmsnorm / add-rmsnorm kernels round it.
#include "common.cuh"

// One block per row of `n` elements, with the block width the standalone rmsnorm kernels use.
//   y     = bf16(x * rsqrt(mean(x^2) + eps) * w_post)
//   sum   = bf16(residual + y)            (stored to `sum_out`)
//   out   = bf16(sum * rsqrt(mean(sum^2) + eps) * w_next)
// `sum_out` may alias `residual`; `out` must not alias `x`.
extern "C" __global__ void scriptor_norm_add_norm_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* residual, const __nv_bfloat16* w_post,
    const __nv_bfloat16* w_next, __nv_bfloat16* sum_out, __nv_bfloat16* out, int n, float eps) {
    __shared__ float scratch[32];
    const long base = (long)blockIdx.x * n;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = scriptor_to_f32(x[base + i]);
        acc += v * v;
    }
    const float inv_post = rsqrtf(scriptor_block_sum(acc, scratch) / (float)n + eps);
    acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const __nv_bfloat16 y = scriptor_to_bf16(scriptor_to_f32(x[base + i]) * inv_post * scriptor_to_f32(w_post[i]));
        const __nv_bfloat16 s = scriptor_to_bf16(scriptor_to_f32(residual[base + i]) + scriptor_to_f32(y));
        sum_out[base + i] = s;
        const float v = scriptor_to_f32(s);
        acc += v * v;
    }
    const float inv_next = rsqrtf(scriptor_block_sum(acc, scratch) / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[base + i] = scriptor_to_bf16(scriptor_to_f32(sum_out[base + i]) * inv_next * scriptor_to_f32(w_next[i]));
    }
}
