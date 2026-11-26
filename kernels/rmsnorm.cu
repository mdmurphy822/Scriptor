// Plain RMSNorm and fused residual-add RMSNorm. bf16 activations, fp32 math.
#include "common.cuh"

// One block per row of `n` elements.
//   out = x / sqrt(mean(x^2) + eps) * w
// `out` may alias `x`.
extern "C" __global__ void scriptor_rmsnorm_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* out, int n, float eps) {
    __shared__ float scratch[32];
    const long base = (long)blockIdx.x * n;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = scriptor_to_f32(x[base + i]);
        acc += v * v;
    }
    const float total = scriptor_block_sum(acc, scratch);
    const float inv = rsqrtf(total / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[base + i] = scriptor_to_bf16(scriptor_to_f32(x[base + i]) * inv * scriptor_to_f32(w[i]));
    }
}

// One block per row of `n` elements.
//   sum = bf16(x + residual)   (stored to `sum_out`)
//   out = sum / sqrt(mean(sum^2) + eps) * w
// `sum_out` and `out` may alias `x` or `residual`; each element is read before
// it is written by the same thread.
extern "C" __global__ void scriptor_add_rmsnorm_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* residual, const __nv_bfloat16* w,
    __nv_bfloat16* sum_out, __nv_bfloat16* out, int n, float eps) {
    __shared__ float scratch[32];
    const long base = (long)blockIdx.x * n;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const __nv_bfloat16 s = scriptor_to_bf16(scriptor_to_f32(x[base + i]) + scriptor_to_f32(residual[base + i]));
        sum_out[base + i] = s;
        const float v = scriptor_to_f32(s);
        acc += v * v;
    }
    const float total = scriptor_block_sum(acc, scratch);
    const float inv = rsqrtf(total / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[base + i] = scriptor_to_bf16(scriptor_to_f32(sum_out[base + i]) * inv * scriptor_to_f32(w[i]));
    }
}
