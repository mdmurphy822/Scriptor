// LayerNorm (weight, bias) followed by erf-GELU. bf16 activations, fp32 math.
#include "common.cuh"

// One block per row of `n` elements; `out` may alias `x`.
extern "C" __global__ void scriptor_layernorm_gelu_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* b,
    __nv_bfloat16* out, int n, float eps) {
    __shared__ float scratch[32];
    const long base = (long)blockIdx.x * n;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += scriptor_to_f32(x[base + i]);
    const float mean = scriptor_block_sum(acc, scratch) / (float)n;
    acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float d = scriptor_to_f32(x[base + i]) - mean;
        acc += d * d;
    }
    const float inv = rsqrtf(scriptor_block_sum(acc, scratch) / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float y = (scriptor_to_f32(x[base + i]) - mean) * inv * scriptor_to_f32(w[i]) + scriptor_to_f32(b[i]);
        out[base + i] = scriptor_to_bf16(scriptor_gelu_erf(y));
    }
}
