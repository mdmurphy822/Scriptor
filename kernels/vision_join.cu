// Vision residual join: projection bias, residual add and RMSNorm in one pass per row.
#include "common.cuh"

constexpr int kJoinChunk = 8;       // elements per 16-byte vector
constexpr int kJoinMaxChunks = 4;   // vectors held per thread

// One block per row of `n` elements (`n` a multiple of 8, at most 4 * 8 * blockDim.x).
//   y = bf16(x + bias)            (skipped when `bias` is null)
//   sum = bf16(y + residual)      (stored to `sum_out`)
//   out = sum / sqrt(mean(sum^2) + eps) * w
// `x` is the bias-free GEMM output rounded to bf16. `sum_out` and `out` must not alias `x`.
extern "C" __global__ void scriptor_vision_bias_add_rmsnorm_bf16(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual, const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ sum_out, __nv_bfloat16* __restrict__ out, int n, float eps) {
    __shared__ float scratch[32];
    const long base = (long)blockIdx.x * n;
    const int chunks = n / kJoinChunk;

    float sum[kJoinMaxChunks][kJoinChunk];
    float acc = 0.0f;
#pragma unroll
    for (int c = 0; c < kJoinMaxChunks; ++c) {
        const int idx = threadIdx.x + c * blockDim.x;
        if (idx < chunks) {
            const long off = base + (long)idx * kJoinChunk;
            const uint4 xv = *reinterpret_cast<const uint4*>(x + off);
            const uint4 rv = *reinterpret_cast<const uint4*>(residual + off);
            const __nv_bfloat16* xe = reinterpret_cast<const __nv_bfloat16*>(&xv);
            const __nv_bfloat16* re = reinterpret_cast<const __nv_bfloat16*>(&rv);
            uint4 bv = make_uint4(0, 0, 0, 0);
            if (bias != nullptr)
                bv = *reinterpret_cast<const uint4*>(bias + (long)idx * kJoinChunk);
            const __nv_bfloat16* be = reinterpret_cast<const __nv_bfloat16*>(&bv);
            uint4 sv;
            __nv_bfloat16* se = reinterpret_cast<__nv_bfloat16*>(&sv);
#pragma unroll
            for (int e = 0; e < kJoinChunk; ++e) {
                float y = scriptor_to_f32(xe[e]);
                if (bias != nullptr) y = scriptor_to_f32(scriptor_to_bf16(y + scriptor_to_f32(be[e])));
                se[e] = scriptor_to_bf16(y + scriptor_to_f32(re[e]));
                sum[c][e] = scriptor_to_f32(se[e]);
                acc += sum[c][e] * sum[c][e];
            }
            *reinterpret_cast<uint4*>(sum_out + off) = sv;
        }
    }
    const float total = scriptor_block_sum(acc, scratch);
    const float inv = rsqrtf(total / (float)n + eps);
#pragma unroll
    for (int c = 0; c < kJoinMaxChunks; ++c) {
        const int idx = threadIdx.x + c * blockDim.x;
        if (idx < chunks) {
            const uint4 wv = *reinterpret_cast<const uint4*>(w + (long)idx * kJoinChunk);
            const __nv_bfloat16* we = reinterpret_cast<const __nv_bfloat16*>(&wv);
            uint4 ov;
            __nv_bfloat16* oe = reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
            for (int e = 0; e < kJoinChunk; ++e)
                oe[e] = scriptor_to_bf16(sum[c][e] * inv * scriptor_to_f32(we[e]));
            *reinterpret_cast<uint4*>(out + base + (long)idx * kJoinChunk) = ov;
        }
    }
}
