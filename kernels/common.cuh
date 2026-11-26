// Shared device helpers for the scriptor kernels.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define SCRIPTOR_WARP_SIZE 32
#define SCRIPTOR_FULL_MASK 0xffffffffu

// Converts a bf16 value to fp32.
__device__ __forceinline__ float scriptor_to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }

// Rounds an fp32 value to bf16 (round-to-nearest-even).
__device__ __forceinline__ __nv_bfloat16 scriptor_to_bf16(float v) { return __float2bfloat16_rn(v); }

// Sums `v` across the 32 lanes of a warp.
__device__ __forceinline__ float scriptor_warp_sum(float v) {
#pragma unroll
    for (int offset = SCRIPTOR_WARP_SIZE / 2; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(SCRIPTOR_FULL_MASK, v, offset);
    return v;
}

// Maximum of `v` across the 32 lanes of a warp.
__device__ __forceinline__ float scriptor_warp_max(float v) {
#pragma unroll
    for (int offset = SCRIPTOR_WARP_SIZE / 2; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(SCRIPTOR_FULL_MASK, v, offset));
    return v;
}

// Sums `v` across the block; `scratch` holds one float per warp (at most 32).
// Every thread receives the block total.
__device__ __forceinline__ float scriptor_block_sum(float v, float* scratch) {
    const int lane = threadIdx.x % SCRIPTOR_WARP_SIZE;
    const int warp = threadIdx.x / SCRIPTOR_WARP_SIZE;
    const int warps = (blockDim.x + SCRIPTOR_WARP_SIZE - 1) / SCRIPTOR_WARP_SIZE;
    v = scriptor_warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = lane < warps ? scratch[lane] : 0.0f;
    total = scriptor_warp_sum(total);
    __syncthreads();
    return total;
}

// Exact (erf) GELU.
__device__ __forceinline__ float scriptor_gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

// SiLU: x * sigmoid(x).
__device__ __forceinline__ float scriptor_silu(float x) { return x / (1.0f + __expf(-x)); }
