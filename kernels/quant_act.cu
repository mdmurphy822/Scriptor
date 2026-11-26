// Activation quantisation for the fp8 GEMM path.
//
// * scriptor_quant_act_fp8_token: per-token dynamic fp8 e4m3 quantisation of a bf16
//   activation matrix with one f32 scale per row.
// * scriptor_fill_f32: writes one f32 value to every element of a vector.
#include <cuda_fp8.h>

#include "common.cuh"

namespace {

constexpr float kE4m3Max = 448.0f;

__device__ __forceinline__ uint8_t scriptor_float_to_e4m3(float v) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

}  // namespace

// For each row r of x [rows, k]: s = amax(x[r, :]) / 448 (1 when the row is all
// zero), xq[r, c] = e4m3(x[r, c] / s), scales[r] = s. One block per row; the
// dequantised value is xq * scales[r].
extern "C" __global__ void scriptor_quant_act_fp8_token(const __nv_bfloat16* __restrict__ x,
                                                  uint8_t* __restrict__ xq,
                                                  float* __restrict__ scales, int k) {
    __shared__ float scratch[32];
    const size_t row = blockIdx.x;
    const __nv_bfloat16* xr = x + row * static_cast<size_t>(k);
    uint8_t* qr = xq + row * static_cast<size_t>(k);

    float amax = 0.0f;
    for (int c = threadIdx.x; c < k; c += blockDim.x) amax = fmaxf(amax, fabsf(scriptor_to_f32(xr[c])));
    amax = scriptor_warp_max(amax);
    const int lane = threadIdx.x % SCRIPTOR_WARP_SIZE;
    const int warp = threadIdx.x / SCRIPTOR_WARP_SIZE;
    const int warps = (blockDim.x + SCRIPTOR_WARP_SIZE - 1) / SCRIPTOR_WARP_SIZE;
    if (lane == 0) scratch[warp] = amax;
    __syncthreads();
    float total = lane < warps ? scratch[lane] : 0.0f;
    total = scriptor_warp_max(total);

    const float scale = total > 0.0f ? total / kE4m3Max : 1.0f;
    const float inv = 1.0f / scale;
    for (int c = threadIdx.x; c < k; c += blockDim.x) qr[c] = scriptor_float_to_e4m3(scriptor_to_f32(xr[c]) * inv);
    if (threadIdx.x == 0) scales[row] = scale;
}

// dst[i] = value for i in [0, n).
extern "C" __global__ void scriptor_fill_f32(float* __restrict__ dst, float value, long long n) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = value;
}
