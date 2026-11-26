// SiLU-and-mul. bf16 activations, fp32 math.
#include "common.cuh"

// One block per row. `gate_up` is `[rows, 2 * width]` with the gate in the
// first `width` columns and the up branch in the last `width`;
// `out[row, i] = silu(gate[i]) * up[i]`.
extern "C" __global__ void scriptor_silu_mul_fused_bf16(
    const __nv_bfloat16* gate_up, __nv_bfloat16* out, int width) {
    const long in_base = (long)blockIdx.x * 2 * width;
    const long out_base = (long)blockIdx.x * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float g = scriptor_to_f32(gate_up[in_base + i]);
        const float u = scriptor_to_f32(gate_up[in_base + width + i]);
        out[out_base + i] = scriptor_to_bf16(scriptor_silu(g) * u);
    }
}

// Elementwise `out[i] = silu(gate[i]) * up[i]` over `count` elements.
extern "C" __global__ void scriptor_silu_mul_bf16(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out, long count) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = scriptor_to_bf16(scriptor_silu(scriptor_to_f32(gate[i])) * scriptor_to_f32(up[i]));
}
