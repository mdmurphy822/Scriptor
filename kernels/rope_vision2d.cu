// Vision attention preparation: per-head q/k RMSNorm fused with 2-D axial NeoX RoPE.
#include "common.cuh"

constexpr int VISION_HEAD_DIM = 64;
constexpr int VISION_HALF = VISION_HEAD_DIM / 2;
constexpr int VISION_AXIS_PAIRS = VISION_HALF / 2;

// `qkv` is `[tokens, 3, heads, 64]` (q, k, v concatenated per token). For each q
// and k head: RMSNorm over its 64 dims with `q_w` / `k_w` (`[64]`, shared by all
// heads), then rotation of pairs (j, j + 32): pairs [0, 16) by the patch row
// `positions[token * 2]`, pairs [16, 32) by the patch column
// `positions[token * 2 + 1]`, with per-axis frequency `theta^(-2k/32)`. The v
// section is untouched. Grid: one block per token; one warp per head.
__device__ __forceinline__ void vision_norm_rope(
    __nv_bfloat16* qkv, const __nv_bfloat16* q_w, const __nv_bfloat16* k_w,
    const int* positions, int heads, float theta, float eps, int first_head) {
    const long token = blockIdx.x;
    const int lane = threadIdx.x % SCRIPTOR_WARP_SIZE;
    const int warp = threadIdx.x / SCRIPTOR_WARP_SIZE;
    const int warps = blockDim.x / SCRIPTOR_WARP_SIZE;
    const int* pos = positions + token * 2;
    const int axis = lane < VISION_AXIS_PAIRS ? 0 : 1;
    const int k_index = lane % VISION_AXIS_PAIRS;
    const float freq = powf(theta, -2.0f * (float)k_index / (float)VISION_HALF);
    float sn, cs;
    sincosf((float)pos[axis] * freq, &sn, &cs);

    for (int h = first_head + warp; h < 2 * heads; h += warps) {
        const bool is_k = h >= heads;
        const __nv_bfloat16* w = is_k ? k_w : q_w;
        __nv_bfloat16* head = qkv + (token * 3 * heads + h) * VISION_HEAD_DIM;
        float a = scriptor_to_f32(head[lane]);
        float b = scriptor_to_f32(head[lane + VISION_HALF]);
        const float ss = scriptor_warp_sum(a * a + b * b);
        const float inv = rsqrtf(ss / (float)VISION_HEAD_DIM + eps);
        a = scriptor_to_f32(scriptor_to_bf16(a * inv * scriptor_to_f32(w[lane])));
        b = scriptor_to_f32(scriptor_to_bf16(b * inv * scriptor_to_f32(w[lane + VISION_HALF])));
        head[lane] = scriptor_to_bf16(a * cs - b * sn);
        head[lane + VISION_HALF] = scriptor_to_bf16(a * sn + b * cs);
    }
}

extern "C" __global__ void scriptor_vision_qk_norm_rope_bf16(
    __nv_bfloat16* qkv, const __nv_bfloat16* q_w, const __nv_bfloat16* k_w,
    const int* positions, int heads, float theta, float eps) {
    vision_norm_rope(qkv, q_w, k_w, positions, heads, theta, eps, 0);
}

// As `scriptor_vision_qk_norm_rope_bf16` over the k heads only; the q section is untouched.
extern "C" __global__ void scriptor_vision_k_norm_rope_bf16(
    __nv_bfloat16* qkv, const __nv_bfloat16* q_w, const __nv_bfloat16* k_w,
    const int* positions, int heads, float theta, float eps) {
    vision_norm_rope(qkv, q_w, k_w, positions, heads, theta, eps, heads);
}
