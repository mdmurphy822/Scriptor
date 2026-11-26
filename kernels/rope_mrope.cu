// GPT-J (interleaved-pair) M-RoPE with 3-D positions read from a device buffer.
#include "common.cuh"

// Rotates `q` (`[tokens, q_heads, head_dim]`) and `k` (`[tokens, k_heads, head_dim]`)
// in place. Pair `i` rotates elements (2i, 2i+1) by `position * theta^(-2i/head_dim)`;
// pairs [0, s0) read component 0 of `positions[token * 3 ..]`, pairs [s0, s0 + s1)
// component 1, the rest component 2. Grid: one block per token.
extern "C" __global__ void scriptor_mrope_gptj_bf16(
    __nv_bfloat16* q, __nv_bfloat16* k, const int* positions, int q_heads, int k_heads,
    int head_dim, float theta, int s0, int s1) {
    const long token = blockIdx.x;
    const int pairs = head_dim / 2;
    const int total = (q_heads + k_heads) * pairs;
    const int* pos = positions + token * 3;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int head = idx / pairs;
        const int pair = idx % pairs;
        const int component = pair < s0 ? 0 : (pair < s0 + s1 ? 1 : 2);
        const float freq = powf(theta, -2.0f * (float)pair / (float)head_dim);
        float sn, cs;
        sincosf((float)pos[component] * freq, &sn, &cs);
        __nv_bfloat16* row = head < q_heads
            ? q + (token * q_heads + head) * head_dim
            : k + (token * k_heads + (head - q_heads)) * head_dim;
        const float a = scriptor_to_f32(row[2 * pair]);
        const float b = scriptor_to_f32(row[2 * pair + 1]);
        row[2 * pair] = scriptor_to_bf16(a * cs - b * sn);
        row[2 * pair + 1] = scriptor_to_bf16(a * sn + b * cs);
    }
}
