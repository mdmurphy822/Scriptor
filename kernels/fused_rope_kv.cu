// GPT-J (interleaved-pair) M-RoPE fused with the paged KV append.
#include "common.cuh"

// Rotates `q` (`[tokens, q_heads, head_dim]`) and `k` (`[tokens, k_heads, head_dim]`) in place
// exactly as scriptor_mrope_gptj_bf16 does (pair `i` rotates elements (2i, 2i+1) by
// `position * theta^(-2i/head_dim)`; pairs [0, s0) read position component 0, [s0, s0 + s1)
// component 1, the rest component 2), and writes each rotated key row and the matching value
// row to flat slot `slots[token]` (= block * block_size + offset) of `k_cache` / `v_cache`,
// each laid out `[slot][kv_head][head_dim]`. Negative slots skip the cache writes only.
// Grid: one block per token.
extern "C" __global__ void scriptor_fused_rope_kv_bf16(
    __nv_bfloat16* q, __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache, const int* positions, const int* slots, int q_heads, int k_heads,
    int head_dim, float theta, int s0, int s1) {
    const long token = blockIdx.x;
    const long slot = slots[token];
    const int pairs = head_dim / 2;
    const int total = (q_heads + k_heads) * pairs;
    const int row_elems = k_heads * head_dim;
    const int* pos = positions + token * 3;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int head = idx / pairs;
        const int pair = idx % pairs;
        const int component = pair < s0 ? 0 : (pair < s0 + s1 ? 1 : 2);
        const float freq = powf(theta, -2.0f * (float)pair / (float)head_dim);
        float sn, cs;
        sincosf((float)pos[component] * freq, &sn, &cs);
        const bool is_key = head >= q_heads;
        __nv_bfloat16* row = is_key ? k + (token * k_heads + (head - q_heads)) * head_dim
                                    : q + (token * q_heads + head) * head_dim;
        const float a = scriptor_to_f32(row[2 * pair]);
        const float b = scriptor_to_f32(row[2 * pair + 1]);
        const __nv_bfloat16 ra = scriptor_to_bf16(a * cs - b * sn);
        const __nv_bfloat16 rb = scriptor_to_bf16(a * sn + b * cs);
        row[2 * pair] = ra;
        row[2 * pair + 1] = rb;
        if (is_key && slot >= 0) {
            __nv_bfloat16* dst = k_cache + slot * row_elems + (head - q_heads) * head_dim;
            dst[2 * pair] = ra;
            dst[2 * pair + 1] = rb;
        }
    }
    if (slot < 0) return;
    if ((row_elems & 7) == 0) {
        const int vecs = row_elems / 8;
        const uint4* vs = reinterpret_cast<const uint4*>(v + token * row_elems);
        uint4* vd = reinterpret_cast<uint4*>(v_cache + slot * row_elems);
        for (int i = threadIdx.x; i < vecs; i += blockDim.x) vd[i] = vs[i];
    } else {
        for (int i = threadIdx.x; i < row_elems; i += blockDim.x)
            v_cache[slot * row_elems + i] = v[token * row_elems + i];
    }
}
