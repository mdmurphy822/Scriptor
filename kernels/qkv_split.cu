// Splits the fused query/key/value projection output into separate q, k and v rows.
#include "common.cuh"

// `qkv` is `[tokens, q_dim + 2 * kv_dim]` (q, k, v side by side per token); writes `q`
// (`[tokens, q_dim]`), `k` and `v` (`[tokens, kv_dim]`). Grid: one block per token.
extern "C" __global__ void scriptor_split_qkv_bf16(const __nv_bfloat16* __restrict__ qkv,
                                             __nv_bfloat16* __restrict__ q,
                                             __nv_bfloat16* __restrict__ k,
                                             __nv_bfloat16* __restrict__ v, int q_dim,
                                             int kv_dim) {
    const long token = blockIdx.x;
    const long stride = (long)q_dim + 2 * kv_dim;
    const __nv_bfloat16* src = qkv + token * stride;
    if (((q_dim | kv_dim) & 7) == 0) {
        const uint4* s = reinterpret_cast<const uint4*>(src);
        const int qv = q_dim / 8, kvv = kv_dim / 8;
        uint4* qd = reinterpret_cast<uint4*>(q + token * q_dim);
        uint4* kd = reinterpret_cast<uint4*>(k + token * kv_dim);
        uint4* vd = reinterpret_cast<uint4*>(v + token * kv_dim);
        for (int i = threadIdx.x; i < qv + 2 * kvv; i += blockDim.x) {
            if (i < qv) qd[i] = s[i];
            else if (i < qv + kvv) kd[i - qv] = s[i];
            else vd[i - qv - kvv] = s[i];
        }
    } else {
        for (int i = threadIdx.x; i < stride; i += blockDim.x) {
            if (i < q_dim) q[token * q_dim + i] = src[i];
            else if (i < q_dim + kv_dim) k[token * kv_dim + i - q_dim] = src[i];
            else v[token * kv_dim + i - q_dim - kv_dim] = src[i];
        }
    }
}

// Splits the fused projection output `qkv` (`[tokens, (q_heads + 2 * k_heads) * head_dim]`)
// into `q`, `k` and `v` rows, rotating `q` and `k` exactly as scriptor_fused_rope_kv_bf16 does,
// and writes each rotated key row and the matching value row to flat slot `slots[token]` of
// `k_cache` / `v_cache` (`[slot][kv_head][head_dim]`; negative slots skip the cache writes
// only). Grid: one block per token.
extern "C" __global__ void scriptor_fused_rope_kv_split_bf16(
    const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache, const int* positions, const int* slots, int q_heads, int k_heads,
    int head_dim, float theta, int s0, int s1) {
    const long token = blockIdx.x;
    const long slot = slots[token];
    const int pairs = head_dim / 2;
    const int total = (q_heads + k_heads) * pairs;
    const int row_elems = k_heads * head_dim;
    const int q_dim = q_heads * head_dim;
    const long stride = (long)q_dim + 2 * row_elems;
    const __nv_bfloat16* src = qkv + token * stride;
    const int* pos = positions + token * 3;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int head = idx / pairs;
        const int pair = idx % pairs;
        const int component = pair < s0 ? 0 : (pair < s0 + s1 ? 1 : 2);
        const float freq = powf(theta, -2.0f * (float)pair / (float)head_dim);
        float sn, cs;
        sincosf((float)pos[component] * freq, &sn, &cs);
        const bool is_key = head >= q_heads;
        const long off = (long)head * head_dim + 2 * pair;
        const float a = scriptor_to_f32(src[off]);
        const float b = scriptor_to_f32(src[off + 1]);
        const __nv_bfloat16 ra = scriptor_to_bf16(a * cs - b * sn);
        const __nv_bfloat16 rb = scriptor_to_bf16(a * sn + b * cs);
        if (is_key) {
            const long o = token * row_elems + (long)(head - q_heads) * head_dim + 2 * pair;
            k[o] = ra;
            k[o + 1] = rb;
            if (slot >= 0) {
                __nv_bfloat16* dst = k_cache + slot * row_elems + (head - q_heads) * head_dim;
                dst[2 * pair] = ra;
                dst[2 * pair + 1] = rb;
            }
        } else {
            const long o = token * q_dim + off;
            q[o] = ra;
            q[o + 1] = rb;
        }
    }
    const __nv_bfloat16* vs = src + q_dim + row_elems;
    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
        v[token * row_elems + i] = vs[i];
        if (slot >= 0) v_cache[slot * row_elems + i] = vs[i];
    }
}
