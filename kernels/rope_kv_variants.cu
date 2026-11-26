// Variants of scriptor_fused_rope_kv_split_bf16 (qkv_split.cu) with identical per-element
// arithmetic: SHARED_TRIG computes each rotary pair's sin/cos once per token block into shared
// memory instead of once per head, and SKIP_SCRATCH leaves the standalone k / v rows unwritten.
#include "common.cuh"

// Rotates the query and key heads of token `blockIdx.x` of `qkv` (`[tokens, (q_heads + 2 *
// k_heads) * head_dim]`) into `q`, and writes each rotated key row and the matching value row
// to flat slot `slots[token]` of `k_cache` / `v_cache` (`[slot][kv_head][head_dim]`; negative
// slots skip the cache writes only). Unless SKIP_SCRATCH, the rotated keys and the values are
// also written to `k` / `v` (`[tokens, k_heads * head_dim]`); with SKIP_SCRATCH those pointers
// are never dereferenced. With SHARED_TRIG, `trig` is dynamic shared memory holding
// `2 * head_dim / 2` floats.
template <bool SHARED_TRIG, bool SKIP_SCRATCH>
__device__ __forceinline__ void scriptor_rope_kv_variant(
    const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* k, __nv_bfloat16* v, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    const int* positions, const int* slots, int q_heads, int k_heads, int head_dim, float theta,
    int s0, int s1, float* trig) {
    const long token = blockIdx.x;
    const long slot = slots[token];
    const int pairs = head_dim / 2;
    const int total = (q_heads + k_heads) * pairs;
    const int row_elems = k_heads * head_dim;
    const int q_dim = q_heads * head_dim;
    const long stride = (long)q_dim + 2 * row_elems;
    const __nv_bfloat16* src = qkv + token * stride;
    const int* pos = positions + token * 3;
    if (SHARED_TRIG) {
        float* sin_row = trig;
        float* cos_row = trig + pairs;
        for (int pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
            const int component = pair < s0 ? 0 : (pair < s0 + s1 ? 1 : 2);
            const float freq = powf(theta, -2.0f * (float)pair / (float)head_dim);
            float sn, cs;
            sincosf((float)pos[component] * freq, &sn, &cs);
            sin_row[pair] = sn;
            cos_row[pair] = cs;
        }
        __syncthreads();
    }
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int head = idx / pairs;
        const int pair = idx % pairs;
        float sn, cs;
        if (SHARED_TRIG) {
            sn = trig[pair];
            cs = trig[pairs + pair];
        } else {
            const int component = pair < s0 ? 0 : (pair < s0 + s1 ? 1 : 2);
            const float freq = powf(theta, -2.0f * (float)pair / (float)head_dim);
            sincosf((float)pos[component] * freq, &sn, &cs);
        }
        const bool is_key = head >= q_heads;
        const long off = (long)head * head_dim + 2 * pair;
        const float a = scriptor_to_f32(src[off]);
        const float b = scriptor_to_f32(src[off + 1]);
        const __nv_bfloat16 ra = scriptor_to_bf16(a * cs - b * sn);
        const __nv_bfloat16 rb = scriptor_to_bf16(a * sn + b * cs);
        if (is_key) {
            if (!SKIP_SCRATCH) {
                const long o = token * row_elems + (long)(head - q_heads) * head_dim + 2 * pair;
                k[o] = ra;
                k[o + 1] = rb;
            }
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
        if (!SKIP_SCRATCH) v[token * row_elems + i] = vs[i];
        if (slot >= 0) v_cache[slot * row_elems + i] = vs[i];
    }
}

#define SCRIPTOR_ROPE_KV_ENTRY(NAME, SHARED, SKIP)                                              \
    extern "C" __global__ void NAME(                                                            \
        const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ q,                   \
        __nv_bfloat16* k, __nv_bfloat16* v, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,     \
        const int* positions, const int* slots, int q_heads, int k_heads, int head_dim,         \
        float theta, int s0, int s1) {                                                          \
        extern __shared__ float trig[];                                                         \
        scriptor_rope_kv_variant<SHARED, SKIP>(qkv, q, k, v, k_cache, v_cache, positions,       \
                                               slots, q_heads, k_heads, head_dim, theta, s0,    \
                                               s1, trig);                                       \
    }

// Shared sin/cos; standalone k / v written. Launch with `2 * (head_dim / 2) * 4` dynamic bytes.
SCRIPTOR_ROPE_KV_ENTRY(scriptor_rope_kv_split_shared_bf16, true, false)
// Per-head sin/cos; standalone k / v skipped (pure decode).
SCRIPTOR_ROPE_KV_ENTRY(scriptor_rope_kv_split_decode_bf16, false, true)
// Shared sin/cos; standalone k / v skipped (pure decode). Dynamic shared bytes as the first.
SCRIPTOR_ROPE_KV_ENTRY(scriptor_rope_kv_split_shared_decode_bf16, true, true)
