// Appends new K/V rows into a paged cache addressed by flat slot ids.
#include "common.cuh"

// New rows are packed `[token][kv_head][head_dim]` bf16. Token `t` is written to
// flat slot `slots[t]` (= block * block_size + offset) of both caches, each laid
// out `[slot][kv_head][head_dim]`; negative slots are skipped. Grid: one block per
// token. `row_elems = kv_heads * head_dim` and must be a multiple of 8 for the
// 16-byte path; other widths take the scalar path.
extern "C" __global__ void scriptor_kv_append_bf16(
    __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, const __nv_bfloat16* k_new,
    const __nv_bfloat16* v_new, const int* slots, int row_elems) {
    const long t = blockIdx.x;
    const long slot = slots[t];
    if (slot < 0) return;
    if ((row_elems & 7) == 0) {
        const int vecs = row_elems / 8;
        const uint4* ks = reinterpret_cast<const uint4*>(k_new + t * row_elems);
        const uint4* vs = reinterpret_cast<const uint4*>(v_new + t * row_elems);
        uint4* kd = reinterpret_cast<uint4*>(k_cache + slot * row_elems);
        uint4* vd = reinterpret_cast<uint4*>(v_cache + slot * row_elems);
        for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
            kd[i] = ks[i];
            vd[i] = vs[i];
        }
    } else {
        for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
            k_cache[slot * row_elems + i] = k_new[t * row_elems + i];
            v_cache[slot * row_elems + i] = v_new[t * row_elems + i];
        }
    }
}
