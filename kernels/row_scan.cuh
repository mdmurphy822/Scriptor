// Shared row-scan contract of the sampling kernels: row parameters, the running top-2 and its
// warp/block reduction, the penalised view of a speculative verification row, and the
// per-virtual-thread scan that every selection kernel runs over a row.
#pragma once

#include "common.cuh"

// Per-row sampling parameters; layout matches `RowParams` in
// crates/scriptor-kernels/src/sampling.rs.
struct ScriptorRowParams {
    uint64_t seed;
    float repetition;
    float presence;
    float frequency;
    float temperature;
    float top_p;
    int32_t top_k;
    uint32_t step;
    uint32_t slot;
};

// Running top-2 of a row: the maximum, the lowest index holding it, and the largest value
// at any other index (the maximum again when it occurs twice).
struct ScriptorTop2 {
    float best;
    uint32_t index;
    float second;
};

#define SCRIPTOR_NO_INDEX 0xffffffffu

// The empty top-2: it leaves any other top-2 unchanged when combined.
__device__ __forceinline__ ScriptorTop2 scriptor_top2_empty() { return {-INFINITY, SCRIPTOR_NO_INDEX, -INFINITY}; }

// Folds the value `v` at index `i` into `t`; indices arrive in ascending order.
__device__ __forceinline__ void scriptor_top2_add(ScriptorTop2& t, float v, uint32_t i) {
    if (v > t.best) {
        t.second = t.best;
        t.best = v;
        t.index = i;
    } else if (v == t.best) {
        if (t.index == SCRIPTOR_NO_INDEX)
            t.index = i;
        else
            t.second = v;
    } else if (v > t.second) {
        t.second = v;
    }
}

// Combines two top-2s over disjoint index sets; commutative and associative.
__device__ __forceinline__ ScriptorTop2 scriptor_top2_merge(const ScriptorTop2& a, const ScriptorTop2& b) {
    ScriptorTop2 r;
    if (a.best > b.best) {
        r.best = a.best;
        r.index = a.index;
        r.second = fmaxf(a.second, b.best);
    } else if (b.best > a.best) {
        r.best = b.best;
        r.index = b.index;
        r.second = fmaxf(b.second, a.best);
    } else {
        r.best = a.best;
        r.index = min(a.index, b.index);
        r.second = a.best;
    }
    return r;
}

// Top-2 across the 32 lanes of a warp.
__device__ __forceinline__ ScriptorTop2 scriptor_warp_top2(ScriptorTop2 t) {
    for (int offset = SCRIPTOR_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        ScriptorTop2 o;
        o.best = __shfl_xor_sync(SCRIPTOR_FULL_MASK, t.best, offset);
        o.index = __shfl_xor_sync(SCRIPTOR_FULL_MASK, t.index, offset);
        o.second = __shfl_xor_sync(SCRIPTOR_FULL_MASK, t.second, offset);
        t = scriptor_top2_merge(t, o);
    }
    return t;
}

// Top-2 across the block; `scratch` holds one entry per warp. Every thread receives the result.
__device__ __forceinline__ ScriptorTop2 scriptor_block_top2(ScriptorTop2 t, ScriptorTop2* scratch) {
    const int lane = threadIdx.x % SCRIPTOR_WARP_SIZE;
    const int warp = threadIdx.x / SCRIPTOR_WARP_SIZE;
    const int warps = (blockDim.x + SCRIPTOR_WARP_SIZE - 1) / SCRIPTOR_WARP_SIZE;
    t = scriptor_warp_top2(t);
    if (lane == 0) scratch[warp] = t;
    __syncthreads();
    ScriptorTop2 total = lane < warps ? scratch[lane] : scriptor_top2_empty();
    total = scriptor_warp_top2(total);
    __syncthreads();
    return total;
}

// Penalty state of one logits row. Row `r` of a sequence group of `q` rows belongs to sequence
// `r / q`, reads that sequence's parameters, and treats the first `r % q` entries of the
// sequence's `k` drafts as already emitted output (`ahead`). `drafts` may be null when `q == 1`.
struct ScriptorRowView {
    ScriptorRowParams p;
    const uint32_t* pb;
    const uint32_t* ob;
    const uint32_t* oc;
    const uint32_t* row_drafts;
    int ahead;
    bool plain;

    // Penalised value of element `i` whose unpenalised value is `v`, given its prompt and
    // output-bitset membership. Per-element output counts are read only under a frequency
    // penalty.
    __device__ __forceinline__ float penalise_bits(int i, float v, bool in_prompt,
                                                   bool in_out_bits) const {
        int extra = 0;
        for (int t = 0; t < ahead; ++t) extra += row_drafts[t] == (uint32_t)i ? 1 : 0;
        const bool in_output = in_out_bits || extra > 0;
        if (p.repetition != 1.0f && (in_prompt || in_output))
            v = v > 0.0f ? v / p.repetition : v * p.repetition;
        const float count = oc ? (float)(oc[i] + (uint32_t)extra) : (in_output ? 1.0f : 0.0f);
        v -= p.frequency * count;
        if (in_output) v -= p.presence;
        return v;
    }

    // Penalised value of element `i` whose unpenalised value is `v`.
    __device__ __forceinline__ float penalise(int i, float v) const {
        return penalise_bits(i, v, (pb[i >> 5] >> (i & 31)) & 1u, (ob[i >> 5] >> (i & 31)) & 1u);
    }

    // Penalised element `i` of a bf16 row.
    __device__ __forceinline__ float at_bf16(const uint16_t* xh, int i) const {
        const float raw = __uint_as_float(((uint32_t)xh[i]) << 16);
        return plain ? raw : penalise(i, raw);
    }
};

// The penalty view of row `row` among `q` rows per sequence.
__device__ __forceinline__ ScriptorRowView scriptor_row_view(
    int row, int q, int vocab, const ScriptorRowParams* params, const uint32_t* prompt_bits,
    const uint32_t* output_bits, const uint32_t* output_counts, int words, const uint32_t* drafts,
    int k) {
    const int seq = row / q;
    ScriptorRowView r;
    r.ahead = row - seq * q;
    r.row_drafts = drafts ? drafts + (size_t)seq * (size_t)k : nullptr;
    r.p = params[seq];
    r.pb = prompt_bits + (size_t)r.p.slot * (size_t)words;
    r.ob = output_bits + (size_t)r.p.slot * (size_t)words;
    r.oc = (output_counts && r.p.frequency != 0.0f)
               ? output_counts + (size_t)r.p.slot * (size_t)vocab
               : nullptr;
    r.plain = r.p.repetition == 1.0f && r.p.frequency == 0.0f && r.p.presence == 0.0f;
    return r;
}

// Top-2 of the elements of a bf16 row owned by virtual thread `first` of a block `stride`
// threads wide. A row whose length is a multiple of 8 and whose base is 16-byte aligned is
// scanned in 16-byte groups of 8 consecutive elements (group `g = first; g += stride`), with
// the bitset words fetched once per group; any other row is scanned element by element
// (`i = first; i += stride`). Elements are folded in ascending index order.
__device__ __forceinline__ ScriptorTop2 scriptor_scan_bf16(const ScriptorRowView& r,
                                                           const uint16_t* xh, int vocab,
                                                           int first, int stride) {
    ScriptorTop2 local = scriptor_top2_empty();
    const bool grouped = (vocab & 7) == 0 && (((uintptr_t)xh) & 15u) == 0;
    if (grouped) {
        const uint4* xv = reinterpret_cast<const uint4*>(xh);
        const int groups = vocab >> 3;
        for (int g = first; g < groups; g += stride) {
            const uint4 w = __ldg(xv + g);
            const uint32_t packed[4] = {w.x, w.y, w.z, w.w};
            const int base = g << 3;
            uint32_t pw = 0u, ow = 0u;
            if (!r.plain) {
                pw = r.pb[base >> 5] >> (base & 31);
                ow = r.ob[base >> 5] >> (base & 31);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t bits = (packed[j >> 1] >> ((j & 1) * 16)) & 0xffffu;
                const float raw = __uint_as_float(bits << 16);
                const float v = r.plain ? raw
                                        : r.penalise_bits(base + j, raw, (pw >> j) & 1u,
                                                          (ow >> j) & 1u);
                scriptor_top2_add(local, v, (uint32_t)(base + j));
            }
        }
    } else {
        for (int i = first; i < vocab; i += stride)
            scriptor_top2_add(local, r.at_bf16(xh, i), (uint32_t)i);
    }
    return local;
}

// As scriptor_scan_bf16 over an fp32 row, which receives the penalised values in place.
__device__ __forceinline__ ScriptorTop2 scriptor_scan_f32(const ScriptorRowView& r, float* x,
                                                          int vocab, int first, int stride) {
    ScriptorTop2 local = scriptor_top2_empty();
    for (int i = first; i < vocab; i += stride) {
        const float v = r.penalise(i, x[i]);
        x[i] = v;
        scriptor_top2_add(local, v, (uint32_t)i);
    }
    return local;
}
