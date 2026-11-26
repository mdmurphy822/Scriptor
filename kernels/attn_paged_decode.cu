// Paged decode attention, head_dim 128, grouped KV heads, split-K.
//
// Grid (kv_heads, splits, sequences); 4 warps per block. Blocks are dispatched in
// descending order of their key range (ties by grid position), so the longest items
// start first and the tail wave holds the shortest; every item computes exactly what
// it computes at its own grid position. Registers and shared memory are bounded so
// `SCRIPTOR_ATTN_RESIDENT_BLOCKS` blocks (default two) are resident per SM: with mixed
// sequence lengths a lower count leaves the tail fewer concurrent long items than a higher
// count would. The query rows of one
// KV head (`group * q_len` of them, at most 16) form one m16 tile, so every KV
// block is read once and serves all query heads of the group and all query rows
// of the sequence. Each warp walks 16-key tiles of its split's key range,
// strided by 4 tiles across the block. Keys are fetched straight into the mma
// B fragments with 16-byte loads (the head dimension is permuted identically in
// the query fragments, which leaves the dot products unchanged); values arrive
// through 16-byte cp.async copies into shared memory. Both are prefetched one
// tile ahead. The four warps merge their running (output, max, sum) in shared
// memory; with one split the block writes the final bf16 output, otherwise one
// partial per split for `scriptor_paged_attn_reduce`.
// Query `j` of a sequence sees positions `0 ..= seq_len - q_len + j`.
#include "common.cuh"

namespace {

constexpr int kHd = 128;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * SCRIPTOR_WARP_SIZE;
constexpr int kTile = 16;
constexpr int kStride = kHd;
constexpr int kMergeRows = 8;
constexpr int kMaxRows = 16;
constexpr float kLog2e = 1.4426950408889634f;

constexpr int kValueBytes = kWarps * 2 * kTile * kStride * 2;
constexpr int kMergeBytes = kWarps * kMergeRows * kHd * 4 + 2 * kWarps * kMaxRows * 4;
constexpr int kSharedBytes = kValueBytes > kMergeBytes ? kValueBytes : kMergeBytes;

// Column of 16-byte chunk `chunk` of value-tile row `row` after the bank swizzle.
__device__ __forceinline__ int swz(int row, int chunk) { return (chunk ^ ((row >> 1) & 7)) * 8; }

__device__ __forceinline__ void mma_16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                          uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ uint32_t pack_pair(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return static_cast<uint32_t>(__bfloat16_as_ushort(lo)) |
           (static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

struct KeyTile {
    uint4 v[2][4];
};

// Loads the keys of the 8-key half `nt` of the tile at `tile` as mma B fragments: `v[p]`
// holds, for key `tile + nt * 8 + g`, dimensions `32p + 8t .. 32p + 8t + 7`.
__device__ __forceinline__ void load_keys(uint4 (&v)[4], const __nv_bfloat16* key_cache,
                                          const int* table, int block_size, int kv_heads,
                                          int kvh, int tile, int nt, int sp_end, int g, int t) {
    const int kpos = min(tile + nt * 8 + g, sp_end - 1);
    const size_t krow = (size_t)table[kpos / block_size] * block_size + kpos % block_size;
    const uint4* kp =
        reinterpret_cast<const uint4*>(key_cache + (krow * kv_heads + kvh) * kHd + 8 * t);
#pragma unroll
    for (int p = 0; p < 4; ++p) v[p] = __ldg(kp + p * 4);
}

// Copies the value tile at `tile` into shared memory; rows past `sp_end` are zero.
__device__ __forceinline__ void load_values(__nv_bfloat16 (*dst)[kStride],
                                            const __nv_bfloat16* value_cache, const int* table,
                                            int block_size, int kv_heads, int kvh, int tile,
                                            int sp_end, int lane) {
    const int c = lane & 15;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int j = 2 * i + (lane >> 4);
        const int pos = tile + j;
        if (pos < sp_end) {
            const size_t row = (size_t)table[pos / block_size] * block_size + pos % block_size;
            cp_async16(&dst[j][swz(j, c)], value_cache + (row * kv_heads + kvh) * kHd + c * 8);
        } else {
            *reinterpret_cast<uint4*>(&dst[j][swz(j, c)]) = make_uint4(0, 0, 0, 0);
        }
    }
}

}  // namespace

// q: [seqs, q_len, q_heads, 128] bf16. Caches: [blocks, block_size, kv_heads, 128] bf16.
// block_tables: [seqs, max_blocks] int32. seq_lens[s] counts cached tokens including
// the q_len new ones. With splits == 1 the block writes `out` ([seqs, q_len, q_heads, 128]
// bf16) directly; otherwise part_o: [seqs, splits, q_len, q_heads, 128] fp32 and
// part_m / part_l: [seqs, splits, q_len, q_heads] fp32 (max in the exp2 domain).
// Blocks per SM the register budget is bounded for.
#ifndef SCRIPTOR_ATTN_RESIDENT_BLOCKS
#define SCRIPTOR_ATTN_RESIDENT_BLOCKS 2
#endif
#ifdef SCRIPTOR_PAGED_ATTN_ORDERED
#define SCRIPTOR_ATTN_KERNEL_NAME scriptor_paged_attn_decode_d128_ordered
#define SCRIPTOR_ATTN_EXTRA_PARAM , const int* __restrict__ pair_order
#else
#define SCRIPTOR_ATTN_KERNEL_NAME scriptor_paged_attn_decode_d128
#define SCRIPTOR_ATTN_EXTRA_PARAM
#endif
// With SCRIPTOR_PAGED_ATTN_ORDERED the kernel takes a trailing `pair_order` ([seqs * splits]
// int32) and block `target` runs pair `pair_order[target]`.
extern "C" __global__ void __launch_bounds__(kThreads, SCRIPTOR_ATTN_RESIDENT_BLOCKS) SCRIPTOR_ATTN_KERNEL_NAME(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache, const int* __restrict__ block_tables,
    int max_blocks, const int* __restrict__ seq_lens, int q_len, int q_heads, int kv_heads,
    int block_size, int splits, int chunk, float scale, float* __restrict__ part_o,
    float* __restrict__ part_m, float* __restrict__ part_l,
    __nv_bfloat16* __restrict__ out SCRIPTOR_ATTN_EXTRA_PARAM) {
    constexpr int kKSteps = kHd / 16;
    constexpr int kDTiles = kHd / 8;

    __shared__ __align__(16) unsigned char smem[kSharedBytes];
    auto sv = reinterpret_cast<__nv_bfloat16(*)[2][kTile][kStride]>(smem);

    int kvh, split, seq;
#ifdef SCRIPTOR_PAGED_ATTN_ORDERED
    {
        const int target = blockIdx.y + gridDim.y * blockIdx.z;
        const int pair = pair_order[target];
        kvh = blockIdx.x;
        split = pair % splits;
        seq = pair / splits;
    }
#else
    __shared__ int slot;
    {
        const int pairs = gridDim.y * gridDim.z;
        const int target = (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) / gridDim.x;
        for (int i = threadIdx.x; i < pairs; i += kThreads) {
            const int wi = min(max(seq_lens[i / splits] - (i % splits) * chunk, 0), chunk);
            int rank = 0;
            for (int j = 0; j < pairs; ++j) {
                const int wj = min(max(seq_lens[j / splits] - (j % splits) * chunk, 0), chunk);
                rank += (wj > wi) | ((wj == wi) & (j < i));
            }
            if (rank == target) slot = i;
        }
        __syncthreads();
        const int pair = slot;
        kvh = blockIdx.x;
        split = pair % splits;
        seq = pair / splits;
    }
#endif
    const int warp = threadIdx.x / SCRIPTOR_WARP_SIZE;
    const int lane = threadIdx.x % SCRIPTOR_WARP_SIZE;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int seq_len = seq_lens[seq];
    const int group = q_heads / kv_heads;
    const int rows = group * q_len;
    const int sp_start = split * chunk;
    const int sp_end = min(sp_start + chunk, seq_len);
    const int* table = block_tables + (size_t)seq * max_blocks;

    const int tile0 = sp_start + warp * kTile;
    constexpr int kTileStride = kWarps * kTile;

    KeyTile kc;
    if (tile0 < sp_end) {
        load_keys(kc.v[0], key_cache, table, block_size, kv_heads, kvh, tile0, 0, sp_end, g, t);
        load_keys(kc.v[1], key_cache, table, block_size, kv_heads, kvh, tile0, 1, sp_end, g, t);
        load_values(sv[warp][0], value_cache, table, block_size, kv_heads, kvh, tile0, sp_end,
                    lane);
    }
    cp_async_commit();

    const int r0 = g, r1 = g + 8;
    const bool ok0 = r0 < rows, ok1 = r1 < rows;
    const int qj0 = ok0 ? r0 / group : 0, qj1 = ok1 ? r1 / group : 0;
    const int h0 = kvh * group + (ok0 ? r0 % group : 0);
    const int h1 = kvh * group + (ok1 ? r1 % group : 0);
    const int lim0 = ok0 ? seq_len - q_len + qj0 : -1;
    const int lim1 = ok1 ? seq_len - q_len + qj1 : -1;
    const float sl2 = scale * kLog2e;

    // Fragment k-slot 2t, 2t+1 (+8) of k-step 2p+h maps to dimensions
    // 32p + 8t + 4h + {0,1} (+{2,3}); keys use the same mapping.
    uint32_t qa[kKSteps][4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        uint4 v0 = make_uint4(0, 0, 0, 0), v1 = make_uint4(0, 0, 0, 0);
        if (ok0)
            v0 = *reinterpret_cast<const uint4*>(
                q + (((size_t)seq * q_len + qj0) * q_heads + h0) * kHd + 32 * p + 8 * t);
        if (ok1)
            v1 = *reinterpret_cast<const uint4*>(
                q + (((size_t)seq * q_len + qj1) * q_heads + h1) * kHd + 32 * p + 8 * t);
        qa[2 * p][0] = v0.x;
        qa[2 * p][1] = v1.x;
        qa[2 * p][2] = v0.y;
        qa[2 * p][3] = v1.y;
        qa[2 * p + 1][0] = v0.z;
        qa[2 * p + 1][1] = v1.z;
        qa[2 * p + 1][2] = v0.w;
        qa[2 * p + 1][3] = v1.w;
    }

    float o[kDTiles][4];
#pragma unroll
    for (int i = 0; i < kDTiles; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    int buf = 0;
    for (int tile = tile0; tile < sp_end; tile += kTileStride) {
        const int next = tile + kTileStride;
        if (next < sp_end) {
            load_values(sv[warp][buf ^ 1], value_cache, table, block_size, kv_heads, kvh, next,
                        sp_end, lane);
        }
        cp_async_commit();
        cp_async_wait<1>();
        __syncwarp();

        float s[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                mma_16816(s[nt], qa[2 * p], kc.v[nt][p].x, kc.v[nt][p].y);
                mma_16816(s[nt], qa[2 * p + 1], kc.v[nt][p].z, kc.v[nt][p].w);
            }
            if (next < sp_end)
                load_keys(kc.v[nt], key_cache, table, block_size, kv_heads, kvh, next, nt, sp_end,
                          g, t);
        }

        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int lim = (e >> 1) ? lim1 : lim0;
                const int key = tile + nt * 8 + 2 * t + (e & 1);
                const bool valid = key < sp_end && key <= lim;
                const float x = valid ? s[nt][e] * sl2 : -INFINITY;
                s[nt][e] = x;
                if (e >> 1) mx1 = fmaxf(mx1, x); else mx0 = fmaxf(mx0, x);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(SCRIPTOR_FULL_MASK, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(SCRIPTOR_FULL_MASK, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(SCRIPTOR_FULL_MASK, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(SCRIPTOR_FULL_MASK, mx1, 2));

        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float u0 = n0 == -INFINITY ? 0.0f : n0;
        const float u1 = n1 == -INFINITY ? 0.0f : n1;
        const float a0 = exp2f(m0 - u0), a1 = exp2f(m1 - u1);
        m0 = n0;
        m1 = n1;

        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            s[nt][0] = exp2f(s[nt][0] - u0);
            s[nt][1] = exp2f(s[nt][1] - u0);
            s[nt][2] = exp2f(s[nt][2] - u1);
            s[nt][3] = exp2f(s[nt][3] - u1);
            sum0 += s[nt][0] + s[nt][1];
            sum1 += s[nt][2] + s[nt][3];
        }
        l0 = l0 * a0 + sum0;
        l1 = l1 * a1 + sum1;

        uint32_t pa[4];
        pa[0] = pack_bf16(s[0][0], s[0][1]);
        pa[1] = pack_bf16(s[0][2], s[0][3]);
        pa[2] = pack_bf16(s[1][0], s[1][1]);
        pa[3] = pack_bf16(s[1][2], s[1][3]);
#pragma unroll
        for (int dt = 0; dt < kDTiles; ++dt) {
            o[dt][0] *= a0;
            o[dt][1] *= a0;
            o[dt][2] *= a1;
            o[dt][3] *= a1;
            const int c0 = swz(2 * t, dt) + g, c1 = swz(2 * t + 8, dt) + g;
            const uint32_t b0 = pack_pair(sv[warp][buf][2 * t][c0], sv[warp][buf][2 * t + 1][c0]);
            const uint32_t b1 =
                pack_pair(sv[warp][buf][2 * t + 8][c1], sv[warp][buf][2 * t + 9][c1]);
            mma_16816(o[dt], pa, b0, b1);
        }
        __syncwarp();
        buf ^= 1;
    }
    cp_async_wait<0>();

    l0 += __shfl_xor_sync(SCRIPTOR_FULL_MASK, l0, 1);
    l0 += __shfl_xor_sync(SCRIPTOR_FULL_MASK, l0, 2);
    l1 += __shfl_xor_sync(SCRIPTOR_FULL_MASK, l1, 1);
    l1 += __shfl_xor_sync(SCRIPTOR_FULL_MASK, l1, 2);

    __syncthreads();
    auto so = reinterpret_cast<float(*)[kMergeRows][kHd]>(smem);
    auto sm = reinterpret_cast<float(*)[kMaxRows]>(smem + kWarps * kMergeRows * kHd * 4);
    auto sl = reinterpret_cast<float(*)[kMaxRows]>(smem + kWarps * kMergeRows * kHd * 4 +
                                                   kWarps * kMaxRows * 4);
    if (t == 0) {
        if (ok0) {
            sm[warp][r0] = m0;
            sl[warp][r0] = l0;
        }
        if (ok1) {
            sm[warp][r1] = m1;
            sl[warp][r1] = l1;
        }
    }

    const int d = threadIdx.x;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int first = half * kMergeRows;
        if (first >= rows) break;
        if (half == 0 ? ok0 : ok1) {
            const int r = half == 0 ? r0 : r1;
#pragma unroll
            for (int dt = 0; dt < kDTiles; ++dt)
                *reinterpret_cast<float2*>(&so[warp][r - first][dt * 8 + 2 * t]) =
                    half == 0 ? make_float2(o[dt][0], o[dt][1]) : make_float2(o[dt][2], o[dt][3]);
        }
        __syncthreads();
        for (int r = first; r < min(rows, first + kMergeRows); ++r) {
            float best = -INFINITY;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) best = fmaxf(best, sm[w][r]);
            float total = 0.0f, acc = 0.0f;
            if (best != -INFINITY) {
#pragma unroll
                for (int w = 0; w < kWarps; ++w) {
                    const float m = sm[w][r];
                    if (m == -INFINITY) continue;
                    const float wt = exp2f(m - best);
                    total += sl[w][r] * wt;
                    acc += so[w][r - first][d] * wt;
                }
            }
            const int qj = r / group;
            const int h = kvh * group + r % group;
            const size_t idx =
                ((size_t)seq * splits + split) * q_len * q_heads + (size_t)qj * q_heads + h;
            if (splits == 1) {
                const float value = total > 0.0f ? acc / total : 0.0f;
                out[idx * kHd + d] = __float2bfloat16_rn(value);
            } else {
                part_o[idx * kHd + d] = acc;
                if (d == 0) {
                    part_m[idx] = best;
                    part_l[idx] = total;
                }
            }
        }
        __syncthreads();
    }
}

// Merges the `parts` partials of each (sequence, query, head) row.
// Grid (q_len * q_heads, seqs); block 128 threads, one per output dimension.
extern "C" __global__ void __launch_bounds__(kHd) scriptor_paged_attn_reduce(
    const float* __restrict__ part_o, const float* __restrict__ part_m,
    const float* __restrict__ part_l, __nv_bfloat16* __restrict__ out, int q_len, int q_heads,
    int parts) {
    const int row = blockIdx.x;
    const int seq = blockIdx.y;
    const size_t stride = (size_t)q_len * q_heads;
    const size_t base = (size_t)seq * parts * stride + row;

    float best = -INFINITY;
    for (int p = 0; p < parts; ++p) best = fmaxf(best, part_m[base + p * stride]);
    float total = 0.0f, acc = 0.0f;
    if (best != -INFINITY) {
        for (int p = 0; p < parts; ++p) {
            const float m = part_m[base + p * stride];
            if (m == -INFINITY) continue;
            const float w = exp2f(m - best);
            total += part_l[base + p * stride] * w;
            acc += part_o[(base + p * stride) * kHd + threadIdx.x] * w;
        }
    }
    const float value = total > 0.0f ? acc / total : 0.0f;
    out[((size_t)seq * stride + row) * kHd + threadIdx.x] = __float2bfloat16_rn(value);
}
