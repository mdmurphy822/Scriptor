// Baseline paged decode attention (one 4-warp partial per split, K read through 4-byte loads),
// kept as the equivalence and bandwidth reference for `attn_paged_decode.cu`.
//
// Grid (kv_heads, splits, sequences); 4 warps per block. The query rows of one
// KV head (`group * q_len` of them, at most 16) form one m16 tile. Each warp
// walks 16-key tiles of its split's key range, strided by 4 tiles across the
// block, and writes an unnormalised partial (output, running max, running sum)
// for its own slice; `scriptor_paged_attn_baseline_reduce` merges the `splits * 4` partials.
// Query `j` of a sequence sees positions `0 ..= seq_len - q_len + j`.
#include "common.cuh"

namespace {

constexpr int kHd = 128;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * SCRIPTOR_WARP_SIZE;
constexpr int kTile = 16;
constexpr int kStride = kHd + 8;
constexpr float kLog2e = 1.4426950408889634f;

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

}  // namespace

// q: [seqs, q_len, q_heads, 128] bf16. Caches: [blocks, block_size, kv_heads, 128] bf16.
// block_tables: [seqs, max_blocks] int32. seq_lens[s] counts cached tokens including
// the q_len new ones. part_o: [seqs, splits * 4, q_len, q_heads, 128] fp32;
// part_m / part_l: [seqs, splits * 4, q_len, q_heads] fp32 (max in the exp2 domain).
extern "C" __global__ void __launch_bounds__(kThreads) scriptor_paged_attn_baseline_d128(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache, const int* __restrict__ block_tables,
    int max_blocks, const int* __restrict__ seq_lens, int q_len, int q_heads, int kv_heads,
    int block_size, int splits, int chunk, float scale, float* __restrict__ part_o,
    float* __restrict__ part_m, float* __restrict__ part_l) {
    constexpr int kKSteps = kHd / 16;
    constexpr int kDTiles = kHd / 8;
    constexpr int kChunksPerRow = kHd / 8;

    __shared__ __align__(16) __nv_bfloat16 sv[kWarps][kTile][kStride];

    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int seq = blockIdx.z;
    const int warp = threadIdx.x / SCRIPTOR_WARP_SIZE;
    const int lane = threadIdx.x % SCRIPTOR_WARP_SIZE;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int seq_len = seq_lens[seq];
    const int group = q_heads / kv_heads;
    const int rows = group * q_len;
    const int sp_start = split * chunk;
    const int sp_end = min(sp_start + chunk, seq_len);
    const int parts = splits * kWarps;
    const int part = split * kWarps + warp;
    const int* table = block_tables + (size_t)seq * max_blocks;

    const int r0 = g, r1 = g + 8;
    const bool ok0 = r0 < rows, ok1 = r1 < rows;
    const int qj0 = ok0 ? r0 / group : 0, qj1 = ok1 ? r1 / group : 0;
    const int h0 = kvh * group + (ok0 ? r0 % group : 0);
    const int h1 = kvh * group + (ok1 ? r1 % group : 0);
    const int lim0 = ok0 ? seq_len - q_len + qj0 : -1;
    const int lim1 = ok1 ? seq_len - q_len + qj1 : -1;
    const float sl2 = scale * kLog2e;

    uint32_t qa[kKSteps][4];
#pragma unroll
    for (int ks = 0; ks < kKSteps; ++ks) {
        const int c = ks * 16 + 2 * t;
        const __nv_bfloat16* p0 = q + (((size_t)seq * q_len + qj0) * q_heads + h0) * kHd + c;
        const __nv_bfloat16* p1 = q + (((size_t)seq * q_len + qj1) * q_heads + h1) * kHd + c;
        qa[ks][0] = ok0 ? *reinterpret_cast<const uint32_t*>(p0) : 0u;
        qa[ks][1] = ok1 ? *reinterpret_cast<const uint32_t*>(p1) : 0u;
        qa[ks][2] = ok0 ? *reinterpret_cast<const uint32_t*>(p0 + 8) : 0u;
        qa[ks][3] = ok1 ? *reinterpret_cast<const uint32_t*>(p1 + 8) : 0u;
    }

    float o[kDTiles][4];
#pragma unroll
    for (int i = 0; i < kDTiles; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    for (int tile = sp_start + warp * kTile; tile < sp_end; tile += kWarps * kTile) {
        for (int idx = lane; idx < kTile * kChunksPerRow; idx += SCRIPTOR_WARP_SIZE) {
            const int j = idx / kChunksPerRow;
            const int c = idx % kChunksPerRow;
            const int pos = tile + j;
            uint4 vv = make_uint4(0, 0, 0, 0);
            if (pos < sp_end) {
                const size_t row = (size_t)table[pos / block_size] * block_size + pos % block_size;
                vv = *reinterpret_cast<const uint4*>(value_cache + (row * kv_heads + kvh) * kHd + c * 8);
            }
            *reinterpret_cast<uint4*>(&sv[warp][j][c * 8]) = vv;
        }
        __syncwarp();

        float s[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
            const int kpos = min(tile + nt * 8 + g, sp_end - 1);
            const size_t krow = (size_t)table[kpos / block_size] * block_size + kpos % block_size;
            const __nv_bfloat16* kp = key_cache + (krow * kv_heads + kvh) * kHd + 2 * t;
#pragma unroll
            for (int ks = 0; ks < kKSteps; ++ks) {
                const uint32_t b0 = __ldg(reinterpret_cast<const uint32_t*>(kp + ks * 16));
                const uint32_t b1 = __ldg(reinterpret_cast<const uint32_t*>(kp + ks * 16 + 8));
                mma_16816(s[nt], qa[ks], b0, b1);
            }
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
            const int col = dt * 8 + g;
            const uint32_t b0 = pack_pair(sv[warp][2 * t][col], sv[warp][2 * t + 1][col]);
            const uint32_t b1 = pack_pair(sv[warp][2 * t + 8][col], sv[warp][2 * t + 9][col]);
            mma_16816(o[dt], pa, b0, b1);
        }
        __syncwarp();
    }

    l0 += __shfl_xor_sync(SCRIPTOR_FULL_MASK, l0, 1);
    l0 += __shfl_xor_sync(SCRIPTOR_FULL_MASK, l0, 2);
    l1 += __shfl_xor_sync(SCRIPTOR_FULL_MASK, l1, 1);
    l1 += __shfl_xor_sync(SCRIPTOR_FULL_MASK, l1, 2);

    const size_t base = (size_t)seq * parts + part;
    if (ok0) {
        const size_t idx = (base * q_len + qj0) * q_heads + h0;
        if (t == 0) {
            part_m[idx] = m0;
            part_l[idx] = l0;
        }
#pragma unroll
        for (int dt = 0; dt < kDTiles; ++dt)
            *reinterpret_cast<float2*>(part_o + idx * kHd + dt * 8 + 2 * t) =
                make_float2(o[dt][0], o[dt][1]);
    }
    if (ok1) {
        const size_t idx = (base * q_len + qj1) * q_heads + h1;
        if (t == 0) {
            part_m[idx] = m1;
            part_l[idx] = l1;
        }
#pragma unroll
        for (int dt = 0; dt < kDTiles; ++dt)
            *reinterpret_cast<float2*>(part_o + idx * kHd + dt * 8 + 2 * t) =
                make_float2(o[dt][2], o[dt][3]);
    }
}

// Merges the `splits * 4` partials of each (sequence, query, head) row.
// Grid (q_len * q_heads, seqs); block 128 threads, one per output dimension.
extern "C" __global__ void __launch_bounds__(kHd) scriptor_paged_attn_baseline_reduce(
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
