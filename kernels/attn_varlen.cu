// Flash-style variable-length attention with grouped KV heads.
//
// One thread block (4 warps) owns 64 consecutive query rows of one head of one
// sequence; each warp owns 16 rows. Keys and values stream through shared
// memory in tiles of 64. Scores and the output accumulate in fp32 through
// bf16 `mma.sync.m16n8k16` tiles; the softmax is online, in the exp2 domain.
#include "common.cuh"

namespace {

constexpr int kBlockQ = 64;
constexpr int kBlockKv = 64;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * SCRIPTOR_WARP_SIZE;
constexpr int kPad = 8;  // bf16 elements of row padding in shared memory

// D += A * B for one m16n8k16 bf16 tile (A row-major, B column-major).
__device__ __forceinline__ void mma_16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                          uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Packs two fp32 values into one register of two bf16 (first value in the low half).
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// Packs two bf16 values into one register (first value in the low half).
__device__ __forceinline__ uint32_t pack_pair(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return static_cast<uint32_t>(__bfloat16_as_ushort(lo)) |
           (static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16);
}

constexpr float kLog2e = 1.4426950408889634f;

template <int HD, bool CAUSAL>
__device__ __forceinline__ void attn_varlen(const __nv_bfloat16* __restrict__ q,
                                            const __nv_bfloat16* __restrict__ k,
                                            const __nv_bfloat16* __restrict__ v,
                                            __nv_bfloat16* __restrict__ out,
                                            const int* __restrict__ cu_seqlens, int q_heads,
                                            int kv_heads, float scale) {
    constexpr int kStride = HD + kPad;
    constexpr int kKSteps = HD / 16;
    constexpr int kDTiles = HD / 8;
    constexpr int kChunksPerRow = HD / 8;  // 16-byte chunks per row

    __shared__ __align__(16) __nv_bfloat16 sk[kBlockKv][kStride];
    __shared__ __align__(16) __nv_bfloat16 sv[kBlockKv][kStride];

    const int seq = blockIdx.z;
    const int head = blockIdx.y;
    const int start = cu_seqlens[seq];
    const int len = cu_seqlens[seq + 1] - start;
    const int q0 = blockIdx.x * kBlockQ;
    if (q0 >= len) return;

    const int group = q_heads / kv_heads;
    const int kvh = head / group;
    const int tid = threadIdx.x;
    const int warp = tid / SCRIPTOR_WARP_SIZE;
    const int lane = tid % SCRIPTOR_WARP_SIZE;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int row0 = q0 + warp * 16 + g;
    const int row1 = row0 + 8;
    const float sl2 = scale * kLog2e;

    uint32_t qa[kKSteps][4];
#pragma unroll
    for (int ks = 0; ks < kKSteps; ++ks) {
        const int c = ks * 16 + 2 * t;
        const __nv_bfloat16* p0 = q + ((size_t)(start + row0) * q_heads + head) * HD + c;
        const __nv_bfloat16* p1 = q + ((size_t)(start + row1) * q_heads + head) * HD + c;
        qa[ks][0] = row0 < len ? *reinterpret_cast<const uint32_t*>(p0) : 0u;
        qa[ks][1] = row1 < len ? *reinterpret_cast<const uint32_t*>(p1) : 0u;
        qa[ks][2] = row0 < len ? *reinterpret_cast<const uint32_t*>(p0 + 8) : 0u;
        qa[ks][3] = row1 < len ? *reinterpret_cast<const uint32_t*>(p1 + 8) : 0u;
    }

    float o[kDTiles][4];
#pragma unroll
    for (int i = 0; i < kDTiles; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    int kv_end = len;
    if (CAUSAL) kv_end = min(len, q0 + kBlockQ);

    for (int kv0 = 0; kv0 < kv_end; kv0 += kBlockKv) {
        __syncthreads();
        for (int idx = tid; idx < kBlockKv * kChunksPerRow; idx += kThreads) {
            const int j = idx / kChunksPerRow;
            const int c = idx % kChunksPerRow;
            const int key = kv0 + j;
            uint4 kk = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (key < len) {
                const size_t off = ((size_t)(start + key) * kv_heads + kvh) * HD + c * 8;
                kk = *reinterpret_cast<const uint4*>(k + off);
                vv = *reinterpret_cast<const uint4*>(v + off);
            }
            *reinterpret_cast<uint4*>(&sk[j][c * 8]) = kk;
            *reinterpret_cast<uint4*>(&sv[j][c * 8]) = vv;
        }
        __syncthreads();

        if (CAUSAL && kv0 > q0 + warp * 16 + 15) continue;

        float s[kBlockKv / 8][4];
#pragma unroll
        for (int nt = 0; nt < kBlockKv / 8; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < kKSteps; ++ks) {
#pragma unroll
            for (int nt = 0; nt < kBlockKv / 8; ++nt) {
                const uint32_t b0 =
                    *reinterpret_cast<const uint32_t*>(&sk[nt * 8 + g][ks * 16 + 2 * t]);
                const uint32_t b1 =
                    *reinterpret_cast<const uint32_t*>(&sk[nt * 8 + g][ks * 16 + 2 * t + 8]);
                mma_16816(s[nt], qa[ks], b0, b1);
            }
        }

        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < kBlockKv / 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = (e >> 1) ? row1 : row0;
                const int key = kv0 + nt * 8 + 2 * t + (e & 1);
                const bool valid = key < len && (!CAUSAL || key <= row);
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
        for (int nt = 0; nt < kBlockKv / 8; ++nt) {
            s[nt][0] = exp2f(s[nt][0] - u0);
            s[nt][1] = exp2f(s[nt][1] - u0);
            s[nt][2] = exp2f(s[nt][2] - u1);
            s[nt][3] = exp2f(s[nt][3] - u1);
            sum0 += s[nt][0] + s[nt][1];
            sum1 += s[nt][2] + s[nt][3];
        }
        l0 = l0 * a0 + sum0;
        l1 = l1 * a1 + sum1;
#pragma unroll
        for (int dt = 0; dt < kDTiles; ++dt) {
            o[dt][0] *= a0;
            o[dt][1] *= a0;
            o[dt][2] *= a1;
            o[dt][3] *= a1;
        }

#pragma unroll
        for (int kk = 0; kk < kBlockKv / 16; ++kk) {
            uint32_t pa[4];
            pa[0] = pack_bf16(s[2 * kk][0], s[2 * kk][1]);
            pa[1] = pack_bf16(s[2 * kk][2], s[2 * kk][3]);
            pa[2] = pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            pa[3] = pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
            for (int dt = 0; dt < kDTiles; ++dt) {
                const int col = dt * 8 + g;
                const uint32_t b0 = pack_pair(sv[kk * 16 + 2 * t][col], sv[kk * 16 + 2 * t + 1][col]);
                const uint32_t b1 =
                    pack_pair(sv[kk * 16 + 2 * t + 8][col], sv[kk * 16 + 2 * t + 9][col]);
                mma_16816(o[dt], pa, b0, b1);
            }
        }
    }

    l0 += __shfl_xor_sync(SCRIPTOR_FULL_MASK, l0, 1);
    l0 += __shfl_xor_sync(SCRIPTOR_FULL_MASK, l0, 2);
    l1 += __shfl_xor_sync(SCRIPTOR_FULL_MASK, l1, 1);
    l1 += __shfl_xor_sync(SCRIPTOR_FULL_MASK, l1, 2);
    const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f;
    const float inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int dt = 0; dt < kDTiles; ++dt) {
        const int col = dt * 8 + 2 * t;
        if (row0 < len) {
            *reinterpret_cast<uint32_t*>(out + ((size_t)(start + row0) * q_heads + head) * HD + col) =
                pack_bf16(o[dt][0] * inv0, o[dt][1] * inv0);
        }
        if (row1 < len) {
            *reinterpret_cast<uint32_t*>(out + ((size_t)(start + row1) * q_heads + head) * HD + col) =
                pack_bf16(o[dt][2] * inv1, o[dt][3] * inv1);
        }
    }
}

}  // namespace

// Entry points. Grid: (ceil(max_seqlen / 64), q_heads, num_seqs); block: 128 threads.
// q/out are [tokens, q_heads, HD], k/v are [tokens, kv_heads, HD], all bf16;
// cu_seqlens is [num_seqs + 1] int32 token offsets.
#define SCRIPTOR_ATTN_VARLEN_ENTRY(NAME, HD, CAUSAL)                                                 \
    extern "C" __global__ void __launch_bounds__(kThreads) NAME(                              \
        const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,               \
        __nv_bfloat16* out, const int* cu_seqlens, int q_heads, int kv_heads, float scale) {  \
        attn_varlen<HD, CAUSAL>(q, k, v, out, cu_seqlens, q_heads, kv_heads, scale);          \
    }

SCRIPTOR_ATTN_VARLEN_ENTRY(scriptor_attn_varlen_d64_full, 64, false)
SCRIPTOR_ATTN_VARLEN_ENTRY(scriptor_attn_varlen_d64_causal, 64, true)
SCRIPTOR_ATTN_VARLEN_ENTRY(scriptor_attn_varlen_d128_full, 128, false)
SCRIPTOR_ATTN_VARLEN_ENTRY(scriptor_attn_varlen_d128_causal, 128, true)
