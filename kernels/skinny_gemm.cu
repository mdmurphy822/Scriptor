// Skinny-M bf16 GEMM `out[m, n] = bf16(x[m, k] . w[n, k]^T)` for m <= 128, fp32 accumulation.
//
// The weight matrix is streamed from global memory exactly once through a multi-stage
// `cp.async` ring while the `m x k` activation slice of the same K range rides along in the
// ring (it stays L2-resident across CTAs). The weights take the MMA "A" role (16 weight rows
// per warp, K steps of 32) and the tokens the "B" role in groups of 8, so a warp accumulates a
// 16 x TM tile in `TM / 2` registers whatever the token count.
//
// CTAs are persistent, one per SM: the work items `(k slice, weight-row tile)` are dealt
// round-robin to `gridDim.x` CTAs and the ring keeps streaming across item boundaries, so the
// next item's loads are in flight while the previous item's epilogue runs. When the K range is split, each
// CTA stores its fp32 partial tile in fragment order into a workspace and bumps a per-tile
// counter; the CTA that completes a tile sums the partials in slice order (so results do not
// depend on arrival order), writes the bf16 rows and resets the counter to zero.
//
// The `scriptor_skinny_argmax_*` entries (single K slice, `m <= 64`) do not store the product: the
// epilogue rounds each element to bf16 exactly as the plain store does and reduces every token
// to its maximum and lowest index of that maximum over the whole weight matrix, writing one
// u32 token index per row. They need `8 * NB * 8 * W` extra bytes of dynamic shared memory
// after the ring.
//
// Requires `k % 128 == 0`, `n % (16 * W) == 0` and `(k / 128) % slices == 0`; `m` is arbitrary up
// to `8 * NB` (rows beyond `m` are zero-filled on load and not stored). Launch with 32 * W
// threads and `stages_for(W, NB) * stage_bytes(W, NB)` bytes of dynamic shared memory, mirrored
// by the host.
#include "common.cuh"

namespace {

// Dynamic shared memory one CTA may use: the opt-in per-block limit less the static flag.
constexpr int kSmemBudget = 101248;
constexpr int kMaxStages = 8;
// Reduction elements per ring stage: each weight row contributes a 256-byte contiguous
// segment per stage, which keeps the DRAM streams long.
constexpr int kBlockK = 128;

__host__ __device__ constexpr int stage_bytes(int w, int nb) {
    return (16 * w + 8 * nb) * kBlockK * 2;
}

// Ring depth of the `(w, nb)` instantiation: as deep as the shared-memory budget allows.
__host__ __device__ constexpr int stages_for(int w, int nb) {
    const int s = kSmemBudget / stage_bytes(w, nb);
    return s > kMaxStages ? kMaxStages : s;
}

__device__ __forceinline__ void mma_16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                          uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Copies 16 bytes global -> shared; zero-fills the destination when `valid` is false.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* gmem, bool valid) {
    const int bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem),
                 "r"(bytes));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(addr));
}

// Byte offset of 16-byte chunk `chunk` of tile row `row` in a ring slot whose rows hold `bk`
// reduction elements; the chunk index is swizzled so eight consecutive rows never share banks.
template <int BK>
__device__ __forceinline__ uint32_t tile_offset(int row, int chunk) {
    constexpr int rb = BK * 2;
    const int swz = row & 7;
    return static_cast<uint32_t>(row * rb + ((chunk ^ swz) << 4));
}

template <int W, int NB, int BK, int ST>
struct Cfg {
    static constexpr int kRows = 16 * W;
    static constexpr int kTokens = 8 * NB;
    static constexpr int kThreads = 32 * W;
    static constexpr int kRowBytes = BK * 2;
    static constexpr int kChunksPerRow = BK / 8;
    static constexpr int kXBytes = kTokens * kRowBytes;
    static constexpr int kStageBytes = (kRows + kTokens) * kRowBytes;
    static constexpr int kChunks = (kRows + kTokens) * kChunksPerRow;
};

// Issues the copies of global step `g` of this CTA into ring slot `slot`.
template <int W, int NB, int BK, int ST>
__device__ __forceinline__ void load_stage(uint32_t smem, int slot, const __nv_bfloat16* x,
                                           const __nv_bfloat16* w, int m, int k, int tiles,
                                           int spi, int g) {
    using C = Cfg<W, NB, BK, ST>;
    const int item_local = g / spi;
    const int kl = g - item_local * spi;
    const int item = blockIdx.x + item_local * gridDim.x;
    const int slice = item / tiles;
    const int tile = item - slice * tiles;
    const int kcol = (slice * spi + kl) * BK;
    const uint32_t base = smem + slot * C::kStageBytes;
#pragma unroll
    for (int p = 0; p < (C::kChunks + C::kThreads - 1) / C::kThreads; ++p) {
        const int idx = threadIdx.x + p * C::kThreads;
        if (idx < C::kChunks) {
            const int row = idx / C::kChunksPerRow;
            const int c = idx % C::kChunksPerRow;
            if (row < C::kTokens) {
                const bool valid = row < m;
                cp_async16(base + tile_offset<BK>(row, c),
                           x + (long)(valid ? row : 0) * k + kcol + c * 8, valid);
            } else {
                const int wr = row - C::kTokens;
                cp_async16(base + C::kXBytes + tile_offset<BK>(wr, c),
                           w + (long)(tile * C::kRows + wr) * k + kcol + c * 8, true);
            }
        }
    }
}

// Rounds the 16 x TM tile held in `acc` to bf16 and stores it as `out[token, row]`.
template <int NB>
__device__ __forceinline__ void store_tile(const float (&acc)[NB][4], __nv_bfloat16* out, int m,
                                           int n, int row_base) {
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const bool odd = g & 1;
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float lo = acc[nb][2 * h];
            const float hi = acc[nb][2 * h + 1];
            const float recv = __shfl_xor_sync(SCRIPTOR_FULL_MASK, odd ? lo : hi, 4);
            const float first = odd ? recv : lo;
            const float second = odd ? hi : recv;
            const int token = nb * 8 + 2 * t + (odd ? 1 : 0);
            if (token < m) {
                const __nv_bfloat162 pair = __floats2bfloat162_rn(first, second);
                *reinterpret_cast<__nv_bfloat162*>(out + (long)token * n + row_base + 8 * h +
                                                   (g & ~1)) = pair;
            }
        }
    }
}

// Running (value, index) maximum of one token: a strictly larger value wins and equal values
// keep the lowest index, so the result does not depend on the order candidates are folded in.
// NaN never wins; an all -inf row resolves to index 0 and an all-NaN row to `kNoIndex`.
constexpr uint32_t kNoIndex = 0xffffffffu;

__device__ __forceinline__ void arg_take(float& bv, uint32_t& bi, float v, uint32_t i) {
    if (v > bv || (v == bv && i < bi)) {
        bv = v;
        bi = i;
    }
}

// Folds the bf16-rounded tile held in `acc` (weight-row tile `tile`) into the per-token
// maxima: the lanes of a warp that share a token reduce by shuffle, the warps meet in `scratch`
// (`W * kTokens` values then `W * kTokens` indices), and threads `< kTokens` keep the running
// maximum of token `threadIdx.x` in `run_v` / `run_i`.
template <int W, int NB>
__device__ __forceinline__ void argmax_tile(const float (&acc)[NB][4], int tile, float* scratch,
                                            float& run_v, uint32_t& run_i) {
    constexpr int kTokens = 8 * NB;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const uint32_t base = static_cast<uint32_t>(tile * 16 * W + warp * 16 + g);
    uint32_t* idx_scratch = reinterpret_cast<uint32_t*>(scratch + W * kTokens);
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            float bv = -INFINITY;
            uint32_t bi = kNoIndex;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float v = __bfloat162float(__float2bfloat16_rn(acc[nb][2 * h + j]));
                arg_take(bv, bi, v, base + 8 * h);
            }
#pragma unroll
            for (int off = 4; off < 32; off <<= 1) {
                const float ov = __shfl_xor_sync(SCRIPTOR_FULL_MASK, bv, off);
                const uint32_t oi = __shfl_xor_sync(SCRIPTOR_FULL_MASK, bi, off);
                arg_take(bv, bi, ov, oi);
            }
            if (lane < 4) {
                const int token = nb * 8 + 2 * t + j;
                scratch[warp * kTokens + token] = bv;
                idx_scratch[warp * kTokens + token] = bi;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < kTokens) {
#pragma unroll
        for (int w = 0; w < W; ++w)
            arg_take(run_v, run_i, scratch[w * kTokens + threadIdx.x],
                     idx_scratch[w * kTokens + threadIdx.x]);
    }
}

template <int W, int NB, int BK, int ST, bool ARGMAX>
__device__ __forceinline__ void skinny_body(const __nv_bfloat16* __restrict__ x,
                                            const __nv_bfloat16* __restrict__ w,
                                            __nv_bfloat16* __restrict__ out,
                                            uint32_t* __restrict__ tok, float4* __restrict__ part,
                                            int* __restrict__ counters, int m, int n, int k,
                                            int slices, int items) {
    using C = Cfg<W, NB, BK, ST>;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    __shared__ int last_flag;
    const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    const int tiles = n / C::kRows;
    const int spi = (k / BK) / slices;
    const int count = blockIdx.x < items ? (items - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
    const int total = count * spi;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    float acc[NB][4];
#pragma unroll
    for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;

    float run_v = -INFINITY;
    uint32_t run_i = kNoIndex;
    float* arg_scratch = reinterpret_cast<float*>(smem_raw + ST * C::kStageBytes);

#pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
        if (s < total) load_stage<W, NB, BK, ST>(smem, s, x, w, m, k, tiles, spi, s);
        cp_async_commit();
    }

    const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int a_chunk = lane >> 4;
    const int b_row = (lane & 7) + (lane >> 4) * 8;
    const int b_chunk = (lane >> 3) & 1;

    for (int g = 0; g < total; ++g) {
        cp_async_wait<ST - 2>();
        __syncthreads();
        const int ahead = g + ST - 1;
        if (ahead < total)
            load_stage<W, NB, BK, ST>(smem, ahead % ST, x, w, m, k, tiles, spi, ahead);
        cp_async_commit();

        const uint32_t x_base = smem + (g % ST) * C::kStageBytes;
        const uint32_t w_base = x_base + C::kXBytes;
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
            uint32_t a[4];
            ldmatrix_x4(a, w_base + tile_offset<BK>(warp * 16 + a_row, ks * 2 + a_chunk));
            if constexpr (NB == 1) {
                uint32_t b[2];
                ldmatrix_x2(b, x_base + tile_offset<BK>(b_row & 7, ks * 2 + b_chunk));
                mma_16816(acc[0], a, b[0], b[1]);
            } else {
#pragma unroll
                for (int nj = 0; nj < NB / 2; ++nj) {
                    uint32_t b[4];
                    ldmatrix_x4(b, x_base + tile_offset<BK>(nj * 16 + b_row, ks * 2 + b_chunk));
                    mma_16816(acc[2 * nj], a, b[0], b[1]);
                    mma_16816(acc[2 * nj + 1], a, b[2], b[3]);
                }
            }
        }

        if ((g + 1) % spi == 0) {
            const int item = blockIdx.x + (g / spi) * gridDim.x;
            const int slice = item / tiles;
            const int tile = item - slice * tiles;
            const int row_base = tile * C::kRows + warp * 16;
            if constexpr (ARGMAX) {
                argmax_tile<W, NB>(acc, tile, arg_scratch, run_v, run_i);
            } else if (slices == 1) {
                store_tile<NB>(acc, out, m, n, row_base);
            } else {
                float4* mine = part + ((long)(tile * slices + slice) * W + warp) * NB * 32 + lane;
#pragma unroll
                for (int j = 0; j < NB; ++j)
                    mine[j * 32] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
                __threadfence();
                __syncthreads();
                if (threadIdx.x == 0) last_flag = atomicAdd(&counters[tile], 1) == slices - 1;
                __syncthreads();
                if (last_flag) {
                    __threadfence();
#pragma unroll
                    for (int j = 0; j < NB; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;
                    for (int s = 0; s < slices; ++s) {
                        const float4* src =
                            part + ((long)(tile * slices + s) * W + warp) * NB * 32 + lane;
#pragma unroll
                        for (int j = 0; j < NB; ++j) {
                            const float4 v = __ldcg(src + j * 32);
                            acc[j][0] += v.x;
                            acc[j][1] += v.y;
                            acc[j][2] += v.z;
                            acc[j][3] += v.w;
                        }
                    }
                    store_tile<NB>(acc, out, m, n, row_base);
                    if (threadIdx.x == 0) counters[tile] = 0;
                }
                __syncthreads();
            }
#pragma unroll
            for (int j = 0; j < NB; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;
        }
    }

    if constexpr (ARGMAX) {
        // Each CTA publishes its per-token maximum; the last CTA to finish folds the grid's
        // maxima into the output tokens and clears the counter.
        float2* partials = reinterpret_cast<float2*>(part);
        if (threadIdx.x < m)
            partials[blockIdx.x * C::kTokens + threadIdx.x] =
                make_float2(run_v, __uint_as_float(run_i));
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) last_flag = atomicAdd(&counters[0], 1) == (int)gridDim.x - 1;
        __syncthreads();
        if (last_flag) {
            __threadfence();
            if (threadIdx.x < m) {
                float bv = -INFINITY;
                uint32_t bi = kNoIndex;
                for (int c = 0; c < (int)gridDim.x; ++c) {
                    const float2 p = __ldcg(partials + c * C::kTokens + threadIdx.x);
                    arg_take(bv, bi, p.x, __float_as_uint(p.y));
                }
                tok[threadIdx.x] = bi;
            }
            if (threadIdx.x == 0) counters[0] = 0;
        }
    }
}

}  // namespace

#ifndef SCRIPTOR_SKINNY_NO_ENTRIES
#define SCRIPTOR_SKINNY_ENTRY(W, NB)                                                                   \
    extern "C" __global__ void __launch_bounds__(32 * W, 1) scriptor_skinny_gemm_w##W##_nb##NB(        \
        const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,                \
        __nv_bfloat16* __restrict__ out, float4* __restrict__ part, int* __restrict__ counters,  \
        int m, int n, int k, int slices, int items) {                                            \
        skinny_body<W, NB, kBlockK, stages_for(W, NB), false>(x, w, out, nullptr, part,          \
                                                              counters, m, n, k, slices,         \
                                                              items);                            \
    }

#define SCRIPTOR_SKINNY_ARGMAX_ENTRY(W, NB)                                                            \
    extern "C" __global__ void __launch_bounds__(32 * W, 1) scriptor_skinny_argmax_w##W##_nb##NB(      \
        const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,                \
        uint32_t* __restrict__ tok, float4* __restrict__ part, int* __restrict__ counters,       \
        int m, int n, int k, int slices, int items) {                                            \
        skinny_body<W, NB, kBlockK, stages_for(W, NB), true>(x, w, nullptr, tok, part,           \
                                                             counters, m, n, k, slices,          \
                                                             items);                             \
    }

#define SCRIPTOR_SKINNY_WIDTH(W) \
    SCRIPTOR_SKINNY_ENTRY(W, 1)  \
    SCRIPTOR_SKINNY_ENTRY(W, 2)  \
    SCRIPTOR_SKINNY_ENTRY(W, 4)  \
    SCRIPTOR_SKINNY_ENTRY(W, 8)  \
    SCRIPTOR_SKINNY_ENTRY(W, 16)

SCRIPTOR_SKINNY_WIDTH(4)
SCRIPTOR_SKINNY_ARGMAX_ENTRY(4, 1)
SCRIPTOR_SKINNY_ARGMAX_ENTRY(4, 2)
SCRIPTOR_SKINNY_ARGMAX_ENTRY(4, 4)
SCRIPTOR_SKINNY_ARGMAX_ENTRY(4, 8)
#endif
