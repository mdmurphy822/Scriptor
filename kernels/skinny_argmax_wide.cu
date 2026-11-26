// Fused skinny-GEMM + argmax for 65..128 tokens: `tok[r]` is the lowest column index holding the
// maximum of row `r` of the bf16-rounded product `x[m, k] . w[n, k]^T`; the product is not stored.
//
// The 128-token tile needs `(16 * 4 + 128) * 128 * 2 = 49152` bytes per ring stage, so the
// two-stage ring fills 98304 of the 101248 budgeted bytes and leaves no room for the cross-warp
// reduction scratch the narrow `scriptor_skinny_argmax_*` kernels keep beside the ring. This
// kernel needs none while streaming: each lane folds the bf16-rounded elements of every finished
// weight-row tile into private per-token running maxima, and the lanes, warps and CTAs are merged
// only after the K loop, with the (by then idle) ring memory serving as the warp-merge scratch.
// Both the main loop and the accumulation order are those of the plain skinny kernel; the
// (value, index) merge is order independent, so the tokens equal those of the narrow kernels and
// of the logits-plus-sampler path.
//
// Single K slice only (`slices == 1`), `65 <= m <= 128`, 128 threads, `stages * stage_bytes`
// bytes of dynamic shared memory with `stages = stages_for(4, 16)`.
#define SCRIPTOR_SKINNY_NO_ENTRIES
#include "skinny_gemm.cu"

namespace {

constexpr int kWideW = 4;
constexpr int kWideNb = 16;
constexpr int kWideStages = stages_for(kWideW, kWideNb);
static_assert(kWideStages >= 2, "the ring needs two stages");
static_assert(kWideStages * stage_bytes(kWideW, kWideNb) <= kSmemBudget,
              "the ring fits the shared-memory budget");
// Warp-merge scratch: one value and one index per warp and token, inside the drained ring.
static_assert(kWideW * 8 * kWideNb * 8 <= kWideStages * stage_bytes(kWideW, kWideNb),
              "the merge scratch fits in the ring");

__device__ __forceinline__ void wide_argmax_body(const __nv_bfloat16* __restrict__ x,
                                                 const __nv_bfloat16* __restrict__ w,
                                                 uint32_t* __restrict__ tok,
                                                 float4* __restrict__ part,
                                                 int* __restrict__ counters, int m, int n, int k,
                                                 int slices, int items) {
    constexpr int W = kWideW;
    constexpr int NB = kWideNb;
    constexpr int BK = kBlockK;
    constexpr int ST = kWideStages;
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
    const int g = lane >> 2;
    const int t = lane & 3;

    float acc[NB][4];
#pragma unroll
    for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;

    // Running maximum of the token `nb * 8 + 2 * t + j` over the weight rows `g` and `g + 8` of
    // every finished tile of this warp.
    float run_v[NB][2];
    uint32_t run_i[NB][2];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            run_v[nb][j] = -INFINITY;
            run_i[nb][j] = kNoIndex;
        }

#pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
        if (s < total) load_stage<W, NB, BK, ST>(smem, s, x, w, m, k, tiles, spi, s);
        cp_async_commit();
    }

    const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int a_chunk = lane >> 4;
    const int b_row = (lane & 7) + (lane >> 4) * 8;
    const int b_chunk = (lane >> 3) & 1;

    for (int step = 0; step < total; ++step) {
        cp_async_wait<ST - 2>();
        __syncthreads();
        const int ahead = step + ST - 1;
        if (ahead < total)
            load_stage<W, NB, BK, ST>(smem, ahead % ST, x, w, m, k, tiles, spi, ahead);
        cp_async_commit();

        const uint32_t x_base = smem + (step % ST) * C::kStageBytes;
        const uint32_t w_base = x_base + C::kXBytes;
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
            uint32_t a[4];
            ldmatrix_x4(a, w_base + tile_offset<BK>(warp * 16 + a_row, ks * 2 + a_chunk));
#pragma unroll
            for (int nj = 0; nj < NB / 2; ++nj) {
                uint32_t b[4];
                ldmatrix_x4(b, x_base + tile_offset<BK>(nj * 16 + b_row, ks * 2 + b_chunk));
                mma_16816(acc[2 * nj], a, b[0], b[1]);
                mma_16816(acc[2 * nj + 1], a, b[2], b[3]);
            }
        }

        if ((step + 1) % spi == 0) {
            const int item = blockIdx.x + (step / spi) * gridDim.x;
            const int tile = item % tiles;
            const uint32_t base = static_cast<uint32_t>(tile * C::kRows + warp * 16 + g);
#pragma unroll
            for (int nb = 0; nb < NB; ++nb)
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float v = __bfloat162float(__float2bfloat16_rn(acc[nb][2 * h + j]));
                        arg_take(run_v[nb][j], run_i[nb][j], v, base + 8 * h);
                    }
#pragma unroll
            for (int j = 0; j < NB; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;
        }
    }

    // The ring is idle once every copy has landed and every warp has left the K loop.
    cp_async_wait<0>();
    __syncthreads();
    float* scratch = reinterpret_cast<float*>(smem_raw);
    uint32_t* idx_scratch = reinterpret_cast<uint32_t*>(scratch + W * C::kTokens);
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            float bv = run_v[nb][j];
            uint32_t bi = run_i[nb][j];
#pragma unroll
            for (int off = 4; off < 32; off <<= 1) {
                const float ov = __shfl_xor_sync(SCRIPTOR_FULL_MASK, bv, off);
                const uint32_t oi = __shfl_xor_sync(SCRIPTOR_FULL_MASK, bi, off);
                arg_take(bv, bi, ov, oi);
            }
            if (lane < 4) {
                const int token = nb * 8 + 2 * t + j;
                scratch[warp * C::kTokens + token] = bv;
                idx_scratch[warp * C::kTokens + token] = bi;
            }
        }
    __syncthreads();
    float cta_v = -INFINITY;
    uint32_t cta_i = kNoIndex;
    if (threadIdx.x < C::kTokens) {
#pragma unroll
        for (int ww = 0; ww < W; ++ww)
            arg_take(cta_v, cta_i, scratch[ww * C::kTokens + threadIdx.x],
                     idx_scratch[ww * C::kTokens + threadIdx.x]);
    }

    // Each CTA publishes its per-token maximum; the last CTA to finish folds the grid's maxima
    // into the output tokens and clears the counter.
    float2* partials = reinterpret_cast<float2*>(part);
    if (threadIdx.x < m)
        partials[blockIdx.x * C::kTokens + threadIdx.x] =
            make_float2(cta_v, __uint_as_float(cta_i));
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

}  // namespace

extern "C" __global__ void __launch_bounds__(32 * kWideW, 1) scriptor_skinny_argmax_wide_w4_nb16(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    uint32_t* __restrict__ tok, float4* __restrict__ part, int* __restrict__ counters, int m,
    int n, int k, int slices, int items) {
    wide_argmax_body(x, w, tok, part, counters, m, n, k, slices, items);
}
