// Vision MLP gate/up projection with the SiLU-and-multiply applied in the GEMM epilogue.
//
// `out[rows, inter] = bf16(silu(gate) * up)` where `gate = bf16(x * w[0:inter]^T)` and
// `up = bf16(x * w[inter:2*inter]^T)`, `w` being the stacked `[2 * inter, k]` gate-then-up
// weight. Gate and up are rounded to bf16 before the activation, exactly as the separate
// GEMM + `scriptor_silu_mul_fused_bf16` pair does, so only the fp32 accumulation order differs from
// a vendor GEMM. An optional bf16 bias `[2 * inter]` is added to the fp32 accumulator before
// that rounding, as the vendor bias epilogue does. The `[rows, 2 * inter]` intermediate is never written.
//
// A 128 x 256 accumulator tile per block (8 warps, 64 x 64 per warp), K in steps of 32 through
// a `cp.async` ring. The 256 weight rows of a tile are 128 gate rows and the 128 matching up
// rows, interleaved per warp: warp column `wn` owns gate columns `[wn * 32, wn * 32 + 32)` in
// its first 32 weight rows and the same up columns in its last 32, so each thread holds a
// gate/up pair for the same output element.
#include "common.cuh"

#ifndef SCRIPTOR_GATE_UP_RASTER
#define SCRIPTOR_GATE_UP_RASTER 8
#endif

#ifndef SCRIPTOR_GATE_UP_STAGES
#define SCRIPTOR_GATE_UP_STAGES 4
#endif

#ifndef SCRIPTOR_GATE_UP_STAGES_SMALL
#define SCRIPTOR_GATE_UP_STAGES_SMALL 3
#endif

namespace {

constexpr int kBlockM = 128;
constexpr int kBlockK = 32;
constexpr int kThreads = 256;
constexpr int kRasterGroup = SCRIPTOR_GATE_UP_RASTER;
constexpr int kATileBytes = kBlockM * kBlockK * 2;
constexpr int kRowBytes = kBlockK * 2;

// Tile shape of one instantiation: `BN` weight rows per block (`BN / 2` output columns, half
// gate and half up) and a `STAGES`-deep copy ring. Eight warps cover the 128 x BN tile; each
// warp owns 64 weight rows (32 gate + 32 up) and `128 / (8 / (BN / 64))` activation rows.
template <int BN, int STAGES>
struct Tile {
    static constexpr int kBlockN = BN;
    static constexpr int kOutCols = BN / 2;
    static constexpr int kStages = STAGES;
    static constexpr int kWarpsN = BN / 64;
    static constexpr int kWarpRows = kBlockM / (8 / kWarpsN);
    static constexpr int kMi = kWarpRows / 16;
    static constexpr int kBTileBytes = BN * kBlockK * 2;
    static constexpr int kStageBytes = kATileBytes + kBTileBytes;
};

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

// Byte offset of 16-byte chunk `chunk` (0..3) of tile row `row`.
__device__ __forceinline__ uint32_t tile_offset(int row, int chunk) {
    return static_cast<uint32_t>(row * kRowBytes + ((chunk ^ ((row >> 1) & 3)) << 4));
}

// Issues the copies of K step `step` into ring slot `slot`.
template <class T>
__device__ __forceinline__ void load_stage(uint32_t smem, int slot, const __nv_bfloat16* x,
                                           const __nv_bfloat16* w, int rows, int inter, int k,
                                           int m0, int n0, int step) {
    const uint32_t base = smem + slot * T::kStageBytes;
    const int c = threadIdx.x & 3;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int row = (threadIdx.x >> 2) + p * 64;
        const int gm = m0 + row;
        const bool valid = gm < rows;
        cp_async16(base + tile_offset(row, c),
                   x + (long)(valid ? gm : 0) * k + step * kBlockK + c * 8, valid);
    }
#pragma unroll
    for (int p = 0; p < T::kBlockN / 64; ++p) {
        const int row = (threadIdx.x >> 2) + p * 64;
        const int wn = row >> 6;
        const int within = row & 63;
        const long wrow = (within < 32 ? 0L : (long)inter) + n0 + wn * 32 + (within & 31);
        cp_async16(base + kATileBytes + tile_offset(row, c),
                   w + wrow * k + step * kBlockK + c * 8, true);
    }
}

}  // namespace

namespace {

template <class T>
__device__ __forceinline__ void gate_up_body(const __nv_bfloat16* __restrict__ x,
                                             const __nv_bfloat16* __restrict__ w,
                                             const __nv_bfloat16* __restrict__ bias,
                                             __nv_bfloat16* __restrict__ out, int rows, int inter,
                                             int k) {
    constexpr int kOutCols = T::kOutCols;
    constexpr int kStages = T::kStages;
    constexpr int kStageBytes = T::kStageBytes;
    constexpr int kMi = T::kMi;
    constexpr int kWarpRows = T::kWarpRows;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    // Blocks are walked in groups of `kRasterGroup` activation-column tiles that sweep every
    // row tile before the next group starts, so a group's weight rows stay resident in L2.
    const int linear = blockIdx.y * gridDim.x + blockIdx.x;
    const int group_span = kRasterGroup * gridDim.y;
    const int group = linear / group_span;
    const int group_cols = min(kRasterGroup, (int)gridDim.x - group * kRasterGroup);
    const int in_group = linear - group * group_span;
    const int n0 = (group * kRasterGroup + in_group % group_cols) * kOutCols;
    const int m0 = (in_group / group_cols) * kBlockM;
    const int steps = k / kBlockK;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int wm = warp / T::kWarpsN;
    const int wn = warp % T::kWarpsN;

    float acc[kMi][8][4];
#pragma unroll
    for (int i = 0; i < kMi; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < steps) load_stage<T>(smem, s, x, w, rows, inter, k, m0, n0, s);
        cp_async_commit();
    }

    const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int a_chunk = lane >> 4;
    const int b_row = (lane & 7) + (lane >> 4) * 8;
    const int b_chunk = (lane >> 3) & 1;

    for (int step = 0; step < steps; ++step) {
        cp_async_wait<kStages - 2>();
        __syncthreads();
        const int ahead = step + kStages - 1;
        if (ahead < steps) load_stage<T>(smem, ahead % kStages, x, w, rows, inter, k, m0, n0, ahead);
        cp_async_commit();

        const uint32_t a_base = smem + (step % kStages) * kStageBytes;
        const uint32_t b_base = a_base + kATileBytes;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            uint32_t a[kMi][4];
            uint32_t b[4][4];
#pragma unroll
            for (int mi = 0; mi < kMi; ++mi)
                ldmatrix_x4(a[mi], a_base + tile_offset(wm * kWarpRows + mi * 16 + a_row, ks * 2 + a_chunk));
#pragma unroll
            for (int nj = 0; nj < 4; ++nj)
                ldmatrix_x4(b[nj], b_base + tile_offset(wn * 64 + nj * 16 + b_row, ks * 2 + b_chunk));
#pragma unroll
            for (int mi = 0; mi < kMi; ++mi)
#pragma unroll
                for (int nt = 0; nt < 8; ++nt)
                    mma_16816(acc[mi][nt], a[mi], b[nt >> 1][(nt & 1) * 2], b[nt >> 1][(nt & 1) * 2 + 1]);
        }
    }

    const int g = lane >> 2;
    const int q = (lane & 3) * 2;
#pragma unroll
    for (int mi = 0; mi < kMi; ++mi) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = m0 + wm * kWarpRows + mi * 16 + g + half * 8;
            if (row >= rows) continue;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int col = n0 + wn * 32 + nt * 8 + q;
                float bg[2] = {0.0f, 0.0f};
                float bu[2] = {0.0f, 0.0f};
                if (bias != nullptr) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        bg[e] = scriptor_to_f32(bias[col + e]);
                        bu[e] = scriptor_to_f32(bias[inter + col + e]);
                    }
                }
                float v[2];
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float gate = scriptor_to_f32(scriptor_to_bf16(acc[mi][nt][half * 2 + e] + bg[e]));
                    const float up = scriptor_to_f32(scriptor_to_bf16(acc[mi][nt + 4][half * 2 + e] + bu[e]));
                    v[e] = scriptor_silu(gate) * up;
                }
                *reinterpret_cast<__nv_bfloat162*>(out + (long)row * inter + col) =
                    __floats2bfloat162_rn(v[0], v[1]);
            }
        }
    }
}

}  // namespace

// Requires `inter % 128 == 0` and `k % 32 == 0`; `rows` is arbitrary. Launch with grid
// `(inter / 128, ceil(rows / 128))`, 256 threads and `SCRIPTOR_GATE_UP_STAGES * 24576` bytes of
// dynamic shared memory: a 128 x 128-output tile with one block per multiprocessor.
extern "C" __global__ void __launch_bounds__(kThreads, 1) scriptor_vision_gate_up_silu_bf16(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out, int rows,
    int inter, int k) {
    gate_up_body<Tile<256, SCRIPTOR_GATE_UP_STAGES>>(x, w, bias, out, rows, inter, k);
}

// The same arithmetic over a 128 x 64-output tile sized for two resident blocks per
// multiprocessor. Requires `inter % 64 == 0` and `k % 32 == 0`; launch with grid
// `(inter / 64, ceil(rows / 128))`, 256 threads and `SCRIPTOR_GATE_UP_STAGES_SMALL * 16384` bytes
// of dynamic shared memory.
extern "C" __global__ void __launch_bounds__(kThreads, 2) scriptor_vision_gate_up_silu_bf16_small(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out, int rows,
    int inter, int k) {
    gate_up_body<Tile<128, SCRIPTOR_GATE_UP_STAGES_SMALL>>(x, w, bias, out, rows, inter, k);
}
