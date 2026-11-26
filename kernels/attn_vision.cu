// Flash-style non-causal attention over the packed vision `qkv` buffer.
//
// `qkv` is `[tokens, 3, heads, 64]` bf16 (q, k, v concatenated per token); `out` is
// `[tokens, heads, 64]`. One thread block (8 warps) owns 128 consecutive query rows of one head
// of one sequence; each warp owns 16 rows. Key and value tiles of 64 rows are staged through
// shared memory with a two-stage `cp.async` pipeline and read as `ldmatrix` fragments. Scores
// and the output accumulate in fp32 through bf16 `mma.sync.m16n8k16` tiles; the softmax is
// online, in the exp2 domain, with probabilities rounded to bf16 per tile.
#include "common.cuh"

namespace {

constexpr int kHeadDim = 64;
constexpr int kBlockQ = 128;
constexpr int kBlockKv = 64;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * SCRIPTOR_WARP_SIZE;
constexpr int kStride = kHeadDim + 8;  // padded shared-memory row, in bf16 elements
constexpr int kChunksPerRow = kHeadDim / 8;
constexpr int kKSteps = kHeadDim / 16;
constexpr int kDTiles = kHeadDim / 8;
constexpr float kLog2e = 1.4426950408889634f;

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

// 2^x through the hardware approximation; inputs are at most 0 and -inf maps to 0.
__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// Copies 16 bytes global -> shared; zero-fills the destination when `valid` is false.
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem),
                 "r"(bytes));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Loads four 8x8 b16 matrices from shared memory; lane `l` supplies the row address of
// matrix `l / 8`, row `l % 8`.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// As `ldmatrix_x4`, delivering each matrix transposed.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* smem) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// Builds the A-operand fragments of this thread's two query rows (`q0`, `q1`: pointers to the
// raw bf16 rows of one head; `pos0`, `pos1`: their [row, col] patch positions) from the
// per-head RMSNorm (`q_w`, 64 weights) and 2-D axial NeoX RoPE of the raw rows, with the same
// bf16 rounding points as the standalone q/k norm + RoPE kernel. Rows flagged invalid yield zeros.
// Fragment layout: qa[ks][0/1] = row0/row1 columns (ks*16+2t, +1); qa[ks][2/3] = columns +8, +9.
__device__ __forceinline__ void load_q_norm_rope(
    uint32_t (&qa)[kKSteps][4], const __nv_bfloat16* q0, const __nv_bfloat16* q1, bool valid0,
    bool valid1, const int* pos0, const int* pos1, const __nv_bfloat16* q_w, float theta,
    float eps, int t) {
    float x[2][kKSteps][4];
    float ss[2] = {0.0f, 0.0f};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const __nv_bfloat16* row = r == 0 ? q0 : q1;
        const bool valid = r == 0 ? valid0 : valid1;
#pragma unroll
        for (int ks = 0; ks < kKSteps; ++ks) {
            const int c = ks * 16 + 2 * t;
            const uint32_t w0 = valid ? *reinterpret_cast<const uint32_t*>(row + c) : 0u;
            const uint32_t w1 = valid ? *reinterpret_cast<const uint32_t*>(row + c + 8) : 0u;
            const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&w0);
            const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&w1);
            x[r][ks][0] = __low2float(b0);
            x[r][ks][1] = __high2float(b0);
            x[r][ks][2] = __low2float(b1);
            x[r][ks][3] = __high2float(b1);
#pragma unroll
            for (int e = 0; e < 4; ++e) ss[r] += x[r][ks][e] * x[r][ks][e];
        }
        ss[r] += __shfl_xor_sync(SCRIPTOR_FULL_MASK, ss[r], 1);
        ss[r] += __shfl_xor_sync(SCRIPTOR_FULL_MASK, ss[r], 2);
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const float inv = rsqrtf(ss[r] / (float)kHeadDim + eps);
        const int* pos = r == 0 ? pos0 : pos1;
#pragma unroll
        for (int ks = 0; ks < kKSteps; ++ks)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int col = ks * 16 + 2 * t + (e & 1) + (e >> 1) * 8;
                x[r][ks][e] = scriptor_to_f32(scriptor_to_bf16(x[r][ks][e] * inv * scriptor_to_f32(q_w[col])));
            }
#pragma unroll
        for (int ks = 0; ks < kKSteps / 2; ++ks) {
            const int axis_pos = pos[ks];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int k_index = 2 * t + (e & 1) + (e >> 1) * 8;
                const float freq = powf(theta, -2.0f * (float)k_index / (float)(kHeadDim / 2));
                float sn, cs;
                sincosf((float)axis_pos * freq, &sn, &cs);
                const float a = x[r][ks][e], b = x[r][ks + 2][e];
                x[r][ks][e] = a * cs - b * sn;
                x[r][ks + 2][e] = a * sn + b * cs;
            }
        }
#pragma unroll
        for (int ks = 0; ks < kKSteps; ++ks) {
            qa[ks][r] = pack_bf16(x[r][ks][0], x[r][ks][1]);
            qa[ks][2 + r] = pack_bf16(x[r][ks][2], x[r][ks][3]);
        }
    }
}

}  // namespace

// Grid: (ceil(max_seqlen / 128), heads, num_seqs); block: 256 threads.
// `cu_seqlens` is `[num_seqs + 1]` int32 token offsets. When `positions` (`[tokens, 2]` int32)
// is non-null the q section of `qkv` is raw: each query row is RMS-normalised with `q_w` and
// rotated by 2-D RoPE (`theta`, `eps`) as it is loaded; otherwise q is used as stored.
extern "C" __global__ void __launch_bounds__(kThreads) scriptor_attn_vision_d64(
    const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ out,
    const int* __restrict__ cu_seqlens, int heads, float scale,
    const int* __restrict__ positions, const __nv_bfloat16* __restrict__ q_w, float theta,
    float eps) {
    __shared__ __align__(16) __nv_bfloat16 sk[2][kBlockKv][kStride];
    __shared__ __align__(16) __nv_bfloat16 sv[2][kBlockKv][kStride];

    const int seq = blockIdx.z;
    const int head = blockIdx.y;
    const int start = cu_seqlens[seq];
    const int len = cu_seqlens[seq + 1] - start;
    const int q0 = blockIdx.x * kBlockQ;
    if (q0 >= len) return;

    const size_t row_stride = (size_t)3 * heads * kHeadDim;
    const size_t out_stride = (size_t)heads * kHeadDim;
    const __nv_bfloat16* qbase = qkv + (size_t)start * row_stride + (size_t)head * kHeadDim;
    const __nv_bfloat16* kbase = qbase + (size_t)heads * kHeadDim;
    const __nv_bfloat16* vbase = kbase + (size_t)heads * kHeadDim;

    const int tid = threadIdx.x;
    const int warp = tid / SCRIPTOR_WARP_SIZE;
    const int lane = tid % SCRIPTOR_WARP_SIZE;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int lm = lane >> 3;  // ldmatrix matrix index
    const int lr = lane & 7;   // ldmatrix row index

    const int ntiles = (len + kBlockKv - 1) / kBlockKv;

    auto load_tile = [&](int tile, int stage) {
        const int kv0 = tile * kBlockKv;
        for (int idx = tid; idx < kBlockKv * kChunksPerRow; idx += kThreads) {
            const int j = idx / kChunksPerRow;
            const int c = idx % kChunksPerRow;
            const int key = kv0 + j;
            const bool valid = key < len;
            const size_t off = (size_t)(valid ? key : 0) * row_stride + c * 8;
            cp_async16(&sk[stage][j][c * 8], kbase + off, valid);
            cp_async16(&sv[stage][j][c * 8], vbase + off, valid);
        }
    };

    load_tile(0, 0);
    cp_async_commit();

    const int row0 = q0 + warp * 16 + g;
    const int row1 = row0 + 8;
    const float sl2 = scale * kLog2e;

    uint32_t qa[kKSteps][4];
    if (positions != nullptr) {
        const int r0 = row0 < len ? row0 : 0, r1 = row1 < len ? row1 : 0;
        load_q_norm_rope(qa, qbase + (size_t)r0 * row_stride, qbase + (size_t)r1 * row_stride,
                         row0 < len, row1 < len, positions + (size_t)(start + r0) * 2,
                         positions + (size_t)(start + r1) * 2, q_w, theta, eps, t);
    } else {
#pragma unroll
        for (int ks = 0; ks < kKSteps; ++ks) {
            const int c = ks * 16 + 2 * t;
            const __nv_bfloat16* p0 = qbase + (size_t)row0 * row_stride + c;
            const __nv_bfloat16* p1 = qbase + (size_t)row1 * row_stride + c;
            qa[ks][0] = row0 < len ? *reinterpret_cast<const uint32_t*>(p0) : 0u;
            qa[ks][1] = row1 < len ? *reinterpret_cast<const uint32_t*>(p1) : 0u;
            qa[ks][2] = row0 < len ? *reinterpret_cast<const uint32_t*>(p0 + 8) : 0u;
            qa[ks][3] = row1 < len ? *reinterpret_cast<const uint32_t*>(p1 + 8) : 0u;
        }
    }

    float o[kDTiles][4];
#pragma unroll
    for (int i = 0; i < kDTiles; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    for (int tile = 0; tile < ntiles; ++tile) {
        const int stage = tile & 1;
        const int kv0 = tile * kBlockKv;
        if (tile + 1 < ntiles) {
            load_tile(tile + 1, stage ^ 1);
            cp_async_commit();
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        float s[kBlockKv / 8][4];
#pragma unroll
        for (int nt = 0; nt < kBlockKv / 8; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < kKSteps; ++ks) {
#pragma unroll
            for (int np = 0; np < kBlockKv / 16; ++np) {
                uint32_t b[4];
                ldmatrix_x4(b, &sk[stage][(np * 2 + (lm >> 1)) * 8 + lr][ks * 16 + (lm & 1) * 8]);
                mma_16816(s[2 * np], qa[ks], b[0], b[1]);
                mma_16816(s[2 * np + 1], qa[ks], b[2], b[3]);
            }
        }

        const bool tail = kv0 + kBlockKv > len;
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < kBlockKv / 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int key = kv0 + nt * 8 + 2 * t + (e & 1);
                const float x = (!tail || key < len) ? s[nt][e] * sl2 : -INFINITY;
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
        const float a0 = fast_exp2(m0 - u0), a1 = fast_exp2(m1 - u1);
        m0 = n0;
        m1 = n1;

        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < kBlockKv / 8; ++nt) {
            s[nt][0] = fast_exp2(s[nt][0] - u0);
            s[nt][1] = fast_exp2(s[nt][1] - u0);
            s[nt][2] = fast_exp2(s[nt][2] - u1);
            s[nt][3] = fast_exp2(s[nt][3] - u1);
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
            for (int dp = 0; dp < kDTiles / 2; ++dp) {
                uint32_t b[4];
                ldmatrix_x4_trans(b, &sv[stage][kk * 16 + (lm & 1) * 8 + lr][(dp * 2 + (lm >> 1)) * 8]);
                mma_16816(o[2 * dp], pa, b[0], b[1]);
                mma_16816(o[2 * dp + 1], pa, b[2], b[3]);
            }
        }
        __syncthreads();
    }

    l0 += __shfl_xor_sync(SCRIPTOR_FULL_MASK, l0, 1);
    l0 += __shfl_xor_sync(SCRIPTOR_FULL_MASK, l0, 2);
    l1 += __shfl_xor_sync(SCRIPTOR_FULL_MASK, l1, 1);
    l1 += __shfl_xor_sync(SCRIPTOR_FULL_MASK, l1, 2);
    const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f;
    const float inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    __nv_bfloat16* obase = out + (size_t)start * out_stride + (size_t)head * kHeadDim;
#pragma unroll
    for (int dt = 0; dt < kDTiles; ++dt) {
        const int col = dt * 8 + 2 * t;
        if (row0 < len) {
            *reinterpret_cast<uint32_t*>(obase + (size_t)row0 * out_stride + col) =
                pack_bf16(o[dt][0] * inv0, o[dt][1] * inv0);
        }
        if (row1 < len) {
            *reinterpret_cast<uint32_t*>(obase + (size_t)row1 * out_stride + col) =
                pack_bf16(o[dt][2] * inv1, o[dt][3] * inv1);
        }
    }
}
