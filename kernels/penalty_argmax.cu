// Logit post-processing: repetition / presence / frequency penalties from
// per-sequence token bitsets, then greedy argmax with a top-2 margin or
// temperature / top-k / top-p sampling with a seeded generator.
#include "row_scan.cuh"

#define SCRIPTOR_SAMPLE_BLOCK 1024

// Maps a float to an unsigned key that orders like the float.
__device__ __forceinline__ uint32_t scriptor_order_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Uniform in [0, 1) from (seed, step): splitmix64 finalizer, top 24 bits.
__device__ __forceinline__ float scriptor_uniform(uint64_t seed, uint32_t step) {
    uint64_t z = seed + (uint64_t)(step + 1u) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);
    return (float)(uint32_t)(z >> 40) * (1.0f / 16777216.0f);
}

// Sum over the row of exp((x - m) * inv_t) for elements whose order key is at
// least `threshold`.
template <typename Get>
__device__ float scriptor_mass_at(Get at, int vocab, uint32_t threshold, float m, float inv_t,
                            float* scratch) {
    float acc = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float v = at(i);
        if (scriptor_order_key(v) >= threshold) acc += __expf((v - m) * inv_t);
    }
    return scriptor_block_sum(acc, scratch);
}

// Processes one logits row: penalties, then argmax or sampling. With `BF16` false `logits` is
// fp32 and the penalties are written back in place; with `BF16` true it is a bf16 matrix read
// directly, and every read of an element recomputes its penalised value from the widened bf16
// bits, which equals the in-place fp32 value. Rows are grouped `q`
// per sequence: row `r` belongs to sequence `r / q`, reads that sequence's parameters, and
// treats the first `r % q` entries of the sequence's `k` drafts as already emitted output
// and advances the sampling step by `r % q`. `drafts` may be null when `q == 1`.
template <bool BF16>
__device__ void scriptor_process_row(int row, bool greedy_only, void* logits, int vocab,
                               const ScriptorRowParams* params, const uint32_t* prompt_bits,
                               const uint32_t* output_bits, const uint32_t* output_counts,
                               int words, const uint32_t* drafts, int k, int q,
                               uint32_t* out_tokens, float* out_margins) {
    __shared__ float sf[32];
    __shared__ ScriptorTop2 st[32];
    __shared__ float chunk_sum[SCRIPTOR_SAMPLE_BLOCK];
    __shared__ uint32_t s_token;
    __shared__ uint32_t s_last;

    const ScriptorRowView view = scriptor_row_view(row, q, vocab, params, prompt_bits,
                                                   output_bits, output_counts, words, drafts, k);
    const ScriptorRowParams p = view.p;
    const int ahead = view.ahead;
    float* x = BF16 ? nullptr : static_cast<float*>(logits) + (size_t)row * (size_t)vocab;
    const uint16_t* xh =
        BF16 ? static_cast<const uint16_t*>(logits) + (size_t)row * (size_t)vocab : nullptr;
    // Penalised element `i` of the row.
    auto at = [&](int i) -> float {
        if (!BF16) return x[i];
        return view.at_bf16(xh, i);
    };

    const ScriptorTop2 local = BF16 ? scriptor_scan_bf16(view, xh, vocab, threadIdx.x, blockDim.x)
                                    : scriptor_scan_f32(view, x, vocab, threadIdx.x, blockDim.x);
    const ScriptorTop2 top = scriptor_block_top2(local, st);
    const float best = top.best;
    const uint32_t argmax = top.index;
    const float second = top.second;

    if (greedy_only || p.temperature <= 0.0f) {
        if (threadIdx.x == 0) {
            out_tokens[row] = argmax;
            out_margins[row] = best - second;
        }
        return;
    }

    const float inv_t = 1.0f / p.temperature;
    uint32_t threshold = 0u;
    if (p.top_k > 0 && p.top_k < vocab) {
        uint32_t t = 0u;
        for (int bit = 31; bit >= 0; --bit) {
            const uint32_t cand = t | (1u << bit);
            float count = 0.0f;
            for (int i = threadIdx.x; i < vocab; i += blockDim.x)
                if (scriptor_order_key(at(i)) >= cand) count += 1.0f;
            if (scriptor_block_sum(count, sf) >= (float)p.top_k) t = cand;
        }
        threshold = t;
    }
    if (p.top_p < 1.0f) {
        const float z = scriptor_mass_at(at, vocab, threshold, best, inv_t, sf);
        const float want = p.top_p * z;
        uint32_t t = 0u;
        for (int bit = 31; bit >= 0; --bit) {
            const uint32_t cand = t | (1u << bit);
            if (scriptor_mass_at(at, vocab, max(cand, threshold), best, inv_t, sf) >= want) t = cand;
        }
        threshold = t;
    }
    const float total = scriptor_mass_at(at, vocab, threshold, best, inv_t, sf);
    const float target = scriptor_uniform(p.seed, p.step + (uint32_t)ahead) * total;

    const int chunk = (vocab + (int)blockDim.x - 1) / (int)blockDim.x;
    const int start = min((int)threadIdx.x * chunk, vocab);
    const int end = min(start + chunk, vocab);
    float mine = 0.0f;
    for (int i = start; i < end; ++i) {
        const float v = at(i);
        if (scriptor_order_key(v) >= threshold) mine += __expf((v - best) * inv_t);
    }
    chunk_sum[threadIdx.x] = mine;
    if (threadIdx.x == 0) {
        s_token = 0xffffffffu;
        s_last = 0u;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float run = 0.0f;
        for (int t = 0; t < (int)blockDim.x; ++t) {
            const float s = chunk_sum[t];
            chunk_sum[t] = run;
            run += s;
        }
    }
    __syncthreads();
    float acc = chunk_sum[threadIdx.x];
    for (int i = start; i < end; ++i) {
        const float v = at(i);
        if (scriptor_order_key(v) >= threshold) {
            acc += __expf((v - best) * inv_t);
            atomicMax(&s_last, (uint32_t)i);
            if (acc > target) {
                atomicMin(&s_token, (uint32_t)i);
                break;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        out_tokens[row] = s_token != 0xffffffffu ? s_token : s_last;
        out_margins[row] = best - second;
    }
}

// Greedy path: penalties, then argmax with the top-2 margin, one block per row.
extern "C" __global__ void scriptor_penalty_argmax(float* logits, int vocab, const ScriptorRowParams* params,
                                             const uint32_t* prompt_bits,
                                             const uint32_t* output_bits,
                                             const uint32_t* output_counts, int words,
                                             uint32_t* out_tokens, float* out_margins) {
    scriptor_process_row<false>(blockIdx.x, true, logits, vocab, params, prompt_bits, output_bits,
                          output_counts, words, nullptr, 0, 1, out_tokens, out_margins);
}

// As scriptor_penalty_argmax over a bf16 logits matrix, which is left unmodified.
extern "C" __global__ void scriptor_penalty_argmax_bf16(const uint16_t* logits, int vocab,
                                                  const ScriptorRowParams* params,
                                                  const uint32_t* prompt_bits,
                                                  const uint32_t* output_bits,
                                                  const uint32_t* output_counts, int words,
                                                  uint32_t* out_tokens, float* out_margins) {
    scriptor_process_row<true>(blockIdx.x, true, const_cast<uint16_t*>(logits), vocab, params,
                         prompt_bits, output_bits, output_counts, words, nullptr, 0, 1,
                         out_tokens, out_margins);
}

// Per-row choice: argmax where temperature is zero, seeded sampling otherwise.
extern "C" __global__ void scriptor_sample_rows(float* logits, int vocab, const ScriptorRowParams* params,
                                          const uint32_t* prompt_bits,
                                          const uint32_t* output_bits,
                                          const uint32_t* output_counts, int words,
                                          uint32_t* out_tokens, float* out_margins) {
    scriptor_process_row<false>(blockIdx.x, false, logits, vocab, params, prompt_bits, output_bits,
                          output_counts, words, nullptr, 0, 1, out_tokens, out_margins);
}

// As scriptor_sample_rows over a bf16 logits matrix, which is left unmodified.
extern "C" __global__ void scriptor_sample_rows_bf16(const uint16_t* logits, int vocab,
                                               const ScriptorRowParams* params,
                                               const uint32_t* prompt_bits,
                                               const uint32_t* output_bits,
                                               const uint32_t* output_counts, int words,
                                               uint32_t* out_tokens, float* out_margins) {
    scriptor_process_row<true>(blockIdx.x, false, const_cast<uint16_t*>(logits), vocab, params,
                         prompt_bits, output_bits, output_counts, words, nullptr, 0, 1,
                         out_tokens, out_margins);
}

// Speculative verification rows: `q = k + 1` rows per sequence, one parameter entry per
// sequence, argmax where temperature is zero and seeded sampling otherwise.
extern "C" __global__ void scriptor_sample_rows_spec(float* logits, int vocab, const ScriptorRowParams* params,
                                               const uint32_t* prompt_bits,
                                               const uint32_t* output_bits,
                                               const uint32_t* output_counts, int words,
                                               const uint32_t* drafts, int k, int q,
                                               uint32_t* out_tokens, float* out_margins) {
    scriptor_process_row<false>(blockIdx.x, false, logits, vocab, params, prompt_bits, output_bits,
                          output_counts, words, drafts, k, q, out_tokens, out_margins);
}

// As scriptor_sample_rows_spec over a bf16 logits matrix, which is left unmodified.
extern "C" __global__ void scriptor_sample_rows_spec_bf16(
    const uint16_t* logits, int vocab, const ScriptorRowParams* params, const uint32_t* prompt_bits,
    const uint32_t* output_bits, const uint32_t* output_counts, int words,
    const uint32_t* drafts, int k, int q, uint32_t* out_tokens, float* out_margins) {
    scriptor_process_row<true>(blockIdx.x, false, const_cast<uint16_t*>(logits), vocab, params,
                         prompt_bits, output_bits, output_counts, words, drafts, k, q,
                         out_tokens, out_margins);
}

// Records accepted tokens: sets the output bit and bumps the output count of
// `slots[i]` for `tokens[i]`; entries with slot 0xffffffff are skipped.
extern "C" __global__ void scriptor_mark_tokens(uint32_t* output_bits, uint32_t* output_counts,
                                          const uint32_t* slots, const uint32_t* tokens, int n,
                                          int words, int vocab) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t token = tokens[i];
    if (token >= (uint32_t)vocab || slots[i] == 0xffffffffu) return;
    const size_t slot = slots[i];
    atomicOr(&output_bits[slot * (size_t)words + (token >> 5)], 1u << (token & 31));
    if (output_counts) atomicAdd(&output_counts[slot * (size_t)vocab + token], 1u);
}
