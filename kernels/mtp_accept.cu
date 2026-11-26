// Speculative-draft acceptance: per sequence, the longest accepted prefix of
// the drafted tokens followed by the verifier's token at the first rejection
// (or the bonus token when every draft is accepted).
#include "common.cuh"

#define SCRIPTOR_NO_TOKEN 0xffffffffu

// One thread per sequence.
//   draft        [rows, k]      drafted tokens
//   target       [rows, k + 1]  verifier token per position: the argmax for
//                               greedy rows, the residual / plain sample for
//                               sampled rows
//   draft_counts [rows]         drafts actually proposed per row (<= k)
//   accept_prob  [rows, k]      probability the verifier assigns the drafted
//                               token, or null for exact-match acceptance
//   uniforms     [rows, k]      acceptance draws in [0, 1), read with accept_prob
//   out_tokens   [rows, k + 1]  emitted tokens, padded with 0xffffffff
//   out_counts   [rows]         number of emitted tokens (accepted + 1)
extern "C" __global__ void scriptor_mtp_accept(const uint32_t* draft, const uint32_t* target,
                                         const int32_t* draft_counts, const float* accept_prob,
                                         const float* uniforms, int rows, int k,
                                         uint32_t* out_tokens, uint32_t* out_counts) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int stride = k + 1;
    const int proposed = min(max(draft_counts[row], 0), k);
    int accepted = 0;
    for (; accepted < proposed; ++accepted) {
        const size_t at = (size_t)row * k + accepted;
        const bool ok = accept_prob ? (uniforms[at] < accept_prob[at])
                                    : (draft[at] == target[(size_t)row * stride + accepted]);
        if (!ok) break;
    }
    uint32_t* out = out_tokens + (size_t)row * stride;
    for (int j = 0; j < accepted; ++j) out[j] = draft[(size_t)row * k + j];
    out[accepted] = target[(size_t)row * stride + accepted];
    for (int j = accepted + 1; j < stride; ++j) out[j] = SCRIPTOR_NO_TOKEN;
    out_counts[row] = (uint32_t)(accepted + 1);
}
