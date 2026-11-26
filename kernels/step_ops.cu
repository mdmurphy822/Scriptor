// Row bookkeeping of the device-resident decode step: input rows of the verification pass, the
// MTP layer passes and the packed per-step result, all derived on the device from the sequence
// state so a captured graph replays with only state buffers rewritten.
#include "common.cuh"

#define SCRIPTOR_NO_TOKEN 0xffffffffu

// Flat KV slot of position `pos` of sequence `seq`, or -1 when the sequence is inactive or the
// position lies beyond the `limit` tokens its block table backs.
__device__ __forceinline__ int scriptor_slot_of(const int* tables, int stride, int block_size, int seq,
                                          int pos, int limit, bool active) {
    if (!active || pos < 0 || pos >= limit) return -1;
    const int block = pos / block_size;
    if (block >= stride) return -1;
    return tables[(size_t)seq * stride + block] * block_size + pos % block_size;
}

// Writes the rotary position of a text row into all three components.
__device__ __forceinline__ void scriptor_put_position(int* positions, int row, int pos) {
    positions[3 * row] = pos;
    positions[3 * row + 1] = pos;
    positions[3 * row + 2] = pos;
}

// Verification rows: sequence `s` contributes `q = k + 1` rows, the last accepted token followed
// by its `k` drafts, at consecutive positions starting at `cached[s]`. A negative `cached`
// marks an inactive row, which writes no KV, attends over block zero and records no output
// token (`mark_slots` is the sequence index of an active row and 0xffffffff otherwise).
extern "C" __global__ void scriptor_step_verify_prepare(
    const int* cached, const int* limit, const uint32_t* last, const int* delta,
    const uint32_t* drafts, const int* tables, int stride, int block_size, int q, int k, int rows,
    int* ids, int* positions, int* slots, int* seq_lens, int* sample_rows,
    uint32_t* mark_slots) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int s = r / q;
    const int j = r % q;
    const int c = cached[s];
    const bool active = c >= 0;
    const int pos = active ? c + j : 0;
    ids[r] = active ? (int)(j == 0 ? last[s] : drafts[(size_t)s * k + j - 1]) : 0;
    scriptor_put_position(positions, r, active ? pos + delta[s] : 0);
    slots[r] = scriptor_slot_of(tables, stride, block_size, s, pos, limit[s], active);
    if (j == 0) seq_lens[s] = active ? c + q : q;
    sample_rows[r] = r;
    mark_slots[r] = active ? (uint32_t)s : SCRIPTOR_NO_TOKEN;
}

// MTP layer rows consuming the verification hidden states: row `j` of sequence `s` pairs the
// hidden state at position `cached[s] + j` with the token that follows it, which is the emitted
// token `j` while `j` is below the emitted count and the last emitted token after that. The
// sampled row of the sequence is the one for its last emitted token.
extern "C" __global__ void scriptor_step_mtp_prepare(
    const int* cached, const int* limit, const int* delta, const uint32_t* emitted,
    const uint32_t* counts, const int* tables, int stride, int block_size, int q, int rows,
    int* ids, int* positions, int* slots, int* seq_lens, int* sample_rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int s = r / q;
    const int j = r % q;
    const int c = cached[s];
    const bool active = c >= 0;
    const int e = active ? (int)counts[s] : 1;
    const int pos = active ? c + j : 0;
    ids[r] = active ? (int)emitted[(size_t)s * q + min(j, e - 1)] : 0;
    scriptor_put_position(positions, r, active ? pos + delta[s] : 0);
    slots[r] = scriptor_slot_of(tables, stride, block_size, s, pos, limit[s], active);
    if (j == 0) {
        seq_lens[s] = active ? c + q : q;
        sample_rows[s] = s * q + e - 1;
    }
}

// MTP chain pass `pass` (1-based): one row per sequence holding the previous draft, at the
// position after the last emitted token advanced by `pass - 1`. Positions the block table does
// not back write no KV and attend over the backed tokens only.
extern "C" __global__ void scriptor_step_chain_prepare(
    const int* cached, const int* limit, const int* delta, const uint32_t* counts,
    const uint32_t* drafts, const int* tables, int stride, int block_size, int k, int pass,
    int seqs, int* ids, int* positions, int* slots, int* seq_lens, int* sample_rows) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= seqs) return;
    const int c = cached[s];
    const bool active = c >= 0;
    const int pos = active ? c + (int)counts[s] - 1 + pass : 0;
    ids[s] = active ? (int)drafts[(size_t)s * k + pass - 1] : 0;
    scriptor_put_position(positions, s, active ? pos + delta[s] : 0);
    slots[s] = scriptor_slot_of(tables, stride, block_size, s, pos, limit[s], active);
    seq_lens[s] = active ? min(pos + 1, limit[s]) : 1;
    sample_rows[s] = s;
}

// Stores one draft token per sequence at draft index `index`.
extern "C" __global__ void scriptor_step_store_drafts(const uint32_t* tokens, uint32_t* drafts, int k,
                                                int index, int seqs) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= seqs) return;
    drafts[(size_t)s * k + index] = tokens[s];
}

// Packs the step result per sequence as `1 + 2q` words: the emitted count, `q` emitted token ids
// (padded) and the `q` top-2 margins of the verification rows as float bits.
extern "C" __global__ void scriptor_step_pack(const uint32_t* emitted, const uint32_t* counts,
                                        const float* margins, int seqs, int q, uint32_t* result) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seqs * (1 + 2 * q)) return;
    const int width = 1 + 2 * q;
    const int s = i / width;
    const int w = i % width;
    if (w == 0) {
        result[i] = counts[s];
    } else if (w <= q) {
        result[i] = emitted[(size_t)s * q + w - 1];
    } else {
        result[i] = __float_as_uint(margins[(size_t)s * q + w - 1 - q]);
    }
}
