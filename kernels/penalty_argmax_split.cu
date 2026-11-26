// Two-stage greedy target selection over a bf16 logits matrix for low row counts. Stage 1
// spreads the scan of each row over several blocks: every virtual thread of the single-block
// kernel's width keeps its exact element set, order and penalties and stores its running top-2
// unchanged. Stage 2 runs the single-block kernel's block reduction over those stored partials,
// so tokens and margins equal scriptor_sample_rows_spec_bf16 on rows whose temperature is not
// positive.
#include "row_scan.cuh"

// Scans the elements owned by the virtual threads of one row slice. Grid x indexes the slice of
// `vt_per_cta` consecutive virtual threads (of `virtual_width` in all), grid y indexes the row;
// each physical thread walks its slice's virtual threads in steps of `blockDim.x` and writes
// the top-2 of virtual thread `t` to `partials[row * virtual_width + t]`. Every stride of the
// scan is `virtual_width`, never the physical block width.
extern "C" __global__ void scriptor_spec_scan_bf16(
    const uint16_t* logits, int vocab, const ScriptorRowParams* params, const uint32_t* prompt_bits,
    const uint32_t* output_bits, const uint32_t* output_counts, int words, const uint32_t* drafts,
    int k, int q, int virtual_width, int vt_per_cta, ScriptorTop2* partials) {
    const int row = blockIdx.y;
    const int lo = blockIdx.x * vt_per_cta;
    const int hi = min(lo + vt_per_cta, virtual_width);
    const ScriptorRowView view = scriptor_row_view(row, q, vocab, params, prompt_bits, output_bits,
                                                   output_counts, words, drafts, k);
    const uint16_t* xh = logits + (size_t)row * (size_t)vocab;
    ScriptorTop2* out = partials + (size_t)row * (size_t)virtual_width;
    for (int t = lo + (int)threadIdx.x; t < hi; t += (int)blockDim.x)
        out[t] = scriptor_scan_bf16(view, xh, vocab, t, virtual_width);
}

// Reduces the partials of one row (grid x) with `blockDim.x == virtual_width` threads, thread
// `t` holding virtual thread `t`'s top-2, through the block reduction of the single-block
// kernel, then writes the argmax token and the best-second margin.
extern "C" __global__ void scriptor_spec_reduce_bf16(const ScriptorTop2* partials,
                                                     uint32_t* out_tokens, float* out_margins) {
    __shared__ ScriptorTop2 st[32];
    const int row = blockIdx.x;
    const ScriptorTop2 local = partials[(size_t)row * (size_t)blockDim.x + threadIdx.x];
    const ScriptorTop2 top = scriptor_block_top2(local, st);
    if (threadIdx.x == 0) {
        out_tokens[row] = top.index;
        out_margins[row] = top.best - top.second;
    }
}
