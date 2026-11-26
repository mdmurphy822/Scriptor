// Embedding gather and image-embedding scatter. bf16 rows.
#include "common.cuh"

// One block per id: `out[t] = table[ids[t]]`; ids outside `[0, vocab)` give a zero row.
extern "C" __global__ void scriptor_embedding_gather_bf16(
    const __nv_bfloat16* table, const int* ids, __nv_bfloat16* out, int dim, int vocab) {
    const long t = blockIdx.x;
    const int id = ids[t];
    const bool valid = id >= 0 && id < vocab;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out[t * dim + i] = valid ? table[(long)id * dim + i] : scriptor_to_bf16(0.0f);
    }
}

// One block per image row: `embeds[positions[i]] = image_embeds[i]`; negative
// positions are skipped.
extern "C" __global__ void scriptor_image_scatter_bf16(
    __nv_bfloat16* embeds, const __nv_bfloat16* image_embeds, const int* positions, int dim) {
    const long i = blockIdx.x;
    const long pos = positions[i];
    if (pos < 0) return;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        embeds[pos * dim + d] = image_embeds[i * dim + d];
    }
}
