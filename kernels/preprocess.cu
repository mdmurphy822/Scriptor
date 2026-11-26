// Batched pixel normalisation and patchify.
//
// One launch converts many interleaved RGB u8 images into the concatenated
// bf16 patch matrix `[sum(patches), 1176]`. Each image is described by four
// 64-bit words: byte offset of its pixels in the source buffer, row offset of
// its first patch in the output, height, width. Patches follow the order
// (h/28, w/28, 2, 2), each flattened as (c, t=2, 14, 14) with the single frame
// repeated along t. Every output element is written exactly once.
#include "common.cuh"

#define SCRIPTOR_PATCH 14
#define SCRIPTOR_PATCH_ELEMS 1176

struct ScriptorPatchifyDesc {
    long long src_offset;
    long long dst_patch;
    long long height;
    long long width;
};

// grid.x: patch index (blocks past an image's patch count exit); grid.y: image.
// block: 196 threads, one per pixel of the patch.
extern "C" __global__ void scriptor_normalise_patchify_bf16(
    const uint8_t* __restrict__ rgb,
    const ScriptorPatchifyDesc* __restrict__ descs,
    __nv_bfloat16* __restrict__ out,
    float scale0, float scale1, float scale2,
    float offset0, float offset1, float offset2) {
    const ScriptorPatchifyDesc d = descs[blockIdx.y];
    const long long grid_w = d.width / SCRIPTOR_PATCH;
    const long long patches = (d.height / SCRIPTOR_PATCH) * grid_w;
    const long long index = blockIdx.x;
    if (index >= patches) return;

    const long long windows_per_row = grid_w / 2;
    const long long window = index / 4;
    const long long inner = index % 4;
    const long long patch_row = (window / windows_per_row) * 2 + inner / 2;
    const long long patch_col = (window % windows_per_row) * 2 + inner % 2;

    const int y = threadIdx.x / SCRIPTOR_PATCH;
    const int x = threadIdx.x % SCRIPTOR_PATCH;
    const long long row = patch_row * SCRIPTOR_PATCH + y;
    const long long col = patch_col * SCRIPTOR_PATCH + x;
    const uint8_t* px = rgb + d.src_offset + (row * d.width + col) * 3;
    __nv_bfloat16* dst = out + (d.dst_patch + index) * SCRIPTOR_PATCH_ELEMS;

    const float scales[3] = {scale0, scale1, scale2};
    const float offsets[3] = {offset0, offset1, offset2};
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const __nv_bfloat16 v = scriptor_to_bf16(fmaf((float)px[c], scales[c], offsets[c]));
        dst[(c * 2 + 0) * (SCRIPTOR_PATCH * SCRIPTOR_PATCH) + threadIdx.x] = v;
        dst[(c * 2 + 1) * (SCRIPTOR_PATCH * SCRIPTOR_PATCH) + threadIdx.x] = v;
    }
}
