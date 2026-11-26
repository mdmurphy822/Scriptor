// Batched separable bicubic antialias resize of interleaved RGB u8 images.
//
// Each image is described by eight 64-bit words: byte offsets of its source
// pixels, its intermediate buffer and its destination pixels, then input
// height and width, output height and width, and one reserved word. The
// horizontal pass maps `[in_h, in_w, 3]` to `[in_h, out_w, 3]`, the vertical
// pass maps that to `[out_h, out_w, 3]`; each pass rounds to u8. Tap geometry
// and weights are evaluated in double so the result tracks the CPU reference.
#include "common.cuh"

struct ScriptorResizeDesc {
    long long src_offset;
    long long mid_offset;
    long long dst_offset;
    long long in_h;
    long long in_w;
    long long out_h;
    long long out_w;
    long long reserved;
};

// Keys cubic convolution kernel, a = -0.5.
__device__ __forceinline__ double scriptor_cubic(double x) {
    x = fabs(x);
    if (x < 1.0) return ((1.5) * x - (2.5)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5;
    return 0.0;
}

// Computes the contiguous tap range [lo, hi) of output index `i` and returns
// the normalising weight sum through `sum`.
__device__ __forceinline__ void scriptor_tap_range(long long input, long long output, long long i,
                                             long long* lo, long long* hi, double* sum) {
    const double scale = (double)input / (double)output;
    const double support = scale >= 1.0 ? 2.0 * scale : 2.0;
    const double inv = scale >= 1.0 ? 1.0 / scale : 1.0;
    const double center = scale * ((double)i + 0.5);
    long long l = (long long)(center - support + 0.5);
    long long h = (long long)(center + support + 0.5);
    if (l < 0) l = 0;
    if (h < 0) h = 0;
    if (h > input) h = input;
    double s = 0.0;
    for (long long j = l; j < h; ++j) s += scriptor_cubic(((double)j - center + 0.5) * inv);
    *lo = l;
    *hi = h;
    *sum = s;
}

__device__ __forceinline__ uint8_t scriptor_round_u8(double v) {
    double r = round(v);
    r = fmin(fmax(r, 0.0), 255.0);
    return (uint8_t)r;
}

// grid.x: output pixel index (row * out_w + col) of the pass, grid.y: image.
// Horizontal pass, one thread per [in_h, out_w] pixel.
extern "C" __global__ void scriptor_resize_bicubic_h_u8(
    const uint8_t* __restrict__ src, const ScriptorResizeDesc* __restrict__ descs,
    uint8_t* __restrict__ mid) {
    const ScriptorResizeDesc d = descs[blockIdx.y];
    const long long n = d.in_h * d.out_w;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const long long row = idx / d.out_w;
    const long long col = idx % d.out_w;
    long long lo, hi;
    double sum;
    scriptor_tap_range(d.in_w, d.out_w, col, &lo, &hi, &sum);
    const double scale = (double)d.in_w / (double)d.out_w;
    const double inv = scale >= 1.0 ? 1.0 / scale : 1.0;
    const double center = scale * ((double)col + 0.5);
    const double divisor = sum != 0.0 ? sum : 1.0;
    const uint8_t* line = src + d.src_offset + row * d.in_w * 3;
    double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0;
    for (long long j = lo; j < hi; ++j) {
        const double w = scriptor_cubic(((double)j - center + 0.5) * inv) / divisor;
        acc0 += w * (double)line[j * 3 + 0];
        acc1 += w * (double)line[j * 3 + 1];
        acc2 += w * (double)line[j * 3 + 2];
    }
    uint8_t* o = mid + d.mid_offset + idx * 3;
    o[0] = scriptor_round_u8(acc0);
    o[1] = scriptor_round_u8(acc1);
    o[2] = scriptor_round_u8(acc2);
}

// Vertical pass, one thread per [out_h, out_w] pixel.
extern "C" __global__ void scriptor_resize_bicubic_v_u8(
    const uint8_t* __restrict__ mid, const ScriptorResizeDesc* __restrict__ descs,
    uint8_t* __restrict__ dst) {
    const ScriptorResizeDesc d = descs[blockIdx.y];
    const long long n = d.out_h * d.out_w;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const long long row = idx / d.out_w;
    const long long col = idx % d.out_w;
    long long lo, hi;
    double sum;
    scriptor_tap_range(d.in_h, d.out_h, row, &lo, &hi, &sum);
    const double scale = (double)d.in_h / (double)d.out_h;
    const double inv = scale >= 1.0 ? 1.0 / scale : 1.0;
    const double center = scale * ((double)row + 0.5);
    const double divisor = sum != 0.0 ? sum : 1.0;
    const uint8_t* base = mid + d.mid_offset + col * 3;
    double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0;
    for (long long j = lo; j < hi; ++j) {
        const double w = scriptor_cubic(((double)j - center + 0.5) * inv) / divisor;
        const uint8_t* p = base + j * d.out_w * 3;
        acc0 += w * (double)p[0];
        acc1 += w * (double)p[1];
        acc2 += w * (double)p[2];
    }
    uint8_t* o = dst + d.dst_offset + idx * 3;
    o[0] = scriptor_round_u8(acc0);
    o[1] = scriptor_round_u8(acc1);
    o[2] = scriptor_round_u8(acc2);
}
