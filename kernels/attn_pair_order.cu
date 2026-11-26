// Dispatch order of the (sequence, split) work pairs of paged decode attention.
//
// Pair `i` (`i = sequence * splits + split`) has the key range length
// `clamp(seq_lens[sequence] - split * chunk, 0, chunk)`. `order[r]` receives the pair of
// rank `r` when pairs are sorted by descending length, ties broken by ascending pair index;
// the ranking is a bijection onto `0 .. pairs`. One block ranks every pair: the lengths are
// staged in dynamic shared memory (4 bytes per pair) and each thread counts the pairs that
// precede its own.
#include "common.cuh"

extern "C" __global__ void __launch_bounds__(256) scriptor_attn_pair_order(
    const int* __restrict__ seq_lens, int splits, int chunk, int pairs, int* __restrict__ order) {
    extern __shared__ int len[];
    for (int i = threadIdx.x; i < pairs; i += blockDim.x)
        len[i] = min(max(seq_lens[i / splits] - (i % splits) * chunk, 0), chunk);
    __syncthreads();
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
        const int wi = len[i];
        int rank = 0;
        for (int j = 0; j < pairs; ++j) {
            const int wj = len[j];
            rank += (wj > wi) | ((wj == wi) & (j < i));
        }
        order[rank] = i;
    }
}
