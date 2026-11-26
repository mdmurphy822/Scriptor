// Paged decode attention whose blocks take their (sequence, split) pair from a precomputed
// dispatch order (`scriptor_attn_pair_order`) instead of ranking every pair in each block.
// The attention body, the partial layout and the reduction are those of
// `attn_paged_decode.cu`, compiled with the ordered pair lookup selected.
#define SCRIPTOR_PAGED_ATTN_ORDERED 1
#include "attn_paged_decode.cu"
