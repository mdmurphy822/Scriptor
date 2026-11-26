// Ordered-pair paged decode attention compiled for three resident blocks per SM; the body,
// partial layout and reduction are those of `attn_paged_decode.cu` with the ordered pair lookup.
#define SCRIPTOR_ATTN_RESIDENT_BLOCKS 3
#define SCRIPTOR_PAGED_ATTN_ORDERED 1
#include "attn_paged_decode.cu"
