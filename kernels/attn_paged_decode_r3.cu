// Paged decode attention compiled for three resident blocks per SM; the body, partial layout
// and reduction are those of `attn_paged_decode.cu`.
#define SCRIPTOR_ATTN_RESIDENT_BLOCKS 3
#include "attn_paged_decode.cu"
