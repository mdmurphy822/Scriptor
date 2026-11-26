# Attention kernels

Two kernels serve every attention in the engine. Both run bf16 `mma.sync`
16x8x16 tiles with fp32 accumulation and an fp32 online softmax in the exp2
domain, and both are specified by `reference::attention`.

## `attn_varlen.cu`

Flash-style attention over concatenated sequences described by `cu_seqlens`.

| Property | Value |
|---|---|
| Entry points | `scriptor_attn_varlen_d{64,128}_{full,causal}` |
| Layout | q/out `[tokens, q_heads, head_dim]`, k/v `[tokens, kv_heads, head_dim]`, bf16 |
| GQA | query head `h` reads KV head `h / (q_heads / kv_heads)` |
| Grid | `(ceil(max_seqlen / 64), q_heads, num_seqs)`, 128 threads |
| Users | vision tower (head 64, 16/16 heads, non-causal, one sequence per image, up to 24k patches) and text prefill (head 128, 16/8 heads, causal, one sequence per request) |

A block owns 64 query rows of one head; each of its 4 warps owns 16 rows and
keeps its Q fragments, the 16x`head_dim` fp32 output accumulator and the
running max and sum in registers. Keys and values stream through shared memory
in 64-key tiles padded by 8 elements per row, which makes the K fragment loads
(32-bit, row `g`, column `2t`) and the V fragment loads (16-bit, row `2t`,
column `g`) bank-conflict free. Causal blocks stop at the diagonal tile and
warps skip tiles entirely above their rows. Rows past the sequence end are
zero-filled on load and never stored. A fully masked tile leaves the row state
untouched, so ragged tails and the first causal tile are exact.

Occupancy (ptxas, sm_121): head 128 uses about 205 registers and 34.8 KB of
shared memory per block, head 64 about 128 registers and 18.4 KB. With 65536
registers per SM that is 2 resident blocks (8 warps) for head 128 and 4
(16 warps) for head 64. Work per block is balanced by the grid being
sequence-major in `z`; for causal prefill the later x blocks of a sequence do
more work, so sequences are launched together in one grid and the scheduler
fills SMs with the short blocks of other sequences.

Throughput is compute bound for prefill and ViT: the 64x64 tile does
`2 * 64 * 64 * head_dim * 2` flops per `2 * 64 * head_dim * 2` bytes of K and V
traffic, so K and V are re-read once per 64 query rows (L2 resident for a
head's sequence below a few thousand tokens). The first improvements to try on
hardware are double-buffering the K/V tile with `cp.async` and `ldmatrix.trans`
for the V fragments.

## `attn_paged_decode.cu`

Decode and draft-verification attention, head dimension 128, 16 Q over 8 KV
heads, `q_len` 1 to `k + 1`.

| Property | Value |
|---|---|
| Entry points | `scriptor_paged_attn_decode_d128`, `scriptor_paged_attn_reduce` |
| Cache | `[blocks, block_size, kv_heads, 128]` bf16, `block_tables [seq, max_blocks]` int32 |
| Rows per launch | `group * q_len <= 16` (one m16 tile per KV head) |
| Mask | query `j` sees positions `0 ..= seq_len - q_len + j` |
| Grid | `(kv_heads, splits, seqs)`, 128 threads; reduce `(q_len * q_heads, seqs)` |

The `group * q_len` query rows of a KV head form one m16 tile, so every K and
V byte of the context is loaded once per KV head and shared by the group and
all verified positions. Each of the 4 warps walks 16-key tiles of its split's
range strided by 4 tiles, reading K fragments straight from global memory
(each lane group fetches 16 contiguous bytes per key, absorbed by L1) and
staging V in a per-warp 4.3 KB shared buffer. Each warp writes its own
unnormalised partial `(o, m, l)`, so a launch produces `splits * 4` partials per
row and `scriptor_paged_attn_reduce` merges them with the standard max-rescaled
combination. A partial that saw no keys carries `m = -inf` and is skipped.

Split-K: `choose_splits` targets `2 * sm_count` blocks over
`batch * kv_heads * splits`, never fewer than one 64-key chunk per split; with
8 KV heads a single sequence at 16384 tokens gets 12 splits on a 48-SM part
and a batch of 32 gets 1. Chunks are multiples of 64 keys.

Bandwidth: decode attention is memory bound. Per token of context a sequence
reads `2 * kv_heads * 128 * 2 = 4096` bytes per layer (64 KB over 16 layers),
so a 16384-token context streams 1.0 GiB per step, about 4.4 ms at 240 GB/s;
batching sequences does not amortise it, which is why split-K parallelism
matters at small batch and why the partial buffers are fp32 but small
(`rows * 128 * 4` bytes, a few hundred KB).

## Hoisted pair order (`attn_pair_order.cu`, `attn_paged_decode_ordered.cu`)

`scriptor_attn_pair_order` ranks the `batch * splits` (sequence, split) pairs of a decode
pass once, by descending key-range length with ties by pair index, into a fixed-address
`order` buffer of the split-K workspace. `scriptor_paged_attn_decode_d128_ordered` is the
default decode kernel compiled with the block's pair read from `order[target]`; its
arithmetic and partial layout are those of `scriptor_paged_attn_decode_d128`. The executor
builds the order after the pass's sequence lengths are staged and before the first layer,
and invalidates it when the pass ends, so every pass (and every graph replay, which
contains the build launch) orders its own lengths. `SCRIPTOR_ATTN_PAIR_ORDER=on` selects it
from 32 pairs, `all` at every pair count; the default is off.

Resident blocks: `SCRIPTOR_ATTN_RESIDENT` selects the decode attention kernel compiled for 2
(default, also when unset) or 3 resident blocks per SM; the kernels differ only in the register
bound of `__launch_bounds__`, so outputs are bitwise equal. Any other value is an error at
executor construction. The ordered kernels follow the same selection.

Memory: the order is `batch * splits` `i32` entries, at most 4096 (16 KiB), held in its own
buffer of the workspace and allocated only when the order is selected on the direct route, in
the same step that sizes the partial rows and before any graph capture. `reserve_resident`
sizes the partial rows and the order independently as the maximum over every reserved shape,
since the shape with the most partial rows need not have the most pairs. A launch whose
pairs the workspace order does not hold runs the unordered kernel; the workspace is never
grown by such a launch. The memory plan charges the 16 KiB maximum once, in the step
workspace.

## Decode routes

Pure-decode attention is staged by default: the decode rows are copied into separate
query/output buffers around the kernel, as for a batch that also holds prefill rows. The
direct route, opted in with `TextExecutor::set_direct_decode(true)`, has the kernel read and
write the activation buffers in place and skips both copies; a batch with prefill rows is
always staged. The hoisted order runs only on the direct route. The routes are compared bitwise on the
GPU before either is relied on.
