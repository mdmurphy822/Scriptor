# Speculative drafting with the MTP layer

GLM-OCR carries one multi-token-prediction layer (layer 16). The engine uses it
as a greedy drafter; the main model verifies every draft, so output is the same
as non-speculative decoding for greedy requests.

## Step

For the live sequences, one step does:

1. **Choose k.** `DraftPolicy::choose_k(batch, limit)` picks the draft length
   (0..=`k_max`), capped per sequence by `max_tokens`, the context window and
   one token reserved for the verifier's own token.
2. **Draft.** The first MTP pass consumes each sequence's pending feeds: for
   every position `i` the MTP KV lacks, the input is the token at `i + 1` plus
   the main model's hidden state at `i`. On a sequence's first step these are
   all prompt positions (the prompt shifted left by one, ending with the first
   generated token; image placeholder ids are kept so the layer sees the
   placeholder embedding there; the row at position zero has a zeroed
   embedding). Later passes feed the previous draft with the MTP layer's own
   output hidden state. Each draft is the argmax of the shared head, without
   penalties.
3. **Verify.** One main pass over `k + 1` rows per sequence: the last accepted
   token followed by the drafts. The main KV grows by `k + 1` positions.
4. **Accept.**
   - Greedy sequences: the penalised argmax at each position (penalties see
     the drafts before it) is compared with the draft; the longest matching
     prefix is accepted and the argmax at the first mismatch, or at the last
     row when everything matched, is emitted. Kernel: `scriptor_mtp_accept`
     (`kernels/mtp_accept.cu`), reference `scriptor_kernels::verify::accept_greedy`.
   - Sampled sequences (temperature > 0): vLLM rejection-sampler semantics for
     one-hot drafts. Draft `j` is accepted when a seeded uniform is below the
     probability the penalised, temperature-scaled, top-k / top-p filtered
     distribution gives it; at the first rejection a token is drawn from that
     distribution with the rejected draft removed; when all are accepted a
     bonus token is drawn from the last row. The emitted stream has the target
     distribution (`accept_rejection`, tested against the target marginal).
5. **Roll back.** The main KV is truncated to the accepted length. The MTP KV
   needs no rollback: the next first pass rewrites it from the first pending
   position with the main model's true hidden states.

When the policy picks `k = 0` the MTP layer is not run; the positions it
misses stay queued as pending feeds and are consumed by the next drafting
step, so the MTP KV is always complete when drafting resumes.

## Adaptive k

`scriptor_engine::spec::policy` keeps, with exponential decay:

- an affine cost fit `seconds = base + per_row * rows` for the verification
  pass and for the draft pass, fed by CUDA-event durations measured by the
  model implementation;
- per-position cumulative acceptance: the fraction of verified chains whose
  drafts `0..=j` were all accepted, shrunk toward a prior and kept
  non-increasing in `j`.

Expected throughput at draft length `k` and batch `b` is
`b * (1 + sum_j c_j) / (main(b * (k + 1)) + k * draft(b))`. The policy picks the
`k` with the highest throughput and uses `k = 0` unless that beats `k = 0` by
`min_gain`. A policy stuck at `k = 0` probes one draft after `reprobe_after`
steps so the acceptance estimate keeps tracking. Before any cost has been
measured the largest permitted `k` is used to gather measurements.

## Draft depth versus measured cost

A draft position in the captured decode graph costs about 1.9 ms at 32
sequences: the MTP layer's paged attention (about 0.62 ms, one pass over the MTP
KV), the shared LM-head product (about 0.86 ms, 182 MB of weights), the layer's
projections (about 0.32 ms) and about 0.07 ms of small kernels including the
draft argmax (about 14 us). The verification pass adds roughly 0.4 ms per extra
row group. Per-position cumulative acceptance and the end-to-end effect at 32
sequences on the benchmark document set (greedy, quiet device):

| k | tokens/step | cumulative acceptance by position | decode + prefill s | WALL s |
|---|---|---|---|---|
| 3 | 3.57 | .929 .858 .786 | 110.5 | 239.4 |
| 4 | 4.24 | .923 .848 .771 .695 | 101.1 | 234.5 |
| 5 | 4.81 | .917 .837 .760 .683 .608 | 95.7 | 222.2 |

A position pays while its cumulative acceptance exceeds
`tokens_per_step * position_cost / step_time`, about 0.36 at these costs; the
graph path captures its depth at boot, so the depth is chosen from this
relation per deployment, and `DraftPolicy` applies the same relation per step
on executors that vary the depth at run time (its `k_max` default is unbounded,
leaving the executor's limit as the only cap).

## Metrics

`SpecStats::prometheus()` renders, under the names the sweep tooling scrapes:
`vllm:spec_decode_num_drafts_total`, `vllm:spec_decode_num_draft_tokens_total`,
`vllm:spec_decode_num_accepted_tokens_total`,
`vllm:spec_decode_num_accepted_tokens_per_pos_total{position="j"}`, plus
`scriptor_spec_proposed_tokens_per_pos_total{position="j"}`,
`scriptor_spec_verify_steps_total` and `scriptor_spec_draft_passes_total`.

## Acceptance versus the vLLM k-sweep

Pending: needs the GPU and the real weights (task T41 gate). Record here, per
position, the engine's accepted-per-position rates next to the vLLM k-sweep
rates; the gate is agreement within 3 percentage points.

| position | vLLM | engine |
|---|---|---|
| 0 | | |
| 1 | | |
| 2 | | |
