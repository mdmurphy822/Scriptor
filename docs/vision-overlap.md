# Vision batching and overlap

Cross-request batching is always on. Overlap is opt-in with `serve --vision-overlap`; the default
runs each batch on the main stream to completion before the step continues.

## Batching

Each scheduler step gathers every unresolved image, running sequences first and then waiting
ones in queue order, into one vision call up to `vision_batch_patches` (the memory plan's
largest-image budget). Images of waiting requests are embedded ahead of admission while the
embeddings already held ahead plus the new ones fit the embedding-cache budget from the same
plan. The batch image counts are exported as the `scriptor_vision_batch_size` histogram.

## Overlap (`--vision-overlap`)

With the flag, on the GPU executor a vision batch runs on its own stream with its own workspace, GEMM engine and
cuBLASLt workspace (all counted in the boot memory plan). The main stream, which carries prefill
and the captured decode graphs, is created at the highest stream priority so decode kernels are
scheduled ahead of vision kernels.

The scheduler starts a batch with `vision_begin`, keeps stepping prefill and decode, polls with
`vision_ready` each step and applies the result with `vision_finish`. It waits for the batch only
when no prefill or decode row can run. Ordering between streams is explicit:

- batch outputs are allocated on the main stream, and the vision stream waits on an event
  recorded after those allocations;
- the host waits on the vision stream's completion event before the embeddings reach prefill, so
  the main stream only reads finished data, and the buffers are freed on the main stream;
- decode graphs, the KV pools and the pinned copy-back are never touched by the vision stream.

Measured end to end, overlap was slower than the synchronous path: 4042 against 4090 pages per
hour on the live workload, and 43.5 against 41.1 s on a synthetic bridge-shaped load. The
decode and vision kernels contend for the same SMs and memory bandwidth on a GPU that is already
about 94% utilised, so the second stream adds scheduling cost without hiding idle time.

`scriptor_vision_host_seconds_total` is the engine-thread time spent on vision;
`scriptor_vision_seconds_total` is the batch wall time from start to application.

## Measuring

`tools/e2e/vision_load.py URL REQUESTS CONCURRENCY` drives the server with generated PNG text
images of varying size and prints wall time and tokens per second.
