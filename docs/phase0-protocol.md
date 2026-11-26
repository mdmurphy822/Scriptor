# Phase-0 measurement protocol

Four measurements fix the numbers the engine plan depends on before any kernel work is
judged. Every step is read-only toward the running seat: the user starts and stops the
seat with the existing seat scripts; the tools here only read its HTTP endpoints and
`docker inspect` output. Paths below use placeholders: `$ENGINE` is this project.

Reference sweep: 15 example documents, 368 pages, `scriptor bridge --serve` (originally measured with the legacy `semantik ocr-bridge --serve`) driving
`vllm/vllm-openai:v0.33.0` at share 0.60 (see `docs/phase0-results.md` for the numbers
already known).

## 1. Metrics delta over a sweep

The goal is the per-step cost and the prefill/decode split of the current seat.

1. With the seat up and idle, take the pre snapshot:
   `tools/phase0/scrape_vllm_metrics.sh snapshot http://127.0.0.1:8002/metrics pre.txt`
2. Start the time series in the background (5 s interval):
   `tools/phase0/scrape_vllm_metrics.sh watch http://127.0.0.1:8002/metrics 5 running.txt`
3. Run the sweep (the bridge plus the 15 documents, as in the reference sweep) and note
   the wall-clock seconds from first submission to last reply.
4. Stop the time series (interrupt it), then take the post snapshot into `post.txt`.
5. Print the table: `tools/phase0/scrape_vllm_metrics.sh table pre.txt post.txt <wall>`.
6. Summarise concurrency: `tools/phase0/scrape_vllm_metrics.sh series running.txt <max_num_seqs>`.

Reported quantities and how they are read:

| Row | Source | Meaning |
|---|---|---|
| prompt / generation tokens | `vllm:prompt_tokens_total`, `vllm:generation_tokens_total` | workload size; tokens per request = divide by requests finished |
| engine steps | `vllm:iteration_tokens_total_count` | scheduler iterations |
| wall / step | wall seconds divided by steps | per-step cost including all host work; compare with the 4.8 ms weight-bandwidth floor |
| mean scheduled tokens / step | `iteration_tokens_total_sum / count` | batch composition (decode rows plus prefill tokens) |
| request prefill / decode time | `vllm:request_{prefill,decode}_time_seconds_{sum,count}` | per-request time spent in each phase, summed over concurrent requests |
| spec drafts, accepted | `vllm:spec_decode_num_*` | present only when speculative decoding is on |
| accept rate by position | `num_accepted_tokens_per_pos_total{position}` divided by drafts | acceptance curve per draft position |

Run the sweep once per configuration of interest: `max_num_seqs` 32 and 128 without
speculation, then with MTP `num_speculative_tokens` = 1, 2, 3, 4 at `max_num_seqs` 128
for the per-position acceptance curves. Use a distinct output prefix per run.

## 2. Boot time

Right after the seat start script returns, run
`tools/phase0/boot_time.sh --container glm-ocr` (defaults: `http://127.0.0.1:8002/v1/models`,
served model `glm-ocr`). It reads `StartedAt` from `docker inspect`, polls the models
endpoint every 0.5 s and prints `boot_seconds`. A line noting the seat was already
serving means the measurement is an upper bound only; repeat from a cold start. Record
cold (page cache dropped by the seat's own procedure) and warm boots separately.

## 3. Bridge-ceiling test

The goal is the bridge's own pages/hour when the seat answers instantly, so seat-bound
and bridge-bound limits can be told apart.

1. Stop the real seat (user action).
2. Start the mock seat from the engine binary (T40): it serves `/v1/models` and
   `/v1/chat/completions` on port 8002 with canned text and no model, e.g.
   `scriptor bench mock-seat --bind 127.0.0.1:8002 --served-model-name glm-ocr`
   (flags follow the T40 CLI; the contract is: instant canned reply, same JSON shape and
   metrics names as the real seat).
3. Run the identical bridge command and document set as the reference sweep, with
   `SCRIPTOR_BRIDGE_MAX_NUM_SEQS` and `SCRIPTOR_BRIDGE_WORKERS` (legacy `SEMANTIK_GLM_OCR_MAX_NUM_SEQS` and `SEMANTIK_GLM_OCR_WORKERS`) at the values under
   test (32 and 128).
4. Record wall seconds and pages/hour (`368 * 3600 / wall`) together with the bridge log
   `schedule` lines (`render_ms`, `layout_ms`, `queue_wait_ms`, `worker_idle_ms`).
   The layout model still shares the GPU, so run it as in the reference sweep.

The resulting pages/hour is the ceiling no engine can exceed with the bridge unchanged.

## 4. GEMM probe

Follow `tools/phase0/gemm_probe.md` once `scriptor bench gemm` exists (T21). It
needs the GPU, so the seat must be down.

## Recording

Paste each run into `docs/phase0-results.md` under the matching heading, keeping the raw
snapshot files with the run notes. Do not commit snapshot files that contain machine
paths or document metadata; the table output contains neither.
