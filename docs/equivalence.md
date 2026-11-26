# Equivalence and throughput harness

`scriptor-bench` posts the exact chat body the Semantik OCR bridge sends (one user
message holding an `image_url` data-URL part then the task text; `temperature 0`,
`top_p 1e-5`, `top_k 1`, `repetition_penalty 1.1`, `max_tokens 8192`, `model`
`glm-ocr`) to OpenAI-compatible endpoints. It never launches a seat; every
subcommand takes endpoints that are already up.

## Region sets

A region set is a directory with `manifest.json` and one JPEG (quality 75) per crop:

```json
{"regions": [{"file": "name.jpg", "width": 448, "height": 420, "tasks": ["text"]}]}
```

- **Synthetic**: `scriptor bench regions --out DIR` renders the 20 crops of
  `tests/fixtures/bench/synthetic-manifest.json` (text lines, tables, formula-like
  strings) with the bundled OFL font. Sizes include non-multiples of 28, a crop
  below the minimum pixel count, and a 2100x2240 crop of exactly 6000 vision tokens.
  Rendering is deterministic.
- **Example**: `tools/bench/render_regions.sh --out DIR PDF...` rasterises PDFs with
  `pdftoppm` at 200 dpi and tiles each page into 28-multiple crops. The public
  example PDFs live under the Semantik checkout's `example/` directory. `DIR` is the
  caller's choice; crops are generated data and are not committed.

## Commands

- `bench compare --regions DIR --a URL --b URL --tokenizer tokenizer.json [--tasks text,table,formula] [--logprobs] [--workers N]`
  posts every (region, task) to both endpoints, tokenizes both `content` strings
  and compares them token for token, and compares `completion_tokens`. The first
  divergence is reported with its token index, both token ids and, when both
  endpoints accept `logprobs`, the top-2 logit margin on each side. Exit code 0 only
  when every case is identical.
- `bench ladder --regions DIR --url URL [--levels 32,64,128,256] [--rounds N]`
  drives each concurrency level with the bridge-shaped task mix (mostly text, a few
  formulas and tables) and reports requests/s, p50 and p99 latency, and prompt and
  generation tokens per second.
- `bench mock-seat --bind ADDR --reply-file FILE` answers `GET /v1/models` and
  `POST /v1/chat/completions` instantly with the file's text, so the bridge's own
  ceiling can be measured.
- Both `compare` and `ladder` accept `--report-md FILE` and `--report-json FILE`.

## Equivalence record

Run `bench compare` against the live vLLM seat (A) and the engine (B) on both
region sets with `--tasks text,table,formula --logprobs`. The acceptance bar is
100 % token identity on both sets.

| Date | Region set | Cases | Identical | Notes |
|---|---|---|---|---|
| pending | synthetic | | | requires GPU, weights and a live seat |
| pending | example | | | requires GPU, weights and a live seat |
