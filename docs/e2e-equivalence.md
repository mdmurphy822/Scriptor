# End-to-end equivalence on the document set

Compares what the engine produces through the OCR bridge (`scriptor bridge --serve`, or the bridge of the combined `scriptor serve`; the legacy `semantik ocr-bridge` is also accepted)
against what the vLLM seat produces for the same documents. The comparison
unit is the recognised text of each region in each reply.

## Running

All paths are arguments; nothing is discovered. With only one OCR server
listening on the chosen port at a time:

1. Start the reference server (the vLLM seat) and run

   ```
   tools/e2e/compare_bridge_runs.sh --bridge-bin <scriptor> --example-dir <dir> \
     --docs-file <list> --out-dir <out> --layout-runtime <libonnxruntime.so> \
     --layout-weights <layout.onnx> --url http://127.0.0.1:8002/v1/chat/completions \
     --seqs 32 --tag ref
   ```

2. Stop it, start the engine on the same URL (or run
   `tools/bench/engine_sweep.sh`, which starts the engine per speculative
   depth and calls the script above), and run the same command with
   `--tag eng`.

   To exercise the combined process instead (`scriptor serve`, which runs
   the engine and the bridge together), pass `--engine-model <checkpoint dir>`
   with `--bridge-bin <scriptor>`; the script starts the engine on the host and
   port of `--url` and the bridge on `--bridge-port`, and the recorded
   replies come from that bridge.

3. Compare:

   ```
   python3 -I tools/e2e/diff_replies.py <out> ref eng --show
   ```

`--docs-file` lists one PDF per line, relative to `--example-dir`; the reply
file for a document is named `<tag>-<parent directory>.reply.json`, so the
parent directory names must be unique within the list. The script never writes
a cache; its bridge checkpoint directory is temporary and removed on exit.

## What is compared

`diff_replies.py` pairs `<tagA>-<class>.reply.json` with
`<tagB>-<class>.reply.json` and compares the `text` of the regions index by
index (reading order). A region present on only one side counts as differing.
`--dir-b` compares sets stored in different directories.

## Reading results

Per document: region counts on both sides and identical regions out of the
larger count. The final line gives totals and the identity percentage. The exit
status is 0 only when every region is identical and the counts match. `--show`
prints, for each differing region, its index, page and id, and both texts.

The gate is zero differing regions. A difference is accepted only when it is
explained by bridge nondeterminism (recovery-band timing, where a region that
the first read left degenerate is re-read at a different moment); the report
names each such document. Differences that are not explained that way, such as
different digits, dropped words or spacing in otherwise equal text, are
engine numerics and are investigated with the token-level comparison.

Each run also writes `<tag>.summary.txt` (per-document seconds, wall, pages
per hour, deltas of the server counters, mean GPU utilisation), the `/metrics`
snapshots before and after, and the bridge log. `vllm:*` and `scriptor_*` counters are
summarised when present; speculative decode acceptance appears when the server
exposes the vLLM-compatible counters.

## Measured results

Reference comparison between two vLLM runs of the 15-document set (32
sequences, baseline vs MTP depth 3): 4150/4150 regions paired, 4016 identical
(96.8%). Engine against vLLM: not yet measured.
