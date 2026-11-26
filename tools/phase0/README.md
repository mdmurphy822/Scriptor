# Phase-0 measurement tools

Read-only helpers the user runs against a seat that is already up. Nothing here
starts, stops or restarts a container, a seat or a bridge.

| Script | Purpose |
|---|---|
| `scrape_vllm_metrics.sh` | `/metrics` snapshots, the sweep delta table, the `num_requests_running` time series |
| `boot_time.sh` | container `StartedAt` to the first 200 listing the served model on `/v1/models` |
| `gemm_probe.md` | recipe for the sustained BF16/FP8 GEMM throughput probe (`scriptor bench gemm`) |

The full procedure, including the bridge-ceiling test, is in `docs/phase0-protocol.md`;
results are recorded in `docs/phase0-results.md`.

Dry run on the committed synthetic sample (prints the delta table):

```
tools/phase0/scrape_vllm_metrics.sh table-sample tests/fixtures/phase0/metrics-sample.txt 100
```

Check the scripts with `shellcheck tools/phase0/*.sh`.
