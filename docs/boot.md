# Boot

## Memory plan

The plan stage refuses a boot whose worst-case plan exceeds the memory free. On
a unified-memory device (CUDA `CU_DEVICE_ATTRIBUTE_INTEGRATED`) the device free
figure excludes reclaimable page cache, so the figure used is the larger of the
device free memory and host `MemAvailable` from `/proc/meminfo`; discrete
devices use the device figure. The refusal names the source used. The
`--gpu-memory-utilization` share of total memory still caps the plan.

## Measuring boot

`tools/bench/boot_times.sh` measures cold and warm boot. Both boots are timed
from process start to the first `/health` 200 (vLLM mode: `/v1/models`), and the
engine's per-stage `scriptor_boot_stage_seconds` samples are read from `/metrics` once
it is ready.

- Cold: a boot with a fresh empty throwaway cache directory created by
  `mktemp -d` under `--scratch-root`.
- Warm: a second boot from that same directory.
- The throwaway directory is removed afterwards, so nothing warm survives a
  measurement. The script never passes any other cache directory.

```
tools/bench/boot_times.sh --scratch-root <dir> --engine <scriptor> --model-dir <model>
```

Output lines are `cold ready_seconds=...`, `warm ready_seconds=...`, and the
`scriptor_boot_stage_seconds` samples prefixed with the boot label.

Page-cache-cold runs: `tools/bench/boot_times.sh --drop-page-cache-note` prints
the root command to run first (`sync && echo 3 > /proc/sys/vm/drop_caches`);
the script never runs sudo. Without it, weights are served from the page cache
and the numbers are page-cache-warm.

vLLM comparison: with `--vllm --start-script <script> --stop-script <script>
--image <image>`, the start script is run twice with `BOOT_CACHE_DIR` exported;
it must bind-mount that directory as the seat's compile cache. The stop script
runs between the boots and after the second. The root-owned throwaway directory
is emptied by `docker run --rm -v <dir>:/d <image> rm -rf /d/*` before removal.

### Results

| Boot | Seconds |
|---|---|
| vLLM seat, cold compile cache | not yet measured (baseline 125-136 s) |
| vLLM seat, warm compile cache | not yet measured |
| Engine, cold cache | not yet measured |
| Engine, warm cache | not yet measured |
