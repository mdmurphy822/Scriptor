#!/usr/bin/env bash
# Sweeps the engine over speculative depth: for each k, starts
# `scriptor serve`, waits for /health, runs
# tools/e2e/compare_bridge_runs.sh against it, then stops the engine by its
# exact pid with SIGTERM and waits for the port to free.
#
# Usage: engine_sweep.sh --engine PATH --model-dir DIR --out-dir DIR [--port N]
#          [--seqs N] [--max-model-len N] [--gpu-share F] [--ks "0 3"]
#          [--cache-dir DIR] [--tag PREFIX] [--engine-args "ARGS"] -- COMPARE_ARGS...
#
#   --ks          space-separated speculative depths (default "0 3").
#   --cache-dir   passed to the engine only when given.
#   --tag         prefix of each run tag; the run tag is <PREFIX>k<K> (default eng).
#   --engine-args extra `scriptor serve` arguments, split on whitespace and appended
#                 to every engine start (for example "--sm-split off").
#   COMPARE_ARGS  passed to compare_bridge_runs.sh after the arguments this
#                 script supplies (--url, --seqs, --model, --tag, --out-dir):
#                 --bridge-bin or --semantik-bin, --example-dir, --docs-file, --layout-runtime,
#                 --layout-weights and any optional ones.
#
# The served model name is glm-ocr. Defaults: port 8002, seqs 32,
# max-model-len 16384, gpu-share 0.60.
set -euo pipefail

engine=""; model_dir=""; out_dir=""; cache_dir=""; engine_args=""
port=8002; seqs=32; max_len=16384; share=0.60; ks="0 3"; tag=eng

usage() { sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --engine) engine=$2; shift 2 ;;
    --model-dir) model_dir=$2; shift 2 ;;
    --out-dir) out_dir=$2; shift 2 ;;
    --port) port=$2; shift 2 ;;
    --seqs) seqs=$2; shift 2 ;;
    --max-model-len) max_len=$2; shift 2 ;;
    --gpu-share) share=$2; shift 2 ;;
    --ks) ks=$2; shift 2 ;;
    --cache-dir) cache_dir=$2; shift 2 ;;
    --tag) tag=$2; shift 2 ;;
    --engine-args) engine_args=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
for v in engine model_dir out_dir; do
  if [ -z "${!v}" ]; then echo "missing --${v//_/-}" >&2; usage >&2; exit 2; fi
done
[ -x "$engine" ] || { echo "not executable: $engine" >&2; exit 2; }

here=$(cd "$(dirname "$0")" && pwd)
compare="$here/../e2e/compare_bridge_runs.sh"
mkdir -p "$out_dir"
epid=""

stop_engine() {
  [ -n "$epid" ] || return 0
  kill -TERM "$epid" 2>/dev/null || true
  wait "$epid" 2>/dev/null || true
  epid=""
  until ! curl -s -m 2 -o /dev/null "http://127.0.0.1:$port/health"; do sleep 1; done
}
trap stop_engine EXIT

for k in $ks; do
  run="${tag}k$k"
  extra=()
  [ -n "$cache_dir" ] && extra=(--cache-dir "$cache_dir")
  # shellcheck disable=SC2206
  eargs=($engine_args)
  t0=$(date +%s.%N)
  "$engine" serve --model "$model_dir" --port "$port" --served-model-name glm-ocr \
    --gpu-memory-utilization "$share" --max-num-seqs "$seqs" --max-model-len "$max_len" \
    --num-speculative-tokens "$k" --no-bridge "${extra[@]}" "${eargs[@]}" > "$out_dir/$run.engine.log" 2>&1 < /dev/null &
  epid=$!
  until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health")" = 200 ]; do
    if ! kill -0 "$epid" 2>/dev/null; then
      echo "engine $run exited before ready" | tee -a "$out_dir/$run.summary.txt"
      tail -20 "$out_dir/$run.engine.log"
      epid=""
      continue 2
    fi
    sleep 0.5
  done
  echo "engine $run ready after $(echo "$(date +%s.%N) - $t0" | bc)s" | tee -a "$out_dir/$run.summary.txt"
  "$compare" --url "http://127.0.0.1:$port/v1/chat/completions" --seqs "$seqs" \
    --model glm-ocr --tag "$run" --out-dir "$out_dir" "$@" || echo "compare failed for $run" >&2
  stop_engine
done
echo ENGINE-SWEEP-DONE
