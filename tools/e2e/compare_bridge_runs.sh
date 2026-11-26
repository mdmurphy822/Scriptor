#!/usr/bin/env bash
# Runs a document set through the OCR bridge (`scriptor bridge --serve`, the combined
# `scriptor serve`, or the legacy `semantik ocr-bridge --serve`) and records per-document
# seconds, wall time, pages per hour, the server's /metrics before and after, and GPU
# utilisation samples. By default the OpenAI-compatible OCR server at --url is already
# listening; with --engine-model the script starts `scriptor serve` itself.
#
# Usage: compare_bridge_runs.sh (--bridge-bin PATH | --semantik-bin PATH) --example-dir DIR --docs-file FILE
#          --out-dir DIR --layout-runtime PATH --layout-weights PATH --url URL
#          [--seqs N] [--model NAME] [--tag TAG] [--bridge-port PORT]
#          [--metrics-url URL] [--layout-batch N] [--engine-model DIR [--serve-arg ARG]...]
#
#   --bridge-bin    a `scriptor` binary, started as `PATH bridge --serve --port N`
#                   against the server at --url, or, with --engine-model, as the
#                   combined `PATH serve --model DIR --bridge-port N`.
#   --engine-model  checkpoint directory; runs the combined `scriptor serve` (engine and
#                   bridge in one process) on the host and port of --url, with --model
#                   as the served name and --seqs as --max-num-seqs. Requires --bridge-bin.
#   --serve-arg     one extra argument for the combined `scriptor serve`; repeatable.
#   --semantik-bin  the legacy bridge, started as `PATH ocr-bridge --serve --port N`.
#                   Exactly one of the two is required; every other argument and
#                   the wire are identical for both.
#   --docs-file     text file, one PDF path relative to --example-dir per line;
#                   blank lines and lines starting with # are ignored.
#   --url           chat-completions endpoint of the OCR server.
#   --seqs          server max sequences (default 32); bridge workers = min(seqs, 128).
#   --model         served model name (default glm-ocr).
#   --tag           file-name prefix for outputs (default run).
#   --metrics-url   defaults to the --url origin plus /metrics.
#
# Outputs in --out-dir: <tag>-<class>.reply.json per document, <tag>.bridge.log,
# <tag>.metrics-pre.txt, <tag>.metrics-post.txt, <tag>.gpu.log, <tag>.summary.txt,
# <tag>.clock.log (one line per second: epoch seconds, SM clock MHz, GPU utilisation percent)
# and <tag>.metrics.ts (the server's /metrics each second, each scrape preceded by a
# "### <epoch seconds>" line).
# <class> is the document's parent directory name. Requires curl, pdfinfo, bc,
# sha256sum, python3. The bridge checkpoint directory is a temporary directory
# inside --out-dir that is removed on exit; no cache is written anywhere else.
set -euo pipefail

semantik_bin=""; bridge_bin=""; example_dir=""; docs_file=""; out_dir=""
layout_runtime=""; layout_weights=""; url=""; metrics_url=""
seqs=32; model=glm-ocr; tag=run; bridge_port=18012; layout_batch=4
engine_model=""; serve_args=()

usage() { sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --semantik-bin) semantik_bin=$2; shift 2 ;;
    --bridge-bin) bridge_bin=$2; shift 2 ;;
    --example-dir) example_dir=$2; shift 2 ;;
    --docs-file) docs_file=$2; shift 2 ;;
    --out-dir) out_dir=$2; shift 2 ;;
    --layout-runtime) layout_runtime=$2; shift 2 ;;
    --layout-weights) layout_weights=$2; shift 2 ;;
    --url) url=$2; shift 2 ;;
    --metrics-url) metrics_url=$2; shift 2 ;;
    --seqs) seqs=$2; shift 2 ;;
    --model) model=$2; shift 2 ;;
    --tag) tag=$2; shift 2 ;;
    --bridge-port) bridge_port=$2; shift 2 ;;
    --layout-batch) layout_batch=$2; shift 2 ;;
    --engine-model) engine_model=$2; shift 2 ;;
    --serve-arg) serve_args+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for v in example_dir docs_file out_dir layout_runtime layout_weights url; do
  if [ -z "${!v}" ]; then echo "missing --${v//_/-}" >&2; usage >&2; exit 2; fi
done
if [ -n "$bridge_bin" ] && [ -n "$semantik_bin" ]; then
  echo "give exactly one of --bridge-bin and --semantik-bin" >&2; usage >&2; exit 2
fi
if [ -z "$bridge_bin" ] && [ -z "$semantik_bin" ]; then
  echo "missing --bridge-bin or --semantik-bin" >&2; usage >&2; exit 2
fi
if [ -n "$engine_model" ] && [ -z "$bridge_bin" ]; then
  echo "--engine-model requires --bridge-bin" >&2; usage >&2; exit 2
fi
if [ -n "$engine_model" ]; then
  engine_hostport=$(printf '%s' "$url" | sed -E 's#^https?://([^/]+)/.*#\1#')
  engine_host=${engine_hostport%:*}; engine_port=${engine_hostport##*:}
  bridge_exe=$bridge_bin
  bridge_pre=(serve --model "$engine_model" --host "$engine_host" --port "$engine_port"
    --served-model-name "$model" --max-num-seqs "$seqs" "${serve_args[@]}")
  bridge_flags=(--bridge-port "$bridge_port")
elif [ -n "$bridge_bin" ]; then
  bridge_exe=$bridge_bin; bridge_pre=(bridge); bridge_flags=(--serve --port "$bridge_port")
else
  bridge_exe=$semantik_bin; bridge_pre=(ocr-bridge); bridge_flags=(--serve --port "$bridge_port")
fi
[ -x "$bridge_exe" ] || { echo "not executable: $bridge_exe" >&2; exit 2; }
[ -f "$docs_file" ] || { echo "no such docs file: $docs_file" >&2; exit 2; }
if [ -z "$metrics_url" ]; then
  origin=$(printf '%s' "$url" | sed -E 's#^(https?://[^/]+)/.*#\1#')
  metrics_url="$origin/metrics"
fi

here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out_dir"
out_dir=$(cd "$out_dir" && pwd)
ck=$(mktemp -d "$out_dir/.checkpoint-$tag.XXXXXX")
summary="$out_dir/$tag.summary.txt"
: > "$summary"

mapfile -t docs < <(grep -Ev '^[[:space:]]*(#|$)' "$docs_file")
pages=0
for d in "${docs[@]}"; do
  pages=$((pages + $(pdfinfo "$example_dir/$d" | awk '/^Pages:/{print $2}')))
done
echo "tag=$tag docs=${#docs[@]} total_pages=$pages seqs=$seqs" | tee -a "$summary"

bridge_pid=""; sampler_pid=""; clock_pid=""; scrape_pid=""
cleanup() {
  if [ -n "$sampler_pid" ]; then kill "$sampler_pid" 2>/dev/null || true; fi
  if [ -n "$clock_pid" ]; then kill "$clock_pid" 2>/dev/null || true; fi
  if [ -n "$scrape_pid" ]; then kill "$scrape_pid" 2>/dev/null || true; fi
  if [ -n "$bridge_pid" ]; then
    kill "$bridge_pid" 2>/dev/null || true
    wait "$bridge_pid" 2>/dev/null || true
  fi
  rm -rf "$ck"
}
trap cleanup EXIT

workers=$(( seqs < 128 ? seqs : 128 ))
SCRIPTOR_BRIDGE_ENGINE_URL="$url" SCRIPTOR_BRIDGE_MODEL="$model" \
SCRIPTOR_BRIDGE_LAYOUT_RUNTIME="$layout_runtime" SCRIPTOR_BRIDGE_LAYOUT_WEIGHTS="$layout_weights" \
SCRIPTOR_BRIDGE_LAYOUT_BATCH_SIZE="$layout_batch" \
SCRIPTOR_BRIDGE_MAX_NUM_SEQS="$seqs" SCRIPTOR_BRIDGE_WORKERS="$workers" \
SCRIPTOR_BRIDGE_CHECKPOINT_DIR="$ck" \
SEMANTIK_GLM_OCR_URL="$url" SEMANTIK_GLM_OCR_MODEL="$model" \
SEMANTIK_LAYOUT_RUNTIME="$layout_runtime" SEMANTIK_LAYOUT_WEIGHTS="$layout_weights" \
SEMANTIK_GLM_OCR_LAYOUT_BATCH_SIZE="$layout_batch" \
SEMANTIK_GLM_OCR_MAX_NUM_SEQS="$seqs" SEMANTIK_GLM_OCR_WORKERS="$workers" \
SEMANTIK_GLM_OCR_CHECKPOINT_DIR="$ck" \
  setsid "$bridge_exe" "${bridge_pre[@]}" "${bridge_flags[@]}" \
  > "$out_dir/$tag.bridge.log" 2>&1 < /dev/null &
bridge_pid=$!
until grep -Eq "(ocr-bridge|scriptor-bridge): ready" "$out_dir/$tag.bridge.log"; do
  kill -0 "$bridge_pid" 2>/dev/null || { echo "bridge exited before ready" | tee -a "$summary"; exit 1; }
  sleep 0.5
done

curl -s -m 10 "$metrics_url" > "$out_dir/$tag.metrics-pre.txt" || true
(
  while kill -0 "$bridge_pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
    sleep 1
  done
) > "$out_dir/$tag.gpu.log" &
sampler_pid=$!
(
  while kill -0 "$bridge_pid" 2>/dev/null; do
    printf '%s ' "$(date +%s.%N)"
    nvidia-smi --query-gpu=clocks.sm,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr ',' ' ' || echo
    sleep 1
  done
) > "$out_dir/$tag.clock.log" &
clock_pid=$!
(
  while kill -0 "$bridge_pid" 2>/dev/null; do
    echo "### $(date +%s.%N)"
    curl -s -m 2 "$metrics_url" || true
    sleep 1
  done
) > "$out_dir/$tag.metrics.ts" &
scrape_pid=$!

t0=$(date +%s.%N)
pids=()
for d in "${docs[@]}"; do
  f="$example_dir/$d"
  sha=$(sha256sum "$f" | cut -d' ' -f1)
  cls=$(basename "$(dirname "$d")")
  (
    s=$(date +%s.%N)
    code=$(curl -s -o "$out_dir/$tag-$cls.reply.json" -w '%{http_code}' -X POST \
      --data-binary @"$f" -H "Content-Type: application/pdf" \
      -H "Semantik-Source-SHA256: $sha" \
      -H "Semantik-Contract: pages=one-based; regions=reading-order; bbox=x0,y0,x1,y1" \
      "http://127.0.0.1:$bridge_port/ocr")
    echo "$tag $cls http=$code secs=$(echo "$(date +%s.%N) - $s" | bc)" >> "$summary"
  ) &
  pids+=($!)
done
wait "${pids[@]}"
wall=$(echo "$(date +%s.%N) - $t0" | bc)
curl -s -m 10 "$metrics_url" > "$out_dir/$tag.metrics-post.txt" || true

echo "$tag WALL=$wall pages_per_hour=$(echo "$pages * 3600 / $wall" | bc)" | tee -a "$summary"
python3 -I "$here/metrics_summary.py" "$out_dir/$tag.metrics-pre.txt" \
  "$out_dir/$tag.metrics-post.txt" "$out_dir/$tag.gpu.log" "$wall" | sed "s/^/$tag /" | tee -a "$summary"
grep -E 'schedule pages' "$out_dir/$tag.bridge.log" >> "$summary" || true
