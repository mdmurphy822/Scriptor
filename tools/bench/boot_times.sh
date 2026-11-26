#!/usr/bin/env bash
# Measures cold versus warm boot. Cold is a boot with a fresh empty throwaway
# cache directory; warm is a second boot from that same directory. Each boot is
# timed from process start to the first /health 200, and the per-stage
# scriptor_boot_stage_seconds samples are read from /metrics. The throwaway directory
# is removed afterwards.
#
# Usage: boot_times.sh --scratch-root DIR --engine PATH --model-dir DIR [--port N]
#          [--seqs N] [--max-model-len N] [--gpu-share F] [--timeout SECONDS]
#        boot_times.sh --vllm --scratch-root DIR --start-script PATH
#          --stop-script PATH --image IMAGE
#          [--port N] [--timeout SECONDS]
#        boot_times.sh --drop-page-cache-note
#
#   --scratch-root          existing directory under which mktemp -d creates the
#                           throwaway cache directory.
#   --drop-page-cache-note  prints the root command to run before a
#                           page-cache-cold measurement; this script never runs sudo.
#   --vllm                  runs the seat start script twice with
#                           BOOT_CACHE_DIR=<throwaway> exported; the start script
#                           must bind-mount that directory as the compile cache and
#                           must start the seat container (it uses docker). The
#                           throwaway directory is emptied by a throwaway container
#                           of --image because it is root-owned. Readiness is
#                           /v1/models answering 200. --stop-script stops the seat
#                           between the two boots and after the second.
#
# Defaults: port 8002, seqs 128, max-model-len 16384, gpu-share 0.60, timeout 600.
set -euo pipefail

mode=engine; scratch=""; engine=""; model_dir=""; start_script=""; stop_script=""; image=""
port=8002; seqs=128; max_len=16384; share=0.60; timeout=600

usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --scratch-root) scratch=$2; shift 2 ;;
    --engine) engine=$2; shift 2 ;;
    --model-dir) model_dir=$2; shift 2 ;;
    --start-script) start_script=$2; shift 2 ;;
    --stop-script) stop_script=$2; shift 2 ;;
    --image) image=$2; shift 2 ;;
    --port) port=$2; shift 2 ;;
    --seqs) seqs=$2; shift 2 ;;
    --max-model-len) max_len=$2; shift 2 ;;
    --gpu-share) share=$2; shift 2 ;;
    --timeout) timeout=$2; shift 2 ;;
    --vllm) mode=vllm; shift ;;
    --drop-page-cache-note)
      echo "For a page-cache-cold run, execute as root before this script:"
      echo "  sync && echo 3 > /proc/sys/vm/drop_caches"
      exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -d "$scratch" ] || { echo "--scratch-root must be an existing directory" >&2; usage >&2; exit 2; }
if [ "$mode" = engine ]; then
  if [ ! -x "$engine" ] || [ -z "$model_dir" ]; then echo "need --engine and --model-dir" >&2; usage >&2; exit 2; fi
else
  if [ ! -f "$start_script" ] || [ ! -f "$stop_script" ] || [ -z "$image" ]; then
    echo "need --start-script, --stop-script and --image" >&2; usage >&2; exit 2
  fi
fi

cache=$(mktemp -d "$scratch/boot-cache.XXXXXX")
epid=""

clean_cache() {
  if [ "$mode" = vllm ]; then
    docker run --rm --entrypoint /bin/sh -v "$cache:/d" "$image" -c 'rm -rf /d/* /d/.[!.]*'
  fi
  rm -rf "$cache"
}

stop_engine() {
  [ -n "$epid" ] || return 0
  kill -TERM "$epid" 2>/dev/null || true
  wait "$epid" 2>/dev/null || true
  epid=""
  until ! curl -s -m 2 -o /dev/null "http://127.0.0.1:$port/health"; do sleep 1; done
}

stop_vllm() { if [ "$mode" = vllm ]; then bash "$stop_script" > /dev/null 2>&1 || true; fi; }

finish() { stop_engine; stop_vllm; clean_cache; }
trap finish EXIT

wait_ready() {
  local path=$1 t0=$2 deadline
  deadline=$(echo "$t0 + $timeout" | bc)
  until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port$path")" = 200 ]; do
    if [ "$(echo "$(date +%s.%N) > $deadline" | bc)" = 1 ]; then
      echo "not ready within ${timeout}s" >&2
      return 1
    fi
    sleep 0.25
  done
}

boot_engine() {
  local label=$1 t0 secs
  t0=$(date +%s.%N)
  "$engine" serve --model "$model_dir" --port "$port" --served-model-name glm-ocr \
    --gpu-memory-utilization "$share" --max-num-seqs "$seqs" --max-model-len "$max_len" \
    --cache-dir "$cache" > "$scratch/boot-$label.engine.log" 2>&1 < /dev/null &
  epid=$!
  wait_ready /health "$t0"
  secs=$(echo "$(date +%s.%N) - $t0" | bc)
  echo "$label ready_seconds=$secs"
  curl -s -m 5 "http://127.0.0.1:$port/metrics" | grep '^scriptor_boot_stage_seconds' | sed "s/^/$label /" || true
  stop_engine
}

boot_vllm() {
  local label=$1 t0 secs
  t0=$(date +%s.%N)
  BOOT_CACHE_DIR="$cache" bash "$start_script" > "$scratch/boot-$label.seat.log" 2>&1
  wait_ready /v1/models "$t0"
  secs=$(echo "$(date +%s.%N) - $t0" | bc)
  echo "$label ready_seconds=$secs"
}

for label in cold warm; do
  if [ "$mode" = engine ]; then boot_engine "$label"; else boot_vllm "$label"; fi
  stop_vllm
done
