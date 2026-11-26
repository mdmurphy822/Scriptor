#!/usr/bin/env bash
# Seat template: GLM-OCR served by scriptor as a detached process.
# Role: ocr. Serves an OpenAI-compatible API at http://127.0.0.1:{{port}}/v1.
# The engine runs alone: --no-bridge keeps the document bridge off.
# Returns once the engine is launched; the seat is ready when GET /v1/models
# answers 200 and lists the served name. SIGTERM drains: the engine stops
# admitting requests, finishes those in flight, then exits.
#
# Params:
#   binary                  path to the scriptor binary
#   model_dir               checkpoint snapshot directory
#   engine_file             engine-native weights file; empty converts the
#                           checkpoint into cache_dir on first start
#   served_name             model name exposed by the API          (glm-ocr)
#   port                    loopback port                          (8002)
#   gpu_memory_utilization  device memory fraction                 (0.60)
#   max_num_seqs            concurrent sequences; empty derives the largest
#                           count that fits the memory share
#   max_model_len           context length                         (16384)
#   max_pixels              image pixel ceiling; empty uses the model default
#   speculative_tokens      draft tokens per step; 0 disables      (0)
#   cache_dir               converted engine files and tuning; empty uses
#                           $XDG_CACHE_HOME/semantik/seat-cache/scriptor
#   state_dir               pid file and log directory
set -euo pipefail

binary="{{binary}}"
model_dir="{{model_dir}}"
engine_file="{{engine_file}}"
state_dir="{{state_dir}}"
max_pixels="{{max_pixels}}"
cache_dir="{{cache_dir}}"
[[ -n "${cache_dir}" ]] || cache_dir="${XDG_CACHE_HOME:-${HOME}/.cache}/semantik/seat-cache/scriptor"
[[ -n "${state_dir}" ]] || { echo "state_dir is required" >&2; exit 1; }
[[ -x "${binary}" ]] || { echo "engine binary is not executable: ${binary}" >&2; exit 1; }
[[ -d "${model_dir}" ]] || { echo "model directory not found: ${model_dir}" >&2; exit 1; }

mkdir -p "${state_dir}" "${cache_dir}"
pid_file="${state_dir}/scriptor.pid"
log_file="${state_dir}/scriptor.log"

if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
  echo "scriptor already running (pid $(cat "${pid_file}"))" >&2
  exit 1
fi

convert=0
if [[ -z "${engine_file}" ]]; then
  engine_file="${cache_dir}/glm-ocr.engine"
  [[ -f "${engine_file}" ]] || convert=1
fi

serve=(
  "${binary}" serve
  --model "${model_dir}"
  --engine-file "${engine_file}"
  --cache-dir "${cache_dir}"
  --bind "127.0.0.1:{{port}}"
  --served-model-name "{{served_name}}"
  --gpu-memory-utilization "{{gpu_memory_utilization}}"
  --max-model-len "{{max_model_len}}"
  --num-speculative-tokens "{{speculative_tokens}}"
  --no-bridge
)
if [[ -n "{{max_num_seqs}}" ]]; then
  serve+=(--max-num-seqs "{{max_num_seqs}}")
fi
if [[ -n "${max_pixels}" ]]; then
  serve+=(--max-pixels "${max_pixels}")
fi

# The detached shell converts when needed and then execs the engine, so the
# recorded pid is the engine's own and SIGTERM reaches it directly.
# shellcheck disable=SC2016
setsid nohup bash -c '
  convert="$1"; shift
  if [[ "${convert}" == 1 ]]; then
    "$1" convert --model-dir "$2" --out "$3" || exit 1
  fi
  shift 3
  exec "$@"
' scriptor "${convert}" "${binary}" "${model_dir}" "${engine_file}" "${serve[@]}" \
  >>"${log_file}" 2>&1 </dev/null &
echo "$!" >"${pid_file}"
