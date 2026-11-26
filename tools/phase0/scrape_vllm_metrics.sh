#!/usr/bin/env bash
# Read-only scraper for a running vLLM-compatible seat's Prometheus endpoint.
#
# Modes:
#   snapshot URL OUT            write the vllm:* sample lines of URL (the /metrics endpoint) to OUT
#   watch URL INTERVAL OUT [N]  append "epoch running waiting kv_usage" rows every INTERVAL
#                               seconds to OUT, forever or for N samples
#   table PRE POST WALL         print the sweep delta table for two snapshot files and a
#                               wall-clock duration in seconds
#   table-sample FILE WALL      like table, for one file holding both snapshots separated by
#                               "# --- snapshot post ---" marker lines
#   series FILE [CEILING]       summarise a watch file: samples, peak and mean running, time
#                               share at or above 90% of CEILING (default: peak)
#
# Only HTTP GET requests are issued; no container or seat is touched.
set -euo pipefail

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

snapshot() {
  local url=$1 out=$2
  curl -fsS --max-time 10 "$url" | grep -E '^vllm:' >"$out"
}

watch_series() {
  local url=$1 interval=$2 out=$3 count=${4:-0} taken=0 body
  while :; do
    if body=$(curl -fsS --max-time 5 "$url"); then
      printf '%s\n' "$body" | awk -v now="$(date +%s)" '
        $1 ~ /^vllm:num_requests_running\{/ { run = $NF }
        $1 ~ /^vllm:num_requests_waiting\{/ { wait = $NF }
        $1 ~ /^vllm:kv_cache_usage_perc\{/ { kv = $NF }
        END { printf "%s %s %s %s\n", now, run + 0, wait + 0, kv + 0 }' >>"$out"
    fi
    taken=$((taken + 1))
    if [ "$count" -gt 0 ] && [ "$taken" -ge "$count" ]; then break; fi
    sleep "$interval"
  done
}

table_awk() {
  awk -v wall="$1" '
    function key(line,   n, p) {
      n = line
      sub(/[ \t]+[^ \t]+$/, "", n)
      return n
    }
    function base(k) { sub(/\{.*/, "", k); return k }
    function pos(k,   m) {
      if (match(k, /position="[0-9]+"/)) {
        m = substr(k, RSTART + 10, RLENGTH - 11)
        return m + 0
      }
      return -1
    }
    /^[[:space:]]*#/ || NF < 2 { next }
    /^vllm:/ {
      k = key($0)
      v = $NF + 0
      if (FILENAME == ARGV[1]) pre[k] = v; else post[k] = v
      seen[k] = 1
    }
    END {
      for (k in seen) {
        d = post[k] - pre[k]
        b = base(k)
        sum[b] += d
        if (b == "vllm:spec_decode_num_accepted_tokens_per_pos_total") {
          p = pos(k)
          if (p >= 0) { perpos[p] += d; if (p > maxpos) maxpos = p }
        }
      }
      pt = sum["vllm:prompt_tokens_total"]
      gt = sum["vllm:generation_tokens_total"]
      steps = sum["vllm:iteration_tokens_total_count"]
      itok = sum["vllm:iteration_tokens_total_sum"]
      pfs = sum["vllm:request_prefill_time_seconds_sum"]
      pfc = sum["vllm:request_prefill_time_seconds_count"]
      dcs = sum["vllm:request_decode_time_seconds_sum"]
      dcc = sum["vllm:request_decode_time_seconds_count"]
      drafts = sum["vllm:spec_decode_num_drafts_total"]
      dtok = sum["vllm:spec_decode_num_draft_tokens_total"]
      acc = sum["vllm:spec_decode_num_accepted_tokens_total"]
      printf "%-34s %s\n", "metric", "value"
      printf "%-34s %.1f s\n", "wall", wall
      printf "%-34s %.0f\n", "prompt tokens", pt
      printf "%-34s %.0f\n", "generation tokens", gt
      if (wall > 0) printf "%-34s %.1f\n", "generation tokens / s", gt / wall
      printf "%-34s %.0f\n", "engine steps", steps
      if (steps > 0) {
        printf "%-34s %.2f ms\n", "wall / step", wall * 1000 / steps
        printf "%-34s %.1f\n", "mean scheduled tokens / step", itok / steps
      }
      printf "%-34s %.0f\n", "requests finished (prefill)", pfc
      if (pfc > 0) printf "%-34s %.1f ms\n", "mean request prefill time", pfs * 1000 / pfc
      printf "%-34s %.1f s\n", "sum request prefill time", pfs
      if (dcc > 0) printf "%-34s %.1f ms\n", "mean request decode time", dcs * 1000 / dcc
      printf "%-34s %.1f s\n", "sum request decode time", dcs
      if (pfc > 0) printf "%-34s %.1f\n", "generation tokens / request", gt / pfc
      if (drafts > 0) {
        printf "%-34s %.0f\n", "spec drafts", drafts
        printf "%-34s %.0f\n", "spec draft tokens", dtok
        printf "%-34s %.0f\n", "spec accepted tokens", acc
        if (dtok > 0) printf "%-34s %.3f\n", "spec accept rate (per draft token)", acc / dtok
        printf "%-34s %.3f\n", "spec tokens / step (1 + accepted)", 1 + acc / drafts
        for (i = 0; i <= maxpos; i++)
          printf "%-34s %.3f\n", "spec accept rate, position " i, perpos[i] / drafts
      }
    }' "$2" "$3"
}

split_sample() {
  local file=$1 pre=$2 post=$3
  awk -v pre="$pre" -v post="$post" '
    /^# --- snapshot pre ---/ { out = pre; next }
    /^# --- snapshot post ---/ { out = post; next }
    out != "" { print > out }' "$file"
}

series_summary() {
  local file=$1 ceiling=${2:-0}
  awk -v ceiling="$ceiling" '
    { n++; r[n] = $2; w[n] = $3; k[n] = $4; if ($2 > peak) peak = $2; if ($4 > kvpeak) kvpeak = $4; rs += $2; ws += $3 }
    END {
      if (n == 0) { print "no samples"; exit 1 }
      c = ceiling > 0 ? ceiling : peak
      for (i = 1; i <= n; i++) if (r[i] >= 0.9 * c) full++
      printf "samples            %d\n", n
      printf "peak running       %g\n", peak
      printf "mean running       %.1f\n", rs / n
      printf "mean waiting       %.1f\n", ws / n
      printf "peak kv usage      %.4f\n", kvpeak
      printf "share at >=90%% of %g running: %.0f%%\n", c, 100 * full / n
    }' "$file"
}

cmd=${1:-}
[ -n "$cmd" ] || usage
shift
case "$cmd" in
  snapshot) [ $# -eq 2 ] || usage; snapshot "$@" ;;
  watch) [ $# -ge 3 ] || usage; watch_series "$@" ;;
  table) [ $# -eq 3 ] || usage; table_awk "$3" "$1" "$2" ;;
  table-sample)
    [ $# -eq 2 ] || usage
    scratch=$(mktemp -d)
    trap 'rm -rf "$scratch"' EXIT
    split_sample "$1" "$scratch/pre" "$scratch/post"
    table_awk "$2" "$scratch/pre" "$scratch/post"
    ;;
  series) [ $# -ge 1 ] || usage; series_summary "$@" ;;
  *) usage ;;
esac
