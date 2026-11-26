#!/usr/bin/env bash
# Measures seat boot time: container StartedAt (read with `docker inspect`) to the
# first HTTP 200 from /v1/models that lists the served model name.
#
# Usage: boot_time.sh [--container NAME] [--url URL] [--model NAME]
#                     [--started-at RFC3339] [--interval SECONDS]
#
# Defaults: container glm-ocr, url http://127.0.0.1:8002/v1/models, model glm-ocr,
# interval 0.5. --started-at replaces the docker query (for a seat started
# elsewhere or for a dry run). Run it right after the seat start script returns:
# a seat that is already serving yields only an upper bound, reported as such.
#
# The script only reads: `docker inspect` and HTTP GET requests.
set -euo pipefail

container=glm-ocr
url=http://127.0.0.1:8002/v1/models
model=glm-ocr
started_at=
interval=0.5

while [ $# -gt 0 ]; do
  case "$1" in
    --container) container=$2; shift 2 ;;
    --url) url=$2; shift 2 ;;
    --model) model=$2; shift 2 ;;
    --started-at) started_at=$2; shift 2 ;;
    --interval) interval=$2; shift 2 ;;
    *) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
  esac
done

if [ -z "$started_at" ]; then
  started_at=$(docker inspect --format '{{.State.StartedAt}}' "$container")
fi
start_epoch=$(date -d "$started_at" +%s.%N)

first_probe=1
already_serving=0
while :; do
  body=$(curl -s --max-time 3 -w '\n%{http_code}' "$url" || true)
  code=${body##*$'\n'}
  if [ "$code" = 200 ] && printf '%s' "$body" | grep -q "\"$model\""; then
    [ "$first_probe" -eq 1 ] && already_serving=1
    break
  fi
  first_probe=0
  sleep "$interval"
done

ready_epoch=$(date +%s.%N)
awk -v s="$start_epoch" -v r="$ready_epoch" -v up="$already_serving" 'BEGIN {
  printf "boot_seconds=%.2f%s\n", r - s, up ? " (seat was already serving at the first probe: upper bound)" : ""
}'
