#!/usr/bin/env bash
# Seat template: stops the scriptor seat.
# Sends SIGTERM so the engine drains (stops admitting, finishes in-flight
# requests) and waits until the process exits.
# Params:
#   state_dir   pid file directory
set -euo pipefail

pid_file="{{state_dir}}/scriptor.pid"
[[ -f "${pid_file}" ]] || exit 0
pid="$(cat "${pid_file}")"
if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
  kill -TERM "${pid}"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 0.2
  done
fi
rm -f "${pid_file}"
