#!/usr/bin/env bash
# Offline checks of install.sh: help text, the dry-run plan, and the refusal on
# a non-aarch64 host. curl is stubbed to fail so any network use is caught.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
installer="$root/install.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
expect() { printf '%s' "$2" | grep -qF -- "$1" || fail "missing: $1"; }

stubs="$work/stubs"
mkdir -p "$stubs"
printf '#!/bin/sh\necho "network access attempted" >&2\nexit 99\n' >"$stubs/curl"
chmod +x "$stubs/curl"
make_uname() {
  # shellcheck disable=SC2016
  printf '#!/bin/sh\ncase "$1" in -s) echo %s;; -m) echo %s;; *) echo %s;; esac\n' \
    "$1" "$2" "$1" >"$stubs/uname"
  chmod +x "$stubs/uname"
}

prefix="$work/prefix"
bindir="$work/bin"

help_out="$(bash "$installer" --help)"
for word in binary model layout onnxruntime all env check --dry-run --prefix --bindir; do
  expect "$word" "$help_out"
done

make_uname Linux aarch64
plan="$(PATH="$stubs:$PATH" bash "$installer" --dry-run --prefix "$prefix" --bindir "$bindir" all)"
expect "aarch64-linux-cuda.tar.gz" "$plan"
expect "zai-org/GLM-OCR" "$plan"
expect "2e85a62840ccac27daa451df36c736c4636b8628" "$plan"
expect "pp-doclayoutv3-e4f489dc.onnx" "$plan"
expect "6a99236055d3efed8b5bb9203ebed4c7ed36836d0259add97ffc884adf4346ca" "$plan"
expect "1bc031df14da045dddf3e12f1c01afb95a39c08917233179855301fb152009b4" "$plan"
expect "onnxruntime-1.28.0-linux-aarch64-cuda13-sm121.tar.gz" "$plan"
expect "$prefix/onnxruntime/1.28.0-cuda13" "$plan"
cpu_plan="$(PATH="$stubs:$PATH" bash "$installer" --dry-run --ort-cpu --prefix "$prefix" --bindir "$bindir" onnxruntime)"
expect "onnxruntime-linux-aarch64-1.28.0.tgz" "$cpu_plan"
expect "$prefix/onnxruntime/1.28.0-cpu" "$cpu_plan"
[ ! -e "$prefix" ] || fail "dry run created the prefix"
[ ! -e "$bindir" ] || fail "dry run created the bindir"

env_out="$(PATH="$stubs:$PATH" bash "$installer" --prefix "$prefix" --bindir "$bindir" env)"
expect "SCRIPTOR_BRIDGE_LAYOUT_RUNTIME=" "$env_out"
expect "SCRIPTOR_BRIDGE_LAYOUT_WEIGHTS=" "$env_out"
expect "scriptor serve --model $prefix/models/glm-ocr" "$env_out"

make_uname Linux x86_64
if refusal="$(PATH="$stubs:$PATH" bash "$installer" --dry-run --prefix "$prefix" all 2>&1)"; then
  fail "x86_64 host was not refused"
fi
expect "unsupported host Linux/x86_64" "$refusal"
expect "docs/install.md" "$refusal"

make_uname Darwin arm64
if PATH="$stubs:$PATH" bash "$installer" --dry-run --prefix "$prefix" binary >/dev/null 2>&1; then
  fail "Darwin host was not refused"
fi

if bash "$installer" --bogus >/dev/null 2>&1; then
  fail "unknown argument accepted"
fi

echo "test_install: ok"
