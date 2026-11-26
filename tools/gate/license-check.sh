#!/usr/bin/env bash
# License gate: cargo-deny license policy, notice-file coverage of every
# dependency, and the adaptation header on every .cu that names an upstream.
#
# Usage: tools/gate/license-check.sh [--write]
#   --write  regenerates the crate table in docs/third-party-notices.md first.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
notices="docs/third-party-notices.md"
begin="<!-- crates:begin -->"
end="<!-- crates:end -->"

crate_rows() {
  cargo metadata --format-version 1 --locked --offline | python3 -I -c '
import json, sys
m = json.load(sys.stdin)
members = set(m["workspace_members"])
rows = set()
for p in m["packages"]:
    if p["id"] in members:
        continue
    src = p["source"] or "path"
    if src.startswith("registry+"):
        src = "crates.io"
    lic = p["license"] or "UNKNOWN"
    rows.add((p["name"], p["version"], lic, src))
for r in sorted(rows):
    print("| " + " | ".join(r) + " |")
'
}

if [[ "${1:-}" == "--write" ]]; then
  tmp="$(mktemp)"
  rows="$(mktemp)"
  crate_rows > "$rows"
  awk -v b="$begin" -v e="$end" -v rows="$rows" '
    $0 == b { print; print "| Crate | Version | License | Source |"; print "|---|---|---|---|";
              while ((getline line < rows) > 0) print line; skip = 1; next }
    $0 == e { skip = 0 }
    !skip   { print }
  ' "$notices" > "$tmp"
  mv "$tmp" "$notices"
  rm -f "$rows"
fi

fail=0

cargo deny check licenses || fail=1

expected="$(crate_rows | sort)"
listed="$(awk -v b="$begin" -v e="$end" '
  $0 == b { on = 1; next } $0 == e { on = 0 }
  on && /^\| / && $0 !~ /^\| Crate / && $0 !~ /^\|---/ { print }
' "$notices" | sort)"
if [[ "$expected" != "$listed" ]]; then
  echo "license-check: $notices crate table differs from cargo metadata (run with --write)" >&2
  diff <(printf '%s\n' "$expected") <(printf '%s\n' "$listed") >&2 || true
  fail=1
fi
dups="$(printf '%s\n' "$listed" | awk -F' [|] ' 'NF { print $1 "|" $2 }' | sort | uniq -d)"
if [[ -n "$dups" ]]; then
  echo "license-check: duplicate notice rows: $dups" >&2
  fail=1
fi

if [[ -d kernels ]]; then
  while IFS= read -r f; do
    if grep -qiE 'vllm|flashinfer|cutlass|flash-attn|flash_attn|triton|llama\.cpp|sglang|tensorrt' "$f"; then
      for field in 'Origin:' 'License:' 'Modifications:'; do
        if ! grep -q "$field" "$f"; then
          echo "license-check: $f names an upstream project but lacks the '$field' notice header field" >&2
          fail=1
        fi
      done
    fi
  done < <(find kernels -type f \( -name '*.cu' -o -name '*.cuh' \) | sort)
fi

if [[ $fail -ne 0 ]]; then
  echo "license-check: FAILED" >&2
  exit 1
fi
echo "license-check: ok"
