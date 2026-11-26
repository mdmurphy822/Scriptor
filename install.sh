#!/usr/bin/env bash
# Scriptor installer: fetches the release binary, the GLM-OCR checkpoint, the
# PP-DocLayoutV3 layout weights and an onnxruntime library, each as an explicit,
# idempotent step. The scriptor server itself never downloads anything.
set -euo pipefail

REPO_URL="https://github.com/mdmurphy822/Scriptor"

MODEL_REPO="zai-org/GLM-OCR"
MODEL_REVISION="2e85a62840ccac27daa451df36c736c4636b8628"

# Dependency assets attached to this Scriptor release: the layout model and the
# CUDA onnxruntime build the published benchmark ran with.
DEPS_URL="$REPO_URL/releases/download/v0.1.0"

# ONNX export of PaddlePaddle/PP-DocLayoutV3_safetensors at revision
# e4f489dc536556fc1be02b973ba06756e0d4a2ac (Apache-2.0).
LAYOUT_FILE="pp-doclayoutv3-e4f489dc.onnx"
LAYOUT_SHA256="6a99236055d3efed8b5bb9203ebed4c7ed36836d0259add97ffc884adf4346ca"

# ONNX Runtime 1.28.0 built with the CUDA execution provider for CUDA 13 and
# compute capability 12.1 (GB10). --ort-cpu selects the official CPU build.
ORT_VERSION="1.28.0"
ORT_CUDA_ASSET="onnxruntime-1.28.0-linux-aarch64-cuda13-sm121.tar.gz"
ORT_CUDA_SHA256="1bc031df14da045dddf3e12f1c01afb95a39c08917233179855301fb152009b4"
ORT_RELEASE_URL="https://github.com/microsoft/onnxruntime/releases/download"
ORT_CPU=0

PREFIX="${HOME}/.local/share/scriptor"
BINDIR="${HOME}/.local/bin"
VERSION="latest"
DRY_RUN=0
ORT_URL=""
ORT_SHA256=""
ORT_LIB=""
COMMAND=""
TMP_DIR=""

usage() {
  cat <<'EOF'
Usage: install.sh <command> [options]

Commands (each is opt-in and safe to repeat):
  binary       download and verify the release tarball, install `scriptor`
  model        fetch the GLM-OCR checkpoint at the pinned revision
  layout       fetch the PP-DocLayoutV3 ONNX weights and verify their sha256
  onnxruntime  fetch an onnxruntime shared library for aarch64 Linux
  all          binary, model, layout and onnxruntime, in that order
  env          print the environment lines and the `scriptor serve` command
  check        verify driver, GPU, installed files, checksums and the binary

Options:
  --prefix DIR       data directory (default ~/.local/share/scriptor)
  --bindir DIR       directory for the `scriptor` binary (default ~/.local/bin)
  --version VER      release to install, such as 0.1.0 (default: latest)
  --ort-cpu          install the official CPU onnxruntime build instead of the
                     CUDA build; the layout step then runs on the CPU
  --ort-version VER  onnxruntime release for --ort-cpu (default 1.28.0)
  --ort-url URL      https URL of an onnxruntime tarball you supply (for example
                     an aarch64 CUDA build); requires --ort-sha256
  --ort-sha256 HEX   expected sha256 of the onnxruntime tarball
  --ort-lib PATH     use an onnxruntime shared library you already have
  --dry-run          print the actions without doing them
  -h, --help         show this help

The server downloads nothing at runtime; this script is the only network step.
EOF
}

log() { printf 'install: %s\n' "$*"; }
die() { printf 'install: error: %s\n' "$*" >&2; exit 1; }

cleanup() {
  if [ -n "$TMP_DIR" ] && [ -d "$TMP_DIR" ]; then
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT

# Runs a mutating action, or prints it under --dry-run.
run() {
  if [ "$DRY_RUN" = 1 ]; then
    printf 'plan: %s\n' "$*"
  else
    "$@"
  fi
}

need() {
  command -v "$1" >/dev/null 2>&1 || die "required tool not found: $1"
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

require_aarch64_linux() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"
  if [ "$os" != "Linux" ] || { [ "$arch" != "aarch64" ] && [ "$arch" != "arm64" ]; }; then
    die "unsupported host ${os}/${arch}: the prebuilt binary and onnxruntime step need aarch64 Linux. Build from source instead; see docs/install.md (Build from source)."
  fi
}

make_tmp() {
  if [ -z "$TMP_DIR" ]; then
    mkdir -p "$PREFIX"
    TMP_DIR="$(mktemp -d "$PREFIX/.tmp.XXXXXX")"
  fi
}

# Downloads an https URL to a file, failing on any HTTP error.
fetch() {
  local url="$1" dest="$2"
  case "$url" in https://*) ;; *) die "refusing non-https URL: $url" ;; esac
  need curl
  mkdir -p "$(dirname "$dest")"
  curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
    --retry 3 --output "$dest" "$url"
}

resolve_version() {
  if [ "$VERSION" != "latest" ]; then
    printf '%s' "${VERSION#v}"
    return
  fi
  if [ "$DRY_RUN" = 1 ]; then
    printf '<latest>'
    return
  fi
  need curl
  local location
  location="$(curl --proto '=https' --tlsv1.2 --silent --head --output /dev/null \
    --write-out '%{redirect_url}' "$REPO_URL/releases/latest")"
  [ -n "$location" ] || die "could not resolve the latest release"
  printf '%s' "${location##*/v}"
}

step_binary() {
  require_aarch64_linux
  local ver asset base
  ver="$(resolve_version)"
  asset="scriptor-v${ver}-aarch64-linux-cuda.tar.gz"
  base="$REPO_URL/releases/download/v${ver}"
  log "binary: $base/$asset (+ .sha256) -> $BINDIR/scriptor"
  if [ "$DRY_RUN" = 0 ] && [ -x "$BINDIR/scriptor" ] \
    && "$BINDIR/scriptor" --version 2>/dev/null | grep -q "$ver"; then
    log "binary: scriptor $ver already installed"
    return
  fi
  if [ "$DRY_RUN" = 1 ]; then
    printf 'plan: download %s/%s and %s/%s.sha256 into a temp dir under %s\n' \
      "$base" "$asset" "$base" "$asset" "$PREFIX"
    printf 'plan: verify sha256, extract, copy scriptor to %s/scriptor atomically\n' "$BINDIR"
    return
  fi
  make_tmp
  fetch "$base/$asset" "$TMP_DIR/$asset"
  fetch "$base/$asset.sha256" "$TMP_DIR/$asset.sha256"
  local want have
  want="$(cut -d' ' -f1 "$TMP_DIR/$asset.sha256" | head -n1)"
  have="$(sha256_of "$TMP_DIR/$asset")"
  [ "$want" = "$have" ] || die "sha256 mismatch for $asset (expected $want, got $have)"
  mkdir -p "$TMP_DIR/extract" "$BINDIR"
  tar -xzf "$TMP_DIR/$asset" -C "$TMP_DIR/extract"
  local found
  found="$(find "$TMP_DIR/extract" -type f -name scriptor | head -n1)"
  [ -n "$found" ] || die "scriptor binary not found in $asset"
  cp "$found" "$BINDIR/.scriptor.new.$$"
  chmod 0755 "$BINDIR/.scriptor.new.$$"
  mv -f "$BINDIR/.scriptor.new.$$" "$BINDIR/scriptor"
  log "binary: installed $BINDIR/scriptor"
}

model_dir() { printf '%s/models/glm-ocr' "$PREFIX"; }

step_model() {
  local dest
  dest="$(model_dir)"
  log "model: $MODEL_REPO @ $MODEL_REVISION -> $dest"
  if [ "$DRY_RUN" = 0 ] && [ -f "$dest/.revision" ] \
    && [ "$(cat "$dest/.revision")" = "$MODEL_REVISION" ] \
    && [ -f "$dest/model.safetensors" ] && [ -f "$dest/config.json" ]; then
    log "model: revision already present"
    return
  fi
  if [ "$DRY_RUN" = 1 ]; then
    printf 'plan: hf download %s --revision %s into a temp dir (else git lfs, else curl per file via the Hugging Face API), then move to %s\n' \
      "$MODEL_REPO" "$MODEL_REVISION" "$dest"
    return
  fi
  make_tmp
  local stage="$TMP_DIR/model"
  mkdir -p "$stage"
  if command -v hf >/dev/null 2>&1; then
    HF_HUB_OFFLINE=0 hf download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$stage"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_HUB_OFFLINE=0 huggingface-cli download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$stage"
  elif command -v git >/dev/null 2>&1 && git lfs version >/dev/null 2>&1; then
    git clone --quiet "https://huggingface.co/$MODEL_REPO" "$stage/repo"
    git -C "$stage/repo" checkout --quiet "$MODEL_REVISION"
    git -C "$stage/repo" lfs pull
    rm -rf "$stage/repo/.git"
    mv "$stage/repo"/* "$stage/"
    rmdir "$stage/repo"
  else
    fetch_model_with_curl "$stage"
  fi
  if [ ! -f "$stage/config.json" ] || [ ! -f "$stage/model.safetensors" ]; then
    die "model download is incomplete (config.json or model.safetensors missing)"
  fi
  rm -rf "$stage/.cache"
  printf '%s\n' "$MODEL_REVISION" >"$stage/.revision"
  mkdir -p "$(dirname "$dest")"
  rm -rf "$dest.old"
  [ -d "$dest" ] && mv "$dest" "$dest.old"
  mv "$stage" "$dest"
  rm -rf "$dest.old"
  log "model: installed $dest"
}

# Lists the files of the pinned revision through the Hugging Face API, then
# downloads each over https and checks the size and, for large files, the sha256.
fetch_model_with_curl() {
  local stage="$1" listing
  need python3
  listing="$TMP_DIR/tree.json"
  fetch "https://huggingface.co/api/models/$MODEL_REPO/tree/$MODEL_REVISION" "$listing"
  python3 -I - "$listing" >"$TMP_DIR/files.tsv" <<'PY'
import json, sys
for entry in json.load(open(sys.argv[1])):
    if entry.get("type") == "file":
        oid = (entry.get("lfs") or {}).get("oid", "")
        print(entry["path"], entry.get("size", 0), oid, sep="\t")
PY
  local path size oid
  while IFS=$'\t' read -r path size oid; do
    [ "$path" = ".gitattributes" ] && continue
    fetch "https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REVISION/$path" "$stage/$path"
    if [ "$(wc -c <"$stage/$path" | tr -d ' ')" != "$size" ]; then
      die "size mismatch for $path (expected $size bytes)"
    fi
    if [ -n "$oid" ] && [ "$(sha256_of "$stage/$path")" != "$oid" ]; then
      die "sha256 mismatch for $path"
    fi
  done <"$TMP_DIR/files.tsv"
}

layout_path() { printf '%s/layout/%s' "$PREFIX" "$LAYOUT_FILE"; }

step_layout() {
  local dest url
  dest="$(layout_path)"
  url="$DEPS_URL/$LAYOUT_FILE"
  log "layout: $url -> $dest"
  if [ "$DRY_RUN" = 0 ] && [ -f "$dest" ] && [ "$(sha256_of "$dest")" = "$LAYOUT_SHA256" ]; then
    log "layout: weights already present and verified"
    return
  fi
  if [ "$DRY_RUN" = 1 ]; then
    printf 'plan: download %s, verify sha256 %s, write %s.sha256, move to %s\n' \
      "$url" "$LAYOUT_SHA256" "$dest" "$dest"
    return
  fi
  make_tmp
  fetch "$url" "$TMP_DIR/$LAYOUT_FILE"
  local have
  have="$(sha256_of "$TMP_DIR/$LAYOUT_FILE")"
  [ "$have" = "$LAYOUT_SHA256" ] || die "sha256 mismatch for $LAYOUT_FILE (expected $LAYOUT_SHA256, got $have)"
  mkdir -p "$(dirname "$dest")"
  printf '%s  %s\n' "$have" "$LAYOUT_FILE" >"$TMP_DIR/$LAYOUT_FILE.sha256"
  mv -f "$TMP_DIR/$LAYOUT_FILE.sha256" "$dest.sha256"
  mv -f "$TMP_DIR/$LAYOUT_FILE" "$dest"
  log "layout: installed $dest"
}

ort_root() {
  if [ -n "$ORT_URL" ]; then
    printf '%s/onnxruntime/custom' "$PREFIX"
  elif [ "$ORT_CPU" = 1 ]; then
    printf '%s/onnxruntime/%s-cpu' "$PREFIX" "$ORT_VERSION"
  else
    printf '%s/onnxruntime/1.28.0-cuda13' "$PREFIX"
  fi
}

ort_library() {
  if [ -n "$ORT_LIB" ]; then
    printf '%s' "$ORT_LIB"
  else
    printf '%s/lib/libonnxruntime.so' "$(ort_root)"
  fi
}

step_onnxruntime() {
  if [ -n "$ORT_LIB" ]; then
    log "onnxruntime: using the library you supplied: $ORT_LIB"
    [ "$DRY_RUN" = 1 ] || [ -e "$ORT_LIB" ] || die "--ort-lib path does not exist: $ORT_LIB"
    return
  fi
  require_aarch64_linux
  local root url sha
  root="$(ort_root)"
  if [ -n "$ORT_URL" ]; then
    [ -n "$ORT_SHA256" ] || die "--ort-url requires --ort-sha256"
    url="$ORT_URL"
    sha="$ORT_SHA256"
    log "onnxruntime: your tarball $url -> $root"
  elif [ "$ORT_CPU" = 1 ]; then
    url="$ORT_RELEASE_URL/v${ORT_VERSION}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz"
    sha="$ORT_SHA256"
    log "onnxruntime: official CPU aarch64 build $url -> $root"
    log "onnxruntime: the layout step will run on the CPU."
  else
    url="$DEPS_URL/$ORT_CUDA_ASSET"
    sha="$ORT_CUDA_SHA256"
    log "onnxruntime: CUDA 13 build for the GB10 $url -> $root"
    log "onnxruntime: needs libcudart.so.13, libcublas.so.13 and libcublasLt.so.13 (CUDA 13 toolkit) on the library path."
  fi
  if [ "$DRY_RUN" = 0 ] && [ -e "$root/lib/libonnxruntime.so" ]; then
    log "onnxruntime: already installed at $root"
    return
  fi
  if [ "$DRY_RUN" = 1 ]; then
    printf 'plan: download %s, verify sha256 (%s), extract into %s\n' \
      "$url" "${sha:-none published; the computed digest is printed}" "$root"
    return
  fi
  make_tmp
  fetch "$url" "$TMP_DIR/ort.tgz"
  local have
  have="$(sha256_of "$TMP_DIR/ort.tgz")"
  if [ -n "$sha" ]; then
    [ "$have" = "$sha" ] || die "sha256 mismatch for onnxruntime tarball (expected $sha, got $have)"
  else
    log "onnxruntime: no checksum supplied; downloaded tarball sha256 is $have"
  fi
  mkdir -p "$TMP_DIR/ort"
  tar -xzf "$TMP_DIR/ort.tgz" -C "$TMP_DIR/ort" --strip-components=1
  [ -e "$TMP_DIR/ort/lib/libonnxruntime.so" ] || die "libonnxruntime.so not found in the tarball"
  mkdir -p "$(dirname "$root")"
  rm -rf "$root.old"
  [ -d "$root" ] && mv "$root" "$root.old"
  mv "$TMP_DIR/ort" "$root"
  rm -rf "$root.old"
  log "onnxruntime: installed $root"
}

has_cuda_provider() {
  [ -n "$ORT_LIB" ] && return 0
  [ -e "$(ort_root)/lib/libonnxruntime_providers_cuda.so" ]
}

step_env() {
  local lib weights model
  lib="$(ort_library)"
  weights="$(layout_path)"
  model="$(model_dir)"
  echo "# Add to your shell profile or service environment"
  echo "export SCRIPTOR_BRIDGE_LAYOUT_RUNTIME=\"$lib\""
  echo "export SCRIPTOR_BRIDGE_LAYOUT_WEIGHTS=\"$weights\""
  if ! has_cuda_provider; then
    echo "export SCRIPTOR_BRIDGE_LAYOUT_DEVICE=cpu   # no CUDA execution provider installed"
  fi
  echo "# Optional bridge settings (see docs/bridge.md): SCRIPTOR_BRIDGE_LAYOUT_BATCH_SIZE,"
  echo "# SCRIPTOR_BRIDGE_WORKERS, SCRIPTOR_BRIDGE_FIGURE_DPI, SCRIPTOR_BRIDGE_CHECKPOINT_DIR"
  echo
  echo "$BINDIR/scriptor serve --model $model \\"
  echo "  --max-model-len 16384 --max-num-seqs 32 \\"
  echo "  --gpu-memory-utilization 0.60 --num-speculative-tokens 4"
}

step_check() {
  local failed=0 name
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "check: driver and GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1)"
  else
    log "check: nvidia-smi not found (CUDA driver missing)"
    failed=1
  fi
  if [ -x "$BINDIR/scriptor" ]; then
    log "check: $("$BINDIR/scriptor" --version 2>&1 | head -n1)"
  else
    log "check: missing $BINDIR/scriptor"
    failed=1
  fi
  for name in config.json model.safetensors .revision; do
    if [ -f "$(model_dir)/$name" ]; then
      log "check: model file present: $name"
    else
      log "check: model file missing: $name"
      failed=1
    fi
  done
  if [ -f "$(layout_path)" ]; then
    if [ "$(sha256_of "$(layout_path)")" = "$LAYOUT_SHA256" ]; then
      log "check: layout weights sha256 ok"
    else
      log "check: layout weights sha256 mismatch"
      failed=1
    fi
  else
    log "check: layout weights missing: $(layout_path)"
    failed=1
  fi
  if [ -e "$(ort_library)" ]; then
    log "check: onnxruntime library present: $(ort_library)"
    has_cuda_provider || log "check: no CUDA execution provider; layout runs on the CPU"
  else
    log "check: onnxruntime library missing: $(ort_library)"
    failed=1
  fi
  [ "$failed" = 0 ] || die "check found problems"
  log "check: ok"
}

parse_args() {
  [ "$#" -gt 0 ] || { usage; exit 2; }
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -h|--help) usage; exit 0 ;;
      --dry-run) DRY_RUN=1 ;;
      --prefix) [ "$#" -ge 2 ] || die "--prefix needs a value"; PREFIX="$2"; shift ;;
      --bindir) [ "$#" -ge 2 ] || die "--bindir needs a value"; BINDIR="$2"; shift ;;
      --version) [ "$#" -ge 2 ] || die "--version needs a value"; VERSION="$2"; shift ;;
      --ort-version) [ "$#" -ge 2 ] || die "--ort-version needs a value"; ORT_VERSION="$2"; shift ;;
      --ort-url) [ "$#" -ge 2 ] || die "--ort-url needs a value"; ORT_URL="$2"; shift ;;
      --ort-sha256) [ "$#" -ge 2 ] || die "--ort-sha256 needs a value"; ORT_SHA256="$2"; shift ;;
      --ort-cpu) ORT_CPU=1 ;;
      --ort-lib) [ "$#" -ge 2 ] || die "--ort-lib needs a value"; ORT_LIB="$2"; shift ;;
      binary|model|layout|onnxruntime|all|env|check)
        [ -z "$COMMAND" ] || die "only one command may be given"
        COMMAND="$1" ;;
      *) die "unknown argument: $1 (see --help)" ;;
    esac
    shift
  done
  [ -n "$COMMAND" ] || { usage; exit 2; }
}

main() {
  parse_args "$@"
  [ "$DRY_RUN" = 0 ] || log "dry run: nothing will be downloaded or written"
  case "$COMMAND" in
    binary) step_binary ;;
    model) step_model ;;
    layout) step_layout ;;
    onnxruntime) step_onnxruntime ;;
    all)
      step_binary
      step_model
      step_layout
      step_onnxruntime
      if [ "$DRY_RUN" = 0 ]; then
        echo
        step_env
      fi
      ;;
    env) step_env ;;
    check) step_check ;;
  esac
}

main "$@"
