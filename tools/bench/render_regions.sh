#!/usr/bin/env bash
# Rasterises PDFs at 200 dpi and tiles each page into 28-multiple JPEG crops.
#
# usage: render_regions.sh --out DIR [--tile-width PX] [--tile-height PX] PDF...
#
# Writes DIR/manifest.json and DIR/<pdf>-p<page>-r<row>-c<col>.jpg (quality 75).
# DIR is created by this script and is never inside the repository tree by default;
# crops are generated data and are not committed.
set -euo pipefail

out=""
tile_w=448
tile_h=476
pdfs=()
while [ $# -gt 0 ]; do
    case "$1" in
        --out) out="$2"; shift 2 ;;
        --tile-width) tile_w="$2"; shift 2 ;;
        --tile-height) tile_h="$2"; shift 2 ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) pdfs+=("$1"); shift ;;
    esac
done

if [ -z "$out" ] || [ "${#pdfs[@]}" -eq 0 ]; then
    echo "usage: $0 --out DIR [--tile-width PX] [--tile-height PX] PDF..." >&2
    exit 2
fi
if [ $((tile_w % 28)) -ne 0 ] || [ $((tile_h % 28)) -ne 0 ]; then
    echo "tile sizes must be multiples of 28" >&2
    exit 2
fi
command -v pdftoppm >/dev/null || { echo "pdftoppm not found" >&2; exit 1; }

mkdir -p "$out"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

for pdf in "${pdfs[@]}"; do
    stem="$(basename "${pdf%.*}")"
    pdftoppm -r 200 -png "$pdf" "$work/$stem"
done

tiler="$work/tile.py"
cat > "$tiler" <<'PY'
import json, os, sys
from PIL import Image

work, out, tw, th = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
regions = []
for name in sorted(os.listdir(work)):
    if not name.endswith(".png"):
        continue
    stem = name[:-4]
    page = Image.open(os.path.join(work, name)).convert("RGB")
    pw, ph = page.size
    for r, top in enumerate(range(0, ph, th)):
        for c, left in enumerate(range(0, pw, tw)):
            box = (left, top, min(left + tw, pw), min(top + th, ph))
            w, h = box[2] - box[0], box[3] - box[1]
            w28, h28 = w // 28 * 28, h // 28 * 28
            if w28 < 112 or h28 < 112:
                continue
            crop = page.crop((box[0], box[1], box[0] + w28, box[1] + h28))
            fname = f"{stem}-r{r}-c{c}.jpg"
            crop.save(os.path.join(out, fname), "JPEG", quality=75)
            regions.append({"file": fname, "width": w28, "height": h28, "tasks": ["text"]})
with open(os.path.join(out, "manifest.json"), "w") as f:
    json.dump({"regions": regions}, f, indent=2)
    f.write("\n")
print(f"wrote {len(regions)} crops to {out}")
PY
python3 -I "$tiler" "$work" "$out" "$tile_w" "$tile_h"
