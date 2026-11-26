#!/usr/bin/env python3
"""Writes the scriptor-image JPEG fixtures and their PIL-derived goldens.

Usage: gen_image_goldens.py [output_dir]
Output: <name>.jpg per case and goldens.json holding the PIL-decoded RGB bytes
(base64), the smart_resize target, token count and a PIL bicubic resize of the
decoded image to the target (non-conforming cases only).
"""
import base64
import io
import json
import math
import os
import sys

from PIL import Image

FACTOR, MIN_PIXELS, MAX_PIXELS, FRAMES = 28, 12544, 9633792, 2


def smart_resize(h, w):
    h_bar = round(h / FACTOR) * FACTOR
    w_bar = round(w / FACTOR) * FACTOR
    if FRAMES * h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt(FRAMES * h * w / MAX_PIXELS)
        h_bar = max(FACTOR, math.floor(h / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(w / beta / FACTOR) * FACTOR)
    elif FRAMES * h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (FRAMES * h * w))
        h_bar = math.ceil(h * beta / FACTOR) * FACTOR
        w_bar = math.ceil(w * beta / FACTOR) * FACTOR
    return h_bar, w_bar


def synth(width, height):
    img = Image.new("RGB", (width, height))
    px = img.load()
    for y in range(height):
        for x in range(width):
            r = (x * 255) // max(1, width - 1)
            g = (y * 255) // max(1, height - 1)
            b = 255 if ((x // 7 + y // 5) % 2 == 0) else 40
            if (x * 3 + y * 5) % 41 < 4:
                r, g, b = 255 - r, 255 - g, 255 - b
            px[x, y] = (r, g, b)
    return img


CASES = [("w28_h28", 28, 28), ("w140_h112", 140, 112), ("w448_h420", 448, 420), ("w777_h333", 777, 333)]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "..", "tests", "fixtures", "image")
    os.makedirs(out, exist_ok=True)
    cases = []
    for name, w, h in CASES:
        buf = io.BytesIO()
        synth(w, h).save(buf, "JPEG", quality=75)
        data = buf.getvalue()
        with open(os.path.join(out, name + ".jpg"), "wb") as f:
            f.write(data)
        dec = Image.open(io.BytesIO(data)).convert("RGB")
        th, tw = smart_resize(h, w)
        resized = dec if (th, tw) == (h, w) else dec.resize((tw, th), Image.BICUBIC)
        cases.append({
            "name": name, "file": name + ".jpg", "width": w, "height": h,
            "conforming": (th, tw) == (h, w),
            "target_height": th, "target_width": tw,
            "tokens": (th // FACTOR) * (tw // FACTOR),
            "pixels_b64": base64.b64encode(dec.tobytes()).decode(),
            "resized_b64": None if (th, tw) == (h, w) else base64.b64encode(resized.tobytes()).decode(),
        })
    with open(os.path.join(out, "goldens.json"), "w") as f:
        json.dump({"cases": cases}, f)


if __name__ == "__main__":
    main()
