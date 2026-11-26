#!/usr/bin/env python3
"""Writes synthetic JPEGs covering the encoder modes the nvJPEG gate compares.

Usage: gen_jpeg_modes.py [output_dir] [--page WIDTHxHEIGHT ...]
Output: <name>.jpg per mode under output_dir (default tests/fixtures/image/modes);
each --page adds a page-sized 4:2:0 baseline JPEG named page_<w>x<h>.jpg.
"""
import os
import sys

from PIL import Image, ImageDraw


def synth(width, height, seed=0):
    img = Image.new("RGB", (width, height))
    px = img.load()
    for y in range(height):
        for x in range(width):
            r = (x * 255) // max(1, width - 1)
            g = (y * 255) // max(1, height - 1)
            b = ((x * 7 + y * 13 + seed * 31) * 5) % 256
            px[x, y] = (r, g, b)
    draw = ImageDraw.Draw(img)
    for i in range(0, max(width, height), 11):
        draw.line([(i, 0), (0, i)], fill=(255 - i % 256, i % 256, 40), width=1)
    draw.rectangle([width // 5, height // 5, width // 2, height // 2], outline=(0, 0, 0))
    draw.text((width // 4, height // 3), "Text 0123", fill=(255, 255, 255))
    return img


def save(img, path, **kw):
    img.save(path, "JPEG", quality=85, **kw)


def main():
    args = sys.argv[1:]
    pages = []
    if "--page" in args:
        i = args.index("--page")
        pages = args[i + 1:]
        args = args[:i]
    out = args[0] if args else os.path.join(
        os.path.dirname(__file__), "../../tests/fixtures/image/modes")
    os.makedirs(out, exist_ok=True)
    cases = {
        "baseline_444": (synth(96, 64), dict(subsampling=0)),
        "baseline_422": (synth(96, 64, 1), dict(subsampling=1)),
        "baseline_420": (synth(96, 64, 2), dict(subsampling=2)),
        "progressive_444": (synth(96, 64, 5), dict(subsampling=0, progressive=True)),
        "progressive_420": (synth(96, 64, 6), dict(subsampling=2, progressive=True)),
        "gray": (synth(96, 64, 7).convert("L"), {}),
        "odd_420": (synth(67, 45, 8), dict(subsampling=2)),
        "odd_422": (synth(67, 45, 9), dict(subsampling=1)),
        "odd_444": (synth(67, 45, 10), dict(subsampling=0)),
        "odd_gray": (synth(67, 45, 11).convert("L"), {}),
        "tiny_420": (synth(3, 5, 12), dict(subsampling=2)),
        "wide_420": (synth(259, 17, 13), dict(subsampling=2)),
    }
    for name, (img, kw) in cases.items():
        save(img, os.path.join(out, name + ".jpg"), **kw)
    exif = Image.Exif()
    exif[0x0112] = 6
    save(synth(96, 64, 14), os.path.join(out, "exif_orient6_420.jpg"),
         subsampling=2, exif=exif.tobytes())
    for spec in pages:
        w, h = (int(v) for v in spec.split("x"))
        save(synth(w, h, 15), os.path.join(out, f"page_{w}x{h}.jpg"), subsampling=2)


if __name__ == "__main__":
    main()
