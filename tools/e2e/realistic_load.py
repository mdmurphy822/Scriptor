"""Synthetic OCR load shaped like the Semantik bridge's region crops.

`realistic_load.py URL REQUESTS CONCURRENCY [--seed S] [--cache DIR] [--scale F] [--out FILE]
[--drain-metrics FILE]`

Each request carries one generated PNG: a crop of rendered English-like text at a
300-dpi-like glyph size whose pixel width and height are drawn from the measured
distribution of the live benchmark's region bounding boxes (per-request image tokens
average ~290 and completions ~75 tokens, like the live load). The chat body uses the
bridge's sampling fields. CONCURRENCY workers keep that many requests in flight, as the
bridge's seat semaphore does. Prints one JSON summary line; `--out` also writes every
request's usage. `--drain-metrics` scrapes /metrics on a side thread when the last request
is dispatched and adds `drain_mark_seconds` to the summary: before the mark every worker
keeps a request in flight, after it the in-flight set only drains. No real documents are
used.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

# Quantiles (p10, p25, p50, p75, p90, p99) of region bbox width and height in page pixels,
# measured from the live benchmark's reply JSONs (4143 regions).
W_Q = [(0.10, 181), (0.25, 360), (0.50, 978), (0.75, 1298), (0.90, 1790), (0.99, 2382)]
H_Q = [(0.10, 44), (0.25, 56), (0.50, 113), (0.75, 227), (0.90, 419), (0.99, 1096)]

WORDS = (
    "the of and to in a is that for it as was with be by on not he this are or his from at "
    "which but have an had they you were their one all we can her has there been if more when "
    "will would who so no out up said what its about into than them only other time new some "
    "could these two may then do first any my now such like our over man me even most made "
    "after also did many before must through back years where much your way well down should "
    "because each just those people how too little state good very make world still own see "
    "men work long get here between both life being under never day same another know while "
    "last might us great old year off come since against go came right used take three "
    "report section figure table value method result system data analysis measure model page "
    "signal process energy policy federal agency program department office notice review"
).split()

FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
)


def quantile_sample(q, u):
    """Piecewise log-linear interpolation of the quantile table at probability `u`."""
    if u <= q[0][0]:
        return q[0][1] * (u / q[0][0]) ** 0.5 if u > 0 else q[0][1] * 0.5
    for (p0, v0), (p1, v1) in zip(q, q[1:]):
        if u <= p1:
            t = (u - p0) / (p1 - p0)
            return v0 * (v1 / v0) ** t
    return q[-1][1]


def font(size):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_png(rng, scale):
    u = rng.random()
    v = min(0.999, max(0.001, u + rng.gauss(0, 0.15)))
    width = max(60, int(quantile_sample(W_Q, v) * scale))
    height = max(40, int(quantile_sample(H_Q, u) * scale))
    image = Image.new("RGB", (width, height), (252, 251, 247))
    draw = ImageDraw.Draw(image)
    size = int(rng.choice((26, 30, 34, 38)) * scale)
    f = font(size)
    line_h = int(size * 1.2)
    y = max(2, (height - line_h * max(1, height // line_h)) // 2)
    char_w = size * 0.5
    while y + size <= height:
        n = max(1, int((width - 16) / char_w / 6.0))
        words = []
        while len(" ".join(words)) < n * 6 and len(words) < 40:
            w = rng.choice(WORDS)
            if rng.random() < 0.08:
                w = w.capitalize()
            if rng.random() < 0.06:
                w += rng.choice((",", ".", ";"))
            words.append(w)
        draw.text((8, y), " ".join(words), fill=(20, 20, 24), font=f)
        y += line_h
    buf = io.BytesIO()
    image.save(buf, "PNG", compress_level=1)
    return buf.getvalue(), width, height


def image_tokens(w, h):
    """Approximate image tokens after the model's smart resize (28 px per token side)."""
    return max(1, round(w / 28)) * max(1, round(h / 28))


def post(url, body, timeout):
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("requests", type=int)
    ap.add_argument("concurrency", type=int)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache", default=None, help="directory holding generated PNGs")
    ap.add_argument("--scale", type=float, default=1.0, help="crop size multiplier")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--out", default=None, help="write per-request usage JSON lines here")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument(
        "--drain-metrics",
        default=None,
        help="write URL/metrics here when the last request is dispatched (the drain mark)",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    images = []
    est_tokens = []
    for i in range(args.requests):
        key = None
        if args.cache:
            os.makedirs(args.cache, exist_ok=True)
            key = os.path.join(
                args.cache,
                hashlib.sha1(f"{args.seed}:{args.scale}:{i}".encode()).hexdigest()[:16] + ".png",
            )
        if key and os.path.exists(key):
            with open(key, "rb") as fh:
                png = fh.read()
            w, h = Image.open(io.BytesIO(png)).size
            # keep the generator's state in step with a fresh run
            make_png(rng, args.scale)
        else:
            png, w, h = make_png(rng, args.scale)
            if key:
                with open(key, "wb") as fh:
                    fh.write(png)
        images.append(base64.b64encode(png).decode())
        est_tokens.append(image_tokens(w, h))

    lock = threading.Lock()
    state = {"next": 0, "errors": 0, "rows": [], "mark": None}
    markers = []

    def scrape_drain_mark():
        state["mark"] = time.time() - started
        try:
            with urllib.request.urlopen(args.url + "/metrics", timeout=60) as r:
                text = r.read()
            with open(args.drain_metrics, "wb") as fh:
                fh.write(text)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"drain-mark scrape failed: {e}\n")

    def worker():
        while True:
            with lock:
                i = state["next"]
                if i >= args.requests:
                    return
                state["next"] += 1
                if args.drain_metrics and i == args.requests - 1:
                    marker = threading.Thread(target=scrape_drain_mark)
                    markers.append(marker)
                    marker.start()
            body = {
                "model": "glm-ocr",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + images[i]}},
                        {"type": "text", "text": "Text Recognition:"},
                    ],
                }],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "top_p": 0.00001,
                "top_k": 1,
                "repetition_penalty": 1.1,
            }
            t0 = time.time()
            try:
                out = post(args.url, body, args.timeout)
                usage = out["usage"]
                row = {
                    "i": i,
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "seconds": round(time.time() - t0, 3),
                    "finish": out["choices"][0].get("finish_reason"),
                    "text_sha1": hashlib.sha1(
                        (out["choices"][0]["message"].get("content") or "").encode()
                    ).hexdigest()[:16],
                }
            except Exception as e:  # noqa: BLE001
                with lock:
                    state["errors"] += 1
                sys.stderr.write(f"request {i} failed: {e}\n")
                continue
            with lock:
                state["rows"].append(row)

    started = time.time()
    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - started
    for marker in markers:
        marker.join()
    rows = state["rows"]
    pt = [r["prompt_tokens"] for r in rows] or [0]
    ct = [r["completion_tokens"] for r in rows] or [0]
    summary = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "errors": state["errors"],
        "wall_seconds": round(wall, 2),
        "requests_per_second": round(len(rows) / wall, 2) if wall else 0,
        "prompt_tokens": sum(pt),
        "prompt_tokens_mean": round(statistics.mean(pt), 1),
        "prompt_tokens_median": statistics.median(pt),
        "completion_tokens": sum(ct),
        "completion_tokens_mean": round(statistics.mean(ct), 1),
        "completion_tokens_median": statistics.median(ct),
        "completion_tokens_p90": sorted(ct)[int(0.9 * (len(ct) - 1))],
        "completion_tokens_per_second": round(sum(ct) / wall, 1) if wall else 0,
        "estimated_image_tokens_mean": round(statistics.mean(est_tokens), 1),
        "length_finishes": sum(1 for r in rows if r["finish"] == "length"),
    }
    if state["mark"] is not None:
        summary["drain_mark_seconds"] = round(state["mark"], 2)
    print(json.dumps(summary))
    if args.out:
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


main()
