"""Synthetic OCR load: `vision_load.py URL REQUESTS CONCURRENCY [SEED]`.

Sends REQUESTS chat completions, CONCURRENCY at a time, each carrying one generated PNG of
rendered pseudo-text whose size varies per request, and prints wall time, completion tokens
and tokens per second as one JSON line. No real documents are used.
"""
import base64
import io
import json
import random
import sys
import threading
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

WORDS = "alpha beta gamma delta sigma omega vector matrix kernel tensor graph stream event batch".split()
FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


def font(size):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_png(rng):
    width = rng.choice((224, 336, 448, 560, 672, 896))
    height = rng.choice((112, 168, 224, 336, 448))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    f = font(rng.choice((14, 18, 22)))
    y = 6
    while y < height - 24:
        line = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 9)))
        draw.text((8, y), line, fill="black", font=f)
        y += f.size + 6
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def post(url, body):
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def main():
    url, total, conc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 7)
    images = [make_png(rng) for _ in range(total)]
    lock = threading.Lock()
    state = {"next": 0, "tokens": 0, "errors": 0}

    def worker():
        while True:
            with lock:
                i = state["next"]
                if i >= total:
                    return
                state["next"] += 1
            body = {
                "model": "glm-ocr",
                "temperature": 0,
                "max_tokens": 256,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + images[i]}},
                        {"type": "text", "text": "Text Recognition:"},
                    ],
                }],
            }
            try:
                out = post(url, body)
                n = out["usage"]["completion_tokens"]
            except Exception:
                with lock:
                    state["errors"] += 1
                continue
            with lock:
                state["tokens"] += n

    started = time.time()
    threads = [threading.Thread(target=worker) for _ in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - started
    print(json.dumps({
        "requests": total,
        "concurrency": conc,
        "errors": state["errors"],
        "wall_seconds": round(wall, 2),
        "completion_tokens": state["tokens"],
        "tokens_per_second": round(state["tokens"] / wall, 1) if wall else 0,
    }))


main()
