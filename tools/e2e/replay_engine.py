"""Deterministic stand-in for an OpenAI-compatible OCR engine.

usage: python3 -I replay_engine.py --replies FILE [--port N] [--host ADDR]
                                   [--model NAME] [--seen-log FILE]

FILE is JSON: {"replies": {"<sha256 hex>": "<reply text>", ...}}. A request to
POST /v1/chat/completions is keyed by the SHA-256 of the first image in the
messages: the decoded bytes of a base64 data URL, or the UTF-8 bytes of any
other image URL. The matching reply text is returned as a chat completion
(or as server-sent events when the request sets "stream"). A request with no
image or with an unrecorded key gets HTTP 404 with a JSON error naming the key,
so a replay never invents text. GET /health returns 200; GET /metrics returns
an empty body. --seen-log appends each requested key, one per line. Binds
127.0.0.1 unless --host is given. Intended for synthetic fixtures only.
"""
import argparse
import base64
import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def image_key(body):
    """SHA-256 hex of the first image in a chat-completions request body, or None."""
    for msg in body.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = part.get("image_url", {})
            url = url.get("url") if isinstance(url, dict) else url
            if not isinstance(url, str):
                continue
            if url.startswith("data:") and ";base64," in url:
                raw = base64.b64decode(url.split(";base64,", 1)[1])
            else:
                raw = url.encode()
            return hashlib.sha256(raw).hexdigest()
    return None


def make_handler(replies, model, seen_log):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def send(self, code, ctype, data):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, code, obj):
            self.send(code, "application/json", json.dumps(obj).encode())

        def do_GET(self):
            if self.path == "/health":
                self.send(200, "text/plain", b"ok")
            elif self.path == "/metrics":
                self.send(200, "text/plain", b"")
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(n))
            except ValueError:
                self.send_json(400, {"error": "invalid json"})
                return
            key = image_key(body)
            if key is None:
                self.send_json(404, {"error": "no image in request"})
                return
            if seen_log:
                with open(seen_log, "a") as f:
                    f.write(key + "\n")
            text = replies.get(key)
            if text is None:
                self.send_json(404, {"error": f"no recorded reply for image sha256 {key}"})
                return
            if body.get("stream"):
                chunk = {"object": "chat.completion.chunk", "model": model,
                         "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                end = {"object": "chat.completion.chunk", "model": model,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                data = f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode()
                self.send(200, "text/event-stream", data)
                return
            self.send_json(200, {
                "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replies", required=True)
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="glm-ocr")
    ap.add_argument("--seen-log")
    a = ap.parse_args()
    with open(a.replies) as f:
        replies = json.load(f)["replies"]
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(replies, a.model, a.seen_log))
    print(f"replay_engine: ready port={a.port} replies={len(replies)}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
