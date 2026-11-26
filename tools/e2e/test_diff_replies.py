"""Synthetic checks of diff_replies.py --strict and replay_engine.py. Run with
`python3 -I tools/e2e/test_diff_replies.py`."""
import base64
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


dr = load("diff_replies")
re_ = load("replay_engine")


def region(i, text="t", bbox=(0, 0, 10, 10), label="text", page=1):
    return {"id": f"r{i}", "page": page, "label": label, "bbox": list(bbox), "text": text}


def reply(*regions, **top):
    return {"regions": list(regions), **top}


def run(a, b, *flags):
    """Runs diff_replies over one document per side; returns (exit code, stdout)."""
    with tempfile.TemporaryDirectory() as d:
        for tag, rep in (("a", a), ("b", b)):
            with open(os.path.join(d, f"{tag}-doc.reply.json"), "w") as f:
                json.dump(rep, f)
        sys.argv = ["diff_replies.py", d, "a", "b", *flags]
        out = io.StringIO()
        code = 0
        with redirect_stdout(out):
            try:
                dr.main()
            except SystemExit as e:
                code = e.code
        return code, out.getvalue()


class Strict(unittest.TestCase):
    def test_identical_passes(self):
        r = reply(region(0), region(1, "x"))
        self.assertEqual(run(r, json.loads(json.dumps(r)), "--strict")[0], 0)

    def test_text_equal_bbox_differs(self):
        a = reply(region(0), region(1, bbox=(1, 2, 3, 4)))
        b = reply(region(0), region(1, bbox=(1, 2, 3, 5)))
        self.assertEqual(run(a, b)[0], 0)
        code, out = run(a, b, "--strict")
        self.assertEqual(code, 1)
        self.assertIn("DIFF [1] page=1 id=r1 fields: bbox", out)
        self.assertNotIn("DIFF [0]", out)

    def test_label_and_text_named(self):
        a = reply(region(0, "a", label="text"))
        b = reply(region(0, "b", label="title"))
        code, out = run(a, b, "--strict")
        self.assertEqual(code, 1)
        self.assertIn("fields: label, text", out)

    def test_order_matters(self):
        a = reply(region(0), region(1, "x"))
        b = reply(region(1, "x"), region(0))
        self.assertEqual(run(a, b)[0], 1)
        code, out = run(a, b, "--strict")
        self.assertEqual(code, 1)
        self.assertIn("DIFF [0]", out)
        self.assertIn("DIFF [1]", out)

    def test_key_order_ignored(self):
        a = reply({"id": "r0", "text": "t", "page": 1})
        b = reply({"page": 1, "text": "t", "id": "r0"})
        self.assertEqual(run(a, b, "--strict")[0], 0)

    def test_missing_region_named(self):
        a = reply(region(0), region(1))
        b = reply(region(0))
        code, out = run(a, b, "--strict")
        self.assertEqual(code, 1)
        self.assertIn("DIFF [1] page=1 id=r1 fields: <absent>", out)

    def test_top_level_field(self):
        a = reply(region(0), empty_pages=[2])
        b = reply(region(0), empty_pages=[3])
        self.assertEqual(run(a, b)[0], 0)
        code, out = run(a, b, "--strict")
        self.assertEqual(code, 1)
        self.assertIn("top-level fields differ: empty_pages", out)


class Replay(unittest.TestCase):
    def setUp(self):
        self.img = b"synthetic-image-bytes"
        key = hashlib.sha256(self.img).hexdigest()
        handler = re_.make_handler({key: "hello"}, "glm-ocr", None)
        self.srv = re_.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def post(self, img, stream=False):
        url = "data:image/png;base64," + base64.b64encode(img).decode()
        body = {"model": "glm-ocr", "stream": stream, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "OCR"}, {"type": "image_url", "image_url": {"url": url}}]}]}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=10)

    def test_recorded_reply(self):
        for _ in range(2):
            data = json.load(self.post(self.img))
            self.assertEqual(data["choices"][0]["message"]["content"], "hello")

    def test_stream(self):
        self.assertIn(b'"hello"', self.post(self.img, True).read())

    def test_unrecorded_fails_closed(self):
        with self.assertRaises(urllib.error.HTTPError) as c:
            self.post(b"other")
        self.assertEqual(c.exception.code, 404)
        self.assertIn(hashlib.sha256(b"other").hexdigest().encode(), c.exception.read())

    def test_health(self):
        self.assertEqual(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=10).status, 200)


if __name__ == "__main__":
    unittest.main()
