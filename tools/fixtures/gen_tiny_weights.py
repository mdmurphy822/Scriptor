#!/usr/bin/env python3
"""Writes the tiny synthetic GLM-OCR model and its float64 goldens.

The model follows tests/fixtures/config/tiny.config.json with the real tensor
names and BF16 storage. Weights are drawn from a seeded generator and rounded
to BF16. Goldens are computed in float64 from the rounded weights by a
self-contained implementation (no numpy), independent of the Rust code.

Usage:
  python3 -I tools/fixtures/gen_tiny_weights.py          # write fixtures
  python3 -I tools/fixtures/gen_tiny_weights.py --check  # verify they reproduce
"""

import json
import math
import os
import random
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.join(ROOT, "tests", "fixtures", "config", "tiny.config.json")
OUT_DIR = os.path.join(ROOT, "tests", "fixtures", "model", "tiny")
WEIGHTS = os.path.join(OUT_DIR, "model.safetensors")
GOLDENS = os.path.join(OUT_DIR, "goldens")

SEED = 20261107
PATCH_ELEMS = 3 * 2 * 14 * 14
DECODE_STEPS = 6


# --------------------------------------------------------------------------- bf16 / safetensors


def bf16_bits(x):
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return bits & 0xFFFF


def bf16_value(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def round_bf16(x):
    return bf16_value(bf16_bits(x))


def write_safetensors(path, tensors):
    """tensors: ordered list of (name, shape, flat float list already bf16-exact)."""
    header = {}
    blobs = []
    offset = 0
    for name, shape, values in tensors:
        assert len(values) == math.prod(shape), name
        blob = b"".join(struct.pack("<H", bf16_bits(v)) for v in values)
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    text = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    text += b" " * ((8 - len(text) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for blob in blobs:
            f.write(blob)


# --------------------------------------------------------------------------- weights


class Weights:
    def __init__(self, cfg):
        self.rng = random.Random(SEED)
        self.cfg = cfg
        self.tensors = []
        self.by_name = {}

    def _add(self, name, shape, values):
        values = [round_bf16(v) for v in values]
        self.tensors.append((name, shape, values))
        self.by_name[name] = (shape, values)

    def matrix(self, name, shape, scale=1.0):
        fan_in = math.prod(shape[1:])
        std = scale / math.sqrt(fan_in)
        self._add(name, shape, [self.rng.gauss(0.0, std) for _ in range(math.prod(shape))])

    def bias(self, name, n):
        self._add(name, [n], [self.rng.gauss(0.0, 0.02) for _ in range(n)])

    def norm(self, name, n):
        self._add(name, [n], [1.0 + 0.1 * self.rng.gauss(0.0, 1.0) for _ in range(n)])

    def table(self, name, rows, dim):
        self._add(name, [rows, dim], [self.rng.gauss(0.0, 1.0) for _ in range(rows * dim)])

    def get(self, name):
        return self.by_name[name]


def build_weights(cfg):
    t, v = cfg["text_config"], cfg["vision_config"]
    w = Weights(cfg)
    vh, vi, oh = v["hidden_size"], v["intermediate_size"], v["out_hidden_size"]
    th, ti = t["hidden_size"], t["intermediate_size"]
    heads, kv, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    vocab = t["vocab_size"]
    vhd = vh // v["num_heads"]

    w.matrix("model.visual.patch_embed.proj.weight", [vh, 3, 2, 14, 14])
    w.bias("model.visual.patch_embed.proj.bias", vh)
    for b in range(v["depth"]):
        p = f"model.visual.blocks.{b}"
        w.norm(f"{p}.norm1.weight", vh)
        w.norm(f"{p}.norm2.weight", vh)
        w.matrix(f"{p}.attn.qkv.weight", [3 * vh, vh])
        w.bias(f"{p}.attn.qkv.bias", 3 * vh)
        w.norm(f"{p}.attn.q_norm.weight", vhd)
        w.norm(f"{p}.attn.k_norm.weight", vhd)
        w.matrix(f"{p}.attn.proj.weight", [vh, vh])
        w.bias(f"{p}.attn.proj.bias", vh)
        for role, shape in (("gate_proj", [vi, vh]), ("up_proj", [vi, vh]), ("down_proj", [vh, vi])):
            w.matrix(f"{p}.mlp.{role}.weight", shape)
            w.bias(f"{p}.mlp.{role}.bias", shape[0])
    w.norm("model.visual.post_layernorm.weight", vh)
    w.matrix("model.visual.downsample.weight", [oh, vh, 2, 2])
    w.bias("model.visual.downsample.bias", oh)
    w.matrix("model.visual.merger.proj.weight", [oh, oh])
    w.norm("model.visual.merger.post_projection_norm.weight", oh)
    w.bias("model.visual.merger.post_projection_norm.bias", oh)
    w.matrix("model.visual.merger.gate_proj.weight", [ti, oh])
    w.matrix("model.visual.merger.up_proj.weight", [ti, oh])
    w.matrix("model.visual.merger.down_proj.weight", [oh, ti])

    w.table("model.language_model.embed_tokens.weight", vocab, th)

    def decoder_layer(prefix):
        w.norm(f"{prefix}.input_layernorm.weight", th)
        w.norm(f"{prefix}.post_self_attn_layernorm.weight", th)
        w.norm(f"{prefix}.post_attention_layernorm.weight", th)
        w.norm(f"{prefix}.post_mlp_layernorm.weight", th)
        w.matrix(f"{prefix}.self_attn.q_proj.weight", [heads * hd, th])
        w.matrix(f"{prefix}.self_attn.k_proj.weight", [kv * hd, th])
        w.matrix(f"{prefix}.self_attn.v_proj.weight", [kv * hd, th])
        w.matrix(f"{prefix}.self_attn.o_proj.weight", [th, heads * hd])
        w.matrix(f"{prefix}.mlp.gate_up_proj.weight", [2 * ti, th])
        w.matrix(f"{prefix}.mlp.down_proj.weight", [th, ti])

    for i in range(t["num_hidden_layers"]):
        decoder_layer(f"model.language_model.layers.{i}")
    w.norm("model.language_model.norm.weight", th)
    w.matrix("lm_head.weight", [vocab, th], scale=2.0)

    mtp = f"model.language_model.layers.{t['num_hidden_layers']}"
    w.table(f"{mtp}.embed_tokens.weight", vocab, th)
    w.norm(f"{mtp}.enorm.weight", th)
    w.norm(f"{mtp}.hnorm.weight", th)
    w.matrix(f"{mtp}.eh_proj.weight", [th, 2 * th])
    decoder_layer(mtp)
    w.norm(f"{mtp}.shared_head.norm.weight", th)
    w.matrix(f"{mtp}.shared_head.head.weight", [vocab, th], scale=2.0)
    return w


# --------------------------------------------------------------------------- float64 model


def matmul_t(x, weight, bias=None):
    """x: list of rows, weight: (shape, flat) with shape [out, in]."""
    (out_dim, in_dim), flat = weight[0][:2], weight[1]
    rows = []
    for row in x:
        r = []
        for o in range(out_dim):
            base = o * in_dim
            acc = 0.0
            for i in range(in_dim):
                acc += row[i] * flat[base + i]
            if bias is not None:
                acc += bias[1][o]
            r.append(acc)
        rows.append(r)
    return rows


def rmsnorm(row, weight, eps):
    ms = sum(x * x for x in row) / len(row)
    inv = 1.0 / math.sqrt(ms + eps)
    return [x * inv * w for x, w in zip(row, weight)]


def layernorm(row, weight, bias, eps):
    n = len(row)
    mean = sum(row) / n
    var = sum((x - mean) ** 2 for x in row) / n
    inv = 1.0 / math.sqrt(var + eps)
    return [(x - mean) * inv * w + b for x, w, b in zip(row, weight, bias)]


def silu(x):
    return x / (1.0 + math.exp(-x))


def gelu(x):
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def vision_positions(t, h, w):
    one = []
    for wr in range(h // 2):
        for wc in range(w // 2):
            for dr in range(2):
                for dc in range(2):
                    one.append((wr * 2 + dr, wc * 2 + dc))
    return one * t


def rope_vision(vec_heads, pos, head_dim, theta):
    """NeoX 2-D axial: pairs (j, j + half); first half of pairs use row, rest column."""
    half = head_dim // 2
    quarter = half // 2
    out = []
    for vec in vec_heads:
        new = list(vec)
        for j in range(half):
            axis, k = (0, j) if j < quarter else (1, j - quarter)
            angle = pos[axis] * theta ** (-2.0 * k / half)
            c, s = math.cos(angle), math.sin(angle)
            a, b = vec[j], vec[j + half]
            new[j] = a * c - b * s
            new[j + half] = a * s + b * c
        out.append(new)
    return out


def rope_text(vec_heads, pos3, head_dim, theta, sections):
    """GPT-J interleaved pairs with chunked t/h/w sections."""
    bounds = []
    for comp, n in enumerate(sections):
        bounds.extend([comp] * n)
    out = []
    for vec in vec_heads:
        new = list(vec)
        for i in range(head_dim // 2):
            angle = pos3[bounds[i]] * theta ** (-2.0 * i / head_dim)
            c, s = math.cos(angle), math.sin(angle)
            a, b = vec[2 * i], vec[2 * i + 1]
            new[2 * i] = a * c - b * s
            new[2 * i + 1] = a * s + b * c
        out.append(new)
    return out


def split_heads(row, heads, dim):
    return [row[h * dim : (h + 1) * dim] for h in range(heads)]


def softmax_weighted(scores, values):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    dim = len(values[0])
    return [sum(e * v[d] for e, v in zip(exps, values)) / z for d in range(dim)]


class Model:
    def __init__(self, cfg, weights):
        self.cfg = cfg
        self.w = weights
        self.t = cfg["text_config"]
        self.v = cfg["vision_config"]

    # ---- vision
    def vision(self, patches, grids):
        v, w = self.v, self.w
        eps = v["rms_norm_eps"]
        theta = v["rope_parameters"]["rope_theta"]
        heads = v["num_heads"]
        hd = v["hidden_size"] // heads
        pe = w.get("model.visual.patch_embed.proj.weight")
        pe = ([pe[0][0], PATCH_ELEMS], pe[1])
        x = matmul_t(patches, pe, w.get("model.visual.patch_embed.proj.bias"))
        pos = []
        spans = []
        start = 0
        for t, h, wd in grids:
            pos.extend(vision_positions(t, h, wd))
            for _ in range(t):
                spans.append((start, start + h * wd))
                start += h * wd
        for b in range(v["depth"]):
            p = f"model.visual.blocks.{b}"
            normed = [rmsnorm(r, w.get(f"{p}.norm1.weight")[1], eps) for r in x]
            qkv = matmul_t(normed, w.get(f"{p}.attn.qkv.weight"), w.get(f"{p}.attn.qkv.bias"))
            hs = v["hidden_size"]
            q = [split_heads(r[:hs], heads, hd) for r in qkv]
            k = [split_heads(r[hs : 2 * hs], heads, hd) for r in qkv]
            val = [split_heads(r[2 * hs :], heads, hd) for r in qkv]
            qn = w.get(f"{p}.attn.q_norm.weight")[1]
            kn = w.get(f"{p}.attn.k_norm.weight")[1]
            q = [[rmsnorm(hv, qn, eps) for hv in tok] for tok in q]
            k = [[rmsnorm(hv, kn, eps) for hv in tok] for tok in k]
            q = [rope_vision(tok, pos[i], hd, theta) for i, tok in enumerate(q)]
            k = [rope_vision(tok, pos[i], hd, theta) for i, tok in enumerate(k)]
            attn = []
            for i in range(len(x)):
                span = next(s for s in spans if s[0] <= i < s[1])
                row = []
                for h in range(heads):
                    scores = [
                        sum(a * c for a, c in zip(q[i][h], k[j][h])) / math.sqrt(hd)
                        for j in range(*span)
                    ]
                    row.extend(softmax_weighted(scores, [val[j][h] for j in range(*span)]))
                attn.append(row)
            proj = matmul_t(attn, w.get(f"{p}.attn.proj.weight"), w.get(f"{p}.attn.proj.bias"))
            x = [[a + b for a, b in zip(r, d)] for r, d in zip(x, proj)]
            normed = [rmsnorm(r, w.get(f"{p}.norm2.weight")[1], eps) for r in x]
            gate = matmul_t(normed, w.get(f"{p}.mlp.gate_proj.weight"), w.get(f"{p}.mlp.gate_proj.bias"))
            up = matmul_t(normed, w.get(f"{p}.mlp.up_proj.weight"), w.get(f"{p}.mlp.up_proj.bias"))
            act = [[silu(g) * u for g, u in zip(gr, ur)] for gr, ur in zip(gate, up)]
            down = matmul_t(act, w.get(f"{p}.mlp.down_proj.weight"), w.get(f"{p}.mlp.down_proj.bias"))
            x = [[a + b for a, b in zip(r, d)] for r, d in zip(x, down)]
        x = [rmsnorm(r, w.get("model.visual.post_layernorm.weight")[1], eps) for r in x]
        # 2x2 stride-2 conv over merge windows: view [G, 2, 2, C] -> [G, C, 2, 2].
        ds_shape, ds = w.get("model.visual.downsample.weight")
        out_dim, in_dim = ds_shape[0], ds_shape[1]
        ds_bias = w.get("model.visual.downsample.bias")[1]
        merged = []
        for g in range(len(x) // 4):
            row = []
            for o in range(out_dim):
                acc = ds_bias[o]
                for c in range(in_dim):
                    for dy in range(2):
                        for dx in range(2):
                            acc += x[g * 4 + dy * 2 + dx][c] * ds[((o * in_dim + c) * 2 + dy) * 2 + dx]
                row.append(acc)
            merged.append(row)
        m = "model.visual.merger"
        x = matmul_t(merged, w.get(f"{m}.proj.weight"))
        x = [
            [
                gelu(a)
                for a in layernorm(
                    r,
                    w.get(f"{m}.post_projection_norm.weight")[1],
                    w.get(f"{m}.post_projection_norm.bias")[1],
                    1e-5,
                )
            ]
            for r in x
        ]
        gate = matmul_t(x, w.get(f"{m}.gate_proj.weight"))
        up = matmul_t(x, w.get(f"{m}.up_proj.weight"))
        act = [[silu(g) * u for g, u in zip(gr, ur)] for gr, ur in zip(gate, up)]
        return matmul_t(act, w.get(f"{m}.down_proj.weight"))

    # ---- text
    def layer(self, prefix, h, positions):
        """One sandwich-norm decoder layer over a full sequence (no cache)."""
        t, w = self.t, self.w
        eps = t["rms_norm_eps"]
        heads, kvh, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
        theta = t["rope_parameters"]["rope_theta"]
        sections = t["rope_parameters"]["mrope_section"]
        normed = [rmsnorm(r, w.get(f"{prefix}.input_layernorm.weight")[1], eps) for r in h]
        q = matmul_t(normed, w.get(f"{prefix}.self_attn.q_proj.weight"))
        k = matmul_t(normed, w.get(f"{prefix}.self_attn.k_proj.weight"))
        v = matmul_t(normed, w.get(f"{prefix}.self_attn.v_proj.weight"))
        q = [rope_text(split_heads(r, heads, hd), positions[i], hd, theta, sections) for i, r in enumerate(q)]
        k = [rope_text(split_heads(r, kvh, hd), positions[i], hd, theta, sections) for i, r in enumerate(k)]
        v = [split_heads(r, kvh, hd) for r in v]
        group = heads // kvh
        attn = []
        for i in range(len(h)):
            row = []
            for hh in range(heads):
                kv = hh // group
                scores = [
                    sum(a * c for a, c in zip(q[i][hh], k[j][kv])) / math.sqrt(hd)
                    for j in range(i + 1)
                ]
                row.extend(softmax_weighted(scores, [v[j][kv] for j in range(i + 1)]))
            attn.append(row)
        o = matmul_t(attn, w.get(f"{prefix}.self_attn.o_proj.weight"))
        o = [rmsnorm(r, w.get(f"{prefix}.post_self_attn_layernorm.weight")[1], eps) for r in o]
        h = [[a + b for a, b in zip(r, d)] for r, d in zip(h, o)]
        normed = [rmsnorm(r, w.get(f"{prefix}.post_attention_layernorm.weight")[1], eps) for r in h]
        gu = matmul_t(normed, w.get(f"{prefix}.mlp.gate_up_proj.weight"))
        inter = t["intermediate_size"]
        act = [[silu(r[i]) * r[inter + i] for i in range(inter)] for r in gu]
        down = matmul_t(act, w.get(f"{prefix}.mlp.down_proj.weight"))
        down = [rmsnorm(r, w.get(f"{prefix}.post_mlp_layernorm.weight")[1], eps) for r in down]
        return [[a + b for a, b in zip(r, d)] for r, d in zip(h, down)]

    def text(self, embeds, positions):
        """Returns the post-final-norm hidden states [n, hidden]."""
        t, w = self.t, self.w
        h = embeds
        for i in range(t["num_hidden_layers"]):
            h = self.layer(f"model.language_model.layers.{i}", h, positions)
        return [rmsnorm(r, w.get("model.language_model.norm.weight")[1], t["rms_norm_eps"]) for r in h]

    def logits(self, hidden):
        return matmul_t(hidden, self.w.get("lm_head.weight"))

    def embed(self, ids, table="model.language_model.embed_tokens.weight"):
        shape, flat = self.w.get(table)
        dim = shape[1]
        return [flat[i * dim : (i + 1) * dim] for i in ids]

    def mtp(self, next_ids, hidden, positions):
        t, w = self.t, self.w
        eps = t["rms_norm_eps"]
        mtp = f"model.language_model.layers.{t['num_hidden_layers']}"
        emb = self.embed(next_ids, f"{mtp}.embed_tokens.weight")
        emb = [[0.0] * len(r) if positions[i][0] == 0 else r for i, r in enumerate(emb)]
        emb = [rmsnorm(r, w.get(f"{mtp}.enorm.weight")[1], eps) for r in emb]
        hid = [rmsnorm(r, w.get(f"{mtp}.hnorm.weight")[1], eps) for r in hidden]
        cat = [e + hh for e, hh in zip(emb, hid)]
        x = matmul_t(cat, w.get(f"{mtp}.eh_proj.weight"))
        x = self.layer(mtp, x, positions)
        normed = [rmsnorm(r, w.get(f"{mtp}.shared_head.norm.weight")[1], eps) for r in x]
        return x, matmul_t(normed, w.get(f"{mtp}.shared_head.head.weight"))


# --------------------------------------------------------------------------- cases


def mrope_positions(ids, image_token, grids, merge=2):
    """Independent M-RoPE position builder: returns (positions, delta)."""
    positions = []
    cursor = 0
    gi = 0
    i = 0
    while i < len(ids):
        if ids[i] == image_token:
            t, h, w = grids[gi]
            gi += 1
            lt, lh, lw = t, h // merge, w // merge
            n = lt * lh * lw
            assert all(x == image_token for x in ids[i : i + n])
            for ti in range(lt):
                for hi in range(lh):
                    for wi in range(lw):
                        positions.append((cursor + ti, cursor + hi, cursor + wi))
            cursor += max(lt, lh, lw)
            i += n
        else:
            positions.append((cursor, cursor, cursor))
            cursor += 1
            i += 1
    return positions, cursor - len(ids)


def penalise(logits, seen, penalty):
    out = list(logits)
    for tok in seen:
        out[tok] = out[tok] / penalty if out[tok] > 0 else out[tok] * penalty
    return out


def argmax_margin(logits):
    order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    return order[0], logits[order[0]] - logits[order[1]]


CASES = [
    # name, grids (t,h,w), text segments around each image, repetition penalty
    ("one_token_image", [(1, 2, 2)], [[11, 12, 13], [14, 15]], 1.0),
    ("two_token_image", [(1, 2, 4)], [[21, 22], [23, 24, 25, 26]], 1.1),
    ("four_token_image", [(1, 4, 4)], [[31, 32, 33, 34], [35]], 1.0),
    ("two_images", [(1, 2, 2), (1, 2, 6)], [[41, 42], [43], [44, 45]], 1.1),
]


def build_case(model, patch_rng, name, grids, segments, penalty):
    cfg = model.cfg
    image_start, image_end, image_token = (
        cfg["image_start_token_id"],
        cfg["image_end_token_id"],
        cfg["image_token_id"],
    )
    eos = set(cfg["text_config"]["eos_token_id"])
    ids = list(segments[0])
    for gi, (t, h, w) in enumerate(grids):
        ids.append(image_start)
        ids.extend([image_token] * (t * (h // 2) * (w // 2)))
        ids.append(image_end)
        ids.extend(segments[gi + 1])
    patches = []
    for t, h, w in grids:
        for _ in range(t * h * w):
            patches.append([round_bf16(patch_rng.gauss(0.0, 1.0)) for _ in range(PATCH_ELEMS)])
    image_embeds = model.vision(patches, grids)
    positions, delta = mrope_positions(ids, image_token, grids)

    embeds = model.embed(ids)
    slot = 0
    for i, tok in enumerate(ids):
        if tok == image_token:
            embeds[i] = image_embeds[slot]
            slot += 1
    assert slot == len(image_embeds)
    hidden = model.text(embeds, positions)
    logits = model.logits(hidden)

    # Greedy decode by recomputing the full sequence each step.
    tokens, margins, step_logits = [], [], []
    seq_ids = list(ids)
    seq_embeds = [list(r) for r in embeds]
    seq_pos = list(positions)
    for step in range(DECODE_STEPS):
        h = model.text(seq_embeds, seq_pos)
        lg = model.logits(h[-1:])[0]
        step_logits.append(lg)
        seen = set(seq_ids)
        pen = penalise(lg, seen, penalty) if penalty != 1.0 else lg
        tok, margin = argmax_margin(pen)
        assert margin > 1e-3, (name, step, margin)
        tokens.append(tok)
        margins.append(margin)
        if tok in eos:
            break
        seq_ids.append(tok)
        seq_embeds.append(model.embed([tok])[0])
        p = len(seq_ids) - 1 + delta
        seq_pos.append((p, p, p))

    mtp_hidden, mtp_logits = model.mtp(ids[1:], hidden[:-1], positions[:-1])
    return {
        "name": name,
        "input_ids": ids,
        "grid_thw": [list(g) for g in grids],
        "patches": [v for row in patches for v in row],
        "image_embeds": [v for row in image_embeds for v in row],
        "positions": [list(p) for p in positions],
        "mrope_delta": delta,
        "hidden": [v for row in hidden for v in row],
        "prefill_logits": [v for row in logits for v in row],
        "repetition_penalty": penalty,
        "greedy_tokens": tokens,
        "greedy_margins": margins,
        "decode_logits": [v for row in step_logits for v in row],
        "mtp_hidden": [v for row in mtp_hidden for v in row],
        "mtp_logits": [v for row in mtp_logits for v in row],
    }


def generate():
    with open(CONFIG) as f:
        cfg = json.load(f)
    weights = build_weights(cfg)
    model = Model(cfg, weights)
    patch_rng = random.Random(SEED + 1)
    cases = [build_case(model, patch_rng, *c) for c in CASES]
    return weights, cases


def render_case(case):
    return json.dumps(case, separators=(",", ":")) + "\n"


def main():
    check = "--check" in sys.argv[1:]
    weights, cases = generate()
    tmp = WEIGHTS + ".check"
    target = tmp if check else WEIGHTS
    os.makedirs(GOLDENS, exist_ok=True)
    write_safetensors(target, weights.tensors)
    ok = True
    if check:
        ok &= open(tmp, "rb").read() == open(WEIGHTS, "rb").read()
        os.remove(tmp)
    for case in cases:
        path = os.path.join(GOLDENS, case["name"] + ".json")
        text = render_case(case)
        if check:
            ok &= os.path.exists(path) and open(path).read() == text
        else:
            with open(path, "w") as f:
                f.write(text)
    if check:
        print("tiny fixtures reproduce" if ok else "tiny fixtures DIFFER")
        sys.exit(0 if ok else 1)
    print(f"wrote {WEIGHTS} and {len(cases)} goldens")


if __name__ == "__main__":
    main()
