#!/usr/bin/env python3
"""Renders the GLM-OCR chat template with Python jinja2 into golden files.

Usage: render_goldens.py [--check] [output_dir]
Writes (or with --check verifies) goldens/*.json and tiny-tokenizer.json under
tests/fixtures/chat. Each golden holds the template inputs and the exact rendered text.
The tiny tokenizer reuses the pre-tokenizer, post-processor and decoder classes of the
real tokenizer.json over a byte-level vocabulary with a handful of merges.
"""
import json
import os
import sys

from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = os.path.normpath(os.path.join(HERE, "..", "..", "tests", "fixtures", "chat"))

IMG = {"type": "image"}


def text(t):
    return {"type": "text", "text": t}


def user(*parts):
    return {"role": "user", "content": list(parts)}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Looks up a term",
            "parameters": {
                "type": "object",
                "properties": {"term": {"type": "string"}},
                "required": ["term"],
            },
        },
    }
]

CASES = [
    ("semantik_text", [user(IMG, text("Text Recognition:"))], {"add_generation_prompt": True}),
    ("semantik_table", [user(IMG, text("Table Recognition:"))], {"add_generation_prompt": True}),
    ("semantik_formula", [user(IMG, text("Formula Recognition:"))], {"add_generation_prompt": True}),
    (
        "semantik_text_nothink",
        [user(IMG, text("Text Recognition:"))],
        {"add_generation_prompt": True, "enable_thinking": False},
    ),
    ("string_content", [{"role": "user", "content": "Hello there"}], {"add_generation_prompt": True}),
    (
        "string_content_nothink",
        [{"role": "user", "content": "Hello there"}],
        {"add_generation_prompt": True, "enable_thinking": False},
    ),
    (
        "system_user",
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Summarise."},
        ],
        {"add_generation_prompt": True},
    ),
    (
        "assistant_history",
        [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Answer one"},
            {"role": "user", "content": "Second"},
        ],
        {"add_generation_prompt": True},
    ),
    (
        "assistant_reasoning",
        [
            {"role": "user", "content": "Think"},
            {
                "role": "assistant",
                "content": "<think>\nstep one\n</think>\nfinal words",
            },
            {"role": "user", "content": "More"},
            {"role": "assistant", "content": "ok", "reasoning_content": " kept reasoning "},
        ],
        {"add_generation_prompt": False},
    ),
    (
        "tools",
        [{"role": "user", "content": "Find it"}],
        {"add_generation_prompt": True, "tools": TOOLS},
    ),
    (
        "tool_round_trip",
        [
            {"role": "user", "content": "Find it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": {"term": "café", "limit": 3},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "result one"},
            {"role": "tool", "content": "result two"},
        ],
        {"add_generation_prompt": True, "tools": TOOLS},
    ),
    (
        "unicode_text_then_image",
        [user(text("Café 日本語 \U0001f600"), IMG, text("Recognise"))],
        {"add_generation_prompt": True},
    ),
]


def environment():
    def tojson(value, ensure_ascii=True, indent=None, separators=None, sort_keys=False):
        return json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    def raise_exception(message):
        raise TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def render_all():
    with open(os.path.join(CHAT, "chat_template.jinja"), encoding="utf-8") as f:
        template = environment().from_string(f.read())
    out = {}
    for name, messages, kwargs in CASES:
        rendered = template.render(messages=messages, **kwargs)
        out[f"goldens/{name}.json"] = dump(
            {"name": name, "messages": messages, "kwargs": kwargs, "rendered": rendered}
        )
    out["tiny-tokenizer.json"] = dump(tiny_tokenizer())
    return out


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=1) + "\n"


def byte_alphabet():
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = {b: chr(b) for b in keep}
    extra = 0
    for b in range(256):
        if b not in chars:
            chars[b] = chr(256 + extra)
            extra += 1
    return [chars[b] for b in range(256)]


SPECIALS = [
    "<|endoftext|>", "[MASK]", "[gMASK]", "[sMASK]", "<sop>", "<eop>", "<|system|>",
    "<|user|>", "<|assistant|>", "<|observation|>", "<|begin_of_image|>",
    "<|end_of_image|>", "<|begin_of_video|>", "<|end_of_video|>", "<|image|>",
    "<|video|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
    "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<tool_response>",
    "</tool_response>", "/nothink",
]


def tiny_tokenizer():
    alphabet = byte_alphabet()
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    merges = []
    for pair in [("Ġ", "t"), ("h", "e"), ("Ġt", "he"), ("e", "r"), ("i", "n")]:
        merges.append(f"{pair[0]} {pair[1]}")
        vocab[pair[0] + pair[1]] = len(vocab)
    added = [
        {
            "id": len(vocab) + i,
            "content": content,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for i, content in enumerate(SPECIALS)
    ]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {
                        "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    },
                    "behavior": "Isolated",
                    "invert": False,
                },
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
            ],
        },
        "post_processor": {
            "type": "Sequence",
            "processors": [
                {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True}
            ],
        },
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": vocab,
            "merges": merges,
        },
    }


def main(argv):
    check = "--check" in argv
    rest = [a for a in argv if a != "--check"]
    root = rest[0] if rest else CHAT
    files = render_all()
    status = 0
    for rel, content in files.items():
        path = os.path.join(root, rel)
        if check:
            try:
                with open(path, encoding="utf-8") as f:
                    same = f.read() == content
            except FileNotFoundError:
                same = False
            if not same:
                print(f"MISMATCH {rel}")
                status = 1
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
    if check and status == 0:
        print(f"{len(files)} goldens reproduced")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
