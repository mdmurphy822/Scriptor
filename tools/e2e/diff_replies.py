"""Region-text identity between two sets of bridge replies.

usage: python3 -I diff_replies.py DIR_A TAG_A TAG_B [--dir-b DIR_B] [--show] [--strict]

Compares DIR_A/TAG_A-<class>.reply.json with DIR_B/TAG_B-<class>.reply.json
(DIR_B defaults to DIR_A) region by region in order. Prints per-document region
counts and identical counts, then a TOTAL line. A region at an index missing on
one side counts as differing. --show lists each differing region with its
index, page, id, and both texts. Exit status is 0 when every region is identical
and region counts match, 1 otherwise, 2 on usage errors.

--strict compares each whole region (every field: text, bbox, label, page, id,
and any other key) and the region order, plus every non-region top-level reply
field. Objects are compared as canonical JSON (keys sorted, no insignificant
whitespace), so only values and array order matter. Every differing region is
named with its index, page, id and the fields that differ, and the first
differing top-level field is named by key; the exit status is 1 on any
difference.
"""
import glob
import json
import os
import sys


def parse(argv):
    pos, dir_b, show, strict = [], None, False, False
    it = iter(argv)
    for a in it:
        if a == "--show":
            show = True
        elif a == "--strict":
            strict = True
        elif a == "--dir-b":
            dir_b = next(it, None)
        elif a in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        else:
            pos.append(a)
    if len(pos) != 3 or (dir_b is None and "--dir-b" in argv):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    return pos[0], pos[1], pos[2], dir_b or pos[0], show, strict


def load(path):
    with open(path) as f:
        return json.load(f)


def regions(path):
    return load(path)["regions"]


def canon(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def field_diff(a, b):
    """Names of the keys whose values differ between two region objects."""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return ["<region>"]
    return sorted(k for k in set(a) | set(b) if canon(a.get(k, "<absent>")) != canon(b.get(k, "<absent>")))


def top_diff(da, db):
    """Names of the non-region top-level reply keys that differ."""
    ka = {k: v for k, v in da.items() if k != "regions"}
    kb = {k: v for k, v in db.items() if k != "regions"}
    return sorted(k for k in set(ka) | set(kb) if canon(ka.get(k, "<absent>")) != canon(kb.get(k, "<absent>")))


def where(r):
    return f"page={r.get('page', '?')} id={r.get('id', '?')}"


def main():
    dir_a, tag_a, tag_b, dir_b, show, strict = parse(sys.argv[1:])
    total = same = count_a = count_b = 0
    clean = True
    for fa in sorted(glob.glob(os.path.join(dir_a, f"{tag_a}-*.reply.json"))):
        cls = os.path.basename(fa)[len(tag_a) + 1:-len(".reply.json")]
        fb = os.path.join(dir_b, f"{tag_b}-{cls}.reply.json")
        if not os.path.exists(fb):
            print(f"missing {fb}")
            clean = False
            continue
        try:
            da, db = load(fa), load(fb)
            ra, rb = da["regions"], db["regions"]
        except (OSError, ValueError, KeyError) as e:
            print(f"bad {cls}: {e}")
            clean = False
            continue
        count_a += len(ra)
        count_b += len(rb)
        n = max(len(ra), len(rb))
        diffs = []
        for i in range(n):
            a = ra[i] if i < len(ra) else None
            b = rb[i] if i < len(rb) else None
            if strict:
                differs = a is None or b is None or canon(a) != canon(b)
            else:
                differs = a is None or b is None or a.get("text", "") != b.get("text", "")
            if differs:
                diffs.append((i, a, b))
        total += n
        same += n - len(diffs)
        if diffs or len(ra) != len(rb):
            clean = False
        tops = top_diff(da, db) if strict else []
        if tops:
            clean = False
        print(f"{cls:22s} regions {len(ra)}/{len(rb)} identical {n - len(diffs)}/{n}")
        if tops:
            print(f"  top-level fields differ: {', '.join(tops)}")
        if strict:
            for i, a, b in diffs:
                loc = where(a) if a else where(b)
                names = ["<absent>"] if a is None or b is None else field_diff(a, b)
                print(f"  DIFF [{i}] {loc} fields: {', '.join(names)}")
        if show:
            for i, a, b in diffs:
                loc = where(a) if a else where(b)
                print(f"  [{i}] {loc}")
                if strict:
                    print(f"    A: {canon(a) if a is not None else '<absent>'}")
                    print(f"    B: {canon(b) if b is not None else '<absent>'}")
                else:
                    print(f"    A: {a.get('text', '') if a else '<absent>'!r}")
                    print(f"    B: {b.get('text', '') if b else '<absent>'!r}")
    pct = 100 * same / total if total else 0.0
    print(f"TOTAL regions {count_a}/{count_b} identical {same}/{total} = {pct:.1f}%")
    sys.exit(0 if clean and total else 1)


if __name__ == "__main__":
    main()
