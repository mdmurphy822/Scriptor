"""Summarises a /metrics before/after pair:
usage `metrics_summary.py PRE POST GPU_LOG WALL_SECONDS [PHASE]`.

Reads vLLM-compatible counters (vllm:*) and engine counters (scriptor_*), prints one
line of deltas: token counts, step time, spec-decode acceptance, mean GPU
utilisation, the decode-step slot occupancy when present (tagged `phase=PHASE` when
given), and the scriptor_* counters that are present. A run scraped at its drain mark
(`realistic_load.py --drain-metrics`) summarises as two pairs: PRE..MARK `steady` and
MARK..POST `tail`.
"""
import re
import sys


def load(path):
    out = {}
    try:
        with open(path) as fh:
            lines = fh.read().splitlines()
    except OSError:
        return out
    for line in lines:
        if line.startswith("#"):
            continue
        m = re.match(r"^((?:vllm:|scriptor_)[A-Za-z0-9_:]+)(\{[^}]*\})?\s+(\S+)$", line.strip())
        if m and m.group(1).endswith(("_total", "_sum", "_count")):
            try:
                out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
            except ValueError:
                pass
    return out


ESTIMATE_NOTE = "est_*=bucket-mean estimate; ignores per-bucket variation and compaction overhead"


def counter_delta(pre, post, name):
    """Summed post-minus-pre delta of every series of counter `name`."""
    return sum(v - pre.get(k, 0) for k, v in post.items() if k.split("{")[0] == name)


def per_bucket(pre, post, name):
    """Post-minus-pre delta of counter `name` keyed by its integer `bucket` label."""
    out = {}
    for k, v in post.items():
        if k.split("{")[0] == name:
            m = re.search(r'bucket="(\d+)"', k)
            if m:
                out[int(m.group(1))] = v - pre.get(k, 0)
    return out


def slot_occupancy(pre, post, phase=None):
    """One line of decode-step slot occupancy over a scrape pair, or "" without occupancy
    counters. Row shares are exact counts. `timed_device_s` is the summed device time of
    the timed graph replays and `timed_coverage` their fraction of all steps. The `est_*`
    fields scale each bucket's mean timed step by its step count, as run and as a gap-free
    slot layout would have run it; they are `n/a` when a bucket they need has no timing."""
    rows = counter_delta(pre, post, "scriptor_decode_bucket_rows_total")
    if not rows:
        return ""
    steps = counter_delta(pre, post, "scriptor_decode_step_shapes_total")
    active = counter_delta(pre, post, "scriptor_decode_active_rows_total")
    padded = counter_delta(pre, post, "scriptor_decode_padded_rows_total")
    sparse = counter_delta(pre, post, "scriptor_decode_sparse_rows_total")
    line = "occupancy" + (f" phase={phase}" if phase else "")
    line += (
        f" steps={steps:.0f}"
        f" active_per_step={active / steps if steps else 0:.2f}"
        f" bucket_per_step={rows / steps if steps else 0:.2f}"
        f" padded_share={padded / rows:.3f} sparse_share={sparse / rows:.3f}"
        f" sparse_steps={counter_delta(pre, post, 'scriptor_decode_sparse_steps_total'):.0f}"
    )
    ran = per_bucket(pre, post, "scriptor_decode_bucket_steps_total")
    compact = per_bucket(pre, post, "scriptor_decode_compact_bucket_steps_total")
    timed = per_bucket(pre, post, "scriptor_decode_bucket_timed_steps_total")
    secs = per_bucket(pre, post, "scriptor_decode_bucket_device_seconds_total")
    timed_steps = sum(timed.values())
    if not timed_steps:
        return line + " timed_steps=0"
    line += (
        f" timed_steps={timed_steps:.0f} timed_coverage={timed_steps / steps if steps else 0:.3f}"
        f" timed_device_s={sum(secs.values()):.3f}"
    )
    mean = {b: secs.get(b, 0) / n for b, n in timed.items() if n}
    line += " bucket_mean_ms=" + ",".join(f"{b}:{mean[b] * 1000:.2f}" for b in sorted(mean))
    needed = {b for b, n in ran.items() if n} | {b for b, n in compact.items() if n}
    untimed = sorted(b for b in needed if b not in mean)
    if untimed:
        missing = ",".join(str(b) for b in untimed)
        return line + f" est_device_s=n/a est_gap_free_device_s=n/a(untimed bucket {missing})"
    spent = sum(n * mean[b] for b, n in ran.items() if n)
    ideal = sum(n * mean[b] for b, n in compact.items() if n)
    line += f" est_device_s={spent:.3f} est_gap_free_device_s={ideal:.3f}"
    if spent:
        line += f" est_reclaimable_share={(spent - ideal) / spent:.3f}"
    return line + f" ({ESTIMATE_NOTE})"


def main():
    pre, post = load(sys.argv[1]), load(sys.argv[2])
    gpu = [int(x) for x in open(sys.argv[3]).read().split() if x.isdigit()]
    wall = float(sys.argv[4])

    def delta(prefix):
        return counter_delta(pre, post, prefix)

    gen = delta("vllm:generation_tokens_total")
    steps = delta("vllm:iteration_tokens_total_count")
    acc = delta("vllm:spec_decode_num_accepted_tokens_total")
    drf = delta("vllm:spec_decode_num_draft_tokens_total")
    out = (
        f"prompt_tok={delta('vllm:prompt_tokens_total'):.0f} gen_tok={gen:.0f} "
        f"gen_tok_per_s={gen / wall if wall else 0:.0f} steps={steps:.0f} "
        f"step_ms={wall * 1000 / steps if steps else 0:.1f} "
        f"req_prefill_s={delta('vllm:request_prefill_time_seconds_sum'):.0f} "
        f"req_decode_s={delta('vllm:request_decode_time_seconds_sum'):.0f} "
        f"gpu_util_mean={sum(gpu) / len(gpu) if gpu else 0:.0f}%"
    )
    if drf:
        out += f" spec_accept_rate={acc / drf:.2f} spec_drafts={drf:.0f}"
    pos = {
        k: v - pre.get(k, 0)
        for k, v in post.items()
        if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total")
    }
    if pos:
        def position(item):
            m = re.search(r'position="(\d+)"', item[0])
            return int(m.group(1)) if m else 0
        out += " per_pos=" + ",".join(f"{v:.0f}" for _, v in sorted(pos.items(), key=position))
    print(out)
    occupancy = slot_occupancy(pre, post, sys.argv[5] if len(sys.argv) > 5 else None)
    if occupancy:
        print(occupancy)
    for key in sorted(k for k in post if k.startswith("scriptor_")):
        print(f"{key} delta={post[key] - pre.get(key, 0):g}")


if __name__ == "__main__":
    main()
