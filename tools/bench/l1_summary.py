#!/usr/bin/env python3
"""Summarizes one Scriptor run's L1 counters: stderr l1-flight and l1-step lines, an optional
PRE/POST /metrics pair of scriptor_l1_* series, and an optional psr_sampler output directory.

Reports vision-stream idle, poll lag, start wait, device time, flight and step closure
residuals, the per-flight cost regression, the turnaround split, the spin or yield verdict and
engine-thread CPU residency. Run with `python3 -I`.
"""

import argparse
import json
import math
import os
import re
import sys

LINE = re.compile(r"scriptor: l1-(flight|step) (.*)$")

CLOSURE_FLIGHT_MEAN_US = 1000.0
CLOSURE_STEP_FRACTION = 0.001
SPIN_AT_LEAST = 0.9
YIELD_AT_MOST = 0.2
T_HOST_OPEN_SECONDS = 4.0
T_HOST_CLOSED_SECONDS = 2.5
IDLE_INSIDE_BAR_SECONDS = 10.0
START_WAIT_LEVER_SECONDS = 10.0
REGRESSION_R2_BAR = 0.9
BIN_LOW_PATCHES = (4000, 6000)
BIN_HIGH_PATCHES = 12000
BIN_DIFFERENCE_BAR = 0.08


def parse_kv(text):
    """Returns the key=value tokens of a line body; `-` values become None, others ints or
    floats, and non-numeric values stay strings."""
    out = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if value == "-":
            out[key] = None
            continue
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def parse_log(lines):
    """Returns (flights, steps), each a list of dicts in log order, from stderr lines."""
    flights, steps = [], []
    for line in lines:
        m = LINE.search(line.rstrip("\n"))
        if not m:
            continue
        rec = parse_kv(m.group(2))
        (flights if m.group(1) == "flight" else steps).append(rec)
    return flights, steps


def parse_metrics(text):
    """Returns {(name, labels): value} for every sample line; labels is a sorted tuple of
    (key, value) pairs."""
    out = {}
    for line in text.splitlines():
        m = re.fullmatch(r"([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)", line.strip())
        if not m:
            continue
        labels = tuple(sorted(re.findall(r'(\w+)="([^"]*)"', m.group(2) or "")))
        try:
            out[(m.group(1), labels)] = float(m.group(3))
        except ValueError:
            continue
    return out


def metrics_delta(pre, post):
    """Per-series POST minus PRE; a series missing from PRE counts from zero."""
    return {k: v - pre.get(k, 0.0) for k, v in post.items()}


def series_value(metrics, name, **labels):
    """Sums the samples of `name` whose labels include every given label."""
    want = set(labels.items())
    return sum(v for (n, ls), v in metrics.items() if n == name and want <= set(ls))


def quantile(values, q):
    """Nearest-rank quantile of `values`; None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def flight_ready_us(f):
    """Host microsecond at which the flight's device work was ready, on the process clock."""
    return round(f["t_begin_ms"] * 1000) + f["launch_us"] + f["counter_us"] - f["lag_us"]


def flight_closure(flights):
    """Residual over flights that carry callback stamps: counter_us - (start_rel_us + device_us
    + cb_lat_us + lag_us) when the signed fields are present, else counter_us - (start_wait_us
    + device_us + lag_us); flights with estimated lag close by construction and are counted
    apart."""
    residuals = []
    estimated = 0
    for f in flights:
        if f.get("start_wait_us") is None or f.get("cb") == 0:
            estimated += 1
            continue
        if f.get("start_rel_us") is not None and f.get("cb_lat_us") is not None:
            lead = f["start_rel_us"] + f["cb_lat_us"]
        else:
            lead = f["start_wait_us"]
        residuals.append(f["counter_us"] - (lead + f["device_us"] + f["lag_us"]))
    absolute = [abs(r) for r in residuals]
    mean = sum(absolute) / len(absolute) if absolute else 0.0
    return {
        "flights_closed": len(residuals),
        "flights_estimated": estimated,
        "mean_abs_residual_us": mean,
        "max_abs_residual_us": max(absolute) if absolute else 0,
        "ok": mean < CLOSURE_FLIGHT_MEAN_US,
    }


def vision_idle(flights):
    """Vision-stream idle: inside the first-begin to last-ready span, and between flights.

    A flight's gap is attributed by the previous flight's pending_after: a gap that started
    with images still pending is dispatch or lag, one that started with none is starvation.
    """
    if not flights:
        return {}
    first_begin = min(round(f["t_begin_ms"] * 1000) for f in flights)
    last_ready = max(flight_ready_us(f) for f in flights)
    device = sum(f["device_us"] for f in flights)
    pending_gap = starved_gap = 0
    previous_pending = None
    for f in flights:
        gap = f.get("gap_us")
        if gap is not None and previous_pending is not None:
            if previous_pending > 0:
                pending_gap += gap
            else:
                starved_gap += gap
        previous_pending = f.get("pending_after")
    lags = [f["lag_us"] for f in flights]
    cb_lats = [f["cb_lat_us"] for f in flights if f.get("cb_lat_us") is not None]
    start_waits = [f["start_wait_us"] for f in flights if f.get("start_wait_us") is not None]
    return {
        "flights": len(flights),
        "m1_idle_inside_s": ((last_ready - first_begin) - device) / 1e6,
        "m1_idle_gap_s": sum(f["gap_us"] for f in flights if f.get("gap_us") is not None) / 1e6,
        "gap_pending_s": pending_gap / 1e6,
        "gap_starved_s": starved_gap / 1e6,
        "device_s": device / 1e6,
        "copy_s": sum(f["copy_us"] for f in flights) / 1e6,
        "launch_s": sum(f["launch_us"] for f in flights) / 1e6,
        "start_wait_s": sum(start_waits) / 1e6,
        "poll_lag_s": sum(lags) / 1e6,
        "poll_lag_us_p50": quantile(lags, 0.5),
        "poll_lag_us_p90": quantile(lags, 0.9),
        "poll_lag_us_p99": quantile(lags, 0.99),
        "cb_lat_us_min": min(cb_lats) if cb_lats else None,
        "cb_lat_us_p50": quantile(cb_lats, 0.5),
        "polls_not_ready": sum(f["polls_not_ready"] for f in flights),
        "polls": sum(f["polls"] for f in flights),
    }


def solve3(matrix, rhs):
    """Solves a 3x3 linear system by Gaussian elimination with partial pivoting."""
    a = [row[:] + [r] for row, r in zip(matrix, rhs)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-300:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(3):
            if r != col:
                factor = a[r][col] / a[col][col]
                a[r] = [x - factor * y for x, y in zip(a[r], a[col])]
    return [a[i][3] / a[i][i] for i in range(3)]


def regress(patches, sum_n2, device_us):
    """Least-squares device_us = a*patches + b*sum_n2 + c; returns (a, b, c, r2) or None."""
    n = len(device_us)
    if n < 3:
        return None
    scale_p = max(patches) or 1.0
    scale_n = max(sum_n2) or 1.0
    rows = [(p / scale_p, s / scale_n, 1.0) for p, s in zip(patches, sum_n2)]
    gram = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    moment = [sum(r[i] * y for r, y in zip(rows, device_us)) for i in range(3)]
    beta = solve3(gram, moment)
    if beta is None:
        return None
    a, b, c = beta[0] / scale_p, beta[1] / scale_n, beta[2]
    mean = sum(device_us) / n
    total = sum((y - mean) ** 2 for y in device_us)
    resid = sum((y - (a * p + b * s + c)) ** 2 for p, s, y in zip(patches, sum_n2, device_us))
    r2 = 1.0 - resid / total if total > 0 else 1.0
    return a, b, c, r2


def flight_cost_model(flights):
    """The H-FLT-2 regression of device_us on patches and sum_n2 and its per-bin per-patch
    comparison, with the verdict it supports."""
    fit = regress(
        [f["patches"] for f in flights],
        [f["sum_n2"] for f in flights],
        [f["device_us"] for f in flights],
    )
    if fit is None:
        return {"verdict": "inconclusive", "reason": "too few flights"}
    a, b, c, r2 = fit

    def per_patch(selected):
        patches = sum(f["patches"] for f in selected)
        if not patches:
            return None
        return sum(f["device_us"] - b * f["sum_n2"] - c for f in selected) / patches

    low = per_patch([f for f in flights if BIN_LOW_PATCHES[0] <= f["patches"] <= BIN_LOW_PATCHES[1]])
    high = per_patch([f for f in flights if f["patches"] >= BIN_HIGH_PATCHES])
    difference = None
    if low and high is not None and low != 0:
        difference = abs(high - low) / abs(low)
    out = {
        "a_us_per_patch": a,
        "b_us_per_n2": b,
        "c_us": c,
        "r2": r2,
        "per_patch_us_low_bin": low,
        "per_patch_us_high_bin": high,
        "bin_difference": difference,
    }
    if r2 < REGRESSION_R2_BAR or difference is None:
        out["verdict"] = "inconclusive"
    elif difference >= BIN_DIFFERENCE_BAR:
        out["verdict"] = "per-patch term differs between bins (needs the bit-equality job to proceed)"
    else:
        out["verdict"] = "per-patch term does not differ between bins"
    return out


def step_partition(steps):
    """Closure of A + S + C + T against the launch-to-launch span over consecutive step lines
    (n and n+1); the span is the difference of the t_ms stamps."""
    residuals = []
    spans = parts = 0
    for cur, nxt in zip(steps, steps[1:]):
        if nxt["n"] != cur["n"] + 1:
            continue
        span = round((nxt["t_ms"] - cur["t_ms"]) * 1000)
        part = cur["a_us"] + cur["sync_us"] + cur["c_us"] + cur["turn_us"]
        residuals.append(part - span)
        spans += span
        parts += part
    fraction = abs(parts - spans) / spans if spans else 0.0
    return {
        "pairs": len(residuals),
        "sum_parts_us": parts,
        "sum_span_us": spans,
        "relative_difference": fraction,
        "max_abs_residual_us": max((abs(r) for r in residuals), default=0),
        "ok": bool(residuals) and fraction <= CLOSURE_STEP_FRACTION,
    }


def turnaround_split(steps):
    """Sums of the turnaround windows and their prefill, blocking vision wait and host parts,
    with the double-buffer lever verdict."""
    total = sum(s["turn_us"] for s in steps)
    prefill = sum(s["turn_prefill_us"] for s in steps)
    vblock = sum(s["turn_vblock_us"] for s in steps)
    host = sum(s["turn_host_us"] for s in steps)
    seconds = host / 1e6
    if seconds >= T_HOST_OPEN_SECONDS:
        lever = "open"
    elif seconds < T_HOST_CLOSED_SECONDS:
        lever = "closed"
    else:
        lever = "between"
    return {
        "steps": len(steps),
        "turnaround_s": total / 1e6,
        "prefill_s": prefill / 1e6,
        "vision_wait_s": vblock / 1e6,
        "host_s": seconds,
        "host_share": host / total if total else None,
        "double_buffer_lever": lever,
    }


def spin_verdict(sync_cpu, sync):
    """`spin` when the synchronize burns at least 0.9 of its wall time on the CPU, `yield` at
    most 0.2, `mixed` between; `unknown` without synchronize time."""
    if sync <= 0:
        return {"ratio": None, "verdict": "unknown"}
    ratio = sync_cpu / sync
    if ratio >= SPIN_AT_LEAST:
        verdict = "spin"
    elif ratio <= YIELD_AT_MOST:
        verdict = "yield"
    else:
        verdict = "mixed"
    return {"ratio": ratio, "verdict": verdict}


def residency(steps, classes, psr_summary):
    """Engine-thread residency: in-engine steps per cpu class from the step lines, next to the
    sampler's running-sample share when a psr summary is given."""
    per_class = {}
    migrations = 0
    previous = None
    for s in steps:
        cpu = s.get("cpu")
        if cpu is None:
            continue
        cls = classes.get(cpu, "unknown")
        per_class[cls] = per_class.get(cls, 0) + 1
        if previous is not None and previous != cpu:
            migrations += 1
        previous = cpu
    total = sum(per_class.values())
    out = {
        "engine_steps_by_class": per_class,
        "engine_little_share": per_class.get("little", 0) / total if total and "unknown" not in per_class else None,
        "engine_cpu_migrations": migrations,
    }
    if psr_summary is not None:
        out["sampler_engine_little_share"] = psr_summary.get("_verdict", {}).get("engine_little_share")
        out["sampler_migrations"] = psr_summary.get("_verdict", {}).get("migrations")
    return out


def summarize(flights, steps, metrics=None, classes=None, psr_summary=None):
    """The full report as a dict."""
    sync = sum(s["sync_us"] for s in steps)
    sync_cpu = sum(s["sync_cpu_us"] for s in steps)
    report = {
        "vision": vision_idle(flights),
        "flight_closure": flight_closure(flights),
        "flight_cost_model": flight_cost_model(flights),
        "step_partition": step_partition(steps),
        "turnaround": turnaround_split(steps),
        "sync_spin": spin_verdict(sync_cpu, sync),
        "residency": residency(steps, classes or {}, psr_summary),
    }
    v = report["vision"]
    if v:
        report["bars"] = {
            "m1_idle_inside_at_least_10s": v["m1_idle_inside_s"] >= IDLE_INSIDE_BAR_SECONDS,
            "poll_lag_at_least_10s": v["poll_lag_s"] >= IDLE_INSIDE_BAR_SECONDS,
            "start_wait_lever_candidate": v["start_wait_s"] >= START_WAIT_LEVER_SECONDS,
        }
    if metrics is not None:
        report["metrics_delta"] = {
            "steps": series_value(metrics, "scriptor_l1_steps_total"),
            "flights": series_value(metrics, "scriptor_l1_vision_flights_total"),
            "flight_patches": series_value(metrics, "scriptor_l1_vision_flight_patches_total"),
            "sync_s": series_value(metrics, "scriptor_l1_step_span_seconds_total", span="sync"),
            "sync_cpu_s": series_value(metrics, "scriptor_l1_step_span_seconds_total", span="sync_cpu"),
            "poll_lag_s": series_value(metrics, "scriptor_l1_vision_seconds_total", part="poll_lag"),
            "device_s": series_value(metrics, "scriptor_l1_vision_seconds_total", part="device"),
            "callbacks_missing": series_value(metrics, "scriptor_l1_vision_callbacks_missing_total"),
            "neg_lag": series_value(metrics, "scriptor_l1_vision_neg_lag_total"),
            "engine_cpu_migrations": series_value(metrics, "scriptor_l1_engine_cpu_migrations_total"),
        }
    return report


def render_text(report):
    """A short text block of the report's headline numbers."""
    lines = []
    v = report["vision"]
    if v:
        lines.append(
            f"vision: {v['flights']} flights, device {v['device_s']:.3f} s, "
            f"start wait {v['start_wait_s']:.3f} s, poll lag {v['poll_lag_s']:.3f} s "
            f"(p50 {v['poll_lag_us_p50']} us, p90 {v['poll_lag_us_p90']} us, p99 {v['poll_lag_us_p99']} us)"
        )
        lines.append(
            f"M1 idle inside {v['m1_idle_inside_s']:.3f} s, between flights {v['m1_idle_gap_s']:.3f} s "
            f"(pending {v['gap_pending_s']:.3f} s, starved {v['gap_starved_s']:.3f} s)"
        )
    c = report["flight_closure"]
    lines.append(
        f"flight closure: mean |residual| {c['mean_abs_residual_us']:.1f} us over {c['flights_closed']} flights "
        f"({c['flights_estimated']} estimated), ok={c['ok']}"
    )
    m = report["flight_cost_model"]
    lines.append(f"flight cost model: {m['verdict']}" + (f", r2 {m['r2']:.3f}" if "r2" in m else ""))
    p = report["step_partition"]
    lines.append(
        f"step partition: {p['pairs']} pairs, relative difference {p['relative_difference']:.6f}, ok={p['ok']}"
    )
    t = report["turnaround"]
    lines.append(
        f"turnaround: {t['turnaround_s']:.3f} s = prefill {t['prefill_s']:.3f} + vision wait "
        f"{t['vision_wait_s']:.3f} + host {t['host_s']:.3f}; double-buffer lever {t['double_buffer_lever']}"
    )
    s = report["sync_spin"]
    lines.append(f"synchronize: cpu/wall {s['ratio']}, verdict {s['verdict']}")
    r = report["residency"]
    lines.append(f"engine residency: {r['engine_steps_by_class']}, little share {r['engine_little_share']}")
    return "\n".join(lines)


def read_text(path):
    with open(path) as f:
        return f.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stderr", required=True, help="engine stderr log holding l1-flight and l1-step lines")
    ap.add_argument("--pre", help="/metrics text before the run")
    ap.add_argument("--post", help="/metrics text after the run")
    ap.add_argument("--psr-dir", help="psr_sampler output directory (summary.json, topology.json)")
    ap.add_argument("--json-out", help="write the report JSON here")
    a = ap.parse_args(argv)
    flights, steps = parse_log(read_text(a.stderr).splitlines())
    metrics = None
    if a.post:
        pre = parse_metrics(read_text(a.pre)) if a.pre else {}
        metrics = metrics_delta(pre, parse_metrics(read_text(a.post)))
    classes, psr_summary = {}, None
    if a.psr_dir:
        with open(os.path.join(a.psr_dir, "topology.json")) as f:
            classes = {int(k): v for k, v in json.load(f)["classes"].items()}
        with open(os.path.join(a.psr_dir, "summary.json")) as f:
            psr_summary = json.load(f)
    report = summarize(flights, steps, metrics, classes, psr_summary)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(report, f, indent=1, sort_keys=True)
    print(json.dumps(report, sort_keys=True))
    print(render_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
