#!/usr/bin/env python3
"""Per-thread CPU residency sampler for Scriptor engine processes.

Writes psr10.tsv (last CPU per thread at 10 Hz), slow1.tsv (CPU time and migrations at 1 Hz),
cpubusy1.tsv (system-wide per-CPU busy ticks at 1 Hz), topology.json (CPU class map) and
summary.json under --out-dir. Run with `python3 -I`.
"""

import argparse
import json
import os
import re
import signal
import sys
import time

STATE_FIELD = 3
UTIME_FIELD = 14
STIME_FIELD = 15
PROCESSOR_FIELD = 39
FIRST_FIELD_AFTER_COMM = 3


def parse_stat(text):
    """Returns (comm, state, utime, stime, processor) from one /proc/<pid>/task/<tid>/stat."""
    i = text.index("(")
    j = text.rindex(")")
    comm = text[i + 1 : j]
    rest = text[j + 2 :].split()

    def field(n):
        return rest[n - FIRST_FIELD_AFTER_COMM]

    return (
        comm,
        field(STATE_FIELD),
        int(field(UTIME_FIELD)),
        int(field(STIME_FIELD)),
        int(field(PROCESSOR_FIELD)),
    )


def read_migrations(path):
    """Returns se.nr_migrations from a sched file, or None when absent."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("se.nr_migrations"):
                    return int(line.split(":")[1].strip())
    except (OSError, ValueError):
        return None
    return None


def cpu_classes(proc_root_sys):
    """Maps cpu number to "big", "little" or "unknown" from cpu_capacity files.

    Returns (classes, capacities). Missing or all-equal capacities give "unknown" for every cpu.
    """
    caps = {}
    try:
        names = os.listdir(proc_root_sys)
    except OSError:
        return {}, {}
    for name in names:
        m = re.fullmatch(r"cpu(\d+)", name)
        if not m:
            continue
        try:
            with open(os.path.join(proc_root_sys, name, "cpu_capacity")) as f:
                caps[int(m.group(1))] = int(f.read().strip())
        except (OSError, ValueError):
            continue
    if not caps or len(set(caps.values())) < 2:
        return {c: "unknown" for c in caps}, caps
    top = max(caps.values())
    return {c: ("big" if v >= 0.9 * top else "little") for c, v in caps.items()}, caps


def sample_threads(proc_root, pid):
    """Yields (tid, comm, state, utime, stime, cpu) for each live thread of pid."""
    task_dir = os.path.join(proc_root, str(pid), "task")
    try:
        tids = os.listdir(task_dir)
    except OSError:
        return
    for tid in tids:
        try:
            with open(os.path.join(task_dir, tid, "stat")) as f:
                comm, state, ut, st, cpu = parse_stat(f.read())
        except (OSError, ValueError, IndexError):
            continue
        yield int(tid), comm, state, ut, st, cpu


def group_of(comm):
    if comm.startswith("scriptor-prep"):
        return "scriptor-prep-*"
    if comm.startswith("tokio"):
        return "tokio-*"
    return comm if comm in ("scriptor-engine", "scriptor-pressure") else "other"


def summarize(rows, classes, clk_tck, migrations=None):
    """Summarizes psr10 rows [(t_ns, pid, tid, comm, state, cpu)] per comm group.

    Only samples in state R count toward residency: the processor field of a sleeping thread is
    stale. Returns a dict keyed by group.
    """
    out = {}
    last_running_cpu = {}
    for _, pid, tid, comm, state, cpu in rows:
        g = out.setdefault(
            group_of(comm),
            {"samples": 0, "running": 0, "sleeping": 0, "class_running": {}, "cpu_running": {}, "cpu_changes": 0},
        )
        g["samples"] += 1
        if state == "S":
            g["sleeping"] += 1
        if state != "R":
            continue
        g["running"] += 1
        cls = classes.get(cpu, "unknown")
        g["class_running"][cls] = g["class_running"].get(cls, 0) + 1
        g["cpu_running"][str(cpu)] = g["cpu_running"].get(str(cpu), 0) + 1
        key = (pid, tid)
        if key in last_running_cpu and last_running_cpu[key] != cpu:
            g["cpu_changes"] += 1
        last_running_cpu[key] = cpu
    for g in out.values():
        running = g["running"]
        g["class_share"] = {c: n / running for c, n in g["class_running"].items()} if running else {}
        g["sleep_share"] = g["sleeping"] / g["samples"] if g["samples"] else 0.0
    engine = out.get("scriptor-engine")
    verdict = {"engine_little_share": None, "migrations": "unavailable" if migrations is None else migrations}
    if engine and engine["running"] and "unknown" not in engine["class_running"]:
        verdict["engine_little_share"] = engine["class_share"].get("little", 0.0)
    out["_verdict"] = verdict
    out["_clk_tck"] = clk_tck
    return out


def read_cpu_busy(proc_root):
    """Returns per-cpu busy ticks (all but idle and iowait) from /proc/stat."""
    busy = []
    try:
        with open(os.path.join(proc_root, "stat")) as f:
            for line in f:
                m = re.match(r"cpu(\d+)\s+(.*)", line)
                if m:
                    v = [int(x) for x in m.group(2).split()]
                    busy.append(sum(v) - v[3] - (v[4] if len(v) > 4 else 0))
    except OSError:
        pass
    return busy


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, action="append", default=[])
    ap.add_argument("--pid-file")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--interval-ms", type=int, default=100)
    ap.add_argument("--slow-s", type=float, default=1.0)
    ap.add_argument("--duration-s", type=float)
    ap.add_argument("--proc-root", default="/proc")
    ap.add_argument("--sys-cpu-root", default="/sys/devices/system/cpu")
    ap.add_argument("--sampler-cpu", type=int, default=1)
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--summarize", metavar="DIR")
    a = ap.parse_args(argv)

    if a.summarize:
        return resummarize(a.summarize)
    os.makedirs(a.out_dir, exist_ok=True)
    pinned = None
    if not a.no_pin:
        try:
            os.sched_setaffinity(0, {a.sampler_cpu})
            pinned = a.sampler_cpu
        except (OSError, AttributeError):
            pinned = None
    classes, caps = cpu_classes(a.sys_cpu_root)
    clk_tck = os.sysconf("SC_CLK_TCK")
    with open(os.path.join(a.out_dir, "topology.json"), "w") as f:
        json.dump({"classes": classes, "capacity": caps, "sampler_cpu": pinned, "clk_tck": clk_tck}, f)

    stop = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.append(1))

    def pids():
        got = list(a.pid)
        if a.pid_file:
            try:
                with open(a.pid_file) as f:
                    got += [int(x) for x in f.read().split()]
            except (OSError, ValueError):
                pass
        return got

    fast = open(os.path.join(a.out_dir, "psr10.tsv"), "w")
    slow = open(os.path.join(a.out_dir, "slow1.tsv"), "w")
    busy = open(os.path.join(a.out_dir, "cpubusy1.tsv"), "w")
    fast.write("t_ns\tpid\ttid\tcomm\tstate\tcpu\n")
    slow.write("t_ns\tpid\ttid\tcomm\tutime\tstime\tnr_migrations\tcpu_last\n")
    start = time.monotonic_ns()
    interval = a.interval_ms * 1_000_000
    slow_every = max(1, int(a.slow_s * 1e9 / interval))
    tick = 0
    while not stop:
        now = time.monotonic_ns()
        if a.duration_s is not None and now - start > a.duration_s * 1e9:
            break
        live = [p for p in pids() if os.path.isdir(os.path.join(a.proc_root, str(p)))]
        if not live and (a.pid or a.pid_file) and tick > 0 and not a.pid_file:
            break
        for pid in live:
            for tid, comm, state, ut, st, cpu in sample_threads(a.proc_root, pid):
                fast.write(f"{now - start}\t{pid}\t{tid}\t{comm}\t{state}\t{cpu}\n")
                if tick % slow_every == 0:
                    mig = read_migrations(os.path.join(a.proc_root, str(pid), "task", str(tid), "sched"))
                    slow.write(f"{now - start}\t{pid}\t{tid}\t{comm}\t{ut}\t{st}\t{'-' if mig is None else mig}\t{cpu}\n")
        if tick % slow_every == 0:
            ticks = "\t".join(str(v) for v in read_cpu_busy(a.proc_root))
            busy.write(f"{now - start}\t{ticks}\n")
        tick += 1
        deadline = start + tick * interval
        delay = (deadline - time.monotonic_ns()) / 1e9
        if delay > 0:
            time.sleep(delay)
    for f in (fast, slow, busy):
        f.close()
    return resummarize(a.out_dir)


def resummarize(directory):
    with open(os.path.join(directory, "topology.json")) as f:
        topo = json.load(f)
    classes = {int(k): v for k, v in topo["classes"].items()}
    rows = []
    with open(os.path.join(directory, "psr10.tsv")) as f:
        next(f)
        for line in f:
            t, pid, tid, comm, state, cpu = line.rstrip("\n").split("\t")
            rows.append((int(t), int(pid), int(tid), comm, state, int(cpu)))
    migrations = None
    try:
        first, last = {}, {}
        with open(os.path.join(directory, "slow1.tsv")) as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if parts[6] == "-":
                    continue
                key = (parts[1], parts[2])
                first.setdefault(key, int(parts[6]))
                last[key] = int(parts[6])
        if last:
            migrations = sum(last[k] - first[k] for k in last)
    except OSError:
        pass
    summary = summarize(rows, classes, topo["clk_tck"], migrations)
    with open(os.path.join(directory, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print(json.dumps(summary["_verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
