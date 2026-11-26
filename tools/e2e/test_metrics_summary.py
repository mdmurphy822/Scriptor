"""Synthetic checks of the slot-occupancy line of metrics_summary.py: exact counts, timed
coverage and the labelled bucket-mean estimates. Run with
`python3 -I tools/e2e/test_metrics_summary.py`."""
import importlib.util
import os
import re
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "metrics_summary", os.path.join(_HERE, "metrics_summary.py")
)
assert _SPEC is not None and _SPEC.loader is not None
ms = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ms)

M = 'model_name="m"'


def series(steps, active, rows, compact_rows, padded, sparse, sparse_steps, buckets):
    """A post scrape: totals, then per bucket (ran, compact, timed, seconds)."""
    out = {
        f"scriptor_decode_step_shapes_total{{{M}}}": steps,
        f"scriptor_decode_active_rows_total{{{M}}}": active,
        f"scriptor_decode_bucket_rows_total{{{M}}}": rows,
        f"scriptor_decode_compact_rows_total{{{M}}}": compact_rows,
        f"scriptor_decode_padded_rows_total{{{M}}}": padded,
        f"scriptor_decode_sparse_rows_total{{{M}}}": sparse,
        f"scriptor_decode_sparse_steps_total{{{M}}}": sparse_steps,
    }
    names = (
        "scriptor_decode_bucket_steps_total",
        "scriptor_decode_compact_bucket_steps_total",
        "scriptor_decode_bucket_timed_steps_total",
        "scriptor_decode_bucket_device_seconds_total",
    )
    for bucket, values in buckets.items():
        for name, value in zip(names, values):
            out[f'{name}{{{M},bucket="{bucket}"}}'] = value
    return out


def field(line, name):
    m = re.search(rf"(?:^| ){re.escape(name)}=(\S+)", line)
    return m.group(1) if m else None


# Bucket 32 ran 10 steps, 6 timed at 10 ms; bucket 1 ran 4 steps, all timed at 2 ms.
# A gap-free layout would run bucket 32 once and bucket 1 thirteen times.
PARTIAL = series(
    steps=14, active=45, rows=324, compact_rows=45, padded=279, sparse=279, sparse_steps=9,
    buckets={32: (10, 1, 6, 0.060), 1: (4, 13, 4, 0.008)},
)


class PartialCoverage(unittest.TestCase):
    def setUp(self):
        self.line = ms.slot_occupancy({}, PARTIAL)

    def test_exact_counts(self):
        self.assertEqual(field(self.line, "steps"), "14")
        self.assertEqual(field(self.line, "padded_share"), f"{279 / 324:.3f}")
        self.assertEqual(field(self.line, "sparse_share"), f"{279 / 324:.3f}")
        self.assertEqual(field(self.line, "sparse_steps"), "9")

    def test_actual_timed_seconds_and_coverage(self):
        self.assertEqual(field(self.line, "timed_steps"), "10")
        self.assertEqual(field(self.line, "timed_coverage"), f"{10 / 14:.3f}")
        self.assertEqual(field(self.line, "timed_device_s"), "0.068")
        self.assertEqual(field(self.line, "bucket_mean_ms"), "1:2.00,32:10.00")

    def test_estimates_scale_bucket_means_by_step_counts(self):
        spent = 10 * 0.010 + 4 * 0.002
        ideal = 1 * 0.010 + 13 * 0.002
        self.assertEqual(field(self.line, "est_device_s"), f"{spent:.3f}")
        self.assertEqual(field(self.line, "est_gap_free_device_s"), f"{ideal:.3f}")
        self.assertEqual(
            field(self.line, "est_reclaimable_share"), f"{(spent - ideal) / spent:.3f}"
        )

    def test_estimates_are_labelled_and_never_measured(self):
        self.assertIn(ms.ESTIMATE_NOTE, self.line)
        self.assertIn("ignores per-bucket variation and compaction overhead", self.line)
        for bare in ("device_s", "gap_free_device_s", "reclaimable_share"):
            self.assertIsNone(field(self.line, bare), bare)
        self.assertNotIn("measured", self.line)
        self.assertNotIn("recoverable", self.line)


class Coverage(unittest.TestCase):
    def test_untimed_needed_bucket_is_not_estimated(self):
        post = series(
            steps=14, active=45, rows=324, compact_rows=45, padded=279, sparse=279,
            sparse_steps=9, buckets={32: (10, 1, 6, 0.060), 1: (4, 13, 0, 0.0)},
        )
        line = ms.slot_occupancy({}, post)
        self.assertEqual(field(line, "timed_device_s"), "0.060")
        self.assertEqual(field(line, "est_device_s"), "n/a")
        self.assertIn("est_gap_free_device_s=n/a(untimed bucket 1)", line)
        self.assertIsNone(field(line, "est_reclaimable_share"))

    def test_no_timing_reports_zero_timed_steps_only(self):
        post = series(
            steps=2, active=2, rows=2, compact_rows=2, padded=0, sparse=0, sparse_steps=0,
            buckets={1: (2, 2, 0, 0.0)},
        )
        line = ms.slot_occupancy({}, post)
        self.assertEqual(field(line, "timed_steps"), "0")
        self.assertNotIn("est_", line)

    def test_no_occupancy_counters_give_no_line(self):
        self.assertEqual(ms.slot_occupancy({}, {"vllm:x_total": 1.0}), "")


class Phases(unittest.TestCase):
    def test_pair_deltas_and_phase_tag(self):
        pre = {k: v / 2 for k, v in PARTIAL.items()}
        line = ms.slot_occupancy(pre, PARTIAL, "tail")
        self.assertTrue(line.startswith("occupancy phase=tail "))
        self.assertEqual(field(line, "steps"), "7")
        self.assertEqual(field(line, "timed_steps"), "5")
        self.assertEqual(field(line, "timed_device_s"), "0.034")

    def test_load_keeps_labelled_counters(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.txt")
            with open(path, "w") as fh:
                fh.write("# TYPE scriptor_decode_bucket_steps_total counter\n")
                fh.write(f'scriptor_decode_bucket_steps_total{{{M},bucket="32"}} 10\n')
                fh.write(f"scriptor_graphs_captured{{{M}}} 6\n")
            loaded = ms.load(path)
        self.assertEqual(loaded, {f'scriptor_decode_bucket_steps_total{{{M},bucket="32"}}': 10.0})


if __name__ == "__main__":
    unittest.main()
