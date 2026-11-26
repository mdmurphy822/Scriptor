import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l1_summary as ls

GOLDEN_FLIGHT = (
    "scriptor: l1-flight id=17 t_begin_ms=1234.567 images=6 frames=6 patches=7340 "
    "sum_n2=18234500 launch_us=312 start_wait_us=41200 copy_us=3100 device_us=262400 "
    "lag_us=28100 counter_us=331800 polls=5 polls_not_ready=4 gap_us=61200 pending_after=3 cb=1"
)


def flight_line(id, t_begin_ms, patches, sum_n2, launch, start_wait, copy, device, lag, counter,
                polls, not_ready, gap, pending, cb=1):
    dash = lambda v: "-" if v is None else v
    return (
        f"scriptor: l1-flight id={id} t_begin_ms={t_begin_ms} images=2 frames=2 patches={patches} "
        f"sum_n2={sum_n2} launch_us={launch} start_wait_us={dash(start_wait)} copy_us={copy} "
        f"device_us={device} lag_us={lag} counter_us={counter} polls={polls} "
        f"polls_not_ready={not_ready} gap_us={dash(gap)} pending_after={pending} cb={cb}"
        + ("" if cb else " lag=est")
    )


def step_line(n, t_ms, rows, a, sync, sync_cpu, c, turn, prefill, vblock, host, between, flight, cpu):
    return (
        f"scriptor: l1-step n={n} t_ms={t_ms} rows={rows} a_us={a} sync_us={sync} "
        f"sync_cpu_us={sync_cpu} c_us={c} turn_us={turn} turn_prefill_us={prefill} "
        f"turn_vblock_us={vblock} turn_host_us={host} between={between} flight={flight} cpu={cpu}"
    )


class SignedStamps(unittest.TestCase):
    def test_signed_fields_parse_and_close_the_flight(self):
        line = GOLDEN_FLIGHT.replace("start_wait_us=41200", "start_wait_us=0").replace(
            "cb=1", "cb=1 start_rel_us=-800 cb_lat_us=-300"
        ).replace("counter_us=331800", "counter_us=%d" % (-800 + 262400 - 300 + 28100))
        flights, _ = ls.parse_log([line])
        self.assertEqual(flights[0]["start_rel_us"], -800)
        self.assertEqual(flights[0]["cb_lat_us"], -300)
        closure = ls.flight_closure(flights)
        self.assertEqual(closure["max_abs_residual_us"], 0)
        self.assertEqual(ls.vision_idle(flights)["cb_lat_us_min"], -300)

    def test_estimated_flight_carries_no_signed_fields(self):
        line = flight_line(3, 10.0, 100, 1000, 50, None, 0, 6000, 3000, 9000, 1, 0, None, 0, cb=0)
        flights, _ = ls.parse_log([line + " start_rel_us=- cb_lat_us=-"])
        self.assertIsNone(flights[0]["cb_lat_us"])
        self.assertEqual(ls.vision_idle(flights)["cb_lat_us_min"], None)


class Closure(unittest.TestCase):
    def test_flight_closure_on_scripted_lines(self):
        lines = [
            "unrelated stderr line",
            flight_line(16, "1000.000", 4000, 9000000, 300, 40000, 3000, 250000, 30000, 320000,
                        4, 3, None, 2),
            GOLDEN_FLIGHT,
            flight_line(18, "1400.000", 900, 400000, 100, None, 800, 60000, 9000, 70000,
                        1, 0, 5000, 0, cb=0),
            flight_line(19, "1500.000", 100, 10000, 50, 1000, 100, 5000, 2000, 8000,
                        1, 0, 7000, 0),
        ]
        flights, steps = ls.parse_log(lines)
        self.assertEqual(len(flights), 4)
        self.assertEqual(steps, [])
        self.assertEqual(flights[1]["id"], 17)
        self.assertEqual(flights[1]["t_begin_ms"], 1234.567)
        self.assertIsNone(flights[0]["gap_us"])
        self.assertIsNone(flights[2]["start_wait_us"])
        closure = ls.flight_closure(flights)
        self.assertEqual(closure["flights_closed"], 3)
        self.assertEqual(closure["flights_estimated"], 1)
        self.assertEqual(closure["mean_abs_residual_us"], 100 / 3)
        self.assertEqual(closure["max_abs_residual_us"], 100)
        self.assertTrue(closure["ok"])
        idle = ls.vision_idle(flights)
        self.assertEqual(idle["gap_pending_s"], 0.0662)
        self.assertEqual(idle["gap_starved_s"], 0.007)
        self.assertEqual(idle["poll_lag_us_p50"], 9000)
        self.assertEqual(idle["start_wait_s"], 0.0822)

    def test_step_partition_sums_to_launch_to_launch(self):
        lines = [
            step_line(1, "100.000", 24, 2000, 45000, 44900, 60, 1340, 0, 0, 1340, 0, 1, 17),
            step_line(2, "148.400", 24, 1900, 45100, 45000, 70, 1230, 200, 0, 1030, 0, 0, 17),
            step_line(3, "196.700", 24, 2100, 44900, 44800, 60, 1400, 0, 0, 1400, 0, 0, 3),
            step_line(7, "500.000", 24, 2000, 45000, 44900, 60, 1340, 0, 0, 1340, 0, 0, 3),
        ]
        _, steps = ls.parse_log(lines)
        self.assertEqual(len(steps), 4)
        part = ls.step_partition(steps)
        self.assertEqual(part["pairs"], 2, "the step number gap is skipped")
        self.assertEqual(part["sum_parts_us"], part["sum_span_us"])
        self.assertEqual(part["max_abs_residual_us"], 0)
        self.assertTrue(part["ok"])
        steps[1]["turn_us"] += 5000
        self.assertFalse(ls.step_partition(steps)["ok"])
        split = ls.turnaround_split(steps[:3])
        self.assertEqual(split["prefill_s"], 0.0002)
        self.assertEqual(split["double_buffer_lever"], "closed")
        residency = ls.residency(steps, {17: "little", 3: "big"}, None)
        self.assertEqual(residency["engine_steps_by_class"], {"little": 2, "big": 2})
        self.assertEqual(residency["engine_cpu_migrations"], 1)


class Regression(unittest.TestCase):
    def test_regression_recovers_known_a_b_c(self):
        a, b, c = 30.0, 0.0002, 5000.0
        patches, sum_n2, device = [], [], []
        for i in range(60):
            p = 500 + 263 * i
            jobs = i % 5 + 1
            n2 = jobs * (p // jobs) ** 2
            patches.append(p)
            sum_n2.append(n2)
            device.append(a * p + b * n2 + c)
        fa, fb, fc, r2 = ls.regress(patches, sum_n2, device)
        self.assertAlmostEqual(fa, a, delta=1e-6 * a)
        self.assertAlmostEqual(fb, b, delta=1e-6 * b)
        self.assertAlmostEqual(fc, c, delta=1e-4 * c)
        self.assertGreater(r2, 0.999999)
        flights = [
            {"patches": p, "sum_n2": s, "device_us": d}
            for p, s, d in zip(patches, sum_n2, device)
        ]
        model = ls.flight_cost_model(flights)
        self.assertLess(model["bin_difference"], 1e-6)
        self.assertEqual(model["verdict"], "per-patch term does not differ between bins")
        self.assertEqual(ls.regress([1, 2], [1, 4], [1.0, 2.0]), None)


class Spin(unittest.TestCase):
    def test_spin_verdict_thresholds(self):
        self.assertEqual(ls.spin_verdict(90, 100)["verdict"], "spin")
        self.assertEqual(ls.spin_verdict(89, 100)["verdict"], "mixed")
        self.assertEqual(ls.spin_verdict(21, 100)["verdict"], "mixed")
        self.assertEqual(ls.spin_verdict(20, 100)["verdict"], "yield")
        self.assertEqual(ls.spin_verdict(0, 100)["verdict"], "yield")
        self.assertEqual(ls.spin_verdict(5, 0)["verdict"], "unknown")
        text = (
            'scriptor_l1_step_span_seconds_total{model_name="m",span="sync"} 3.5\n'
            'scriptor_l1_step_span_seconds_total{model_name="m",span="sync_cpu"} 3.2\n'
        )
        pre = ls.parse_metrics(text.replace("3.5", "0.5").replace("3.2", "0.2"))
        delta = ls.metrics_delta(pre, ls.parse_metrics(text))
        sync = ls.series_value(delta, "scriptor_l1_step_span_seconds_total", span="sync")
        cpu = ls.series_value(delta, "scriptor_l1_step_span_seconds_total", span="sync_cpu")
        self.assertEqual((sync, cpu), (3.0, 3.0))
        self.assertEqual(ls.spin_verdict(cpu, sync)["verdict"], "spin")


if __name__ == "__main__":
    unittest.main()
