import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psr_sampler as ps


def stat_line(comm, state, utime, stime, processor, tid=100):
    """A synthetic 52-field stat line: fields 1 and 2 are tid and (comm)."""
    fields = ["0"] * 50
    fields[0] = state
    fields[11] = str(utime)
    fields[12] = str(stime)
    fields[36] = str(processor)
    return f"{tid} ({comm}) " + " ".join(fields)


class StatParse(unittest.TestCase):
    def test_stat_parse_comm_with_spaces_and_parens(self):
        text = stat_line("scr (eng) x", "R", 7, 9, 17)
        self.assertEqual(ps.parse_stat(text), ("scr (eng) x", "R", 7, 9, 17))

    def test_stat_field_indices_match_proc_man(self):
        self.assertEqual(ps.STATE_FIELD, 3)
        self.assertEqual(ps.UTIME_FIELD, 14)
        self.assertEqual(ps.STIME_FIELD, 15)
        self.assertEqual(ps.PROCESSOR_FIELD, 39)
        text = stat_line("t", "S", 11, 22, 33)
        self.assertEqual(len(text.split(")")[1].split()) + 2, 52)
        self.assertEqual(ps.parse_stat(text)[1:], ("S", 11, 22, 33))


class Classes(unittest.TestCase):
    def build(self, caps):
        root = tempfile.mkdtemp()
        for cpu, cap in caps.items():
            os.makedirs(os.path.join(root, f"cpu{cpu}"))
            with open(os.path.join(root, f"cpu{cpu}", "cpu_capacity"), "w") as f:
                f.write(f"{cap}\n")
        return root

    def test_class_map_from_capacity_table(self):
        classes, _ = ps.cpu_classes(self.build({0: 718, 1: 731, 5: 997, 6: 1017, 7: 1024}))
        self.assertEqual([classes[c] for c in (0, 1, 5, 6, 7)], ["little", "little", "big", "big", "big"])
        classes, _ = ps.cpu_classes(self.build({0: 1024, 1: 1024}))
        self.assertEqual(set(classes.values()), {"unknown"})


class Summary(unittest.TestCase):
    def test_summary_counts_only_running_samples(self):
        classes = {0: "little", 5: "big"}
        rows = [
            (0, 1, 1, "scriptor-engine", "R", 0),
            (1, 1, 1, "scriptor-engine", "R", 5),
            (2, 1, 1, "scriptor-engine", "S", 0),
            (3, 1, 1, "scriptor-engine", "S", 0),
        ]
        s = ps.summarize(rows, classes, 100)
        e = s["scriptor-engine"]
        self.assertEqual((e["samples"], e["running"], e["sleeping"]), (4, 2, 2))
        self.assertEqual(e["class_share"], {"little": 0.5, "big": 0.5})
        self.assertEqual(e["cpu_changes"], 1)
        self.assertEqual(s["_verdict"]["engine_little_share"], 0.5)
        self.assertEqual(s["_verdict"]["migrations"], "unavailable")

    def test_unknown_classes_refuse_the_residency_verdict(self):
        rows = [(0, 1, 1, "scriptor-engine", "R", 0)]
        s = ps.summarize(rows, {0: "unknown"}, 100)
        self.assertIsNone(s["_verdict"]["engine_little_share"])


class ProcTree(unittest.TestCase):
    def tree(self):
        root = tempfile.mkdtemp()
        task = os.path.join(root, "42", "task", "100")
        os.makedirs(task)
        with open(os.path.join(task, "stat"), "w") as f:
            f.write(stat_line("scriptor-engine", "R", 1, 2, 5))
        return root

    def test_missing_sched_file_gives_blank_migrations(self):
        root = self.tree()
        self.assertIsNone(ps.read_migrations(os.path.join(root, "42", "task", "100", "sched")))

    def test_sched_file_with_migrations_is_read(self):
        root = self.tree()
        path = os.path.join(root, "42", "task", "100", "sched")
        with open(path, "w") as f:
            f.write("se.nr_migrations                              :                   12\n")
        self.assertEqual(ps.read_migrations(path), 12)

    def test_exited_thread_is_skipped(self):
        root = self.tree()
        os.makedirs(os.path.join(root, "42", "task", "101"))
        got = list(ps.sample_threads(root, 42))
        self.assertEqual([g[0] for g in got], [100])
        self.assertEqual(list(ps.sample_threads(root, 999)), [])


if __name__ == "__main__":
    unittest.main()
