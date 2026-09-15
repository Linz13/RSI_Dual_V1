"""CPU-only checks for priority, per-task smoke gates and safe resumption."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import midasheng_v4_v5_eval as app


def candidate(version, number=0):
    return app.existing.Candidate(f"midasheng_rewardv{version}_r{number}",
        f"midasheng_rewardv{version}", "test", "midasheng", number,
        "/model", "/adapter", "/python", "sdpa", "ready")


class SchedulingTests(unittest.TestCase):
    def test_v5_first_three_tasks(self):
        v4, v5 = candidate(4), candidate(5)
        jobs = app.ordered_tasks([v4, candidate(4, 1), v5])
        self.assertEqual([c for c, _ in jobs[:3]], [v5] * 3)
        self.assertEqual({s for _, s in jobs[:3]}, set(app.existing.BENCHMARKS))
        self.assertEqual(len(jobs), 9)
        self.assertEqual(jobs[3][0], v4)

    def test_four_distinct_gpus(self):
        self.assertEqual(app.parse_gpus("4,5,6,7"), ["4", "5", "6", "7"])
        for spec in ("0,1", "0,1,2,2", "0,1,2,x"):
            with self.assertRaises(ValueError):
                app.parse_gpus(spec)

    def exercise(self, failing_smoke=False, already_complete=False):
        c = candidate(5)
        done, calls = set(), []
        if already_complete:
            done.update((s, "full") for s in app.existing.BENCHMARKS)
        def status(output, suite, size, unused):
            return ((suite, size) in done, "unfinished")
        def commands(unused, suite, size, output):
            return [[suite, size]]
        def run(command, env, log):
            suite, size = command
            calls.append((suite, size))
            if failing_smoke and suite == "emotiontalk" and size == "smoke":
                raise RuntimeError("simulated smoke failure")
            done.add((suite, size))
        with tempfile.TemporaryDirectory() as tmp:
            runner = app.Runner(Path(tmp))
            with patch.object(app, "task_status", side_effect=status), \
                 patch.object(app, "commands", side_effect=commands), \
                 patch.object(app, "summarize"), \
                 patch.object(runner, "run_command", side_effect=run):
                if failing_smoke:
                    with self.assertRaises(RuntimeError):
                        runner.stage([c], "full", ["4", "5", "6", "7"])
                else:
                    runner.stage([c], "full", ["4", "5", "6", "7"])
        return calls

    def test_smoke_before_each_full_task(self):
        calls = self.exercise()
        for suite in app.existing.BENCHMARKS:
            self.assertLess(calls.index((suite, "smoke")), calls.index((suite, "full")))

    def test_failed_smoke_blocks_only_its_full_task(self):
        calls = self.exercise(failing_smoke=True)
        self.assertNotIn(("emotiontalk", "full"), calls)
        self.assertIn(("stylecap", "full"), calls)
        self.assertIn(("paraspeechcaps", "full"), calls)

    def test_completed_full_tasks_skip_all_commands(self):
        self.assertEqual(self.exercise(already_complete=True), [])


if __name__ == "__main__":
    unittest.main()
