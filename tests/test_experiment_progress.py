"""Console, non-interactive, and subprocess paths without training a model."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import experiment_progress as progress


class ConsoleTests(unittest.TestCase):
    def test_epoch_rollover_and_protocol_switch_replace_both_lines(self):
        stream = io.StringIO()
        with patch.object(progress, "_interactive", return_value=True):
            console = progress._Console(stream)
        for epoch, step, completed in [(1, 9, 0), (1, 10, 1), (2, 0, 1)]:
            event = dict(phase="training", epoch=epoch, max_epochs=50, step=step, steps=10)
            lines = progress.progress_lines(event, 100)
            self.assertEqual(len(lines), 2)
            self.assertIn(f"{completed}/50", lines[0])
            self.assertIn(f"{step}/10", lines[1])
            console.update(event)
            self.assertEqual(console.height, 2)
        console.update(dict(phase="test", protocol="imp", unit="rows", done=5, total=20))
        self.assertEqual(console.height, 1)
        event = dict(phase="test", protocol="total", unit="rows", done=0, total=100,
                     scored=10, score_total=50)
        lines = progress.progress_lines(event, 100)
        self.assertEqual(len(lines), 1)
        self.assertIn("0/100", lines[0])
        self.assertIn("chunk scores 10/50", lines[0])
        console.update({"phase": "idle"})
        self.assertEqual(console.height, 0)
        self.assertIn("\x1b[1A", stream.getvalue())

    def test_noninteractive_hour_is_quiet_even_across_phase_changes(self):
        stream = io.StringIO()
        console = progress._Console(stream)
        event = dict(phase="test", protocol="IMP", unit="rows", done=1, total=100)
        with patch.object(progress.time, "monotonic", side_effect=[0, 300, 3599, 3600]):
            for protocol in ("IMP", "IMP", "RS30", "RS30"):
                console.update(dict(event, protocol=protocol), "NAML")
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("Test IMP", lines[0])
        self.assertIn("Test RS30", lines[1])
        self.assertNotIn("\x1b", stream.getvalue())

    def test_current_score_percentage_is_separate_from_ranked_rows(self):
        event = dict(phase="test", protocol="total", unit="rows", done=5000, total=20000,
                     scored=400, score_total=800)
        line = progress.progress_lines(event, 140)[0]
        self.assertIn("25.0%", line)
        self.assertIn("5,000/20,000", line)
        self.assertIn("chunk scores 400/800 (50.0%)", line)

    def test_short_redirected_batch_announces_and_flushes_actual_snapshot(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            env = {}
            with progress.batch_progress(env, "DIGAT / rs50 / total"):
                snapshot = dict(phase="test", protocol="total", unit="rows", done=30, total=100,
                                scored=4, score_total=8)
                Path(env[progress._FILE_ENV]).write_text(json.dumps(snapshot), encoding="utf-8")
        self.assertIn("Running DIGAT / rs50 / total", stream.getvalue())
        self.assertIn("30/100", stream.getvalue())
        self.assertIn("chunk scores 4/8 (50.0%)", stream.getvalue())
        self.assertNotIn("100/100", stream.getvalue())

    def test_redirected_first_measurement_does_not_wait_one_hour(self):
        observed = threading.Event()

        class Stream(io.StringIO):
            def write(self, text):
                result = super().write(text)
                if "Test total" in text:
                    observed.set()
                return result

        stream = Stream()
        with redirect_stdout(stream), patch.object(progress, "REFRESH_SECONDS", 0.01):
            env = {}
            with progress.batch_progress(env, "DIGAT"):
                snapshot = dict(phase="test", protocol="total", unit="rows", done=5, total=100)
                Path(env[progress._FILE_ENV]).write_text(json.dumps(snapshot), encoding="utf-8")
                self.assertTrue(observed.wait(2), "first measured status was deferred to the hourly interval")
        self.assertIn("5/100", stream.getvalue())

    def test_supervised_child_keeps_stdout_log_and_reports_across_processes(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SCRIPTS)
        before = env.copy()
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "worker.log"
            with progress.batch_progress(env, "DIGAT"):
                path = Path(env[progress._FILE_ENV])
                with log.open("w", encoding="utf-8") as stream:
                    child = subprocess.run([sys.executable, "-B", "-c",
                        "from experiment_progress import test_progress; "
                        "test_progress('total', 5000, 111383, scored=256, score_total=900); "
                        "print('worker log')"], env=env, stdout=stream, check=True)
                self.assertEqual(child.returncode, 0)
                self.assertEqual(log.read_text(encoding="utf-8").strip(), "worker log")
                snapshot = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual((snapshot["protocol"], snapshot["done"], snapshot["scored"]),
                                 ("total", 5000, 256))
            self.assertFalse(path.exists())
        self.assertEqual(env, before)


if __name__ == "__main__":
    unittest.main()
