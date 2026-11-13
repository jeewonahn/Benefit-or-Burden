import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from digat_source_snapshot import SCRIPTS, freeze, verify


class SnapshotTests(unittest.TestCase):
    def test_snapshot_keeps_actual_bytes_and_rejects_tamper_or_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, run = base / "repo", base / "run"
            (source / "scripts").mkdir(parents=True)
            (source / "third_party/DIGAT").mkdir(parents=True)
            for name in SCRIPTS:
                (source / "scripts" / name).write_bytes(("# " + name + "\r\n").encode())
            (source / "third_party/DIGAT/model.py").write_bytes(b"# official model\r\n")
            metadata = freeze(source, run)
            snapshot = run / "source_snapshot"
            self.assertEqual(metadata, verify(snapshot, run))
            self.assertTrue(metadata["runtime_executes_snapshot"])
            self.assertIn("scripts/training_validation.py", metadata["files"])
            self.assertIn("scripts/prepare_development_candidates.py", metadata["files"])
            self.assertEqual((snapshot / "third_party/DIGAT/model.py").read_bytes(), b"# official model\r\n")
            with self.assertRaises(ValueError):
                freeze(source, run)
            (snapshot / "scripts/run_small_digat.py").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                verify(snapshot, run)

if __name__ == "__main__":
    unittest.main()
