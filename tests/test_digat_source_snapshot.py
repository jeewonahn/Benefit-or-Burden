import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from digat_source_snapshot import SCRIPTS, freeze, verify
from evaluate_existing_digat import frozen_model_directory


class SnapshotTests(unittest.TestCase):
    def test_snapshot_keeps_actual_bytes_and_rejects_tamper_or_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, run = base / "repo", base / "run"
            (source / "scripts").mkdir(parents=True)
            (source / "DIGAT").mkdir(parents=True)
            for name in SCRIPTS:
                (source / "scripts" / name).write_bytes(("# " + name + "\r\n").encode())
            (source / "DIGAT/model.py").write_bytes(b"# official model\r\n")
            metadata = freeze(source, run)
            snapshot = run / "source_snapshot"
            self.assertEqual(metadata, verify(snapshot, run))
            self.assertTrue(metadata["runtime_executes_snapshot"])
            self.assertIn("scripts/training_validation.py", metadata["files"])
            self.assertIn("scripts/prepare_development_candidates.py", metadata["files"])
            self.assertEqual((snapshot / "DIGAT/model.py").read_bytes(), b"# official model\r\n")
            with self.assertRaises(ValueError):
                freeze(source, run)
            (snapshot / "scripts/run_small_digat.py").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                verify(snapshot, run)

    def test_evaluation_loads_model_from_verified_snapshot_layout(self):
        snapshot = Path("run/source_snapshot")
        for folder in ("DIGAT", "third_party/DIGAT"):
            with self.subTest(folder=folder):
                model = folder + "/model.py"
                verified_files = {model: {"path": str(snapshot / model)}}
                self.assertEqual(frozen_model_directory(verified_files), snapshot / folder)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            frozen_model_directory({})

    def test_snapshot_imports_artifact_policy_without_original_source_path(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run = base / "run"
            metadata = freeze(root, run)
            snapshot = Path(metadata["snapshot_root"])
            isolated = base / "isolated"
            isolated.mkdir()
            code = "\n".join([
                "import json, sys",
                "from pathlib import Path",
                "snapshot = Path(sys.argv[1]).resolve()",
                "sys.path.insert(0, str(snapshot / 'scripts'))",
                "import nnr_artifacts",
                "import context_feature_policy",
                "modules = (nnr_artifacts, context_feature_policy)",
                "paths = [str(Path(module.__file__).resolve().relative_to(snapshot)) for module in modules]",
                "assert not any(name in sys.modules for name in ('torch', 'numpy'))",
                "print(json.dumps({'paths': paths, 'models_imported': False}))",
            ])
            # -I ignores the original checkout supplied through PYTHONPATH;
            # -S also prevents installed site packages from satisfying imports.
            environment = dict(os.environ, PYTHONPATH=str(root / "scripts"),
                               CUDA_VISIBLE_DEVICES="-1")
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-c", code, str(snapshot)],
                cwd=str(isolated), env=environment, capture_output=True,
                text=True, encoding="utf-8", timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            proof = json.loads(completed.stdout)
            self.assertEqual(set(proof["paths"]), {
                str(Path("scripts/nnr_artifacts.py")),
                str(Path("scripts/context_feature_policy.py")),
            })
            self.assertFalse(proof["models_imported"])
            self.assertIn("scripts/context_feature_policy.py", metadata["files"])
            self.assertEqual(
                (snapshot / "scripts/context_feature_policy.py").read_bytes(),
                (root / "scripts/context_feature_policy.py").read_bytes())

if __name__ == "__main__":
    unittest.main()
