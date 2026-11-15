"""Freeze runtime source bytes and execute DIGAT from the verified snapshot."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import sys

SCRIPTS = (
    "run_small_digat.py", "digat_source_snapshot.py", "digat_small_assets.py",
    "digat_cached_eval.py", "digat_projection_cache.py", "digat_eval_benchmark.py",
    "digat_sparse_attention.py",
    "digat_run.py", "digat_prepare.py", "nnr_artifacts.py", "naml_cached_eval.py",
    "model_batch_defaults.py", "experiment_progress.py", "evaluation_progress.py",
    "training_validation.py", "prepare_development_candidates.py", "scheduled_cooling.py",
    "candidate_policy.py", "consistent_candidate_profile.py", "context_feature_policy.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def freeze(root, run_dir):
    started_unix = time.time()
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    if run_dir.exists():
        raise ValueError("Run directory exists; preserve it and choose a new run")
    paths = [root / "scripts" / name for name in SCRIPTS]
    paths += sorted(path for path in (root / "DIGAT").iterdir() if path.is_file())
    blobs = {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}
    # Detect concurrent changes while reading the source set; never mix versions silently.
    for name, data in blobs.items():
        if (root / name).read_bytes() != data:
            raise ValueError("Source changed during snapshot: " + name)
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, data in blobs.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if target.read_bytes() != data:
            raise ValueError("Snapshot write mismatch: " + name)
        files[name] = {"bytes": len(data), "sha256": digest(data)}
    manifest = {"schema": 1, "started_unix": started_unix, "original_root": str(root), "snapshot_root": str(snapshot),
                "runtime_executes_snapshot": True, "files": files}
    (snapshot / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify(snapshot, run_dir):
    snapshot, run_dir = Path(snapshot).resolve(), Path(run_dir).resolve()
    if snapshot != run_dir / "source_snapshot" or (run_dir / "experiment.json").exists():
        raise ValueError("Invalid snapshot execution location or existing experiment")
    manifest = json.loads((snapshot / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    if Path(manifest["snapshot_root"]).resolve() != snapshot or not manifest["runtime_executes_snapshot"]:
        raise ValueError("Source snapshot identity mismatch")
    for name, info in manifest["files"].items():
        path = (snapshot / name).resolve()
        path.relative_to(snapshot)
        data = path.read_bytes()
        if len(data) != info["bytes"] or digest(data) != info["sha256"]:
            raise ValueError("Frozen runtime source changed: " + name)
    return manifest


def execute(root, run_dir, argv):
    manifest = freeze(root, run_dir)
    child = Path(manifest["snapshot_root"]) / "scripts/run_small_digat.py"
    return subprocess.call([sys.executable, str(child)] + list(argv) + ["--execute-snapshot"])
