"""Shared DIGAT asset-copy and verified evaluation-backend utilities."""
import argparse
import builtins
import gc
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DIGAT"))
from digat_small_assets import config_for, identity
from digat_run import verify_upstream
from nnr_artifacts import check_inputs, fingerprint


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cuda_memory_snapshot(torch):
    """Synchronize phase boundaries and record live versus reserved CUDA bytes."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "free_bytes": free, "total_bytes": total}


def verified_evaluator(factory, behavior_path, *, sparse_eval, projection_cache,
                       score_batch_size, rejection_policy, fallback_path, release=None):
    """Only an explicit score/rank rejection may select fresh dense256 evaluation."""
    from digat_sparse_attention import SparseEquivalenceError
    if rejection_policy not in ("abort", "dense256"):
        raise ValueError("Unknown sparse rejection policy")
    ev = factory(sparse_eval=sparse_eval, projection_cache=projection_cache,
                 score_batch_size=score_batch_size)
    try:
        verification = ev.verify_against_forward(behavior_path)
        return ev, verification, None
    except SparseEquivalenceError as exc:
        if not sparse_eval or rejection_policy != "dense256":
            raise
        # Changed weights/corpus or missing evidence must stop, not authorize fallback.
        ev._assert_current()
        fallback = {"status": "dense_verification_pending", "policy": rejection_policy,
            "reason": str(exc), "rejected_cache_identity": ev.identity,
            "model_state_sha256": ev.model_hash,
            "failed_gate_evidence": fingerprint(Path(exc.evidence_path)),
            "selected_evaluation": {"graph_backend": "dense", "batch_size": 256,
                                    "projection_cache": False},
            "retraining": False, "tolerance_changed": False}
        save(fallback_path, fallback)
    # The sparse wrapper never replaces any model method. Release it before making
    # a fresh dense evaluator with a separate backend/projection cache identity.
    del ev
    gc.collect()
    if release is not None:
        release()
    print(json.dumps({"stage": "sparse_rejection_dense256", **fallback}), flush=True)
    ev = factory(sparse_eval=False, projection_cache=False, score_batch_size=256)
    if ev.model_hash != fallback["model_state_sha256"]:
        raise ValueError("Checkpoint changed before dense fallback")
    if ev.identity == fallback["rejected_cache_identity"]:
        raise ValueError("Dense fallback must use a distinct cache identity")
    verification = ev.verify_against_forward(behavior_path)
    fallback.update(status="dense_verified", dense_cache_identity=ev.identity,
                    dense_verification=verification)
    save(fallback_path, fallback)
    return ev, verification, fallback


def copy_shared_assets(shared, data_root, embedding_root, run_dir, seed):
    info = json.loads((shared / "asset_prep.json").read_text(encoding="utf-8"))
    if info["status"] != "complete" or identity(info["spec"]) != info["identity"]:
        raise ValueError("Shared DIGAT assets are incomplete or have invalid provenance")
    spec = info["spec"]
    if seed != spec["seed"] or verify_upstream() != spec["upstream"]:
        raise ValueError("Seed or official source differs from shared preparation")
    for split in ("train", "dev", "test"):
        if fingerprint(data_root / split / "news.tsv")["sha256"] != spec["news"][split]["sha256"]:
            raise ValueError("News membership/order differs from semantic graph: " + split)
    if fingerprint(embedding_root / "glove.6B.300d.txt")["sha256"] != spec["glove"]["sha256"]:
        raise ValueError("GloVe 6B source differs from shared preparation")
    required = ("news_ID-MIND-small.json", "category-MIND-small.json", "subCategory-MIND-small.json",
                "vocabulary-3-32-MIND-small.json", "word_embedding-3-300-32-MIND-small.pkl",
                "news_graph-2-5-MIND-small.pkl")
    copied = {}
    for name in required:
        source = shared / name
        expected = info["files"][name]["sha256"]
        if fingerprint(source)["sha256"] != expected:
            raise ValueError("Shared cache hash mismatch: " + name)
        shutil.copyfile(source, run_dir / name)
        copied[name] = fingerprint(run_dir / name)
        if copied[name]["sha256"] != expected:
            raise ValueError("Shared cache copy mismatch")
    return info, copied




