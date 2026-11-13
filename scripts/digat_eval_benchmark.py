"""One bounded off/on and batch-size sweep for the same DIGAT checkpoint."""
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from digat_cached_eval import DigatCachedEvaluator
from naml_cached_eval import _file_hash, _save_json, model_fingerprint


def paired_inputs(evaluator, behavior_path, max_rows=32, max_candidates=128):
    keys, chunks, offsets = [], [], [0]
    with Path(behavior_path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            _, user, _, text, candidates = line.rstrip("\r\n").split("\t")
            keys.append(evaluator._key(user, evaluator._history(text)))
            news = [evaluator.corpus.news_ID_dict[item.rsplit("-", 1)[0]]
                    for item in candidates.split()[:max_candidates]]
            chunks.append(news); offsets.append(offsets[-1] + len(news))
            if index + 1 >= max_rows: break
    users = evaluator._ensure_users(keys)
    flat_users = np.concatenate([np.full(len(news), users[i], dtype=np.int64) for i, news in enumerate(chunks)])
    flat_news = np.asarray([item for news in chunks for item in news], dtype=np.int64)
    return flat_users, flat_news, offsets


def benchmark(model, corpus, categories, cache_root, behavior_path, output,
              batch_sizes=(64, 256, 512), atol=1e-4, rtol=1e-5):
    if next(model.parameters()).device.type != "cuda":
        raise ValueError("The performance sweep requires an exclusive CUDA slot")
    output = Path(output)
    if output.exists():
        raise ValueError("Benchmark output exists; preserve it")
    model.eval()
    reference = None
    report = {"schema": 1, "model_state_sha256": model_fingerprint(model),
              "input_sha256": _file_hash(behavior_path), "variants": [],
              "historical_R2_reproduction": False,
              "scope": "One bounded sweep; not full-evaluation accuracy or a large-run time measurement",
              "atol": atol, "rtol": rtol, "selection_requires_identical_stable_ranks": True,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    all_start = time.monotonic()
    for projection in (False, True):
        begin = time.monotonic()
        ev = DigatCachedEvaluator(model, corpus, categories, cache_root,
                                  news_batch_size=256, score_batch_size=64,
                                  projection_cache=projection)
        construction = time.monotonic() - begin
        try:
            verification = ev.verify_against_forward(behavior_path, max_entries=8, max_candidates=16)
        except ValueError as error:
            if not projection:
                raise
            report["variants"].append({"projection_cache": True, "batch_size": None,
                                       "status": "rejected_native_equivalence", "error": str(error)})
            del ev; gc.collect(); torch.cuda.empty_cache()
            continue
        users, news, offsets = paired_inputs(ev, behavior_path)
        report["rows"], report["pairs"] = len(offsets) - 1, len(news)
        prior_oom = False
        for size in batch_sizes:
            entry = {"projection_cache": projection, "batch_size": int(size),
                     "cache_construction_seconds": construction,
                     "cache_encoding_timings": dict(ev.timings),
                     "projection_bytes": ev.projections.bytes if ev.projections else 0,
                     "native_verification": verification}
            if prior_oom:
                entry.update(status="skipped_after_smaller_batch_oom")
                report["variants"].append(entry)
                continue
            ev.score_batch_size = int(size)
            try:
                ev._compute_pairs(users[:min(size, len(users))], news[:min(size, len(news))])
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                free, total = torch.cuda.mem_get_info()
                started = time.monotonic()
                scores = ev._compute_pairs(users, news)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                if reference is None:
                    reference = scores.copy()
                maximum = float(np.max(np.abs(scores - reference)))
                equal = bool(np.allclose(scores, reference, atol=atol, rtol=rtol))
                rank_disagreements = 0
                for start, end in zip(offsets[:-1], offsets[1:]):
                    if not np.array_equal(np.argsort(-scores[start:end], kind="stable"),
                                          np.argsort(-reference[start:end], kind="stable")):
                        rank_disagreements += 1
                entry.update(status="passed" if equal and rank_disagreements == 0 else "rejected_equivalence",
                             seconds=elapsed, pairs_per_second=len(news) / elapsed,
                             max_absolute_error=maximum, allclose=equal,
                             rows_with_rank_disagreement=rank_disagreements,
                             peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                             cuda_free_bytes_before=free, cuda_total_bytes=total)
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                prior_oom = True
                entry.update(status="out_of_memory", error=str(error))
            report["variants"].append(entry)
            _save_json(output, dict(report, status="running"))
            gc.collect(); torch.cuda.empty_cache()
        del ev; gc.collect(); torch.cuda.empty_cache()
    passed = [entry for entry in report["variants"] if entry["status"] == "passed"]
    if not passed:
        raise RuntimeError("No equivalent DIGAT evaluation configuration passed")
    selected = max(passed, key=lambda entry: entry["pairs_per_second"])
    report["selected"] = {key: selected[key] for key in
                          ("projection_cache", "batch_size", "pairs_per_second", "peak_cuda_allocated_bytes")}
    report["status"] = "completed"
    report["elapsed_seconds"] = time.monotonic() - all_start
    _save_json(output, report)
    return report
