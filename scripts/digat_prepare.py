#!/usr/bin/env python
"""DIGAT behavior-file staging and candidate-profile validation utilities."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

TRAIN_TYPES = ("imp", "rs50")
EVAL_TYPES = ("imp", "tt", "rs30", "rs50", "rs60", "rs90", "rs120", "rs150")
DATASETS = ("small", "200k")
SCHEMA_VERSION = 1
MATCHED_PROFILE = "request_time_history_and_current_positives_v1"
MATCHED_CERTIFICATE = "matched_rands50_manifest.json"

from training_validation import resolve_dev_protocol, validate_candidate_profile


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def behavior_name(protocol):
    return "behaviors.tsv" if protocol == "imp" else "behaviors_" + protocol + ".tsv"


def matched_candidate_profile(data_root, train_type):
    """Require certified request-time RandS50 inputs before preparing or training."""
    return validate_candidate_profile(Path(data_root).resolve(), train_type)


def validate_prepared_candidate_profile(prepared, manifest):
    """Bind the copied certificate to the actual staged train/dev bytes."""
    if manifest["protocols"]["train"] == "imp":
        return
    proof = manifest.get("matched_candidate_profile")
    if not proof or proof.get("relative_path") != MATCHED_CERTIFICATE:
        raise ValueError("Matched RandS candidate certificate is required; prepare a new input directory")
    certificate = Path(prepared) / MATCHED_CERTIFICATE
    if not certificate.is_file() or sha256_file(certificate) != proof.get("sha256"):
        raise ValueError("Prepared RandS candidate certificate is missing or changed")
    record = json.loads(certificate.read_text(encoding="utf-8-sig"))
    if record.get("schema") != 1 or record.get("status") != "complete" or record.get("profile") != MATCHED_PROFILE:
        raise ValueError("Unsupported matched RandS candidate profile")
    from prepare_development_candidates import RECIPE
    if (type(record.get("seed")) is not int or not 0 <= record["seed"] < 2**32
            or record.get("sampling") != RECIPE or set(record.get("outputs", {})) != {"train", "dev"}):
        raise ValueError("Prepared matched RandS sampling recipe differs")
    for name in ("publication_proxy", "source_dataset_manifest"):
        evidence = record.get(name, {})
        digest = evidence.get("sha256", "")
        if (type(evidence.get("bytes")) is not int or evidence["bytes"] < 0
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Prepared matched RandS source evidence is incomplete")
    for split in ("train", "dev"):
        expected = record["outputs"][split]
        if expected["path"] != split + "/behaviors_rs50.tsv":
            raise ValueError("Matched RandS certificate has an unexpected split path")
        path = Path(prepared) / manifest["files"][split]["behaviors"]["relative_path"]
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError("Prepared behavior bytes differ from the matched RandS certificate")


def inspect_inputs(files):
    """Validate all rows without changing candidate order, histories or labels."""
    news = {}
    summaries = {}
    for split in ("train", "dev", "test"):
        path = Path(files[split]["news"])
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != 8:
                    raise ValueError(f"{path}:{line_no}: expected 8 news columns")
                news_id, category = fields[:2]
                if not news_id or any(c.isspace() for c in news_id):
                    raise ValueError(f"{path}:{line_no}: invalid news ID")
                if not category or category in (".", "..") or any(c in category for c in '/\\'):
                    raise ValueError(f"{path}:{line_no}: invalid category for upstream SAG filenames")
                fingerprint = hashlib.sha256("\t".join(fields).encode("utf-8")).hexdigest()
                if news_id in news and news[news_id] != fingerprint:
                    raise ValueError(f"{path}:{line_no}: conflicting metadata for {news_id}")
                news[news_id] = fingerprint
                count += 1
        if count == 0:
            raise ValueError(f"{path}: empty news file")
        summaries[split] = {"news_rows": count}

    sessions = {}
    for split in ("train", "dev", "test"):
        path = Path(files[split]["behaviors"])
        rows = positives = negatives = candidates = 0
        session_keys = set()
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != 5:
                    raise ValueError(f"{path}:{line_no}: expected 5 behavior columns")
                impression_id, user_id, timestamp, history, impressions = fields
                if not impression_id or not user_id or not timestamp:
                    raise ValueError(f"{path}:{line_no}: empty ID/user/timestamp")
                history_ids = history.split(" ") if history else []
                if any(not item or item not in news for item in history_ids):
                    raise ValueError(f"{path}:{line_no}: unknown/empty history news")
                row_labels = []
                candidate_ids = set()
                for item in impressions.split(" "):
                    try:
                        news_id, label = item.rsplit("-", 1)
                    except ValueError:
                        raise ValueError(f"{path}:{line_no}: every candidate must have a binary label") from None
                    if news_id not in news or label not in ("0", "1"):
                        raise ValueError(f"{path}:{line_no}: unknown news or nonbinary label: {item}")
                    if news_id in candidate_ids:
                        raise ValueError(f"{path}:{line_no}: duplicate candidate news ID: {news_id}")
                    candidate_ids.add(news_id)
                    row_labels.append(int(label))
                if set(row_labels) != {0, 1}:
                    raise ValueError(
                        f"{path}:{line_no}: positive and negative candidates both required; "
                        "rows are never silently skipped (sampling/AUC would be undefined)"
                    )
                # Candidate protocols/impression IDs may differ for the same
                # event. Compare the shared user/time/history identity instead.
                session_keys.add(hashlib.sha256(
                    "\t".join((user_id, timestamp, history)).encode("utf-8")
                ).digest())
                rows += 1
                positives += sum(row_labels)
                negatives += len(row_labels) - sum(row_labels)
                candidates += len(row_labels)
        if not rows:
            raise ValueError(f"{path}: empty behaviors file")
        sessions[split] = session_keys
        summaries[split].update(
            behavior_rows=rows, candidate_count=candidates,
            positive_count=positives, negative_count=negatives,
        )
    overlap = len(sessions["dev"] & sessions["test"])
    if overlap:
        raise ValueError(
            f"Dev/test share {overlap} user/time/history fingerprints; "
            "resolve split provenance before running; rows are not silently removed"
        )
    return summaries


def prepare(dataset, data_root, train_type, dev_type, test_type, output):
    dev_type = resolve_dev_protocol(train_type, dev_type)
    if dataset not in DATASETS or train_type not in TRAIN_TYPES or any(
        protocol not in EVAL_TYPES for protocol in (dev_type, test_type)
    ):
        raise ValueError("Unsupported dataset or train/dev/test protocol")
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output must be a new directory: {output}")
    if output == data_root or data_root in output.parents:
        raise ValueError("Prepared output must be outside the source dataset directory")
    protocols = dict(train=train_type, dev=dev_type, test=test_type)
    files = {
        split: {"news": data_root / split / "news.tsv",
                "behaviors": data_root / split / behavior_name(protocols[split])}
        for split in protocols
    }
    for split in files:
        for path in files[split].values():
            if not path.is_file():
                raise FileNotFoundError(f"Required prepared TORS input is missing: {path}")
    summaries = inspect_inputs(files)
    candidate_profile = matched_candidate_profile(data_root, train_type)
    original_hashes = {
        (split, kind): sha256_file(path)
        for split, entries in files.items() for kind, path in entries.items()
    }
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "dataset_display_name": "MIND-Small" if dataset == "small" else "MIND-200K",
        "protocols": protocols,
        "candidate_generation": "not performed; pre-existing TORS TSVs copied byte-for-byte",
        "candidate_generation_seed": None,
        "candidate_generation_seed_status": "unknown unless recovered from original preprocessing records",
        "historical_R2_DIGAT_identity": "unverified",
        "dev_test_session_overlap": 0,
        "validation_policy": "training_matched",
        "files": {},
        "summaries": summaries,
    }
    for split, entries in files.items():
        destination = output / "dataset" / split
        destination.mkdir(parents=True)
        manifest["files"][split] = {}
        for kind, source in entries.items():
            target = destination / ("news.tsv" if kind == "news" else "behaviors.tsv")
            shutil.copyfile(source, target)
            digest = sha256_file(target)
            if digest != original_hashes[(split, kind)] or sha256_file(source) != digest:
                raise RuntimeError(f"Input changed while staging: {source}; incomplete output: {output}")
            manifest["files"][split][kind] = {
                "source": str(source), "relative_path": target.relative_to(output).as_posix(),
                "sha256": digest, "bytes": target.stat().st_size,
            }
    if candidate_profile is not None:
        certificate = output / MATCHED_CERTIFICATE
        shutil.copyfile(candidate_profile["path"], certificate)
        if sha256_file(certificate) != candidate_profile["sha256"]:
            raise RuntimeError("Matched candidate certificate changed while staging")
        manifest["matched_candidate_profile"] = {
            "relative_path": MATCHED_CERTIFICATE, "sha256": candidate_profile["sha256"],
            "profile": candidate_profile["profile"], "seed": candidate_profile["seed"],
        }
    validate_prepared_candidate_profile(output, manifest)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def validate_prepared(prepared):
    prepared = Path(prepared).resolve()
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("dataset") not in DATASETS:
        raise ValueError("Unsupported prepared manifest")
    protocols = manifest["protocols"]
    resolve_dev_protocol(protocols["train"], protocols["dev"])
    if protocols["train"] not in TRAIN_TYPES or any(protocols[s] not in EVAL_TYPES for s in ("dev", "test")):
        raise ValueError("Unsupported protocol in manifest")
    source_dev_name = Path(manifest["files"]["dev"]["behaviors"]["source"]).name
    if source_dev_name != behavior_name(protocols["dev"]):
        raise ValueError("Prepared dev source name does not match its declared candidate protocol")
    files = {}
    for split in ("train", "dev", "test"):
        files[split] = {}
        for kind, expected_name in (("news", "news.tsv"), ("behaviors", "behaviors.tsv")):
            record = manifest["files"][split][kind]
            expected = Path("dataset") / split / expected_name
            if Path(record["relative_path"]) != expected:
                raise ValueError("Prepared manifest must use its fixed dataset/split layout")
            path = (prepared / expected).resolve()
            if prepared not in path.parents:
                raise ValueError("Prepared file resolves outside its directory")
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Prepared input missing or modified: {path}")
            files[split][kind] = path
    summaries = inspect_inputs(files)
    if summaries != manifest["summaries"]:
        raise ValueError("Prepared data statistics differ from manifest")
    validate_prepared_candidate_profile(prepared, manifest)
    return manifest




