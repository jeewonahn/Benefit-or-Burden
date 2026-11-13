"""Prepare matched RandS50 train/dev candidate bundles without test inputs.

This is a new request-time profile. It does not reuse or relabel the legacy
closed-period RandS candidates, whose exclusions include future test clicks.
The output is a candidate bundle, not a complete dataset. Preserve the source
dataset and assemble a separate run dataset with its unchanged IMP/news inputs.
"""
import argparse
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

PROFILE = "request_time_history_and_current_positives_v1"
MANIFEST_NAME = "matched_rands50_manifest.json"
TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
RECIPE = {
    "negative_pool_size": 50,
    "window_hours": 24,
    "window_endpoints": "inclusive",
    "exclusion": "request_history_and_current_positives",
    "test_inputs_used": False,
    "rng_stream": "{seed}:matched_rs50:{split}:{source_row}:{positive_index}",
}


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise ValueError("Missing or invalid candidate provenance: " + str(path)) from error


def _expected_fingerprint(value, label):
    if not isinstance(value, dict) or not isinstance(value.get("sha256"), str) or len(value["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in value["sha256"]):
        raise ValueError("Missing SHA-256 evidence for " + label)
    if type(value.get("bytes")) is not int or value["bytes"] < 0:
        raise ValueError("Missing file-size evidence for " + label)
    return value


def _verify(path, expected, label):
    expected = _expected_fingerprint(expected, label)
    try:
        actual = fingerprint(path)
    except OSError as error:
        raise ValueError("Required input is absent: " + label) from error
    if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
        raise ValueError("File differs from candidate provenance: " + label)
    return actual


def validate_matched_candidate_profile(data_root):
    """Verify both matched files and their held-out IMP input identities.

    Returns a certificate fingerprint plus its profile, seed and outputs.
    It never reads test data, initializes a model, or repairs legacy files.
    """
    root = Path(data_root).resolve()
    if (root / "consistent_candidate_manifest.json").is_file():
        from consistent_candidate_profile import validate_consistent_candidate_profile
        return validate_consistent_candidate_profile(root)
    path = root / MANIFEST_NAME
    certificate = _read_json(path)
    if (type(certificate.get("schema")) is not int or certificate.get("schema") != 1 or certificate.get("status") != "complete"
            or certificate.get("profile") != PROFILE):
        raise ValueError("A complete matched RandS50 train/dev candidate profile is required")
    if (type(certificate.get("seed")) is not int or certificate.get("sampling") != RECIPE
            or certificate.get("sampling", {}).get("test_inputs_used") is not False):
        raise ValueError("Matched RandS50 sampling recipe differs")
    _expected_fingerprint(certificate.get("publication_proxy"), "publication proxy")
    _expected_fingerprint(certificate.get("source_dataset_manifest"), "source dataset manifest")
    inputs = certificate.get("input_files", {})
    outputs = certificate.get("outputs", {})
    if set(outputs) != {"train", "dev"}:
        raise ValueError("Both matched RandS50 train and dev outputs are required")
    for split in ("train", "dev"):
        relative = split + "/behaviors_rs50.tsv"
        item = outputs.get(split, {})
        if item.get("path") != relative:
            raise ValueError("Matched RandS50 output escaped its prescribed split")
        _verify(root / relative, item, relative)
        relative_imp = split + "/behaviors.tsv"
        _verify(root / relative_imp, inputs.get(relative_imp), relative_imp)
    result = fingerprint(path)
    result.update(profile=PROFILE, seed=certificate["seed"], outputs=outputs)
    return result


def read_imp(path):
    records, groups = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for index, raw in enumerate(stream, 1):
            parts = raw.rstrip("\r\n").split("\t")
            if len(parts) != 5:
                raise ValueError("Expected five IMP behavior fields")
            datetime.strptime(parts[2], TIME_FORMAT)
            values = []
            for token in parts[4].split():
                news, label = token.rsplit("-", 1)
                if label not in ("0", "1") or not news:
                    raise ValueError("Invalid IMP candidate")
                values.append((news, int(label)))
            if len({news for news, _ in values}) != len(values) or {label for _, label in values} != {0, 1}:
                raise ValueError("IMP source requires unique candidates and both labels")
            key = (parts[1], parts[2], " ".join(parts[3].split()))
            groups.add(key)
            records.append((index, parts, values))
    if not records:
        raise ValueError("Empty IMP source")
    return records, groups


def load_news_ids(path):
    ids = set()
    with Path(path).open(encoding="utf-8") as stream:
        for row in stream:
            news = row.split("\t", 1)[0].rstrip("\r\n")
            if not news or news in ids:
                raise ValueError("Missing or duplicate news metadata ID")
            ids.add(news)
    return ids


class PublicationIndex:
    def __init__(self, path):
        publication = {}
        with Path(path).open(encoding="utf-8") as stream:
            for row in stream:
                news, stamp = row.rstrip("\r\n").split("\t", 1)
                current = datetime.strptime(stamp, TIME_FORMAT)
                if news in publication and publication[news] != current:
                    raise ValueError("Conflicting publication proxy timestamps")
                publication[news] = current
        if not publication:
            raise ValueError("Empty publication proxy")
        self.ids = sorted(publication, key=lambda n: (publication[n], n))
        self.times = [publication[n] for n in self.ids]

    def eligible(self, parts, positives, news_ids):
        current = datetime.strptime(parts[2], TIME_FORMAT)
        left = bisect_left(self.times, current - timedelta(hours=24))
        right = bisect_right(self.times, current)
        # Do not aggregate later train/dev/test clicks by user. Only information
        # contained in this request is used to exclude negative candidates.
        excluded = set(parts[3].split()) | set(positives)
        pool = [news for news in self.ids[left:right] if news not in excluded]
        if any(news not in news_ids for news in pool):
            raise ValueError("Eligible proxy candidate has no train/dev news metadata")
        if not pool:
            raise ValueError("No eligible RandS50 negatives for this request")
        return pool


def write_split(records, index, news_ids, target, split, seed):
    stats = dict(rows=0, source_rows=len(records), candidates=0, positive_labels=0,
                 negative_min=None, negative_max=0)
    with Path(target).open("x", encoding="utf-8", newline="\n") as output:
        for row_index, parts, values in records:
            positives = [news for news, label in values if label]
            if not (set(parts[3].split()) | set(positives)) <= news_ids:
                raise ValueError("History or positive article has no train/dev metadata")
            pool = index.eligible(parts, positives, news_ids)
            for positive_index, positive in enumerate(positives):
                rng = random.Random(f"{seed}:matched_rs50:{split}:{row_index}:{positive_index}")
                chosen = rng.sample(pool, min(50, len(pool)))
                candidates = [positive + "-1"] + [news + "-0" for news in chosen]
                identifier = f"{parts[0]}_rs50_{positive_index}"
                output.write("\t".join([identifier] + parts[1:4] + [" ".join(candidates)]) + "\n")
                stats["rows"] += 1
                stats["candidates"] += len(candidates)
                stats["positive_labels"] += 1
                stats["negative_min"] = len(chosen) if stats["negative_min"] is None else min(stats["negative_min"], len(chosen))
                stats["negative_max"] = max(stats["negative_max"], len(chosen))
    return stats


def prepare(args):
    """Generate a new atomic candidate bundle from verified IMP train/dev only."""
    root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Output already exists; no file will be overwritten")
    if output == root or root in output.parents:
        raise ValueError("Output must be outside the immutable input dataset")
    if type(args.seed) is not int or args.split not in ("train", "dev", "both"):
        raise ValueError("Invalid seed or selected split")
    manifest_path = root / "manifest.json"
    source = _read_json(manifest_path)
    if source.get("status") != "complete":
        raise ValueError("A complete source dataset manifest is required")
    historical_root = Path(source["source"]).resolve() if isinstance(source.get("source"), str) else None
    if historical_root and (output == historical_root or historical_root in output.parents):
        raise ValueError("Output must also be outside the immutable historical source")
    source_fp = fingerprint(manifest_path)
    inputs, records, groups, metadata = {}, {}, {}, set()
    checked = [(manifest_path, source_fp)]
    for split in ("train", "dev"):
        relative = split + "/behaviors.tsv"
        expected = source.get("output_files", {}).get(relative)
        inputs[relative] = _verify(root / relative, expected, relative)
        checked.append((root / relative, inputs[relative]))
        records[split], groups[split] = read_imp(root / relative)
        news_relative = split + "/news.tsv"
        expected_news = source.get("assets", {}).get(news_relative, {}).get("output")
        news_fp = _verify(root / news_relative, expected_news, news_relative)
        checked.append((root / news_relative, news_fp))
        metadata.update(load_news_ids(root / news_relative))
    if groups["train"] & groups["dev"]:
        raise ValueError("Train/dev request groups overlap")
    proxy_evidence = source.get("RandS_training_construction", {}).get("publication_proxy")
    _expected_fingerprint(proxy_evidence, "publication proxy")
    if not proxy_evidence.get("path"):
        raise ValueError("Publication proxy path is missing; test data is not a fallback")
    proxy_path = Path(proxy_evidence["path"]).resolve()
    if any(part.casefold() == "test" for part in proxy_path.parts):
        raise ValueError("Test-directory inputs are not permitted as publication proxies")
    proxy_fp = _verify(proxy_path, proxy_evidence, "publication proxy")
    checked.append((proxy_path, proxy_fp))
    index = PublicationIndex(proxy_path)
    selected = ("train", "dev") if args.split == "both" else (args.split,)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".matched-rands50-", dir=output.parent) as temporary:
        staged = Path(temporary) / "candidate_bundle"
        staged.mkdir()
        outputs, statistics = {}, {}
        for split in selected:
            (staged / split).mkdir()
            relative = split + "/behaviors_rs50.tsv"
            statistics[split] = write_split(records[split], index, metadata, staged / relative, split, args.seed)
            item = fingerprint(staged / relative)
            item["path"] = relative
            outputs[split] = item
        # Source changes during preparation invalidate the whole staged bundle.
        for path, expected in checked:
            _verify(path, expected, str(path))
        certificate = dict(schema=1, status="complete" if args.split == "both" else "partial",
            profile=PROFILE, created_utc=datetime.now(timezone.utc).isoformat(), seed=args.seed,
            sampling=dict(RECIPE), source_dataset_manifest=source_fp, input_files=inputs,
            publication_proxy=proxy_fp, outputs=outputs, statistics=statistics,
            validation=dict(train_dev_group_overlap=0, test_files_opened=False),
            producer_sha256=fingerprint(Path(__file__))["sha256"],
            notes=[
                "This is a candidate bundle, not a complete dataset.",
                "Copy unchanged IMP/news assets into a separate run dataset; never overwrite the historical source.",
                "Legacy RandS train candidates are incompatible with this new profile and are not reused.",
                "A first-observed-impression timestamp is a publication-time proxy, not actual publication time.",
                "The 50-negative pool is distinct from the optimizer's negative_sample_num.",
                "source_row is the one-based row of the selected IMP split; its order is pinned by the input hash.",
                "Partial train-only/dev-only bundles cannot validate as a matched training profile."])
        (staged / MANIFEST_NAME).write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
        os.rename(staged, output)
    return certificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New candidate-bundle directory outside the input dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("train", "dev", "both"), default="both")
    args = parser.parse_args()
    result = prepare(args)
    print(json.dumps(dict(status=result["status"], profile=PROFILE, output=str(args.output), statistics=result["statistics"])))


if __name__ == "__main__":
    main()

