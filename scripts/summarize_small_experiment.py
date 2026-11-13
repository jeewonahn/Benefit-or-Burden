"""News-recommendation artifact alignment, aggregation and table utilities."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np
from experiment_timing import collect_timing_report, timing_csv_rows, timing_protocol_csv_rows


METRICS = ("AUC", "MRR", "nDCG@5", "nDCG@10")
PROTOCOLS = ("imp", "rs30", "rs90", "rs150", "total5")
MODELS = ("NAML", "NRMS", "NPA", "LSTUR", "CNE-SUE", "DIGAT")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_time(value):
    value = " ".join(str(value).split())
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        parsed = None
        for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S"):
            try:
                parsed = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValueError("Unrecognized request timestamp: " + value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat()


def group_key(record):
    """Use unchanged history hashes, never truncated model-history identities."""
    return (str(record["user_id"]), normalize_time(record["time"]),
            str(record["history_sha256"]))


def group_hash(key):
    return hashlib.sha256(json.dumps(key, ensure_ascii=False).encode("utf-8")).hexdigest()


def train_name(value):
    value = str(value).lower()
    if value in ("imp", "imptrain", "imp-based"):
        return "imp"
    if value in ("rs50", "rands", "rs", "rands-based"):
        return "rs50"
    raise ValueError("Unknown training protocol: " + value)


@dataclass
class Artifact:
    model: str
    train_type: str
    protocol: str
    prefix: Path
    report: dict
    values: np.ndarray
    groups: dict
    positives: dict
    candidate_counts: np.ndarray
    source_hashes: dict

    def grouped(self, keys):
        sums = np.array([self.values[self.groups[k]].sum(axis=0) for k in keys])
        counts = np.array([len(self.groups[k]) for k in keys], dtype=np.int64)
        return sums, counts


def artifact_path(prefix, suffix):
    return Path(str(prefix) + "." + suffix)


def load_artifact(model, training, protocol, prefix):
    prefix = Path(prefix)
    meta_path = artifact_path(prefix, "metrics.json")
    if not meta_path.is_file():
        return None
    with meta_path.open(encoding="utf-8") as source:
        report = json.load(source)
    if tuple(report.get("metric_order", METRICS)) != METRICS:
        raise ValueError("Unexpected metric order: " + str(prefix))
    values_path = artifact_path(prefix, "entry_metrics.npy")
    records_path = artifact_path(prefix, "records.jsonl")
    values = np.load(values_path, allow_pickle=False)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
        raise ValueError("Expected nonempty rows x 4 metrics: " + str(prefix))
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid per-entry metric values: " + str(prefix))
    groups, positives, candidates = defaultdict(list), defaultdict(Counter), []
    with records_path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            record = json.loads(line)
            key = group_key(record)
            positive = record["positive_news"]
            count = int(record["candidate_count"])
            if not positive or len(positive) != len(set(positive)) or count <= len(positive):
                raise ValueError("Invalid candidate/positive record: " + str(prefix))
            groups[key].append(index)
            positives[key].update(positive)
            candidates.append(count)
    if len(candidates) != len(values) or report.get("rows") != len(values):
        raise ValueError("Record/metric row count mismatch: " + str(prefix))
    if report.get("candidates") != sum(candidates):
        raise ValueError("Candidate count mismatch: " + str(prefix))
    logged = np.array([report["metrics"][name] for name in METRICS])
    if not np.allclose(logged, values.mean(axis=0), rtol=0, atol=1e-10):
        raise ValueError("Logged mean differs from per-entry metrics: " + str(prefix))
    return Artifact(model, training, protocol, prefix, report, values,
                    dict(groups), dict(positives), np.array(candidates),
                    {p.name: file_hash(p) for p in (meta_path, values_path, records_path)})


def aligned_groups(artifacts):
    """Match request support AND each request's multiset of GT articles."""
    if not artifacts:
        return [], {"union_groups": 0, "common_groups": 0, "gt_mismatch_groups": 0}
    union = set().union(*(set(a.groups) for a in artifacts))
    common = set(artifacts[0].groups).intersection(*(set(a.groups) for a in artifacts[1:]))
    valid = [key for key in sorted(common)
             if all(a.positives[key] == artifacts[0].positives[key] for a in artifacts[1:])]
    return valid, {"union_groups": len(union), "common_groups": len(common),
                   "gt_mismatch_groups": len(common) - len(valid),
                   "included_groups": len(valid), "excluded_groups": len(union) - len(valid)}


def average_ranks(values, tolerance):
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks, start = [0.0] * len(values), 0
    while start < len(order):
        end = start + 1
        while end < len(order) and abs(values[order[end]] - values[order[start]]) <= tolerance:
            end += 1
        rank = (start + 1 + end) / 2.0
        for i in order[start:end]:
            ranks[i] = rank
        start = end
    return ranks


def direction(difference, tolerance):
    return 0 if abs(difference) <= tolerance else (1 if difference > 0 else -1)


def means(grouped, chosen, aggregation):
    sums, counts = grouped
    if aggregation == "native_entry_macro":
        return sums[chosen].sum(axis=0) / counts[chosen].sum()
    return (sums[chosen] / counts[chosen, None]).mean(axis=0)


def pair_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["aggregation"], row["protocol"], row["metric"])].append(row)
    results = []
    for (aggregation, protocol, metric), items in sorted(grouped.items()):
        eligible = [row for row in items if row["eligible"]]
        per_pair = defaultdict(list)
        for row in eligible:
            per_pair[(row["model_a"], row["model_b"])].append(row["matched"])
        matched = sum(row["matched"] for row in eligible)
        macro = float(np.mean([np.mean(v) for v in per_pair.values()]) * 100) if per_pair else None
        results.append({"aggregation": aggregation, "protocol": protocol, "metric": metric,
                        "reference": "total5", "comparisons": len(items),
                        "reference_ties_excluded": len(items) - len(eligible),
                        "eligible_comparisons": len(eligible), "matched": matched,
                        "candidate_ties_with_nontied_reference": sum(row["candidate_sign"] == 0 for row in eligible),
                        "matching_percent": macro,
                        "macro_pair_matching_percent": macro,
                        "pooled_matching_percent": 100 * matched / len(eligible) if eligible else None,
                        "eligible_pairs": len(per_pair)})
    return results


def compute_tables(artifacts, fraction=0.5, repetitions=10, seed=42, tolerance=1e-12,
                   ranking_training="imp"):
    if not 0 < fraction <= 1 or repetitions < 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid resampling fraction, repetition count or tolerance")
    ranking_training = train_name(ranking_training)
    by_key = {(a.model, a.train_type, a.protocol): a for a in artifacts}
    if len(by_key) != len(artifacts):
        raise ValueError("Duplicate model/training/evaluation combination")
    # Identical candidate inputs are required for comparisons between models.
    hashes = defaultdict(set)
    for a in artifacts:
        source_hash = a.report.get("input", {}).get("sha256")
        if not source_hash:
            raise ValueError("Missing evaluated input hash: " + str(a.prefix))
        hashes[a.protocol].add(source_hash)
    if any(len(value) > 1 for value in hashes.values()):
        raise ValueError("Different evaluated input files within the same protocol")
    models = sorted(set(a.model for a in artifacts))
    cohort = [m for m in models if all((m, ranking_training, p) in by_key for p in PROTOCOLS)]
    cohort_artifacts = [by_key[m, ranking_training, p] for m in cohort for p in PROTOCOLS]
    common, alignment = aligned_groups(cohort_artifacts)
    table2, pairs, samples, table5, table5_alignment = [], [], [], [], []
    grouped = {(a.model, a.protocol): a.grouped(common) for a in cohort_artifacts} if common else {}
    if common:
        all_rows = np.arange(len(common))
        for aggregation in ("native_entry_macro", "request_macro"):
            for protocol in PROTOCOLS:
                scores = np.array([means(grouped[m, protocol], all_rows, aggregation) for m in cohort])
                for j, metric in enumerate(METRICS):
                    ranks = average_ranks(scores[:, j], tolerance)
                    for m, value, rank in zip(cohort, scores[:, j], ranks):
                        table2.append({"train_type": ranking_training, "aggregation": aggregation,
                                       "protocol": protocol, "metric": metric, "model": m,
                                       "value": float(value), "rank": rank,
                                       "cohort_models": len(cohort), "request_groups": len(common),
                                       "entries": int(grouped[m, protocol][1].sum())})
        if len(cohort) >= 2:
            sample_size = max(1, math.floor(len(common) * fraction))
            for repetition in range(repetitions):
                chosen = np.array(sorted(random.Random(seed + repetition).sample(range(len(common)), sample_size)))
                samples.append({"repetition": repetition + 1, "seed": seed + repetition,
                                "group_count": sample_size,
                                "group_sha256": [group_hash(common[i]) for i in chosen]})
                for aggregation in ("native_entry_macro", "request_macro"):
                    scores = {(m, p): means(grouped[m, p], chosen, aggregation)
                              for m in cohort for p in PROTOCOLS}
                    for a, b in itertools.combinations(cohort, 2):
                        for j, metric in enumerate(METRICS):
                            reference_delta = float(scores[a, "total5"][j] - scores[b, "total5"][j])
                            reference_sign = direction(reference_delta, tolerance)
                            for protocol in PROTOCOLS[:-1]:
                                delta = float(scores[a, protocol][j] - scores[b, protocol][j])
                                sign = direction(delta, tolerance)
                                pairs.append({"train_type": ranking_training,
                                              "aggregation": aggregation, "repetition": repetition + 1,
                                              "seed": seed + repetition, "protocol": protocol,
                                              "metric": metric, "model_a": a, "model_b": b,
                                              "reference_difference": reference_delta,
                                              "candidate_difference": delta,
                                              "reference_sign": reference_sign, "candidate_sign": sign,
                                              "eligible": int(reference_sign != 0),
                                              "matched": int(reference_sign != 0 and sign == reference_sign)})
    for model in models:
        if not all((model, training, "total5") in by_key for training in ("imp", "rs50")):
            continue
        compared = [by_key[model, training, "total5"] for training in ("imp", "rs50")]
        keys, audit = aligned_groups(compared)
        table5_alignment.append(dict(audit, model=model))
        if not keys:
            continue
        for aggregation in ("native_entry_macro", "request_macro"):
            scores = [means(a.grouped(keys), np.arange(len(keys)), aggregation) for a in compared]
            for j, metric in enumerate(METRICS):
                imp, rs = float(scores[0][j]), float(scores[1][j])
                table5.append({"model": model, "protocol": "total5", "aggregation": aggregation,
                               "metric": metric, "imp": imp, "rs50": rs,
                               "difference": rs - imp,
                               "relative_gain_percent": 100 * (rs - imp) / imp if imp != 0 else None,
                               "request_groups": len(keys)})
    return {"table2": table2, "table3_pairs": pairs, "table3": pair_summary(pairs),
            "table5": table5, "table2_3_cohort": cohort, "table2_3_alignment": alignment,
            "table5_alignment": table5_alignment, "samples": samples}


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as output:
        if rows:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def collect(index_path):
    index_path = Path(index_path).resolve()
    with index_path.open(encoding="utf-8") as source:
        index = json.load(source)
    runs = index if isinstance(index, list) else index["runs"]
    context = {} if isinstance(index, list) else index.get("context", {})
    artifacts, statuses, seen = [], [], set()
    for run in runs:
        model, training = str(run["model"]), train_name(run["train_type"])
        if (model, training) in seen:
            raise ValueError("Duplicate indexed model/training run")
        seen.add((model, training))
        root = Path(run["run_dir"])
        if not root.is_absolute():
            root = index_path.parent / root
        for protocol in PROTOCOLS:
            value = run.get("evaluations", {}).get(protocol, "evaluation/test_" + protocol)
            if value is None:
                statuses.append({"model": model, "train_type": training, "protocol": protocol,
                                 "status": "pending", "reason": "No evaluation prefix supplied"})
                continue
            prefix = Path(value)
            if not prefix.is_absolute():
                prefix = root / prefix
            artifact = load_artifact(model, training, protocol, prefix)
            statuses.append({"model": model, "train_type": training, "protocol": protocol,
                             "status": "complete" if artifact else "pending", "prefix": str(prefix.resolve())})
            if artifact:
                artifacts.append(artifact)
    for model in MODELS:
        for training in ("imp", "rs50"):
            if (model, training) not in seen:
                statuses.append({"model": model, "train_type": training, "protocol": "all",
                                 "status": "pending", "reason": "Run not indexed"})
    return artifacts, statuses, context


def summarize(args):
    artifacts, statuses, context = collect(args.run_index)
    result = compute_tables(artifacts, args.resample_fraction, args.repetitions, args.seed, args.tie_tolerance)
    auxiliary = compute_tables(artifacts, args.resample_fraction, args.repetitions, args.seed,
                               args.tie_tolerance, ranking_training="rs50")
    timing_report = collect_timing_report(args.run_index, getattr(args, "supervisor_status", None), getattr(args, "thermal_events", None))
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty; existing results are preserved")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("table2", "table3", "table3_pairs", "table5"):
        write_csv(output / (name + ".csv"), result[name])
    for name in ("table2", "table3", "table3_pairs"):
        write_csv(output / (name + "_rs50.csv"), auxiliary[name])
    write_json(output / "timings.json", timing_report)
    write_csv(output / "timings.csv", timing_csv_rows(timing_report))
    write_csv(output / "timing_protocols.csv", timing_protocol_csv_rows(timing_report))
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "run_index": str(Path(args.run_index).resolve()), "run_index_sha256": file_hash(args.run_index),
                "context": context, "status": statuses,
                "timing_report": {"file": "timings.json", "sha256": file_hash(output / "timings.json")},
                "method": {"ranking_training": "imp", "reference": "historical_total5_not_full_Total",
                           "ranking_training_scope": "IMP training is the present comparison choice; inspected R2 Table 2 text does not unambiguously assign its training profile",
                           "sampling": "without_replacement_shared_request_groups",
                           "fraction": args.resample_fraction, "repetitions": args.repetitions,
                           "seeds": list(range(args.seed, args.seed + args.repetitions)),
                           "tie_tolerance": args.tie_tolerance,
                           "tie_rule": "Exclude reference ties; candidate tie versus ordered reference is mismatch",
                           "sampling_scope": "New operational choice; original R2 does not specify sampling fraction",
                           "native_entry_macro": "Mean of input per-entry metrics; IMP rows may contain multiple positives",
                           "request_macro": "Mean per request after averaging that request's entry metrics",
                           "matching_percent": "Mean of each model pair's matched/eligible repetitions (primary)",
                           "macro_pair_matching_percent": "Mean of each model pair's matched/eligible repetitions",
                           "pooled_matching_percent": "Matched/eligible comparisons pooled across model pairs"},
                "cohort": result["table2_3_cohort"], "alignment": result["table2_3_alignment"],
                "cohorts": {"imp": result["table2_3_cohort"], "rs50": auxiliary["table2_3_cohort"]},
                "auxiliary_rs50_alignment": auxiliary["table2_3_alignment"],
                "table5_alignment": result["table5_alignment"],
                "source_artifacts": [{"model": a.model, "train_type": a.train_type,
                                      "protocol": a.protocol, "prefix": str(a.prefix.resolve()),
                                      "evaluated_input": a.report.get("input"),
                                      "cache_identity": a.report.get("cache_identity"),
                                      "files_sha256": a.source_hashes,
                                      "tie_policy": a.report.get("tie_policy"),
                                      "entries": len(a.values), "request_groups": len(a.groups),
                                      "mean_candidates": float(a.candidate_counts.mean())} for a in artifacts],
                "limitations": ["Current subset/Total5 results are not the original full-data, full-Total R2 tables.",
                                "Ten resamplings reuse fixed checkpoints; they are not ten training seeds.",
                                "RS30/90/150 nested candidate samples are dependent, not independent replications.",
                                "No missing model or uncomputed metric is filled with zero.",
                                "Model superiority uses unrounded means, not a statistical significance test.",
                                "Cohort and input support must be read with partial-run tables."]}
    write_json(output / "summary.json", manifest)
    write_json(output / "resampling_manifest.json", result["samples"])
    write_json(output / "resampling_manifest_rs50.json", auxiliary["samples"])
    write_json(output / "auxiliary_rs50.json", {
        "train_type": "rs50", "cohort": auxiliary["table2_3_cohort"],
        "alignment": auxiliary["table2_3_alignment"],
        "table2": auxiliary["table2"], "table3": auxiliary["table3"],
        "scope": "Separate RandS-trained cohort; not mixed with IMP-trained models."})
    lines = ["# Current small-experiment aggregates", "", "Created UTC: " + manifest["created_utc"], "",
             "Reference: historical Total5. These are subset diagnostics, not reproduced original R2 Tables 2/3/5.", "",
             "Completed IMP-trained comparison cohort: " + (", ".join(result["table2_3_cohort"]) or "none") + ".",
             "Table 3 status: " + ("computed" if result["table3"] else "pending; at least two complete models and shared GT-aligned requests are required") + ".", "",
             "- table2.csv: model scores and average tie ranks; native-entry and request-macro aggregation are separate.",
             "- table3.csv: mean of per-model-pair matching rates against Total5; reference ties are excluded and counted. Pooled comparison rates are also retained.",
             "- table3_pairs.csv: every seed, pair, metric and comparison, before aggregation.",
             "- table5.csv: IMP versus rs50 training on identical Total5 inputs; unavailable pairs remain absent.",
             "- table2_rs50.csv / table3_rs50.csv / auxiliary_rs50.json: separately ranked RandS-trained models; never pooled with the IMP-trained cohort.",
             "- timings.json / timings.csv / timing_protocols.csv: training, all-test sum, process wall and internal timers with explicit scope; missing process observations stay null.",
             "- summary.json: pending runs, provenance, alignment losses and limitations.",
             "- resampling_manifest.json: hashed request identities selected in each repetition.", "",
             "Table 3 samples " + str(args.resample_fraction) + " of shared request groups without replacement, "
             + str(args.repetitions) + " times, with seeds " + str(args.seed) + " onward. This fraction and pairing rule are new choices; R2 states ten different test sets but does not specify a fraction.", "",
             "Candidate rankings over common news under the same fixed model are not used as agreement evidence."]
    with (output / "README.md").open("x", encoding="utf-8", newline="\n") as destination:
        destination.write("\n".join(lines) + "\n")
    return manifest




