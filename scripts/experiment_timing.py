"""Normalize saved timing scopes without importing or executing any model."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from thermal_timing import read_thermal_log, attach_net_times

PROTOCOLS = ("imp", "rs30", "rs90", "rs150", "total5")


def finite_seconds(value):
    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def complete_sum(values):
    values = [finite_seconds(value) for value in values]
    return sum(values) if values and all(value is not None for value in values) else None


def normalize_timing(info, supervisor=None, evaluation_metadata=None):
    supervisor, evaluation_metadata = supervisor or {}, evaluation_metadata or {}
    status = info.get("status", supervisor.get("status", "pending"))
    if supervisor.get("status") == "interrupted_for_reboot" and status not in ("completed", "complete"):
        status = "interrupted_for_reboot"
    timings = info.get("timings_seconds", {})
    evidence, warnings = {}, []
    completed_epochs = {}
    for record in info.get("trained_epochs", []) + info.get("epoch_records", []):
        if "epoch" in record and "train_seconds" in record:
            completed_epochs[record["epoch"]] = record["train_seconds"]
    train = finite_seconds(timings.get("training"))
    if train is not None and (train > 0 or completed_epochs or status in ("completed", "complete")):
        evidence["training_seconds"] = "experiment.timings_seconds.training"
    else:
        train = complete_sum(completed_epochs.values())
        evidence["training_seconds"] = "unique completed epochs in trained_epochs/epoch_records" if train is not None else None
    dev_records = [record["dev"].get("elapsed_seconds") for record in info.get("epoch_records", []) if "dev" in record]
    dev = finite_seconds(timings.get("dev_evaluation"))
    if dev is not None and (dev > 0 or dev_records or status in ("completed", "complete")):
        evidence["dev_seconds"] = "experiment.timings_seconds.dev_evaluation"
    else:
        dev = complete_sum(dev_records)
        evidence["dev_seconds"] = "sum(epoch_records.dev.elapsed_seconds)" if dev is not None else None
    protocol_times = {}
    for protocol in PROTOCOLS:
        candidates = [
            (evaluation_metadata.get(protocol, {}).get("elapsed_seconds"), "evaluation artifact elapsed_seconds"),
            (info.get("evaluations", {}).get(protocol, {}).get("elapsed_seconds"), "experiment.evaluations elapsed_seconds"),
            (timings.get("tests", {}).get(protocol), "experiment.timings_seconds.tests"),
        ]
        available = [(finite_seconds(value), source) for value, source in candidates if finite_seconds(value) is not None]
        if available:
            protocol_times[protocol] = available[0][0]
            evidence["test_" + protocol + "_seconds"] = available[0][1]
            if any(abs(value - available[0][0]) > 1e-6 for value, _ in available[1:]):
                warnings.append("Different saved timing values for test " + protocol + "; completed evaluation artifact takes precedence")
    all_tests = all(protocol in protocol_times for protocol in PROTOCOLS)
    cache_records = info.get("cache_construction_records") or info.get("embedding_builds") or []
    cache = complete_sum(record.get("seconds", record.get("constructor_seconds")) for record in cache_records)
    if cache is not None:
        evidence["cache_constructor_seconds"] = "sum(cache_construction_records.seconds or embedding_builds.constructor_seconds)"
    elif finite_seconds(timings.get("cache_construction")) not in (None, 0):
        cache = finite_seconds(timings["cache_construction"])
        evidence["cache_constructor_seconds"] = "experiment.timings_seconds.cache_construction"
    preprocessing = finite_seconds(timings.get("preprocessing"))
    if preprocessing is None:
        preprocessing = finite_seconds(info.get("preprocess_seconds"))
    process = finite_seconds(supervisor.get("process_wall_seconds")) if supervisor.get("status") in ("completed", "failed") else None
    runner = finite_seconds(timings.get("total"))
    legacy = finite_seconds(info.get("elapsed_seconds"))
    internal, internal_scope = (runner, "R:runner_entry_wall") if runner is not None else (legacy, "M:manifest_internal_elapsed")
    segments = info.get("execution_segments", [])
    recovery_process = None
    if segments:
        segment_sum = complete_sum(segment.get("duration_seconds") for segment in segments)
        if segment_sum is None:
            raise ValueError("Execution segments require finite nonnegative recorded durations")
        if runner is not None and abs(runner - segment_sum) > max(.5, segment_sum * .001):
            raise ValueError("Segmented total differs from the sum of recorded execution segments")
        internal, internal_scope = segment_sum, "S:recorded_execution_segments_mixed_timer_scopes"
        recovery_process, process, runner = process, None, None
        warnings.append("Segmented total excludes the reboot gap and unmeasured interrupted work; it is not full-process time")
        evidence["internal_total_seconds"] = "sum(execution_segments.duration_seconds); segment timing_basis/coverage retain each measurement scope"
    if internal is None:
        internal_scope = None
    evidence["process_wall_seconds"] = "supervisor.process_wall_seconds (launch to observed process completion)" if process is not None else None
    evidence["runner_entry_total_seconds"] = "experiment.timings_seconds.total (runner entry; excludes interpreter launch)" if runner is not None else None
    evidence["manifest_elapsed_seconds"] = "experiment.elapsed_seconds (starts after upfront input/source work)" if legacy is not None else None
    test_loop = finite_seconds(timings.get("test_total")) if protocol_times else None
    loop_all_recorded = all(protocol in timings.get("tests", {}) or protocol in info.get("evaluations", {})
                            for protocol in PROTOCOLS)
    return {
        "status": status, "training_seconds": train,
        "test_protocol_sum_seconds": sum(protocol_times.values()) if all_tests else None,
        "test_completed_protocol_sum_seconds": sum(protocol_times.values()) if protocol_times else None,
        "test_protocols_completed": len(protocol_times), "test_protocols_expected": len(PROTOCOLS),
        "process_wall_seconds": process, "runner_entry_total_seconds": runner,
        "recovery_process_wall_seconds": recovery_process,
        "execution_segments": segments,
        "manifest_elapsed_seconds": legacy, "internal_total_seconds": internal, "internal_total_scope": internal_scope,
        "preprocessing_seconds": preprocessing, "preprocessing_reused_from": info.get("preprocessing_reused_from"),
        "cache_constructor_seconds": cache, "dev_seconds": dev,
        "test_loop_wall_seconds": test_loop if all_tests and loop_all_recorded else None,
        "test_loop_partial_wall_seconds": test_loop,
        "evaluation_configuration_benchmark_seconds": finite_seconds(timings.get("evaluation_configuration_benchmark")),
        "protocol_seconds": {protocol: protocol_times.get(protocol) for protocol in PROTOCOLS},
        "evidence": evidence, "warnings": warnings,
    }


def read_evidence(path, required=False):
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}, None
    payload = path.read_bytes()
    return json.loads(payload.decode("utf-8")), {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}


def collect_timing_report(run_index_path, supervisor_path=None, thermal_events_path=None):
    run_index_path = Path(run_index_path).resolve()
    index, index_evidence = read_evidence(run_index_path, required=True)
    runs = index if isinstance(index, list) else index["runs"]
    supervisor_path = Path(supervisor_path) if supervisor_path else run_index_path.parent / "visible_supervisor_status.json"
    supervisor, supervisor_evidence = read_evidence(supervisor_path)
    rows, sources, thermal_inputs = [], [], []
    for run in runs:
        root = Path(run["run_dir"])
        if not root.is_absolute():
            root = run_index_path.parent / root
        info, source = read_evidence(root / "experiment.json")
        if not source:
            info, source = read_evidence(root / "run_manifest.json")
        if source:
            sources.append(source)
        interruption, interruption_source = read_evidence(root / "INTERRUPTED_FOR_REBOOT.json")
        if interruption_source and info.get("status") not in ("completed", "complete"):
            sources.append(interruption_source)
            info = dict(info, status="interrupted_for_reboot")
        metadata = {}
        for protocol in PROTOCOLS:
            prefix = run.get("evaluations", {}).get(protocol, "evaluation/test_" + protocol)
            if prefix is None:
                continue
            prefix = Path(prefix)
            if not prefix.is_absolute():
                prefix = root / prefix
            report, source = read_evidence(Path(str(prefix) + ".metrics.json"))
            if source:
                sources.append(source)
                metadata[protocol] = report
        job = supervisor.get("jobs", {}).get(root.name, {})
        command = job.get("command", [])
        if "--run-dir" in command and Path(command[command.index("--run-dir") + 1]).resolve() != root.resolve():
            raise ValueError("Supervisor job points to another run directory: " + root.name)
        row = normalize_timing(info, job, metadata)
        if interruption_source:
            row["evidence"]["interruption_marker"] = interruption_source
            row["warnings"].append("Reboot interruption marker is a state record, not an observed process stop timestamp")
        row.update(model=run["model"], train_type=run["train_type"], run_dir=str(root.resolve()))
        rows.append(row)
        thermal_inputs.append((info, job, root.name))
    thermal_path = Path(thermal_events_path) if thermal_events_path else run_index_path.parent / "thermal_guard_events.jsonl"
    thermal_log = read_thermal_log(thermal_path)
    for row, (info, job, job_name) in zip(rows, thermal_inputs):
        segment_jobs = {segment.get("job_name"): supervisor.get("jobs", {}).get(segment.get("job_name"), {})
                        for segment in info.get("execution_segments", [])}
        attach_net_times(row, info, job, thermal_log, job_name, segment_jobs=segment_jobs)
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "rows": rows,
            "run_index": index_evidence, "supervisor": supervisor_evidence, "sources": sources,
            "thermal_events": thermal_log["source"],
            "thermal_policy_events": [event for event in thermal_log["events"] if event.get("event") == "guard_start"],
            "definitions": {
                "training": "Completed recorded training epochs only; excludes validation and preprocessing",
                "test_sum": "Sum of the five completed evaluate_file elapsed_seconds; NA until all five exist",
                "process_wall": "Supervisor launch-to-observed-exit wall time; not inferred from legacy elapsed",
                "runner_entry_total": "Early runner-entry wall timer when recorded (DIGAT); excludes Python process startup",
                "segmented_total": "Sum of individually recorded execution segments, excluding reboot gap; coverage may omit interrupted work and uses explicitly recorded mixed timer scopes",
                "manifest_elapsed": "Legacy internal timer; starts after upfront input checks/source work",
                "cache_constructor": "Only independently timed cache constructors; lazy encoding inside evaluation stays in evaluation time",
                "test_loop": "Outer test loop timer if recorded; includes runner bookkeeping between protocols",
                "net_time": "Wall time minus the union of evidenced managed suspension intervals; not GPU-kernel execution time",
                "net_phase_unknown": "A positive pause without matching phase boundaries leaves phase net time null; original wall is retained",
            }, "limitations": [
                "Old runs without a supervisor observation have no measured full-process time.",
                "Do not sum cumulative cache encoding snapshots across protocols; they overlap.",
                "Phase timers and full-process timers have different boundaries; their sums need not match.",
                "Shared data/asset preparation and earlier failed attempts are not allocated to each successful run.",
                "Protocol order and checkpoint-local cache reuse affect test timing; this is not a cold-start comparison.",
                "Thermal intervals use suspend-return to earliest resume-request boundaries, with event monotonic cross-checks; OS transition overhead remains in net time.",
                "No thermal log does not by itself prove zero managed pause; operator-confirmed history or covered job start is required.",
            ]}


def timing_csv_rows(report):
    return [dict({"model": row["model"], "train_type": row["train_type"], "run_dir": row["run_dir"]},
                 **{key: value for key, value in row.items()
                    if key not in ("model", "train_type", "run_dir", "protocol_seconds", "protocol_net_seconds",
                                   "execution_segments", "thermal_accounting", "net_time_status", "evidence", "warnings")})
            for row in report["rows"]]


def timing_protocol_csv_rows(report):
    return [{"model": row["model"], "train_type": row["train_type"], "status": row["status"],
             "protocol": protocol, "test_seconds": row["protocol_seconds"][protocol],
             "test_wall_seconds": row["protocol_seconds"][protocol],
             "test_net_seconds": row.get("protocol_net_seconds", {}).get(protocol),
             "net_status": row.get("net_time_status", {}).get("test_protocols", {}).get(protocol),
             "run_dir": row["run_dir"]}
            for row in report["rows"] for protocol in PROTOCOLS]
