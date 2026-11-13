"""Subtract evidenced managed suspension intervals; never control a process."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


def numeric(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Backward interval")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def duration(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def overlap(intervals, windows):
    return duration([(max(a, c), min(b, d)) for a, b in intervals for c, d in windows
                     if max(a, c) < min(b, d)])


def read_thermal_log(path):
    path = Path(path)
    if not path.is_file():
        return {"events": [], "source": None, "complete_read": False, "warnings": ["Thermal log absent; absence alone is not zero pause evidence"]}
    raw = path.read_bytes()
    events, warnings = [], []
    for index, line in enumerate(raw.splitlines(), 1):
        try:
            events.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            warnings.append("Unparseable thermal event line " + str(index))
    complete = bool(raw) and raw.endswith(b"\n") and not warnings
    return {"events": events, "complete_read": complete, "warnings": warnings,
            "source": {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}}


def pause_account(log, job_name, supervisor=None):
    supervisor = supervisor or {}
    events = log.get("events", [])
    starts = [event for event in events if event.get("event") == "guard_start"]
    confirmed_prior_zero = any(event.get("no_managed_pauses_before_guard_start") is True for event in starts)
    job_started = None
    if supervisor.get("started_kst"):
        job_started = datetime.fromisoformat(supervisor["started_kst"]).timestamp()
    starts_before_job = bool(job_started is not None and any(numeric(event.get("event_unix")) and event["event_unix"] <= job_started for event in starts))
    covered = bool(log.get("complete_read") and (confirmed_prior_zero or starts_before_job))
    related = [event for event in events if event.get("job") == job_name
               and event.get("event") in ("pause_start", "pause_end", "watchdog_resume")]
    grouped = {}
    for event in related:
        # Guard targets the verified deepest worker; supervisor PID can be its launcher.
        identity = (event.get("interval_id"), event.get("pid"), event.get("create_time"))
        grouped.setdefault(identity, []).append(event)
    intervals, unresolved, details = [], [], []
    for identity, records in grouped.items():
        beginning = [e for e in records if e.get("event") == "pause_start" and numeric(e.get("event_unix"))]
        endings = [e for e in records if e.get("event") in ("pause_end", "watchdog_resume") and numeric(e.get("resume_requested_unix"))]
        if not identity[0] or not beginning or not endings:
            unresolved.append({"interval_id": identity[0], "reason": "Missing start or explicit resume-request boundary"})
            continue
        start = min(beginning, key=lambda e: e["event_unix"])
        end = min(endings, key=lambda e: e["resume_requested_unix"])
        first, last = start["event_unix"], end["resume_requested_unix"]
        if last < first:
            unresolved.append({"interval_id": identity[0], "reason": "Resume request predates recorded suspension"})
            continue
        if not (numeric(start.get("event_monotonic")) and numeric(end.get("event_monotonic")) and numeric(end.get("event_unix"))):
            unresolved.append({"interval_id": identity[0], "reason": "No monotonic cross-check for wall-clock interval"})
            continue
        drift = abs((end["event_unix"] - first) - (end["event_monotonic"] - start["event_monotonic"]))
        if drift > .25:
            unresolved.append({"interval_id": identity[0], "reason": "Wall/monotonic clock discrepancy", "seconds": drift})
            continue
        intervals.append((first, last))
        details.append({"interval_id": identity[0], "worker_pid": identity[1], "worker_create_time": identity[2],
                        "start_unix": first, "end_unix": last, "duration_seconds": last - first,
                        "resume_event": end["event"], "clock_discrepancy_seconds": drift})
    known = covered and not unresolved
    merged = merge_intervals(intervals)
    return {"status": "known" if known else "unconfirmed", "managed_pause_seconds": duration(merged) if known else None,
            "closed_interval_count": len(details), "merged_interval_count": len(merged),
            "intervals": merged, "interval_evidence": details, "unresolved": unresolved,
            "history_basis": "operator-confirmed zero before guard start" if confirmed_prior_zero else "job began after guard start" if starts_before_job else "unconfirmed",
            "scope": "Recorded definite suspension from suspend return to earliest explicit resume request; OS-call transition overhead is not subtracted",
            "warnings": log.get("warnings", [])}


def net_phase(wall, pause, phase_intervals, phase=None, protocol=None):
    if wall is None:
        return None, "wall_time_unavailable"
    if pause["managed_pause_seconds"] is None:
        return None, "managed_pause_history_unconfirmed"
    if pause["managed_pause_seconds"] == 0:
        return wall, "confirmed_zero_managed_pause"
    selected = [item for item in phase_intervals if item.get("phase") == phase
                and (protocol is None or item.get("protocol") == protocol)]
    windows = []
    for item in selected:
        if item.get("start_boundary_derived") or item.get("end_boundary_derived"):
            return None, "derived_phase_boundary_cannot_allocate_positive_pause"
        first, last = item.get("start_unix"), item.get("end_unix")
        if not numeric(first) or not numeric(last) or last < first:
            return None, "phase_boundaries_invalid"
        if numeric(item.get("start_monotonic")) and numeric(item.get("end_monotonic")):
            if abs((last - first) - (item["end_monotonic"] - item["start_monotonic"])) > .25:
                return None, "phase_clock_discrepancy"
        windows.append((first, last))
    if not windows:
        return None, "phase_boundaries_unavailable_wall_not_adjusted"
    if abs(duration(windows) - wall) > max(.5, wall * .001):
        return None, "phase_boundary_duration_differs_from_reported_wall"
    removed = overlap(pause["intervals"], merge_intervals(windows))
    if removed > wall + .01:
        return None, "pause_exceeds_phase_wall"
    return max(0., wall - removed), "recorded_pause_overlap_subtracted"


def attach_net_times(row, info, supervisor, log, job_name, segment_jobs=None):
    pause = pause_account(log, job_name, supervisor)
    segments = info.get("execution_segments", [])
    if segments:
        names = list(dict.fromkeys(segment["job_name"] for segment in segments))
        accounts = {name: pause_account(log, name, (segment_jobs or {}).get(name, {})) for name in names}
        known = all(account["managed_pause_seconds"] is not None for account in accounts.values())
        intervals = merge_intervals([interval for account in accounts.values() for interval in account["intervals"]])
        pause = {"status": "known" if known else "unconfirmed",
                 "managed_pause_seconds": duration(intervals) if known else None,
                 "closed_interval_count": sum(account["closed_interval_count"] for account in accounts.values()),
                 "merged_interval_count": len(intervals), "intervals": intervals,
                 "segment_job_accounts": accounts,
                 "history_basis": "separate job histories for recorded execution segments; reboot gap excluded",
                 "unresolved": [item for account in accounts.values() for item in account["unresolved"]]}
    row["managed_cooldown_seconds"] = pause["managed_pause_seconds"]
    row["cooldown_interval_count"] = pause["closed_interval_count"]
    row["thermal_accounting"] = pause
    intervals = info.get("timing_intervals", [])
    row["training_net_seconds"], training_status = net_phase(row.get("training_seconds"), pause, intervals, "training")
    row["dev_net_seconds"], dev_status = net_phase(row.get("dev_seconds"), pause, intervals, "dev")
    protocol_net, protocol_status = {}, {}
    for protocol, wall in row["protocol_seconds"].items():
        protocol_net[protocol], protocol_status[protocol] = net_phase(wall, pause, intervals, "test", protocol)
    row["protocol_net_seconds"] = protocol_net
    row["test_protocol_sum_net_seconds"] = sum(protocol_net.values()) if all(v is not None for v in protocol_net.values()) else None
    process_wall = row.get("process_wall_seconds")
    if process_wall is not None and pause["managed_pause_seconds"] is not None and pause["managed_pause_seconds"] <= process_wall:
        row["process_net_seconds"] = process_wall - pause["managed_pause_seconds"]
    else:
        row["process_net_seconds"] = None
    row["internal_total_net_seconds"], internal_status = net_phase(row.get("internal_total_seconds"), pause, intervals, "total")
    row["net_time_status"] = {"training": training_status, "dev": dev_status, "test_protocols": protocol_status,
                              "internal_total": internal_status}
    return row
