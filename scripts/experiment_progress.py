"""Display measured worker progress in the launch terminal. No model or GPU dependencies."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time


REFRESH_SECONDS = 1
REPORT_SECONDS = 3600
_FILE_ENV = "TORS_EXPERIMENT_PROGRESS_FILE"
_INTERVAL_ENV = "TORS_EXPERIMENT_PROGRESS_INTERVAL"


def _interactive(stream):
    if not stream.isatty():
        return False
    if os.name != "nt":
        return os.environ.get("TERM") != "dumb"
    # A native CMD can inherit TERM=dumb from its launcher and still support VT.
    import ctypes
    import msvcrt
    from ctypes import wintypes
    try:
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
        mode = wintypes.DWORD()
        kernel = ctypes.windll.kernel32
        return bool(kernel.GetConsoleMode(handle, ctypes.byref(mode))
                    and kernel.SetConsoleMode(handle, mode.value | 0x0004))
    except (OSError, ValueError):
        return False


def _counter(label, done, total, width):
    if not total:
        return f"{label}  {done:,} / ?"
    fraction = done / total
    count = f"{fraction:5.1%}  {done:,}/{total:,}"
    cells = min(24, max(0, width - len(count) - len(label) - 6))
    if cells < 4:
        return f"{label}  {count}"
    filled = min(cells, max(0, int(cells * fraction)))
    return f"{label}  [{'#' * filled}{'-' * (cells - filled)}] {count}"


def progress_lines(event, width):
    if event["phase"] == "training":
        epoch, step, steps = event["epoch"], event["step"], event["steps"]
        finished = epoch - 1 + int(steps > 0 and step == steps)
        return [_counter("Epochs (max)", finished, event["max_epochs"], width),
                _counter(f"Epoch {epoch} batches", step, steps, width)]
    if event["phase"] == "test":
        label = f"Test {event['protocol']} ({event['unit']})"
        detail = ""
        if event.get("score_total"):
            detail = (f" | chunk scores {event['scored']:,}/{event['score_total']:,} "
                      f"({event['scored']/event['score_total']:.1%})")
        line = _counter(label, event["done"], event["total"], width - len(detail))
        if detail and len(line + detail) > width:
            detail = f" | chunk {event['scored']/event['score_total']:.0%}"
            line = _counter(label, event["done"], event["total"], width - len(detail))
        return [line + detail]
    return []


class _Console:
    def __init__(self, stream):
        self.stream = stream
        self.live = _interactive(stream)
        self.height = 0
        self.last_report = None
        self.lock = threading.RLock()

    def clear(self):
        with self.lock:
            if self.height:
                self.stream.write("\r\x1b[2K" + "\x1b[1A\r\x1b[2K" * (self.height - 1))
                self.stream.flush()
                self.height = 0

    def update(self, event, job_name="", *, force=False):
        with self.lock:
            now = time.monotonic()
            if not force and not self.live and self.last_report is not None and now - self.last_report < REPORT_SECONDS:
                return
            width = max(1, shutil.get_terminal_size((100, 24)).columns - 1)
            lines = progress_lines(event, width)
            if self.live:
                self.clear()
                if lines:
                    self.stream.write("\n".join(line[:width] for line in lines))
                    self.height = len(lines)
            elif lines or job_name:
                detail = " | ".join(lines) if lines else "waiting for measured progress"
                self.stream.write(f"[{time.strftime('%H:%M:%S')}] {job_name} {detail}\n")
            self.stream.flush()
            self.last_report = now


_console = None
_last_phase = None
_last_update = None


def clear_console_progress():
    if _console is not None:
        try:
            _console.clear()
        except (OSError, ValueError):
            pass


def _publish(event, key, force=False):
    global _console, _last_phase, _last_update
    now = time.monotonic()
    path = os.environ.get(_FILE_ENV)
    if path:
        # Keep the hourly snapshot fresh even when the worker and launcher clocks differ.
        interval = 30 if os.environ.get(_INTERVAL_ENV) == str(REPORT_SECONDS) else REFRESH_SECONDS
    else:
        if _console is None:
            _console = _Console(sys.stderr)
        interval = REFRESH_SECONDS if _console.live else REPORT_SECONDS
    if not force and key == _last_phase and _last_update is not None and now - _last_update < interval:
        return
    _last_phase, _last_update = key, now
    try:
        if path:
            tmp = Path(path).with_suffix(".tmp")
            tmp.write_text(json.dumps(event), encoding="utf-8")
            os.replace(tmp, path)
        else:
            _console.update(event)
    except (OSError, ValueError):
        # Loss of the display or temporary status file must not interrupt an optimizer step.
        return


def train_progress(epoch, max_epochs, step, steps):
    _publish(dict(phase="training", epoch=epoch, max_epochs=max_epochs, step=step, steps=steps),
             ("training", epoch), force=step == steps and steps > 0)


def test_progress(protocol, done, total, *, unit="rows", scored=None, score_total=None):
    _publish(dict(phase="test", protocol=protocol, done=done, total=total, unit=unit,
                  scored=scored, score_total=score_total), ("test", protocol), force=bool(total and done == total))


def finish_progress():
    if _last_phase is not None:
        _publish({"phase": "idle"}, ("idle",))


@contextmanager
def batch_progress(env, job_name):
    """The child reports to a private local file even when its stdout goes to a detailed log."""
    global _console
    with tempfile.TemporaryDirectory(prefix="tors-progress-") as folder:
        path = Path(folder) / "latest.json"
        console = _Console(sys.stdout)
        previous = _console
        old_env = {key: env.get(key) for key in (_FILE_ENV, _INTERVAL_ENV)}
        _console = console
        env[_FILE_ENV] = str(path)
        interval = REFRESH_SECONDS if console.live else REPORT_SECONDS
        env[_INTERVAL_ENV] = str(interval)
        stop = threading.Event()
        console.stream.write(f"Running {job_name}; waiting for measured progress.\n")
        console.stream.flush()

        def display():
            previous_event = None
            # Poll the lightweight snapshot even for redirected output. The console
            # itself keeps hourly log lines, but the first measurement appears now.
            while not stop.wait(REFRESH_SECONDS):
                try:
                    if not path.is_file():
                        continue
                    event = json.loads(path.read_text(encoding="utf-8"))
                    if not console.live or event != previous_event or not console.height:
                        console.update(event, job_name)
                        previous_event = event
                except (OSError, ValueError, KeyError, TypeError):
                    continue

        thread = threading.Thread(target=display, name="experiment-progress", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()
            try:
                # Short redirected runs still expose their last measured counter.
                # This is not a completion claim: only display the actual snapshot.
                if not console.live and path.is_file():
                    try:
                        event = json.loads(path.read_text(encoding="utf-8"))
                        if event.get("phase") in ("training", "test"):
                            console.update(event, job_name, force=True)
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                clear_console_progress()
            finally:
                _console = previous
                for key, value in old_env.items():
                    if value is None:
                        env.pop(key, None)
                    else:
                        env[key] = value
