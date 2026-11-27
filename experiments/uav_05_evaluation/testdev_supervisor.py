"""Supervise one Test-dev worker with the accepted NVML temperature checks."""
from collections import deque
import ctypes
import importlib.util
import json
from pathlib import Path
import subprocess
from threading import Event, Thread
from time import perf_counter, sleep

from pipeline import disk_guard, write
from resume_development import POLICY, observe
from supervisor import thermal_reason


def guarded_run(command, out, cap, env, expected_uuid):
    path = Path(__file__).resolve().parent.parent / "experiments_test/vm37_yolo26_rq4/post_cost/safety.py"
    spec = importlib.util.spec_from_file_location("testdev_thermal_policy", path)
    safety = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(safety)
    disk_guard(out)
    queue, history, errors = deque(), deque(maxlen=96), []
    stopped, worker_started = Event(), Event()
    observer = Thread(target=observe, args=(queue, stopped, errors, worker_started), daemon=True)
    wake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    began = perf_counter()
    process, reason, detail, peak, count = None, None, None, 0, 0
    observer.start()
    try:
        with (out / "gpu_telemetry.jsonl").open("x", encoding="utf-8") as telemetry, (
                out / "worker.log").open("xb") as log:
            while not reason:
                now = perf_counter()
                if now - began >= cap - 15:
                    reason = "WALL_CLOCK_HARD_STOP"
                while queue and not reason:
                    row = queue.popleft()
                    telemetry.write(json.dumps(row) + "\n")
                    telemetry.flush()
                    count += 1
                    peak = max(peak, row["temperature_C"])
                    if row["uuid"] != expected_uuid:
                        reason, detail = "TELEMETRY_FAILURE", "GPU identity changed."
                    history.append(row)
                    if not reason:
                        reason = thermal_reason(safety, POLICY, history, expected_uuid) if len(history) > 1 else safety.decide(POLICY, history)
                if not reason and (errors or now - (history[-1]["monotonic_seconds"] if history else began) >= 2):
                    reason = "TELEMETRY_FAILURE"
                    detail = errors[-1] if errors else "Temperature sampling exceeded two seconds."
                if reason:
                    break
                if process is None and history and now - began >= 12:
                    last = history[-1]
                    util = last["utilization_percent"]
                    if last["free_MiB"] < 3072 or util is None or util > 20 or last["temperature_C"] >= 87:
                        reason = "GPU_PREFLIGHT_FAILED"
                        break
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                    worker_started.set()
                if process is not None and process.poll() is not None:
                    reason = "WORKER_ERROR" if process.returncode else None
                    break
                sleep(.15)
    except BaseException:
        reason = reason or "SUPERVISOR_INTERRUPTED"
        raise
    finally:
        stopped.set()
        if process is not None and process.poll() is None:
            safety.stop_owned(process)
        if wake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        write(out / "supervisor.json", {
            "returncode": process.returncode if process is not None else None,
            "stop_reason": reason, "detail": detail, "peak_temp_C": peak,
            "budget_seconds": cap, "worker_wall_seconds": perf_counter() - began, "samples": count,
        })
    if reason or process is None:
        raise RuntimeError(f"Evaluation stopped: {reason}.")
