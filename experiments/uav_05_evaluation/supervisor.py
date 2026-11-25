"""Guard one authorized Windows worker tree and enforce thermal and wall time limits."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import time

from pipeline import disk_guard, require, write


def sample():
    cmd = ["nvidia-smi", "--query-gpu=uuid,temperature.gpu,memory.free,utilization.gpu",
           "--format=csv,noheader,nounits"]
    rows = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1).stdout.strip().splitlines()
    require(len(rows)==1, "Exactly one NVIDIA device is required.")
    uuid, deg, memory, utilization = [v.strip() for v in rows[0].split(",")]
    return {"uuid":uuid, "temperature_C":int(deg), "free_MiB":int(memory),
            "utilization_percent":int(utilization), "monotonic_seconds":time.perf_counter(),
            "utc":datetime.now(timezone.utc).isoformat()}


def thermal_reason(safety, policy, samples, expected_uuid):
    latest, previous = samples[-1], samples[-2]
    if latest["uuid"] != expected_uuid or latest["monotonic_seconds"] - previous["monotonic_seconds"] > 2:
        return "TELEMETRY_FAILURE"
    return safety.decide(policy, samples)


def guarded_run(command, out, cap, env, expected_uuid):
    safety_file = Path(__file__).resolve().parent.parent/"experiments_test/vm37_yolo26_rq4/post_cost/safety.py"
    spec = importlib.util.spec_from_file_location("uav_existing_thermal_policy", safety_file)
    safety = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(safety)
    policy = {"immediate_stop_temperature_C":90, "sustained_stop_temperature_C":87, "sustained_seconds":30}
    started = time.perf_counter()
    initial = sample()
    disk_guard(out)
    require(initial["uuid"]==expected_uuid and initial["temperature_C"]<87
            and initial["utilization_percent"]<=20 and initial["free_MiB"]>=3072,
            "GPU UUID, temperature, contention or available memory preflight failed.")
    samples, reason, code = [initial], None, None
    with (out/"gpu_telemetry.jsonl").open("x", encoding="utf-8") as telemetry, (out/"worker.log").open("xb") as log:
        telemetry.write(json.dumps(initial)+"\n")
        telemetry.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while process.poll() is None:
                if time.perf_counter()-started >= cap-15:
                    reason = "WALL_CLOCK_HARD_STOP"
                else:
                    try:
                        observation = sample()

                        samples.append(observation)
                        telemetry.write(json.dumps(observation)+"\n")
                        telemetry.flush()
                        reason = thermal_reason(safety, policy, samples, expected_uuid)
                    except Exception:
                        reason = "TELEMETRY_FAILURE"
                if reason:
                    safety.stop_owned(process)
                    break
                time.sleep(.45)
            code = process.wait(timeout=15)
        except BaseException:
            reason = "SUPERVISOR_INTERRUPTED"
            if process.poll() is None:
                safety.stop_owned(process)
            raise
        finally:
            write(out/"supervisor.json", {"returncode":code, "stop_reason":reason, "budget_seconds":cap,
                  "worker_wall_seconds":time.perf_counter()-started, "retry":False, "samples":len(samples)})
    if reason or code!=0:
        raise RuntimeError(f"Evaluation worker stopped: {reason or code}.")
