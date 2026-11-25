"""Read GPU telemetry while the owned training lease is active."""
import argparse
import json
import subprocess
import time
from study import *


def watch(identity):
    out=run_path(identity);lease=load(out/'WORKER_LEASE.json')
    while (RUNS/'.gpu.lock').exists() and load(RUNS/'.gpu.lock')['RunID']==identity:
        query=subprocess.run(['nvidia-smi','--query-gpu=memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],
            capture_output=True,text=True,timeout=10)
        if query.returncode:
            if not (out/'SHA256SUMS.txt').exists():save(out/'telemetry_failure.json',{'UTC':utc().isoformat(),'stderr':query.stderr},out)
            break
        memory,temperature,utilization=[int(s.strip()) for s in query.stdout.strip().split(',')]
        row={'UTC':utc().isoformat(),'used_MiB':memory,'temperature_C':temperature,'utilization_percent':utilization,
            'driver_target_temperature_C':87,'target_evidence':'nvidia-smi-q-d-TEMPERATURE; no system setting changed.'}
        if not (RUNS/'.gpu.lock').exists() or load(RUNS/'.gpu.lock')['RunID']!=identity or (out/'SHA256SUMS.txt').exists():break
        with (out/'gpu_telemetry.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
        time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);watch(parser.parse_args().run)
