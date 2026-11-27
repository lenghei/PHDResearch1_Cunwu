"""Continue the frozen 548-image evaluator from a verified partial run."""
import argparse
from collections import deque
import ctypes as c
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
from time import perf_counter, sleep
from uuid import uuid4

import run_development as dev

POLICY = {"immediate_stop_temperature_C":90, "sustained_stop_temperature_C":87, "sustained_seconds":30}


class Memory(c.Structure):
    _fields_ = [(n,c.c_ulonglong) for n in ("total","free","used")]


class Utilization(c.Structure):
    _fields_ = [(n,c.c_uint) for n in ("gpu","memory")]


def read_telemetry(native, fallback):
    try:
        return native()
    except RuntimeError as error:
        try:
            return {**fallback(), "source":"nvidia-smi", "nvml_error":str(error)}
        except Exception as exc:
            raise RuntimeError(f"{error}; fallback failed: {exc}") from exc


def read_nvml_snapshot(core, utilization, require_utilization):
    snapshot = core()
    try:
        snapshot["utilization_percent"] = utilization()
    except RuntimeError as exc:
        if require_utilization:
            raise
        snapshot.update(utilization_percent=None, utilization_error=str(exc))
    return snapshot


def observe(samples, stop, errors, worker_started):
    lib = None
    ready = False
    try:
        lib = c.WinDLL("nvml.dll")
        def call(name, *args):
            status = getattr(lib,name)(*args)
            if status:
                raise RuntimeError(f"{name}: NVML {status}")
        call("nvmlInit_v2")
        ready = True
        count, handle = c.c_uint(), c.c_void_p()
        call("nvmlDeviceGetCount_v2",c.byref(count))
        dev.require(count.value==1,"Exactly one GPU is required.")
        call("nvmlDeviceGetHandleByIndex_v2",0,c.byref(handle))
        uuid = c.create_string_buffer(96)
        call("nvmlDeviceGetUUID",handle,uuid,len(uuid))
        def core():
            temp, mem = c.c_uint(),Memory()
            call("nvmlDeviceGetTemperature",handle,0,c.byref(temp))
            call("nvmlDeviceGetMemoryInfo",handle,c.byref(mem))
            return {"uuid":uuid.value.decode("ascii"),"temperature_C":temp.value,
                "free_MiB":mem.free//2**20,"monotonic_seconds":perf_counter(),
                "utc":datetime.now(timezone.utc).isoformat(),"source":"nvml"}
        def utilization():
            util = Utilization()
            call("nvmlDeviceGetUtilizationRates",handle,c.byref(util))
            return util.gpu
        def native():
            return read_nvml_snapshot(core,utilization,not worker_started.is_set())
        while not stop.is_set():
            begin = perf_counter()
            samples.append(read_telemetry(native,dev.sample))
            stop.wait(max(0,.45-(perf_counter()-begin)))
    except Exception as exc:
        errors.append(str(exc))
    finally:
        if ready:
            lib.nvmlShutdown()


def check_seal(run, names):
    checks = {name:sha for sha,name in (line.split("  ",1)
              for line in (run/"SHA256SUMS.txt").read_text(encoding="utf-8").splitlines())}
    for name in names:
        dev.require(dev.sha256(run/name)==checks.get(name),f"Incorrect source SHA: {run.name}/{name}")


def archived(source, continuations=()):
    runs = (source, *continuations)
    dev.require(len(set(runs))==len(runs), "Repeated source run.")
    check_seal(source, ("protocol.json","result.json","lease.json","members.json","group_progress.jsonl"))
    lease = dev.load(source/"lease.json")
    matrix, protocol = dev.design(lease["groups"])
    specs = {spec["pair_id"]:spec for group in matrix for spec in group}
    existing = {}
    for i, run in enumerate(runs):
        receipt = dev.load(run/"result.json") if i==0 else None
        before = len(existing)
        if i:
            check_seal(run, ("protocol.json","result.json","resume.json","supervisor.json"))
            receipt, ticket = (dev.load(run/name) for name in ("result.json","resume.json"))
            lineage = [parent.name for parent in runs[:i]]
            dev.require(receipt["source_RunID"]==source.name and
                        ticket.get("source_RunIDs",[ticket["source_RunID"]])==lineage and
                        ticket["source_pairs"]==before and
                        ticket["original_deadline_utc"]==lease["deadline_utc"] and
                        receipt["formal_pair_credit"]==0, "Continuation lineage or deadline differs.")
        dev.require(receipt["RunID"]==run.name and dev.load(run/"protocol.json")==protocol,
                    "Archived protocol or runtime differs.")
        for ident, spec in specs.items():
            folder = run/"pairs"/ident
            if not folder.is_dir():
                continue
            binding = dev.load(folder/"configuration.json")
            dev.require(ident not in existing and binding["spec"]==spec and
                        binding["gpu_uuid"]==lease["gpu_uuid"] and
                        all(binding[k]==v for k,v in protocol.items()), "Source binding differs.")
            dev.verify_pair(folder,binding)
            existing[ident] = binding
        if i:
            dev.require(len(existing)==receipt["total_unique_pairs"] and
                        len(existing)-before==receipt["new_pairs"],"Continuation pair count differs.")
        else:
            dev.require(len(existing)==receipt["evaluated_unique_pairs"],"Source pair count differs.")
    return matrix, protocol, existing, lease


def remaining_seconds(lease):
    if lease["deadline_utc"] is None:
        return float("inf")
    return (datetime.fromisoformat(lease["deadline_utc"])-datetime.now(timezone.utc)).total_seconds()


def bootstrap_for_workload(evaluator, samples, seed, progress_callback):
    params = evaluator._paramsEval
    n = len(params.imgIds)
    stride = n*len(params.areaRng)
    detections = sum(len(entry["dtScores"])
                     for offset in range(0,len(evaluator.evalImgs),stride)
                     for entry in evaluator.evalImgs[offset:offset+n] if entry is not None)
    if detections>=500_000:
        from bootstrap_workers import parallel_bootstrap
        return parallel_bootstrap(evaluator,samples,seed,progress_callback)
    return dev.bootstrap_coco(evaluator,samples,seed,progress_callback)


def worker(source, continuations, out, matrix, protocol, existing, lease):
    ticket = dev.load(out/"resume.json")
    runs = (source, *continuations)
    dev.require(ticket["token"]==os.environ.get("UAV04_RESUME_TOKEN") and
                ticket["source_RunIDs"]==[run.name for run in runs] and
                ticket["original_deadline_utc"]==lease["deadline_utc"] and
                ticket["code_sha256"]==dev.sha256(Path(__file__)) and
                ticket["source_pairs"]==len(existing),"Invalid worker lease.")
    if "deadline_utc" in ticket:
        deadline = ticket["deadline_utc"]
        dev.require(deadline==lease["deadline_utc"] or
                    (deadline is None and bool(ticket.get("authorization_note"))),
                    "Invalid continuation deadline.")
        lease = {**lease, "deadline_utc":deadline}
    bootstrap = dev.bootstrap_coco
    if ticket.get("parallel_bootstrap_sha256"):
        source_file = Path(__file__).with_name("bootstrap_workers.py")
        dev.require(dev.sha256(source_file)==ticket["parallel_bootstrap_sha256"],
                    "Parallel bootstrap code changed after admission.")
        bootstrap = bootstrap_for_workload
    if ticket.get("sparse_bootstrap_sha256"):
        source_file = Path(__file__).with_name("sparse_bootstrap.py")
        dev.require(dev.sha256(source_file)==ticket["sparse_bootstrap_sha256"],
                    "Sparse bootstrap code changed after admission.")
        from sparse_bootstrap import sparse_bootstrap
        bootstrap = partial(sparse_bootstrap, fallback=bootstrap)
    replay = dev.evaluate_saved
    if ticket.get("sparse_matching_sha256"):
        source_file = Path(__file__).with_name("sparse_matching.py")
        dev.require(dev.sha256(source_file)==ticket["sparse_matching_sha256"],
                    "Sparse matching code changed after admission.")
        from sparse_matching import with_sparse_matching
        replay = partial(with_sparse_matching, replay)
    from cv2 import __version__ as cv2_version
    import numpy as np
    members = dev.load(source/"members.json")
    dev.runtime(out)
    device = dev.Device(protocol["checkpoint_sha256"])
    torch = device.torch
    dev.require(device.params==11139470,"Parameter count differs.")
    fmp, _, _ = dev.models()
    attrs = fmp.read_uvl_attributes(dev.FMP)
    render = dev.generator()
    dev.module("new_fast_coco_accumulate",dev.HERE.parent/"experiments_test/cv261207/development_fast_coco.py")
    history = [json.loads(s)["group_seconds"] for s in (source/"group_progress.jsonl").read_text().splitlines()]
    reader = dev.OriginalZIP(dev.VAL,dev.IMAGES)
    try:
        for order,specs in enumerate(matrix,1):
            pending = [spec for spec in specs if spec["pair_id"] not in existing]
            if not pending:
                continue
            if remaining_seconds(lease) < max(600,1.5*max(history))*len(pending)/5+120:
                dev.write(out/"queue_stop.json",{"reason":"BUDGET_RESERVE","next_group":order})
                break
            dev.disk_guard(out,3*2**30)
            start = perf_counter()
            p = specs[0]["degradation"]
            recipe = {leaf:attrs[leaf] for leaf in p["leaves"]}
            key = dev.identity({"P":p,"recipe":recipe,"render_seed":protocol["render_seed"],
                 "dataset_sha256":protocol["dataset_sha256"],"sources":protocol["code_sha256"],
                 "numpy":np.__version__,"opencv":cv2_version})
            caches = next((run/"cache" for run in runs if (run/"cache"/key).is_dir()), out/"cache")
            cache, data, _ = dev.shared_cache(caches,key,members,reader,
                lambda image,stem:render.generate(image,p["leaves"],attrs,stem,protocol["render_seed"]))
            cache_sha, gt_sha = (dev.sha256(cache/n) for n in ("manifest.json","ground_truth.json"))
            for old in (existing[s["pair_id"]] for s in specs if s["pair_id"] in existing):
                dev.require((old["cache_manifest_sha256"],old["gt_sha256"])==(cache_sha,gt_sha),"Resumed inputs differ.")
            print(f"P {order}/{len(matrix)} | {p['degradation_id']} | {len(pending)} remaining D",flush=True)
            for spec in pending:
                before = perf_counter()
                ident, d = spec["pair_id"], spec["detector"]
                binding = {**protocol,"spec":spec,"cache_manifest_sha256":cache_sha,"gt_sha256":gt_sha,
                    "P_recipe_sha256":dev.identity({"P":p,"recipe":recipe}),"gpu_uuid":lease["gpu_uuid"],
                    "d_config_sha256":dev.identity(d)}
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                predictor = device.detector(d)
                def predict(image):
                    result,tensor = predictor(image)
                    return result.boxes.data.detach().cpu().numpy(),tensor,result.speed
                print(f"D {d['resolution']} {d['precision']} | {ident}",flush=True)
                dev.measure_pair(out/"pairs"/ident,binding,cache,data,predict,replay,bootstrap,
                    torch.cuda.synchronize,torch.cuda.reset_peak_memory_stats,
                    lambda:{"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes":torch.cuda.max_memory_reserved()},
                    clock_origin=before,prepare_seconds=perf_counter()-before)
                with (out/"pair_progress.jsonl").open("a",encoding="utf-8") as f:
                    f.write(json.dumps({"group":order,"pair_id":ident,"images":dev.IMAGES,
                                        "wall_seconds":perf_counter()-before})+"\n")
            spent = perf_counter()-start
            if len(pending)==5:
                history.append(spent)
            with (out/"group_progress.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps({"group":order,"P":p["degradation_id"],"new_pairs":len(pending),
                                    "wall_seconds":spent})+"\n")
    finally:
        reader.close()


def supervise(source, continuations, out, lease, token):
    safety = dev.module("uav04_resume_safety",dev.HERE.parent/
                        "experiments_test/vm37_yolo26_rq4/post_cost/safety.py")
    queue, history, errors, stop = deque(),deque(maxlen=100),[],Event()
    worker_started = Event()
    Thread(target=observe,args=(queue,stop,errors,worker_started),daemon=True).start()
    process, reason, detail = None,None,None
    began = perf_counter()
    count, peak = 0,None
    command = [sys.executable,"-B","-X","faulthandler",str(Path(__file__).resolve()),str(source),"--worker",str(out)]
    for run in continuations:
        command.extend(("--previous",str(run)))
    dev.write(out/"command.json",{"worker":command,"thermal_policy":POLICY})
    try:
        with (out/"gpu_telemetry.jsonl").open("x",encoding="utf-8") as monitor, (out/"worker.log").open("xb") as log:
            while not reason:
                now = perf_counter()
                if remaining_seconds(lease)<=15:
                    reason = "WALL_CLOCK_HARD_STOP"
                while queue and not reason:
                    obs = queue.popleft()
                    if obs["uuid"]!=lease["gpu_uuid"]:
                        reason,detail = "TELEMETRY_FAILURE","GPU UUID differs"
                    elif history and obs["monotonic_seconds"]-history[-1]["monotonic_seconds"]>2:
                        reason,detail = "TELEMETRY_FAILURE","sample gap exceeded 2 seconds"
                    history.append(obs)
                    count += 1
                    peak = max(peak or 0,obs["temperature_C"])
                    monitor.write(json.dumps(obs)+"\n")
                    monitor.flush()
                    reason = reason or safety.decide(POLICY,list(history))
                if not reason and (errors or (now-(history[-1]["monotonic_seconds"] if history else began))>2):
                    reason,detail = "TELEMETRY_FAILURE",errors[-1] if errors else "sample is stale"
                if reason:
                    break
                if process is None and history and now-began>=12:
                    last = history[-1]
                    util = last["utilization_percent"]
                    if last["free_MiB"]<3072 or util is None or util>20 or last["temperature_C"]>=87:
                        reason = "GPU_PREFLIGHT_FAILED"
                        break
                    process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,UAV04_RESUME_TOKEN=token))
                    worker_started.set()
                if process is not None and process.poll() is not None:
                    reason = "WORKER_ERROR" if process.returncode else None
                    break
                sleep(.15)
    finally:
        stop.set()
        if process is not None and process.poll() is None:
            safety.stop_owned(process)
        dev.write(out/"supervisor.json",{"stop_reason":reason,"detail":detail,"samples":count,"peak_temp_C":peak,
             "returncode":process.returncode if process else None,"worker_wall_seconds":perf_counter()-began})


def launch(source, continuations, matrix, protocol, existing, lease,
           authorization_note=None, parallel_bootstrap=False, sparse_bootstrap=False, sparse_matching=False):
    effective = {**lease, "deadline_utc":None} if authorization_note else lease
    dev.require(remaining_seconds(effective)>=720,"Original deadline is too close.")
    dev.require(len(existing)<len(matrix)*5,"All pairs are already complete.")
    out = dev.output_folder(dev.HERE.name)
    lock = dev.ROOT/"datasets/local_runs/uav_05_evaluation/.gpu.lock"
    with lock.open("x",encoding="utf-8") as stream:
        stream.write(str(out))
    print(out,flush=True)
    wake = c.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        token = uuid4().hex
        dev.write(out/"protocol.json",protocol)
        dev.write(out/"resume.json",{"source_RunID":source.name,"source_RunIDs":[r.name for r in (source,*continuations)],
            "original_deadline_utc":lease["deadline_utc"],
            "deadline_utc":effective["deadline_utc"],"authorization_note":authorization_note,
            "parallel_bootstrap_sha256":dev.sha256(Path(__file__).with_name("bootstrap_workers.py"))
                if parallel_bootstrap else None,
            "sparse_bootstrap_sha256":dev.sha256(Path(__file__).with_name("sparse_bootstrap.py"))
                if sparse_bootstrap else None,
            "sparse_matching_sha256":dev.sha256(Path(__file__).with_name("sparse_matching.py"))
                if sparse_matching else None,
            "source_pairs":len(existing),"authorization":"Author approved GPU continuation",
            "code_sha256":dev.sha256(Path(__file__)),"token":token})
        supervise(source,continuations,out,effective,token)
        found = set(existing)
        for specs in matrix:
            for spec in specs:
                ident = spec["pair_id"]
                folder = out/"pairs"/ident
                if folder.is_dir():
                    binding = dev.load(folder/"configuration.json")
                    dev.require(binding["spec"]==spec and ident not in found
                                and all(binding[k]==v for k,v in protocol.items()),"Invalid resumed binding.")
                    dev.verify_pair(folder,binding)
                    found.add(ident)
        record = dev.load(out/"supervisor.json")
        result = {"ExperimentID":dev.HERE.name,"RunID":out.name,"source_RunID":source.name,"scope":dev.SCOPE,
                  "status":"COMPLETE" if len(found)==len(matrix)*5 and not record["stop_reason"] else "PARTIAL",
                  "new_pairs":len(found)-len(existing),"total_unique_pairs":len(found),"formal_pair_credit":0,
                  "completed_groups":sum(all(s["pair_id"] in found for s in group) for group in matrix),
                  "stop_reason":record["stop_reason"] or ("BUDGET_RESERVE" if (out/"queue_stop.json").is_file() else None)}
        dev.write(out/"result.json",result)
        dev.index(out,{"ExperimentID":dev.HERE.name,"status":result["status"],
                       "CPU_wall_seconds":record["worker_wall_seconds"]})
        dev.text_file(out/"SHA256SUMS.txt","\n".join(f"{dev.sha256(p)}  {p.name}" for p in sorted(out.iterdir()) if p.is_file()))
        print(json.dumps(result),flush=True)
        return out
    finally:
        if wake:
            c.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink()


def wait_for_run(run):
    lock = dev.ROOT/"datasets/local_runs/uav_05_evaluation/.gpu.lock"
    while True:
        try:
            owner = lock.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            break
        dev.require(owner==str(run),"A different run owns the GPU lock.")
        sleep(5)
    check_seal(run, ("result.json","supervisor.json","resume.json"))
    return dev.load(run/"result.json")


def analyze_runs(runs):
    subprocess.run([sys.executable,"-B",str(dev.HERE/"analyze_development.py"),
                    *(str(run) for run in runs)],check=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("--previous",type=Path,action="append",default=[],help="Sealed continuation, in execution order.")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--no-deadline",help="Author's explicit permission for a new continuation without a wall-clock limit.")
    parser.add_argument("--after",type=Path,help="Continue after an active run reaches its planned time limit.")
    parser.add_argument("--parallel-bootstrap",action="store_true",
                        help="Use two CPU threads for the frozen bootstrap draws.")
    parser.add_argument("--sparse-bootstrap",action="store_true",
                        help="Reuse shared detection order and accumulate precision at true positives.")
    parser.add_argument("--sparse-matching",action="store_true",
                        help="Send only detections with sufficient IoU to the COCO matcher.")
    parser.add_argument("--worker",type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    source=args.source.resolve(strict=True)
    continuations=tuple(run.resolve(strict=True) for run in args.previous)
    active = args.after.resolve(strict=True) if args.after else None
    runs = (source,*continuations,*((active,) if active else ()))
    dev.require(all(run.parent==dev.RUNS.resolve() for run in runs),
                "Expected an existing Week 4 run.")
    dev.require(len(set(runs))==len(runs), "Repeated source run.")
    if args.no_deadline is not None:
        dev.require(args.execute and args.no_deadline.strip(),"A new explicit authorization is required.")
    if active:
        dev.require(args.execute and args.no_deadline,"Handoff requires a new explicit authorization.")
        prior = wait_for_run(active)
        continuations += (active,)
        inputs = archived(source,continuations)
        if prior["stop_reason"] in ("BUDGET_RESERVE","WALL_CLOCK_HARD_STOP") and len(inputs[2])<len(inputs[0])*5:
            continuations += (launch(source,continuations,*inputs,authorization_note=args.no_deadline,
                                     parallel_bootstrap=args.parallel_bootstrap,
                                     sparse_bootstrap=args.sparse_bootstrap,
                                     sparse_matching=args.sparse_matching),)
        analyze_runs((source,*continuations))
        sys.exit()
    inputs=archived(source,continuations)
    if args.worker:
        worker(source,continuations,args.worker,*inputs)
    elif args.execute:
        out = launch(source,continuations,*inputs,authorization_note=args.no_deadline,
                     parallel_bootstrap=args.parallel_bootstrap,sparse_bootstrap=args.sparse_bootstrap,
                     sparse_matching=args.sparse_matching)
        analyze_runs((source,*continuations,out))
    else:
        print(json.dumps({"verified_pairs":len(inputs[2]),"remaining_pairs":len(inputs[0])*5-len(inputs[2]),
                          "deadline_utc":inputs[3]["deadline_utc"]}))
