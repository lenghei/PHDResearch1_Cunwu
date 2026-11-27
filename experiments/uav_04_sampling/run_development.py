
"""Evaluate frozen Week 4 pairs on the complete 548-image development split."""
import argparse
import ctypes
from datetime import datetime, timedelta, timezone
from itertools import groupby
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1]
ROOT = MAIN.parent
WEEK5 = HERE.parent / "uav_05_evaluation"
RUNS = ROOT / "datasets/local_runs/uav_04_sampling"
sys.path.insert(0, str(MAIN / "code/tools"))
from uav_reproduction import VAL, output_folder
from uav_week01_06 import index, table, text_file

sys.path.insert(0, str(WEEK5))
from run import (CHECKPOINT, FMP, Device, bootstrap_coco, code_binding, evaluate_saved, frozen,
                 generator, module, runtime, runtime_binding)
from archives import OriginalZIP
from sampling import models
from pipeline import (disk_guard, identity, load, measure_pair, require, sha256, shared_cache,
                      verify_pair, verify_repeat, write)
from supervisor import guarded_run, sample

IMAGES = 548
BOOTSTRAP_SEED = 20261207
PROTOCOL = "uav_04_shared_png_native10_dev548_v1"
SCOPE = "DEVELOPMENT_ONLY_VISDRONE_VAL_CHECKPOINT_SELECTION_EXPOSURE"
MAX_SECONDS = 24*3600


def design(groups):
    source, selection, _ = frozen()
    val = load(source/"dataset_gate.json")["archives"]["val"]
    matrix = [list(rows) for _,rows in groupby(selection["pairs"],
              key=lambda r:r["degradation"]["degradation_id"])]
    checkpoint_sha = matrix[0][0]["detector"]["checkpoint_sha256"]
    require(val["images"]==IMAGES and sha256(VAL)==val["sha256"], "Validation ZIP differs.")
    require(sha256(CHECKPOINT)==checkpoint_sha, "Frozen checkpoint differs.")
    files = code_binding()
    files[Path(__file__).relative_to(MAIN).as_posix()] = sha256(Path(__file__))
    return matrix[:groups], {"protocol":PROTOCOL, "scope":SCOPE,"images":IMAGES,
        "source_split":"VisDrone2019-DET-val", "dataset_sha256":val["sha256"],
        "selection_sha256":sha256(source/"selection_manifest.json"), "source_RunID":source.name,
        "checkpoint_sha256":checkpoint_sha, "render_seed":selection["render_seed"], "bootstrap_seed":BOOTSTRAP_SEED,
        "bootstrap_samples":200, "code_sha256":files, "runtime":runtime_binding(),
        "formal_pair_credit":0, "planned_unique_pairs":groups*5}


def worker(out):
    started = perf_counter()
    lease = load(out/"lease.json")
    require(os.environ.get("UAV04_DEV_WORKER_TOKEN")==lease["token"], "Invalid worker lease.")
    matrix, protocol = design(lease["groups"])
    require(protocol==load(out/"protocol.json"), "Development protocol changed after admission.")
    reader = OriginalZIP(VAL,IMAGES)
    try:
        members = [reader.record(i)[2] for i in range(IMAGES)]
        write(out/"members.json",members)
        runtime(out)
        device = Device(protocol["checkpoint_sha256"])
        torch = device.torch
        require(device.params==11139470, "Unexpected checkpoint parameter count.")
        write(out/"environment.json",{"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,
              "cuda":torch.version.cuda,"checkpoint_parameters":device.params})
        from cv2 import __version__ as cv2_version
        import numpy as np
        pmodel, _, _ = models()
        attrs = pmodel.read_uvl_attributes(FMP)
        render = generator()
        module("new_fast_coco_accumulate", HERE.parent/"experiments_test/cv261207/development_fast_coco.py")
        reserve_seconds, reserve_bytes = 20*60, 3*1024**3
        group_costs = []
        for order, specs in enumerate(matrix,1):
            elapsed = perf_counter()-started
            if elapsed+reserve_seconds+120 > lease["cap"]:
                print("Budget reserve reached at a complete P-group boundary.",flush=True)
                write(out/"queue_stop.json",{"reason":"BUDGET_RESERVE","next_group":order,
                      "remaining_seconds":lease["cap"]-elapsed,"reserved_group_seconds":reserve_seconds})
                break
            disk_guard(out,reserve_bytes)
            group_started = perf_counter()
            p = specs[0]["degradation"]
            recipe = {leaf:attrs[leaf] for leaf in p["leaves"]}
            cache_identity = {"P":p, "recipe":recipe,"render_seed":protocol["render_seed"],
                              "dataset_sha256":protocol["dataset_sha256"],
                              "sources":protocol["code_sha256"],"numpy":np.__version__,"opencv":cv2_version}
            cache, cache_rows, _ = shared_cache(out/"cache",identity(cache_identity),members,reader,
                lambda image, stem:render.generate(image,p["leaves"],attrs,stem,protocol["render_seed"]))
            print(f"P {order}/{len(matrix)}: {p['degradation_id']} | 548 verified images",flush=True)
            schedule = [(spec,False) for spec in specs]
            if order==1:
                schedule.insert(1,(specs[0],True))
            for spec,repeated in schedule:
                pair_start = perf_counter()
                d = spec["detector"]
                binding = {**protocol,"spec":spec,"cache_manifest_sha256":sha256(cache/"manifest.json"),
                    "gt_sha256":sha256(cache/"ground_truth.json"),"P_recipe_sha256":identity({"P":p,"recipe":recipe}),
                    "gpu_uuid":lease["gpu_uuid"],"d_config_sha256":identity(d)}
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                predictor = device.detector(d)

                def predict(image):
                    result,tensor = predictor(image)
                    return result.boxes.data.detach().cpu().numpy(),tensor,result.speed

                suffix = "__repeat" if repeated else ""
                target = out/"pairs"/(spec["pair_id"]+suffix)
                print(f"D {d['resolution']} {d['precision']}{suffix} | {spec['pair_id']}",flush=True)
                measure_pair(target,binding,cache,cache_rows,predict,evaluate_saved,bootstrap_coco,
                    torch.cuda.synchronize,torch.cuda.reset_peak_memory_stats,
                    lambda:{"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes":torch.cuda.max_memory_reserved()},
                    clock_origin=pair_start,prepare_seconds=perf_counter()-pair_start)
                metrics = load(target/"metrics.json")
                metrics_all = next(row for row in metrics["per_area"] if row["size"]=="all")
                with (out/"pair_progress.jsonl").open("a",encoding="utf-8") as log:
                    log.write(json.dumps({"order":order,"pair_id":spec["pair_id"],"resolution":d["resolution"],
                        "precision":d["precision"],"repeat":repeated,"images":metrics["images"],
                        "AP50_95":metrics_all["map50_95"],"seconds":perf_counter()-pair_start})+"\n")
                if repeated:
                    first = out/"pairs"/spec["pair_id"]
                    verify_repeat(first,target,binding)
                    write(out/"first_pair_repeat.json",{"status":"PASS","pair_id":spec["pair_id"],"images_each":IMAGES})
            spent = perf_counter()-group_started
            group_costs.append(spent)
            # The first group includes an extra repeat. Retain a conservative upper planning cost.
            reserve_seconds = max(600, max(group_costs)*1.5)
            pair_bytes = sum(f.stat().st_size for spec,repeated in schedule for f in
                             (out/"pairs"/(spec["pair_id"]+("__repeat" if repeated else ""))).iterdir() if f.is_file())
            reserve_bytes = max(reserve_bytes, 2*(pair_bytes+cache_rows["cache_png_bytes"]))
            with (out/"group_progress.jsonl").open("a",encoding="utf-8") as log:
                log.write(json.dumps({"group":order,"P":p["degradation_id"],"group_seconds":spent,
                    "completed_unique":5*order,"remaining_seconds":lease["cap"]-(perf_counter()-started),
                    "planning_seconds_per_group":reserve_seconds,
                    "total_groups":len(matrix)})+"\n")
    finally:
        reader.close()


def summarize(out, matrix, protocol, error):
    import numpy as np
    results = []
    completed = 0
    for specs in matrix:
        group = []
        for spec in specs:
            folder = out/"pairs"/spec["pair_id"]
            if not folder.is_dir():
                continue
            binding = load(folder/"configuration.json")
            require(all(binding[k]==v for k,v in protocol.items()) and binding["spec"]==spec,
                    "Saved pair identity differs.")
            verify_pair(folder,binding)
            metric = load(folder/"metrics.json")
            timing = load(folder/"timing.json")
            boot = load(folder/"bootstrap.json")
            area = {r["size"]:r for r in metric["per_area"]}
            tiny = boot["draws"]["values"]["tiny"]["map50_95"]
            lo,hi = np.quantile(tiny,[.025,.975])
            latency_lo,latency_hi = timing["cached_pipeline_mean_ci95_ms"]
            status = "PASS" if lo>=.10 and latency_hi<=100 else "FAIL" if hi<.10 or latency_lo>100 else "UNCERTAIN"
            d = spec["detector"]
            group.append({"pair_id":spec["pair_id"],"P":spec["degradation"]["degradation_id"],
                "D":d["detector_id"],"images":metric["images"],"resolution":d["resolution"],
                "precision":d["precision"],"AP50":area["all"]["map50"],"AP50_95":area["all"]["map50_95"],
                "AR50_95":area["all"]["ar50_95"],"Tiny_AP50_95":area["tiny"]["map50_95"],
                "Small_AP50_95":area["small"]["map50_95"],"Medium_AP50_95":area["medium"]["map50_95"],
                "Large_AP50_95":area["large"]["map50_95"],"Tiny_CI95_low":lo,"Tiny_CI95_high":hi,
                "cached_pipeline_ms":timing["cached_pipeline_mean_ms"],
                "cached_pipeline_CI95_low":latency_lo,"cached_pipeline_CI95_high":latency_hi,
                "pair_seconds":timing["wall_seconds_to_sealing_including_setup"],
                "peak_allocated_MiB":timing["peak_allocated_bytes"]/2**20,"tested_scope_status":status,
                "predictions_sha256":sha256(folder/"predictions.jsonl.gz"),"GT_sha256":binding["gt_sha256"],
                "cache_manifest_sha256":binding["cache_manifest_sha256"]})
        require(len({(r["GT_sha256"],r["cache_manifest_sha256"]) for r in group})<=1,
                "Detectors did not share identical cached pixels and GT.")
        completed += len(group)==5
        results.extend(group)
    if results:
        table(out/"configuration_metrics.csv",results)
    supervisor = load(out/"supervisor.json") if (out/"supervisor.json").is_file() else {}
    stopped = load(out/"queue_stop.json") if (out/"queue_stop.json").is_file() else {}
    repeated = (out/"first_pair_repeat.json").is_file()
    max_deg = None
    if (out/"gpu_telemetry.jsonl").is_file():
        with (out/"gpu_telemetry.jsonl").open(encoding="utf-8") as monitor:
            max_deg = max((json.loads(line)["temperature_C"] for line in monitor),default=None)
    state = "COMPLETE" if len(results)==len(matrix)*5 and repeated and not error else "PARTIAL" if results else "BLOCKED"
    summary = {"RunID":out.name,"status":state,"scope":SCOPE,"images_per_pair":IMAGES,
               "completed_groups":completed,"planned_groups":len(matrix),"evaluated_unique_pairs":len(results),
               "image_evaluations":IMAGES*(len(results)+int(repeated)),"first_pair_repeat_verified":repeated,
               "formal_pair_credit":0,"reason":supervisor.get("stop_reason") or error or stopped.get("reason"),
               "elapsed_seconds":supervisor.get("worker_wall_seconds"),"peak_temp_C":max_deg}
    write(out/"result.json",summary)
    index(out,{"ExperimentID":HERE.name,"status":state,"CPU_wall_seconds":summary["elapsed_seconds"] or 0.0})
    text_file(out/"SHA256SUMS.txt", "\n".join(f"{sha256(f)}  {f.name}" for f in sorted(out.iterdir()) if f.is_file()))
    return summary


def launch(groups, hours):
    matrix, protocol = design(groups)
    out = output_folder(HERE.name)
    print(out,flush=True)
    reserved_cap = hours*3600
    before = datetime.now(timezone.utc)
    gpu = sample()
    lock = ROOT/"datasets/local_runs/uav_05_evaluation/.gpu.lock"
    with lock.open("x",encoding="utf-8") as log:
        log.write(str(out))
    error = None
    sleep_override = False
    try:
        disk_guard(out,3*1024**3)
        write(out/"protocol.json",protocol)
        token = uuid4().hex
        write(out/"lease.json",{"token":token,"groups":groups,"cap":reserved_cap-15,
              "deadline_utc":(before+timedelta(seconds=reserved_cap)).isoformat(),"gpu_uuid":gpu["uuid"],
              "authorization":"Author approved up to 24 hours for the 548-image development evaluation."})
        command = [sys.executable,"-B",str(Path(__file__).resolve()),"--worker",str(out)]
        write(out/"command.json",{"worker":command,"hours":hours,"groups":groups})
        if os.name=="nt":
            sleep_override = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        available = reserved_cap-(datetime.now(timezone.utc)-before).total_seconds()
        guarded_run(command,out,available,dict(os.environ,UAV04_DEV_WORKER_TOKEN=token),gpu["uuid"])
    except Exception as exc:
        error = str(exc)
        print(error,file=sys.stderr,flush=True)
    finally:
        try:
            summary = summarize(out,matrix,protocol,error)
            print(json.dumps(summary),flush=True)
        finally:
            if sleep_override:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.unlink()
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--worker",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--groups",type=int,choices=range(1,101),default=100,metavar="1..100")
    parser.add_argument("--hours",type=int,choices=range(1,25),default=24,metavar="1..24")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.execute:
        launch(args.groups,args.hours)
    else:
        _, protocol = design(args.groups)
        print(json.dumps(protocol))
