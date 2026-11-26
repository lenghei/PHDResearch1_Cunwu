"""Check historical native-ten models and recover their development results."""
import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean
import sys
from tempfile import TemporaryDirectory

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code/tools"))
from uav_reproduction import EXPS, RESULTS, ROOT, RUNS, digest, output_folder, write_json
from uav_week01_06 import BASE

CATALOG = ROOT / "datasets/research_archive/migration_20261104"
PHASE4J = "full20_2ba95688f0cb43a0af8e2a839c82cb07"
GROUPS = {
    "yolov8s_cft640": "WEEK1_BASELINE",
    "ck05_yolov8s_ft1280": "FT1280_SMALL",
    "ck06_yolo11s_ft1280": "FT1280_SMALL",
    "ck01_yolo26s_ft1280": "FT1280_SMALL",
    "yolov8n_ft640": "LEGACY_NANO",
    "ck04_yolo26n_native10_full20": "PHASE4J_NANO",
}


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def dump(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verified(path, expected):
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"Cannot verify {path}")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def healthy_state(net, torch):
    for name, tensor in net.state_dict().items():
        if tensor.is_floating_point() and (not torch.isfinite(tensor).all()
                or name.endswith("running_var") and (tensor < 0).any()):
            raise ValueError(f"Unhealthy tensor: {name}")


def reusable(out):
    import torch
    from ultralytics import YOLO
    import yaml
    from main.labels import VIS_NAMES

    torch.set_num_threads(2)
    registry = {r["model_id"]: r for r in table(CATALOG / "model_catalog.csv")}
    history = table(CATALOG / "checkpoint_index.csv")
    inventory, private, requests = [], [], []
    for model_id, cohort in GROUPS.items():
        row = registry[model_id]
        path = ROOT / row["local_checkpoint"]
        binding = next((r for r in history if r["model_id"] == model_id and r["sha256"] == row["sha256"]), None)
        origin = (BASE/"training/yolov8s/weights/best.pt" if cohort == "WEEK1_BASELINE"
                  else Path(binding["source_path"]))
        if not path.is_file():
            verified(origin, row["sha256"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=path.parent) as staged:
                temporary = Path(staged) / path.name
                shutil.copyfile(origin, temporary)
                verified(temporary, row["sha256"])
                os.link(temporary, path)
        verified(path, row["sha256"])
        if origin.is_file():
            verified(origin, row["sha256"])
        original = YOLO(str(path))
        if list(original.names.values()) != VIS_NAMES:
            raise ValueError(f"Native-ten classes differ for {model_id}.")
        healthy_state(original.model, torch)
        for key in ("ema", "model"):
            component = original.ckpt.get(key)
            if isinstance(component, torch.nn.Module):
                healthy_state(component, torch)
        train_args = ROOT / row["training_config"]
        train_log = (BASE/"training/yolov8s/results.csv" if cohort == "WEEK1_BASELINE"
                     else ROOT / row["train_curve"])
        args = yaml.safe_load(train_args.read_text(encoding="utf-8-sig"))
        log = table(train_log)
        epochs = [int(r["epoch"]) for r in log]
        if epochs != list(range(1, int(args["epochs"])+1)):
            raise ValueError(f"Training log is incomplete for {model_id}.")
        dataname = {"PHASE4J_NANO": "VisDrone A5841 train / B627 development",
                    "WEEK1_BASELINE": "VisDrone DET train 6471 / val 548"}.get(
                    cohort, "VisDrone official train / val (historical YAML)")
        role = {"WEEK1_BASELINE":"best EMA, epoch 76", "PHASE4J_NANO":"fixed epoch 20 EMA"}.get(
                    cohort, "validation-selected best EMA")
        item = {"checkpoint":model_id,"sha256":row["sha256"],"cohort":cohort,"data":dataname,
                "train_imgsz":args["imgsz"],"recorded_epochs":len(epochs),"seed":args["seed"],
                "optimizer":args["optimizer"],"lr0":args["lr0"],"batch":args["batch"],
                "weights_role":role,"classes":len(VIS_NAMES),"parameters":sum(p.numel() for p in original.model.parameters()),
                "saved_ultralytics":original.ckpt.get("version"),"health":"CPU_TENSOR_PASS"}
        inventory.append(item)
        private.append({**item,"file":str(path),"original_file":str(origin),
                    "original_bytes_checked":origin.is_file(),"training_dataset_argument":args["data"],
                    "init_or_model":args.get("model"),"args":str(train_args),"args_sha256":digest(train_args),
                    "training_log":str(train_log),"training_log_sha256":digest(train_log)})
        requests.append({"id":model_id,"path":str(path),"sha256":row["sha256"],"resolutions":[640,1280]})
        print(f"{model_id}: healthy, {len(epochs)} epochs logged",flush=True)
    dump(out/"checkpoint_pool.csv",inventory)
    write_json(out/"checkpoint_sources.json",private)
    return inventory, requests


def clean_history(out, inventory):
    rows, source_tables = [], {}

    def append(item, data, protocol, resolution, run, origin):
        rows.append({"checkpoint":item["checkpoint"],"checkpoint_sha256":item["sha256"],"dataset":data,
                    "protocol":protocol,"inference_size":resolution,"images":origin["image_count"],
                    **{k:origin[k] for k in ("map50","map50_95","ar50_95","gt_count")},
                    "source_run":run})

    mapping = {r["sha256"]:r for r in inventory}
    for r in table(CATALOG/"paper1_clean_metrics.csv"):
        if r["checkpoint_sha256"] not in mapping or r["size"]!="all" or r["protocol"]!="native10":
            continue
        path=ROOT/r["source_csv"]
        if path not in source_tables:
            source_tables[path]=table(path)
        original=source_tables[path][int(r["source_csv_line"])-2]
        if any(r[k]!=original[k] for k in original):
            raise ValueError("Archived metric differs from original CSV row.")
        append(mapping[r["checkpoint_sha256"]], "VisDrone DET val 548", "PAPER1_NATIVE10_CLEAN",
               r["resolution"], path.parent.name, original)
    baseline = next(r for r in inventory if r["cohort"]=="WEEK1_BASELINE")
    result = BASE/"csv/metrics_by_area.csv"
    verified(result,json.loads((BASE/"manifest.json").read_bytes())["result"]["csv_SHA256"][result.name])
    append(baseline,"VisDrone DET val 548","CV261102_NATIVE10_CLEAN",1280,BASE.name,
           next(r for r in table(result) if r["size"]=="all"))
    phase = next(r for r in inventory if r["cohort"]=="PHASE4J_NANO")
    path=ROOT/"datasets/research_archive/degradation_adaptive/outputs/phase4j"/PHASE4J
    if json.loads((path/"epoch_020_completed.json").read_bytes())["checkpoint"]["sha256"]!=phase["sha256"]:
        raise ValueError("Phase 4J epoch 20 and last SHA differ.")
    phase_csv=path/"validation/epoch_020/size_metrics.csv"
    append(phase,"VisDrone B627 development","PHASE4J_NATIVE10_CLEAN",640,PHASE4J,
           next(r for r in table(phase_csv) if r["size"]=="all"))
    dump(out/"historical_clean.csv",rows)
    return len(rows)


def timing_history(inventory):
    profiles = RESULTS/HERE.name/"uav_03_fmd_static_gap_20261124T153932_aafb49d2/development_detector_profiles.csv"
    by_sha={r["sha256"]:r for r in inventory}
    count=0
    for row in table(profiles):
        folder = RUNS/"cv261207"/row["RunID"]
        checksums={name:checksum for checksum,name in
                   (line.split(maxsplit=1) for line in (folder/"SHA256SUMS.txt").read_text().splitlines())}
        for name in ("timing.json","environment.json","configuration.json"):
            verified(folder/name,checksums[name])
        setting = json.loads((folder/"configuration.json").read_bytes())
        env = json.loads((folder/"environment.json").read_bytes())
        samples = json.loads((folder/"timing.json").read_bytes())["samples"]
        if (setting["checkpoint_sha256"]!=row["checkpoint_sha256"] or row["checkpoint_sha256"] not in by_sha
                or env["gpu"]!="NVIDIA GeForce RTX 4060 Laptop GPU" or len(samples)!=int(row["images"])
                or not math.isclose(mean(r["pipeline_ms"] for r in samples),float(row["mean_pipeline_ms"]),abs_tol=1e-9)):
            raise ValueError("Archived detector timing binding differs.")
        count+=1
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    out=output_folder(HERE.name,args.output)
    print(out,flush=True)
    runtime=out/"runtime"
    (runtime/"Ultralytics").mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES="", YOLO_CONFIG_DIR=str(runtime), YOLO_AUTOINSTALL="false",
                      YOLO_OFFLINE="true")
    inventory, requests=reusable(out)
    history=clean_history(out, inventory)
    gpu=timing_history(inventory)
    sys.path.insert(0,str(EXPS/"cv261207"))
    from supervisor_profile import profile
    request=out/"profile_inputs.json"
    raw=out/"operation_profile.json"
    write_json(request,{"checkpoints":requests})
    profile(request,raw)
    original=json.loads(raw.read_bytes())["profiles"]
    columns=("checkpoint","checkpoint_sha256","resolution","precision","checkpoint_parameters","GFLOPs_2xMAC")
    dump(out/"checkpoint_compute.csv",[{k:r[k] for k in columns} for r in original])
    write_json(out/"result.json",{"status":"PASS","models":len(inventory),"cpu_profiles":len(original),
               "historical_clean_rows":history,"historical_gpu_rows_verified":gpu,
               "new_GPU_runs":0,"official_pairs_added":0})


if __name__=="__main__":
    main()
