"""Export verified zero-shot coverage and inference-option results."""
import argparse
import csv
from datetime import datetime, timezone
from math import isclose
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory

from validate_checkpoints import FILES, MAIN, digest, load, percentile95, require, verify, write_json
from validate_zeroshot import MODELS, ZERO
from profile_inference_options import CASES, operations, read_models


def table(path):
    with path.open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def proof(path):
    return {"artifact":path.name,"SHA256":digest(path)}


def completed(run, count):
    r = load(run/"result.json")
    supervisor = load(run/"supervisor.json")
    require(r["RunID"]==run.name and r["status"]=="COMPLETE" and supervisor["returncode"]==0
            and supervisor["stop_reason"] is None and count in (r.get("cells"),r.get("cases")),
            f"Run did not pass: {run.name}")


def collect_profiles(runs, members_sha):
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    records, sources = {}, []
    for run in runs:
        plan = load(run/"plan.json")
        guard = load(run/"supervisor.json")
        require(plan["members_sha256"]==members_sha and plan["images"]==548 and not guard["stop_reason"],
                "Resource profile input or thermal guard differs.")
        if guard["returncode"] == 0:
            completed(run,len(plan["order"]))
        else:
            require(guard["returncode"]==1 and
                    "ValueError: Live tensors remained from the preceding case." in (run/"worker.log").read_text(encoding="utf-8"),
                    "Unexpected resource profile failure.")
        artifacts = []
        for name in plan["order"]:
            path = run/(name+".json")
            if not path.exists() and guard["returncode"]!=0:
                continue
            require(name in CASES and name not in records,"Unexpected or repeated resource case.")
            record = load(path)
            require(record["summary"]["case"]==name
                    and record["summary"]["FM_D_configuration_id"]==plan["cases"][name]["configuration_id"],
                    "Resource case identity differs.")
            records[name] = record
            artifacts.append(proof(path))
        sources.append({"kind":"inference_options","RunID":run.name,
                        "plan":proof(run/"plan.json"),"supervisor":proof(run/"supervisor.json"),"cells":artifacts})
    require(set(records)==set(CASES),"Incomplete resource case coverage.")
    member_ids, profiles, rows = None, {}, []
    models = read_models()
    for name in CASES:
        record = records[name]
        row, samples = record["summary"], record["samples"]
        ids = [s["image_id"] for s in samples]
        if member_ids is None:
            member_ids = ids
        entry = models[row["checkpoint"]]
        require(row["allocated_before_case_bytes"]==0 and row["checkpoint_sha256"]==entry["sha256"]
                and len(ids)==len(set(ids))==548 and ids==member_ids, "Resource profile population differs.")
        for field in ("predict","pipeline"):
            vals = [s[field+"_ms"] for s in samples]
            for statistic, function in (("mean",mean),("median",median),("p95",percentile95)):
                require(isclose(function(vals),row[f"{field}_{statistic}_ms"]),"Latency did not reproduce.")
        row["mean_GFLOPs_2xMAC"] = operations(row,samples,entry,torch,YOLO,profiles)
        rows.append(row)
    return rows, sources


def export(zero, profiles, static, existing, out, complements=()):
    completed(zero,4)
    pre = load(static)
    require(pre["source_version"]=="8.4.33" and sorted(pre["coverage_counts"].values())==[1,3,18],
            "Source and static coverage record differs.")
    zero_plan = load(zero/"inputs.json")
    require(zero_plan["protocol"]==ZERO,"Zero-shot protocol differs.")
    latency, profile_sources = collect_profiles((profiles,*complements),zero_plan["members_sha256"])
    rows = table(zero/"zeroshot_metrics.csv")
    metadata = table(zero/"checkpoint_metadata.csv")
    classes = table(zero/"zeroshot_per_class.csv")
    require(list(r["checkpoint"] for r in rows)==list(MODELS) and len(metadata)==4 and len(classes)==48,
            "Incomplete or duplicated rows.")
    sources = [{"kind":"zero_shot","RunID":zero.name,"result":proof(zero/"result.json"),
                "supervisor":proof(zero/"supervisor.json"),"cells":[]},*profile_sources]
    for r in rows:
        name = r["checkpoint"]
        binding_sha = zero_plan["ground_truth_sha256"]
        cell = zero/name/"infer640"
        verify(cell,FILES,{"checkpoint":name,"resolution":640,"images":548})
        binding = load(cell/"configuration.json")
        summary = load(cell/"summary.json")
        require(binding["gt_sha256"]==binding_sha==digest(zero/"ground_truth.json")
                and binding["checkpoint_sha256"]==MODELS[name][2]==r["checkpoint_sha256"]
                and binding["protocol"]==ZERO and all(str(value)==r[key] for key,value in summary.items()),
                "Zero-shot result provenance differs.")
        sources[0]["cells"].append({"artifact":f"{name}/infer640/COMPLETE.json",
                                    "SHA256":digest(cell/"COMPLETE.json")})
    roster = []
    old = table(existing/"checkpoint_training.csv")
    require(len(old)==8 and sum(r["FM_D_architecture_scale_eligible"]=="YES" for r in old)==7,
            "Earlier checkpoint roster differs.")
    for r in old:
        roster.append({"checkpoint":r["checkpoint"],"weights":"VisDrone_FT","training_data":r["training_data"],
                       "checkpoint_sha256":r["checkpoint_sha256"],
                       "counted_in_FM_D_pool":r["FM_D_architecture_scale_eligible"],"evaluation_classes":"native10"})
    for r in metadata:
        name = r["checkpoint"]
        require(r["sha256"]==MODELS[name][2] and Path(r["pretrained_data_argument"]).name=="coco.yaml",
                "Pretrained checkpoint metadata differs.")
        r["pretrained_data_argument"]="coco.yaml"
        roster.append({"checkpoint":name,"weights":"COCO_ZeroShot","training_data":"COCO",
                       "checkpoint_sha256":r["sha256"],"counted_in_FM_D_pool":"YES","evaluation_classes":"common6"})
    require(len({r["checkpoint_sha256"] for r in roster})==12,"Duplicate checkpoint identity.")
    sources.extend([{"kind":"native10_training_roster","RunID":existing.name,
                     **proof(existing/"checkpoint_training.csv")},
                    {"kind":"static_source_audit",**proof(static)}])

    require(not out.exists(),"Output directory already exists.")
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage=Path(tmp)/out.name
        stage.mkdir()
        for name,data in (("checkpoint_roster.csv",roster),("zeroshot_checkpoints.csv",metadata),
                          ("zeroshot_metrics.csv",rows),("zeroshot_per_class.csv",classes),
                          ("inference_options.csv",latency)):
            with (stage/name).open("x",encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        identity={"ExperimentID":"uav_03_fmd_static_gap","RunID":profiles.name,"status":"COMPLETE_WITH_LIMITATIONS",
                  "mode":"WEEK3_ZERO_SHOT_AND_INFERENCE_OPTIONS","scope":"DEVELOPMENT_VALIDATION"}
        summary={**identity,"images_per_condition":548,
                 "checkpoints":[{"eligible_checkpoints":11,"external_controls":1,"zero_shot_checkpoints":4}],
                 "inference_profiles":[{"new_profiles":10,"images_each":548,"scope":"RESOURCE_ONLY",
                     "INT8":"NOT_ADMITTED_NO_VERIFIED_CALIBRATED_BACKEND"}],
                 "static_coverage":[{"identity":1,"brightness_only":3,"plan_proxy":18,"population":46368}]}
        write_json(stage/"snapshot.json",{**identity,"files":{"summary.json":summary,"result.json":identity}})
        write_json(stage/"manifest.json",{**identity,"schema":2,"bundle":"snapshot.json",
            "started_at":datetime.fromtimestamp((profiles/"plan.json").stat().st_mtime,timezone.utc).isoformat(),
            "SHA256":digest(stage/"snapshot.json"),"sources":sources,
            "files":{p.name:digest(p) for p in stage.iterdir()}})
        stage.rename(out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("zero","profiles","static","existing"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--profile-complement",type=Path,action="append",default=[])
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    export(args.zero,args.profiles,args.static,args.existing,args.output
           or MAIN/"reports/results/uav_03_fmd_static_gap"/args.profiles.name,args.profile_complement)
