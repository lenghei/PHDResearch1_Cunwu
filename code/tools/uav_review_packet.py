"""Stage verified Week 1–4 tables for the four local result directories."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
from uav_week01_06 import AUDIT, BASE, EXP_IDS, RUNS, digest, get_fmp, load, require
from uav_week_public import validated_file


FIRST = (
    "uav_01_baseline_20261124T154238_4a518050",
    "uav_02_fmp_20261124T153924_1245ff9f",
    "uav_03_fmd_static_gap_20261124T153932_aafb49d2",
    "uav_04_sampling_20261124T153934_fb755db2",
)
FOLLOWUP = RUNS/"checks/uav_week01_06_20261125"
EXPLORE = RUNS/"uav_05_evaluation/exploratory_executions/uav05-9fe57f16643041aaa41f00ef4e396080"


def emit_json(path, data):
    with path.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(data,f,indent=2,allow_nan=False)
        f.write("\n")


def emit_csv(path, rows):
    with path.open("x",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def emit_copy(src,dst,expected):
    require(digest(src)==expected, f"Source sha differs for {src.name}")
    with src.open("rb") as a,dst.open("xb") as b:
        shutil.copyfileobj(a,b)


def manifest(out, week, sources):
    emit_json(out/"provenance.json",{"week":week,"type":"verified_evidence_export",
        "sources":[{"source_id":p.parent.name,"artifact":p.name,"sha256":digest(p)} for p in sources],
        "files":{p.name:digest(p) for p in out.iterdir() if p.is_file()}})


def packet(staging, evidence):
    baseline, p_run, d_run, sample = [RUNS/name/run for name,run in zip(EXP_IDS[:4],FIRST,strict=True)]
    out1,out2,out3,out4 = [staging/name for name in EXP_IDS[:4]]
    for out in (out1,out2,out3,out4):
        out.mkdir()
    train = load(BASE/"manifest.json")
    results = train["result"]
    replay = load(evidence/"week1_replay.json")
    require(replay["status"]=="PASS" and replay["coco_recomputed"] and replay["RunID"]==BASE.name
            and (replay["images"],replay["epochs"])==(548,100)
            and replay["AP50"]==results["AP50"] and replay["AP50_95"]==results["AP50_95"],
            "Unexpected Week 1 replay")
    paths1=[BASE/"manifest.json", evidence/"week1_replay.json", baseline/"ignore_and_gt_replay.json"]
    for title in ("metrics_by_area.csv","metrics_by_class.csv"):
        p=BASE/"csv"/title
        emit_copy(p,out1/title,results["csv_SHA256"][title])
        paths1.append(p)
    proof = load(BASE/"coco_regeneration.json")
    emit_json(out1/"replay.json",{**replay,"AP50_95":results["AP50_95"],"AR50_95":results["AR50_95"],
        "original_ground_truth_sha256":proof["GT_SHA256"], "coco_prediction_sha256":proof["predictions_COCO_SHA256"],
        "full_validation_native10":True})
    manifest(out1,1,paths1+[BASE/"coco_regeneration.json"])

    params, params_sha = validated_file(p_run,"feature_parameters.csv")
    receipt=load(validated_file(FOLLOWUP,"rolling_geometry.json")[0])
    demo_path=evidence/"week2_demo/demo_receipt.json"
    demo=load(demo_path)
    require(receipt["images"]==548 and receipt["source_archive_sha256"]==demo["source_archive_sha256"]
            and demo["repeated_pixel_identity"] and digest(demo_path.with_name("preview.png"))==demo["preview_sha256"],
            "Unexpected Week 2 demo or validation")
    require(digest(demo_path.with_name("ground_truth.json"))==demo["ground_truth_sha256"],
            "Week 2 demo ground truth differs")
    require(digest(params)==params_sha, "Feature parameter table differs")
    with params.open(encoding="utf-8-sig", newline="") as stream:
        feature_rows = list(csv.DictReader(stream))
    for row in feature_rows:
        row.pop("author_review", None)
        if row["feature"] == "RollingShutter":
            row["interpretation"] = "Sinusoidal row shift with derived boxes and ignore masks."
            row["geometry"] = "DERIVED_GT_AND_IGNORE"
    emit_csv(out2/params.name, feature_rows)
    fields=("images","original_retained_targets","warped_targets","clipped_out_targets",
        "images_with_verified_ignore_union","logical_configurations","geometry_supported_configurations",
        "unsupported_geometry_configurations","source_archive_sha256","derived_gt_sha256","seed","elapsed_seconds")
    emit_json(out2/"geometry_validation.json", {k:receipt[k] for k in fields})
    emit_json(out2/"demo.json", {k:demo[k] for k in ("configuration_id","leaves","selection_seed","selection_mode","render_seed",
        "population","annotation_counts","pixel_sha256","preview_sha256","repeated_pixel_identity")})
    manifest(out2,2,[params,FOLLOWUP/"rolling_geometry.json",demo_path])

    attr_path=validated_file(FOLLOWUP,"five_detector_attributes.json")[0]
    attrs=load(attr_path)
    first=load(EXPLORE/"verified_summary.json")
    require(first["scope"]=="EXPLORATORY_FIRST_PAIR_ONLY" and first["repeat_verified"]
            and digest(EXPLORE/"result.json")==first["result_sha256"],"Unexpected Week 3 timing evidence")
    detector_id=first["pair_id"].split("__")[1]
    reference=attrs["detectors"][0]
    binding_fields=("family","train_resolution","train_epochs_logged","train_seed","train_batch","inference")
    require(all(all(d[k]==reference[k] for k in binding_fields) for d in attrs["detectors"]),
            "Frozen detector training or inference settings differ")
    profiles=[]
    for d in attrs["detectors"]:
        same=d["detector_id"]==detector_id and d["checkpoint_sha256"]==first["checkpoint_sha256"]
        profiles.append({"detector_id":d["detector_id"],"checkpoint":d["checkpoint"],"checkpoint_sha256":d["checkpoint_sha256"],
            "resolution":d["resolution"],"precision":d["precision"],"parameters":d["checkpoint_parameters"],
            "gflops_2xmac":d["GFLOPs_2xMAC"],"gpu_scope":"EXPLORATORY_TEST_DEV_FIRST_PAIR" if same else "UNMEASURED_MATCHED_SCOPE",
            "cached_pipeline_mean_ms":first["cached_pipeline_mean_ms"][0] if same else "",
            "repeat_cached_mean_ms":first["cached_pipeline_mean_ms"][1] if same else "",
            "peak_allocated_bytes":first["peak_allocated_bytes"][0] if same else ""})
    require(len(profiles)==5 and sum(p["gpu_scope"]=="EXPLORATORY_TEST_DEV_FIRST_PAIR" for p in profiles)==1,
            "Frozen and measured detector IDs do not join uniquely")
    emit_csv(out3/"frozen_detector_attributes.csv", profiles)
    counts=load(AUDIT/"static_gap.json")
    leaf=get_fmp()
    population=list(leaf.combinations())
    flags={name:set(value["covered_leaves"]) for name,value in counts["definitions"].items()}
    independently_counted={name:sum(set(p)<=selected for p in population) for name,selected in flags.items()}
    require(len(population)==46368 and independently_counted==attrs["static_policies"],"Static rule count differs")
    emit_csv(out3/"static_rules_by_leaf.csv", [{"leaf":name,"family":x["family"],
        "support":x["classification"], "reference_transform":x["transform"] or "",
        "rule_explanation":x["rationale"],
        **{rule:name in allowed for rule,allowed in flags.items()}} for name,x in sorted(counts["leaves"].items())])
    logic,_=validated_file(d_run,"flamapy_verification.json")
    emit_json(out3/"model_and_policies.json",{
        "fmd_logic":load(logic),"static_policy_counts":independently_counted,
        "fmp_population":len(population),"ultralytics_plan":"8.4.33",
        "shared_detector_settings":{k:reference[k] for k in binding_fields},
        "historical_training_ultralytics":attrs["training_logged_ultralytics"],
        "historical_albumentations_activation":attrs["historical_albumentations_activation"],
        "separate_legacy_gpu_pilot":attrs["comparison_sources"]["prior_GPU"]})
    manifest(out3,3,[attr_path,AUDIT/"static_gap.json",logic,EXPLORE/"verified_summary.json",EXPLORE/"reproduced.json"])

    week4=[]
    for fn in ("p_configurations.csv","d_configurations.csv","sampling_coverage.csv","first500_pairs_PLAN_ONLY.csv"):
        src,sha=validated_file(sample,fn)
        emit_copy(src,out4/fn,sha)
        week4.append(src)
    manifest(out4,4,week4)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging",type=Path,required=True)
    parser.add_argument("--evidence",type=Path,required=True)
    args=parser.parse_args()
    args.staging.mkdir(parents=True)
    packet(args.staging,args.evidence)
