"""Test the static gap proxy against verified development measurements."""
import argparse
from collections import Counter
import csv
import json
from math import isclose
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
MAIN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAIN / "code/tools"))
from uav_reproduction import ROOT, RUNS, digest, output_folder, write_json
from uav_week01_06 import pair_status, require

FMD_FIXED = {
    "ck01_Infer640", "ck01_Infer1280", "ck05_Infer640", "ck05_Infer1280"
}
NORMAL = "p_cdbfeb94b010c58d"
SCENARIOS = [(tau, limit) for tau in (.05, .10, .15) for limit in (50, 100)]
VM36 = RUNS / "vm36_supervisor_reproduction/vm36_supervisor_reproduction_normal4_analysis_20261110T234523_1ce40f03"
ARCHIVE = ROOT / "datasets/research_archive/paper1"
STATIC = MAIN / "experiments/uav_03_fmd_static_gap/coverage_rules.json"
ADMISSION = RUNS / "uav_05_evaluation/uav_05_evaluation_20261124T153937_5f1368ab/test_dev_admission.json"


def load(path):
    return json.loads(path.read_bytes())


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(out, name, rows):
    with (out / name).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def recorded_file(run, name):
    want = {n.removeprefix("*"): h for h, n in
            (line.split(maxsplit=1) for line in (run / "SHA256SUMS.txt").read_text().splitlines())}
    path = run / name
    require(want.get(name) == digest(path), f"Run checksum differs: {path}")
    return path


def interval(values):
    return tuple(float(v) for v in np.quantile(values, (.025, .975)))


def evidence(source_dir):
    manifest = recorded_file(source_dir, "METRIC_PROVENANCE.json")
    provenance = load(manifest)["pairs"]
    require(len(provenance) == 84, "Expected 84 accepted VM36 cells.")
    found = {}
    images = tuple(range(1, 549))
    indices = np.random.default_rng(20261108).integers(0, 548, (200, 548))
    source_members, gt_sha, pixels = None, None, {}
    model_shas = {}
    for item in provenance:
        folder = Path(item["source_folder"]).resolve(strict=True)
        require(folder.is_relative_to(source_dir.parent), "Metric reference escapes VM36 archive.")
        for name, sha in item["SHA256"].items():
            require(Path(name).name == name and digest(folder/name) == sha,
                    f"Pair checksum differs: {folder.name}/{name}")
        binding = load(folder / "configuration.json")
        p, d = binding["p"], binding["d"]
        require(item["pair_id"] == binding["id"] == f"{p['id']}__{d['id']}" == folder.name
                and d["id"] in FMD_FIXED and binding["images"] == 548
                and d["precision"] == "FP32", "Invalid experimental binding.")
        sha = item["SHA256"]["ground_truth.json"]
        require(gt_sha is None or gt_sha == sha, "Matched GT differs.")
        gt_sha = sha
        expected_sha = model_shas.setdefault(d["checkpoint"], d["sha256"])
        require(expected_sha == d["sha256"], "Checkpoint SHA varies within model.")
        records = [json.loads(line) for line in (folder/"image_manifest.jsonl").read_bytes().splitlines()]
        members = tuple((r["image_id"], r["source"]["image_SHA256"], r["source"]["annotation_SHA256"])
                        for r in records)
        require(tuple(r[0] for r in members) == images and (source_members is None or source_members == members),
                "Matched source image or annotation differs.")
        source_members = members
        this_pixels = tuple(r["pixel_sha256"] for r in records)
        require(pixels.setdefault(p["id"], this_pixels) == this_pixels,
                "Detectors did not share the same P pixels.")
        bootstrap = load(folder/"bootstrap.json")
        draws = bootstrap["draws"]
        require(tuple(draws["image_ids"]) == images and draws["samples"] == 200 and draws["seed"] == 20261108
                and draws["sampling"] == "numpy.default_rng.integers, sorted source image positions",
                "Bootstrap positions, seed or count differ.")
        ap = np.asarray(draws["values"]["tiny"]["map50_95"], dtype=float)
        ai = next(r["ci95"] for r in bootstrap["intervals"] if r["size"] == "tiny" and r["metric"] == "map50_95")
        timing = load(folder/"timing.json")
        require(tuple(r["image_id"] for r in timing["samples"]) == images
                and timing["gpu"] == "NVIDIA GeForce RTX 4060 Laptop GPU" and timing["FP32"] and not timing["TF32"]
                and ap.shape == (200,) and np.isfinite(ap).all() and np.allclose(interval(ap), ai),
                "Saved image interval or timing record differs.")
        ms = np.asarray([r["pipeline_ms"] for r in timing["samples"]])
        require(np.isfinite(ms).all() and (ms >= 0).all()
                and isclose(ms.mean(), timing["mean_pipeline_ms"]), "Invalid pipeline time.")
        ld = ms[indices].mean(axis=1)
        metrics = load(folder/"metrics.json")
        tiny = next(r for r in metrics["per_area"] if r["size"] == "tiny")
        require(tiny["image_count"] == 548 and tiny["gt_count"] == 11842
                and tiny["evaluated_categories"] == 10 and tiny["max_det"] == 3000
                and load(folder/"verification.json")["COCO_recomputed_from_raw"],
                "Tiny AP protocol differs.")
        key = p["id"], d["id"]
        require(key not in found, "Duplicate P–D pair.")
        found[key] = {"ap": tiny["map50_95"], "ms": float(ms.mean()), "ap_ci": tuple(ai),
                      "ms_ci": interval(ld), "ap_draws": ap, "ms_draws": ld, "leaves": p["leaves"],
                      "source_run": folder.parents[1].name}
    require(len(found) == 84 and len(pixels) == 21 and NORMAL in pixels
            and all((p,d) in found for p in pixels for d in FMD_FIXED), "Incomplete VM36 matrix.")
    return found, {"pairs":84,"source_manifest_sha256":digest(manifest),"GT_SHA256":gt_sha,
                   "checkpoint_SHA256":model_shas,"images":548,"bootstrap_seed":20261108,"draws":200}


def analyze(found, source_dir):
    p_ids = sorted({p for p, _ in found})
    rules = load(STATIC)["covered_levels"]
    policies = {"identity": {"Normal"},
                "brightness_only": {"Normal","Overexposed_mild","Underexposed_mild"},
                "plan_proxy": {leaf for values in rules.values() for leaf in values}}
    reference = {(row["P"],row["D"],float(row["tau"]),int(row["L_cached_ms"])): row
                 for row in read_csv(recorded_file(source_dir,"PAIR_STATES.csv"))}
    empirical, confusion, detectors, dominance, joint = [], [], [], [], []
    for tau, limit in SCENARIOS:
        outcomes, flags = {}, []
        states_by_d = {d:[] for d in sorted(FMD_FIXED)}
        for p in p_ids:
            cells = {d:found[p,d] for d in sorted(FMD_FIXED)}
            leaves = next(iter(cells.values()))["leaves"]
            require(all(r["leaves"] == leaves for r in cells.values()), "P recipe differs within a comparison.")
            classified = {d:pair_status(r["ap_ci"],r["ms_ci"],tau,limit) for d,r in cells.items()}
            for d,state in classified.items():
                require(reference[p,d,tau,limit]["latency_interval_state"] == state, "Historical oracle differs.")
                states_by_d[d].append(state)
            state = "COVERED" if "PASS" in classified.values() else (
                    "GAP_TESTED_D" if all(s=="FAIL" for s in classified.values()) else "UNRESOLVED")
            outcomes[p] = state
            passes = [d for d,s in classified.items() if s=="PASS"]
            coverage_draws = np.logical_or.reduce([(r["ap_draws"]>=tau)&(r["ms_draws"]<=limit)
                                                  for r in cells.values()])
            flags.append(coverage_draws)
            empirical.append({"P":p,"features":"|".join(leaves),"tau":tau,"latency_limit_ms":limit,
                              "static_covered":int(set(leaves).issubset(policies["plan_proxy"])),
                              "empirical_status":state,"pass_D":"|".join(passes),
                              "fail_D":sum(s=="FAIL" for s in classified.values()),
                              "uncertain_D":sum(s=="UNCERTAIN" for s in classified.values()),
                              "draw_fraction_covered":float(coverage_draws.mean())})
        gap_count_draws = len(p_ids)-np.sum(flags,axis=0)
        l,h = interval(gap_count_draws)
        hist = Counter(outcomes.values())
        joint.append({"tau":tau,"latency_limit_ms":limit,"P":len(p_ids),"covered":hist["COVERED"],
                      "gap_tested_D":hist["GAP_TESTED_D"],"unresolved":hist["UNRESOLVED"],
                      "bootstrap_gap_count_median":float(np.median(gap_count_draws)),
                      "bootstrap_gap_count_low":l,"bootstrap_gap_count_high":h,"paired_image_draws":200})
        for policy, allowed in policies.items():
            counts = Counter()
            for p,status in outcomes.items():
                leaves = found[p,sorted(FMD_FIXED)[0]]["leaves"]
                prediction = "COVERED" if set(leaves).issubset(allowed) else "GAP_TESTED_D"
                counts[prediction,status] += 1
            tp, fp = counts["GAP_TESTED_D","GAP_TESTED_D"], counts["GAP_TESTED_D","COVERED"]
            fn, tn = counts["COVERED","GAP_TESTED_D"], counts["COVERED","COVERED"]
            decisive = tp+fp+fn+tn
            confusion.append({"policy":policy,"tau":tau,"latency_limit_ms":limit,
                "static_covered_P":sum(v for (pred,_),v in counts.items() if pred=="COVERED"),
                "decisive_P":decisive,"unresolved_P":hist["UNRESOLVED"],"TP_gap":tp,"FP_gap":fp,
                "FN_gap":fn,"TN_gap":tn,"accuracy":(tp+tn)/decisive if decisive else None,
                "always_gap_accuracy":(tp+fn)/decisive if decisive else None,
                "balanced_accuracy":(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,
                "gap_precision":tp/(tp+fp) if tp+fp else None,
                "covered_recall":tn/(tn+fp) if tn+fp else None})
        for d, states in states_by_d.items():
            unique = sum(state=="PASS" and sum(states_by_d[other][i]=="PASS" for other in states_by_d)==1
                         for i,state in enumerate(states))
            detectors.append({"D":d,"tau":tau,"latency_limit_ms":limit,
                "confirmed_pass":states.count("PASS"),"confirmed_fail":states.count("FAIL"),
                "uncertain":states.count("UNCERTAIN"),"unique_confirmed_pass":unique,
                "tested_P":len(p_ids)})
    for a in sorted(FMD_FIXED):
        for b in sorted(FMD_FIXED-{a}):
            pairs = [(found[p,a],found[p,b]) for p in p_ids]
            weak = [x["ap_ci"][0]>=y["ap_ci"][1] and x["ms_ci"][1]<=y["ms_ci"][0] for x,y in pairs]
            strict = any(x["ap_ci"][0]>y["ap_ci"][1] or x["ms_ci"][1]<y["ms_ci"][0] for x,y in pairs)
            dominance.append({"candidate":a,"reference":b,"pointwise_interval_dominating_P":sum(weak),
                              "tested_P":len(p_ids),"all_tested_P_dominance":int(all(weak) and strict)})
    return empirical, confusion, detectors, dominance, joint



def archive_interactions():
    v3 = ARCHIVE / "ieee_results_v3"
    lower = ARCHIVE / "ieee_results_resolution_640"
    mf = [base/"dataset_manifest.json" for base in (v3, lower)]
    protocols = [base/"protocol.json" for base in (v3, lower)]
    require(load(mf[0]) == load(mf[1]), "Historical image manifests differ.")
    require(all({k:v for k,v in load(p).items() if k!="bootstrap"} ==
                {k:v for k,v in load(protocols[0]).items() if k!="bootstrap"} for p in protocols),
            "Historical evaluator protocols differ.")
    conditions = sorted(load(mf[0])["conditions"])
    ci_file = v3/"paired_resolution_ci.csv"
    leaf = "leaf_yolo_n_external"
    intervals = {r["condition"]:r for r in read_csv(ci_file)
                 if (r["left_model"],r["right_model"],r["left_resolution"],r["right_resolution"],
                     r["protocol"],r["size"],r["metric"],r["same_checkpoint"]) ==
                    (leaf,leaf,"1280","640","native10","all","map50_95","True")}
    require(set(intervals) == set(conditions), "Incomplete LEAF paired intervals.")
    results, summary = [], []
    settings = ("backend","precision","preprocessing","conf","iou_nms","multi_label_nms",
                "global_evaluation_max_det","class_names","inference_head")
    for model in (leaf, "yolov8s_ft", "yolo26s_ft"):
        model_rows = []
        for condition in conditions:
            files = [(v3 if s==1280 or model==leaf else lower)/model/str(s)/condition for s in (640,1280)]
            raw = [load(f/"raw_inference.json") for f in files]
            require(all(raw[0][k]==raw[1][k] for k in (*settings,"checkpoint_sha256"))
                    and all(len(r["per_image_counts"])==548 for r in raw),
                    f"Unmatched checkpoint or inference settings: {model}/{condition}")
            metric_files = [f/"native10/metrics.json" for f in files]
            m = [next(r for r in load(f) if r["size"]=="all") for f in metric_files]
            require(all(r["checkpoint_sha256"]==a["checkpoint_sha256"] and r["model"]==model
                        and r["condition"]==condition and r["resolution"]==s and r["image_count"]==548
                        and r["gt_count"]==38593 and r["max_det"]==3000 and r["evaluated_categories"]==10
                        and r["area_reference"]=="original_image_pixels" for r,a,s in zip(m,raw,(640,1280))),
                    "Historical metric binding differs.")
            gt = [f/"native10/ground_truth.json" for f in files]
            require(digest(gt[0])==digest(gt[1]), "Historical matched GT differs.")
            x,y = (r["map50_95"] for r in m)
            ci = intervals[condition] if model==leaf else None
            if ci is not None:
                require(isclose(float(ci["delta"]), y-x) and isclose(float(ci["right_estimate"]),x)
                        and isclose(float(ci["left_estimate"]),y),
                        "LEAF interval differs from source point estimate.")
            paired = ci is not None and ci["ci_status"]=="available" and ci["n_valid_paired_draws"]=="1000"
            model_rows.append({"model":model,"condition":condition,"ap_640":x,"ap_1280":y,"delta":y-x,
                "paired_ci95_low":float(ci["ci_low"]) if paired else None,
                "paired_ci95_high":float(ci["ci_high"]) if paired else None,
                "candidate_pool_640":raw[0]["candidate_pool"],"candidate_pool_1280":raw[1]["candidate_pool"],
                "matched_inference_settings":raw[0]["candidate_pool"]==raw[1]["candidate_pool"],
                "checkpoint_sha256":raw[0]["checkpoint_sha256"],"gt_sha256":digest(gt[0]),
                "metric_640":metric_files[0].relative_to(ROOT).as_posix(),"metric_sha_640":digest(metric_files[0]),
                "metric_1280":metric_files[1].relative_to(ROOT).as_posix(),"metric_sha_1280":digest(metric_files[1])})
        clean = next(r for r in model_rows if r["condition"]=="CLEAN")
        degraded = [r for r in model_rows if r is not clean]
        for row in model_rows:
            row["clean_adjusted_delta"] = row["delta"]-clean["delta"]
        worst640 = min(degraded,key=lambda r:r["ap_640"])
        worst1280 = min(degraded,key=lambda r:r["ap_1280"])
        summary.append({"model":model,"mean_degraded_640":float(np.mean([r["ap_640"] for r in degraded])),
            "mean_degraded_1280":float(np.mean([r["ap_1280"] for r in degraded])),
            "mean_degraded_delta":float(np.mean([r["delta"] for r in degraded])),
            "clean_delta":clean["delta"],"worst_640":worst640["condition"],"worst_ap_640":worst640["ap_640"],
            "worst_1280":worst1280["condition"],"worst_ap_1280":worst1280["ap_1280"],
            "matched_inference_settings":all(r["matched_inference_settings"] for r in model_rows)})
        results.extend(model_rows)
    return results, summary, {p.relative_to(ROOT).as_posix():digest(p) for p in (*mf,*protocols,ci_file)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cells, provenance = evidence(VM36)
    empirical, confusion, detectors, dominance, sensitivity = analyze(cells,VM36)
    archive, rank, archive_sources = archive_interactions()
    provenance.update({"historical_resolution_inputs":archive_sources,"static_policy_sha256":digest(STATIC),
                       "test_dev_admission_status":load(ADMISSION)["status"],
                       "scope":"DEVELOPMENT_ONLY; fixed 21 P and four D; no population inference"})
    out = output_folder("uav_06_gap_analysis",args.output)
    for name, records in (("static_empirical",empirical),("static_agreement",confusion),
                          ("detectors",detectors),("dominance",dominance),("threshold_sensitivity",sensitivity),
                          ("historical_resolution",archive),("historical_resolution_summary",rank)):
        write_csv(out,name+".csv",records)
    write_json(out/"sources.json",provenance)
    with (out/"SHA256SUMS.txt").open("x",encoding="utf-8") as stream:
        for f in sorted(out.glob("*")):
            if f.name != "SHA256SUMS.txt":
                stream.write(f"{digest(f)}  {f.name}\n")
    print(out)
    print(next(r for r in confusion if r["policy"]=="plan_proxy" and r["tau"]==.10 and r["latency_limit_ms"]==100))
    print(rank)


if __name__ == "__main__":
    main()
