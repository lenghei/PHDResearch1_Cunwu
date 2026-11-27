"""Verify saved 548-image cells and calculate Gap on the frozen detector set."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np

import run_development as dev
from uav_week01_06 import gap_status, pair_status

THRESHOLDS = tuple((t,l) for t in (.05,.10,.15) for l in (50,100))
REFERENCE = (.10,100)
SETTINGS = ((640,"FP32"),(960,"FP32"),(1280,"FP32"),(640,"FP16"),(1280,"FP16"))
CONTRASTS = ((SETTINGS[2], SETTINGS[0]),(SETTINGS[1], SETTINGS[0]),(SETTINGS[2], SETTINGS[1]),
             (SETTINGS[3], SETTINGS[0]),(SETTINGS[4], SETTINGS[2]))


def read_cells(matrix, protocol, runs):
    cells = {}
    image_ids = list(range(1,dev.IMAGES+1))
    indices = np.random.default_rng(protocol["bootstrap_seed"]).integers(0,dev.IMAGES,(200,dev.IMAGES))
    for specs in matrix:
        shared = set()
        for spec in specs:
            name = spec["pair_id"]
            matches = [(run,run/"pairs"/name) for run in runs if (run/"pairs"/name).is_dir()]
            if not matches:
                continue
            dev.require(len(matches)==1,f"Duplicate sealed source: {name}")
            run, pair = matches[0]
            binding = dev.load(pair/"configuration.json")
            dev.require(binding["spec"]==spec and all(binding[k]==v for k,v in protocol.items()),"Pair binding differs.")
            seal = dev.verify_pair(pair,binding)
            dev.require(seal["sha256"]["ground_truth.json"]==binding["gt_sha256"],"GT checksum differs.")
            shared.add((binding["gt_sha256"],binding["cache_manifest_sha256"]))
            raw = dev.load(pair/"metrics.json")
            areas = {r["size"]:r for r in raw["per_area"]}
            dev.require(all(r["image_count"]==dev.IMAGES and r["evaluated_categories"]==10
                            and r["area_reference"]=="original_image_pixels" and r["max_det"]==3000
                            and r["map50_95"] is not None for r in areas.values()),"Metric scope differs.")
            boot = dev.load(pair/"bootstrap.json")
            timing = dev.load(pair/"timing.json")
            dev.require(boot["image_ids"]==[r["image_id"] for r in timing["samples"]]==image_ids
                        and boot["samples"]==200 and boot["seed"]==protocol["bootstrap_seed"],"Unmatched resamples.")
            with gzip.open(pair/"bootstrap_indices.npy.gz","rb") as stream:
                dev.require(np.array_equal(np.load(stream,allow_pickle=False),indices),"Bootstrap indices differ.")
            draws = boot["draws"]["values"]
            estimates = {k: np.asarray(draws[k]["map50_95"]) for k in ("all","tiny")}
            dev.require(all(len(a)==200 and np.isfinite(a).all() for a in estimates.values()),"Invalid AP draws.")
            tiny_ci = tuple(map(float,np.quantile(estimates["tiny"],(.025,.975))))
            latency_ci = tuple(timing["cached_pipeline_mean_ci95_ms"])
            roll = np.array([r["pipeline_ms"] for r in timing["samples"]])
            dev.require(np.allclose(latency_ci,np.quantile(roll[indices].mean(axis=1),(.025,.975))),
                        "Latency interval does not match the saved draws.")
            d = spec["detector"]
            cells[name] = {"pair_id":name,"P":spec["degradation"]["degradation_id"],"D":d["detector_id"],
                "resolution":d["resolution"],"precision":d["precision"],"images":dev.IMAGES,
                "AP50":areas["all"]["map50"],"AP50_95":areas["all"]["map50_95"],"AR50_95":areas["all"]["ar50_95"],
                "Tiny_AP50_95":areas["tiny"]["map50_95"],"Tiny_low":tiny_ci[0],"Tiny_high":tiny_ci[1],
                "latency_ms":timing["cached_pipeline_mean_ms"],"latency_low":latency_ci[0],"latency_high":latency_ci[1],
                "bootstrap_seconds":timing["bootstrap_seconds"], "pair_seconds":timing["wall_seconds_to_sealing_including_setup"],
                "predictions_sha256":seal["sha256"]["predictions.jsonl.gz"],"gt_sha256":binding["gt_sha256"],
                "source_RunID":run.name,"draws":estimates}
        dev.require(len(shared)<=1,"The detectors did not share the same GT and cache.")
    return cells


def analyze(matrix, cells, allowed):
    pair_rows, p_rows, counts, paired, missing = [],[],[],[],[]
    for order, specs in enumerate(matrix,1):
        p = specs[0]["degradation"]
        group = [cells.get(s["pair_id"]) for s in specs]
        measured = [r for r in group if r is not None]
        for spec, cell in zip(specs,group):
            if cell is None:
                missing.append({"P_order":order,"pair_id":spec["pair_id"],
                    "features":"|".join(p["leaves"]),"resolution":spec["detector"]["resolution"],
                    "precision":spec["detector"]["precision"]})
        for tau,limit in THRESHOLDS:
            states = ["UNMEASURED" if c is None else pair_status((c["Tiny_low"],c["Tiny_high"]),
                (c["latency_low"],c["latency_high"]),tau,limit) for c in group]
            decision = gap_status(states)
            p_rows.append({"order":order,"P":p["degradation_id"],"features":"|".join(p["leaves"]),
                "Tiny_threshold":tau,"latency_limit_ms":limit,"measured_D":len(measured),
                "pass_D":states.count("PASS"),"fail_D":states.count("FAIL"),
                "uncertain_D":states.count("UNCERTAIN"),"missing_D":states.count("UNMEASURED"),"status":decision,
                "static_proxy":"COVERED" if set(p["leaves"])<=allowed else "GAP",
                "best_measured_Tiny":max((r["Tiny_AP50_95"] for r in measured),default=None)})
            if (tau,limit)==REFERENCE:
                pair_rows.extend({**{k:v for k,v in item.items() if k!="draws"},"status":state}
                                 for item,state in zip(group,states) if item is not None)
        by_d = {(c["resolution"],c["precision"]):c for c in measured}
        for left,right in CONTRASTS:
            if left not in by_d or right not in by_d:
                continue
            a,b = by_d[left],by_d[right]
            for area,col in (("all","AP50_95"),("tiny","Tiny_AP50_95")):
                ci = np.quantile(a["draws"][area]-b["draws"][area],(.025,.975))
                paired.append({"P":p["degradation_id"],"left":f"{left[0]} {left[1]}",
                    "right":f"{right[0]} {right[1]}","area":area,"delta_pp":100*(a[col]-b[col]),
                    "low_pp":float(100*ci[0]),"high_pp":float(100*ci[1]),"shared_resamples":200})
    for t,l in THRESHOLDS:
        subset = [p for p in p_rows if (p["Tiny_threshold"],p["latency_limit_ms"])==(t,l)]
        full = [p for p in subset if p["missing_D"]==0]
        c = Counter(p["status"] for p in subset)
        counts.append({"Tiny_threshold":t,"latency_limit_ms":l,"complete_P":len(full),
            "covered":c["COVERED"],"tested_candidate_gap":c["TESTED_CANDIDATE_GAP"],"unknown":c["UNKNOWN"],
            "covered_complete_P":sum(p["status"]=="COVERED" for p in full),
            "static_covered_complete_P":sum(p["static_proxy"]=="COVERED" for p in full)})
    return pair_rows,p_rows,counts,paired,missing


def heatmap(out, p_rows, cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tested = [p for p in p_rows if (p["Tiny_threshold"],p["latency_limit_ms"])==REFERENCE and p["measured_D"]]
    scores = {(r["P"],r["resolution"],r["precision"]):100*r["Tiny_AP50_95"] for r in cells.values()}
    data = np.array([[scores.get((r["P"],*setting),np.nan) for setting in SETTINGS] for r in tested])
    fig,ax = plt.subplots(figsize=(9,max(4,1.6+.34*len(tested))),layout="constrained")
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#e5e7eb")
    im=ax.imshow(data,aspect="auto",vmin=0,vmax=max(10,float(np.nanmax(data))),cmap=cmap)
    ax.set_yticks(range(len(tested)),[f"{p['order']:02d}  {p['P']}" for p in tested])
    ax.set_xticks(range(5),[f"{size} {prec}" for size,prec in SETTINGS])
    for i,j in np.ndindex(data.shape):
        value=data[i,j]
        ax.text(j,i,"Pending" if np.isnan(value) else f"{value:.2f}",ha="center",va="center",
                color="white" if value>7 else "#152634")
    ax.set_title("Week 4 development evaluation on 548 validation images")
    ax.set_xlabel("Inference size and precision")
    fig.colorbar(im,ax=ax,label="Tiny AP50:95 (%)")
    fig.savefig(out/"tiny_ap.png",dpi=170)
    plt.close(fig)


def main():
    began = perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs",nargs="+",type=Path)
    args=parser.parse_args()
    runs=[p.resolve(strict=True) for p in args.runs]
    dev.require(all(p.parent==dev.RUNS.resolve() for p in runs),"Expected Week 4 local run directories.")
    matrix, protocol = dev.design(100)
    dev.require(all(dev.load(r/"protocol.json")==protocol for r in runs),"Run protocols do not match.")
    cells=read_cells(matrix,protocol,runs)
    dev.require(bool(cells),"No sealed cells.")
    rules=dev.load(dev.HERE.parent/"uav_03_fmd_static_gap/coverage_rules.json")
    allowed={n for levels in rules["covered_levels"].values() for n in levels}
    result,p,thresholds,paired,missing=analyze(matrix,cells,allowed)
    out=dev.output_folder(dev.HERE.name)
    for name,rows in (("verified_metrics",result),("gap_by_configuration",p),("threshold_sensitivity",thresholds),
                      ("paired_detector_differences",paired),("remaining_pairs",missing)):
        if rows:
            dev.table(out/(name+".csv"),rows)
    heatmap(out,p,cells)
    cost=[json.loads(x)["group_seconds"] for x in (runs[0]/"group_progress.jsonl").read_text().splitlines()]
    summary={"RunID":out.name,"source_RunIDs":[r.name for r in runs],"status":"MEASURED_SUBSET_VERIFIED",
        "scope":dev.SCOPE,"formal_pair_credit":0,"verified_pairs":len(cells),"unmeasured_pairs":len(missing),
        "primary":next(r for r in thresholds if (r["Tiny_threshold"],r["latency_limit_ms"])==REFERENCE),
        "bootstrap_samples":200,"original_complete_P_group_seconds_mean":mean(cost),
        "original_complete_P_group_seconds_median":median(cost),"original_complete_P_group_seconds_min":min(cost),
        "original_complete_P_group_seconds_max":max(cost),"protocol_SHA256":dev.sha256(runs[0]/"protocol.json")}
    dev.write(out/"summary.json",summary)
    dev.index(out,{"ExperimentID":dev.HERE.name,"status":summary["status"],"CPU_wall_seconds":perf_counter()-began})
    dev.text_file(out/"SHA256SUMS.txt","\n".join(f"{dev.sha256(f)}  {f.name}" for f in sorted(out.iterdir()) if f.is_file()))
    print(out,flush=True)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
