"""Summarize sealed Test-dev results without running the detector."""
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
from html import escape
import json
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np

from run_testdev_101p import CAMPAIGN, MAIN, base_binding, frozen_plan, load, require, verify_pair

sys.path.insert(0, str(MAIN / "code/tools"))
from uav_week01_06 import gap_status, pair_status


def publish(path, emit):
    temp = path.with_name(f".{path.name}.{uuid4().hex}")
    try:
        with temp.open("x", encoding="utf-8", newline="") as stream:
            emit(stream)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def p18_contrasts(pairs):
    p18 = {int(r["d_order"]): r for r in pairs if r["p_label"] == "P18"}
    if set(p18) != {1, 2, 3, 4, 5}:
        return []
    archives = {i: CAMPAIGN / "pairs" / p["pair_id"] for i, p in p18.items()}
    bootstrap = {i: load(path / "bootstrap.json") for i, path in archives.items()}
    source = bootstrap[1]
    require(all((b["image_ids"], b["seed"], b["samples"]) ==
                (source["image_ids"], source["seed"], source["samples"]) for b in bootstrap.values()),
            "P18 Bootstrap inputs differ.")
    indices = None
    for path in archives.values():
        with gzip.open(path / "bootstrap_indices.npy.gz", "rb") as stream:
            current = np.load(stream, allow_pickle=False)
        if indices is None:
            indices = current
        require(np.array_equal(current, indices), "P18 Bootstrap draws are not aligned.")
    records = []
    for area in ("all", "tiny"):
        point, draws = {}, {}
        for name, high, low in (("YOLOv8s", 2, 1), ("YOLO26s", 4, 3)):
            point[name] = p18[high][f"{area}_map50_95"] - p18[low][f"{area}_map50_95"]
            draws[name] = np.asarray(bootstrap[high]["draws"]["values"][area]["map50_95"]) - np.asarray(
                bootstrap[low]["draws"]["values"][area]["map50_95"])
        models = "YOLO26s effect - YOLOv8s effect"
        point[models] = point["YOLO26s"] - point["YOLOv8s"]
        draws[models] = draws["YOLO26s"] - draws["YOLOv8s"]
        for name, value in point.items():
            low, high = np.quantile(draws[name], [0.025, 0.975])
            records.append({"P": p18[1]["p_id"], "area": area, "metric": "AP50:95",
                            "contrast": name, "delta_pp": 100 * value,
                            "ci95_low_pp": 100 * low, "ci95_high_pp": 100 * high,
                            "bootstrap_samples": source["samples"], "images": len(source["image_ids"])})
    return records


def summarize(out):
    spec, groups = frozen_plan()
    study = load(CAMPAIGN / "study.json")
    require(all(study[k] == v for k, v in spec.items()), "Campaign identity changed.")
    pairs, gaps = [], []
    for group in groups:
        first = group[0]
        decisions = []
        for row in group:
            target = CAMPAIGN / "pairs" / row["pair_id"]
            if not target.is_dir():
                decisions.append("UNMEASURED")
                continue
            binding = load(target / "configuration.json")
            require(all(binding.get(k) == v for k, v in base_binding(study, row).items()),
                    "A pair is outside the frozen Test-dev campaign.")
            verify_pair(target, binding)
            m, t, b = (load(target / f"{name}.json") for name in ("metrics", "timing", "bootstrap"))
            tiny = next(i for i in b["metrics"] if i["size"] == "tiny" and i["metric"] == "map50_95")
            decision = pair_status(tiny["ci95"], t["cached_pipeline_mean_ci95_ms"],
                                   float(row["tiny_ap_threshold"]), float(row["cached_latency_limit_ms"]))
            decisions.append(decision)
            pair = {k: row[k] for k in ("pair_id", "p_id", "p_order", "p_label",
                "analysis_group", "p_features", "d_order", "d_id", "checkpoint", "infer_px", "images")}
            for area in m["per_area"]:
                for metric in ("map50", "map50_95", "ar50_95"):
                    pair[f"{area['size']}_{metric}"] = area[metric]
            pair.update(tiny_low=tiny["ci95"][0], tiny_high=tiny["ci95"][1],
                        latency_ms=t["cached_pipeline_mean_ms"], pair_decision=decision)
            pairs.append(pair)
        gaps.append({
            "p_id": first["p_id"], "p_order": int(first["p_order"]), "p_label": first["p_label"],
            "analysis_group": first["analysis_group"], "features": first["p_features"],
            "measured_D": sum(d != "UNMEASURED" for d in decisions),
            "decision": gap_status(decisions),
        })
    uniform = [r for r in gaps if r["analysis_group"] == "UNIFORM"]
    count = Counter(r["decision"] for r in uniform)
    summary = {
        "scope": study["scope"], "sealed_pairs": len(pairs), "planned_pairs": 505,
        "complete_P": sum(r["measured_D"] == 5 for r in gaps),
        "uniform_pairs": sum(m["analysis_group"] == "UNIFORM" for m in pairs),
        "P18_pairs": gaps[-1]["measured_D"], "uniform_decisions": dict(count),
        "selection_sha256": spec["selection_sha256"],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.mkdir(exist_ok=True)

    def table(name, rows):
        if rows:
            def emit(stream):
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            publish(out / name, emit)

    table("metrics.csv", pairs)
    table("gap_by_P.csv", gaps)
    contrasts = p18_contrasts(pairs)
    table("p18_resolution_contrasts.csv", contrasts)
    publish(out / "summary.json", lambda s: json.dump(summary, s, indent=2))

    def pct(value):
        return "NA" if value is None else f"{value:.2%}"

    result_rows = "".join(
        "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in [
            f"{r['p_label'] or r['p_order']}", r["checkpoint"], r["infer_px"],
            pct(r["all_map50_95"]), pct(r["tiny_map50_95"]), f"{r['latency_ms']:.2f}",
            r["pair_decision"],
        ]) + "</tr>" for r in pairs
    )
    links = '<a href="metrics.csv">Metrics CSV</a> · ' if pairs else ""
    p18_link = ' · <a href="p18_resolution_contrasts.csv">P18 paired effects</a>' if contrasts else ""
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Test-dev evaluation</title>
<style>
body{{margin:3rem auto;max-width:70rem;padding:0 1rem;background:#f8fafc;color:#10223a;font:16px system-ui}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{text-align:left;padding:.6rem;border-bottom:1px solid #dde6f0}}
a{{color:#0759a1}}h1{{font-size:2rem}}
</style>
<h1>Test-dev evaluation</h1>
<p>Limited benchmark · 1,610 images per pair</p>
<p><strong>{len(pairs)} / 505 sealed pairs</strong> · {summary['complete_P']} / 101 complete P · P18 {summary['P18_pairs']} / 5</p>
<p>{links}<a href="gap_by_P.csv">Gap by P</a> · <a href="summary.json">Summary</a>{p18_link}</p>
<table><thead><tr><th>P</th><th>Checkpoint</th><th>Infer</th><th>AP50:95</th><th>Tiny AP50:95</th><th>Latency (ms)</th><th>Decision</th></tr></thead>
<tbody>{result_rows}</tbody></table>
</html>"""
    publish(out / "index.html", lambda s: s.write(html))
    return summary


if __name__ == "__main__":
    results = CAMPAIGN / "results"
    print(json.dumps(summarize(results)))
    print(results / "index.html")
