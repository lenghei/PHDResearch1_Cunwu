"""Rebuild the Test-dev review tables and charts from verified summary files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE = Path(__file__).resolve().parents[2] / "reports/results/uav_05_evaluation/testdev_101p_5d_review"
D = {"1": "YOLOv8s T1280 / I640", "2": "YOLOv8s T1280 / I1280",
     "3": "YOLO26s T1280 / I640", "4": "YOLO26s T1280 / I1280",
     "5": "YOLOv8s T640 / I1280"}
BLUE, TEAL = "#225482", "#146C65"
SIZES = ("all", "tiny", "small", "medium", "large")


def load_input(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ranked(rows, metric):
    result = {}
    key = lambda r: (-float(r[metric]), int(r["p_order"]), int(r["d_order"]))
    for row in sorted(rows, key=key):
        result.setdefault(row["p_id"], row)
    return list(result.values())


def plot_clean(clean, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.35), gridspec_kw={"width_ratios": [1.65, 1]})
    for ax, group, title in zip(axes, ("VAL_MATCHED_CHECKPOINTS", "TESTDEV_SEED42_PAIRED"),
                                ("Val · 548 · campaign weights", "Test-dev · 1,610 · seed 42 pair")):
        cells = [r for r in clean if r["series"] == group]
        x = np.arange(len(cells))
        names = [r["configuration"] if r["configuration"].startswith("D") else
                 ("Control" if r["configuration"] == "CONTROL" else "Guided") for r in cells]
        for shift, metric, color, label in ((-.18, "AP50_95", BLUE, "Overall"),
                                             (.18, "Tiny_AP50_95", TEAL, "Tiny")):
            bars = ax.bar(x + shift, [100 * float(r[metric]) for r in cells], .34, color=color, label=label)
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
        ax.set(title=title, xticks=x, xticklabels=names, ylim=(0, 42))
        ax.grid(axis="y", color="#dbe3eb")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("AP50:95 (%)")
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "normal_baselines.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ranks(ranks, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, key, title, metric, color in zip(axes, ("overall", "tiny"),
            ("Highest overall AP", "Highest Tiny AP"), ("all_map50_95", "tiny_map50_95"), (BLUE, TEAL)):
        chosen = ranks[key][:10]
        y = np.arange(len(chosen))
        values = [100 * float(r[metric]) for r in chosen]
        bars = ax.barh(y, values, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set(yticks=y, yticklabels=[f"{r['p_label'] or 'Draw ' + r['p_order']} · D{r['d_order']}" for r in chosen],
               xlabel="AP50:95 (%)", title=title, xlim=(0, 25 if key == "overall" else 11))
        if key == "tiny":
            ax.axvline(10, ls="--", lw=1, color="#aa433a", label="10% threshold")
            ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#dbe3eb")
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(out / "testdev_top_rankings.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_detectors(rows):
    summary = []
    for order, label in D.items():
        cells = [r for r in rows if r["d_order"] == order]
        latency = [float(r["latency_ms"]) for r in cells]
        summary.append({
            "d_order": order, "detector": label, "P": len(cells),
            **{f"mean_{metric}": np.mean([100 * float(r[metric]) for r in cells]) for metric in
               ("all_map50", "all_map50_95", "all_ar50_95", *(f"{s}_map50_95" for s in SIZES[1:]))},
            "max_tiny_pct": max(100 * float(r["tiny_map50_95"]) for r in cells),
            "latency_median_ms": np.median(latency), "latency_min_ms": min(latency), "latency_max_ms": max(latency),
        })
    return summary


def resolution_effects(rows):
    cell = {(r["p_id"], r["d_order"]): r for r in rows}
    p_ids = list(dict.fromkeys(r["p_id"] for r in rows))
    effects = []
    for model, lo, hi in (("YOLOv8s", "1", "2"), ("YOLO26s", "3", "4")):
        for size in SIZES:
            key = f"{size}_map50_95"
            delta = [100 * (float(cell[p, hi][key]) - float(cell[p, lo][key])) for p in p_ids]
            increased = sum(x > 1e-12 for x in delta)
            decreased = sum(x < -1e-12 for x in delta)
            effects.append(dict(model=model, size=size, P=len(p_ids), mean_delta_pp=np.mean(delta),
                                increased=increased, decreased=decreased, tied=len(p_ids) - increased - decreased))
    return effects


def plot_campaign(rows, detector_summary, out):
    fig, (ap, latency) = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw={"width_ratios": [1.35, 1]})
    scores = sorted(100 * float(r["tiny_map50_95"]) for r in ranked(rows, "tiny_map50_95"))
    ap.plot(range(1, len(scores) + 1), scores, color=TEAL, lw=2.4)
    ap.axhline(10, color="#aa433a", lw=1.2, ls="--", label="10% threshold")
    ap.set(xlabel="Uniform P, sorted by best Tiny AP", ylabel="Best Tiny AP50:95 (%)", ylim=(0, 11))
    ap.legend(loc="upper left", frameon=False, fontsize=9)
    y = np.arange(len(detector_summary))
    lows = [r["latency_min_ms"] for r in detector_summary]
    highs = [r["latency_max_ms"] for r in detector_summary]
    medians = [r["latency_median_ms"] for r in detector_summary]
    latency.hlines(y, lows, highs, color="#87aac5", lw=3, label="Min–max across P")
    latency.scatter(medians, y, color=BLUE, zorder=3, label="Median across P")
    latency.axvline(100, color="#aa433a", lw=1.2, ls="--")
    latency.set(xlabel="Mean cached latency per cell (ms/image)", yticks=y, yticklabels=[f"D{i}" for i in D],
                xlim=(20, 106), ylim=(4.7, -.7))
    latency.legend(loc="upper right", frameon=False, fontsize=8)
    for ax in (ap, latency):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e3e9ef")
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2)
    fig.savefig(out / "testdev_campaign.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_augmentation(rows, out):
    fig, ax = plt.subplots(figsize=(9.6, 3.7))
    y = np.arange(len(SIZES))
    for condition, color, offset, label in (("Normal", BLUE, -.15, "Normal"),
                                            ("Gaussian_s15", TEAL, .15, "Gaussian σ15")):
        by_size = {r["size"]: r for r in rows if r["condition"] == condition}
        points = [by_size[s] for s in SIZES]
        delta = np.asarray([100 * float(r["delta"]) for r in points])
        bounds = [np.asarray([100 * float(r[k]) for r in points]) for k in ("ci95_low", "ci95_high")]
        ypos = y + offset
        ax.hlines(ypos, bounds[0], bounds[1], color=color, lw=1.6)
        ax.scatter(delta, ypos, color=color, label=label, zorder=3)
    ax.axvline(0, color="#69737c", lw=1)
    ax.set(xlabel="GUIDED − CONTROL AP50:95 (percentage points)", yticks=y,
           yticklabels=["Overall", "Tiny", "Small", "Medium", "Large"])
    ax.invert_yaxis()
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e3e9ef")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "augmentation_by_size.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_augmentation_levels(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    positions = np.arange(len(SIZES))
    for ax, condition, label in zip(axes, ("Normal", "Gaussian_s15"),
                                    ("Normal (original images)", "Gaussian σ15 (degraded images)")):
        by_size = {row["size"]: row for row in rows if row["condition"] == condition}
        for method, color, offset, legend in (("control", BLUE, -.17, "Ordinary training"),
                                               ("guided", TEAL, .17, "Gaussian-guided training")):
            scores = [100 * float(by_size[size][method]) for size in SIZES]
            bars = ax.barh(positions + offset, scores, .31, color=color, label=legend)
            ax.bar_label(bars, labels=[f"{score:.2f}%" for score in scores], padding=4, fontsize=9)
        change = 100 * float(by_size["all"]["delta"])
        ax.set(title=f"{label}\nOverall change {change:+.2f} pp", xlim=(0, 34),
               yticks=positions, yticklabels=["Overall", "Tiny", "Small", "Medium", "Large"],
               xticks=(0, 10, 20, 30), xlabel="AP50:95 (%)")
        ax.xaxis.grid(color="#dae3ec", linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("YOLO26s T960 / I1280 on VisDrone Test-dev", x=.53, y=.985, fontsize=14, fontweight="bold")
    fig.text(.53, .915, "1,610 images per condition · fixed epoch 10 EMA · seed 42", ha="center", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .88), ncol=2, frameon=False)
    fig.subplots_adjust(left=.10, right=.98, bottom=.20, top=.68, wspace=.14)
    fig.text(.10, .046, "Shared initialization before VisDrone fine-tuning: no verified Normal Test-dev score.",
             fontsize=9, color="#435365")
    fig.savefig(out / "gaussian_ap_levels.png", dpi=215, facecolor="white")
    plt.close(fig)


def selected_resolution_cases(rows):
    cases = []
    cells = {(r["p_order"], r["d_order"]): r for r in rows}
    for draw in ("69", "38", "101", "62"):
        for model, lo, hi in (("YOLOv8s", "1", "2"), ("YOLO26s", "3", "4")):
            low, high = cells[draw, lo], cells[draw, hi]
            cases.append({
                "condition": high["p_label"] or draw, "p_id": high["p_id"], "model": model,
                "i640_pair_id": low["pair_id"], "i1280_pair_id": high["pair_id"],
                **{f"{size}_i{resolution}_ap_pct": 100 * float(row[f"{size}_map50_95"])
                   for size in SIZES for resolution, row in ((640, low), (1280, high))},
                "latency_i640_ms": float(low["latency_ms"]), "latency_i1280_ms": float(high["latency_ms"]),
                **{f"{size}_delta_pp": 100 * (float(high[f"{size}_map50_95"]) - float(low[f"{size}_map50_95"]))
                   for size in SIZES},
                "latency_delta_ms": float(high["latency_ms"]) - float(low["latency_ms"]),
            })
    return cases


def resolution_highlights(effects, clean_pairs, p18):
    val = {r["benchmark_D_reference"]: r["tiny_ap_gain_pp"] for r in clean_pairs}
    p18_tiny = {r["contrast"]: r for r in p18 if r["area"] == "tiny"}
    results = []
    for model, settings in (("YOLOv8s", "D1/D2"), ("YOLO26s", "D3/D4")):
        val_delta = val[settings]
        test_delta = next(r["mean_delta_pp"] for r in effects if r["model"] == model and r["size"] == "tiny")
        p18_row = p18_tiny[model]
        results.extend([
            dict(model=model, scope="Clean Val", images=548, P=1, delta_pp=val_delta, ci95_low_pp="", ci95_high_pp=""),
            dict(model=model, scope="100 Test-dev P", images=1610, P=100, delta_pp=test_delta,
                 ci95_low_pp="", ci95_high_pp=""),
            dict(model=model, scope="P18 Test-dev", images=1610, P=1, delta_pp=float(p18_row["delta_pp"]),
                 ci95_low_pp=p18_row["ci95_low_pp"], ci95_high_pp=p18_row["ci95_high_pp"]),
        ])
    return results


def plot_resolution(effects, highlights, target):
    fig, (sizes, tiny) = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})
    positions = np.arange(len(SIZES))
    labels = list(dict.fromkeys(r["scope"] for r in highlights))
    for model, color, shift in (("YOLOv8s", BLUE, -.17), ("YOLO26s", TEAL, .17)):
        delta = [r["mean_delta_pp"] for r in effects if r["model"] == model]
        sizes.barh(positions + shift, delta, .31, color=color, label=model)
        group = [r["delta_pp"] for r in highlights if r["model"] == model]
        bars = tiny.barh(np.arange(len(labels)) + shift, group, .31, color=color)
        tiny.bar_label(bars, labels=[f"{v:+.2f}" for v in group], padding=3, fontsize=8)
    sizes.set(title="100 randomly sampled Test-dev conditions", xlabel="I1280 − I640 AP50:95 (pp)",
              yticks=positions, yticklabels=["Overall", "Tiny", "Small", "Medium", "Large"], xlim=(-4.6, 0.7))
    right_edge = max(r["delta_pp"] for r in highlights) * 1.2
    tiny.set(title="Tiny AP across evaluation settings", yticks=range(len(labels)), yticklabels=labels,
             xlabel="I1280 − I640 Tiny AP50:95 (pp)", xlim=(-0.5, right_edge))
    sizes.legend(loc="upper left", frameon=False, fontsize=8)
    for ax in (sizes, tiny):
        ax.axvline(0, color="#69737c", lw=.9)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e3e9ef")
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2)
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def tiny_band(row):
    score = 100 * float(row["tiny_map50_95"])
    if score <= 2:
        return "0–2%"
    if score <= 5:
        return ">2–5%"
    if score <= 8:
        return ">5–8%"
    return ">8%"


def report_tables(top, detectors, clean, references, augmentation, training, cases, bands, schedule, budget):
    def pct(value):
        return f"{100 * float(value):.2f}"

    val = [r for r in clean if r["series"] == "VAL_MATCHED_CHECKPOINTS"]
    aug = {(r["condition"], r["size"]): r for r in augmentation}
    pilots = {(r["strategy"], r["condition"], r["size"]): r for r in training}

    def before_after(row, other="comparison"):
        return f"{pct(row['control'])} → {pct(row[other])}\n({100 * float(row['delta']):+.2f})"

    def resolution_cell(row, key, unit):
        return (f"{row[f'{key}_i640_{unit}']:.2f} → {row[f'{key}_i1280_{unit}']:.2f}\n"
                f"({row[f'{key}_delta_' + ('pp' if unit == 'ap_pct' else 'ms')]:+.2f})")
    first_stage = [r for r in schedule if r["phase"] == "1"]
    return {
        "benchmark_detectors": [
            [f"D{r['d_order']}", r["detector"], f"{r['mean_all_map50_95']:.2f}",
             f"{r['mean_tiny_map50_95']:.2f}", f"{r['latency_median_ms']:.2f}"]
            for r in detectors
        ],
        **{
            f"{ranking}_top_10": [
                [str(r["rank"]), r["p_label"] or r["p_order"], f"D{r['d_order']}",
                 *[pct(r[f"{size}_map50_95"]) for size in SIZES], f"{float(r['latency_ms']):.2f}"]
                for r in top if r["ranking"] == ranking
            ]
            for ranking in ("overall", "tiny")
        },
        "selected_input_size_cases": [
            [r["condition"], r["model"], *[resolution_cell(r, size, "ap_pct") for size in SIZES],
             resolution_cell(r, "latency", "ms")]
            for r in cases
        ],
        "training_case_overview": [
            ["Gaussian guided", "Test-dev · Gaussian σ15", *[before_after(aug["Gaussian_s15", size], "guided")
                                                              for size in ("all", "tiny")]],
            *[[strategy, f"Val · {condition}", *[before_after(pilots[strategy, condition, size])
                                                   for size in ("all", "tiny")]]
              for strategy, condition in (("MOTION_TARGETED", "Motion_k21"), ("MOTION_TARGETED", "Normal"),
                                          ("ALL_FIVE", "Normal"), ("ALL_FIVE", "Gaussian_s15"),
                                          ("ALL_FIVE", "JPEG_Q25"), ("ALL_FIVE", "JPEG_Q75"))],
        ],
        "motion_targeted_by_size": [
            [size, pct(row["control"]), pct(row["comparison"]), f"{100 * float(row['delta']):+.2f}",
             f"[{100 * float(row['ci95_low']):+.2f}, {100 * float(row['ci95_high']):+.2f}]"]
            for size in SIZES for row in [pilots["MOTION_TARGETED", "Motion_k21", size]]
        ],
        "all_five_by_size": [
            [condition, *[before_after(pilots["ALL_FIVE", condition, size]) for size in SIZES]]
            for condition in ("Normal", "Gaussian_s15", "JPEG_Q25", "JPEG_Q75")
        ],
        "normal_baseline": [
            [pct(r[key]) for r in val] for key in ("AP50_95", "Tiny_AP50_95")
        ],
        "other_models_i1280": [
            [r["model"], pct(r["AP50_95"]), pct(r["Tiny_AP"]), f"{float(r['pipeline_mean_ms']):.2f}"]
            for r in references
        ],
        "augmentation_changes_pp": [
            [f"{100 * float(aug[condition, size]['delta']):+.2f}" for size in SIZES]
            for condition in ("Normal", "Gaussian_s15")
        ],
        "gaussian_ap_before_after": [
            [size, pct(aug["Gaussian_s15", size]["control"]), pct(aug["Gaussian_s15", size]["guided"]),
             f"{100 * float(aug['Gaussian_s15', size]['delta']):+.2f}",
             f"[{100 * float(aug['Gaussian_s15', size]['ci95_low']):+.2f}, "
             f"{100 * float(aug['Gaussian_s15', size]['ci95_high']):+.2f}]"]
            for size in SIZES
        ],
        "gaussian_ap_tradeoff": [
            ["Overall" if size == "all" else size.title(),
             *[value for condition in ("Normal", "Gaussian_s15")
               for row in (aug[condition, size],)
               for value in (f"{pct(row['control'])}% → {pct(row['guided'])}%",
                             f"{100 * float(row['delta']):+.2f}")]]
            for size in SIZES
        ],
        "gaussian_ap_intervals": [
            ["Overall" if size == "all" else size.title(),
             *[f"[{100 * float(row['ci95_low']):+.2f}, {100 * float(row['ci95_high']):+.2f}]"
               for condition in ("Normal", "Gaussian_s15") for row in (aug[condition, size],)]]
            for size in SIZES
        ],
        "proposed_timing": [
            [str(stage["P"]), str(stage["pairs"]), f"{stage['images']:,}",
             f"{stage['minutes_low']}–{stage['minutes_high']}"]
            for stage in budget.values()
        ],
        "proposed_tiny_bands": [
            dict(band=row["band"], available=row["unique_P"],
                 selected=[[r["draw_order"], pct(r["reference_tiny_AP"])] for r in first_stage
                           if r["selection_band"] == row["band"]])
            for row in bands
        ],
    }


def clean_resolution_comparison(rows):
    pairs = {}
    for row in rows:
        pairs.setdefault(row["checkpoint"], {})[row["inference_size"]] = row
    comparison = []
    for pair in pairs.values():
        low, high = pair["640"], pair["1280"]
        assert low["checkpoint_sha256"] == high["checkpoint_sha256"]
        item = {key: high[key] for key in
                ("model", "benchmark_D_reference", "checkpoint_sha256", "dataset", "images", "run")}
        for metric in ("overall_ap", "tiny_ap"):
            i640, i1280 = 100 * float(low[metric]), 100 * float(high[metric])
            item[f"{metric}_i640_pct"] = i640
            item[f"{metric}_i1280_pct"] = i1280
            item[f"{metric}_gain_pp"] = i1280 - i640
        comparison.append(item)
    return comparison


def tiny_threshold_summary(uniform, extras):
    thresholds = (3, 5, 7, 10)
    groups = {
        **{f"D{d}": [r for r in uniform if r["d_order"] == d] for d in D},
        "All 500 cells": uniform,
        "Best D per P": ranked(uniform, "tiny_map50_95"),
        "P18 (extra)": extras,
    }
    return [
        {"population": name, "denominator": len(rows),
         **{f"ge_{level}pct": sum(float(r["tiny_map50_95"]) >= level / 100 for r in rows)
            for level in thresholds}}
        for name, rows in groups.items()
    ]


def tiny_score_bins(counts):
    thresholds = (3, 5, 7, 10)
    labels = ("<3%", "3–<5%", "5–<7%", "7–<10%", "≥10%")
    bins = {"band": labels}
    for population, column in (("All 500 cells", "cells"), ("Best D per P", "unique_P")):
        row = next(r for r in counts if r["population"] == population)
        levels = [row["denominator"], *[row[f"ge_{level}pct"] for level in thresholds], 0]
        bins[column] = [low - high for low, high in zip(levels, levels[1:])]
    return [dict(zip(bins, row)) for row in zip(*bins.values())]


def plot_table(headers, rows, widths, out):
    multiline = any("\n" in str(cell) for row in (headers, *rows) for cell in row)
    height = (0.36 if multiline else 0.26) * (len(rows) + 1)
    fig, ax = plt.subplots(figsize=(7.5, height))
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=headers, colWidths=widths, bbox=[0, 0, 1, 1],
                     cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9 if multiline else 10)
    for (row, _), cell in table.get_celld().items():
        cell.set_facecolor("#e6edf5" if row == 0 else ("#f4f7fb" if row % 2 == 0 else "white"))
        cell.set_edgecolor("#dce4ed")
        cell.set_linewidth(0.45)
        cell.PAD = 0.055
        cell.set_text_props(color="#162536", weight="bold" if row == 0 else "normal")
    fig.subplots_adjust(left=0.005, bottom=0.01, right=0.995, top=0.99)
    fig.savefig(out, dpi=260, facecolor="white")
    plt.close(fig)


def testdev_normal_refs(clean, augmentation, references, out):
    """Keep Normal measurements on the same image split as the Test-dev report."""
    normal = {r["size"]: r for r in augmentation if r["condition"] == "Normal"}
    cells = []
    for r in clean:
        if r["series"] != "TESTDEV_SEED42_PAIRED":
            continue
        key = {"CONTROL": "control", "VARIABILITY_GUIDED": "guided"}[r["configuration"]]
        ap = {size: float(normal[size][key]) for size in SIZES}
        assert r["condition"] == "Normal" and r["images"] == "1610" and r["infer_px"] == "1280"
        assert r["split"] == "VisDrone2019-DET-test-dev" and len(normal) == len(SIZES)
        assert ap["all"] == float(r["AP50_95"]) and ap["tiny"] == float(r["Tiny_AP50_95"])
        for value in normal.values():
            assert value[f"{key}_sha256"] == r["checkpoint_sha256"]
            assert value["source_run"] == r["source_run"] and value["metrics_sha256"] == r["source_sha256"]
        cells.append({**r, **{f"{size}_AP50_95": ap[size] for size in SIZES}})
    assert len(cells) == 2 and len({r["checkpoint_sha256"] for r in cells}) == 2
    write_table(out / "normal_testdev_by_size.csv", cells)
    plot_table(("YOLO26s T960", "Input", "Overall", "Tiny", "Small", "Medium", "Large"),
               [[{"CONTROL": "Control", "VARIABILITY_GUIDED": "Gaussian guided"}[r["configuration"]],
                 f"I{r['infer_px']}", *[f"{100 * r[f'{size}_AP50_95']:.2f}" for size in SIZES]] for r in cells],
               (.25, .12, .13, .12, .12, .13, .13), out / "normal_testdev_by_size_table.png")

    availability = [("YOLOv8s T1280", "D1", "D2"), ("YOLO26s T1280", "D3", "D4"),
                    ("YOLOv8s T640", "—", "D5")]
    availability.extend((r["model"], "—", "—") for r in references)
    plot_table(("Other weight", "I640", "I1280"),
               [[name, f"{low} · —" if low != "—" else low, f"{high} · —" if high != "—" else high]
                for name, low, high in availability],
               (.51, .245, .245), out / "normal_testdev_other_models_table.png")


def plot_report_tables(tables, out):
    rank_headers = ("Rank", "P", "D", "Overall", "Tiny", "Small", "Medium", "Large", "Latency (ms)")
    rank_widths = (.075, .070, .069, .111, .101, .108, .118, .103, .145)
    plots = {
        "clean_resolution_comparison": (
            ("Model", "AP I640", "AP I1280", "Δ AP (pp)", "Tiny I640", "Tiny I1280", "Δ Tiny (pp)"),
            tables["clean_resolution_comparison"], (.22, .133, .138, .124, .124, .136, .125)),
        "tiny_threshold_summary": (
            ("Detector or group", "Total", "≥3%", "≥5%", "≥7%", "≥10%"),
            tables["tiny_threshold_summary"], (.42, .10, .12, .12, .12, .12)),
        "tiny_threshold_bins": (
            ("Tiny AP50:95 band", "Pairs (of 500)", "Best D per P (of 100)"),
            tables["tiny_threshold_bins"], (.34, .27, .39)),
        "benchmark_detectors": (
            ("ID", "Detector setting", "Overall AP (%)", "Tiny AP (%)", "Median latency (ms)"),
            tables["benchmark_detectors"], (.07, .33, .18, .17, .25)),
        "selected_input_size_cases": (
            ("P", "Model", "Overall", "Tiny", "Small", "Medium", "Large", "Latency (ms)"),
            tables["selected_input_size_cases"], (.058, .142, .128, .128, .128, .128, .128, .16)),
        "training_case_overview": (
            ("Strategy", "Dataset / test case", "Overall AP", "Tiny AP"),
            tables["training_case_overview"], (.27, .35, .19, .19)),
        "normal_baseline": (
            ("548 clean Val images", "D1", "D2", "D3", "D4", "D5"),
            [(name, *row) for name, row in zip(("Overall AP (%)", "Tiny AP (%)"), tables["normal_baseline"])],
            (.27, .146, .146, .146, .146, .146)),
        "other_models_i1280": (
            ("Model at I1280", "Overall AP (%)", "Tiny AP (%)", "Latency (ms)"),
            tables["other_models_i1280"], (.37, .21, .21, .21)),
        "overall_top_10": (rank_headers, tables["overall_top_10"], rank_widths),
        "tiny_top_10": (rank_headers, tables["tiny_top_10"], rank_widths),
        "gaussian_ap_before_after": (
            ("Size", "Control (%)", "Guided (%)", "Change (pp)", "95% interval (pp)"),
            [(("Overall" if size == "all" else size.title()), *values)
             for size, *values in tables["gaussian_ap_before_after"]],
            (.17, .17, .17, .17, .32)),
        "gaussian_ap_tradeoff": (
            ("Size", "Normal (clean)\nControl → Guided", "Change (pp)",
             "Gaussian σ15 (noisy)\nControl → Guided", "Change (pp)"),
            tables["gaussian_ap_tradeoff"], (.12, .29, .13, .33, .13)),
        "gaussian_ap_intervals": (
            ("Size", "Normal 95% interval (pp)", "Gaussian σ15 95% interval (pp)"),
            tables["gaussian_ap_intervals"], (.20, .38, .42)),
        "proposed_tiny_bands": (
            ("Tiny AP band", "Available P", "First-stage P"),
            [[r["band"], str(r["available"]),
              ", ".join(f"{order} ({ap}%)" for order, ap in r["selected"]) or "—"]
             for r in tables["proposed_tiny_bands"]],
            (.25, .21, .54)),
        "proposed_timing": (
            ("Stage", "Conditions", "Evaluation cells", "Image evaluations", "Estimated minutes"),
            [(name, *row) for name, row in zip(("First pass", "Follow-up"), tables["proposed_timing"])],
            (.19, .15, .20, .23, .23)),
    }
    for name, (headers, rows, widths) in plots.items():
        plot_table(headers, rows, widths, out / f"{name}_table.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        with (SOURCE / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Source checksum differs: {name}")
    all_rows = load_input(SOURCE / "testdev_metrics.csv")
    assert len(all_rows) == len({r["pair_id"] for r in all_rows}) == 505
    assert {r["images"] for r in all_rows} == {"1610"}
    uniform = [r for r in all_rows if r["analysis_group"] == "UNIFORM"]
    assert len(uniform) == 500 and len({r["p_id"] for r in uniform}) == 100
    assert all(r["p_features"] != "Normal" for r in all_rows)
    p_rows = load_input(SOURCE / "p_configurations.csv")
    d_rows = load_input(SOURCE / "d_configurations.csv")
    assert len(p_rows) == 101 and len(d_rows) == 5
    assert {(r["p_id"], r["d_id"]) for r in all_rows} == {
        (p["p_id"], d["d_id"]) for p in p_rows for d in d_rows
    }
    assert {r["pair_decision"] for r in all_rows} == {"FAIL"}
    assert all(float(r["tiny_high"]) < .10 and float(r["latency_ms"]) <= 100 for r in all_rows)
    ranks = {"overall": ranked(all_rows, "all_map50_95"), "tiny": ranked(all_rows, "tiny_map50_95")}
    screen_d = ("2", "4")
    uniform_tiny = ranked([r for r in uniform if r["d_order"] in screen_d], "tiny_map50_95")
    cases = selected_resolution_cases(all_rows)
    write_table(out / "selected_input_size_cases.csv", cases)
    detector_summary = summarize_detectors(uniform)
    write_table(out / "detector_summary.csv", detector_summary)
    effects = resolution_effects(uniform)
    clean = load_input(SOURCE / "normal_baselines.csv")
    p18 = load_input(SOURCE / "p18_resolution_contrasts.csv")
    clean_pairs = clean_resolution_comparison(load_input(SOURCE / "clean_val_resolution_source.csv"))
    highlights = resolution_highlights(effects, clean_pairs, p18)
    write_table(out / "resolution_effects.csv", effects)
    write_table(out / "input_size_benefits.csv", highlights)
    top = []
    for ranking, selected in ranks.items():
        for i, r in enumerate(selected[:10], 1):
            top.append({"ranking": ranking, "rank": i, **{k: r[k] for k in
                        ("p_order", "p_label", "analysis_group", "p_id", "p_features", "d_order", "pair_id", "all_map50_95", "tiny_map50_95",
                         "small_map50_95", "medium_map50_95", "large_map50_95",
                         "tiny_low", "tiny_high", "latency_ms")}})
    write_table(out / "top_rankings.csv", top)
    bands = [{"band": band, "unique_P": sum(tiny_band(r) == band for r in uniform_tiny)}
             for band in ("0–2%", ">2–5%", ">5–8%", ">8%")]
    write_table(out / "tiny_bands.csv", bands)
    by_order = {r["p_order"]: r for r in uniform_tiny}
    p18_i1280 = [r for r in all_rows if r["p_label"] == "P18" and r["d_order"] in screen_d]
    by_order["101"] = ranked(p18_i1280, "tiny_map50_95")[0]
    timers = load_input(SOURCE / "timing_reference.csv")
    chosen = (("1", "62", "Low Tiny AP with Gaussian noise"),
              ("1", "54", "Low Tiny AP without Gaussian noise"),
              ("1", "69", "Gaussian plus JPEG"), ("1", "38", "Gaussian in composite degradation"),
              ("1", "51", "Lower-scoring Gaussian condition"), ("1", "12", "No-Gaussian transfer control"),
              ("2", "10", "No-Gaussian transfer control"), ("2", "4", "Motion transfer control"),
              ("2", "101", "Prespecified P18 transfer case"))
    assert {r["p_order"] for r in uniform_tiny if float(r["tiny_map50_95"]) > .02} == {
        order for _, order, _ in chosen if order not in {"101", "62", "54"}}
    schedule = [{"phase": "1", "draw_order": "Normal", "p_id": "p_cdbfeb94b010c58d",
                 "p_features": "Normal", "role": "Clean control", "selection_band": "Clean control",
                 "reference_D": "", **{f"reference_{s}_AP": "" for s in SIZES},
                 "pair_count": 2, "images": 3220, "proxy_low_seconds": 900, "proxy_high_seconds": 1800}]
    for phase, order, role in chosen:
        r = by_order[order]
        matched = [x for x in timers if x["draw_order"] == order]
        assert len(matched) == 2
        pair_time = [float(x["pair_seconds"]) for x in matched]
        cache = max(float(x["cache_seconds"]) for x in matched)
        schedule.append({"phase": phase, "draw_order": order if order != "101" else "P18",
                         "p_id": r["p_id"], "p_features": r["p_features"], "role": role,
                         "selection_band": "Targeted P18" if order == "101" else tiny_band(r),
                         "reference_D": r["d_order"],
                         **{f"reference_{s}_AP": r[f"{s}_map50_95"] for s in SIZES},
                         "pair_count": 2, "images": 3220,
                         "proxy_low_seconds": round(cache + 2 * min(pair_time), 1),
                         "proxy_high_seconds": round(cache + 2 * max(pair_time), 1)})
    write_table(out / "augmentation_pilot_plan.csv", schedule)
    summary = {
        "scope": manifest["testdev"]["scope"], "pairs": len(all_rows),
        "image_evaluations": sum(int(r["images"]) for r in all_rows), "P": len(p_rows), "D": len(d_rows),
        "uniform_P": len(uniform_tiny), "uniform_P_with_no_passing_D": len(uniform_tiny),
        "P_with_transformed_GT": sum(r["annotation_mode"] == "WARP_GT_AND_IGNORE_ORIGINAL_AREA" for r in p_rows),
        "max_tiny_ci_upper_pct": 100 * max(float(r["tiny_high"]) for r in all_rows),
        "max_mean_latency_ms": max(float(r["latency_ms"]) for r in all_rows),
        "supervised_worker_hours": sum(r["worker_wall_seconds"] for r in manifest["testdev"]["batches"]) / 3600,
        "bands": bands, "pilot_screen_D": [f"D{d}" for d in screen_d], "checkpoints": D, "phase_budget": {},
    }
    for phase in ("1", "2"):
        selected = [r for r in schedule if r["phase"] == phase]
        summary["phase_budget"][phase] = {
            "P": len(selected), "pairs": len(selected) * 2, "images": len(selected) * 3220,
            "minutes_low": round(sum(r["proxy_low_seconds"] for r in selected) / 60 + 10),
            "minutes_high": round(sum(r["proxy_high_seconds"] for r in selected) / 60 * 1.25 + 10),
        }
    (out / "review_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_clean(clean, out)
    plot_resolution(effects, highlights, out / "input_size_benefits.png")
    plot_resolution(effects, [r for r in highlights if r["images"] == 1610],
                    out / "input_size_benefits_testdev.png")
    plot_ranks(ranks, out)
    plot_campaign(uniform, detector_summary, out)
    augmentation = load_input(SOURCE / "augmentation_reference.csv")
    references = load_input(SOURCE / "normal_i1280_reference.csv")
    training = load_input(SOURCE / "training_case_reference.csv")
    testdev_normal_refs(clean, augmentation, references, out)
    tables = report_tables(top, detector_summary, clean, references, augmentation, training, cases, bands, schedule,
                           summary["phase_budget"])
    counts = tiny_threshold_summary(uniform, [r for r in all_rows if r["p_label"] == "P18"])
    bins = tiny_score_bins(counts)
    write_table(out / "clean_reference_comparison.csv", clean_pairs)
    write_table(out / "tiny_threshold_summary.csv", counts)
    write_table(out / "tiny_threshold_bins.csv", bins)
    tables["clean_resolution_comparison"] = [
        [r["model"],
         *[value for field in ("overall_ap", "tiny_ap")
           for value in (f"{r[f'{field}_i640_pct']:.2f}", f"{r[f'{field}_i1280_pct']:.2f}",
                         f"{r[f'{field}_gain_pp']:+.2f}")]]
        for r in clean_pairs
    ]
    tables["tiny_threshold_summary"] = [
        [f"{r['population']} · {D[r['population'][1:]]}" if r["population"][1:] in D else r["population"],
         *[str(r[key]) for key in
          ("denominator", "ge_3pct", "ge_5pct", "ge_7pct", "ge_10pct")]]
        for r in counts
    ]
    tables["tiny_threshold_bins"] = [[r["band"], str(r["cells"]), str(r["unique_P"])] for r in bins]
    (out / "report_tables.json").write_text(json.dumps(tables, indent=2) + "\n", encoding="utf-8")
    plot_augmentation(augmentation, out)
    plot_augmentation_levels(augmentation, out)
    plot_report_tables(tables, out)


if __name__ == "__main__":
    main()
