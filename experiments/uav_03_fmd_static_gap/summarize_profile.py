"""Verify a full validation profile and export portable tables and a chart."""
import argparse
import csv
import json
from math import ceil, isclose, isfinite
from pathlib import Path
from statistics import mean, median


def load(path):
    return json.loads(path.read_bytes())


def percentile95(values):
    return sorted(values)[ceil(.95 * len(values)) - 1]


def require(value, message):
    if not value:
        raise ValueError(message)


def write_csv(path, values):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def summarize(run, out):
    receipt = load(run / "measurements.json")
    guard = load(run / "supervisor.json")
    expected = [(640, "FP32"), (960, "FP32"), (1280, "FP32"), (640, "FP16"), (1280, "FP16")]
    require(receipt["status"] == "COMPLETE" and receipt["profile"] == "FULL_VALIDATION"
            and guard["returncode"] == 0 and not guard["stop_reason"],
            "Full validation did not complete.")
    require(guard["worker_wall_seconds"] <= guard["budget_seconds"], "Worker exceeded budget.")
    require([(r["resolution"], r["precision"]) for r in receipt["results"]] == expected,
            "Detector matrix differs.")
    require(len({r["detector_id"] for r in receipt["results"]}) == 5, "Repeated detector ID.")
    sample = load(run / "sample.json")
    reference = [(r["image_id"], r["pixel_sha256"]) for r in sample["images"]]
    require(len(reference) == len({pair[0] for pair in reference}) == 548
            and sample["configuration_id"] == receipt["configuration_id"], "Image manifest differs.")
    rows, passes = [], []
    for row in receipt["results"]:
        ident, size, precision = (row[k] for k in ("detector_id", "resolution", "precision"))
        dtype = {"FP32": "torch.float32", "FP16": "torch.float16"}[precision]
        require((row["images"], row["repeats"], row["warmups"], row["batch"]) == (548, 3, 3, 1)
                and row["checkpoint_sha256"] == receipt["checkpoint_sha256"], "Setting metadata differ.")
        require(0 < row["peak_allocated_bytes"] <= row["peak_reserved_bytes"], "Invalid memory peaks.")
        pass_summary = load(run / f"{ident}_passes.json")
        require(len(pass_summary) == 3, "Incomplete pass summary.")
        detector, pipeline = [], []
        for number, timing in enumerate(pass_summary, start=1):
            raw = load(run / f"{ident}_pass_{number}_timing.json")
            require([(r["image_id"], r["pixel_sha256"]) for r in raw] == reference, "Image order or pixels differ.")
            require(all(r["repeat"] == number - 1 and r["batch_shape"] == [1, 3, size, size]
                        and r["dtype"] == dtype and r["detections"] >= 0
                        and 0 < r["predict_ms"] <= r["cached_to_boxes_ms"]
                        and isfinite(r["cached_to_boxes_ms"]) for r in raw), "Invalid timing or input trace.")
            require((timing["detector_id"], timing["pass"], timing["images"]) == (ident, number, 548),
                    "Pass identity differs.")
            for metric in ("predict", "cached_to_boxes"):
                require(isclose(mean(r[f"{metric}_ms"] for r in raw), timing[f"{metric}_mean_ms"]),
                        "Pass mean does not reproduce.")
            detector.extend(r["predict_ms"] for r in raw)
            pipeline.extend(r["cached_to_boxes_ms"] for r in raw)
            passes.append({"resolution": size, "precision": precision, **timing})
        for key, timings in (("predict", detector), ("cached_to_boxes", pipeline)):
            for statistic, function in (("mean", mean), ("median", median), ("p95", percentile95)):
                require(isclose(function(timings), row[f"{key}_{statistic}_ms"]),
                        f"{ident}: {key} {statistic} does not reproduce.")
        require(isclose(sum(r["wall_seconds"] for r in pass_summary), row["three_pass_wall_seconds"]),
                "Total pass time differs.")
        rows.append({"RunID": run.name, **row, "gpu": receipt["gpu"],
                     "peak_allocated_MiB": row["peak_allocated_bytes"] / 2**20,
                     "peak_reserved_MiB": row["peak_reserved_bytes"] / 2**20})
    out.mkdir(parents=True)
    write_csv(out / "full_validation_profiles.csv", rows)
    write_csv(out / "full_validation_passes.csv", passes)
    plot(rows, out)


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['resolution']}\n{r['precision']}" for r in rows]
    positions = range(len(rows))
    fig, (timing, memory) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for key, label, shift, color in (("predict", "Detector", -.18, "#146c91"),
                                     ("cached_to_boxes", "Cached pipeline", .18, "#dc8127")):
        x = [i + shift for i in positions]
        timing.bar(x, [r[f"{key}_mean_ms"] for r in rows], .34, color=color, label=label)
        timing.scatter(x, [r[f"{key}_p95_ms"] for r in rows], color="#222222", s=18, zorder=3)
    timing.scatter([], [], color="#222222", s=18, label="P95")
    timing.set_ylabel("Latency (ms/image)")
    timing.set_ylim(0, max(r["cached_to_boxes_p95_ms"] for r in rows) * 1.25)
    timing.legend(loc="upper center", ncols=3, frameon=False)
    memory.bar([i - .18 for i in positions], [r["peak_allocated_MiB"] for r in rows], .34,
               color="#146c91", label="Allocated")
    memory.bar([i + .18 for i in positions], [r["peak_reserved_MiB"] for r in rows], .34,
               color="#9cabc0", label="Reserved")
    memory.set_ylabel("Peak PyTorch memory (MiB)")
    memory.legend(frameon=False)
    for ax in (timing, memory):
        ax.set_xticks(positions, labels)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("YOLOv8s · VisDrone val · 548 images × 3 passes")
    fig.savefig(out / "full_validation_profiles.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summarize(args.run.resolve(), args.output or args.run / "summary")


if __name__ == "__main__":
    main()
