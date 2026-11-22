"""Verify and replay the published UAV-GAP result summaries on CPU."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


RESULTS = Path(__file__).resolve().parents[2] / "reports" / "results"
STUDIES = (
    "vm36_supervisor_reproduction",
    "vm39_multi_seed_confirmation",
    "vm43_external_pilot",
    "vm44_tiny_error_variability_decision",
    "vm45_motion_haze_gap_learning",
    "vm47_gaussian_jpeg_factorial",
    "vm48_formal_rq4",
)
SHA256 = re.compile(r"[0-9a-f]{64}")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,219}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9), "Calculated value differs from the published value.")


def state(tiny_low, tiny_high, latency_high, tiny_threshold, latency_threshold):
    if tiny_high < tiny_threshold or latency_high > latency_threshold:
        return "FAIL"
    return "PASS" if tiny_low >= tiny_threshold else "UNCERTAIN"


def replay(summary):
    mode = summary["mode"]
    if mode == "VM36_MEASURED_GAP":
        pairs = summary["measured_pairs"]
        require(len({(r["condition_id"], r["D"]) for r in pairs}) == len(pairs), "Duplicate P/D cells.")
        by_p = defaultdict(list)
        for r in pairs:
            by_p[r["condition_id"]].append(state(r["Tiny_CI_low"], r["Tiny_CI_high"], r["latency_CI_high_ms"],
                                                 summary["tiny_threshold"], summary["latency_upper_threshold_ms"]))
        d = len(summary["detector_settings"])
        require(all(len(v) == d for v in by_p.values()), "Incomplete measured cohort.")
        states = Counter("COVERED" if "PASS" in v else "GAP" if set(v) == {"FAIL"} else "UNCERTAIN" for v in by_p.values())
        require(states == Counter(summary["gap"]), "Gap counts differ.")
        require(Counter(s for v in by_p.values() for s in v) == Counter(summary["pair_states"]), "Cell counts differ.")
        return {"conditions":len(by_p), "detector_settings":d, "gap":summary["gap"]}
    if mode == "FACTORIAL_INTERACTION":
        values = {}
        for row in summary["factorial"]:
            normal, a, b, joint = row["ap"]
            did = joint - a - b + normal
            close(did, row["did"])
            values[row["det"]] = {"DID_pp":did, "saved_image_CI_pp":row["ci"]}
        return values
    if mode == "MOTION_TARGETED_SEED45":
        counts = {arm:Counter() for arm in summary["states_by_strategy"]}
        closed = 0
        for row in summary["paired_conditions"]:
            arms = row["arms"]
            control, targeted = arms["CONTROL"], arms["MOTION_TARGETED"]
            for key in ("ap","tiny","ms"):
                close(targeted[key] - control[key], row["delta"][key]["value"])
            for arm, cell in arms.items():
                decision = state(*cell["tiny_ci"], cell["lat_ci"][1],
                                 summary["tiny_threshold_percent"], summary["latency_upper_threshold_ms"])
                require(decision == cell["state"], "Cell decision differs.")
                counts[arm][decision] += 1
            closed += control["state"] == "FAIL" and targeted["state"] == "PASS"
        require(counts == summary["states_by_strategy"] and closed == summary["confirmed_gap_closures"], "Gap counts differ.")
        return {"confirmed_gap_closures":closed,"states_by_strategy":counts}
    if mode == "VM44_TINY_AND_SELECTOR":
        for row in summary["tiny_recall_diagnostic"]:
            close(row["TP"] / row["GT"], row["recall"])
        return summary["lookup"]
    return {"scope":summary["scope"],"status":summary["status"]}


def verify(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["RunID"] == folder.name and manifest["ExperimentID"] == folder.parent.name, "Run identity mismatch.")
    require(manifest["schema"] == 2 and manifest["bundle"] == "snapshot.json", "Unknown snapshot format.")
    require(manifest["SHA256"] == manifest["files"]["snapshot.json"], "Bundle hash fields disagree.")
    require({f.name for f in folder.iterdir()} == {"manifest.json", *manifest["files"]}, "Unexpected public files.")
    for name, digest in manifest["files"].items():
        f = folder / name
        require(NAME.fullmatch(name) and SHA256.fullmatch(digest) and not f.is_symlink(), "Invalid artifact entry.")
        require(f.stat().st_size <= 262144, "Artifact exceeds the public size limit.")
        require(hashlib.sha256(f.read_bytes()).hexdigest() == digest, "File hash mismatch: " + name)
    for source in manifest["sources"]:
        artifact = source["artifact"]
        require(NAME.fullmatch(source["RunID"]) and SHA256.fullmatch(source["SHA256"]) and
                all(NAME.fullmatch(p) and p not in (".","..") for p in artifact.split("/")), "Invalid source reference.")
    snapshot = json.loads((folder / "snapshot.json").read_text(encoding="utf-8"))
    summary = snapshot["files"]["summary.json"]
    for key in ("RunID","ExperimentID","mode","scope","status"):
        require(snapshot[key] == summary[key] == manifest[key], "Snapshot metadata differs.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", nargs="?", choices=STUDIES, help="Verify one study, or omit to verify all seven.")
    parser.add_argument("--run", help="Select a RunID within the study.")
    args = parser.parse_args()
    if args.run and (not args.study or not NAME.fullmatch(args.run)):
        parser.error("--run requires a study and a valid RunID.")
    for study in (args.study,) if args.study else STUDIES:
        group = RESULTS / study
        folders = [group / args.run] if args.run else sorted(p for p in group.iterdir() if (p / "manifest.json").is_file())
        require(folders, "No summaries for " + study)
        for folder in folders:
            require(not folder.is_symlink(), "Run folder must not be a symlink.")
            summary = verify(folder)
            print(json.dumps({"RunID":folder.name,"verified":True,"replay":replay(summary)}, sort_keys=True))


if __name__ == "__main__":
    main()
