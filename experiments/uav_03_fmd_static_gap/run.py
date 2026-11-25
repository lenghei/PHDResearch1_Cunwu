"""Reproduce FM_D counts, static coverage, and frozen detector CPU profiles."""
import argparse
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code/tools"))
from uav_reproduction import RESULTS, check_uvl, output_folder, write_json
import uav_week01_06 as audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--flamapy-python", type=Path)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    check_uvl("FM_P")
    audit.FMD = check_uvl("FM_D")
    if args.flamapy_python:
        audit.FLAMA = args.flamapy_python.resolve(strict=True)
    out = output_folder(HERE.name, args.output)
    summary = audit.week3(out)
    published = RESULTS / HERE.name / "uav_03_fmd_static_gap_20261124T153932_aafb49d2/static_gap_counts.csv"
    if audit.rows(out / "static_gap_counts.csv") != audit.rows(published):
        raise ValueError("Replayed static coverage differs from the verified result.")
    from profile_frozen import main as profile
    profiling = out / "frozen_detectors"
    profiling.mkdir()
    profile(profiling)
    observed = audit.load(profiling / "five_detector_attributes.json")["detectors"]
    published = audit.rows(RESULTS / HERE.name / "review/frozen_detector_attributes.csv")
    expected = {r["detector_id"]: (int(r["parameters"]), float(r["gflops_2xmac"])) for r in published}
    actual = {r["detector_id"]: (r["checkpoint_parameters"], r["GFLOPs_2xMAC"]) for r in observed}
    if actual != expected:
        raise ValueError("Replayed detector profiles differ from the verified result.")
    write_json(out / "reproduction.json", {**summary, "detector_cpu_profiles_match": True,
                                           "new_gpu_measurements": 0})
    audit.finish(out)
    print(f"FM_D: {summary['FlamaPy']['valid_count']} valid | Static coverage: 1 / 3 / 18 | Results: {out}")


if __name__ == "__main__":
    main()
