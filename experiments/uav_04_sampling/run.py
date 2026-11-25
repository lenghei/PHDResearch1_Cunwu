"""Regenerate the frozen 100 × 5 design and verify it against the review tables."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code/tools"))
from uav_reproduction import EXPS, RESULTS, check_uvl, digest, output_folder, write_json
import uav_week01_06 as audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    check_uvl("FM_P")
    check_uvl("FM_D")
    out = output_folder(HERE.name, args.output)
    published = RESULTS / HERE.name / "review"
    checkpoints = audit.rows(RESULTS / "uav_01_baseline/models/checkpoints.csv")
    baseline, = [row for row in checkpoints if row["role"] == "validation_selected_baseline"]
    sys.path.insert(0, str(EXPS / "cv261207"))
    from sampling import detector_design, freeze, models
    fmp, fmd, sampler = models()
    freeze(out, sampler.enumerate_configs(fmp), fmp, sampler,
           detector_design(fmd, baseline["sha256"]))
    plan = out / "selection_manifest.json"
    summary = audit.week4(out, regenerated=plan)
    for name in ("p_configurations.csv", "d_configurations.csv", "sampling_coverage.csv", "first500_pairs_PLAN_ONLY.csv"):
        if audit.rows(out / name) != audit.rows(published / name):
            raise ValueError(f"Replayed design differs from the verified result: {name}")
    archived = audit.AUDIT / "selection_manifest.json"
    historical_match = None
    if archived.is_file():
        reference = audit.design()
        historical_match = audit.load(plan) == reference
        if not historical_match:
            raise ValueError("Regenerated plan differs from the historical selection manifest.")
    write_json(out / "reproduction.json", {**summary, "published_tables_match": True,
              "historical_manifest_match": historical_match, "regenerated_manifest_sha256": digest(plan),
              "measured_pairs": 0})
    audit.finish(out)
    print(f"{summary['p']} P x {summary['d']} D | {summary['pairs']} planned pairs | Results: {out}")


if __name__ == "__main__":
    main()
