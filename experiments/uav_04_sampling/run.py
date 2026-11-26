"""Regenerate the frozen design and validate the full-space sampling candidates."""
import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code/tools"))
from uav_reproduction import EXPS, RESULTS, check_uvl, digest, flama_python, output_folder, write_json
import uav_week01_06 as audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument('--frozen-only', action='store_true')
    args = parser.parse_args()
    began = perf_counter()
    check_uvl("FM_P")
    check_uvl("FM_D")
    out = output_folder(HERE.name, args.output)
    published = RESULTS / HERE.name / "review"
    checkpoints = audit.rows(RESULTS / "uav_01_baseline/models/checkpoints.csv")
    baseline, = [row for row in checkpoints if row["role"] == "validation_selected_baseline"]
    sys.path.insert(0, str(EXPS / "cv261207"))
    from sampling import detector_design, freeze, models
    fmp, fmd, sampler = models()
    configs = sampler.enumerate_configs(fmp)
    freeze(out, configs, fmp, sampler,
           detector_design(fmd, baseline["sha256"]))
    plan = out / "selection_manifest.json"
    summary = audit.week4(out, regenerated=plan)
    review = audit.load(published / "manifest.json")
    for name in ("p_configurations.csv", "d_configurations.csv", "sampling_coverage.csv", "first500_pairs_PLAN_ONLY.csv"):
        audit.source(published / name, review["files"][name])
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
    if not args.frozen_only:
        from full_space_plan import export
        followup = export(out, audit.load(plan), configs, fmp, sampler, audit, RESULTS, flama_python)
        write_json(out / "full_space_summary.json", followup)
        print(f"{followup['p']['pairwise_n']} pairwise P, {followup['p']['uniform_n']} uniform P; "
              f"{followup['d']['SAT_checked']} D bindings passed SAT.")
    audit.index(out, {"ExperimentID": HERE.name, "status":"SAMPLING_VERIFIED",
                      "CPU_wall_seconds":perf_counter()-began})
    audit.finish(out)
    print(f"{summary['p']} P x {summary['d']} D | {summary['pairs']} planned pairs | Results: {out}")


if __name__ == "__main__":
    main()
