"""Recompute the 1 / 3 / 18 static-coverage sensitivity policies from FM_P."""
from __future__ import annotations
import csv
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "reports" / "results" / HERE.name


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def main():
    fmp = load_module("fm_p_reference", HERE / "reference" / "fm_p_model_reference.py")
    cvmodel = load_module("fmd_coverage_rules", HERE / "model.py")
    rules = cvmodel.load_rules(HERE / "coverage_rules.json")
    configurations = list(fmp.combinations())
    strict = sum(1 for x in configurations if fmp.clean_selection(x))
    basic_levels = {"Normal", "Overexposed_mild", "Underexposed_mild"}
    basic = sum(1 for x in configurations if set(x).issubset(basic_levels))
    broad = cvmodel.gap_analysis(fmp, rules)
    if len(configurations) != 46368 or strict != 1 or basic != 3 or broad["proxy_covered"] != 18:
        raise ValueError("Static-gap policy counts differ from the recorded FM_P model/rules.")
    saved = json.loads((RESULTS / "static_gap.json").read_text(encoding="utf-8"))
    if sorted(saved["proxy"]["covered_configuration_ids"]) != sorted(broad["covered_configuration_ids"]):
        raise ValueError("Recomputed proxy-covered IDs differ from supplied static_gap.json.")

    policy_rows = [
        {"policy_id": "STRICT_IDENTITY", "covered": strict, "total": len(configurations),
         "gap": len(configurations)-strict, "gap_fraction": (len(configurations)-strict)/len(configurations),
         "coverage_interpretation": "EXACT identity only; Normal without degradation",
         "evidence_status": "RECOMPUTED_FROM_FM_P_REFERENCE_MODEL"},
        {"policy_id": "BRIGHTNESS_ONLY_INTERMEDIATE", "covered": basic, "total": len(configurations),
         "gap": len(configurations)-basic, "gap_fraction": (len(configurations)-basic)/len(configurations),
         "coverage_interpretation": "Normal + mild overexposure + mild underexposure; HSV gain proxy only",
         "evidence_status": "RECOMPUTED_INTERMEDIATE_SENSITIVITY_POLICY_NOT_A_SEPARATE_PLAN_TARGET"},
        {"policy_id": "SUPERVISOR_BROAD_PROXY", "covered": broad["proxy_covered"], "total": len(configurations),
         "gap": broad["proxy_gap"], "gap_fraction": broad["proxy_gap_fraction"],
         "coverage_interpretation": "3 illumination x 2 defocus states x 3 altitude states; weak proxy assumptions",
         "evidence_status": rules["verification_status"]},
    ]
    with (RESULTS / "static_gap_policy_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(policy_rows[0]))
        w.writeheader(); w.writerows(policy_rows)

    family_by_leaf = {leaf: family for family, leaves in fmp.GROUPS.items() for leaf in leaves}
    proxy_levels = rules["covered_levels"]
    exact = {"Normal"}
    leaf_rows = []
    for leaf in sorted(fmp.ALL_LEAVES):
        family = family_by_leaf[leaf]
        if leaf in exact:
            status = "EXACT"
            reason = "No degradation; identity baseline only."
        elif leaf in proxy_levels[family]:
            status = "APPROXIMATE"
            reason = next(r["rationale"] for r in rules["rules"] if r["feature_family"] == family)
        else:
            status = "NOT_COVERED"
            reason = next(r["rationale"] for r in rules["rules"] if r["feature_family"] == family)
        counts = broad["per_leaf"][leaf]
        leaf_rows.append({
            "FM_P_leaf": leaf, "family": family, "default_8433_proxy_status": status,
            "strict_identity_status": "EXACT" if leaf == "Normal" else "NOT_COVERED",
            "present_total": counts["present_total"], "covered_when_present_under_broad_policy": counts["covered_when_present"],
            "historical_checkpoint_actual_coverage": "UNKNOWN",
            "reason": reason,
            "evidence_needed": next(r["evidence_needed"] for r in rules["rules"] if r["feature_family"] == family),
            "limitation": "Proxy mapping from plan/rules; actual saved training args and Albumentations activation are not fully bound to this checkpoint."
        })
    with (RESULTS / "static_gap_leaf_rules.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(leaf_rows[0]))
        w.writeheader(); w.writerows(leaf_rows)

    summary = {
        "status": "PASS_STATIC_GAP_RECOMPUTATION",
        "fm_p_configuration_count": len(configurations),
        "strict_identity_covered": strict,
        "brightness_only_intermediate_covered": basic,
        "broad_proxy_covered": broad["proxy_covered"],
        "broad_proxy_gap": broad["proxy_gap"],
        "broad_proxy_gap_fraction": broad["proxy_gap_fraction"],
        "broad_proxy_covered_ids_match_supplied_evidence": True,
        "historical_checkpoint_actual_coverage": "UNKNOWN",
        "interpretation": "Static coverage proxy, not empirical detector pass/fail or AP evidence."
    }
    (RESULTS / "static_gap_recomputation.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
