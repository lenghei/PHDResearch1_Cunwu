
"""Export the full-space P cohorts and checkpoint-specific D candidates."""
import csv
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
W2 = HERE.parent / "uav_02_fmp"
W3 = HERE.parent / "uav_03_fmd_static_gap"


def checked(base, filename, audit):
    path = base / filename
    audit.source(path, audit.load(base / "manifest.json")["files"][filename])
    return path


def p_cohorts(manifest, configs, fmp, sampler, w2_proof):
    lookup = {fmp.configuration_id(c): c for c in configs}
    if len(lookup) != 46368:
        raise ValueError("FM_P IDs are not unique.")
    catalog_hash = w2_proof["sat_catalog"]["leaf_catalog_sha256"]
    payload = "".join("|".join(c) + "\n" for c in sorted(configs)).encode("ascii")
    if sha256(payload).hexdigest() != catalog_hash:
        raise ValueError("The sampling catalog differs from the Week 2 FlamaPy catalog.")

    pairwise = manifest["pairwise"]["cohort"]
    uniform = manifest["full_space_uniform_candidates"]
    groups = (("PAIRWISE_FULL", pairwise), ("UNIFORM_FULL", uniform))
    prefix = manifest["pairwise"]["greedy_cover_size"]
    all_ids, table = {}, []
    for cohort, records in groups:
        ids = set()
        for order, row in enumerate(records, 1):
            ident = row["configuration_id"]
            if ident in ids or lookup[ident] != tuple(row["selected_leaves"]):
                raise ValueError(f"Duplicate or invalid {cohort} row: {ident}")
            ids.add(ident)
            role = ("GREEDY_COVER" if order <= prefix else "SEEDED_FILL") if cohort == "PAIRWISE_FULL" else "UNIFORM_DRAW"
            table.append({"cohort": cohort, "selection_order": order, "p_id": ident,
                          "selected_leaves": "|".join(lookup[ident]), "selection_role": role,
                          "rolling_shutter": "RollingShutter" in lookup[ident], "status": "PLAN_ONLY"})
        all_ids[cohort] = ids

    safe = [c for c in configs if "RollingShutter" not in c]
    frozen = {r["degradation"]["degradation_id"] for r in manifest["pairs"]}
    leaves = sorted(fmp.ALL_LEAVES)
    _, full_mask = sampler.compute_binary_pair_masks(configs, leaves)
    _, safe_mask = sampler.compute_binary_pair_masks(safe, leaves)
    names = list(combinations(leaves, 2))
    gaps = []
    for cohort, ids, denominator in (
        ("FIRST500_UNIFORM_SAFE", frozen, safe_mask),
        ("PAIRWISE_FULL", all_ids["PAIRWISE_FULL"], full_mask),
        ("UNIFORM_FULL", all_ids["UNIFORM_FULL"], full_mask),
    ):
        _, observed = sampler.compute_binary_pair_masks((lookup[i] for i in ids), leaves)
        missing = denominator & ~observed
        if cohort == "PAIRWISE_FULL" and missing:
            raise ValueError("Pairwise cohort no longer covers all feasible states.")
        for i, (a,b) in enumerate(names):
            for first, second in ((0,0),(0,1),(1,0),(1,1)):
                if missing & (1 << (4*i + 2*first + second)):
                    gaps.append({"cohort": cohort, "leaf_a": a, "selected_a": first,
                                 "leaf_b": b, "selected_b": second})
    overlap = len(all_ids["PAIRWISE_FULL"] & all_ids["UNIFORM_FULL"])
    return table, gaps, {"pairwise_n": len(pairwise), "uniform_n": len(uniform),
                         "greedy_cover_n": prefix, "overlap_n": overlap,
                         "unique_full_space_candidates": len(pairwise)+len(uniform)-overlap,
                         "eligible_population": len(configs),
                         "uniform_inclusion_probability": len(uniform)/len(configs),
                         "frozen_safe_population": len(safe),
                         "frozen_uniform_inclusion_probability": len(frozen)/len(safe)}


def d_candidates(roster_path, trained_path, coco_path, audit):
    from export_configuration_validation import MANDATORY_AUG, PREPROCESS, sha_id, valid_config
    from plan_inference_pairwise import AXES, POSITIONS, plan

    trained = {r["checkpoint"]: r for r in audit.rows(trained_path)}
    coco = {r["checkpoint"]: r for r in audit.rows(coco_path)}
    roster = {r["checkpoint"]: r for r in audit.rows(roster_path)
              if r["counted_in_FM_D_pool"] == "YES"}
    design = plan(roster_path)
    rows, coverage = [], []
    for name, kind in design["checkpoint_templates"].items():
        entry = roster[name]
        zero = entry["weights"] == "COCO_ZeroShot"
        if zero:
            meta = coco[name]
            arch, scale, train, sha = (meta[k] for k in ("architecture","scale","FM_D_train_leaf","sha256"))
        else:
            meta = trained[name]
            arch = next(a for a in ("YOLOv8","YOLO11","YOLO26") if a.lower() in name) if "leaf_yolo" not in name else "LEAF_YOLO"
            scale = "Nano" if any(token in name for token in ("yolov8n","yolo26n","leaf_yolo_n")) else "Small"
            train, sha = "Train"+meta["train_imgsz"], meta["checkpoint_sha256"]
        if sha != entry["checkpoint_sha256"]:
            raise ValueError("Checkpoint roster SHA conflicts with its training record.")
        template = design["inference_pairwise_templates"][kind]
        levels = set()
        for setting in template["settings"]:
            selected = {*MANDATORY_AUG, arch, scale, entry["weights"], train,
                        *(setting[k] for k in ("resolution","strategy","postprocessing","precision")),
                        *(p for p in (*PREPROCESS,"TTA") if setting[p]=="on")}
            if not valid_config(selected):
                raise ValueError(f"Invalid inference assignment: {name} {setting}")
            did = sha_id("d_", selected)
            binding_id = sha_id("binding_", {did, sha})
            if setting["precision"]=="INT8":
                pending = "INT8_CALIBRATION_REQUIRED"
            elif setting["postprocessing"]=="EndToEnd_NMSFree" and setting["TTA"]=="on":
                pending = "END_TO_END_TTA_NEEDS_VALIDATION"
            else:
                pending = "CHECK_RUNTIME_AND_EVALUATION_PROTOCOL"
            levels.update((i,j,setting[AXES[i]],setting[AXES[j]]) for i,j in POSITIONS)
            rows.append({"checkpoint":name, "checkpoint_sha256":sha, "binding_id":binding_id,
                         "d_id":did, "declared_leaves":"|".join(sorted(selected)),
                         "evaluation_classes":entry["evaluation_classes"],
                         "training_augmentation_evidence":"NOT_APPLICABLE" if zero else "UNVERIFIED",
                         **setting, "readiness":pending, "status":"PLAN_ONLY"})
        if len(levels) != template["legal_pairs"]:
            raise ValueError(f"Missing inference pairs for {name}")
        coverage.append({"checkpoint":name, "template":kind, "sample_n":len(template["settings"]),
                         "covered_inference_level_pairs":len(levels),
                         "feasible_inference_level_pairs":template["legal_pairs"],
                         "evaluation_classes":entry["evaluation_classes"]})
    if len(roster) != 11 or len({r["binding_id"] for r in rows}) != len(rows):
        raise ValueError("Missing checkpoint or duplicate detector binding.")
    return rows, coverage


def export(out, manifest, configs, fmp, sampler, audit, results, flama_python):
    reports_3 = results / W3.name
    followup = reports_3 / "reproduce_20261125T142040_7a322c8d"
    native = reports_3 / "reproduce_20261125T134035_61c7c6f2"
    paths = [checked(followup,"checkpoint_roster.csv",audit),
             checked(native,"checkpoint_training.csv",audit),
             checked(followup,"zeroshot_checkpoints.csv",audit)]
    sys.path.insert(0, str(W3))
    proof_path = checked(results / W2.name / "review","full_space_validation.json",audit)
    proof = audit.load(proof_path)
    if proof["passed"]!=46368 or proof["failed"]!=0 or proof["sat_catalog"]["uvl_sha256"]!=audit.digest(W2/"fm_p.uvl"):
        raise ValueError("Week 2 full-space renderer proof is missing or differs.")
    p, missing, p_summary = p_cohorts(manifest, configs, fmp, sampler, proof)
    d, coverage = d_candidates(*paths,audit)
    audit.table(out/"full_space_p_samples.csv",p)
    audit.table(out/"uncovered_p_states.csv",missing,["cohort","leaf_a","selected_a","leaf_b","selected_b"])
    audit.table(out/"inference_pairwise_candidates.csv",d)
    audit.table(out/"inference_coverage.csv",coverage)
    checked_d = subprocess.run(
        [str(flama_python()),"-B",str(Path(__file__).resolve()),str(out/"inference_pairwise_candidates.csv")],
        check=True,capture_output=True,text=True,encoding="utf-8")
    count = int(checked_d.stdout.strip())
    if count != len(d):
        raise ValueError("FM_D SAT verification is incomplete.")
    code_paths = (Path(__file__), HERE/"run.py", W3/"plan_inference_pairwise.py",
                  W3/"export_configuration_validation.py", HERE.parent/"experiments_test/cv261123/sampling.py",
                  HERE.parent/"experiments_test/cv261207/sampling.py")
    return {"status":"SAMPLING_VERIFIED", "p":p_summary,"d":{"checkpoint_count":len(coverage),"logical_bindings":len(d),
                                "SAT_checked":count,"needs_int8_calibration":sum(r["precision"]=="INT8" for r in d),
                                "native10_candidates":sum(r["evaluation_classes"]=="native10" for r in d)},
            "active_selection":"WAITING_FOR_ADMITTED_TRAINING_LABELS", "AP_used_for_selection":False,
            "seed":manifest["seed"],"render_seed":manifest["render_seed"],
            "FM_P_UVL_sha256":{"weekly":audit.digest(W2/"fm_p.uvl"),"historical":manifest["FM_P_sha256"]},
            "FM_D_UVL_sha256":{"weekly":audit.digest(W3/"fm_d.uvl"),"historical":manifest["FM_D_sha256"]},
            "source_sha256":{p.relative_to(results).as_posix():audit.digest(p) for p in (*paths,proof_path)},
            "code_sha256":{p.relative_to(HERE.parents[1]).as_posix():audit.digest(p) for p in code_paths},
            "frozen_selection_sha256":audit.digest(out/"selection_manifest.json"),
            "target_split":manifest["target_split"],"measured_pairs_in_this_run":0}


def sat_check(path):
    from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader
    from flamapy.metamodels.pysat_metamodel.transformations.fm_to_pysat import FmToPysat
    from pysat.solvers import Solver

    model = UVLReader(str(W3/"fm_d.uvl")).transform()
    sat = FmToPysat(model).transform()
    leaves = {f.name:sat.variables[f.name] for f in model.get_features() if not f.relations}
    with path.open(encoding="utf-8-sig",newline="") as f:
        rows = list(csv.DictReader(f))
    with Solver(bootstrap_with=sat.get_all_clauses().clauses) as solver:
        for row in rows:
            selected = set(row["declared_leaves"].split("|"))
            if not selected <= leaves.keys() or not solver.solve(
                    assumptions=[v if k in selected else -v for k,v in leaves.items()]):
                raise ValueError("Declared FM_D candidate is unsatisfiable: "+row["binding_id"])
    return len(rows)


if __name__ == "__main__":
    print(sat_check(Path(sys.argv[1])))
