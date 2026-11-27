"""Build the measured FM_D UVL and the per-leaf static coverage argument."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader
from flamapy.metamodels.pysat_metamodel.transformations.fm_to_pysat import FmToPysat

from export_configuration_validation import valid_config
from verify_static_source import FMP, SOURCE, SOURCE_SHA, verify

HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parents[1]
RESULTS = PUBLIC / "reports/results" / HERE.name
BASELINE = RESULTS / "review"
NATIVE = RESULTS / "reproduce_20261125T134035_61c7c6f2"
FOLLOWUP = RESULTS / "reproduce_20261125T142040_7a322c8d"
FMD = HERE / "fm_d.uvl"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def rows(folder, filename):
    path = folder / filename
    manifest = json.loads((folder / "manifest.json").read_bytes())
    if manifest["files"][filename] != digest(path):
        raise ValueError(f"Published evidence checksum differs: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def profile(source, row, training, record_id, run_id, scope, repeats=1):
    name = row.get("checkpoint", "yolov8s_cft640")
    checkpoint_sha = row["checkpoint_sha256"]
    architectures = {"yolov8":"YOLOv8", "yolo11":"YOLO11", "yolo26":"YOLO26", "leaf_yolo":"LEAF_YOLO"}
    arch = next(value for prefix,value in architectures.items() if prefix in name)
    scale = "Nano" if any(token in name for token in ("yolov8n","yolo26n","leaf_yolo_n")) else "Small"
    state = "COCO_ZeroShot" if scope == "common6" else "VisDrone_FT"
    train = "Train640" if state == "COCO_ZeroShot" else "Train" + training[name]["train_imgsz"]
    if state == "VisDrone_FT" and checkpoint_sha != training[name]["checkpoint_sha256"]:
        raise ValueError("Measured source and training checkpoint differ.")
    size = int(row["resolution"])
    infer = f"Infer{size}"
    strategy = row.get("strategy", "FullFrame")
    post = row.get("postprocessing", "NMS")
    precision = row.get("precision", "FP32")
    pre = row.get("preprocessing", "None")
    tta = row.get("TTA", "False") == "True"
    selected = {arch, scale, state, train, infer, strategy, post, precision,
                "HSV_Default","Geometric_Default"}
    if pre != "None":
        selected.add(pre)
    if tta:
        selected.add("TTA")
    if not valid_config(selected) or int(row["images"]) != 548:
        raise ValueError(f"Measurement does not bind to a legal 548-image setting: {record_id}")
    gflops = next(row[k] for k in ("mean_GFLOPs_2xMAC","GFLOPs_2xMAC","gflops_2xmac") if k in row)
    detector_ms = row.get("detector_mean_ms",row.get("predict_mean_ms"))
    detector_p95 = row.get("detector_p95_ms",row.get("predict_p95_ms"))
    latency = row.get("pipeline_mean_ms", row.get("cached_to_boxes_mean_ms"))
    p95 = row.get("pipeline_p95_ms", row.get("cached_to_boxes_p95_ms"))
    output = {"id":record_id, "checkpoint":name, "checkpoint_sha256":checkpoint_sha,
              "architecture":arch, "scale":scale, "weights":state, "train_resolution":train,
              "infer_resolution":infer, "strategy":strategy, "postprocessing":post, "precision":precision,
              "preprocessing":pre, "tta":tta, "training_augmentation_evidence":"NOT_APPLICABLE" if state=="COCO_ZeroShot" else "UNVERIFIED",
              "scope":scope, "images_per_pass":548,"repeats":repeats,
              "parameters":int(row["parameters"]),"gflops_2xmac_per_image":float(gflops),
              "detector_mean_ms_per_image":float(detector_ms),
              "detector_p95_ms_per_image":float(detector_p95),
              "cached_pipeline_mean_ms_per_image":float(latency),"cached_pipeline_p95_ms_per_image":float(p95),
              "peak_allocated_mib":float(row["peak_allocated_MiB"]),"peak_reserved_mib":float(row["peak_reserved_MiB"]),
              "run_id":run_id,
              "source_table":(source / {"resource_baseline":"full_validation_profiles.csv",
                                        "native10":"validation_metrics.csv","common6":"zeroshot_metrics.csv",
                                        "resource_broad":"inference_options.csv"}[scope]).relative_to(PUBLIC).as_posix()}
    if any(not isfinite(x) or x<=0 for x in output.values() if type(x) in (int,float)):
        raise ValueError(f"Invalid measured attribute: {record_id}")
    if row.get("FM_D_configuration_id"):
        output["declared_fmd_id"] = row["FM_D_configuration_id"]
    return output


def collect():
    training = {r["checkpoint"]:r for r in rows(NATIVE, "checkpoint_training.csv")}
    zero_sources = {r["checkpoint"]:r["sha256"] for r in rows(FOLLOWUP, "zeroshot_checkpoints.csv")}
    collected = []
    for r in rows(BASELINE, "full_validation_profiles.csv"):
        collected.append(profile(BASELINE,r,training,f"baseline_{r['resolution']}_{r['precision']}",
                                 r["RunID"],"resource_baseline",int(r["repeats"])))
    for r in rows(NATIVE, "validation_metrics.csv"):
        if r["FM_D_architecture_scale_eligible"] != "YES":
            continue
        collected.append(profile(NATIVE,r,training,f"native_{r['checkpoint']}_{r['resolution']}",
                                 r["RunID"],"native10"))
    for r in rows(FOLLOWUP, "zeroshot_metrics.csv"):
        if r["checkpoint_sha256"] != zero_sources[r["checkpoint"]]:
            raise ValueError("COCO weight identity differs.")
        collected.append(profile(FOLLOWUP,r,training,f"zero_{r['checkpoint']}",r["RunID"],"common6"))
    runs = {cell["artifact"].removesuffix(".json"):source["RunID"]
            for source in json.loads((FOLLOWUP/"manifest.json").read_bytes())["sources"]
            if source["kind"]=="inference_options" for cell in source["cells"]}
    for r in rows(FOLLOWUP, "inference_options.csv"):
        if r["allocated_before_case_bytes"] != "0":
            raise ValueError("Resource measurement did not isolate memory.")
        collected.append(profile(FOLLOWUP,r,training,r["case"],runs[r["case"]],"resource_broad"))
    if Counter(r["scope"] for r in collected) != {"resource_baseline":5,"native10":14,"common6":4,"resource_broad":10}:
        raise ValueError("Measurement roster differs.")
    return collected


def literal(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        if "'" in value:
            raise ValueError("Unexpected quote in a measured attribute.")
        return f"'{value}'"
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def render(records):
    base = FMD.read_text(encoding="utf-8")
    prefix = ("Detector {abstract, measurement_split 'VisDrone2019-DET-val Normal', "
              "measurement_hardware 'NVIDIA GeForce RTX 4060 Laptop GPU', "
              "measurement_binding 'Checkpoint SHA and runtime choices; training augmentation is separate',\n"
              "        measured_profiles [\n")
    lines = []
    for r in records:
        fields = ", ".join(f"{key} {literal(value)}" for key,value in r.items())
        lines.append("            {" + fields + "}")
    target = prefix + ",\n".join(lines) + "\n        ]}"
    return base.replace("Detector {abstract}", target, 1)


def check_uvl(path, records):
    old, new = (UVLReader(str(p)).transform() for p in (FMD,path))
    u, v = (FmToPysat(model).transform() for model in (old,new))
    loaded = next(a.default_value for a in new.root.attributes if a.name=="measured_profiles")
    expected = [{k:float(literal(x)) if isinstance(x,float) else x for k,x in record.items()} for record in records]
    if (loaded != expected or u.variables != v.variables
            or u.get_all_clauses().clauses != v.get_all_clauses().clauses
            or len(new.ctcs)!=4 or sum(not f.relations for f in new.get_features())!=34):
        raise ValueError("Measured UVL failed numeric read-back or changed Boolean semantics.")


def argument(leaf, family):
    if leaf == "Normal":
        return "IDENTITY", "No additional transform", "FM_P Normal gain 1 bias 0",
    if family == "illumination":
        mild = leaf.endswith("_mild")
        return ("APPROXIMATE" if mild else "GAP", "RandomHSV; gain 0.6-1.4, zero bias",
                "Gain is inside the range, but the nonzero additive offset is absent."
                if mild else "The required additive offset is absent; LowLight gain 0.4 also lies outside the HSV range.")
    if family == "motion":
        return "GAP", "Albumentations.Blur p=0.01", "Generic blur lacks a directional motion trajectory and cannot implement the FM_P motion kernel."
    if family == "defocus":
        return (("WEAK_PROXY" if leaf=="Defocus_r3" else "GAP"), "Albumentations.Blur and MedianBlur p=0.01 each",
                "Generic local smoothing only approximates small defocus; radius 3 is not pixel-equivalent."
                if leaf=="Defocus_r3" else "Default smoothing has no verified equivalent of the radius-6 defocus kernel.")
    if family == "gaussian":
        return "GAP", "Albumentations default list has no additive Gaussian noise", "Blur and HSV cannot add a stochastic Gaussian field."
    if family == "shot":
        return "GAP", "Albumentations default list has no photon-noise transform", "No default transform samples signal-dependent photon counts."
    if family == "compression":
        return "GAP", "Albumentations.ImageCompression p=0.0", "The transform is disabled, so none of the specified extra JPEG qualities is generated."
    if family == "atmosphere":
        return "GAP", "No haze or rain transform in the default list", ("No streak generator is included." if leaf=="Rain" else "No scattering or airlight model is included.")
    if family == "altitude":
        return "WEAK_PROXY", "v8_transforms: Mosaic and RandomPerspective(scale=hyp.scale)", "Plan scale=0.5 and Mosaic change object scale; they do not reproduce fixed downsample-upsample pixels."
    if family == "rolling":
        return "GAP", "v8_transforms: shear/perspective fed to RandomPerspective", "Plan sets both to zero; a global affine or perspective transform would still lack row-dependent motion."
    raise ValueError(f"Missing rationale for {leaf}.")


def static_table(source_path):
    source = verify(source_path)
    if source["coverage_counts"]["SUPERVISOR_PROXY"] != 18:
        raise ValueError("Unexpected static proxy coverage.")
    fm = UVLReader(str(FMP)).transform()
    leaves = {f.name:{a.name:a.default_value for a in f.attributes} for f in fm.get_features() if not f.relations}
    prior = rows(BASELINE,"static_rules_by_leaf.csv")
    if len(prior) != 25 or {r["leaf"] for r in prior} != set(leaves):
        raise ValueError("FM_P leaf coverage differs.")
    output = []
    for r in prior:
        name, family = r["leaf"], r["family"]
        decision, source_ref, why = argument(name, family)
        allowed = r["supervisor_proxy"] == "True"
        if allowed != (decision != "GAP"):
            raise ValueError(f"Static rationale differs from frozen proxy: {name}")
        output.append({"leaf":name, "family":family, "FM_P_UVL_parameters":json.dumps(leaves[name],sort_keys=True),
                       "default_training_reference":source_ref,"reason":why,"decision":decision,
                       "exact_identity":r["exact_identity"],"brightness_only":r["conservative"],
                       "plan_proxy_covered":r["supervisor_proxy"],
                       "default_source_version":"8.4.33","default_source_SHA256":SOURCE_SHA,
                       "default_cfg_SHA256":"3ef9fcf96280ba18b761a6165f7d77e14630b6759d5b32b28cd9d31025734dca",
                       "historical_activation":"NOT_REQUIRED_FOR_IDENTITY" if name=="Normal" else "UNVERIFIED"})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uvl",type=Path,default=HERE/"fm_d_measured.uvl")
    parser.add_argument("--rules",type=Path,default=RESULTS/"static_gap_rules.csv")
    parser.add_argument("--source",type=Path,default=SOURCE,help="Pinned Ultralytics v8.4.33 augment.py")
    args = parser.parse_args()
    if args.uvl.exists() or args.rules.exists():
        raise FileExistsError("Choose unused destinations for a new export.")
    records, rules = collect(), static_table(args.source)
    text = render(records)
    with TemporaryDirectory() as tmp:
        candidate = Path(tmp)/"fm_d_measured.uvl"
        candidate.write_text(text, encoding="utf-8",newline="\n")
        check_uvl(candidate, records)
    with args.uvl.open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(text)
    with args.rules.open("x",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,list(rules[0]))
        writer.writeheader()
        writer.writerows(rules)
    print(f"UVL: {args.uvl} | {len(records)} measured records; identical Boolean clauses.")
    print(f"Static Gap: {args.rules} | {len(rules)} argued leaf rules; SAT coverage 1 / 3 / 18.")


if __name__ == "__main__":
    main()
