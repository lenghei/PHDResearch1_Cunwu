"""Plan pairwise inference settings for the Week 3 checkpoint roster."""
import argparse
from itertools import combinations, product
import json
from pathlib import Path

from export_configuration_validation import INFER_RES, STRATEGY, POST, PRECISION, PREPROCESS
from export_week3_followup import table

AXES = ("resolution","strategy","postprocessing","precision",*PREPROCESS,"TTA")
POSITIONS = tuple(combinations(range(len(AXES)),2))


def pairwise(allow_end2end):
    legal = [row for row in product(INFER_RES,STRATEGY,POST if allow_end2end else ("NMS",),
                                    PRECISION,*(("off","on"),)*5)
             if row[1]!="SAHI_Slicing" or row[-1]=="off"]
    coverage = [frozenset((i,j,row[i],row[j]) for i,j in POSITIONS) for row in legal]
    missing = set().union(*coverage)
    target = len(missing)
    selected = []
    while missing:
        index = max(range(len(legal)),key=lambda i:len(missing & coverage[i]))
        hit = coverage[index] & missing
        if not hit:
            raise ValueError("A legal pair is still missing.")
        selected.append(dict(zip(AXES,legal[index])))
        missing -= hit
    return {"logical_candidates":len(legal),"legal_pairs":target,"covered_pairs":target,
            "settings":selected}


def plan(path):
    checkpoints=[row["checkpoint"] for row in table(path) if row["counted_in_FM_D_pool"]=="YES"]
    templates={"NMS":pairwise(False),"YOLO26":pairwise(True)}
    return {"scope":"LOGIC_ONLY_PROPOSAL",
            "inference_pairwise_templates":templates,
            "checkpoint_templates":{name:"YOLO26" if "yolo26" in name else "NMS" for name in checkpoints},
            "technical_admission":{"INT8":"CALIBRATED_BACKEND_REQUIRED",
                                   "TTA_with_EndToEnd_NMSFree":"BACKEND_VALIDATION_REQUIRED",
                                   "all_other_rows":"CHECK_SETTINGS_BEFORE_EXECUTION"}}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roster",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    payload=json.dumps(plan(args.roster),indent=2)+"\n"
    if args.output:
        with args.output.open("x",encoding="utf-8") as stream:
            stream.write(payload)
    else:
        print(payload,end="")
