"""Shared paths and output handling for the weekly entry points."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from uav_week01_06 import EXPS, PUBLIC, ROOT, RUNS, digest


RESULTS = PUBLIC / "reports/results"
MODELS = {
    "FM_P": (PUBLIC / "experiments/uav_02_fmp/fm_p.uvl",
             "e6eadd633d1c14e0d6bc425f5e54421d592d766c65a6e5f20c61bc34dd76fb5c"),
    "FM_D": (PUBLIC / "experiments/uav_03_fmd_static_gap/fm_d.uvl",
             "4022d674822b97dbbe3712646b9a0f970af95d175ae358e63582029bc107571a"),
}
VAL = ROOT / "datasets/VisDrone/VisDrone2019-DET-val.zip"


def output_folder(experiment, requested=None):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = (requested or RUNS / experiment / f"reproduce_{stamp}_{uuid4().hex[:8]}").resolve()
    if path.is_relative_to(PUBLIC.resolve()):
        raise ValueError("Choose an output directory outside the publication root.")
    path.mkdir(parents=True)
    return path


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check_uvl(domain):
    path, expected = MODELS[domain]
    if digest(path) != expected:
        raise ValueError(f"{domain} differs from the verified PDF transcription.")
    return path


def flama_python(python=None):
    python = python or ROOT / ".venv_flamapy/Scripts/python.exe"
    if not python.is_file():
        python = Path(sys.executable)
    return python


def validate_uvl(domain, seed=20261109, python=None):
    path = check_uvl(domain)
    validator = EXPS / "vm36_supervisor_reproduction/validate_models.py"
    process = subprocess.run([str(flama_python(python)), "-B", str(validator), "--model", str(path),
                              "--domain", domain, "--seed", str(seed)],
                             check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(process.stdout.splitlines()[-1])
