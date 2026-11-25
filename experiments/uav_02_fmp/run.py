"""Validate FM_P and render an annotated VisDrone example."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code/tools"))
from uav_reproduction import VAL, output_folder, validate_uvl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="render",
                        choices=("render", "validate", "geometry", "generate", "audit-space"))
    parser.add_argument("--val", type=Path, default=VAL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--flamapy-python", type=Path)
    parser.add_argument("--sample-seed", type=int, default=20261109)
    parser.add_argument("--config-id")
    parser.add_argument("--image-index", type=int, choices=range(548), default=0, metavar="0..547")
    args = parser.parse_args()
    out = output_folder(HERE.name, args.output)
    if args.mode == "audit-space":
        from audit_space import audit_space
        print(f"FM_P validation output: {out}", flush=True)
        audit_space(out, args.flamapy_python)
        return
    result = validate_uvl("FM_P", args.sample_seed, args.flamapy_python)
    write_json(out / "flamapy_verification.json", result)
    if args.mode == "render":
        from demo import render
        render(args.val, out / "preview", args.config_id, args.sample_seed, args.image_index)
        receipt = json.loads((out / "preview/demo_receipt.json").read_text(encoding="utf-8"))
        expected = args.config_id or result["configuration_id"]
        if receipt["configuration_id"] != expected:
            raise ValueError("Rendered configuration differs from the chosen configuration.")
        print(f"Configuration {expected} | Preview: {out / 'preview/preview.png'}")
    if args.mode == "geometry":
        from validate_geometry import audit_geometry
        audit_geometry(args.val, out)
    if args.mode == "generate":
        from render_set import render_set
        folder = render_set(args.val, out, args.config_id or result["configuration_id"])
        print(f"548-image configuration: {folder}")
    print(f"FM_P: {result['valid_count']} valid configurations | Results: {out}")


if __name__ == "__main__":
    main()
