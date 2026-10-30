"""Backfill VM1 root summary from saved experiment artifacts (CPU only)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main.run_report import generate_run_report, format_terminal


def main():
    parser = argparse.ArgumentParser(description="Create a VM1 run-level summary without GPU inference")
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    if not (root / "run.json").is_file() or not (root / "result.json").is_file():
        parser.error("Pass an existing completed VM1 RunID directory containing run.json and result.json")
    if not root.name.startswith("vm1_flamapy_yolo26s_validation_"):
        parser.error("This tool only supports the VM1 experiment")
    report = generate_run_report(root)
    print(format_terminal(report), flush=True)
    print("Report files:", root / "summary.json", root / "summary.txt", sep="\n", flush=True)


if __name__ == "__main__":
    main()
