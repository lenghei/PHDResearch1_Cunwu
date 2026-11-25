"""Compare both weekly UVL files with the original research plan PDF."""
import argparse
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from uav_reproduction import check_uvl, digest, write_json
from uav_week01_06 import PLAN_SHA


def blocks(pages):
    pages = [re.split(r"\nResearch plan: the gap", p, maxsplit=1)[0] for p in pages]
    fm_p = (pages[4][pages[4].index("namespace UAVDegradation"):] + "\n"
            + pages[5].split("Your tasks on this model:")[0])
    fm_d = (pages[6][pages[6].index("namespace YOLODetector"):].split("\nFeature\n")[0] + "\n"
            + pages[7] + "\n" + pages[8].split("Your tasks on this model:")[0])
    return {"FM_P": fm_p, "FM_D": fm_d}


def normalized(text):
    text = re.sub(r"\{[^{}]+\}", lambda m: " ".join(m[0].split()), text)
    text = re.sub(r"([&|])[ \t]*\n[ \t]*(?=!)", r"\1 ", text)
    return tuple((len(line) - len(line.lstrip()), " ".join(line.split()))
                 for line in text.splitlines() if line.strip())


def compare(plan):
    from pypdf import PdfReader

    if digest(plan) != PLAN_SHA:
        raise ValueError("The PDF differs from the frozen October 22 research plan.")
    pdf = PdfReader(plan)
    result = {}
    for domain, block in blocks([p.extract_text() for p in pdf.pages[:9]]).items():
        model = check_uvl(domain)
        actual, expected = normalized(model.read_text(encoding="utf-8")), normalized(block)
        if actual != expected:
            from difflib import unified_diff
            diff = "\n".join(unified_diff([str(x) for x in expected], [str(x) for x in actual]))
            raise ValueError(f"{domain} differs from the PDF:\n{diff}")
        result[domain] = {"uvl_sha256": digest(model), "pdf_pages": [5, 6] if domain == "FM_P" else [7, 8, 9],
                          "matching_nonempty_lines": len(actual)}
    return {"status": "MATCH", "source_pdf_sha256": PLAN_SHA,
            "normalization": "PDF page breaks, wrapped lines and nonsemantic spacing; tree indentation checked",
            "models": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.plan)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))
