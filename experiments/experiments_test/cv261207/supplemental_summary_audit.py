"""Recalculate *reported* RQ figures from the human-readable CV261207 acceptance record.

This script audits arithmetic and report consistency only. It does not rerun models,
bootstrap, derive independent detector observations, or establish test independence.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

SOURCE_INTERACTION_RE = re.compile(
    r'^\|\s*(YOLO26s|YOLOv8s|YOLO11s)\s*\|\s*\+([0-9.]+)\s*\|\s*\[([0-9.]+),\s*([0-9.]+)\]\s*\|$',
    re.MULTILINE)
SOURCE_POLICY_RE = re.compile(
    r'^\|\s*(exact identity|conservative source-supported proxy|supervisor provisional proxy)\s*\|\s*([0-9,]+)\s*\|\s*([0-9,]+)\s*\|$',
    re.MULTILINE)
MODEL_COUNTS = (46368, 635904)
HISTORICAL_RESPONSE_ITEMS = (243, 54, 18, 279)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_evidence(path: Path) -> dict:
    source = path.read_text(encoding='utf-8-sig')
    interactions = []
    for model, effect, lo, hi in SOURCE_INTERACTION_RE.findall(source):
        effect, lo, hi = map(float, (effect, lo, hi))
        if not lo <= effect <= hi or lo <= 0:
            raise ValueError(f'Invalid effect or bounds: {model}')
        interactions.append({'model':model,'interaction_pp':effect,
            'simultaneous_ci_low_pp':lo,'simultaneous_ci_high_pp':hi,
            'interval_width_pp':round(hi-lo, 3), 'lower_bound_positive': lo > 0})
    coverage = []
    for policy, covered, gap in SOURCE_POLICY_RE.findall(source):
        covered = int(covered.replace(',', ''))
        gap = int(gap.replace(',', ''))
        if covered + gap != MODEL_COUNTS[0]:
            raise ValueError(f'Gap count inconsistency: {policy}')
        coverage.append({'policy':policy,'covered':covered,'gap':gap,
            'coverage_percent':round(100*covered/MODEL_COUNTS[0],8),
            'gap_percent':round(100*gap/MODEL_COUNTS[0],8)})
    if len(interactions) != 3 or len(coverage) != 3:
        raise ValueError(f'Expected 3 interaction and 3 coverage rows; got {len(interactions)}, {len(coverage)}')
    if 'ML_INSUFFICIENT_EVIDENCE' not in source or 'NOT_EVALUATED' not in source:
        raise ValueError('Missing scientific limitation markers')
    if '0/500' not in source:
        raise ValueError('Missing formal test-dev pair count')
    source_audited = 'report_arithmetic_only_not_raw_prediction_validation'
    return {'scope':source_audited,'source_filename':path.name,
        'source_sha256':sha256(path), 'model_counts':{'FM_P':MODEL_COUNTS[0],'FM_D':MODEL_COUNTS[1],
        'cartesian_pairs':MODEL_COUNTS[0]*MODEL_COUNTS[1]},
        'historical_response_reconciliation':{'VM09':243,'VM26':54,'overlap':18,
        'unique_responses':243+54-18,'reported_total':279},
        'interaction':interactions,'coverage':coverage,
        'planned_image_evaluations':{'v1_500_x_1610':500*1610,'v2_900_x_1610':900*1610},
        'scientific_limits':['All interactions are development-only, not independent test evidence.',
        'Joint CIs are copied from the source report, not recomputed from bootstrap samples.',
        'Static gap counts are policy-dependent coverage counts, not observed detector failures.',
        'No new GPU experiment, trained model, held-out ML evaluation, or RQ4 result is created.']}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    result = read_evidence(a.report.resolve())
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir/'supplemental_report_audit.json').write_text(json.dumps(result,ensure_ascii=False,
       indent=2,allow_nan=False)+'\n', encoding='utf-8')
    write_csv(a.output_dir/'reported_interaction_arithmetic.csv', result['interaction'])
    write_csv(a.output_dir/'reported_static_gap_arithmetic.csv', result['coverage'])
    print('REPORT ARITHMETIC PASS: 3 interactions; 3 coverage policies; no new experiments claimed')


if __name__ == '__main__':
    main()
