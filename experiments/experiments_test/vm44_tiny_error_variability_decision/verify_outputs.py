"""Independent VM44 aggregate checks and read-only selector CLI checks."""
from collections import Counter
import argparse
import csv
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse, unquote

from vm44_io import HERE, RUNS, sha, save, read


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main(identity):
    started, cpu = time.perf_counter(), time.process_time()
    if Path(identity).name != identity:
        raise ValueError('Expected a VM44 RunID.')
    out = (RUNS/identity).resolve(strict=True)
    verified = 0
    for line in (out/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if sha(name) != digest: raise ValueError('Sealed output changed: '+name)
        verified += 1
    counts = Counter()
    with gzip.open(out/'matched_GT_transitions.csv.gz', 'rt', encoding='utf-8', newline='') as f:
        n = 0
        for row in csv.DictReader(f):
            for cat in ('0', row['category_id']):
                for area in ('all', row['size']):
                    counts[row['condition'], cat, area, row['transition']] += 1
            n += 1
    if n != 74687*2: raise ValueError('GT-level transition population differs.')
    transitions = rows(out/'VM43_object_match_transitions.csv')
    for r in transitions:
        for label in ('BOTH_DETECT', 'CONTROL_ONLY_LOST', 'GUIDED_ONLY_IMPROVED', 'BOTH_MISS'):
            assert int(r[label]) == counts[r['condition'], r['category_id'], r['size'], label]
        assert int(r['GT']) == sum(int(r[k]) for k in ('BOTH_DETECT', 'CONTROL_ONLY_LOST', 'GUIDED_ONLY_IMPROVED', 'BOTH_MISS'))
    fn = rows(out/'VM43_false_negative_analysis.csv')
    for r in fn:
        t = next(x for x in transitions if x['condition'] == r['condition'] and x['category_id'] == r['category_id'] and x['size'] == r['size'])
        tp = int(t['BOTH_DETECT'])+int(t['GUIDED_ONLY_IMPROVED' if r['strategy'] == 'VARIABILITY_GUIDED' else 'CONTROL_ONLY_LOST'])
        assert int(r['TP']) == tp and int(r['FN']) == int(t['GT'])-tp
    images = rows(out/'VM43_per_image_diagnostics.csv')
    image_counts = Counter()
    sums = Counter()
    seen = set()
    for r in images:
        key = (r['cell_id'], r['image_id'])
        assert key not in seen
        seen.add(key); image_counts[r['cell_id']] += 1
        assert int(r['raw_predictions'])-int(r['ignored_predictions']) == int(r['retained_predictions'])
        assert int(r['TP'])+int(r['FP']) == int(r['selected_predictions'])
        assert int(r['TP'])+int(r['FN']) == int(r['valid_GT'])
        for name in ('TP', 'FP', 'FN', 'retained_predictions'): sums[r['cell_id'], name] += int(r[name])
    assert len(image_counts) == 4 and set(image_counts.values()) == {1610}
    confidence = rows(out/'VM43_detection_confidence_analysis.csv')
    for cell in image_counts:
        total = sum(int(r['count']) for r in confidence if r['cell_id'] == cell and r['category_id'] == '0' and r['status'] == 'ALL_RETAINED')
        assert total == sums[cell, 'retained_predictions']
    rules = out/'FM_P_to_FM_D_decision_rules.json'
    cases = [(['Normal'], False, 'SELECTED_MEASURED_DETECTOR'), (['Normal', 'Motion_k21'], False, 'UNKNOWN_UNMEASURED_CONDITION'),
             (['LowLight'], False, 'INVALID_FM_P_CONFIGURATION'), (['Normal'], True, 'NO_CONFIRMED_FEASIBLE_DETECTOR')]
    commands, results = [], []
    for leaves, strict, expected in cases:
        command = [sys.executable, '-B', '-X', 'utf8', str(HERE/'run.py'), '--select', str(rules), '--leaves', *leaves]
        if strict: command.append('--strict-training')
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=60)
        assert result.returncode == 0, result.stderr
        answer = json.loads(result.stdout)
        assert answer['status'] == expected
        commands.append(command)
        results.append({'leaves': leaves, 'strict_training': strict, 'status': answer['status'], 'D': answer['selected']['D'] if answer['selected'] else None})
    class Links(HTMLParser):
        def __init__(self): super().__init__(); self.targets = []
        def handle_starttag(self, tag, attrs):
            self.targets.extend(value for key, value in attrs if key in ('src', 'href'))
    parser = Links(); parser.feed((out/'index.html').read_text(encoding='utf-8'))
    for value in parser.targets:
        u = urlparse(value)
        target = Path(unquote(u.path).lstrip('/')) if u.scheme == 'file' else out/value
        assert target.exists(), target
    findings = read(out/'mechanism_evidence.json')['findings']
    normalized = []
    for item in findings:
        item = dict(item)
        if item['status'] == 'DIRECTLY_MEASURED_FALSE':
            item.update(status='DIRECTLY_MEASURED', claim_supported=False)
        assert item['status'] in ('DIRECTLY_MEASURED', 'SUPPORTED_ASSOCIATION', 'PLAUSIBLE_NOT_PROVEN', 'NOT_EVALUATED')
        normalized.append(item)
    save(out/'verified_scientific_findings.json', {'findings': normalized, 'source_SHA256': sha(out/'mechanism_evidence.json'),
        'normalization': 'Separate evidence grade from the false claim value; no metric, diagnosis or conclusion changed.'})
    receipt = {'status': 'PASS', 'sealed_files_verified': verified, 'GT_transition_rows_independently_counted': n,
        'summary_transition_rows_verified': len(transitions), 'per_image_rows_verified': len(images), 'FN_rows_verified': len(fn),
        'selector_entrypoint_cases': results, 'actual_commands': commands, 'HTML_link_targets_verified': len(parser.targets),
        'IoU_distribution_population': 'Matched TP at score>=.25 andIoU>=.50; FN have null matched IoU',
        'confidence_population': 'Post-ignore retained scores, plus separately labelled diagnostic TP/FP',
        'CPU_wall_seconds': time.perf_counter()-started, 'CPU_process_seconds': time.process_time()-cpu, 'GPU_seconds': 0,
        'source': {'path': str(Path(__file__).resolve()), 'SHA256': sha(__file__)}, 'original_sealed_artifacts_modified': False}
    save(out/'independent_output_verification.json', receipt)
    files = [out/'verified_scientific_findings.json', out/'independent_output_verification.json', Path(__file__).resolve()]
    with (out/'QA_SHA256SUMS.txt').open('x', encoding='utf-8') as f:
        for p in files: f.write(sha(p)+'  '+str(p)+'\n')
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    main(parser.parse_args().run)
