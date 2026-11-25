"""Tests for the read-only arithmetic check of reported RQ figures."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'supplemental_summary_audit.py'
spec = importlib.util.spec_from_file_location('supplemental_summary_audit', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

BASE = '''
| YOLO26s | +3.399 | [2.537, 4.260] |
| YOLOv8s | +3.637 | [2.754, 4.520] |
| YOLO11s | +3.823 | [2.923, 4.724] |
| exact identity | 1 | 46,367 |
| conservative source-supported proxy | 3 | 46,365 |
| supervisor provisional proxy | 18 | 46,350 |
Formal count 0/500. State: ML_INSUFFICIENT_EVIDENCE. RQ4: NOT_EVALUATED.
'''

class AuditTests(unittest.TestCase):
    def parse(self, contents):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'report.md'
            path.write_text(contents, encoding='utf-8')
            return audit.read_evidence(path)

    def test_historical_arithmetic(self):
        a = self.parse(BASE)
        self.assertEqual(a['model_counts']['cartesian_pairs'], 29485596672)
        self.assertEqual(a['historical_response_reconciliation']['unique_responses'],279)
        self.assertEqual([x['gap'] for x in a['coverage']], [46367,46365,46350])
        self.assertTrue(all(x['lower_bound_positive'] for x in a['interaction']))

    def test_reject_bad_gap_total(self):
        with self.assertRaises(ValueError):
            self.parse(BASE.replace('46,350','46,351'))

    def test_reject_invalid_ci(self):
        with self.assertRaises(ValueError):
            self.parse(BASE.replace('[2.537, 4.260]','[3.900, 4.260]'))

    def test_reject_missing_scope(self):
        with self.assertRaises(ValueError):
            self.parse(BASE.replace('ML_INSUFFICIENT_EVIDENCE','DATA_PRESENT'))

if __name__ == '__main__':
    unittest.main()
