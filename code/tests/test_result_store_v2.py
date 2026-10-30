"""Snapshot store tests; no Streamlit or GPU required."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PUBLIC / 'code/tools'))
from dashboard_data import DashboardStore, DataError


class PublishedRunTests(unittest.TestCase):
    def test_public_runs_are_independent(self):
        store = DashboardStore('cloud', public=PUBLIC)
        vm1 = store.history('vm1_flamapy_yolo26s_validation')
        vm2 = store.history('vm2_degradation_gap_analysis')
        self.assertEqual(len(vm1), 2)
        self.assertEqual(len(vm2), 1)
        latest = store.view('vm1_flamapy_yolo26s_validation', vm1[0]['RunID'])
        self.assertEqual(latest.read('summary.json')['dataset']['images_per_condition'], 548)
        self.assertEqual(latest.read('summary.json')['model']['parameter_count'], 9955604)
        self.assertAlmostEqual(latest.read('summary.json')['conditions'][1]['AP50_95'], .20729035461491044)
        self.assertEqual(len(store.view('vm2_degradation_gap_analysis', vm2[0]['RunID']).read('measured_grids.json')), 297)

    def test_remove_one_run_only(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / 'repo'
            shutil.copytree(PUBLIC / 'reports/results', repo / 'reports/results')
            store = DashboardStore('cloud', public=repo)
            runs = store.history('vm1_flamapy_yolo26s_validation')
            assert len(runs) == 2
            doomed = repo / 'reports/results/vm1_flamapy_yolo26s_validation' / runs[0]['RunID']
            shutil.rmtree(doomed)
            survivors = store.history('vm1_flamapy_yolo26s_validation')
            self.assertEqual(len(survivors), 1)
            self.assertEqual(len(store.history('vm2_degradation_gap_analysis')), 1)
            self.assertEqual(store.view('vm1_flamapy_yolo26s_validation', survivors[0]['RunID']).read('result.json')['status'], 'PASS')

    def test_missing_or_modified_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / 'repo'
            shutil.copytree(PUBLIC / 'reports/results', repo / 'reports/results')
            store = DashboardStore('cloud', public=repo)
            record = store.history('vm1_flamapy_yolo26s_validation')[0]
            path = repo / 'reports/results/vm1_flamapy_yolo26s_validation' / record['RunID'] / 'snapshot.json'
            path.write_bytes(path.read_bytes() + b' ')
            with self.assertRaisesRegex(DataError, 'checksum mismatch'):
                store.view(record['ExperimentID'], record['RunID'])
            path.unlink()
            with self.assertRaisesRegex(DataError, 'missing'):
                store.view(record['ExperimentID'], record['RunID'])

    def test_path_safety(self):
        store = DashboardStore('cloud', public=PUBLIC)
        for invalid in ['../private', '../../models', '/root', 'foo\\bar', '.']:
            with self.assertRaises(DataError):
                store.child(store.root, invalid)


if __name__ == '__main__':
    unittest.main()
