"""Automatic dashboard result discovery tests; no GPU or Streamlit required."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))

from dashboard_autodiscovery import (  # noqa: E402
    DataError, discover_studies, load_experiment, public_runs, safe_child,
)
from install_dashboard_autodiscovery import patch  # noqa: E402


class FakeStore:
    def __init__(self, root):
        self.root = root
        self.mode = 'cloud'

    def history(self, study):
        return []


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'reports/results'
        self.root.mkdir(parents=True)
        self.study = 'cv261102'
        self.run_id = 'cv261102_full_20261031T000013_d0be8587'
        self.folder = self.root / self.study / self.run_id
        self.folder.mkdir(parents=True)
        self.store = FakeStore(self.root)
        self.create_sample()

    def create_sample(self):
        data = self.folder / 'csv'
        data.mkdir()
        metrics = ('RunID,size,map50,map50_95,ar50_95,image_count\n'
                   f'{self.run_id},all,0.5041689748827122,0.29624432837203357,0.5317225023268554,548\n')
        (data / 'metrics_overall.csv').write_text(metrics, encoding='utf-8')
        (data / 'metrics_by_area.csv').write_text('size,map50,map50_95\ntiny,0.3562,0.1483\n', encoding='utf-8')
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.glob('*.csv')}
        self.manifest = {
            'RunID': self.run_id,
            'ExperimentID': self.study,
            'status': 'PASS',
            'scope': 'DEVELOPMENT_VALIDATION',
            'created_at_UTC': '2026-10-31T00:00:13+00:00',
            'configuration': {'train_zip': 'E:\\Sensitive\\Private\\Data.zip'},
            'result': {'images': 548, 'trained_epochs': 100,
                       'AP50': 0.5041689748827122,
                       'AP50_95': 0.29624432837203357,
                       'AR50_95': 0.5317225023268554,
                       'csv_SHA256': hashes},
            'training_wall_seconds': 26979.132,
        }
        self.save_manifest()

    def save_manifest(self):
        (self.folder / 'manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_cv_appears_without_study_registration(self):
        self.assertEqual(list(discover_studies(self.store).values()), [self.study])
        rows = public_runs(self.store, self.study)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['format'], 'experiment')

    def test_metrics_and_private_paths(self):
        loaded = load_experiment(self.store, self.study, self.run_id)
        self.assertEqual(loaded['images'], 548)
        self.assertEqual(loaded['epochs'], 100)
        self.assertAlmostEqual(loaded['metrics']['AP50_95'], .29624432837203357)
        self.assertEqual(len(loaded['tables']['metrics_by_area.csv']), 1)
        self.assertNotIn('Sensitive', json.dumps(loaded))
        self.assertNotIn('train_zip', json.dumps(loaded))

    def test_crlf_to_lf_conversion_is_accepted(self):
        path = self.folder / 'csv/metrics_overall.csv'
        original = path.read_bytes().replace(b'\n', b'\r\n')
        expected = hashlib.sha256(original).hexdigest()
        self.manifest['result']['csv_SHA256']['metrics_overall.csv'] = expected
        self.save_manifest()
        value = load_experiment(self.store, self.study, self.run_id)
        self.assertEqual(value['images'], 548)

    def test_csv_checksum_change_is_rejected(self):
        path = self.folder / 'csv/metrics_overall.csv'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(DataError, 'modified published CSV'):
            load_experiment(self.store, self.study, self.run_id)

    def test_manifest_csv_mismatch_is_rejected(self):
        self.manifest['result']['AP50'] = .999
        self.save_manifest()
        with self.assertRaisesRegex(DataError, 'metrics disagree'):
            load_experiment(self.store, self.study, self.run_id)

    def test_future_study_automatically_discovered(self):
        other_study = 'cv261109'
        other_id = 'cv261109_full_20261109T093000_sample'
        path = self.root / other_study / other_id
        shutil.copytree(self.folder, path)
        manifest = self.manifest.copy()
        manifest['RunID'] = other_id
        manifest['ExperimentID'] = other_study
        (path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        self.assertEqual(set(discover_studies(self.store).values()), {self.study, other_study})

    def test_schema2_snapshot_is_discovered(self):
        other_study = 'vm3_future_study'
        other_run = 'vm3_run_20261116T090000_example'
        run = self.root / other_study / other_run
        run.mkdir(parents=True)
        snapshot = b'{"RunID":"sample"}\n'
        (run / 'snapshot.json').write_bytes(snapshot)
        (run / 'manifest.json').write_text(json.dumps({
            'schema': 2,
            'RunID': other_run,
            'ExperimentID': other_study,
            'mode': 'full',
            'status': 'PASS',
            'bundle': 'snapshot.json',
            'SHA256': hashlib.sha256(snapshot).hexdigest(),
        }), encoding='utf-8')
        entries = public_runs(self.store, other_study)
        self.assertEqual(entries[0]['format'], 'snapshot')
        self.assertIn(other_study, discover_studies(self.store).values())

    def test_unlisted_csv_is_safely_read(self):
        path = self.folder / 'csv/extra_table.csv'
        path.write_text('label,score,private_path\nnew,0.25,E:/Sensitive/data\n', encoding='utf-8')
        info = load_experiment(self.store, self.study, self.run_id)
        self.assertEqual(info['tables']['extra_table.csv'][0], {'label': 'new', 'score': 0.25})

    def test_invalid_identity_is_not_listed(self):
        self.manifest['ExperimentID'] = 'another_experiment'
        self.save_manifest()
        self.assertEqual(discover_studies(self.store), {})

    def test_path_traversal_is_blocked(self):
        for bad in ('../private', '../../models', '..', 'x\\y', '/tmp/private'):
            with self.subTest(bad=bad), self.assertRaises(DataError):
                safe_child(self.root, bad, self.root)

    def test_checksum_metadata_is_required(self):
        self.manifest['result']['csv_SHA256']['metrics_overall.csv'] = 'not-a-sha'
        self.save_manifest()
        with self.assertRaisesRegex(DataError, 'Invalid CSV manifest entry'):
            load_experiment(self.store, self.study, self.run_id)

    def test_other_experiment_folder_isolated(self):
        (self.root / 'scratch' / 'not_a_run').mkdir(parents=True)
        self.assertEqual(len(discover_studies(self.store)), 1)


class InstallerTests(unittest.TestCase):
    def test_install_preserves_existing_pages(self):
        source = '''from dashboard_data import DashboardStore, DataError
store = DashboardStore()
STUDIES = {'VM1': 'vm1_flamapy_yolo26s_validation',
           'VM2': 'vm2_degradation_gap_analysis',
           'VM03': 'vm03_gpu_pilot'}
MODELS = {}
with st.sidebar:
    options = list(STUDIES) if store.mode == 'local' else list(STUDIES)[:2]
    label = st.selectbox('Experiment', options)
try:
    records = store.history(STUDIES[label])
except DataError:
    pass
'''
        output = patch(source)
        self.assertIn('STUDIES = discover_studies(store)', output)
        self.assertIn('render_discovered_study(store, STUDIES[label])', output)
        self.assertIn('records = store.history(STUDIES[label])', output)
        self.assertEqual(output, patch(output))
        compile(output, 'dashboard.py', 'exec')

    def test_unknown_dashboard_layout_refused(self):
        with self.assertRaises(ValueError):
            patch('print("Unexpected source")\n')


if __name__ == '__main__':
    unittest.main()
