"""Local + published experiment discovery tests; no GPU or Streamlit required."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import dashboard_autodiscovery as discovery
from dashboard_data import DashboardStore, DataError


class LocalDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.public = self.base / 'main'
        self.public.mkdir()
        self.local_root = self.base / 'datasets/local_runs'
        self.local_root.mkdir(parents=True)
        self.study = 'cv261109'
        self.run_id = 'cv261109_fmp_p_936a456fc700e8d8_20261101T173419_099c950c'
        self.patcher = patch.object(discovery, 'PUBLIC', self.public)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.env = patch.dict('os.environ', {'UAVGAP_RUNS_ROOT': str(self.local_root)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.store = DashboardStore(mode='local', public=self.public)

    def publish(self):
        root = self.public / 'reports/results' / self.study / self.run_id
        root.mkdir(parents=True, exist_ok=True)
        data = {'RunID': self.run_id, 'ExperimentID': self.study, 'mode': 'FM_P_GENERATION',
                'status': 'PASS', 'files': {
                'result.json': {'status': 'PASS', 'scope': 'FULL_548_FM_P_GENERATION',
                                'generated_images': 548, 'configuration_id': 'p_936a456fc700e8d8',
                                'FlamaPy_valid_configurations': 46368, 'elapsed_seconds': 62.74},
                'summary.json': {'configuration': {'features': ['Normal', 'Motion_k15', 'JPEG_q50'],
                                'numeric_parameters': {'JPEG_q50': {'quality': 50}},
                                'effect_order': ['atmosphere','illumination','blur','rolling_shutter',
                                                 'noise','downscale','compression']},
                                'metrics': {'AP': 'NOT_EVALUATED', 'model_inference': 'NOT_RUN'}},
                'provenance.json': {'audit': {'status': 'PASS', 'files_checked': 548,
                                               'images_regenerated': 3}}}}
        content = json.dumps(data).encode('utf-8')
        (root / 'snapshot.json').write_bytes(content)
        (root / 'manifest.json').write_text(json.dumps({'schema': 2, 'RunID': self.run_id,
                    'ExperimentID': self.study, 'mode': 'FM_P_GENERATION', 'status': 'PASS',
                    'started_at': '2026-11-01T17:34:19+00:00', 'bundle': 'snapshot.json',
                    'SHA256': hashlib.sha256(content).hexdigest()}), encoding='utf-8')
        return root

    def create_raw_run(self, status='PASS'):
        folder = self.local_root / self.study / self.run_id
        folder.mkdir(parents=True, exist_ok=True)
        payload = b'{"image_index":1}\n'
        (folder / 'image_manifest.jsonl').write_bytes(payload)
        result = {'RunID': self.run_id, 'ExperimentID': self.study,
                  'status': status, 'scope': 'FULL_548_FM_P_GENERATION',
                  'generated_images': 548, 'configuration_id': 'p_936a456fc700e8d8',
                  'FlamaPy_valid_configurations': 46368, 'elapsed_seconds': 65.14,
                  'image_manifest_SHA256': hashlib.sha256(payload).hexdigest(),
                  'model_inference': 'NOT_RUN', 'AP_metrics': 'NOT_EVALUATED'}
        selection = {'RunID': self.run_id, 'ExperimentID': self.study,
                     'started_at_UTC': '2026-11-01T17:34:19+00:00',
                     'selection': {'configuration_id': 'p_936a456fc700e8d8',
                                   'selected_leaves': ['Normal', 'Motion_k15', 'JPEG_q50']},
                     'parameters_from_UVL': {'JPEG_q50': {'quality': 50}},
                     'effect_order': ['atmosphere','illumination','blur','rolling_shutter',
                                      'noise','downscale','compression']}
        (folder / 'result.json').write_text(json.dumps(result), encoding='utf-8')
        (folder / 'selection.json').write_text(json.dumps(selection), encoding='utf-8')
        return folder

    def test_published_fallback_without_local_run(self):
        self.publish()
        self.assertIn(self.study, discovery.discover_studies(self.store).values())
        records = discovery.public_runs(self.store, self.study)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['format'], 'published_snapshot')
        value = discovery.load_snapshot(self.store, self.study, self.run_id)
        self.assertEqual(value['images'], 548)
        self.assertEqual(value['generation']['valid_configurations'], 46368)
        self.assertEqual(value['generation']['features'], ['Normal', 'Motion_k15', 'JPEG_q50'])
        self.assertEqual(value['generation']['elapsed_seconds'], 62.74)
        self.assertEqual(value['generation']['audit']['files_checked'], 548)
        self.assertEqual(value['generation']['AP'], 'NOT_EVALUATED')

    def test_local_raw_run_auto_discovered_without_run_json(self):
        self.create_raw_run()
        records = discovery.public_runs(self.store, self.study)
        self.assertEqual(records[0]['format'], 'local_fmp')
        self.assertIn(self.study, discovery.discover_studies(self.store).values())
        value = discovery.load_local_run(self.store, self.study, self.run_id)
        self.assertEqual(value['images'], 548)
        self.assertEqual(value['generation']['features'][1], 'Motion_k15')
        self.assertEqual(value['generation']['elapsed_seconds'], 65.14)
        self.assertEqual(value['generation']['inference'], 'NOT_RUN')

    def test_local_run_preferred_over_same_published_run(self):
        self.publish()
        self.create_raw_run()
        records = discovery.public_runs(self.store, self.study)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['source'], 'local')
        self.assertEqual(records[0]['format'], 'local_fmp')

    def test_modified_public_snapshot_is_rejected(self):
        folder = self.publish()
        with (folder / 'snapshot.json').open('ab') as stream:
            stream.write(b' ')
        with self.assertRaisesRegex(DataError, 'checksum'):
            discovery.load_snapshot(self.store, self.study, self.run_id)

    def test_modified_local_image_manifest_is_rejected(self):
        folder = self.create_raw_run()
        (folder / 'image_manifest.jsonl').write_bytes(b'altered\n')
        with self.assertRaisesRegex(DataError, 'checksum'):
            discovery.load_local_run(self.store, self.study, self.run_id)

    def test_false_local_run_id_is_not_discovered(self):
        folder = self.create_raw_run()
        meta = folder / 'result.json'
        record = json.loads(meta.read_text())
        record['RunID'] = 'another_run'
        meta.write_text(json.dumps(record))
        self.assertEqual(discovery.public_runs(self.store, self.study), [])

    def test_valid_preview_pairs_only(self):
        folder = self.create_raw_run()
        preview = folder / 'preview'
        preview.mkdir()
        for prefix in ('original', 'degraded'):
            (preview / f'{prefix}_0001.jpg').write_bytes(b'JPEG bytes fixture')
        items = discovery.local_preview_pairs(folder)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0][0], b'JPEG bytes fixture')
        (preview / 'degraded_0001.jpg').unlink()
        self.assertEqual(discovery.local_preview_pairs(folder), [])

    def test_cloud_only_reads_public_results(self):
        self.create_raw_run()
        cloud = DashboardStore(mode='cloud', public=self.public)
        self.assertNotIn(self.study, discovery.discover_studies(cloud).values())
        self.publish()
        self.assertIn(self.study, discovery.discover_studies(cloud).values())


if __name__ == '__main__':
    unittest.main()
