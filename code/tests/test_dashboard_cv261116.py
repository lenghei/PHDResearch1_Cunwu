"""CV261116 dashboard regression tests; no GPU and no Streamlit server required."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import dashboard_autodiscovery as discovery
from dashboard_data import DashboardStore, DataError


RUN_ID = 'cv261116_static_20261101T223010_016ba75c'


def dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


class FMDashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.public = base / 'main'
        self.public.mkdir()
        self.root = base / 'datasets/local_runs'
        self.root.mkdir(parents=True)
        self.study = 'cv261116'
        patcher = patch.object(discovery, 'PUBLIC', self.public)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = patch.dict(os.environ, {'UAVGAP_RUNS_ROOT': str(self.root)})
        env.start()
        self.addCleanup(env.stop)
        self.store = DashboardStore(mode='local', public=self.public)

    def local(self):
        folder = self.root / self.study / RUN_ID
        folder.mkdir(parents=True)
        model = {'FlamaPy': {'valid_count': 635904, 'training_variants': 844,
                            'leaf_count': 34, 'constraint_count': 4,
                            'dead_feature_count': 0, 'satisfiable': True,
                            'count_backend': 'EXACT_FACTORIZED_COUNT_OF_STRICT_UVL_V1',
                            'count_limitation': 'Not an independent FlamaPy BDD count.'},
                 'independent': {'independent_training_variants': 844}}
        gap = {'proxy': {'FM_P_configurations': 46368, 'proxy_covered': 18,
                         'proxy_gap': 46350, 'coverage_status': 'PROVISIONAL'},
               'strict_identity': {'covered': 1}}
        rules = {'rules': [{'feature_family': 'illumination', 'default_status': 'weak_proxy',
                            'rationale': 'Approximation', 'evidence_needed': 'Review training args'}]}
        audit = {'status': 'REQUIRES_HUMAN_REVIEW', 'actual_training_args': 'NOT_PROVIDED',
                 'installed_source': {'ultralytics_version': '8.4.61'}}
        run = {'RunID': RUN_ID, 'ExperimentID': self.study, 'status': 'PARTIAL',
               'mode': 'FM_D_STATIC_GAP'}
        for name, value in [('model_count.json', model), ('static_gap.json', gap),
                            ('coverage_rules.json', rules), ('training_source_audit.json', audit),
                            ('run.json', run)]:
            dump(folder / name, value)
        (folder / 'coverage_by_feature.csv').write_text(
            'feature,present_total,covered_when_present\n'
            'Defocus_r3,15456,9\nMotion_k15,13248,0\n', encoding='utf-8')
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in folder.iterdir() if p.is_file()}
        result = {'RunID': RUN_ID, 'ExperimentID': self.study, 'status': 'PARTIAL',
                  'mode': 'FM_D_STATIC_GAP', 'scope': 'SUPERVISOR_PROXY_GAP_NEEDS_TRAINING_SOURCE_REVIEW',
                  'model_validation': 'PASS', 'FM_D': model, 'FM_P_configurations': 46368,
                  'proxy_covered': 18, 'proxy_gap': 46350, 'strict_identity_covered': 1,
                  'gpu_measurements': 'NOT_COLLECTED', 'AP_metrics': 'NOT_EVALUATED',
                  'elapsed_seconds': 3.514, 'files_SHA256': files}
        dump(folder / 'result.json', result)
        return folder

    def published(self, valid=True):
        location = self.public / 'reports/results' / self.study / RUN_ID
        location.mkdir(parents=True)
        data = {'RunID': RUN_ID, 'ExperimentID': self.study, 'mode': 'FM_D_STATIC_GAP',
                'status': 'PARTIAL', 'files': {'run.json': {'RunID': RUN_ID},
                'result.json': {'mode': 'FM_D_STATIC_GAP', 'status': 'PARTIAL',
                                'FM_D_valid_count': 635904, 'FM_D_training_variants': 844,
                                'FM_P_total': 46368, 'proxy_covered': 18,
                                'proxy_gap': 46350 if valid else 46349,
                                'scope': 'PROVISIONAL_SUPERVISOR_PROXY_NO_AP'},
                'summary.json': {'status': 'PARTIAL', 'scope': 'PROVISIONAL_SUPERVISOR_PROXY_NO_AP'}}}
        path = location / 'snapshot.json'
        dump(path, data)
        dump(location / 'manifest.json', {'schema': 2, 'RunID': RUN_ID,
            'ExperimentID': self.study, 'mode': 'FM_D_STATIC_GAP', 'status': 'PARTIAL',
            'started_at': '2026-11-01T22:30:10+00:00', 'bundle': 'snapshot.json',
            'SHA256': hashlib.sha256(path.read_bytes()).hexdigest()})
        return location

    def test_local_fmd_discovered_and_renderable(self):
        self.local()
        self.assertIn(self.study, discovery.discover_studies(self.store).values())
        rows = discovery.public_runs(self.store, self.study)
        self.assertEqual(rows[0]['format'], 'local_json')
        data = discovery.load_local_run(self.store, self.study, RUN_ID)
        fmd = data['static_gap']
        self.assertEqual(data['status'], 'PARTIAL')
        self.assertEqual(fmd['valid_configurations'], 635904)
        self.assertEqual(fmd['training_variants'], 844)
        self.assertEqual(fmd['leaves'], 34)
        self.assertEqual(fmd['constraints'], 4)
        self.assertEqual(fmd['strict_covered'], 1)
        self.assertEqual(fmd['proxy_gap'], 46350)
        self.assertEqual(len(fmd['per_feature']), 2)
        self.assertEqual(len(fmd['coverage_rules']), 1)
        self.assertEqual(fmd['source_audit_status'], 'REQUIRES_HUMAN_REVIEW')
        self.assertEqual(fmd['gpu_status'], 'NOT_COLLECTED')
        self.assertEqual(fmd['ap_status'], 'NOT_EVALUATED')
        self.assertFalse(fmd['published_summary_only'])

    def test_local_evidence_sha_tampering_rejected(self):
        folder = self.local()
        (folder / 'static_gap.json').write_text('{}\n', encoding='utf-8')
        with self.assertRaisesRegex(DataError, 'checksum'):
            discovery.load_local_run(self.store, self.study, RUN_ID)

    def test_csv_tampering_rejected(self):
        folder = self.local()
        with (folder / 'coverage_by_feature.csv').open('a') as f:
            f.write('Fake,50,50\n')
        with self.assertRaisesRegex(DataError, 'checksum'):
            discovery.load_local_run(self.store, self.study, RUN_ID)

    def test_unchanged_sha_but_conflicting_result_count_rejected(self):
        folder = self.local()
        value = json.loads((folder / 'result.json').read_text())
        value['proxy_covered'] = 35
        dump(folder / 'result.json', value)
        with self.assertRaisesRegex(DataError, 'disagree'):
            discovery.load_local_run(self.store, self.study, RUN_ID)

    def test_local_run_status_mismatch_rejected(self):
        folder = self.local()
        value = json.loads((folder / 'run.json').read_text())
        value['status'] = 'PASS'
        dump(folder / 'run.json', value)
        result = json.loads((folder / 'result.json').read_text())
        result['files_SHA256']['run.json'] = hashlib.sha256((folder / 'run.json').read_bytes()).hexdigest()
        dump(folder / 'result.json', result)
        with self.assertRaisesRegex(DataError, 'status or identity'):
            discovery.load_local_run(self.store, self.study, RUN_ID)

    def test_published_fmd_summary_automatically_discovered(self):
        self.published()
        cloud = DashboardStore(mode='cloud', public=self.public)
        self.assertIn(self.study, discovery.discover_studies(cloud).values())
        fmd = discovery.load_snapshot(cloud, self.study, RUN_ID)['static_gap']
        self.assertEqual(fmd['valid_configurations'], 635904)
        self.assertEqual(fmd['proxy_covered'], 18)
        self.assertEqual(fmd['proxy_gap'], 46350)
        self.assertTrue(fmd['published_summary_only'])
        self.assertEqual(fmd['coverage_rules'], [])

    def test_invalid_public_gap_rejected(self):
        self.published(valid=False)
        with self.assertRaisesRegex(DataError, 'inconsistent'):
            discovery.load_snapshot(self.store, self.study, RUN_ID)

    def test_local_priority_over_published(self):
        self.published()
        self.local()
        rows = discovery.public_runs(self.store, self.study)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['source'], 'local')

    def test_fmp_generation_normalizer_is_unchanged(self):
        generated = discovery._generated_summary(
            {'configuration_id': 'p_sample', 'FlamaPy_valid_configurations': 46368,
             'elapsed_seconds': 62.74},
            {'configuration': {'features': ['Normal', 'JPEG_q50']}}, {})
        self.assertEqual(generated['valid_configurations'], 46368)
        self.assertEqual(generated['features'], ['Normal', 'JPEG_q50'])
        self.assertEqual(generated['elapsed_seconds'], 62.74)

    def test_dedicated_fmd_renderer_shows_scientific_results(self):
        class Recorder:
            def __init__(self):
                self.labels = []
            def subheader(self, x): self.labels.append(str(x))
            def metric(self, label, value): self.labels.append((label, str(value)))
            def caption(self, value): self.labels.append(str(value))
            def warning(self, value): self.labels.append(str(value))
            def info(self, value): self.labels.append(str(value))
            def progress(self, value): self.labels.append(('progress', value))
            def columns(self, count): return [self for _ in range(count)]
            def dataframe(self, frame, **kwargs): self.labels.append(('table', len(frame)))
            def bar_chart(self, frame): self.labels.append(('chart', len(frame)))
        import pandas as pd
        self.local()
        obj = discovery.load_local_run(self.store, self.study, RUN_ID)
        fake = Recorder()
        discovery._render_fmd_static(fake, pd, obj, obj['static_gap'])
        self.assertIn(('Valid FM_D configurations', '635,904'), fake.labels)
        self.assertIn(('Proxy covered', '18'), fake.labels)
        self.assertIn(('Proxy uncovered', '46,350'), fake.labels)
        self.assertIn(('Training source audit', 'REQUIRES_HUMAN_REVIEW'), fake.labels)
        self.assertIn(('GPU profiling', 'NOT_COLLECTED'), fake.labels)
        self.assertIn(('table', 1), fake.labels)
        self.assertIn(('chart', 2), fake.labels)

    def test_gpu_profile_fields_are_not_replaced_with_ap(self):
        value = discovery._fmd_gpu_summary({'mode': 'FM_D_GPU_PILOT',
            'sample_images': 16, 'mean_ms_per_image': 7.7,
            'peak_pytorch_allocated_bytes': 200000, 'GFLOPs_estimate': None})
        self.assertEqual(value['mean_ms_per_image'], 7.7)
        self.assertNotIn('AP50', value)
        self.assertNotIn('GFLOPs_estimate', value)


if __name__ == '__main__':
    unittest.main()
