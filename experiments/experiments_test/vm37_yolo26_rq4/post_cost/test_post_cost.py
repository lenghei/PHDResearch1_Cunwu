"""CPU checks for budgets, execution gates and paired contrasts."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST))
sys.path.insert(0, str(POST.parent))
from safety import available_seconds, decide
from portable import ROOT, authorize, within, matrix, load, renderer, install_accumulator, digest, execution_remaining_seconds, verify_bundle
from evaluate import paired_contrasts


class PostCostTests(unittest.TestCase):
    def test_absolute_deadline_limits_unused_allocation(self):
        now = datetime(2026, 11, 12, tzinfo=timezone.utc)
        ledger = {'total_seconds': 172800, 'completed_workers': [{'charged_seconds': 6000}],
                  'absolute_deadline_UTC': (now + timedelta(seconds=3600)).isoformat()}
        self.assertEqual(available_seconds(ledger, now), 3600)
        self.assertEqual(available_seconds(ledger, now + timedelta(days=2)), 0)

    def test_cumulative_usage_limits_deadline(self):
        now = datetime.now(timezone.utc)
        ledger = {'total_seconds': 100, 'completed_workers': [{'charged_seconds': 90}],
                  'absolute_deadline_UTC': (now + timedelta(days=2)).isoformat()}
        self.assertEqual(available_seconds(ledger, now), 10)

    def test_temperature_hard_stop(self):
        policy = {'immediate_stop_temperature_C': 90, 'sustained_stop_temperature_C': 87, 'sustained_seconds': 30}
        self.assertEqual(decide(policy, [{'temperature_C': 90, 'monotonic_seconds': 0}]), 'TEMPERATURE_HARD_STOP')
        self.assertEqual(decide(policy, [], False), 'TELEMETRY_FAILURE')

    def test_temperature_sustained_and_reset(self):
        policy = {'immediate_stop_temperature_C': 90, 'sustained_stop_temperature_C': 87, 'sustained_seconds': 30}
        samples = [{'temperature_C': 88, 'monotonic_seconds': t} for t in (0, 15, 30)]
        self.assertEqual(decide(policy, samples), 'SUSTAINED_TEMPERATURE_STOP')
        self.assertIsNone(decide(policy, samples + [{'temperature_C': 86, 'monotonic_seconds': 31}]))

    def test_default_authorization_refuses_execution(self):
        with self.assertRaises(PermissionError):
            authorize(Path('.'), {'id': 'VM37_FORMAL100'}, {'approved': False})

    def test_full_pair_resource_gate(self):
        folder = ROOT / 'datasets/local_runs'
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            bundle = Path(temp)
            (bundle / 'bundle_manifest.json').write_text('{}', encoding='utf-8')
            (bundle / 'cost.json').write_text('{"fixture":true}', encoding='utf-8')
            protocol = {'id': 'TEST_ONLY', 'architectures': ['YOLO26s'], 'seeds': [42], 'arms': ['CONTROL', 'ALL_FIVE']}
            auth = {'approved': True, 'protocol_id': 'TEST_ONLY', 'bundle_manifest_SHA256': digest(bundle / 'bundle_manifest.json'),
                'arms': [list(row) for row in matrix(protocol)], 'cooling_confirmed': True, 'power_confirmed': True,
                'machine_admitted': True, 'total_seconds': 1000, 'completed_workers': [],
                'absolute_deadline_UTC': (datetime.now(timezone.utc) + timedelta(seconds=600)).isoformat(),
                'resource_admission': {'status': 'PASS', 'ALL_FIVE_GPU_cost_measured': True,
                    'required_complete_matrix_seconds': 100, 'cost_evidence_relative': 'cost.json',
                    'cost_evidence_SHA256': digest(bundle / 'cost.json')}}
            self.assertGreater(authorize(bundle, protocol, auth), 100)
            auth['resource_admission']['required_complete_matrix_seconds'] = 700
            with self.assertRaises(PermissionError):
                authorize(bundle, protocol, auth)
            auth['resource_admission']['required_complete_matrix_seconds'] = 100
            auth['arms'] = auth['arms'][:1]
            with self.assertRaises(PermissionError):
                authorize(bundle, protocol, auth)
            auth['arms'] = [list(row) for row in matrix(protocol)]
            auth['resource_admission']['ALL_FIVE_GPU_cost_measured'] = False
            with self.assertRaises(PermissionError):
                authorize(bundle, protocol, auth)

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            within(POST, '../training.py')

    def test_original_charges_and_current_execution_both_count(self):
        now = datetime.now(timezone.utc)
        auth = {'total_seconds': 1000, 'completed_workers': [{'charged_seconds': 200}],
                'absolute_deadline_UTC': (now + timedelta(seconds=2000)).isoformat()}
        self.assertEqual(execution_remaining_seconds(auth, 300, now), 500)
        auth['absolute_deadline_UTC'] = (now + timedelta(seconds=100)).isoformat()
        self.assertEqual(execution_remaining_seconds(auth, 300, now), 100)
        self.assertEqual(execution_remaining_seconds(auth, 900, now), 0)

    def test_later_arm_does_not_readmit_the_original_whole_matrix(self):
        folder = ROOT / 'datasets/local_runs'
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            bundle = Path(temp)
            (bundle / 'bundle_manifest.json').write_text('{}', encoding='utf-8')
            (bundle / 'cost.json').write_text('{"fixture":true}', encoding='utf-8')
            protocol = {'id': 'TEST_ONLY', 'architectures': ['YOLO26s'], 'seeds': [42], 'arms': ['CONTROL', 'ALL_FIVE']}
            auth = {'approved': True, 'protocol_id': 'TEST_ONLY', 'bundle_manifest_SHA256': digest(bundle / 'bundle_manifest.json'),
                'arms': [list(row) for row in matrix(protocol)], 'cooling_confirmed': True, 'power_confirmed': True,
                'machine_admitted': True, 'total_seconds': 1000, 'completed_workers': [],
                'absolute_deadline_UTC': (datetime.now(timezone.utc) + timedelta(seconds=2000)).isoformat(),
                'resource_admission': {'status': 'PASS', 'ALL_FIVE_GPU_cost_measured': True,
                    'required_complete_matrix_seconds': 900, 'cost_evidence_relative': 'cost.json',
                    'cost_evidence_SHA256': digest(bundle / 'cost.json')}}
            self.assertEqual(authorize(bundle, protocol, auth), 1000)
            auth['completed_workers'].append({'charged_seconds': 500})
            with self.assertRaises(PermissionError):
                authorize(bundle, protocol, auth)
            self.assertEqual(authorize(bundle, protocol, auth, initial_admission=False), 500)

    def test_changed_frozen_asset_is_rejected(self):
        folder = ROOT / 'datasets/local_runs'
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            bundle = Path(temp)
            (bundle / 'asset.txt').write_text('original', encoding='utf-8')
            (bundle / 'bundle_manifest.json').write_text(json.dumps({'files': {'asset.txt': digest(bundle / 'asset.txt')}}), encoding='utf-8')
            verify_bundle(bundle)
            (bundle / 'asset.txt').write_text('changed', encoding='utf-8')
            with self.assertRaises(ValueError):
                verify_bundle(bundle)

    def test_frozen_horizons_and_counts(self):
        from study import expected_steps
        pilot = load(POST / 'PILOT20_PROTOCOL.json')
        formal = load(POST / 'FORMAL100_PROTOCOL.json')
        self.assertEqual(len(matrix(pilot)), 2)
        self.assertEqual(len(matrix(formal)), 12)
        self.assertEqual(expected_steps(6471, 20)['updates'], 2827)
        self.assertEqual(expected_steps(6471, 20)['batches'], 64720)
        self.assertEqual(expected_steps(6471, 100)['updates'], 10917)
        self.assertEqual(pilot['schedule_horizon_epochs'], 20)
        self.assertEqual(formal['schedule_horizon_epochs'], 100)
        self.assertEqual(pilot['shared_training_args'], formal['shared_training_args'])

    def test_paired_difference_and_loss_change(self):
        values = {}
        for arm, normal, degraded in (('CONTROL', .4, .1), ('ALL_FIVE', .5, .3)):
            for context, value in (('Normal', normal), ('JPEG_Q25', degraded)):
                areas = [{'size': size, **{metric: value for metric in ('map50', 'map50_95', 'ar50_95')}}
                         for size in ('all', 'tiny', 'small', 'medium', 'large')]
                draws = {'image_ids': [1, 2], 'samples': 3, 'seed': 42, 'sampling': 'test',
                    'values': {r['size']: {metric: [value] * 3 for metric in ('map50', 'map50_95', 'ar50_95')} for r in areas}}
                for resolution in (640, 1280):
                    values[('YOLO26s', 42, arm, resolution, context)] = {'metrics': {'per_area': areas}, 'draws': draws}
        rows = paired_contrasts(values)
        row = next(r for r in rows if r['context'] == 'JPEG_Q25' and r['size'] == 'all')
        self.assertAlmostEqual(row['ALL_FIVE_minus_CONTROL'], .2)
        self.assertAlmostEqual(row['degradation_loss_change'], -.1)
        self.assertAlmostEqual(row['difference_CI95'][0], .2)
        broken = json.loads(json.dumps(values[('YOLO26s', 42, 'ALL_FIVE', 640, 'JPEG_Q25')]))
        broken['draws']['seed'] = 999
        values[('YOLO26s', 42, 'ALL_FIVE', 640, 'JPEG_Q25')] = broken
        with self.assertRaises(ValueError):
            paired_contrasts(values)

    def test_renderer_geometry_and_repeat(self):
        import numpy as np
        ops = renderer()
        image = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
        case = load(POST / 'PILOT20_PROTOCOL.json')['evaluation']['contexts'][-1]
        first = ops.generate(image, case['p']['leaves'], case['p']['attributes'], 'synthetic_fixture', 20261108)
        second = ops.generate(image, case['p']['leaves'], case['p']['attributes'], 'synthetic_fixture', 20261108)
        self.assertEqual(first[0].shape, image.shape)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_bootstrap_reaccumulates_duplicate_images(self):
        install_accumulator()
        from main.statistics import evaluate_coco, resample_evaluator, _metric_rows
        gt = {'info': {}, 'images': [{'id': 1}, {'id': 2}], 'categories': [{'id': 1, 'name': 'fixture'}],
              'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 8, 8], 'area': 64, 'iscrowd': 0},
                              {'id': 2, 'image_id': 2, 'category_id': 1, 'bbox': [0, 0, 8, 8], 'area': 64, 'iscrowd': 0}]}
        predictions = [{'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 8, 8], 'score': .9}]
        _, _, evaluator = evaluate_coco(gt, predictions, 3000)
        rows, _ = _metric_rows(resample_evaluator(evaluator, [0, 0]))
        self.assertAlmostEqual(next(r['map50_95'] for r in rows if r['size'] == 'all'), 1.)
        rows, _ = _metric_rows(resample_evaluator(evaluator, [1, 1]))
        self.assertEqual(next(r['map50_95'] for r in rows if r['size'] == 'all'), 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
