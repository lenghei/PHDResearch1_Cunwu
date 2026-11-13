"""CPU checks for the v3 integration and server admission boundaries."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST))
import portable_v3 as v3


class V3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.protocol = {'id': 'SYNTHETIC_TEST_ONLY', 'architectures': ['YOLO26s'], 'seeds': [42],
                         'arms': ['CONTROL', 'ALL_FIVE']}
        for name in ('benchmark26.json', 'formal100_protocol.json'):
            (self.folder / name).write_text(json.dumps(self.protocol))
        (self.folder / 'bundle_manifest.json').write_text('{"files":{}}')
        (self.folder / 'receipt.json').write_text('{}')
        (self.folder / 'cost.json').write_text('{"scope":"SYNTHETIC_UNIT_TEST_NOT_A_MEASUREMENT"}')
        self.auth = {'approved': True, 'authorization_scope': 'NEW_TARGET_SERVER_ONLY',
            'protocol_id': self.protocol['id'], 'bundle_manifest_SHA256': v3.base.digest(self.folder / 'bundle_manifest.json'),
            'input_pipeline': 'optimized', 'arms': [list(row) for row in v3.base.matrix(self.protocol)],
            'cooling_confirmed': True, 'power_confirmed': True, 'machine_admitted': True,
            'CPU_receipt_SHA256': v3.base.digest(self.folder / 'receipt.json'), 'total_seconds': 1000,
            'completed_workers': [], 'absolute_deadline_UTC': (datetime.now(timezone.utc) + timedelta(seconds=900)).isoformat(),
            'purpose': 'TARGET_SERVER_FULL_EPOCH_COST_ONLY', 'per_arm_hard_seconds': 300,
            'resource_admission': {'status': 'PASS', 'ALL_FIVE_GPU_cost_measured': True,
                'required_complete_matrix_seconds': 600, 'cost_evidence_relative': 'cost.json',
                'cost_evidence_SHA256': v3.base.digest(self.folder / 'cost.json'),
                'measured_architectures': ['YOLO26s'], 'evaluation_and_bootstrap_included': True}}

    def tearDown(self):
        self.temp.cleanup()

    def admitted(self, kind='benchmark26', initial=True):
        with patch.object(v3, 'verify_receipt', return_value={'scope': 'SYNTHETIC_TEST'}):
            return v3.authorize(self.folder, kind, self.auth, self.folder / 'receipt.json', initial_admission=initial)

    def test_default_authorization_refused(self):
        with self.assertRaises(PermissionError):
            v3.authorize(self.folder, 'benchmark26', {'approved': False}, self.folder / 'receipt.json')

    def test_historical_local_budget_refused(self):
        del self.auth['authorization_scope']
        with self.assertRaises(PermissionError):
            self.admitted()

    def test_complete_benchmark_pair_admission(self):
        self.assertGreater(self.admitted(), 600)
        self.auth['total_seconds'] = 500
        with self.assertRaises(PermissionError):
            self.admitted()

    def test_later_worker_does_not_require_original_pair_budget_again(self):
        self.auth['completed_workers'] = [{'charged_seconds': 650}]
        self.assertGreater(self.admitted(initial=False), 60)
        with self.assertRaises(PermissionError):
            self.admitted(initial=True)

    def test_expired_authorization_refused(self):
        self.auth['absolute_deadline_UTC'] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        with self.assertRaises(PermissionError):
            self.admitted()

    def test_single_arm_or_mixed_pipeline_refused(self):
        self.auth['arms'] = self.auth['arms'][:1]
        with self.assertRaises(PermissionError):
            self.admitted()
        self.auth['arms'] = [list(row) for row in v3.base.matrix(self.protocol)]
        self.auth['input_pipeline'] = 'unreviewed'
        with self.assertRaises(PermissionError):
            self.admitted()

    def test_benchmark_cap_cannot_be_infinite_or_expanded(self):
        for value in (True, float('inf'), 7201, 0):
            self.auth['per_arm_hard_seconds'] = value
            with self.assertRaises(PermissionError):
                self.admitted()

    def test_formal_requires_all_architecture_costs_and_evaluation(self):
        self.assertGreater(self.admitted('formal100'), 600)
        self.auth['resource_admission']['measured_architectures'] = []
        with self.assertRaises(PermissionError):
            self.admitted('formal100')
        self.auth['resource_admission']['measured_architectures'] = ['YOLO26s']
        self.auth['resource_admission']['evaluation_and_bootstrap_included'] = False
        with self.assertRaises(PermissionError):
            self.admitted('formal100')

    def test_parent_wall_time_and_prior_charges_both_count(self):
        import time
        self.auth['completed_workers'] = [{'charged_seconds': 700}]
        value = v3.remaining(self.auth, time.perf_counter() - 100)
        self.assertGreater(value, 198)
        self.assertLess(value, 201)

    def test_preflight_cannot_spend_the_complete_pair_reserve(self):
        import time
        self.auth['total_seconds'] = 650
        with patch.object(v3, 'verify_receipt', return_value={'scope': 'SYNTHETIC_TEST'}):
            with self.assertRaises(PermissionError):
                v3.execution_admission(self.folder, 'benchmark26', self.auth,
                    self.folder / 'receipt.json', time.perf_counter() - 100)

    def test_source_and_data_output_boundary(self):
        data = self.folder / 'raw'
        data.mkdir()
        with self.assertRaises(ValueError):
            v3.execute(self.folder, data, 'formal100', data / 'runs')
        with self.assertRaises(ValueError):
            v3.execute(self.folder, data, 'formal100', self.folder / 'assets/runs')

    def test_invalid_pipeline_rejected_before_plan_creation(self):
        with self.assertRaises(ValueError):
            v3.make_plan(self.folder, {}, self.protocol, 'YOLO26s', 42, 'CONTROL', self.folder, 'formal100', 'invalid')

    def test_external_run_metadata_can_be_finalized(self):
        folder = self.folder / 'vm37_fixture'
        folder.mkdir()
        (folder / 'run.json').write_text(json.dumps({'RunID': folder.name, 'ExperimentID': 'vm37_yolo26_rq4'}))
        v3.finish_run(folder, 'CPU_FIXTURE_ONLY')
        self.assertEqual(v3.base.load(folder / 'run.json')['status'], 'CPU_FIXTURE_ONLY')
        self.assertTrue((folder / 'SHA256SUMS.txt').exists())

    def test_legacy_writer_is_narrowed_to_the_external_run(self):
        from study import save
        folder = self.folder / 'external_output'
        folder.mkdir()
        old = save.__globals__['ROOT']
        with v3.output_scope(folder):
            save(folder / 'allowed.json', {'scope': 'SYNTHETIC_TEST'}, folder)
            with self.assertRaises(ValueError):
                save(self.folder / 'outside.json', {}, folder)
        self.assertEqual(save.__globals__['ROOT'], old)
        self.assertFalse((self.folder / 'outside.json').exists())

    def test_evaluation_adapter_keeps_the_original_metric_implementation(self):
        import evaluate_v3
        original = evaluate_v3.evaluate.check_worker_authorization
        with patch.object(evaluate_v3, 'check_evaluation_authorization', return_value=None) as check:
            with patch.object(evaluate_v3.evaluate, 'evaluate', return_value='CPU_ADAPTER_FIXTURE') as evaluator:
                self.assertEqual(evaluate_v3.execute(self.folder, self.folder, self.folder, 'formal100'), 'CPU_ADAPTER_FIXTURE')
                evaluator.assert_called_once_with(self.folder, self.folder, self.folder, 'formal100')
                check.assert_called_once()
        self.assertIs(evaluate_v3.evaluate.check_worker_authorization, original)

    def test_benchmark_cannot_enter_research_evaluation(self):
        with self.assertRaises(PermissionError):
            v3.check_evaluation_authorization(self.folder, self.folder, 'benchmark26')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    v3.base.runtime(args.output, gpu=False)
    import torch
    import cv2
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    v3.base.renderer()
    import test_post_cost
    import test_input
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(test_post_cost.PostCostTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(test_input.NativePrefetchTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(V3Tests)])
    with (args.output / 'tests.log').open('x', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    v3.base.save(args.output / 'test_results.json', {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
         'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'GPU_seconds': 0})
    print(json.dumps({'status': 'PASS' if result.wasSuccessful() else 'FAIL', 'tests': result.testsRun}), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
