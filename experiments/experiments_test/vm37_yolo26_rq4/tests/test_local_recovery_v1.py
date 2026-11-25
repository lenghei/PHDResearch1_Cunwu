"""New CPU checks for the resolution adapter and stop diagnostics."""
from datetime import datetime, timedelta, timezone
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_recovery_v1 as recovery


class RecoveryTests(unittest.TestCase):
    def pair(self):
        value = {key: [] for key in ('architecture', 'COCO_source_SHA256', 'train', 'development',
                                    'train_labels', 'val_labels', 'expected', 'execution_epochs', 'selection')}
        value.update(initialization={'sha256': 'same'}, recovery_protocol=recovery.candidate_protocol('B'),
                     arm='CONTROL', targeted_recipe=None, args={'imgsz': 960, 'seed': 42, 'data': 'control.yaml'})
        target = copy.deepcopy(value)
        target.update(arm='ALL_FIVE', targeted_recipe=recovery.baseline.load(recovery.baseline.bundle() / 'assets/TARGETED_POLICY.json'))
        target['args']['data'] = 'target.yaml'
        return value, target

    def test_matched_pair_allows_only_policy_and_output_paths(self):
        recovery.verify_pair(*self.pair())

    def test_mismatched_initialization_population_or_seed_rejected(self):
        for key, change in (('initialization', {'sha256': 'different'}), ('train', ['other_image']),
                            ('args', {'imgsz': 960, 'seed': 43})):
            control, target = self.pair()
            target[key] = change
            with self.assertRaises(ValueError):
                recovery.verify_pair(control, target)

    def test_changed_all_five_policy_rejected(self):
        control, target = self.pair()
        target['targeted_recipe'] = {'families': {}}
        with self.assertRaises(ValueError):
            recovery.verify_pair(control, target)

    def test_candidates_preserve_shared_protocol(self):
        a, b = (recovery.candidate_protocol(key) for key in ('A', 'B'))
        self.assertEqual(a['shared_training_args'] | {'imgsz': 960}, b['shared_training_args'])
        for spec in (a, b):
            recovery.verify_candidate(spec)
            self.assertEqual(spec['epochs'], 10)
            self.assertEqual(spec['inference_resolutions'], [640, 1280])
            self.assertFalse(spec['GPU_execution_authorized'])
        self.assertEqual(a['execution_epochs'], 1)
        self.assertEqual(b['execution_epochs'], 10)

    def test_parameter_changes_rejected(self):
        for key, value in (('batch', 1), ('amp', True), ('workers', 1), ('lr0', .001),
                           ('close_mosaic', 0), ('imgsz', 1280)):
            spec = recovery.candidate_protocol('B')
            spec['shared_training_args'][key] = value
            with self.assertRaises(ValueError):
                recovery.verify_candidate(spec)

    def test_epoch_horizon_and_arm_changes_rejected(self):
        for key, value in (('epochs', 100), ('schedule_horizon_epochs', 100),
                           ('arms', ['CONTROL']), ('execution_epochs', 1)):
            spec = recovery.candidate_protocol('B')
            spec[key] = value
            with self.assertRaises(ValueError):
                recovery.verify_candidate(spec)

    def test_resolution_does_not_change_exposure_schedule(self):
        for option, batches, steps, pending in (('A', 3236, 784, 1), ('B', 32360, 1816, 17)):
            counters = recovery.candidate_protocol(option)['planned_counters_per_arm']
            self.assertEqual((counters['batches'], counters['updates'], counters['pending_microbatches']),
                             (batches, steps, pending))
            self.assertEqual(counters['warmup_batches'], 9708)

    def test_thermal_reason_precedes_missing_worker_file(self):
        with patch.object(recovery.baseline, 'load', side_effect=AssertionError('Must not read worker output')):
            result = recovery.stage_diagnosis(Path('absent'), {'stop_reason': 'TEMPERATURE_HARD_STOP', 'returncode': 15})
        self.assertEqual(result['primary_reason'], 'TEMPERATURE_HARD_STOP')

    def test_thermal_reason_precedes_even_stale_success(self):
        with patch.object(recovery.baseline, 'load', side_effect=AssertionError('Must not read worker output')):
            result = recovery.stage_diagnosis(Path('absent'), {'stop_reason': 'SUSTAINED_TEMPERATURE_STOP', 'returncode': 0})
        self.assertEqual(result['status'], 'SUPERVISOR_STOPPED')

    def test_worker_exit_precedes_missing_output(self):
        with patch.object(recovery.baseline, 'load', side_effect=AssertionError('Must not read worker output')):
            result = recovery.stage_diagnosis(Path('absent'), {'stop_reason': None, 'returncode': 7})
        self.assertEqual(result['primary_reason'], 'WORKER_EXIT')

    def test_missing_output_after_normal_exit_is_incomplete(self):
        with tempfile.TemporaryDirectory() as name:
            result = recovery.stage_diagnosis(Path(name), {'stop_reason': None, 'returncode': 0})
        self.assertEqual(result['primary_reason'], 'MISSING_WORKER_RESULT')

    def test_corrupt_output_and_failed_status_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'worker_result.json'
            path.write_text('{')
            self.assertEqual(recovery.stage_diagnosis(Path(name), {'returncode': 0})['primary_reason'], 'INVALID_WORKER_RESULT')
            path.write_text(json.dumps({'status': 'FAILED'}))
            self.assertEqual(recovery.stage_diagnosis(Path(name), {'returncode': 0})['status'], 'INCOMPLETE')

    def test_success_requires_explicit_worker_success(self):
        with tempfile.TemporaryDirectory() as name:
            (Path(name) / 'worker_result.json').write_text(json.dumps({'status': 'TRAINING_COMPLETED'}))
            self.assertEqual(recovery.stage_diagnosis(Path(name), {'returncode': 0})['status'], 'STAGE_COMPLETED')

    def test_no_fresh_physical_clearance_from_generic_reply(self):
        self.assertFalse(recovery.thermal_clearance({'author_reply': 'Rechecked; details to follow'},
                                                  '2026-11-13T22:11:53.602370+00:00'))

    def test_old_confirmation_rejected(self):
        now = datetime(2026, 11, 14, tzinfo=timezone.utc)
        record = {'checked_UTC': (now - timedelta(days=2)).isoformat(),
                  'hard_ventilated_surface': True, 'unobstructed_intake_exhaust': True,
                  'fans_normal': True, 'reliable_AC_adapter': True, 'no_visible_airflow_problem': True,
                  'OEM_mode': 'Recorded by user', 'room_observation': 'Temperature unknown; ventilation checked'}
        self.assertFalse(recovery.thermal_clearance(record, '2026-11-13T22:11:53+00:00', now))
        record['checked_UTC'] = (now - timedelta(minutes=1)).isoformat()
        self.assertTrue(recovery.thermal_clearance(record, '2026-11-13T22:11:53+00:00', now))

    def test_budget_does_not_waive_thermal_admission(self):
        result = recovery.resource_admission({'remaining_seconds': 126999.4}, {'A_CONTROL': 3600, 'A_ALL_FIVE': 3600})
        self.assertEqual(result['status'], 'BLOCKED')
        self.assertIn('PHYSICAL_CLEARANCE_PENDING', result['reasons'])

    def test_checks_cannot_be_added_free_to_full_pilot_caps(self):
        caps = {'A_CONTROL': 3600, 'A_ALL_FIVE': 3600, 'CONTROL': 39000, 'ALL_FIVE': 68000, 'evaluation': 15000}
        result = recovery.resource_admission({'remaining_seconds': 126999.4}, caps, clearance=True, measured=True)
        self.assertIn('CAPS_EXCEED_REMAINING_AUTHORIZATION', result['reasons'])

    def test_expired_or_unknown_caps_rejected(self):
        for caps in ({}, {'training': None}, {'training': float('nan')}):
            result = recovery.resource_admission({'remaining_seconds': 0}, caps, True, True)
            self.assertIn('STAGE_CAPS_NOT_FROZEN', result['reasons'])
            self.assertIn('AUTHORIZATION_EXPIRED_OR_EXHAUSTED', result['reasons'])

    def test_cpu_tensor_shape_precision_and_health(self):
        import torch
        for size in (960, 1280):
            recovery.assert_tensor({'img': torch.zeros(2, 3, size, size)}, size)
        for tensor in (torch.zeros(2, 3, 640, 640), torch.zeros(2, 3, 960, 960, dtype=torch.float16),
                       torch.full((1, 3, 960, 960), float('nan'))):
            with self.assertRaises(ValueError):
                recovery.assert_tensor({'img': tensor}, 960)


if __name__ == '__main__':
    unittest.main()
