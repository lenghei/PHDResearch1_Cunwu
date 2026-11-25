"""CPU admission checks for the separately frozen ten-epoch pilot."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import pilot10 as pilot


class Pilot10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = pilot.study.run_path(sys.argv[-1]) if sys.argv[-1].startswith('vm37_') else None

    def spec(self):
        return pilot.load(pilot.bundle() / 'formal100_protocol.json') | {
            'id': pilot.ID, 'epochs': 10, 'schedule_horizon_epochs': 10,
            'architectures': ['YOLO26s'], 'seeds': [42],
            'planned_counters_per_arm': pilot.study.expected_steps(6471, 10),
            'shared_training_args': pilot.load(pilot.bundle() / 'formal100_protocol.json')['shared_training_args'] | {'close_mosaic': 10}}

    def test_ten_epoch_protocol(self):
        pilot.verify_protocol(self.spec())

    def test_no_formal_horizon_truncation(self):
        value = self.spec()
        value['schedule_horizon_epochs'] = 100
        with self.assertRaises(ValueError):
            pilot.verify_protocol(value)

    def test_no_arm_deletion(self):
        value = self.spec()
        value['arms'] = ['CONTROL']
        with self.assertRaises(ValueError):
            pilot.verify_protocol(value)

    def test_no_precision_or_worker_change(self):
        for key, change in (('amp', True), ('workers', 1), ('close_mosaic', 1)):
            value = self.spec()
            value['shared_training_args'][key] = change
            with self.assertRaises(ValueError):
                pilot.verify_protocol(value)

    def test_expected_partial_batch_and_steps(self):
        value = pilot.study.expected_steps(6471, 10)
        self.assertEqual((value['batches'], value['updates'], value['pending_microbatches']), (32360, 1816, 17))
        self.assertEqual(6471 % 2, 1)

    def test_smoke_has_one_real_update(self):
        self.assertEqual(pilot.study.expected_steps(16, 10, stop_epochs=1)['updates'], 1)

    def test_cumulative_new_budget_and_expiry(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name)
            (p / 'charges').mkdir()
            now = datetime.now(timezone.utc)
            (p / 'authorization.json').write_text(json.dumps({'max_cumulative_GPU_worker_wall_seconds': 129600,
                'absolute_expiry_UTC': (now + timedelta(days=7)).isoformat()}))
            (p / 'charges/first.json').write_text(json.dumps({'charged_seconds': 600}))
            self.assertEqual(pilot.available(p, now=now), 129000)
            self.assertEqual(pilot.available(p, now=now + timedelta(days=8)), 0)

    def test_internal_health_output_scope(self):
        approved = (pilot.ROOT / 'datasets/local_runs').resolve()
        self.assertTrue((approved / 'vm37_yolo26_rq4/future').resolve().is_relative_to(approved))
        self.assertFalse((pilot.ROOT / 'main/experiments/future').resolve().is_relative_to(approved))


if __name__ == '__main__':
    unittest.main()
