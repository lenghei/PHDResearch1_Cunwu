"""Offline acceptance checks for CV261102 CSV export without GPU or dataset."""
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[2]
EXPORT_PATH = PUBLIC / 'experiments/cv261102/export.py'
VERIFY_PATH = PUBLIC / 'experiments/cv261102/verify.py'
RUN_PATH = PUBLIC / 'experiments/cv261102/run.py'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


exporter = module('cv261102_export', EXPORT_PATH)
verify = module('cv261102_verify', VERIFY_PATH)


def sha(data: bytes):
    return hashlib.sha256(data).hexdigest()


class CsvExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / 'cv261102_yolov8s_fullval_test_00000000'
        self.run.mkdir()
        data = {'per_area': [
            {'size': 'all', 'map50': 0.5644934038564043,
             'map50_95': 0.3491918838837957, 'ar50': 0.9, 'ar50_95': 0.5638779985041216,
             'gt_count': 38593, 'evaluated_categories': 10, 'image_count': 548, 'max_det': 3000,
             'area_reference': 'original_image_pixels'},
            {'size': 'tiny', 'map50': 0.35, 'map50_95': 0.15, 'ar50': 0.7,
             'ar50_95': 0.3, 'gt_count': 12000, 'evaluated_categories': 10,
             'image_count': 548, 'max_det': 3000, 'area_reference': 'original_image_pixels'}],
            'per_class': [{'size': 'all', 'map50': 0.3, 'map50_95': 0.2,
                           'ar50': 0.7, 'ar50_95': 0.4, 'gt_count': 500,
                           'evaluated_categories': 1, 'image_count': 548, 'max_det': 3000,
                           'area_reference': 'original_image_pixels', 'category_id': 1,
                           'category_name': 'pedestrian'}]}
        metrics = json.dumps(data).encode()
        (self.run/'metrics.json').write_bytes(metrics)
        summary = {'RunID': self.run.name, 'ExperimentID': 'cv261102', 'Status': 'PASS',
                   'checkpoint_id': 'ck05', 'images': 548, 'dataset': 'VisDrone2019-DET-val',
                   'scope': 'FULL_DEVELOPMENT_VALIDATION', 'protocol': exporter.PROTOCOL,
                   'metrics_SHA256': sha(metrics), 'checkpoint_SHA256': 'c'*64,
                   'dataset_ZIP_SHA256': 'd'*64, 'network_seconds': 20.4,
                   'evaluation_seconds': 101.1, 'total_seconds': 179.0,
                   'aggregate': data['per_area'][0]}
        (self.run/'summary.json').write_text(json.dumps(summary), encoding='utf8')

    def test_full_csv_outputs_and_manifests(self):
        result = exporter.export_csv(self.run)
        target = Path(result['directory'])
        self.assertEqual(len(result['csv_files']), 6)
        with (target/'cv261102_metrics_overall.csv').open(encoding='utf-8-sig', newline='') as f:
            row = next(csv.DictReader(f))
        self.assertAlmostEqual(float(row['AP50']), 0.5644934038564043)
        self.assertEqual(row['csv_role'], exporter.ROLE)
        self.assertEqual(row['source_run_id'], self.run.name)
        self.assertEqual(row['value_unit'], 'fraction_0_to_1')
        with (target/'cv261102_training_epoch_status.csv').open(encoding='utf-8-sig') as f:
            e = next(csv.DictReader(f))
        self.assertEqual(e['training_epoch_status'], 'MISSING_TRAINING_EPOCH_TIMING')
        self.assertEqual(e['mean_epoch_seconds'], '')
        self.assertEqual(result['reference_status'], 'MISSING_CSV')
        manifest = json.loads(Path(result['manifest']).read_text())
        for filename, digest in manifest['csv_files_SHA256'].items():
            self.assertEqual(exporter.sha256(target/filename), digest)
        self.assertFalse(any('epoch' in k.lower() for k in (target/'cv261102_timing_evaluation.csv').read_text()))

    def test_reference_export_rejected(self):
        result = exporter.export_csv(self.run)
        csv_file = Path(result['directory'])/'cv261102_metrics_overall.csv'
        with self.assertRaisesRegex(ValueError, 'not an independent historical reference'):
            verify.reference_values(csv_file)

    def test_metric_sha_failure_blocks(self):
        (self.run/'metrics.json').write_text('bad', encoding='utf8')
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            exporter.export_csv(self.run)

    def test_missing_epoch_reference_never_invented(self):
        result = exporter.export_csv(self.run, audit_report={'reference_CSV': {'status': 'MATCHED',
            'values': {'AP50': 0.56449, 'AP50_95': 0.34919}, 'source_SHA256': 'f'*64},
            'training_epoch_timing': {'status': 'RECORDED_FROM_SUPPLIED_EPOCH_CSV',
                                     'rows': 100, 'mean_seconds': 18.2, 'min_seconds': 17,
                                     'max_seconds': 22, 'source_SHA256': 'e'*64}})
        target = Path(result['directory'])
        with (target/'cv261102_reference_comparison.csv').open(encoding='utf-8-sig') as f:
            self.assertEqual(next(csv.DictReader(f))['comparison_status'], 'MATCHED')
        with (target/'cv261102_training_epoch_status.csv').open(encoding='utf-8-sig') as f:
            self.assertEqual(next(csv.DictReader(f))['mean_epoch_seconds'], '18.2')

    def test_command_export_only(self):
        r = subprocess.run([sys.executable, '-B', str(RUN_PATH), '--export-run', str(self.run)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('no GPU', r.stdout)
        self.assertTrue((self.run/'cv261102_reports/cv261102_metrics_overall.csv').is_file())


if __name__ == '__main__':
    unittest.main()
