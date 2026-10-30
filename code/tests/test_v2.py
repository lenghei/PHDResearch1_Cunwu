import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from main.bridge import numeric_parameters
from main.degradation import degrade, field_for
from main.gap_analysis import empirical_gap, finite_difference, static_gap, verify_measurement
from main.visualization import update_run_index, build_run_index


class V2Tests(unittest.TestCase):
    def test_vm2_browser_default_and_no_open(self):
        import runpy
        import sys
        entry = Path(__file__).resolve().parents[2] / 'experiments/vm2_degradation_gap_analysis/run.py'
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            for options in [[], ['--no-open']]:
                with patch.object(sys, 'argv', [str(entry), *options]), \
                        patch('main.gap_analysis.replay', return_value=run), \
                        patch('webbrowser.open', return_value=True) as browser, \
                        contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(entry), run_name='__main__')
                    if options:
                        browser.assert_not_called()
                    else:
                        browser.assert_called_once_with((run / 'index.html').as_uri())

    def test_result_index_rejects_failed_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            group = Path(folder)
            first, failed = group / 'first', group / 'failed'
            for run, status in [(first, 'PASS'), (failed, 'FAILED')]:
                run.mkdir()
                (run / 'index.html').write_text('Result', encoding='utf-8')
                (run / 'run.json').write_text(json.dumps({'RunID': run.name, 'ExperimentID': 'test',
                                                         'mode': 'demo', 'status': status}), encoding='utf-8')
            update_run_index(first)
            with self.assertRaisesRegex(ValueError, 'Only completed runs'):
                update_run_index(failed)
            actual = json.loads((group / 'latest.json').read_text(encoding='utf-8'))
            self.assertEqual(actual['modes']['demo']['RunID'], 'first')

    def test_run_history_order_and_latest_success(self):
        with tempfile.TemporaryDirectory() as folder:
            group = Path(folder)
            identities = [('test_demo_20260101T010000_old', 'demo', 'PASS'),
                          ('test_demo_20260103T010000_failed', 'demo', 'FAILED'),
                          ('test_check_20260102T010000_check', 'check', 'PASS')]
            for name, mode, status in identities:
                run = group / name
                run.mkdir()
                (run / 'run.json').write_text(json.dumps({'ExperimentID': 'test', 'mode': mode, 'status': status}), encoding='utf-8')
                if status == 'PASS':
                    (run / 'index.html').write_text('Result', encoding='utf-8')
            old_metadata = group / identities[0][0] / 'run.json'
            os.utime(old_metadata, (2000000000, 2000000000))
            before = {p: p.read_bytes() for p in group.rglob('*') if p.is_file()}
            build_run_index(group, 'test')
            actual = json.loads((group / 'latest.json').read_text(encoding='utf-8'))
            self.assertEqual([r['RunID'] for r in actual['history']], [identities[1][0], identities[2][0], identities[0][0]])
            self.assertEqual(actual['modes']['demo']['RunID'], identities[0][0])
            self.assertEqual(actual['history'][0]['page_kind'], 'Metadata')
            for path, contents in before.items():
                self.assertEqual(path.read_bytes(), contents)

    def test_history_keeps_legacy_and_unreadable_records(self):
        with tempfile.TemporaryDirectory() as folder:
            group = Path(folder)
            legacy = group / 'git03_vm03_20260101T010000_old'
            legacy.mkdir()
            (legacy / 'summary.json').write_text(json.dumps({'ExperimentID': 'test', 'Status': 'PASS',
                                                           'scope': 'FULL_DEVELOPMENT_VALIDATION'}), encoding='utf-8')
            bad = group / 'test_20260102T010000_bad'
            bad.mkdir()
            (bad / 'run.json').write_text('{', encoding='utf-8')
            build_run_index(group, 'test')
            actual = json.loads((group / 'latest.json').read_text(encoding='utf-8'))
            self.assertEqual(actual['history'][0]['status'], 'UNREADABLE')
            self.assertEqual(actual['history'][1]['mode'], 'full-validation')
            self.assertEqual(actual['modes']['full-validation']['RunID'], legacy.name)

    def test_degradation_contract(self):
        image = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
        condition = [0, 1, 2]
        first, meta = degrade(image, condition, field_for(image, 'fixed', 42))
        second, _ = degrade(image, condition, field_for(image, 'fixed', 42))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, image.shape)
        for key, value in numeric_parameters(condition).items():
            self.assertEqual(meta[key], value)

    def test_four_gap_states(self):
        self.assertEqual(empirical_gap(.4, .01), 'NOT_EVALUATED')
        self.assertEqual(empirical_gap(.4, .01, .3, .02), 'PASS')
        self.assertEqual(empirical_gap(.2, None, .3, .02), 'FAIL')
        self.assertEqual(empirical_gap(.4, None, .3, .02), 'UNCERTAIN')
        self.assertEqual(static_gap({'blur': 2}, []), 'NOT_EVALUATED')
        rule = {'factor': 'blur', 'covered_levels': [0], 'exhaustive': False}
        self.assertEqual(static_gap({'blur': 2}, [rule]), 'UNCERTAIN')
        rule['exhaustive'] = True
        self.assertEqual(static_gap({'blur': 2}, [rule]), 'FAIL')

    def test_required_corners(self):
        rows = [{'r': r, 'b': b, 'AP50_95': ap} for r, b, ap in [(640, 0, .2), (1280, 0, .5), (640, 2, .1), (1280, 2, .2)]]
        value = finite_difference(rows, ['r', 'b'], [[640, 1280], [0, 2]])
        self.assertAlmostEqual(value['estimate'], -.2)
        self.assertEqual(finite_difference(rows[:-1], ['r', 'b'], [[640, 1280], [0, 2]])['status'], 'NOT_EVALUATED')

    def test_rejects_inconsistent_metric_extraction(self):
        keys = ['AP50', 'AP50_95', 'AR', 'Tiny_AP', 'Small_AP', 'Medium_AP', 'Large_AP']
        row = {key: .25 for key in keys}
        measured = {'per_area': [{'size': area, 'map50': .25, 'map50_95': .25, 'ar50_95': .25}
                                  for area in ['all', 'tiny', 'small', 'medium', 'large']]}
        verify_measurement(row, measured)
        measured['per_area'][1]['map50_95'] = .3
        with self.assertRaisesRegex(ValueError, 'Tiny_AP'):
            verify_measurement(row, measured)

    def test_vm1_full_paired_summary_from_saved_metrics(self):
        from main.config import save
        from main.pipeline import save_paired_comparison
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            for name, value in [('CLEAN', .5), ('DEGRADED', .3)]:
                condition = run / name
                condition.mkdir()
                save(condition / 'metrics.json', {'per_area': [
                    {'size': 'all', 'map50': value, 'map50_95': value / 2,
                     'ar50_95': value / 3}]})
                save(condition / 'summary.json', {'checkpoint_id': 'ck01'})
            sources = [{'configuration': {'name': name, 'configuration_id': name.lower()}}
                       for name in ['CLEAN', 'DEGRADED']]
            save_paired_comparison(run, sources, 548)
            data = json.loads((run / 'paired_comparison.json').read_text(encoding='utf-8'))
            self.assertEqual(data['images_per_condition'], 548)
            self.assertEqual(data['checkpoint_id'], 'ck01')
            self.assertAlmostEqual(data['rows'][0]['delta_degraded_minus_clean'], -.2)
            self.assertTrue((run / 'paired_comparison.csv').is_file())

    def test_entry_routing_without_forward(self):
        path = Path(__file__).resolve().parents[2] / 'experiments/vm1_flamapy_yolo26s_validation/run.py'
        spec = importlib.util.spec_from_file_location('v2_entry_test', path)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        for argv, mode in [([], 'full-paired'), (['--full-paired'], 'full-paired'), (['--check'], 'check'), (['--demo'], 'demo'), (['--full-clean'], 'full-clean'), (['--regenerate', 'saved-run'], 'regenerate')]:
            with patch.dict(os.environ), patch('main.pipeline.execute', side_effect=RuntimeError('ROUTE_ONLY')) as call:
                with self.assertRaisesRegex(RuntimeError, 'ROUTE_ONLY'):
                    entry.main(argv)
                self.assertEqual(call.call_args.args[1], mode)
