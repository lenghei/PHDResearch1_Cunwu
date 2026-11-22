"""Verify published snapshot tables and the VM48 cloud page on CPU."""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

PUBLIC = Path(__file__).resolve().parents[2]
TOOLS = PUBLIC / 'code/tools'
sys.path.insert(0, str(TOOLS))

from dashboard_autodiscovery import CLASS_TABLE, DataError, load_experiment, load_snapshot, public_runs
from dashboard_data import DashboardStore

STUDY = 'vm48_formal_rq4'
RUN = 'motion_targeted_seed45_execution_20261118T210143_5ada2476'
OLD_RUN = 'cv261102_full_20261031T000013_d0be8587'


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.public = Path(temp.name)
        self.path = self.public / 'reports/results' / STUDY / RUN
        shutil.copytree(PUBLIC / 'reports/results' / STUDY / RUN, self.path)
        self.store = DashboardStore(mode='cloud', public=self.public)

    def load(self):
        return load_snapshot(self.store, STUDY, RUN)

    def declare(self, name, data):
        if '/' not in name:
            (self.path / name).write_bytes(data)
        path = self.path / 'manifest.json'
        manifest = json.loads(path.read_bytes())
        manifest['files'][name] = hashlib.sha256(data).hexdigest()
        if name == 'snapshot.json':
            manifest['SHA256'] = manifest['files'][name]
        path.write_bytes(json.dumps(manifest).encode())

    def test_snapshot_sha_and_public_figures(self):
        data = self.load()
        self.assertEqual(data['RunID'], RUN)
        self.assertEqual(data['scope'], 'DEVELOPMENT_ONLY_SINGLE_SEED')
        self.assertEqual(set(data['figures']), {'gap_tradeoffs.png', 'Figure_C_class_degradation_heatmap.png'})
        self.assertTrue(all(value.startswith(b'\x89PNG') for value in data['figures'].values()))
        p = self.path / 'snapshot.json'
        p.write_bytes(p.read_bytes() + b' ')
        with self.assertRaisesRegex(DataError, 'snapshot checksum mismatch'):
            self.load()

    def test_snapshot_git_line_ending_conversion(self):
        path = self.path / 'snapshot.json'
        original = path.read_bytes()
        normalized = original.replace(b'\r\n', b'\n')
        alternate = normalized if original != normalized else original.replace(b'\n', b'\r\n')
        self.assertNotEqual(original, alternate)
        path.write_bytes(alternate)
        self.assertEqual(self.load()['RunID'], RUN)

    def test_snapshot_identity_even_with_valid_hash(self):
        bundle = json.loads((self.path / 'snapshot.json').read_bytes())
        bundle['RunID'] = 'other_run'
        self.declare('snapshot.json', json.dumps(bundle).encode())
        with self.assertRaisesRegex(DataError, 'snapshot identity mismatch'):
            self.load()

    def test_manifest_identity(self):
        p = self.path / 'manifest.json'
        manifest = json.loads(p.read_bytes())
        manifest['ExperimentID'] = 'other_study'
        p.write_bytes(json.dumps(manifest).encode())
        with self.assertRaisesRegex(DataError, 'manifest identity mismatch'):
            self.load()

    def test_registered_csv_is_discovered(self):
        self.declare('other_summary.csv', b'label,ap\nother,0.25\n')
        self.assertEqual(self.load()['tables']['other_summary.csv'], [{'label': 'other', 'ap': 0.25}])

    def test_csv_sha_mismatch(self):
        p = self.path / CLASS_TABLE
        p.write_bytes(p.read_bytes() + b' ')
        with self.assertRaisesRegex(DataError, 'CSV checksum mismatch.*per_class'):
            self.load()

    def test_40_unique_class_rows(self):
        rows = self.load()['tables'][CLASS_TABLE]
        self.assertEqual(len(rows), 40)
        conditions = ['Normal', 'Motion_k21', 'Haze_dense', 'Motion_Haze']
        self.assertEqual({(row['condition'], row['category_id']) for row in rows},
                         {(name, cid) for name in conditions for cid in range(1, 11)})
        motion = sorted((r for r in rows if r['condition'] == 'Motion_k21'),
                        key=lambda r: r['delta_AP50_95_pp'], reverse=True)
        self.assertEqual([r['category_name'] for r in motion[:3]], ['car', 'van', 'bus'])
        self.assertEqual([round(r['delta_AP50_95_pp'], 3) for r in motion[:3]], [15.806, 6.375, 5.86])

    def test_summary_sizes_and_seed(self):
        data = self.load()
        for key, value in (('images', 548), ('epochs', 10), ('training_images', 6471),
                           ('cells', 8), ('image_evaluations', 4384), ('seed', 45)):
            self.assertEqual(data[key], value)
        self.assertEqual(data['paired']['arm_names'], ('CONTROL', 'MOTION_TARGETED'))

    def test_all_paired_values_preserve_snapshot_scale_and_gap(self):
        data = self.load()
        paired = data['paired']
        bundle = json.loads((self.path / 'snapshot.json').read_bytes())
        source = bundle['files']['summary.json']['paired_conditions']
        self.assertEqual(len(paired['conditions']), 4)
        for current, original in zip(paired['conditions'], source):
            self.assertEqual(current['condition'], original['condition'])
            for arm, values in current['arms'].items():
                for key in ('ap', 'tiny', 'ms', 'state'):
                    self.assertEqual(values[key], original['arms'][arm][key])
            for key, value in current['delta'].items():
                self.assertEqual(value, original['delta'][key]['value'])
        motion = paired['conditions'][1]
        a, b = motion['arms']['CONTROL'], motion['arms']['MOTION_TARGETED']
        self.assertEqual((round(a['ap'], 3), round(b['ap'], 3)), (2.313, 6.563))
        self.assertEqual((round(a['tiny'], 3), round(b['tiny'], 3)), (0.109, 0.581))
        self.assertEqual((a['state'], b['state'], paired['closures']), ('FAIL', 'FAIL', 0))

    def test_missing_declared_csv(self):
        (self.path / CLASS_TABLE).unlink()
        with self.assertRaisesRegex(DataError, 'Missing published asset.*per_class'):
            self.load()

    def test_private_and_unregistered_files_are_excluded(self):
        self.declare('extra.csv', b'name,ap,private_path,credential,comment\n'
                     b'car,0.25,E:/private,never_display,/Users/member/private\n')
        (self.path / 'unregistered.csv').write_text('name\nnever_display\n', encoding='utf-8')
        tables = self.load()['tables']
        self.assertNotIn('unregistered.csv', tables)
        self.assertEqual(tables['extra.csv'], [{'name': 'car', 'ap': 0.25}])

    def test_manifest_cannot_read_another_run(self):
        self.declare('../private.csv', b'key\nprivate\n')
        with self.assertRaises(DataError):
            self.load()

    def test_rejects_invalid_class_types_and_oversized_files(self):
        content = (self.path / CLASS_TABLE).read_bytes()
        header, row, *remaining = content.splitlines()
        values = row.split(b',')
        col = header.split(b',').index(b'CONTROL_AP50_95_pct')
        values[col] = b'bad-number'
        self.declare(CLASS_TABLE, b'\n'.join([header, b','.join(values), *remaining]))
        with self.assertRaisesRegex(DataError, 'Invalid per-class CSV metric'):
            self.load()
        self.declare(CLASS_TABLE, b'x' * 1_000_001)
        with self.assertRaisesRegex(DataError, 'CSV exceeds size limit'):
            self.load()

    def test_unverified_plot_is_rejected(self):
        path = self.path / 'gap_tradeoffs.png'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(DataError, 'figure checksum'):
            self.load()


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.store = DashboardStore(mode='cloud', public=PUBLIC)

    def test_existing_legacy_schema(self):
        data = load_experiment(self.store, 'cv261102', OLD_RUN)
        self.assertEqual((data['images'], data['epochs']), (548, 100))
        self.assertIn('metrics_by_class.csv', data['tables'])

    def test_published_snapshots_with_no_csv_have_saved_tables(self):
        studies = {'vm39_multi_seed_confirmation': ('differences_pp', 24),
                   'vm43_external_pilot': ('metrics', 8),
                   'vm44_tiny_error_variability_decision': ('tiny_recall_diagnostic', 2),
                   'vm45_motion_haze_gap_learning': ('factorial', 2),
                   'vm47_gaussian_jpeg_factorial': ('factorial', 2)}
        for study, (name, count) in studies.items():
            with self.subTest(study=study):
                run = public_runs(self.store, study)[0]['RunID']
                data = load_snapshot(self.store, study, run)
                self.assertEqual(len(data['summary_tables'][name]), count)
                self.assertEqual(data['tables'], {})

    def test_older_vm48_run_is_available_without_class_csv(self):
        run = 'laptop_transfer_pilot_execution_20261118T100506_d3fa1688'
        data = load_snapshot(self.store, STUDY, run)
        self.assertEqual((data['images'], data['image_evaluations']), (548, 13152))
        self.assertEqual(len(data['summary_tables']['mean_differences_pp']), 4)
        self.assertNotIn(CLASS_TABLE, data['tables'])


@unittest.skipUnless(all(find_spec(name) for name in ('streamlit', 'pandas', 'plotly')), 'dashboard dependencies missing')
class CloudPageTests(unittest.TestCase):
    def test_vm48_cloud_renders_charts_filters_and_download(self):
        from streamlit.testing.v1 import AppTest

        with patch.dict(os.environ, {'UAVGAP_DASHBOARD_MODE': 'cloud', 'UAVGAP_RUNS_ROOT': 'unavailable_private_runs'}):
            app = AppTest.from_file(str(TOOLS / 'dashboard.py'), default_timeout=20).run()
            next(box for box in app.selectbox if box.label == 'Experiment').select('VM48 · formal rq4').run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            metrics = {x.label: x.value for x in app.metric}
            for key, value in (('Images', '548 per condition'), ('Epochs', '10 per arm'),
                               ('Training arms', '2'), ('Total evaluations', '4384'), ('Seed', '45'),
                               ('Status', 'MATCHED PILOT'), ('Confirmed Gap closures', '0')):
                self.assertEqual(metrics[key], value)
            captions = ' '.join(x.value for x in app.caption)
            self.assertIn('2.313% to 6.563% (+4.250 pp)', captions)
            self.assertIn('0.109% to 0.581% | Gap: FAIL to FAIL', captions)
            self.assertIn('MATCHED_PILOT_COMPLETED_WITH_LIMITATIONS', captions)
            self.assertEqual(len(app.get('plotly_chart')), 3)
            self.assertEqual(len(app.get('image')), 2)
            self.assertIn('Download all class results', [button.proto.label for button in app.get('download_button')])
            next(box for box in app.selectbox if box.label == 'Degradation condition').select('All conditions').run()
            result = next(table.value for table in app.dataframe if 'category_name' in table.value)
            self.assertEqual(len(result), 40)
            next(box for box in app.multiselect if box.label == 'Classes').set_value(['car']).run()
            result = next(table.value for table in app.dataframe if 'category_name' in table.value)
            self.assertEqual(len(result), 4)
            self.assertEqual(set(result['category_name']), {'car'})

    def test_saved_measurements_are_visible_in_another_study(self):
        from streamlit.testing.v1 import AppTest

        with patch.dict(os.environ, {'UAVGAP_DASHBOARD_MODE': 'cloud'}):
            app = AppTest.from_file(str(TOOLS / 'dashboard.py'), default_timeout=20).run()
            next(box for box in app.selectbox if box.label == 'Experiment').select('VM43 · external pilot').run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(len(app.dataframe[0].value), 8)
            self.assertIn('ap50_95', app.dataframe[0].value)


if __name__ == '__main__':
    unittest.main()
