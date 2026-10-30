"""Dashboard tests; run with the lightweight dashboard environment."""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PUBLIC = Path(__file__).resolve().parents[2]


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Use the dashboard environment')
class DashboardTests(unittest.TestCase):
    def render(self, app_file):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(app_file), default_timeout=45).run()

    def test_local_studies(self):
        with patch.dict(os.environ, {'UAVGAP_DASHBOARD_MODE': 'local'}):
            app = self.render(PUBLIC / 'code/tools/dashboard.py')
            self.assertFalse(app.exception)
            for study in ['VM1 · FlamaPy–YOLO26s validation', 'VM03 · Historical YOLOv8s baseline']:
                app.sidebar.selectbox[0].select(study).run()
                self.assertFalse(app.exception)

    def test_cloud_copy_without_private_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            clone = Path(temporary) / 'repo'
            shutil.copytree(PUBLIC, clone, ignore=shutil.ignore_patterns('config.local.yaml', '__pycache__', '*.pyc'))
            self.assertFalse((clone.parent / 'datasets').exists())
            old_module = sys.modules.pop('dashboard_data', None)
            previous = Path.cwd()
            sys.path.insert(0, str(clone / 'code/tools'))
            forbidden = [str(PUBLIC.parent / p).lower() for p in ['datasets', 'models', '.venv', '.venv_flamapy']]
            active = [False]

            def guard(event, args):
                if active[0] and event in ['open', 'os.listdir', 'os.scandir'] and args and isinstance(args[0], (str, bytes, os.PathLike)):
                    path = str(Path(os.fsdecode(args[0])).absolute()).lower()
                    if any(path == p or path.startswith(p + os.sep) for p in forbidden):
                        raise AssertionError('Cloud accessed private workspace: ' + path)

            sys.addaudithook(guard)
            try:
                os.chdir(clone)
                active[0] = True
                with patch.dict(os.environ, {'UAVGAP_DASHBOARD_MODE': 'cloud', 'UAVGAP_RUNS_ROOT': str(clone / 'absent')}):
                    app = self.render(clone / 'code/tools/dashboard.py')
                    self.assertFalse(app.exception)
                    self.assertGreaterEqual(len(app.get('plotly_chart')), 4)
                    app.sidebar.selectbox[0].select('VM1 · FlamaPy–YOLO26s validation').run()
                    self.assertFalse(app.exception)
                    app.sidebar.selectbox[0].select('VM2 · Degradation interactions').run()
                    bundle = clone / 'reports/results/vm2_summary.json'
                    payload = bundle.read_bytes()
                    bundle.unlink()
                    app.run()
                    self.assertFalse(app.exception)
                    self.assertTrue(any('missing' in e.value for e in app.error))
                    bundle.write_bytes(payload + b' ')
                    app.run()
                    self.assertFalse(app.exception)
                    self.assertTrue(any('checksum' in e.value for e in app.error))
                    bundle.write_bytes(payload)
                    manifest = clone / 'reports/results/dashboard_manifest.json'
                    manifest.unlink()
                    app.run()
                    self.assertFalse(app.exception)
                    self.assertTrue(app.info)
                    self.assertFalse(any(m in sys.modules for m in ['torch', 'ultralytics', 'cv2']))
            finally:
                active[0] = False
                os.chdir(previous)
                sys.path.remove(str(clone / 'code/tools'))
                sys.modules.pop('dashboard_data', None)
                if old_module is not None:
                    sys.modules['dashboard_data'] = old_module

    def test_cloud_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            sys.path.insert(0, str(PUBLIC / 'code/tools'))
            try:
                from dashboard_data import DashboardStore, DataError
                store = DashboardStore('cloud', public=Path(temporary))
                with self.assertRaises(DataError):
                    store.child(store.root, '../configuration.json')
                with self.assertRaises(DataError):
                    store.child(store.root, '/private/file.json')
            finally:
                sys.path.remove(str(PUBLIC / 'code/tools'))


if __name__ == '__main__':
    unittest.main()
