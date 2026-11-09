from pathlib import Path
import copy
import hashlib
import json
import os
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
PUBLIC=HERE.parents[1]
ROOT=PUBLIC.parent
sys.path.insert(0,str(PUBLIC/'code/tools'))


class SummaryTests(unittest.TestCase):
    def test_summary_tampering_is_rejected(self):
        from dashboard_vm36 import checked_summary
        from dashboard_data import DataError
        value={'schema':'VM36_READ_ONLY_SUMMARY_V1','RunID':'example','new_measured_pairs':0}
        digest=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        class View:
            def read(self,name,default=None):
                return value if name=='summary.json' else {'RunID':'example','summary_canonical_sha256':digest}
        self.assertEqual(checked_summary(View())['new_measured_pairs'],0)
        value['new_measured_pairs']=36
        with self.assertRaises(DataError):checked_summary(View())

    def test_local_app_shows_no_new_measurements(self):
        try:from streamlit.testing.v1 import AppTest
        except ImportError:self.skipTest('Use the existing dashboard environment for AppTest.')
        with patch.dict(os.environ,{'UAVGAP_DASHBOARD_MODE':'local'}):
            app=AppTest.from_file(str(PUBLIC/'code/tools/dashboard.py')).run(timeout=25)
            selector=next(s for s in app.selectbox if s.label=='Experiment')
            target=next(s for s in selector.options if s.startswith('VM36'))
            with patch('subprocess.Popen',side_effect=AssertionError('Viewer must not start a worker')):
                selector.select(target).run(timeout=25)
                saved=next(s for s in app.selectbox if s.label=='Saved VM36 run')
                preparation=next(i for i,name in enumerate(saved.options) if '_protocol_v2_' in name)
                saved.set_value(preparation).run(timeout=25)
            self.assertFalse(app.exception)
            self.assertIn('548',[m.value for m in app.metric])
            self.assertEqual(next(m.value for m in app.metric if m.label=='New measured pairs'),'0')
            self.assertEqual(next(m.value for m in app.metric if m.label=='Formal measured pairs'),'0')

    def test_measured_component_renders_without_starting_a_worker(self):
        try:from streamlit.testing.v1 import AppTest
        except ImportError:self.skipTest('Use the dashboard environment.')
        script='''from dashboard_vm36 import render_measurements
render_measurements({'validated_validation_images':548,'new_measured_pairs':2,'combined_measured_pairs':8,'formal_measured_pairs':0,
    'worker_wall_seconds':60,'status':'PARTIAL','forward_profiler_seconds':1.0,'source_hashes':{},
    'effects':[],'gates':[{'branch':'test fixture','status':'NO_EXPERIMENT'}],
    'metrics':[{'checkpoint':'ck05','resolution':640,'JPEG':25,'images':548,'map50':.3,
        'map50_95':.2,'AP_CI_low':.1,'AP_CI_high':.3,'ar50_95':.4}]})'''
        with patch('subprocess.Popen',side_effect=AssertionError('No worker in viewer')):
            app=AppTest.from_string(script).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(next(m.value for m in app.metric if m.label=='New measured pairs'),'2')
        self.assertEqual(next(m.value for m in app.metric if m.label=='Combined measured pairs'),'8')
        self.assertTrue(app.warning)

    def test_cloud_never_falls_back_to_private_datasets(self):
        try:from streamlit.testing.v1 import AppTest
        except ImportError:self.skipTest('Use the existing dashboard environment for AppTest.')
        original=Path.open
        def guarded(path,*args,**kwargs):
            if path.resolve().is_relative_to((ROOT/'datasets').resolve()):raise AssertionError('Cloud read private data.')
            return original(path,*args,**kwargs)
        script='from dashboard_vm36 import render\nfrom dashboard_data import DashboardStore\nrender(DashboardStore(mode="cloud"))'
        with patch.dict(os.environ,{'UAVGAP_DASHBOARD_MODE':'cloud'}),patch.object(Path,'open',guarded),patch('subprocess.Popen',side_effect=AssertionError('No shell in viewer')):
            app=AppTest.from_string(script).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertTrue(app.info)


if __name__=='__main__':unittest.main()
