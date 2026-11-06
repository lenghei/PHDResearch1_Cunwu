"""Historical dashboard evidence stays read-only and cloud-safe."""
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
from dashboard_data import DashboardStore, PUBLIC


class HistoricalDashboardTests(unittest.TestCase):
    def test_snapshot_has_no_private_sources_and_preserves_identity(self):
        store=DashboardStore(mode='cloud')
        found=0
        for run in store.history('cv261207'):
            value=store.view('cv261207',run['RunID']).read('summary.json',{})
            history=value.get('historical_vm26')
            if not history:
                continue
            found+=1
            self.assertEqual(history['configurations'],54)
            self.assertEqual(len({c['config_id'] for c in history['cells']}),54)
            self.assertEqual(len(history['gains']),18)
            self.assertEqual(len(history['headline_intervals']),9)
            self.assertEqual(history['validation']['held_out_status'],'NOT_EVALUATED')
            self.assertEqual(value['formal_evaluation'],'DATASET_BLOCKED')
            self.assertFalse(re.search(r'[A-Za-z]:[\\/]|/Users/|/home/',json.dumps(value)))
            self.assertEqual({c['checkpoint'] for c in history['cells']},{'ck01','ck05','ck06'})
        self.assertGreater(found,0)

    def test_dashboard_renders_without_local_data(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,{
                'UAVGAP_DASHBOARD_MODE':'cloud','UAVGAP_RUNS_ROOT':temp}):
            app=AppTest.from_file(str(TOOLS/'dashboard.py'),default_timeout=20).run()
            self.assertFalse(app.exception)
            option=next(x for x in app.sidebar.selectbox[0].options if 'cv261207' in x.lower())
            app.sidebar.selectbox[0].select(option).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get('plotly_chart')),3)
            self.assertTrue(any('admission remains blocked' in w.value for w in app.warning))
            self.assertTrue(any('NOT_EVALUATED' in i.value for i in app.info))


if __name__=='__main__':
    unittest.main()
