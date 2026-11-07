"""RQ dashboard reads verified public summaries without local research data."""
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
from dashboard_data import DashboardStore


class RQDashboardTests(unittest.TestCase):
    def test_public_summary_preserves_gates_and_has_no_private_paths(self):
        store=DashboardStore(mode='cloud');found=0
        for run in store.history('cv261207'):
            s=store.view('cv261207',run['RunID']).read('summary.json',{}).get('rq_acceptance')
            if not s:continue
            found+=1
            self.assertEqual(s['formal_pairs'],0)
            self.assertEqual(s['test_gate'],'DATASET_BLOCKED')
            self.assertEqual(s['historical_unique_responses'],279)
            self.assertEqual(len(s['feature_compatibility']),34)
            self.assertTrue(all(v is None for v in s['held_out_metrics'].values()))
            self.assertFalse(re.search(r'[A-Za-z]:[\\/]|/Users/|/home/',json.dumps(s)))
        self.assertGreater(found,0)

    def test_acceptance_renders_in_cloud_with_local_store_absent(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{
                'UAVGAP_DASHBOARD_MODE':'cloud','UAVGAP_RUNS_ROOT':temp}):
            app=AppTest.from_file(str(TOOLS/'dashboard.py'),default_timeout=20).run()
            app.sidebar.selectbox[0].select('CV261207').run()
            self.assertFalse(app.exception)
            self.assertTrue(any('admission remains blocked' in x.value for x in app.warning))
            self.assertTrue(any('ML_INSUFFICIENT_EVIDENCE' in x.value for x in app.info))
            self.assertTrue(any('NOT_EVALUATED' in x.value for x in app.info))


if __name__=='__main__':unittest.main()
