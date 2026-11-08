"""New development summaries must work without private datasets."""
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


class DevelopmentDashboardTests(unittest.TestCase):
    def test_verified_summary_preserves_scope_and_pair_count(self):
        store=DashboardStore(mode='cloud');found=0
        for run in store.history('cv261207'):
            item=store.view('cv261207',run['RunID']).read('summary.json',{}).get('development_study')
            if not item:continue
            found+=1
            self.assertEqual(item['formal_pairs'],0)
            self.assertEqual(item['independent_test_gate'],'DATASET_BLOCKED')
            self.assertEqual(item['image_IDs'],248)
            self.assertEqual(item['unique_image_hashes'],247)
            self.assertEqual(item['unique_new_pairs'],len(item['pairs']))
            self.assertEqual(len({p['RunID'] for p in item['pairs']}),len(item['pairs']))
            self.assertEqual(item['learning']['classifier'],'ML_INSUFFICIENT_EVIDENCE')
            self.assertIsNone(item['learning']['balanced_accuracy'])
            self.assertIsNone(item['execution_cost']['repeated_run_worker_seconds'])
            self.assertFalse(re.search(r'[A-Za-z]:[\\/]|/Users/|/home/',json.dumps(item)))
        self.assertGreater(found,0)

    def test_charts_render_with_private_results_unavailable(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{
                'UAVGAP_DASHBOARD_MODE':'cloud','UAVGAP_RUNS_ROOT':temp}):
            app=AppTest.from_file(str(TOOLS/'dashboard.py'),default_timeout=30).run()
            app.sidebar.selectbox[0].select('CV261207').run()
            saved=next(s for s in app.selectbox if s.label=='Saved run')
            index=next(i for i,name in enumerate(saved.options) if 'development_delivery' in name)
            saved.set_value(index).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get('plotly_chart')),2)
            self.assertTrue(any('admission remains blocked' in w.value for w in app.warning))
            self.assertTrue(any('ML_INSUFFICIENT_EVIDENCE' in i.value for i in app.info))
            self.assertTrue(any('NOT_EVALUATED' in i.value for i in app.info))

    def test_local_bare_summary_uses_the_same_measured_view(self):
        from streamlit.testing.v1 import AppTest
        store=DashboardStore(mode='cloud')
        saved=next(r for r in store.history('cv261207') if 'development_delivery' in r['RunID'])
        brief=store.view('cv261207',saved['RunID']).read('summary.json')['development_study']
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{
                'UAVGAP_DASHBOARD_MODE':'local','UAVGAP_RUNS_ROOT':temp}):
            path=Path(temp)/'cv261207'/saved['RunID'];path.mkdir(parents=True)
            (path/'summary.json').write_text(json.dumps(brief),encoding='utf-8')
            (path/'run.json').write_text(json.dumps({'RunID':saved['RunID'],'mode':'DEVELOPMENT_DELIVERY','status':'PARTIAL'}),encoding='utf-8')
            app=AppTest.from_file(str(TOOLS/'dashboard.py'),default_timeout=30).run()
            app.sidebar.selectbox[0].select('CV261207').run()
            self.assertFalse(app.exception)
            self.assertTrue(any(m.label=='New measured pairs' for m in app.metric))
            self.assertEqual(len(app.get('plotly_chart')),2)


if __name__=='__main__':unittest.main()
