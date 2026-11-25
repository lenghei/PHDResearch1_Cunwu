from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from protocol import ROOT
from joint_analysis import empirical_tables,refresh,failure_associations
from historical_review import table
from protocol import load,save


class JointOutcomeTests(unittest.TestCase):
    def test_missing_frozen_recipe_stays_unknown_for_all_settings(self):
        accepted={}
        for ck in ('ck01','ck05'):
            for resolution in (640,1280):
                identity=ck+str(resolution)
                spec={'p':{'id':'observed','leaves':['Normal','JPEG_q25']},
                    'd':{'checkpoint':ck,'id':identity}}
                bootstrap={'intervals':[{'size':'tiny','metric':'map50_95','ci95':[.2,.3]}]}
                timing={'samples':[{'pipeline_ms':10.} for _ in range(548)]}
                accepted[identity]=(spec,{},bootstrap,timing)
        plan={'pairs':[{'p':{'id':'frozen_not_run'}}],'seed':42,'thresholds':[.1],'latency_limits_ms':[100]}
        parent=ROOT/'datasets/local_runs/checks';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_joint_',dir=parent) as folder:
            target=Path(folder)
            self.assertTrue(target.resolve().is_relative_to(parent.resolve()))
            states,gaps,static=empirical_tables(target,plan,accepted)
            self.assertEqual(len(states),4)
            missing=next(r for r in gaps if r['p_id']=='frozen_not_run')
            self.assertEqual(missing['measured_detector_settings'],0)
            self.assertEqual(missing['state'],'UNRESOLVED')
            text=(target/'ALL_SETTING_OUTCOMES.csv').read_text()
            self.assertEqual(text.count('NOT_EVALUATED'),4)

    def test_latest_counts_come_from_joint_results(self):
        parent=ROOT/'datasets/local_runs/checks'
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_joint_latest_',dir=parent) as folder:
            root=Path(folder);out=root/'TEST_ONLY';out.mkdir()
            save(out/'summary.json',{'status':'PARTIAL','combined_measured_pairs':9,
                'new_measured_pairs':1,'coverage':{'measured_states':500,'scope':'TEST_ONLY'}},out)
            def history(target):
                save(root/'latest.json',{'history':[],'measured_conditions':8,'new_conditions':2},out)
            with patch('factorial_analysis.refresh_history',side_effect=history):refresh(out)
            latest=load(root/'latest.json')
            self.assertEqual(latest['measured_conditions'],9)
            self.assertEqual(latest['new_conditions'],1)
            self.assertEqual(latest['measured_states'],500)

    def test_association_counts_do_not_invent_absent_outcomes(self):
        parent=ROOT/'datasets/local_runs/checks'
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_joint_counts_',dir=parent) as folder:
            out=Path(folder)
            accepted={'fixture':({'p':{'leaves':['feature']}},None,None,None)}
            states=[{'pair_id':'fixture','tau':.1,'L_cached_ms':100,'state':'UNCERTAIN'}]
            failure_associations(out,accepted,states)
            absent=next(row for row in table(out/'DESCRIPTIVE_FAILURE_ASSOCIATIONS.csv') if row['present']=='False')
            self.assertEqual(absent['measured_pairs'],'0')
            self.assertEqual(absent['FAIL'],'0')


if __name__=='__main__':unittest.main()
