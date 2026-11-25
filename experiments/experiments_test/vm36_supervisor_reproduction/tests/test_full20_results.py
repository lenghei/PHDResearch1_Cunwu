from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from full20_results import outcomes,finite20_marginals,latency_interval_state,single_factor_identifiability,resolution_effects


class FiniteSubsetOutcomeTests(unittest.TestCase):
    def test_single_factor_claims_reject_multi_factor_changes(self):
        found={}
        recipes=[['Normal','JPEG_q25'],['Normal','JPEG_q75'],['Underexposed_strong','Gauss_s30','JPEG_q50']]
        for p,leaves in enumerate(recipes):
            for ck in ('ck01','ck05'):
                for resolution in (640,1280):
                    found[f'TEST_{p}_{ck}_{resolution}']={'spec':{'p':{'id':f'TEST_P{p}','leaves':leaves},'d':{'id':f'{ck}_Infer{resolution}'}}}
        rows=single_factor_identifiability(found)
        self.assertEqual(sum(r['degradation_family']=='compression' and r['status']=='DIRECTLY_ESTIMABLE' for r in rows),2)
        self.assertTrue(all(r['status'].startswith('NOT_ESTIMABLE') for r in rows if r['degradation_family']=='gaussian'))

    def test_latency_interval_crossing_is_uncertain_not_confirmed_failure(self):
        self.assertEqual(latency_interval_state([.2,.3],[45,55],.1,50),'UNCERTAIN')
        self.assertEqual(latency_interval_state([.2,.3],[51,55],.1,50),'FAIL')
        self.assertEqual(latency_interval_state([0.,.01],[45,55],.1,50),'FAIL')
        self.assertEqual(latency_interval_state([.2,.3],[40,49],.1,50),'PASS')
        self.assertEqual(latency_interval_state([.08,.12],[40,49],.1,50),'UNCERTAIN')

    def test_context_mean_uses_existing_AP_and_missing_cells_are_not_filled(self):
        found={};sizes=('all','tiny','small','medium','large');metrics=('map50_95','map50','ar50_95')
        for p in range(20):
            for ck in ('ck01','ck05'):
                for res in (640,1280):
                    value=.2 if res==1280 else .1
                    found[f'{p}_{ck}_{res}']={'spec':{'p':{'id':str(p),'leaves':['TEST_ONLY']},'d':{'id':f'{ck}_Infer{res}'}},
                        'metrics':{'per_area':[{'size':size,**{m:value for m in metrics}} for size in sizes]},
                        'bootstrap':{'draws':{'image_ids':list(range(548)),'samples':200,'seed':42,'sampling':'TEST_ONLY',
                            'values':{size:{m:[value]*200 for m in metrics} for size in sizes}}}}
        rows=finite20_marginals(found)
        delta=next(r for r in rows if r['contrast']=='ck01_1280_minus640_mean' and r['size']=='all' and r['metric']=='map50_95')
        self.assertAlmostEqual(delta['estimate'],.1)
        matched=next(r for r in resolution_effects(found) if r['p_id']=='0' and r['checkpoint']=='ck01' and r['size']=='all' and r['metric']=='map50_95')
        self.assertAlmostEqual(matched['estimate'],.1)
        self.assertAlmostEqual(matched['CI95_low'],.1)
        del found['0_ck01_1280']
        missing=next(r for r in finite20_marginals(found) if r['contrast']=='ck01_1280_minus640_mean')
        self.assertEqual(missing['status'],'NOT_ESTIMABLE')
        self.assertIsNone(missing['estimate'])
        missing_effect=next(r for r in resolution_effects(found) if r['p_id']=='0' and r['checkpoint']=='ck01')
        self.assertEqual(missing_effect['status'],'NOT_ESTIMABLE')
        self.assertIsNone(missing_effect['estimate'])

    def test_incomplete_cells_stay_unknown_and_pass_is_a_witness(self):
        found={}
        for p in range(20):
            settings=range(4) if p==0 else range(1)
            for d in settings:
                identity=f'TEST_P{p}__TEST_D{d}'
                found[identity]={'spec':{'id':identity,'p':{'id':f'TEST_P{p}'},'d':{'id':f'TEST_D{d}'}},
                    'bootstrap':{'intervals':[{'size':'tiny','metric':'map50_95','ci95':[.2,.3] if p==1 else [0.,.001]}]},
                    'timing':{'samples':[{'pipeline_ms':10.} for _ in range(548)]},'checks':{'metrics.json':'TEST_ONLY'}}
        pairs,cells,gaps=outcomes(found,42)
        primary={r['p_id']:r['subset_state'] for r in gaps if r['tau']==.1 and r['L_cached_ms']==100}
        self.assertEqual(primary['TEST_P0'],'GAP_TESTED_SUBSET')
        self.assertEqual(primary['TEST_P1'],'COVERED_TESTED_SUBSET')
        self.assertEqual(primary['TEST_P2'],'UNRESOLVED')
        self.assertEqual(len(cells),20*4*6)
        self.assertTrue(any(r['state']=='NOT_EVALUATED' for r in cells))


if __name__=='__main__':unittest.main()
