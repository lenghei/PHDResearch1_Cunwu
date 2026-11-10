from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from normal_reference_analysis import contrasts,timing_difference,subset_states,NORMAL,CONDITIONS,SIZES,METRICS


def fixture():
    found={}
    for p in (NORMAL,*CONDITIONS):
        for ck in ('ck01','ck05'):
            for res in (640,1280):
                value=(.2 if res==640 else .3) if p==NORMAL else (.1 if res==640 else .15)
                identity=p+'__'+ck+'_Infer'+str(res);extra=0 if p==NORMAL else 1
                found[identity]={'folder':'UNIT_TEST_ONLY','metrics':{'per_area':[{'size':s,**{m:value for m in METRICS}} for s in SIZES],
                    'per_class':[{'size':'large','category_id':1,'category_name':'UNIT_TEST_ONLY',**{m:None for m in METRICS}}]},
                    'spec':{'p':{'id':p}},'timing':{'samples':[{'image_id':i,'forward_ms':10+extra,'predict_ms':20+2*extra,'pipeline_ms':30+3*extra} for i in range(548)]},
                    'bootstrap':{'intervals':[{'size':'tiny','metric':'map50_95','ci95':[value,value]}],
                        'draws':{'seed':20261108,'samples':200,'sampling':'UNIT_TEST_ONLY','image_ids':list(range(548)),
                            'values':{s:{m:[value]*200 for m in METRICS} for s in SIZES}}}}
    return found


class NormalContrastTests(unittest.TestCase):
    def test_corruption_difference_and_resolution_did_have_distinct_signs(self):
        changes,times,did,classes=contrasts(fixture(),20261108)
        self.assertEqual(len(changes),180);self.assertEqual(len(did),90)
        row=next(r for r in changes if r['resolution']==1280)
        self.assertAlmostEqual(row['estimate'],-.15)
        self.assertTrue(all(abs(r['estimate']+.05)<1e-12 for r in did))
        self.assertTrue(all(abs(r['CI95_low']-r['estimate'])<1e-12 for r in did))
        self.assertTrue(all(r['difference'] is None for r in classes))
        self.assertEqual(next(r['difference_ms'] for r in times if r['component']=='pipeline_ms'),3)

    def test_missing_reference_stays_not_estimable(self):
        found=fixture();del found[NORMAL+'__ck01_Infer640']
        changes,_,did,_=contrasts(found,20261108)
        missing=[r for r in changes if r['checkpoint']=='ck01' and r['resolution']==640]
        self.assertTrue(all(r['estimate'] is None and r['status']=='NOT_ESTIMABLE' for r in missing))
        self.assertTrue(all(r['estimate'] is None for r in did if r['checkpoint']=='ck01'))
        found={k:v for k,v in found.items() if not k.startswith(NORMAL)}
        _,gaps=subset_states(found,20261108)
        self.assertTrue(all(r['subset_state']=='UNRESOLVED' for r in gaps if r['P']==NORMAL))

    def test_timing_pairing_checks_image_identity(self):
        found=fixture();n=found[NORMAL+'__ck01_Infer640'];d=found[next(iter(CONDITIONS))+'__ck01_Infer640']
        d['timing']['samples'][0]['image_id']=-1
        with self.assertRaises(ValueError):timing_difference(n,d,'pipeline_ms',20261108)


if __name__=='__main__':unittest.main()
