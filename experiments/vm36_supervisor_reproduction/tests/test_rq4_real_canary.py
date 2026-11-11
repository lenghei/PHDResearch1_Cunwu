import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rq4_real_canary import canary_contract,TimedLoader,save_without_counter_hook
from rq4_preparation import optimizer_exposure_plan


class CanaryTests(unittest.TestCase):
    def plan(self):
        return {'args':{'epochs':1,'imgsz':1280,'batch':2,'workers':0,'nbs':64,'optimizer':'AdamW','lr0':.01,
            'cos_lr':True,'warmup_epochs':3.,'seed':42,'amp':True,'resume':False},'worker_seconds':600,
            'new_target_augmentations':[],'train':[{'image_id':str(i)} for i in range(64)],
            'development':[{'stem':str(i)} for i in range(16)]}

    def test_scope_rejects_full_training_or_numeric_fallback(self):
        canary_contract(self.plan())
        for key,value in (('epochs',100),('batch',1),('amp',False),('lr0',.001)):
            plan=self.plan();plan['args'][key]=value
            with self.assertRaises(ValueError):canary_contract(plan)
        plan=self.plan();plan['train'][1]=plan['train'][0]
        with self.assertRaises(ValueError):canary_contract(plan)

    def test_short_warmup_has_seven_real_update_opportunities(self):
        plan=optimizer_exposure_plan(64,1,2,64,3.)
        self.assertEqual(plan['forward_backward_batches'],32)
        self.assertEqual(plan['warmup_batches'],100)
        self.assertEqual(plan['optimizer_and_EMA_updates'],7)

    def test_fp32_requires_explicit_failed_run_diagnostic(self):
        plan=self.plan();plan['args']['amp']=False
        plan['precision_diagnostic']={'mode':'EXPLICIT_FP32_REPAIR'}
        with self.assertRaises(ValueError):canary_contract(plan)
        plan['precision_diagnostic']['failed_run']={'path':'UNIT_TEST_ONLY'}
        canary_contract(plan)
        plan['args']['lr0']=.001
        with self.assertRaises(ValueError):canary_contract(plan)

    def test_counter_hook_is_removed_before_snapshot_serialization(self):
        import torch
        from unittest.mock import patch
        from rq4_training_backend import RQ4HealthHooks
        obj=SimpleNamespace(model=torch.nn.Linear(2,2))
        obj.rq4_forward_handle=obj.model.register_forward_hook(lambda *args:None)
        def saving(trainer):
            import pickle
            self.assertFalse(trainer.model._forward_hooks)
            pickle.dumps(trainer.model)
            return True
        with patch.object(RQ4HealthHooks,'save_model',saving):self.assertTrue(save_without_counter_hook(obj))

    def test_repair_keeps_the_original_absolute_deadline(self):
        from datetime import datetime,timezone,timedelta
        from rq4_fp32_repair import remaining_allocation_seconds
        start=datetime(2026,11,11,12,tzinfo=timezone.utc)
        allocation={'start_UTC':start.isoformat(),'deadline_UTC':(start+timedelta(seconds=600)).isoformat(),
            'worker_seconds':600,'budget_reset_allowed':False,'formal_training_seconds':0}
        self.assertEqual(remaining_allocation_seconds(allocation,start+timedelta(seconds=599)),1)
        self.assertLess(remaining_allocation_seconds(allocation,start+timedelta(seconds=601)),0)
        allocation['deadline_UTC']=(start+timedelta(seconds=1200)).isoformat()
        with self.assertRaises(ValueError):remaining_allocation_seconds(allocation,start)

    def test_loader_records_actual_anchor_identity_without_resampling(self):
        trainer=SimpleNamespace();data=[{'im_file':['a.jpg','b.jpg']},{'im_file':['c.jpg','d.jpg']}]
        loader=TimedLoader(data,trainer)
        self.assertEqual(len(loader),2);observed=[]
        for batch in loader:observed.append((batch,trainer.rq4_anchor_ids))
        self.assertEqual([r[1] for r in observed],[['a','b'],['c','d']])


if __name__=='__main__':unittest.main()
