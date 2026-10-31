"""No-GPU tests of genuine training time parsing, comparison and evidence separation."""
from __future__ import annotations
import csv, importlib.util, json, sys, tempfile, unittest, hashlib
from pathlib import Path

PUBLIC=Path(__file__).resolve().parents[2]
SCRIPTS=PUBLIC/'experiments/cv261102'
sys.path.insert(0,str(SCRIPTS))

def load_module(name):
    path=SCRIPTS/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

epoch=load_module('epoch_evidence')
verify=load_module('verify')
export=load_module('export')

class SupervisorEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
    def makecsv(self,name,fields,rows):
        p=self.root/name
        with p.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        return p
    def test_original_cumulative_epoch_derivation_skips_first(self):
        source=self.makecsv('results.csv',['epoch','time'],[
            {'epoch':1,'time':100.0},{'epoch':2,'time':175.25},
            {'epoch':3,'time':258.5}])
        r=epoch.training_timing(training_results=source)
        self.assertEqual(r['status'],'DERIVED_FROM_TRAINER_CUMULATIVE_TIME')
        self.assertEqual(r['rows'],2)
        self.assertAlmostEqual(r['mean_seconds'],79.25)
        self.assertTrue(r['first_epoch_excluded'])
    def test_epoch_gap_rejected(self):
        source=self.makecsv('results.csv',['epoch','time'],[
            {'epoch':1,'time':100},{'epoch':3,'time':240}])
        with self.assertRaisesRegex(ValueError,'Missing epoch'):
            epoch.ultralytics_results_csv(source)
    def test_no_fake_original_training_time_from_probe(self):
        source=self.makecsv('probe.csv',['epoch','wall_seconds'],[
            {'epoch':1,'wall_seconds':32.5},{'epoch':2,'wall_seconds':30}])
        r=epoch.training_timing(train_probe_csv=source)
        self.assertEqual(r['status'],'NEW_TRAINING_PROBE_ONLY')
        self.assertIn('NEW_CALIBRATION',r['scope'])
    def test_duplicate_or_negative_epoch_rejected(self):
        source=self.makecsv('bad.csv',['epoch','wall_seconds'],[
            {'epoch':1,'wall_seconds':2},{'epoch':1,'wall_seconds':-4}])
        with self.assertRaises(ValueError):epoch.measured_epoch_csv(source)
    def test_dataset_yaml_10_classes_required(self):
        from train_timing import check_training_data
        p=self.root/'data.yaml';p.write_text('train: images/train\nval: images/val\nnc: 2\nnames: [a,b]\n')
        with self.assertRaisesRegex(ValueError,'VisDrone native10'):
            check_training_data(p)
    def test_reference_same_run_rejected(self):
        root=self.root/'a';root.mkdir()
        with self.assertRaisesRegex(ValueError,'cannot serve'):
            verify.independent_run_reference(root,root)
    def test_independent_run_needs_provenance(self):
        a,b=(self.root/'a',self.root/'b');a.mkdir();b.mkdir()
        (a/'summary.json').write_text(json.dumps({'checkpoint_id':'ck05','images':548,'RunID':'a',
             'Status':'PASS','scope':'FULL_DEVELOPMENT_VALIDATION','protocol':'x',
             'checkpoint_SHA256':'q','dataset_ZIP_SHA256':'r'}))
        (b/'summary.json').write_text(json.dumps({'checkpoint_id':'ck05','images':548,'RunID':'b',
             'Status':'PASS','scope':'FULL_DEVELOPMENT_VALIDATION','protocol':'y',
             'checkpoint_SHA256':'q','dataset_ZIP_SHA256':'r'}))
        with self.assertRaisesRegex(ValueError,'protocol'):
            verify.independent_run_reference(a,b)
    def test_missing_epoch_report_stays_missing(self):
        self.assertEqual(epoch.training_timing()['status'],'MISSING_TRAINING_EPOCH_TIMING')
    def test_csv_detail_export_with_measured_epoch(self):
        run=self.root/'run1';run.mkdir()
        metrics={'per_area':[{'size':'all','map50':0.5,'map50_95':0.3,'ar50_95':0.4,
                  'ar50':0.8,'gt_count':2,'image_count':548}], 'per_class':[]}
        (run/'metrics.json').write_text(json.dumps(metrics))
        summary={'RunID':'run1','Status':'PASS','checkpoint_id':'ck05','images':548,
                 'scope':'FULL_DEVELOPMENT_VALIDATION','protocol':export.PROTOCOL,
                 'metrics_SHA256':export.sha256(run/'metrics.json'),'aggregate':metrics['per_area'][0],
                 'dataset':'VisDrone2019-DET-val','checkpoint_SHA256':'x','dataset_ZIP_SHA256':'y'}
        (run/'summary.json').write_text(json.dumps(summary))
        source=self.makecsv('original.csv',['epoch','wall_seconds'],[
            {'epoch':1,'wall_seconds':45},{'epoch':2,'wall_seconds':50}])
        result=export.export_csv(run,epoch_source=source,epoch_source_kind='original')
        t=Path(result['directory'])/'cv261102_timing_training_epochs.csv'
        with t.open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[1]['wall_seconds'],'50.0')
        self.assertEqual(rows[0]['training_scope'],'ORIGINAL_LOG_UNVERIFIED_CHECKPOINT_LINEAGE')

if __name__=='__main__':unittest.main()
