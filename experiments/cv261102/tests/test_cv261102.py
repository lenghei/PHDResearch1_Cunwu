"""Pure CPU tests: no GPU, YOLO weights or real VisDrone archives required."""
from __future__ import annotations
import csv
import importlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TASK))
from prepare_data import label_rows,list_members,convert_split,NAMES,digest,scan_archive_annotations
from run import export_csv,clock_epochs,verify_replay,export_reports,parse,read_config,validate_existing_split
from validate_run import audit


class Tests(unittest.TestCase):
    def test_publishable_files_use_english_names_and_no_cjk(self):
        for path in TASK.rglob('*'):
            if path.is_dir() or '__pycache__' in path.parts:
                continue
            self.assertTrue(path.name.isascii(), str(path))
            if path.suffix.lower() in {'.py', '.md', '.ps1', '.yaml'}:
                data = path.read_text(encoding='utf-8')
                self.assertFalse(any(0x3400 <= ord(char) <= 0x9fff for char in data), str(path))

    def test_label_class_mapping_and_exclusions(self):
        x = b'10,20,20,10,1,1,0,0\n5,5,10,10,0,3,0,0\n6,7,4,4,1,11,0,0\n'
        result=label_rows(x,100,100)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0].split()[0],'0')
        self.assertAlmostEqual(float(result[0].split()[1]),.2)

    def test_label_clips(self):
        self.assertEqual(len(label_rows(b'-10,-10,20,20,1,4,0,0\n',100,100)),1)
        self.assertEqual(label_rows(b'120,120,1,1,1,4,0,0\n',100,100),[])

    def test_visdrone_trailing_comma_is_valid(self):
        plain = b'440,541,271,152,1,6,0,0\n'
        trailing = b'440,541,271,152,1,6,0,0,\n'
        spaced = b'440,541,271,152,1,6,0,0,   \r\n'
        expected = label_rows(plain, 1200, 900)
        self.assertEqual(len(expected), 1)
        self.assertEqual(label_rows(trailing, 1200, 900), expected)
        self.assertEqual(label_rows(spaced, 1200, 900), expected)

    def test_visdrone_trailing_comma_is_not_extra_field(self):
        with self.assertRaisesRegex(ValueError, 'expected 8 VisDrone fields'):
            label_rows(b'440,541,271,152,1,6,0,0,5\n', 1200, 900)
        with self.assertRaisesRegex(ValueError, 'expected 8 VisDrone fields'):
            label_rows(b'440,541,271,152,1,6,0,0,,\n', 1200, 900)
        with self.assertRaisesRegex(ValueError, 'non-numeric VisDrone record'):
            label_rows(b'440,541,271,152,1,6,0,not-number,\n', 1200, 900)

    def test_trailing_comma_scan_counts_expected_objects(self):
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d) / 'source.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('VisDrone2019-DET-train/images/a.jpg', b'image fixture')
                z.writestr('VisDrone2019-DET-train/annotations/a.txt',
                           b'440,541,271,152,1,6,0,0,\n1,1,0,4,1,4,0,0,\n')
            with zipfile.ZipFile(archive) as z:
                counts = scan_archive_annotations(z)
            self.assertEqual(counts['rows'], 2)
            self.assertEqual(counts['candidate_objects'], 1)
            self.assertEqual(counts['zero_area'], 1)

    def test_trailing_comma_is_converted_and_accounted(self):
        import cv2, numpy as np
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            archive = base / 'input.zip'
            ok, blob = cv2.imencode('.jpg', np.zeros((900, 1200, 3), dtype=np.uint8))
            self.assertTrue(ok)
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('V/images/a.jpg', blob.tobytes())
                z.writestr('V/annotations/a.txt', b'440,541,271,152,1,6,0,0,\n')
            results = convert_split(archive, base/'prepared', 'train', expected=1)
            self.assertEqual(results['labels'], 1)
            self.assertEqual(results['annotation_accounting']['kept'], 1)
            self.assertTrue((base/'prepared/labels/train/a.txt').read_text().startswith('5 '))

    def test_label_invalid(self):
        self.assertEqual(label_rows(b'1,1,0,4,1,1,0,0\n',100,100), [])
        with self.assertRaisesRegex(ValueError,'negative box dimensions'):
            label_rows(b'1,1,-1,4,1,1,0,0\n',100,100)
        with self.assertRaises(ValueError):label_rows(b'1,1,nan,4,1,1,0,0\n',100,100)

    def test_zero_area_annotations_are_excluded_and_counted(self):
        stats = dict(rows=0, ignored=0, zero_area=0, outside_image=0,
                     kept=0, zero_area_examples=[])
        raw = (b'1,1,0,4,1,1,0,0\n2,2,4,0,1,4,0,0\n'
               b'2,2,5,5,0,0,0,0\n5,5,10,10,1,4,0,0\n')
        labels = label_rows(raw, 100, 100, annotation_path='V/annotations/a.txt',
                            statistics=stats)
        self.assertEqual(len(labels), 1)
        self.assertEqual(stats['zero_area'], 2)
        self.assertEqual(stats['ignored'], 1)
        self.assertEqual(stats['kept'], 1)
        self.assertEqual(stats['rows'], 4)
        self.assertEqual(stats['zero_area_examples'][0]['line'], 1)

    def test_annotation_scan_catches_corruption_before_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d) / 'sample.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('V/images/a.jpg', b'fixture')
                z.writestr('V/annotations/a.txt',
                           b'1,1,0,4,1,1,0,0\n1,1,4,4,1,2,0,0\n')
            with zipfile.ZipFile(archive) as z:
                scan = scan_archive_annotations(z)
            self.assertEqual(scan['zero_area'], 1)
            self.assertEqual(scan['candidate_objects'], 1)
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('V/images/a.jpg', b'fixture')
                z.writestr('V/annotations/a.txt',
                           b'1,1,4,4,1,12,0,0\n')
            with zipfile.ZipFile(archive) as z:
                with self.assertRaisesRegex(ValueError,
                                            r'V/annotations/a\.txt:1: invalid VisDrone category'):
                    scan_archive_annotations(z)

    def test_conversion_zeros_are_reported_in_provenance(self):
        import cv2,numpy as np
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); zpath = root/'train.zip'
            ok, buffer = cv2.imencode('.jpg', np.zeros((40,50,3), dtype=np.uint8))
            self.assertTrue(ok)
            with zipfile.ZipFile(zpath, 'w') as z:
                z.writestr('V/images/a.jpg', buffer.tobytes())
                z.writestr('V/annotations/a.txt',
                           b'1,1,0,4,1,1,0,0\n10,10,20,10,1,4,0,0\n')
            result = convert_split(zpath, root/'converted', 'train', expected=1)
            self.assertEqual(result['labels'], 1)
            self.assertEqual(result['annotation_accounting']['zero_area'], 1)
            self.assertEqual(result['annotation_accounting']['kept'], 1)
            self.assertEqual(result['annotation_accounting']['zero_area_examples'][0]['line'], 1)

    def test_zip_images_and_labels(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'v.zip'
            with zipfile.ZipFile(f,'w') as z:
                z.writestr('VisDrone/images/a.jpg',b'fake')
                z.writestr('VisDrone/annotations/a.txt',b'')
            with zipfile.ZipFile(f) as z:
                img,lab=list_members(z)
                self.assertEqual(len(img),1)
                self.assertEqual(len(lab),1)

    def test_zip_rejects_duplicate_annotation_stem(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'v.zip'
            with zipfile.ZipFile(f,'w') as z:
                z.writestr('V/images/a.jpg',b'fake')
                z.writestr('V/annotations/a.txt',b'')
                z.writestr('Other/annotations/a.txt',b'')
            with zipfile.ZipFile(f) as z:
                with self.assertRaises(ValueError):list_members(z)

    def test_real_image_bytes_and_yolo_label_conversion(self):
        import cv2,numpy as np
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);zpath=root/'train.zip'
            ok,buffer=cv2.imencode('.jpg',np.zeros((40,50,3),dtype=np.uint8))
            self.assertTrue(ok)
            blob=buffer.tobytes()
            with zipfile.ZipFile(zpath,'w') as z:
                z.writestr('V/images/a.jpg',blob)
                z.writestr('V/annotations/a.txt',b'10,10,20,10,1,4,0,0\n')
            result=convert_split(zpath,root/'converted','train',expected=1)
            self.assertEqual(result['images'],1)
            self.assertEqual(result['labels'],1)
            self.assertEqual((root/'converted/images/train/a.jpg').read_bytes(),blob)
            row=(root/'converted/labels/train/a.txt').read_text().split()
            self.assertEqual(row[0],'3')
            self.assertAlmostEqual(float(row[1]),.4)

    def test_csv_written(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'t.csv'
            export_csv(f,[dict(name='AP50',value=.56)])
            with f.open(encoding='utf-8-sig',newline='') as stream:
                self.assertEqual(list(csv.DictReader(stream))[0]['value'],'0.56')

    def test_csv_schema_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):export_csv(Path(d)/'no.csv',[])
            with self.assertRaises(ValueError):export_csv(Path(d)/'no.csv',[dict(x=1),dict(y=2)])

    def test_epoch_clock_records_actual_callback_values(self):
        class FakeModel:
            def __init__(self):self.calls={}
            def add_callback(self,name,fn):self.calls[name]=fn
        with tempfile.TemporaryDirectory() as d:
            model=FakeModel();p=Path(d)/'epoch.csv';recorded=clock_epochs(model,p)
            trainer=SimpleNamespace(epoch=0,device=SimpleNamespace(type="cpu"))
            model.calls['on_train_epoch_start'](trainer)
            model.calls['on_train_epoch_end'](trainer)
            model.calls['on_fit_epoch_end'](trainer)
            model.calls['on_fit_epoch_end'](trainer)
            self.assertEqual(recorded,{1})
            with p.open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),1)
            self.assertGreater(float(rows[0]['train_seconds']),0)

    def test_local_config_and_override(self):
        with tempfile.TemporaryDirectory() as d:
            import yaml
            p=Path(d)/'config.local.yaml'
            p.write_text(yaml.safe_dump(dict(train_zip='train.zip',data_yaml=None,
                val_zip='val.zip',weights='yolov8s.pt',epochs=100,batch=4)),encoding='utf-8')
            x=read_config(parse(['--config',str(p),'--batch','2']))
            self.assertEqual(x['batch'],2)
            self.assertEqual(x['train_zip'],(Path(d)/'train.zip').resolve())

    def test_config_rejects_two_datasets(self):
        with tempfile.TemporaryDirectory() as d:
            import yaml
            p=Path(d)/'cfg.yaml';p.write_text(yaml.safe_dump(dict(train_zip='train.zip',data_yaml='data.yaml',val_zip='val.zip',weights='yolov8s.pt')),encoding='utf-8')
            with self.assertRaises(ValueError):read_config(parse(['--config',str(p)]))

    def test_fake_run_csv_and_missing_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'validation').mkdir();(p/'training').mkdir()
            (p/'training/epoch_times.csv').write_text('epoch,train_seconds,train_plus_validation_seconds\n1,5,6\n',encoding='utf-8')
            (p/'validation/summary.json').write_text(json.dumps(dict(images=548,full_validation_status='FULL_VALIDATION_PASS',network_seconds=5,evaluation_seconds=30,total_seconds=40)),encoding='utf-8')
            (p/'validation/ground_truth.json').write_text(json.dumps(dict(images=[dict(id=i) for i in range(548)])),encoding='utf-8')
            (p/'validation/metrics.json').write_text(json.dumps(dict(per_area=[dict(size='all',map50=.55,map50_95=.35,ar50_95=.53)],per_class=[dict(size='all',category_name='car',map50=.9)])),encoding='utf-8')
            (p/'validation/predictions_raw.json').write_text('[]',encoding='utf-8')
            r=export_reports(p,100,40,1)
            self.assertEqual(len(r['csv_SHA256']),5)
            with self.assertRaises(ValueError):export_reports(p,100,40,2)

    def make_audit_fixture(self, p: Path):
        (p/'validation').mkdir()
        (p/'csv').mkdir()
        (p/'training/yolov8s/weights').mkdir(parents=True)
        (p/'training/yolov8s/weights/best.pt').write_bytes(b'synthetic metadata fixture only')
        (p/'training/yolov8s/results.csv').write_text('epoch\n1\n', encoding='utf-8')
        (p/'training/yolov8s/args.yaml').write_text('epochs: 1\n', encoding='utf-8')
        epochs = [dict(epoch='1', train_seconds='2.0', train_plus_validation_seconds='3.0',
                       clock='time.perf_counter; live trainer callbacks',
                       recorded_at_UTC='2026-11-02T12:00:00+00:00')]
        export_csv(p/'training/epoch_times.csv', epochs)
        metrics = dict(per_area=[dict(size='all', map50=.56, map50_95=.35, ar50_95=.5)],
                       per_class=[dict(size='all', category_name='car', map50=.5)])
        (p/'validation/metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
        (p/'validation/predictions_raw.json').write_text('[]', encoding='utf-8')
        (p/'validation/predictions_coco.json').write_text('[]', encoding='utf-8')
        (p/'validation/ground_truth.json').write_text(
            json.dumps(dict(images=[dict(id=i) for i in range(548)])), encoding='utf-8')
        summary = dict(images=548, full_validation_status='FULL_VALIDATION_PASS',
                       RunID=p.name, ExperimentID='cv261102', network_seconds=5,
                       evaluation_seconds=30, total_seconds=40)
        (p/'validation/summary.json').write_text(json.dumps(summary), encoding='utf-8')
        export_csv(p/'csv/metrics_overall.csv', [dict(RunID=p.name, **metrics['per_area'][0])])
        export_csv(p/'csv/metrics_by_area.csv', metrics['per_area'])
        export_csv(p/'csv/metrics_by_class.csv', metrics['per_class'])
        export_csv(p/'csv/training_epoch_times.csv', epochs)
        export_csv(p/'csv/evaluation_timing.csv', [dict(images=548, network_seconds=5,
                   coco_seconds=30, evaluation_pipeline_seconds=40,
                   evaluation_wall_seconds=45, train_stage_wall_seconds=100,
                   meaning='measured training/evaluation wall clocks, not GPU-only kernel time')])
        checkpoint = dict(best_checkpoint_SHA256=digest(p/'training/yolov8s/weights/best.pt'),
                          actual_epochs=1,
                          ultralytics_results_SHA256=digest(p/'training/yolov8s/results.csv'),
                          ultralytics_args_SHA256=digest(p/'training/yolov8s/args.yaml'))
        (p/'training/checkpoint.json').write_text(json.dumps(checkpoint), encoding='utf-8')
        replay = dict(status='PASS', metric_records={'per_area': 1, 'per_class': 1},
                      GT_SHA256=digest(p/'validation/ground_truth.json'),
                      predictions_COCO_SHA256=digest(p/'validation/predictions_coco.json'),
                      metrics_SHA256=digest(p/'validation/metrics.json'))
        (p/'coco_regeneration.json').write_text(json.dumps(replay), encoding='utf-8')
        result = dict(csv_SHA256={x.name: digest(x) for x in (p/'csv').glob('*.csv')},
                      trained_epochs=1, images=548, AP50=.56, AP50_95=.35, AR50_95=.5,
                      metrics_SHA256=digest(p/'validation/metrics.json'),
                      predictions_raw_SHA256=digest(p/'validation/predictions_raw.json'),
                      GT_SHA256=digest(p/'validation/ground_truth.json'))
        manifest = dict(RunID=p.name, ExperimentID='cv261102', status='PASS', stage='COMPLETE',
                        result=result, training_wall_seconds=100, evaluation_wall_seconds=45,
                        source_checkpoint_SHA256=digest(p/'training/yolov8s/weights/best.pt'))
        (p/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return replay

    def test_audit_accepts_consistent_synthetic_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            replay = self.make_audit_fixture(p)
            self.assertEqual(audit(p, recompute=False)['status'], 'INTEGRITY_ONLY')
            with patch('run.verify_replay', return_value=replay) as mocked:
                self.assertEqual(audit(p)['status'], 'PASS')
                self.assertTrue(mocked.called)
            (p/'csv/training_epoch_times.csv').write_text('BAD', encoding='utf-8')
            with self.assertRaises(ValueError):audit(p, recompute=False)

    def test_audit_rejects_coordinated_csv_falsification(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); self.make_audit_fixture(p)
            path = p/'csv/metrics_overall.csv'
            export_csv(path, [dict(RunID=p.name, size='all', map50=.95, map50_95=.35, ar50_95=.5)])
            manifest=json.loads((p/'manifest.json').read_text())
            manifest['result']['csv_SHA256']['metrics_overall.csv']=digest(path)
            (p/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'mismatched map50'):
                audit(p, recompute=False)

    def test_audit_rejects_unrecomputed_data(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); self.make_audit_fixture(p)
            with patch('run.verify_replay', side_effect=ValueError('Mismatch from raw outputs')):
                with self.assertRaisesRegex(ValueError,'Mismatch from raw outputs'):
                    audit(p)

    def test_existing_yaml_images_must_match_original_zip_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d); (base/'images/val').mkdir(parents=True)
            original = base/'source.zip'
            with zipfile.ZipFile(original,'w') as archive:
                archive.writestr('VisDrone/images/a.jpg', b'original')
                archive.writestr('VisDrone/annotations/a.txt', b'')
            image = base/'images/val/a.jpg'
            image.write_bytes(b'original')
            validate_existing_split(dict(path=str(base), val='images/val'),
                                    base/'data.yaml', original, 'val', 1)
            image.write_bytes(b'other-image-data')
            with self.assertRaisesRegex(ValueError, 'differs from original'):
                validate_existing_split(dict(path=str(base), val='images/val'),
                                        base/'data.yaml', original, 'val', 1)

    def test_refuse_incomplete_audit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'manifest.json').write_text(json.dumps(dict(RunID=p.name,ExperimentID='cv261102',status='PARTIAL')),encoding='utf-8')
            with self.assertRaises(ValueError):audit(p)

    def test_replay_detects_missing_files(self):
        mocked=ModuleType('main.evaluation')
        mocked.regenerate=lambda path: None
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(sys.modules, {'main.evaluation': mocked}):
                with self.assertRaises(FileNotFoundError):verify_replay(Path(d))

    def test_replay_detects_metric_change(self):
        mocked=ModuleType('main.evaluation')
        mocked.regenerate=lambda path: dict(per_area=[dict(size='all',map50=.10)],per_class=[])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'metrics.json').write_text(json.dumps(dict(per_area=[dict(size='all',map50=.56)],per_class=[])),encoding='utf-8')
            with patch.dict(sys.modules, {'main.evaluation': mocked}):
                with self.assertRaises(ValueError):verify_replay(p)

if __name__=='__main__':unittest.main()
