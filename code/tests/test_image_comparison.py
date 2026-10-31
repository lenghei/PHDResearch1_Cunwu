"""Four-panel previews: provenance, publication, and tamper rejection."""
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import types

import numpy as np
from PIL import Image

PUBLIC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PUBLIC / 'code'))
sys.path.insert(0, str(PUBLIC / 'code/tools'))
from dashboard_data import DashboardStore, DataError
from publish_image_comparison import create_comparisons, sha256

STUDY = 'vm1_flamapy_yolo26s_validation'
RUNID = 'vm1_flamapy_yolo26s_validation_full-paired_20261030T194618_3f0454d3'


def write_json(path, data):
    path.write_text(json.dumps(data), encoding='utf8')


def image_bytes(color):
    out = io.BytesIO()
    Image.new('RGB', (80, 60), color).save(out, 'JPEG')
    return out.getvalue()


class CloudPreviewTests(unittest.TestCase):
    def test_compare_discovery_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            public = Path(tmp) / 'repo'
            original = PUBLIC / 'reports/results' / STUDY / RUNID
            folder = public / 'reports/results' / STUDY / RUNID
            folder.mkdir(parents=True)
            for name in ['manifest.json', 'snapshot.json']:
                shutil.copyfile(original / name, folder / name)
            entries = {}
            for name, color in [('original.jpg', 'red'), ('degraded.jpg', 'gray'),
                                ('clean_predictions.jpg', 'green'), ('degraded_predictions.jpg', 'blue')]:
                p = folder / 'image_comparison' / '0001' / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(image_bytes(color))
                entries[name] = sha256(p)
            preview_index = folder / 'image_comparison' / 'manifest.json'
            write_json(preview_index, {'schema': 1, 'RunID': RUNID, 'images': [
                {'image_id': 1, 'files': entries, 'source_file': 'sample.jpg', 'recipe': {'kernel': 7}}]})
            publication = json.loads((folder / 'manifest.json').read_text())
            for p in (folder / 'image_comparison').rglob('*'):
                if p.is_file():
                    publication['files'][p.relative_to(folder).as_posix()] = sha256(p)
            write_json(folder / 'manifest.json', publication)
            store = DashboardStore('cloud', public=public)
            view = store.view(STUDY, RUNID)
            previews = view.comparison_sets()
            self.assertEqual(len(previews), 1)
            self.assertEqual(len(previews[0]['pictures']), 4)
            target = folder / 'image_comparison/0001/original.jpg'
            target.write_bytes(target.read_bytes() + b'X')
            with self.assertRaisesRegex(DataError, 'checksum mismatch'):
                view.comparison_sets()

    def test_existing_snapshot_has_no_preview_yet(self):
        store = DashboardStore('cloud', public=PUBLIC)
        self.assertEqual(store.view(STUDY, RUNID).comparison_sets(), [])


class LocalRendererTests(unittest.TestCase):
    def test_render_synthetic_inputs_no_inference(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / RUNID
            run.mkdir()
            dataset_zip = root / 'val.zip'
            dataset_zip.write_bytes(b'synthetic test-only fixture')
            bgr = np.zeros((60, 80, 3), dtype=np.uint8)
            bgr[..., 1] = 120
            bgr[:, :20, 2] = 255
            transformed = cv2.GaussianBlur(bgr, (7, 7), 2)
            self.assertFalse(np.array_equal(bgr, transformed))
            image_sha = hashlib.sha256(b'fixture original jpeg bytes').hexdigest()
            source_proof = {'image_SHA256': image_sha}
            write_json(run / 'run.json', {'RunID': RUNID, 'ExperimentID': STUDY,
                                           'status': 'PASS', 'mode': 'full-paired'})
            write_json(run / 'result.json', {'status': 'PASS', 'images_per_condition': 548, 'conditions': 2})
            write_json(run / 'configuration.json', {'dataset_zip': str(dataset_zip),
                             'dataset_zip_sha256': sha256(dataset_zip)})
            gt = {'categories': [{'id': 4, 'name': 'car'}],
                  'images': [{'id': 1, 'file_name': 'fixture.jpg'}]}
            pred = [{'image_id': 1, 'category_id': 4, 'bbox': [10, 12, 16, 20], 'score': .91}]
            for name, frame in [('CLEAN', bgr), ('DEGRADED', transformed)]:
                folder = run / name
                folder.mkdir()
                if name == 'DEGRADED':
                    (folder / 'images').mkdir()
                    cv2.imwrite(str(folder / 'images' / 'fixture.png'), frame)
                write_json(folder / 'ground_truth.json', gt)
                write_json(folder / 'predictions_coco.json', pred)
                write_json(folder / 'metrics.json', {'fixture': True})
                (folder / 'predictions_raw.jsonl').write_text('{}\n')
                trace = {'image_id': 1, 'image_SHA256': image_sha,
                         'output_pixel_SHA256': hashlib.sha256(frame.tobytes()).hexdigest(),
                         'configuration_id': 'ck01_I0B1N2_R1280',
                         'recipe': {'kernel': 7, 'noise_variance': 30.0}}
                if name == 'DEGRADED':
                    trace.update(transformed_file='images/fixture.png',
                                 transformed_file_SHA256=sha256(folder / 'images/fixture.png'))
                write_json(folder / 'trace.json', [trace])
                write_json(folder / 'summary.json', {'Status': 'PASS', 'images': 548,
                    'GT_SHA256': sha256(folder / 'ground_truth.json'),
                    'predictions_coco_SHA256': sha256(folder / 'predictions_coco.json'),
                    'metrics_SHA256': sha256(folder / 'metrics.json'),
                    'predictions_SHA256': sha256(folder / 'predictions_raw.jsonl'),
                    'raw_prediction_file': 'predictions_raw.jsonl'})

            class FakeArchive:
                def __init__(self, *args):
                    pass
                def record(self, i):
                    assert i == 0
                    return bgr, {'id': 1, 'file_name': 'fixture.jpg'}, source_proof
                def close(self):
                    pass
            fake_main = types.ModuleType('main')
            fake_main.__path__ = []
            fake_dataset = types.ModuleType('main.dataset')
            fake_dataset.VisDroneZIP = FakeArchive
            with patch.dict(sys.modules, {'main': fake_main, 'main.dataset': fake_dataset}):
                path = create_comparisons(run, ids=[1], max_width=640)
            items = sorted(p.name for p in (path / '0001').glob('*.jpg'))
            self.assertEqual(len(items), 4)
            self.assertTrue((path / 'manifest.json').is_file())
            self.assertTrue((path / 'index.html').is_file())
            self.assertEqual(json.loads((path / 'manifest.json').read_text())['images'][0]['displayed_clean_boxes'], 1)


if __name__ == '__main__':
    unittest.main()
