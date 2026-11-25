"""CPU checks of the rolling geometry and formal cache/pair adapters."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1]
sys.path.insert(0, str(MAIN/"code"))
sys.path.insert(0, str(MAIN/"experiments/experiments_test/cv261207"))
sys.path.insert(0, str(MAIN/"experiments/uav_02_fmp"))

from evaluation import evaluate_saved
from geometry import RollingRaster, transform_annotations
from main.statistics import bootstrap_coco, evaluate_coco
from pipeline import (identity, indices_for, load, measure_pair, read_cache, read_cache_image,
                      sha256, shared_cache, verify_pair, verify_repeat)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class GeometryTests(unittest.TestCase):
    def test_nonlinear_boxes_match_opencv_support(self):
        h, w = 67, 113
        for phase in (0, .41, 2.81):
            effect = {"feature": "RollingShutter", "params": {"max_shift_fraction": .08, "waves": 2},
                      "phase_rad":phase}
            raster = RollingRaster((h,w), effect)
            for x,y,ww,hh in ((0,0,3,66), (109,5,4,61), (35,1,31,64), (-3,9,5,4), (160,8,2,3)):
                mask = np.zeros((h,w), np.float32)
                mask[y:y+hh,max(x,0):min(x+ww,w)] = 1
                yy,xx = np.nonzero(raster.warp(mask))
                expected = [xx.min(),yy.min(),np.ptp(xx)+1,np.ptp(yy)+1] if len(xx) else None
                self.assertEqual(raster.bbox([x,y,ww,hh]), expected)

    def test_warped_ignore_union_is_not_a_bounding_rectangle(self):
        raster = RollingRaster((48,100), {"feature":"RollingShutter",
            "params":{"max_shift_fraction":.07,"waves":2}, "phase_rad":1.3})
        mask = np.zeros((48,100), np.float32)
        mask[4:43, 15:18] = 1
        mask[15:25, 17:23] = 1
        out = np.zeros((48,100), bool)
        for x,y,w,h in raster.regions([[15,4,3,39],[17,15,6,10]]):
            out[y:y+h,x:x+w] = True
        self.assertTrue(np.array_equal(out,raster.warp(mask)>0))
        x,y,w,h = raster.bbox([15,4,3,39])
        self.assertFalse(out[y:y+h,x:x+w].all())

    def test_original_area_keeps_tiny_ap_bin(self):
        record = {"id":1,"file_name":"case.png","width":100,"height":90,"objects":[
            {"class_id":0,"bbox":[24,4,12,18]}],"ignore_regions":[],"other_regions":[]}
        frozen = deepcopy(record)
        original, gt, _ = transform_annotations(record, {"feature":"RollingShutter",
             "params":{"max_shift_fraction":.08,"waves":2}, "phase_rad":1.3})
        self.assertEqual(record, frozen)
        self.assertEqual(original["annotations"][0]["area"],216)
        ann = gt["annotations"][0]
        self.assertGreaterEqual(ann["bbox"][2]*ann["bbox"][3],256)
        areas,_,_ = evaluate_coco(gt,[{"image_id":1,"category_id":1,"bbox":ann["bbox"],"score":.9}])
        tiny = next(x for x in areas if x["size"]=="tiny")
        self.assertEqual(tiny["gt_count"],1)
        self.assertAlmostEqual(tiny["map50_95"],1.)
        ann["area"] = 200
        with self.assertRaisesRegex(ValueError,"original box"):
            evaluate_coco(gt,[])


class Reader:
    def __init__(self):
        self.calls = 0
        self.members = [{"image_id":i, "shape":[40,50,3], "source":str(i)} for i in range(1,4)]

    def record(self,i):
        self.calls += 1
        proof = self.members[i]
        record = {"id":i+1,"stem":str(i+1), "file_name":f"{i+1}.png","height":40,"width":50,
                  "objects":[{"class_id":0,"bbox":[10,11,8,8]}],"ignore_regions":[],"other_regions":[]}
        return np.full((40,50,3),i+1,np.uint8),record,proof


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_module("new_fast_coco_accumulate",MAIN/"experiments/experiments_test/cv261207/development_fast_coco.py")

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reader = Reader()
        self.key = identity({"P":"fixture"})

    def cache(self):
        return shared_cache(self.root/"cache",self.key,self.reader.members,self.reader,
                            lambda image,stem:(image+10,[]),monitor_disk=False)

    def binding(self, folder, d):
        return {"images":3,"D":d,"gt_sha256":sha256(folder/"ground_truth.json"),
                "cache_manifest_sha256":sha256(folder/"manifest.json")}

    def predictor(self, pixels):
        self.seen.append(int(pixels[0,0,0]))
        return np.array([[10,11,18,19,.8,0]]), {"fixture_tensor":[1,3,40,50]}, {}

    def measure(self, target, binding, folder, manifest, predictor=None):
        self.seen = []
        return measure_pair(target,binding,folder,manifest,predictor or self.predictor,evaluate_saved,
                            bootstrap_coco,monitor_disk=False,samples=3,seed=7)

    def test_same_cached_pixels_and_gt_for_five_D_and_completed_skip(self):
        folder, manifest, hit = self.cache()
        self.assertFalse(hit)
        for i in range(5):
            same, verify, hit = self.cache()
            self.assertTrue(hit)
            self.assertEqual((same,verify),(folder,manifest))
            binding = self.binding(folder,i)
            target = self.root/f"pair{i}"
            self.assertEqual(self.measure(target,binding,folder,manifest),"COMPLETE")
            self.assertEqual(self.seen,[11,11,11,11,12,13])
            self.assertEqual(self.measure(target,binding,folder,manifest),"VERIFIED_SKIP")
            self.assertEqual(self.seen,[])
            self.assertAlmostEqual(load(target/"metrics.json")["per_area"][0]["map50_95"],1.)
        self.assertEqual(self.reader.calls,3)

    def test_cache_changed_png_and_gt_fail_closed(self):
        folder, manifest, _ = self.cache()
        (folder/"0002.png").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError,"PNG hash"):
            read_cache_image(folder,manifest["rows"][1])
        with self.assertRaisesRegex(ValueError,"PNG hash"):
            read_cache(folder,self.key,[1,2,3],verify_pixels=True)
        with (folder/"ground_truth.json").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(ValueError,"Corrupted artifact"):
            self.cache()

    def test_failure_is_separate_and_verified_rerun_skips(self):
        folder, manifest, _ = self.cache()
        binding = self.binding(folder,1)
        target = self.root/"paired"
        def interrupt(pixels):
            if pixels[0,0,0]==12:
                raise RuntimeError("controlled stop")
            return self.predictor(pixels)
        with self.assertRaisesRegex(RuntimeError,"controlled stop"):
            self.measure(target,binding,folder,manifest,interrupt)
        self.assertFalse(target.exists())
        partial = list(self.root.glob(".partial-paired-*"))
        self.assertEqual(len(partial),1)
        self.assertFalse((partial[0]/"COMPLETE.json").exists())
        self.assertEqual(self.measure(target,binding,folder,manifest),"COMPLETE")
        self.assertEqual(len(list(self.root.glob(".partial-paired-*"))),1)
        with self.assertRaisesRegex(ValueError,"Pair configuration|Completion"):
            self.measure(target,{**binding,"D":2},folder,manifest)

    def test_replay_repeat_bootstrap_indices_and_corrupt_completed_receipt(self):
        import gzip
        folder, manifest, _ = self.cache()
        binding = self.binding(folder,0)
        a,b = self.root/"a",self.root/"b"
        for target in (a,b):
            self.measure(target,binding,folder,manifest)
        verify_repeat(a,b,binding)
        self.assertEqual(evaluate_saved(a)[0],load(a/"metrics.json"))
        with gzip.open(a/"bootstrap_indices.npy.gz","rb") as stream:
            self.assertTrue(np.array_equal(np.load(stream,allow_pickle=False),indices_for(3,3,7)))
        (a/"metrics.json").write_text("{}",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"Corrupted artifact"):
            verify_pair(a,binding)

    def test_member_hash_or_order_mismatch_rejects_cache_build(self):
        other = list(reversed(self.reader.members))
        with self.assertRaisesRegex(ValueError,"membership or order"):
            shared_cache(self.root/"cache",self.key,other,self.reader,lambda i,s:(i,[]),monitor_disk=False)
        self.assertFalse((self.root/"cache"/self.key).exists())

    def test_cli_missing_authorizations_cannot_start(self):
        command = [sys.executable,"-B",str(HERE/"run.py"),"--execute"]
        result = subprocess.run(command,capture_output=True,text=True,timeout=15)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("separate science and GPU",result.stderr)

    def test_exact_frozen_identity_without_GPU(self):
        command = [sys.executable,"-B",str(HERE/"run.py"),"--identity"]
        result = subprocess.run(command,capture_output=True,text=True,check=True,timeout=15)
        data = json.loads(result.stdout)
        self.assertEqual(len(data["first_group"]),5)
        self.assertEqual(data["formal_images_per_pair"],1610)
        self.assertEqual(data["selection_sha256"],"b59d3b19f2a1260c1464576631e6161bf22b858c5ea66e7df82b0ea7acd42b35")


class ThermalTests(unittest.TestCase):
    def test_thermal_guard_stops_hard_sustained_and_stale_samples(self):
        from supervisor import thermal_reason
        policy = {"immediate_stop_temperature_C":90, "sustained_stop_temperature_C":87, "sustained_seconds":30}
        safety = load_module("test_existing_safety",MAIN/"experiments/experiments_test/vm37_yolo26_rq4/post_cost/safety.py")
        def row(seconds, deg, uuid="test"):
            return {"monotonic_seconds":seconds,"temperature_C":deg,"uuid":uuid}

        self.assertEqual(thermal_reason(safety,policy,[row(0,66),row(1,90)],"test"),"TEMPERATURE_HARD_STOP")
        self.assertEqual(thermal_reason(safety,policy,[row(x,87) for x in range(31)],"test"),"SUSTAINED_TEMPERATURE_STOP")
        self.assertEqual(thermal_reason(safety,policy,[row(0,72),row(3,72)],"test"),"TELEMETRY_FAILURE")
        self.assertEqual(thermal_reason(safety,policy,[row(0,72),row(1,72,"other")],"test"),"TELEMETRY_FAILURE")
        self.assertIsNone(thermal_reason(safety,policy,[row(x,87) for x in range(30)],"test"))


if __name__ == "__main__":
    unittest.main()
