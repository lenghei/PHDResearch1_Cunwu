"""Exercise the new cache, resume and admission paths without CUDA."""
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_testdev_101p import (  # noqa: E402
    LEGACY, ROOT, RUNS, base_binding, bootstrap_coco, build_cache, completed_pairs,
    degradation, evaluate_saved, frozen_plan, generator, identity, load, measure_pair,
    module, OriginalZIP, run_groups, sha256, sparse_bootstrap, with_sparse_matching,
)


class ValidationFixture:
    def __init__(self):
        source = ROOT / "datasets/VisDrone/VisDrone2019-DET-val.zip"
        reader = OriginalZIP(source, 548)
        try:
            self.records = [reader.record(i) for i in range(3)]
        finally:
            reader.close()
        self.members = [member for _, _, member in self.records]

    def record(self, index):
        return self.records[index]


class TestDevCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study, cls.plan = frozen_plan()
        cls.fixture = ValidationFixture()
        cls.render = staticmethod(generator().generate)
        module("new_fast_coco_accumulate", LEGACY / "development_fast_coco.py")

    def setUp(self):
        work = TemporaryDirectory(prefix="cpu-campaign-", dir=RUNS)
        self.addCleanup(work.cleanup)
        self.out = Path(work.name)

    def test_rolling_cache_rebuild_is_exact_and_keeps_source_area(self):
        p = degradation(self.plan[0][0])
        self.assertEqual(p["annotation_mode"], "WARP_GT_AND_IGNORE_ORIGINAL_AREA")
        a, b = self.out / "a", self.out / "b"
        for folder in (a, b):
            build_cache(folder, identity(p), p, self.fixture.members, self.fixture, self.render)
        for name in ("manifest.json", "ground_truth.json", "COMPLETE.json", "0001.png"):
            self.assertEqual(sha256(a / name), sha256(b / name))
        gt = load(a / "ground_truth.json")
        annotations = gt["coco"]["annotations"]
        self.assertTrue(annotations)
        self.assertTrue(any(a["bbox"] != a["source_bbox"] for a in annotations))
        self.assertTrue(all(a["area"] == a["source_bbox"][2] * a["source_bbox"][3] for a in annotations))
        self.assertTrue(any(source[1]["ignore_regions"] != derived["ignore_regions"]
                            for source, derived in zip(self.fixture.records, gt["records"])))

    def test_p18_uses_original_annotations_and_historical_pixels(self):
        p = degradation(self.plan[-1][0])
        path = self.out / "p18"
        b = build_cache(path, identity(p), p, self.fixture.members, self.fixture, self.render)
        source = ROOT / "datasets/local_runs/uav_04_sampling/reproduce_20261126T000929_14dcac74"
        prior = source / "pairs/p_7b49139cd7ab294c__d_806d430bf3db1913/image_manifest.jsonl"
        with prior.open(encoding="utf-8") as stream:
            checks = [json.loads(next(stream)) for _ in self.fixture.members]
        self.assertEqual([r["pixel_sha256"] for r in b["rows"]], [r["pixel_sha256"] for r in checks])
        self.assertEqual(load(path / "ground_truth.json")["records"],
                         [record for _, record, _ in self.fixture.records])

    def test_interrupt_and_resume_reuses_completion_and_cache_identity(self):
        groups = [self.plan[0][:2], self.plan[-1][:2]]
        study = {"protocol": "CPU_TEST", "images": len(self.fixture.members)}
        root = self.out / "pairs"
        measured = []

        def cpu_fixture(target, binding, cache, cached, row):
            measured.append(row["pair_id"])
            gt = load(cache / "ground_truth.json")["coco"]
            ann = next(a for a in gt["annotations"] if a["image_id"] == 1)
            x, y, w, h = ann["bbox"]
            boxes = np.array([[x, y, x + w, y + h, .95, ann["category_id"] - 1]])
            status = measure_pair(
                target, binding, cache, cached, lambda _: (boxes, {}, {}),
                partial(with_sparse_matching, evaluate_saved),
                partial(sparse_bootstrap, fallback=bootstrap_coco),
                samples=binding["bootstrap"]["samples"], seed=binding["bootstrap"]["seed"],
            )
            self.assertGreater(load(target / "metrics.json")["per_area"][0]["map50_95"], 0)
            return status

        def interrupted(target, binding, cache, cached, row):
            if measured:
                raise InterruptedError("CPU resume test")
            return cpu_fixture(target, binding, cache, cached, row)

        with self.assertRaises(InterruptedError):
            run_groups(self.out, root, groups, study, set(), self.fixture.members, self.fixture,
                       self.render, interrupted, 2)
        recovered = completed_pairs(root, groups, study)
        first = groups[0][0]
        self.assertEqual(recovered, {first["pair_id"]})
        checkpoint = root / first["pair_id"] / "COMPLETE.json"
        before = sha256(checkpoint)
        done = run_groups(self.out, root, groups, study, recovered, self.fixture.members, self.fixture,
                          self.render, cpu_fixture, 2)
        self.assertEqual(len(done), 4)
        self.assertEqual(len(set(measured)), len(measured))
        self.assertEqual(sha256(checkpoint), before)
        self.assertEqual(completed_pairs(root, groups, study), done)
        self.assertFalse(list(self.out.glob("pixels-*")))

    def test_execute_requires_both_authorizations(self):
        result = subprocess.run([sys.executable, "-B", str(HERE / "run_testdev_101p.py"), "--execute"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires Test-dev science approval", result.stderr)


if __name__ == "__main__":
    unittest.main()
