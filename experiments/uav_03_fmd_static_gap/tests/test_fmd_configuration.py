import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import export_configuration_validation as exporter


class FMDFullConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "YOLOv8", "Small", "VisDrone_FT", "Train640", "HSV_Default", "Geometric_Default",
            "Infer640", "FullFrame", "NMS", "FP32"
        }

    def test_original_model_has_34_leaves(self):
        self.assertEqual(len(exporter.FEATURE_COLUMNS), 34)

    def test_training_variants_count(self):
        self.assertEqual(sum(1 for _ in exporter.training_variants()), 844)

    def test_expected_configuration_count_is_declared(self):
        self.assertEqual(exporter.EXPECTED_CONFIGS, 635904)
        self.assertEqual(exporter.EXPECTED_TRAIN_VARIANTS, 844)

    def test_four_constraints(self):
        # LEAF_YOLO must use Nano + VisDrone_FT + Train640 + NMS.
        leaf_ok = (self.base - {"YOLOv8", "Small"}) | {"LEAF_YOLO", "Nano"}
        self.assertTrue(exporter.valid_config(leaf_ok))
        self.assertFalse(exporter.valid_config(leaf_ok - {"Nano", "LEAF_YOLO"} | {"Small", "LEAF_YOLO"}))

        # EndToEnd_NMSFree is only legal for YOLO26.
        end2end_v8 = (self.base - {"NMS"}) | {"EndToEnd_NMSFree"}
        self.assertFalse(exporter.valid_config(end2end_v8))
        end2end_v26 = (end2end_v8 - {"YOLOv8"}) | {"YOLO26"}
        self.assertTrue(exporter.valid_config(end2end_v26))

        # COCO_ZeroShot requires Train640 and excludes five added augmentations.
        coco_ok = (self.base - {"VisDrone_FT"}) | {"COCO_ZeroShot"}
        self.assertTrue(exporter.valid_config(coco_ok))
        self.assertFalse(exporter.valid_config(coco_ok | {"MotionBlurAug"}))

        # SAHI_Slicing and TTA cannot be selected together in the original FM_D.
        sahi_ok = (self.base - {"FullFrame"}) | {"SAHI_Slicing"}
        self.assertTrue(exporter.valid_config(sahi_ok))
        self.assertFalse(exporter.valid_config(sahi_ok | {"TTA"}))

    def test_uvl_source_contains_original_constraints(self):
        self.assertEqual(len(exporter.read_and_check_uvl(HERE / "fm_d.uvl")), 64)


if __name__ == "__main__":
    unittest.main()
