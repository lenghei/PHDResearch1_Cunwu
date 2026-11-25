"""Fast deterministic CPU checks for CV261116, requiring no GPU or FlamaPy."""
from __future__ import annotations
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import model
import run
import gpu_profile


class CV261116Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fmp = run.read_fmp_model(run.PUBLIC / 'experiments/experiments_test/cv261109/model.py')
        cls.rules = model.load_rules(HERE/'coverage_rules.json')
        cls.gap = model.gap_analysis(cls.fmp, cls.rules)

    def test_leaf_count(self):
        self.assertEqual(len(model.LEAVES), 34)

    def test_independent_configuration_count(self):
        self.assertEqual(model.enumerate_count()['independent_configurations'], 635904)

    def test_independent_training_variants(self):
        self.assertEqual(len(list(model.training_variants())), 844)

    def test_four_explicit_constraints(self):
        value = (HERE/'fm_d.uvl').read_text(encoding='utf-8')
        for name in ('LEAF_YOLO', 'EndToEnd_NMSFree', 'COCO_ZeroShot', 'SAHI_Slicing'):
            self.assertEqual(value.count('\n    ' + name + ' =>'), 1)

    def test_all_leaves_written_to_uvl(self):
        lines = (HERE/'fm_d.uvl').read_text(encoding='utf-8').split('constraints')[0].splitlines()
        names = {line.strip().split(' ',1)[0] for line in lines}
        self.assertTrue(model.LEAVES.issubset(names))

    def test_fmp_count(self):
        self.assertEqual(self.gap['FM_P_configurations'], 46368)

    def test_proxy_coverage(self):
        self.assertEqual(self.gap['proxy_covered'], 18)

    def test_proxy_gap(self):
        self.assertEqual(self.gap['proxy_gap'], 46350)

    def test_conservative_coverage(self):
        self.assertEqual(model.strict_identity_gap(self.fmp)['covered'], 1)

    def test_covered_ids_are_distinct(self):
        self.assertEqual(len(set(self.gap['covered_configuration_ids'])), 18)

    def test_feature_presence_provenance(self):
        self.assertTrue(all(v['covered_when_present'] <= v['present_total']
                            for v in self.gap['per_leaf'].values()))

    def test_all_rules_explicit(self):
        self.assertEqual(len(self.rules['rules']), 9)

    def test_no_false_verified_status(self):
        self.assertIn('PROVISIONAL', self.rules['verification_status'])

    def test_unknown_family_rejected(self):
        value=json.loads(json.dumps(self.rules))
        value['covered_levels']['fake_family']=[]
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'rules.json'
            file.write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaises(ValueError):
                model.load_rules(file)

    def test_unknown_feature_rejected(self):
        value=json.loads(json.dumps(self.rules))
        value['covered_levels']['motion']=['Fake_motion']
        with self.assertRaises(ValueError):
            model.gap_analysis(self.fmp, value)

    def test_coco_no_strong_training_aug(self):
        self.assertFalse(any('COCO_ZeroShot' == tr[2] and tr[4].intersection(model.AUG[1:])
                             for tr in model.training_variants()))

    def test_leaf_training_restrictions(self):
        for architecture, scale, weights, res, aug in model.training_variants():
            if architecture == 'LEAF_YOLO':
                self.assertEqual((scale,weights,res),('Nano','VisDrone_FT','Train640'))

    def test_no_cuda_profiler_import_at_module_load(self):
        self.assertTrue(callable(gpu_profile.run_profile))

    def test_missing_fmp_fails(self):
        with self.assertRaises(FileNotFoundError):
            run.read_fmp_model(HERE/'nonexistent_model.py')

    def test_training_source_audit_unavailable_is_explicit(self):
        from source_audit import audit_training
        self.assertEqual(audit_training()['coverage_claim'], 'NOT_VERIFIED')

    def test_result_json_writer_is_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'result.json'
            run.save(path,{'status':'PASS'})
            with self.assertRaises(FileExistsError):
                run.save(path,{'status':'PARTIAL'})

    def test_published_snapshot_identity_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'snapshot.json'
            run.save(path,{'RunID':'id','ExperimentID':'cv261116','files':{}})
            h=model.sha256(path)
            self.assertEqual(len(h),64)
            self.assertNotEqual(h,'0'*64)

    def test_cover_table_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            run.write_coverage_tables(Path(temp),self.gap,self.rules)
            with (Path(temp)/'coverage_by_feature.csv').open(encoding='utf-8',newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),25)
            self.assertIn('covered_when_present',rows[0])

    def test_public_root_is_main(self):
        self.assertEqual(run.PUBLIC.name,'main')


class ProfileBranchTests(unittest.TestCase):
    def test_valid_declaration(self):
        value = model.declared_profile_configuration(
            'YOLOv8','Small','VisDrone_FT','Train640',1280,'FP32','FullFrame','NMS')
        self.assertEqual(len(value['configuration_id']),18)
        self.assertIn('HSV_Default',value['selected_features'])

    def test_end_to_end_restriction(self):
        with self.assertRaises(ValueError):
            model.declared_profile_configuration(
                'YOLOv8','Small','VisDrone_FT','Train640',1280,'FP32','FullFrame','EndToEnd_NMSFree')

    def test_leaf_restriction(self):
        with self.assertRaises(ValueError):
            model.declared_profile_configuration(
                'LEAF_YOLO','Small','VisDrone_FT','Train640',1280,'FP32','FullFrame','NMS')

    def test_zero_shot_restriction(self):
        with self.assertRaises(ValueError):
            model.declared_profile_configuration(
                'YOLOv8','Small','COCO_ZeroShot','Train1280',1280,'FP32','FullFrame','NMS')

    def test_sahi_tta_restriction(self):
        with self.assertRaises(ValueError):
            model.declared_profile_configuration(
                'YOLOv8','Small','VisDrone_FT','Train640',1280,'FP32','SAHI_Slicing','NMS',tta=True)


class LargeModelTimeoutRegressionTests(unittest.TestCase):
    """Regression tests for the 600-second enumerative counting timeout."""

    def test_guard_accepts_original_uvl(self):
        import flamapy_count
        digest = flamapy_count.validate_uvl_contract(HERE / 'fm_d.uvl')
        self.assertEqual(len(digest), 64)

    def test_factorized_proof_terms(self):
        import flamapy_count
        proof = flamapy_count.factorized_count()
        self.assertEqual(proof['valid_count'], 635904)
        self.assertEqual(proof['training_variants'], 844)
        self.assertEqual(proof['proof_terms']['weighted_postprocessing_cases'], 1104)
        self.assertEqual(proof['proof_terms']['inference_cases_per_postprocessor'], 576)

    def test_guard_rejects_modified_constraint(self):
        import flamapy_count
        original = (HERE / 'fm_d.uvl').read_text(encoding='utf-8')
        changed = original.replace('SAHI_Slicing => !TTA', 'SAHI_Slicing => TTA')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fm_d.uvl'
            path.write_text(changed, encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'structure or constraints changed'):
                flamapy_count.validate_uvl_contract(path)

    def test_guard_rejects_changed_cardinality(self):
        import flamapy_count
        original = (HERE / 'fm_d.uvl').read_text(encoding='utf-8')
        changed = original.replace('                    YOLO11', '                    YOLONew')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fm_d.uvl'
            path.write_text(changed, encoding='utf-8')
            with self.assertRaises(ValueError):
                flamapy_count.validate_uvl_contract(path)

    def test_guard_accepts_extra_blank_lines_not_semantics(self):
        import flamapy_count
        original = (HERE / 'fm_d.uvl').read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fm_d.uvl'
            path.write_text(original.replace('\nconstraints', '\n\nconstraints') + '\n', encoding='utf-8')
            self.assertEqual(len(flamapy_count.validate_uvl_contract(path)), 64)

    def test_flama_sat_admission_no_count_enumeration(self):
        import flamapy_count
        import types
        calls = []
        fake_features = [SimpleNamespace(name=n, relations=[]) for n in sorted(model.LEAVES)]
        fm = SimpleNamespace(ctcs=[object() for _ in range(4)],
                             get_features=lambda: fake_features)
        class Reader:
            def __init__(self, path):
                self.path = path
            def transform(self):
                return fm
        class Discover:
            def use_operation_from_file(self, name, path):
                calls.append(name)
                if name == 'PySATSatisfiable':
                    return True
                if name == 'PySATDeadFeatures':
                    return []
                raise AssertionError('An enumerative or unsupported backend was invoked: ' + name)
        mapping = {}
        packages = [
            'flamapy', 'flamapy.metamodels', 'flamapy.metamodels.fm_metamodel',
            'flamapy.metamodels.fm_metamodel.transformations',
            'flamapy.metamodels.fm_metamodel.transformations.uvl_reader',
            'flamapy.core', 'flamapy.core.discover',
        ]
        for name in packages:
            module = types.ModuleType(name)
            module.__path__ = []
            mapping[name] = module
        mapping[packages[4]].UVLReader = Reader
        mapping[packages[-1]].DiscoverMetamodels = Discover
        with patch.dict(sys.modules, mapping):
            report = flamapy_count.analyze(HERE / 'fm_d.uvl')
        self.assertEqual(report['valid_count'], 635904)
        self.assertEqual(report['count_backend'], 'EXACT_FACTORIZED_COUNT_OF_STRICT_UVL_V1')
        self.assertEqual(calls, ['PySATSatisfiable', 'PySATDeadFeatures'])

    def test_flama_satisfiability_failure_is_not_pass(self):
        import flamapy_count
        import types
        fake_features = [SimpleNamespace(name=n, relations=[]) for n in sorted(model.LEAVES)]
        class Reader:
            def __init__(self, path): pass
            def transform(self):
                return SimpleNamespace(ctcs=[None] * 4, get_features=lambda: fake_features)
        class Discover:
            def use_operation_from_file(self, name, path):
                return False
        mapping = {}
        names = ['flamapy', 'flamapy.metamodels', 'flamapy.metamodels.fm_metamodel',
                 'flamapy.metamodels.fm_metamodel.transformations',
                 'flamapy.metamodels.fm_metamodel.transformations.uvl_reader',
                 'flamapy.core', 'flamapy.core.discover']
        for name in names:
            mod = types.ModuleType(name)
            mod.__path__ = []
            mapping[name] = mod
        mapping[names[4]].UVLReader = Reader
        mapping[names[-1]].DiscoverMetamodels = Discover
        with patch.dict(sys.modules, mapping):
            with self.assertRaisesRegex(ValueError, 'did not confirm'):
                flamapy_count.analyze(HERE / 'fm_d.uvl')


if __name__ == "__main__":
    unittest.main()
