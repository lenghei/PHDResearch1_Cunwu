"""CPU-only unit tests; planned GPU pairs are never executed in tests."""
from __future__ import annotations
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PUBLIC = HERE.parents[2]
sys.path.insert(0, str(HERE))
from sampling import (load_fmp_module, enumerate_configs, make_plan,
                      compute_binary_pair_masks, write_plan_files)
from run import sha, read_evidence
from figures import read_uvl_top_groups


class TestCurrentModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fmp = load_fmp_module(PUBLIC/'experiments/experiments_test/cv261109/model.py')
        cls.configs = enumerate_configs(cls.fmp)
        cls.plan = make_plan(cls.fmp, cls.configs, seed=20261104)

    def test_01_exact_fmp_space(self):
        self.assertEqual(len(self.configs), 46368)
        self.assertEqual(len(set(self.configs)), 46368)

    def test_02_exact_distribution(self):
        hist = Counter(map(len, self.configs))
        self.assertEqual([hist[i] for i in range(1,10)],
                         [5,85,631,2678,7125,12201,13167,8208,2268])
        self.assertEqual(sum(hist.values()), 46368)

    def test_03_valid_selected_leaves(self):
        self.assertTrue(all(self.fmp.valid_leaves(c) for c in self.configs))
        self.assertEqual(len(self.fmp.ALL_LEAVES), 25)

    def test_04_greedy_binary_pairwise_coverage(self):
        leaves = sorted(self.fmp.ALL_LEAVES)
        masks, feasible = compute_binary_pair_masks(self.configs, leaves)
        self.assertEqual(feasible.bit_count(),1163)
        selected = {item['valid_space_index'] for item in self.plan['pairwise']}
        actual = 0
        for i in selected:actual |= masks[i]
        self.assertEqual(actual, feasible)
        self.assertEqual(self.plan['greedy_cover_count'],21)

    def test_05_sample_budgets(self):
        self.assertEqual(len(self.plan['pairwise']),80)
        self.assertEqual(len(self.plan['uniform']),100)
        self.assertEqual(len({x['configuration_id'] for x in self.plan['pairwise']}),80)
        self.assertEqual(len({x['configuration_id'] for x in self.plan['uniform']}),100)
        self.assertEqual(len(self.plan['first_phase_fmp']),100)
        self.assertEqual(len({x['configuration_id'] for x in self.plan['first_phase_fmp']}),100)
        self.assertEqual(self.plan['first_phase']['planned_total_image_inferences'],805000)

    def test_06_deterministic_seed(self):
        other = make_plan(self.fmp,self.configs,seed=20261104)
        self.assertEqual([x['configuration_id'] for x in self.plan['pairwise']],
                         [x['configuration_id'] for x in other['pairwise']])
        self.assertEqual([x['configuration_id'] for x in self.plan['uniform']],
                         [x['configuration_id'] for x in other['uniform']])

    def test_07_different_seed_changes_sampling(self):
        other = make_plan(self.fmp,self.configs,seed=20261105)
        self.assertNotEqual([x['configuration_id'] for x in self.plan['uniform']],
                            [x['configuration_id'] for x in other['uniform']])

    def test_08_export_has_no_invented_ap(self):
        with tempfile.TemporaryDirectory() as temp:
            write_plan_files(self.plan,Path(temp))
            with (Path(temp)/'first_500_pairs_PLAN_ONLY.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows),500)
            self.assertTrue(all(row['AP50_95']=='' and row['latency_ms']=='' for row in rows))
            self.assertTrue(all(row['status']=='PLANNED_NOT_EXECUTED' and row['checkpoint_sha256']=='NOT_BOUND' for row in rows))
            self.assertTrue(all(row['planned_split']=='VisDrone2019-DET-test-dev' for row in rows))

    def test_09_fmd_uvl_tree(self):
        groups=read_uvl_top_groups(PUBLIC/'experiments/experiments_test/cv261116/fm_d.uvl')
        self.assertIn('Architecture',groups)
        self.assertIn('TrainAugmentation',groups)
        self.assertIn('Preprocessing',groups)
        self.assertIn('TTA',groups)

    def test_10_missing_static_audit_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'bad';folder.mkdir()
            (folder/'result.json').write_text(json.dumps({'RunID':'bad','ExperimentID':'cv261116',
                                               'model_validation':'PASS','files_SHA256':{}}))
            with self.assertRaises((ValueError,FileNotFoundError)):
                read_evidence(folder,PUBLIC/'experiments/experiments_test/cv261109/fm_p.uvl',
                              PUBLIC/'experiments/experiments_test/cv261116/fm_d.uvl')

    def test_11_public_preview_has_nine_figures(self):
        root=HERE/'results';self.assertTrue((root/'index.html').is_file())
        for i in range(1,10):
            image=list(root.glob(f'fig{i:02d}_*.png'))
            vector=list(root.glob(f'fig{i:02d}_*.svg'))
            self.assertEqual(len(image),1)
            self.assertEqual(len(vector),1)
        self.assertTrue((root/'cv261123_supervisor_figures.pdf').read_bytes().startswith(b'%PDF'))

    def test_12_public_preview_hashes(self):
        root=HERE/'results';meta=json.loads((root/'figures_manifest.json').read_text())
        for name, expected in meta['files_SHA256'].items():
            self.assertEqual(sha(root/name),expected,msg=name)
        self.assertEqual(meta['pair_design'],'500_PLANNED_NOT_EXECUTED')


if __name__=='__main__':unittest.main()
