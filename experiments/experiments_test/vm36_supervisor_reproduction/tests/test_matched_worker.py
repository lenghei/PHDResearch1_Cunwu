from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from protocol import ROOT,save
from matched_worker import validate_authorization,completed_from
import matched_worker
import joint_batch


class MatchedWorkerGateTests(unittest.TestCase):
    def test_measurement_function_is_reused_without_scientific_changes(self):
        self.assertIs(matched_worker.measure_pair,joint_batch.measure_pair)

    def test_old_or_missing_budget_cannot_authorize_execution(self):
        parent=ROOT/'datasets/local_runs/checks';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_matched_auth_',dir=parent) as folder:
            out=Path(folder);save(out/'EXECUTION_PROTOCOL.json',{'scope':'TEST_ONLY'},out)
            save(out/'old_authorization.json',{'scope':'VM36_MATCHED3_FULL548','worker_budget_seconds':14400},out)
            with self.assertRaises(ValueError):validate_authorization(out,out/'old_authorization.json')

    def test_changed_resume_protocol_is_rejected(self):
        parent=ROOT/'datasets/local_runs/checks'
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_matched_resume_',dir=parent) as folder:
            out=Path(folder);save(out/'EXECUTION_PROTOCOL.json',{'pairs':[],'seed':42},out)
            with self.assertRaises(ValueError):completed_from(out,{'pairs':[],'seed':43})
            self.assertEqual(completed_from(out,{'pairs':[],'seed':42}),{})


if __name__=='__main__':unittest.main()
