import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from development_analysis import outcome
from development_recovery import durable


class RecoveryTests(unittest.TestCase):
    def test_joint_outcomes_preserve_uncertainty_and_missingness(self):
        self.assertEqual(outcome([.1,.2],[30,40],.1,50),'PASS')
        self.assertEqual(outcome([.1,.2],[45,55],.1,50),'UNCERTAIN')
        self.assertEqual(outcome([.01,.03],None,.1,50),'FAIL')
        self.assertEqual(outcome([.1,.2],None,.1,50),'NOT_EVALUATED')
        self.assertEqual(outcome(None,[51,55],.1,50),'FAIL')
        self.assertEqual(outcome([.09,.11],[30,40],.1,50),'UNCERTAIN')

    def test_durable_flush_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);f=p/'result.json';data=json.dumps({'status':'PASS'}).encode();f.write_bytes(data)
            durable(p);self.assertEqual(f.read_bytes(),data)


if __name__=='__main__':unittest.main()
