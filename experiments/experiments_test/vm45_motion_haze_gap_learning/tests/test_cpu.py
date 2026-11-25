"""VM45 admission and incomplete-interaction tests."""
import unittest
class Checks(unittest.TestCase):
 def test_incomplete_DID(self):
  from interaction import did
  self.assertIsNone(did({'Normal':.2}))
 def test_four_corner_DID(self):
  from interaction import did
  self.assertAlmostEqual(did({'Normal':.4,'Motion':.3,'Haze':.2,'MotionHaze':.05}),-.05)
 def test_GPU_default_rejection(self):
  from evaluate import execute
  with self.assertRaises(PermissionError):execute()
 def test_group_metadata(self):
  from vm45_common import read,V44
  rs=read(V44/'FM_P_to_FM_D_decision_rules.json')['development_rows']
  for p in {r['condition_id'] for r in rs}:self.assertFalse({r['condition_id'] for r in rs if r['condition_id']!=p}&{p})
if __name__=='__main__':unittest.main()
