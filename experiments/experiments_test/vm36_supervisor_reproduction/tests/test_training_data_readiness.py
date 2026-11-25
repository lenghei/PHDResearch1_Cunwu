from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training_data_readiness import native_rows


class TrainingAnnotationTests(unittest.TestCase):
    def test_trailing_delimiter_preserves_the_eight_original_values(self):
        first=native_rows('10,20,30,40,1,4,0,0',100,100)[0]
        second=native_rows('10,20,30,40,1,4,0,0,',100,100)[0]
        self.assertTrue(np.array_equal(first,second))

    def test_zero_area_is_logged_without_clipping_or_repair(self):
        values,errors,_,_=native_rows('10,20,30,0,1,4,0,0',100,100)
        self.assertEqual(len(values),0)
        self.assertTrue(errors[0]['positive_native10'])
        self.assertEqual(errors[0]['raw_row'],'10,20,30,0,1,4,0,0')

    def test_internal_missing_field_and_nonfinite_values_reject(self):
        with self.assertRaises(ValueError):native_rows('10,,30,40,1,4,0,0',100,100)
        with self.assertRaises(ValueError):native_rows('nan,20,30,40,1,4,0,0',100,100)

    def test_duplicate_handling_matches_library_unique_rows(self):
        values,_,_,duplicates=native_rows('10,20,30,40,1,4,0,0\n10,20,30,40,1,4,0,0',100,100)
        self.assertEqual((len(values),duplicates),(1,1))


if __name__=='__main__':unittest.main()
