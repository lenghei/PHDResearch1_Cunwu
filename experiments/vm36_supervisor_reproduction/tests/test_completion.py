from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from complete_factorial import missing_specs


class CompletionTests(unittest.TestCase):
    def test_existing_pairs_cannot_be_selected_again(self):
        specs=[{'id':'old','d':{'checkpoint':'ck05','resolution':640},'p':{'leaves':['Normal','JPEG_q25']}}]
        specs += [{'id':q,'d':{'checkpoint':'ck01','resolution':1280},'p':{'leaves':['Normal',q]}} for q in ('JPEG_q25','JPEG_q75')]
        self.assertEqual([s['id'] for s in missing_specs({'pairs':specs},{'old'})],['JPEG_q25','JPEG_q75'])
        with self.assertRaises(ValueError):missing_specs({'pairs':specs},{'old','JPEG_q25'})

    def test_scientific_measurement_block_matches_the_accepted_worker(self):
        before=(HERE/'full_development.py').read_text(encoding='utf-8')
        after=(HERE/'complete_factorial.py').read_text(encoding='utf-8')
        begin="        predict=device.detector(spec['d'])"
        end='        seal(folder);completed.append('
        self.assertEqual(before[before.index(begin):before.index(end)],after[after.index(begin):after.index(end)])


if __name__=='__main__':unittest.main()
