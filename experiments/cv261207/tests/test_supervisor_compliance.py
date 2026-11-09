from pathlib import Path
import json
import tempfile
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import supervisor_compliance as audit
from supervisor_requirements import requirements


class SupervisorAuditTests(unittest.TestCase):
    def test_attribute_origin_is_not_silently_reconciled(self):
        pdf={('a','x'):1,('b','y'):2,('c','z'):3}
        uvl={('a','x'):1,('c','z'):4,('d','p'):5}
        self.assertEqual([r['status'] for r in audit.attribute_diff(pdf,uvl)],
            ['MATCH','MISSING_FROM_REPOSITORY_UVL','VALUE_MISMATCH','IMPLEMENTATION_EXTENSION'])

    def test_numeric_attributes_preserve_decimals_and_sign(self):
        values=audit.numeric_attributes('F { sigma 5.5 bias -20 } G { photons 100 }')
        self.assertEqual(values,{('F','sigma'):5.5,('F','bias'):-20,('G','photons'):100})

    def test_atomic_output_refuses_unknown_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'run';out.mkdir()
            with patch.object(audit,'ROOT',root):
                audit.write_atomic(out/'a.json','old',out)
                with self.assertRaises(FileExistsError):audit.write_atomic(out/'a.json','new',out)
                self.assertEqual((out/'a.json').read_text(),'old')
                audit.write_atomic(out/'a.json','new',out,replace=True)
                self.assertEqual((out/'a.json').read_text(),'new')
                with self.assertRaises(ValueError):audit.write_atomic(root.parent/'unsafe.txt','x',out)

    def test_requirements_have_unique_ids_and_explicit_acceptance(self):
        rows=requirements();self.assertEqual(len({r['id'] for r in rows}),len(rows))
        for row in rows:
            self.assertTrue(1<=row['pdf_page']<=17)
            self.assertTrue(row['acceptance_condition']);self.assertTrue(row['next_executable_action'])
        c4=next(r for r in rows if r['id']=='SUP_C4')
        self.assertEqual(c4['evidence_key'],'release')

    def test_no_experimental_value_replaces_missing_attribute(self):
        row=audit.attribute_diff({('Nano','params_m'):2.5},{})[0]
        self.assertIsNone(row['repository_value'])
        self.assertEqual(row['status'],'MISSING_FROM_REPOSITORY_UVL')

    def test_visual_review_requires_current_recorded_evidence(self):
        from supervisor_accept import validate_visual_review
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'figures').mkdir();(out/'pdf_qa_final').mkdir()
            (out/'figures/supervisor_figures.pdf').write_bytes(b'test fixture')
            with self.assertRaises(ValueError):validate_visual_review(out)
            pages={}
            for i in range(1,13):
                name=f'page-{i:02}.png';(out/'pdf_qa_final'/name).write_bytes(bytes([i]))
                pages[name]=audit.sha256(out/'pdf_qa_final'/name)
            record={'status':'PASS','reviewer':'test fixture only','PDF_sha256':audit.sha256(out/'figures/supervisor_figures.pdf'),
                    'rendered_pages':pages}
            (out/'visual_review_input.json').write_text(json.dumps(record))
            self.assertEqual(validate_visual_review(out)['status'],'PASS')
            (out/'pdf_qa_final/page-01.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_visual_review(out)


if __name__=='__main__':unittest.main()
