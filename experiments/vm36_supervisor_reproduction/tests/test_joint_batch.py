from pathlib import Path
import ast
import sys
import textwrap
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from joint_batch import coverage_contract,BUDGET
from protocol import ROOT,EXPERIMENT
from historical_review import table


class JointBatchTests(unittest.TestCase):
    def test_frozen_contract_and_budget(self):
        path=ROOT/'datasets/local_runs'/EXPERIMENT/'vm36_supervisor_reproduction_joint_design_20261109T211826_4ada1a21/JOINT_COVERAGE_PLAN.csv'
        result=coverage_contract(table(path))
        self.assertEqual(result['projected_pairs'],7488)
        self.assertEqual(result['existing_states'],459)
        self.assertEqual(result['planned_total_states'],1033)
        self.assertEqual(BUDGET,14400)

    def test_measurement_math_matches_accepted_source(self):
        source=(HERE/'complete_factorial.py').read_text(encoding='utf-8')
        before=source[source.index("        predict=device.detector(spec['d'])"):source.index('        seal(folder);completed.append(')]
        target=(HERE/'joint_batch.py').read_text(encoding='utf-8')
        after=target[target.index("    predict=device.detector(spec['d'])"):target.index('    return timing')]
        expected=ast.parse(textwrap.dedent(before));actual=ast.parse(textwrap.dedent(after))
        class RemoveGuard(ast.NodeTransformer):
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='resource_guard':return None
                return self.generic_visit(node)
        actual=RemoveGuard().visit(actual)
        self.assertEqual(ast.dump(expected,include_attributes=False),ast.dump(actual,include_attributes=False))


if __name__=='__main__':unittest.main()
