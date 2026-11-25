from datetime import datetime,timezone,timedelta
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from protocol import ROOT,save,source
from full20_completion import validate_day_budget,measure_pair
import joint_batch


class Full20BudgetTests(unittest.TestCase):
    def test_original_measurement_function_is_unchanged(self):
        self.assertIs(measure_pair,joint_batch.measure_pair)

    def test_total_budget_cannot_restart_from_a_later_time(self):
        parent=ROOT/'datasets/local_runs/checks';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_full20_budget_',dir=parent) as folder:
            out=Path(folder);start=datetime(2026,11,10,12,tzinfo=timezone.utc)
            save(out/'parent.json',{'recorded_at_UTC':start.isoformat()},out)
            a={'global_start_UTC':start.isoformat(),'global_deadline_UTC':(start+timedelta(days=1)).isoformat(),
                'parent_authorization':source(out/'parent.json')}
            validate_day_budget(a)
            with self.assertRaises(ValueError):validate_day_budget({**a,'global_deadline_UTC':(start+timedelta(days=1,seconds=1)).isoformat()})
            with self.assertRaises(ValueError):validate_day_budget({**a,'global_start_UTC':(start+timedelta(hours=1)).isoformat()})


if __name__=='__main__':unittest.main()
