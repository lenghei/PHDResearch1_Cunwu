"""Fail closed until a new VM45 scientific and GPU admission exists."""
from vm45_common import csv_write
def record_unmeasured(out):
 for name in ('matched_evaluation_metrics.csv','paired_bootstrap_intervals.csv','factorial_interactions.csv','cross_detector_interactions.csv'):
  csv_write(out/name,[{'status':'NOT_EVALUATED','reason':'VM45 scientific bindings and专项GPU permission absent; no new predictions or AP','images_measured':0}],['status','reason','images_measured'])
def execute(*args,**kwargs):
 raise PermissionError('VM45 has no admitted GPU executor. Reuse the accepted VM43 backend only after a separately authorized, source-bound8-cell implementation.')
