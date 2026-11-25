"""VM45 local paths and evidence helpers."""
import csv, hashlib, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]; PUBLIC=ROOT/'main'
sys.path.insert(0,str(PUBLIC/'code'))
V44=ROOT/'datasets/local_runs/vm44_tiny_error_variability_decision/vm44_tiny_error_variability_decision_analysis_20261117T031628_9ebff202'
V41=ROOT/'datasets/local_runs/vm41_degradation_coverage/vm41_degradation_coverage_coverage_20261116T020508_dfa7879d'
UVL=HERE.parent/'vm36_supervisor_reproduction/feature_models'
SOURCES={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ref(p):
 p=Path(p).resolve(strict=True); r={'path':str(p),'SHA256':sha(p),'bytes':p.stat().st_size}; SOURCES[str(p)]=r; return r
def read(p):ref(p); return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def csv_write(p,rs,fields=None):
 with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
