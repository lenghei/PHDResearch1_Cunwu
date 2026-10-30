"""Configuration and local artifact utilities."""
import os,re,json,hashlib
from pathlib import Path

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def load_config(path):
 import yaml
 cfg=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
 for key in ['dataset_zip','checkpoint','output_root']:
  value=os.path.expandvars(cfg[key])
  if '$' in value or '%' in value:raise ValueError('Set the environment variable for '+key)
  cfg[key]=str(Path(value).expanduser().resolve())
 for key in ['dataset_zip','checkpoint']:
  if not Path(cfg[key]).is_file():raise FileNotFoundError(key)
 if cfg['checkpoint_sha256']!=sha256(cfg['checkpoint']):raise ValueError('Checkpoint SHA256 mismatch')
 return cfg
def save(path,obj):
 path=Path(path);temp=path.with_suffix(path.suffix+'.partial')
 with temp.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
 temp.replace(path)
