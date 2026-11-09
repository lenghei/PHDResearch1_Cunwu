"""Prepare and audit VM36. This entry never executes YOLO."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from protocol import ROOT,HERE,EXPERIMENT,prepare,save,load,source
from main.pipeline import new_run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','audit'))
    p.add_argument('--run');p.add_argument('--meeting-dir',type=Path);p.add_argument('--source-run');args=p.parse_args()
    if args.mode=='prepare':
        out,logger=new_run({'output_root':str(ROOT/'datasets/local_runs'/EXPERIMENT)},EXPERIMENT,'protocol_v2')
        try:
            protocol=prepare(out)
            save(out/'preparation.json',{'status':'PLANNED','protocol_hash_file':'PROTOCOL_SHA256.txt','GPU_seconds':0},out)
            print(out,flush=True)
        except Exception:
            save(out/'failure.json',{'traceback':traceback.format_exc(),'stage':'PROTOCOL_PREPARATION'},out)
            raise
    else:
        if not args.run or Path(args.run).name!=args.run:p.error('audit requires a plain RunID.')
        out=ROOT/'datasets/local_runs'/EXPERIMENT/args.run
        if not out.is_dir():raise FileNotFoundError(out)
        from evidence import audit
        try:audit(out,args.meeting_dir,args.source_run)
        except Exception:
            save(out/'audit_failure.json',{'traceback':traceback.format_exc(),'stage':'CPU_AUDIT'},out)
            raise


if __name__=='__main__':main()
