"""CV261116: audit baseline FM_D and list missing evidence before static-gap claims."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))


def cli(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,
                   default=ROOT/'experiments/vm1_flamapy_yolo26s_validation/config.local.yaml')
    args=p.parse_args(argv)
    from main.pipeline import settings, bridge
    from main.config import sha256
    start=time.perf_counter()
    cfg=settings(args.config)
    output=(ROOT.parent/'datasets/local_runs/cv261116').resolve()
    if output.is_relative_to(ROOT.resolve()):
        p.error('Output must remain outside the public Git root.')
    output.mkdir(parents=True, exist_ok=True)
    run_id='cv261116_fmd_audit_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    folder=output/run_id
    folder.mkdir(exist_ok=False)
    try:
        value=bridge(cfg,folder)
        if value['status'] != 'PASS' or value['yolo']['valid_count'] != 9:
            raise ValueError('Existing FM_D baseline UVL count changed.')
        uvl=ROOT/'experiments/vm1_flamapy_yolo26s_validation/yolo.uvl'
        result={
            'RunID':run_id,'ExperimentID':'cv261116','status':'PARTIAL',
            'mode':'FM_D_BASELINE_AUDIT','checkpoint_id':'ck01',
            'uvl':{'file':'experiments/vm1_flamapy_yolo26s_validation/yolo.uvl',
                   'SHA256':sha256(uvl),'valid_configurations':value['yolo']['valid_count'],
                   'constraint_count':value['yolo']['constraint_count']},
            'configuration_example':{'model_features':value['model_features'],
                                     'resolution':value['resolution'],
                                     'checkpoint_id':value['checkpoint_id']},
            'training_augmentation_coverage':'NOT_VERIFIED',
            'validity_constraints':'INCOMPLETE_SUPERVISOR_FM_D',
            'per_configuration_measured_latency':'NOT_COLLECTED',
            'per_configuration_measured_memory':'NOT_COLLECTED',
            'static_gap':'NOT_EVALUATED',
            'empirical_gap':'NOT_EVALUATED',
            'scope':'Engineering inventory of existing nine-config FM_D; not the proposed full supervisor model.',
            'required_next_evidence':['Document actual training augmentation with code and version SHA',
                'Add realistic FM_D features and explicit constraints',
                'Measure latency/memory for each supported detector setting',
                'Compute static gap only from verified exhaustive coverage rules'],
            'elapsed_seconds':round(time.perf_counter()-start,3),
        }
        (folder/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n', encoding='utf-8')
        print('CV261116: baseline FM_D valid configurations:',value['yolo']['valid_count'])
        print('Static gap: NOT_EVALUATED (no evidenced exhaustive coverage rules).')
        print('Report:',folder/'result.json')
    except BaseException as exc:
        (folder/'failure.json').write_text(json.dumps({'status':'FAILED','error':repr(exc)},indent=2)+'\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    cli()
