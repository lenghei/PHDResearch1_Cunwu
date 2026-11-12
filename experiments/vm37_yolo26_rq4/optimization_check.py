"""Compare saved training states and traces from the two implementations."""
import argparse
import json
from study import *


def execute(reference_id,optimized_id):
    out=create('optimization_parity_cpu');runtime(out,cuda=False)
    import torch
    from main.training_health import model_health,same_tree
    torch.set_num_threads(2)
    folders=[run_path(identity) for identity in (reference_id,optimized_id)]
    plans=[];payloads=[];inputs=[]
    for folder in folders:
        verify_hashes(folder,('training_acceptance.json','protocol.json','training_batches.jsonl'))
        accepted=load(folder/'training_acceptance.json');checkpoint=Path(accepted['checkpoint']['path'])
        if sha256(checkpoint)!=accepted['checkpoint']['sha256']:raise ValueError('Saved checkpoint changed.')
        plans.append(load(folder/'protocol.json'));payloads.append(torch.load(checkpoint,map_location='cpu',weights_only=False))
        inputs.append({'RunID':folder.name,'acceptance':source(folder/'training_acceptance.json'),'checkpoint':source(checkpoint)})
    if plans[0]['initialization']['sha256']!=plans[1]['initialization']['sha256']:raise ValueError('Different starting state.')
    comparisons={}
    for key in ('online_model','ema'):
        left,right=(payload[key] for payload in payloads);model_health(left);model_health(right)
        a,b=left.state_dict(),right.state_dict()
        if a.keys()!=b.keys():raise ValueError('Different model structure.')
        unequal=[];maximum=0.
        for name,value in a.items():
            if not torch.equal(value,b[name]):
                unequal.append(name);maximum=max(maximum,float((value.double()-b[name].double()).abs().max()))
        comparisons[key]={'bitwise_equal':not unequal,'unequal_tensors':unequal,'maximum_absolute_difference':maximum,'tensor_count':len(a)}
    for key in ('optimizer','scheduler','scaler','updates'):
        comparisons[key]={'equal':same_tree(payloads[0][key],payloads[1][key])}
    traces=[[json.loads(line) for line in (folder/'training_batches.jsonl').read_text().splitlines()] for folder in folders]
    if len(traces[0])!=len(traces[1]):raise ValueError('Different batch populations.')
    comparisons['traces']={'anchor_order_equal':all(a['image_ids']==b['image_ids'] for a,b in zip(*traces)),
        'tensor_shapes_equal':all(a['NCHW']==b['NCHW'] for a,b in zip(*traces)),
        'loss_values_equal':all(a['loss']==b['loss'] for a,b in zip(*traces)),
        'maximum_loss_difference':max(abs(a['loss']-b['loss']) for a,b in zip(*traces)),
        'update_cadence_equal':all(a['optimizer_updates']==b['optimizer_updates'] for a,b in zip(*traces))}
    equal=all(comparisons[k]['bitwise_equal'] for k in ('online_model','ema')) and all(comparisons[k]['equal'] for k in ('optimizer','scheduler','scaler','updates'))
    status='NUMERICAL_PARITY_PASS' if equal else 'NON_IDENTICAL_STATE_REQUIRES_INTERPRETATION'
    save(out/'state_comparison.json',{'status':status,'inputs':inputs,'comparisons':comparisons,
        'file_SHA_note':'Serialized checkpoints include different run metadata. State equality is checked separately.',
        'scope':'Same-seed implementation regression, not an augmentation effect or independent training-seed replication.'},out)
    write_atomic(out/'diary.md','# Saved-state optimization regression\n\nCompared sealed model/EMA/optimizer/scheduler/scaler states and actual batch traces on CPU. No new training or inference.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'status':status,'GPU_seconds':0},out,replace=True)
    seal(out);print(out,flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reference',required=True);parser.add_argument('--optimized',required=True)
    args=parser.parse_args();execute(args.reference,args.optimized)
