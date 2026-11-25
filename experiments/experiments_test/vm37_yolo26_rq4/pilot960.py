"""One admitted 960 cost check, then a budget-gated matched pilot."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import local_recovery_v1 as recovery

old = recovery.baseline
study = recovery.study
save, load = old.save, old.load
SOURCE = Path(__file__).resolve()
PREPARED = 'vm37_yolo26_rq4_local_recovery_v1_cpu_20261113T232417_d06c615b'


def check(parent):
    for path, digest in load(parent / 'execution_freeze.json')['files'].items():
        if study.sha256(Path(path)) != digest:
            raise ValueError('Execution source changed: ' + path)
    old.verify_freeze(study.run_path(recovery.AUTH))
    if old.available(study.run_path(recovery.AUTH)) <= 0:
        raise PermissionError('Original budget expired or exhausted.')


def context(out):
    ticket = load(out / 'worker_ticket.json')
    parent = study.run_path(ticket['parent_RunID'])
    check(parent)
    if not out.resolve().is_relative_to(parent.resolve()):
        raise PermissionError('Worker output escapes its run.')
    if study.sha256(out / 'protocol.json') != ticket['protocol_SHA256']:
        raise ValueError('Worker plan changed.')
    if ticket.get('evaluation_stage_SHA256') and study.sha256(out / 'evaluation_stage.json') != ticket['evaluation_stage_SHA256']:
        raise ValueError('Evaluation stage changed.')
    if datetime.now(timezone.utc) >= datetime.fromisoformat(load(out / 'WORKER_LEASE.json')['deadline_UTC']):
        raise PermissionError('Worker lease expired.')
    return load(out / 'protocol.json'), parent


def worker(out, mode):
    study.runtime(out, cuda=True)
    plan, parent = context(out)
    if mode == 'train':
        Factory = recovery.trainer_factory(plan, out)
        with patch.object(old, 'worker_context', context), patch.object(old.v3, 'trainer_factory', lambda *_: Factory):
            return old.train_worker(out)
    with patch.object(old, 'worker_context', context):
        return old.evaluate_worker(out)


def run_stage(parent, out, stage, cap, mode='train'):
    check(parent)
    authorization = study.run_path(recovery.AUTH)
    if cap + 120 > old.available(authorization):
        raise PermissionError('Stage and stop reserve exceed the original budget.')
    old.cooldown(out)
    save(out / 'resource_before.json', old.resource_check())
    deadline = min(datetime.now(timezone.utc) + timedelta(seconds=cap),
                   datetime.fromisoformat(load(authorization / 'authorization.json')['absolute_expiry_UTC']))
    save(out / 'WORKER_LEASE.json', {'deadline_UTC': deadline.isoformat(), 'cap_seconds': cap})
    save(out / 'worker_ticket.json', {'parent_RunID': parent.name,
        'protocol_SHA256': study.sha256(out / 'protocol.json'),
        'evaluation_stage_SHA256': study.sha256(out / 'evaluation_stage.json') if mode == 'evaluate' else None})
    command = [sys.executable, '-B', '-X', 'utf8', str(SOURCE), '--worker', str(out), '--mode', mode]
    save(out / 'command.json', command)
    started = time.perf_counter()
    result = None
    try:
        result = old.run_owned(command, out, cap, load(old.bundle() / 'safety_policy.json'),
            dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2'))
        diagnosis = recovery.stage_diagnosis(out, result, mode)
        save(out / 'stage_diagnosis.json', diagnosis)
        if diagnosis['status'] != 'STAGE_COMPLETED':
            raise RuntimeError(diagnosis['primary_reason'])
    finally:
        save(authorization / 'charges' / (parent.name + '_' + stage + '.json'), {
            'RunID': parent.name, 'stage': stage, 'stage_path': str(out),
            'charged_seconds': time.perf_counter() - started, 'hard_cap': cap,
            'supervisor': result, 'finished_UTC': datetime.now(timezone.utc).isoformat()})
    return result


def finish(parent, status, error=None):
    stages = []
    for folder in sorted(parent.iterdir()):
        if not folder.is_dir() or not (folder / 'protocol.json').exists():
            continue
        item = {'stage': folder.name}
        for name in ('safety_supervisor.json', 'training_acceptance.json', 'stage_diagnosis.json', 'evaluation_result.json'):
            if (folder / name).exists():
                item[name] = load(folder / name)
        if (folder / 'gpu_telemetry.jsonl').exists():
            values = [json.loads(line) for line in (folder / 'gpu_telemetry.jsonl').read_text().splitlines()]
            item['maximum_temperature_C'] = max((v['temperature_C'] for v in values), default=None)
            item['peak_NVML_used_MiB'] = max((v['used_MiB'] for v in values), default=None)
        if (folder / 'training_batches.jsonl').exists():
            item['completed_training_batches'] = sum(1 for _ in (folder / 'training_batches.jsonl').open())
        stages.append(item)
    budget = recovery.budget_snapshot()
    save(parent / 'execution_result.json', {'status': status, 'error': error, 'stages': stages, 'budget': budget,
        'scope': '960 ten-epoch single-seed DEVELOPMENT_ONLY; not formal100 or full empirical Gap.', 'automatic_retry': False})
    meta = load(parent / 'run.json')
    meta.update(status=status, finished_UTC=datetime.now(timezone.utc).isoformat())
    study.save(parent / 'run.json', meta, parent, replace=True)
    report = study.ROOT / 'datasets/reportcn' / ('vm37_yolo26_rq4_960配对实际结果_' + parent.name[-8:] + '.md')
    text = ['# YOLO26s 960配对Pilot实际结果', '', '**' + status + '**', '',
            'RunID：`' + parent.name + '`。', '预算来源：`' + recovery.AUTH + '`；没有新建或延长额度。',
            '原始参数、SHA、命令、日志与逐阶段证据：' + str(parent), '',
            'close_mosaic=10且总训练10轮，因此epoch0关闭Mosaic；原1280/100轮正式方案不变。',
            '室温、风扇转速及未提供OEM设置均为UNKNOWN。']
    if error:
        text += ['', '停止原因：`' + error + '`。不自动重试。']
    for item in stages:
        accepted = item.get('training_acceptance.json', {})
        text += ['', '## ' + item['stage'],
                 '完成epoch：' + str(accepted.get('completed_epochs', 0)) +
                 '；已记录完整batch：' + str(item.get('completed_training_batches', 'NOT_APPLICABLE')),
                 '温度峰值°C：' + str(item.get('maximum_temperature_C', 'NOT_RECORDED')) +
                 '；NVML显存峰值MiB：' + str(item.get('peak_NVML_used_MiB', 'NOT_RECORDED')),
                 '监督器墙钟秒：' + str(item.get('safety_supervisor.json', {}).get('wall_seconds', 'NOT_RECORDED'))]
    index = parent / 'evaluation/result_index.json'
    if index.exists():
        text += ['', '## 科学评价与配对区间',
                 '16项配置的AP50、AP50:95、尺寸AP、AR和延迟详见下列原始索引及逐配置metrics.json；不混用库内验证指标。',
                 str(index), str(parent / 'evaluation/paired_contrasts.json'),
                 '以下为原始可追溯指标，不据此推断训练seed显著性或独立泛化。']
        for row in load(index):
            text += ['', '```json', json.dumps({'configuration': row['configuration'],
                        'metrics': load(parent / 'evaluation' / row['folder'] / 'metrics.json')}, ensure_ascii=False), '```']
        text += ['', '配对差异、相对Normal退化损失变化及200次图像Bootstrap区间：', '```json',
                 json.dumps(load(parent / 'evaluation/paired_contrasts.json'), ensure_ascii=False), '```']
    else:
        text += ['', '科学AP50/AP50:95、Tiny/Small AP、鲁棒性变化、Bootstrap区间及科学推理延迟：**NOT_EVALUATED**。',
                 '不使用工程单轮或历史AP代替本次配对结果。']
    text += ['', '原额度累计已计费秒：' + str(budget['charged_seconds']),
             '原额度剩余秒：' + str(budget['remaining_seconds']),
             '固定截止：2026-11-20 21:20:36 UTC。',
             '单seed、开发集选模暴露、缩短日程和仅四个退化上下文仍限制结论；没有独立测试或完整Gap缩减结论。']
    report.write_text('\n'.join(text) + '\n', encoding='utf-8')
    save(parent / 'final_report.json', {'path': str(report), 'sha256': study.sha256(report)})
    (parent / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>YOLO26s 960 pilot</h1><p>' +
        status + '</p><a href="execution_result.json">Measured evidence</a>', encoding='utf-8')
    study.seal(parent)
    print('PILOT960_FINISHED', parent, status, flush=True)


def execute(confirmation_file):
    parent = study.create('pilot960_execution')
    study.runtime(parent, cuda=False)
    authorization = study.run_path(recovery.AUTH)
    old.verify_freeze(authorization)
    previous = study.run_path(PREPARED)
    for ref in load(previous / 'inputs.json')['files']:
        if study.sha256(Path(ref['path'])) != ref['sha256']:
            raise ValueError('Accepted source changed.')
    if load(previous / 'test_results.json')['status'] != 'PASS':
        raise ValueError('Accepted CPU gate failed.')
    confirmation = load(confirmation_file)
    required = ('hard_ventilated_surface', 'unobstructed_intake_exhaust', 'fans_normal', 'reliable_AC_adapter')
    if not all(confirmation.get(k) is True for k in required):
        raise PermissionError('Explicit current physical confirmation is missing.')
    save(parent / 'physical_confirmation.json', confirmation)
    save(parent / 'author_authorization.json', {'source': study.source(confirmation_file),
        'original_budget': recovery.AUTH, 'new_allocation': False, 'continuous_execution': True,
        'stability_attempts': 1, 'no_automatic_retry': True, 'formal100_modified': False})
    save(parent / 'execution_freeze.json', {'files': {str(p): study.sha256(p) for p in
        [SOURCE, Path(recovery.__file__), *old.source_paths()]}})
    spec = recovery.candidate_protocol('B')
    save(parent / 'pilot10_protocol.json', spec)
    lock = study.RUNS / '.pilot10_gpu.lock'
    if lock.exists() or (study.RUNS / '.gpu.lock').exists():
        raise RuntimeError('Existing GPU worker lock.')
    old.resource_check()
    if old.available(authorization) < 3600 + 15000 + 1800:
        raise PermissionError('Insufficient original budget for cost admission and evaluation reserve.')
    save(lock, {'RunID': parent.name, 'PID': os.getpid()})
    status, error = 'PARTIAL_WITH_LIMITATIONS', None
    try:
        population = old.v3.base.resolve_data(old.bundle(), study.ROOT / 'datasets/VisDrone', verify=False)
        admission = parent / 'stability_cost'
        admission.mkdir()
        plan = recovery.make_plan(spec, admission, 'CONTROL', population)
        plan.update(execution_epochs=1, stop_after_completed_epochs=1,
                    expected=study.expected_steps(6471, 10, stop_epochs=1),
                    scope='ENGINEERING_ONLY_DEVELOPMENT_ONLY', selection='EPOCH1_ENGINEERING_ONLY_NEVER_REUSE')
        save(admission / 'protocol.json', plan)
        observed = run_stage(parent, admission, 'stability_cost', 3600)
        measured = observed['wall_seconds']
        caps = {'CONTROL': max(18000, int(measured * 10 * 1.35)), 'ALL_FIVE': 68000,
                'evaluation': 15000, 'reserve': 1800}
        save(parent / 'measured_resource_admission.json', {'CONTROL_single_full_epoch_observed_seconds': measured,
            'CONTROL_10_epoch_cap_extrapolated': caps['CONTROL'], 'ALL_FIVE_cap_historical_reference_not_960_measurement': 68000,
            'reference_note': 'Historical 1280 ALL_FIVE full epoch ~5430s; 10x plus25% uncertainty ceiling. No pixel-ratio speed assumption.',
            'required_caps': caps, 'remaining_seconds': old.available(authorization),
            'all_required_caps_fit': sum(caps.values()) <= old.available(authorization),
            'short_check_does_not_prove_long_thermal_stability': True})
        if sum(caps.values()) > old.available(authorization):
            raise PermissionError('RESOURCE_BLOCKED_COMPLETE_MATCHED_WORK_DOES_NOT_FIT')
        for arm in ('CONTROL', 'ALL_FIVE'):
            required = caps[arm] + caps['evaluation'] + caps['reserve'] + (caps['ALL_FIVE'] if arm == 'CONTROL' else 0)
            if old.available(authorization) < required:
                raise PermissionError('RESOURCE_BLOCKED_NO_UNPAIRED_LAUNCH')
            folder = parent / ('YOLO26s_seed42_' + arm)
            folder.mkdir()
            save(folder / 'protocol.json', recovery.make_plan(spec, folder, arm, population))
            run_stage(parent, folder, arm, caps[arm])
        evaluation = parent / 'evaluation'
        evaluation.mkdir()
        save(evaluation / 'protocol.json', {'scope': spec['id'], 'completed_arms_required': 2})
        save(evaluation / 'evaluation_stage.json', {'contexts': load(old.bundle() / 'evaluation_manifest.json')['contexts']})
        run_stage(parent, evaluation, 'evaluation', caps['evaluation'], 'evaluate')
        if len(load(evaluation / 'result_index.json')) != 16:
            raise ValueError('Incomplete frozen evaluation matrix.')
        status = 'PILOT960_MATCHED_COMPLETE_DEVELOPMENT_ONLY'
    except BaseException as exc:
        error = str(exc)
        save(parent / 'failure.json', {'error': error, 'traceback': traceback.format_exc(), 'automatic_retry': False})
    finally:
        if lock.exists() and load(lock).get('RunID') == parent.name:
            lock.unlink()
        finish(parent, status, error)
    return 0 if status == 'PILOT960_MATCHED_COMPLETE_DEVELOPMENT_ONLY' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', type=Path)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--mode', choices=('train', 'evaluate'), default='train')
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(0 if worker(args.worker.resolve(strict=True), args.mode) in ('TRAINING_COMPLETED', 'COMPLETE') else 1)
    if not args.execute:
        parser.error('An explicit authorization/confirmation file is required.')
    raise SystemExit(execute(args.execute.resolve(strict=True)))
