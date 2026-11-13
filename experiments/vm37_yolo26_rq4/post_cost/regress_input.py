"""One bounded optimized-input regression against saved real training state."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from unittest.mock import patch

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST))
sys.path.insert(0, str(POST.parent))
from study import *
from safety import available_seconds, query, run_owned
from short_cost import BATCHES, PrefixComplete, ac_connected, cpu_tree, prefix_steps

PREFIX = 'vm37_yolo26_rq4_all_five_short_cost_20261112T195422_06010323'
CAP = 600
CORRECTED_TICKET = RUNS / 'input_regression_v2_ticket.json'


def compared_dataset(data):
    import numpy as np
    import torch
    from fast_input import ArithmeticPrefetchDataset
    from verify_input import value_signature, rng_signature
    reference = copy.deepcopy(data)

    class ComparedDataset(ArithmeticPrefetchDataset):
        def __getitem__(self, index):
            state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            expected = self.reference[index]
            expected_rng = rng_signature(self.reference)
            random.setstate(state[0])
            np.random.set_state(state[1])
            torch.set_rng_state(state[2])
            actual = super().__getitem__(index)
            actual_events = self.effect_events[self.checked_event_count:]
            if (value_signature(expected) != value_signature(actual)
                    or self.reference.effect_events != actual_events
                    or self.reference.buffer != self.buffer
                    or self.reference.occurrences != self.occurrences
                    or expected_rng != rng_signature(self)):
                raise ValueError('Original versus optimized training input differs.')
            self.checked_event_count = len(self.effect_events)
            self.reference.effect_events.clear()
            self.checked_samples += 1
            return actual

    value = ComparedDataset.adopt(data)
    value.reference = reference
    value.checked_samples = 0
    value.checked_event_count = 0
    return value


def prepare(input_run, unit_run):
    if CORRECTED_TICKET.exists():
        raise RuntimeError('The corrected regression attempt is already consumed.')
    out = create('optimized_input_regression')
    runtime(out, cuda=False)
    if importlib.metadata.version('ultralytics') != '8.4.210':
        raise RuntimeError('Use the frozen RQ4 environment.')
    accepted = run_path(input_run)
    unit = run_path(unit_run)
    verify_hashes(accepted, ('input_acceptance.json', 'inputs.json', 'protocol.json'))
    verify_hashes(unit, ('test_results.json',))
    if load(accepted / 'input_acceptance.json')['status'] != 'CPU_EXACT_INPUT_PASS':
        raise ValueError('CPU exact input gate failed.')
    if load(accepted / 'protocol.json')['optimized_implementation'] != 'arithmetic2':
        raise ValueError('The optimized input version was not compared.')
    if load(unit / 'test_results.json')['status'] != 'PASS':
        raise ValueError('CPU failure-injection tests failed.')
    for ref in load(accepted / 'inputs.json')['sources']:
        if sha256(Path(ref['path'])) != ref['sha256']:
            raise ValueError('Input implementation changed after CPU validation.')
    prefix = run_path(PREFIX)
    verify_hashes(prefix, ('short_measurement.json', 'final_result.json', 'protocol.json'))
    reference_state = load(prefix / 'short_measurement.json')['state']
    if sha256(Path(reference_state['path'])) != reference_state['sha256']:
        raise ValueError('Reference engineering state changed.')
    plan = load(prefix / 'protocol.json')
    plan['args'].update(data=str(out / 'dataset.yaml'), project=str(out))
    plan.update(mode='INPUT_NUMERICAL_REGRESSION_ONLY', scope='ENGINEERING_ONLY_NOT_PILOT',
                stop_after_batches=BATCHES, expected=prefix_steps(BATCHES), worker_hard_seconds=CAP,
                CPU_input_gate_RunID=input_run, CPU_unit_RunID=unit_run, reference_RunID=PREFIX,
                reference_state=reference_state, completed_epochs=0, scientific_training_arm=False,
                performance_timing='Shadow reference computation is included; not a throughput benchmark.')
    plan['normalization_verifier'] = 'V2: original DetectionTrainer preprocessing on the same GPU; CPU normalization is diagnostic only.'
    plan['previous_failed_verifier_RunID'] = 'vm37_yolo26_rq4_optimized_input_regression_20261113T100530_a0942530'
    plan['corrected_attempt_limit'] = 1
    files = [Path(__file__), POST / 'fast_input.py', POST / 'input_ops.py', POST / 'verify_input.py',
             POST / 'profile_input.py', POST / 'short_cost.py', POST / 'safety.py', HERE / 'study.py',
             HERE / 'training.py', HERE / 'TARGETED_POLICY.json', HERE.parent / 'cv261109/generator.py',
             PUBLIC / 'code/main/training_data.py', PUBLIC / 'code/main/training_effects.py',
             PUBLIC / 'code/main/training_health.py']
    plan['sources'] = {str(path.relative_to(PUBLIC)): sha256(path) for path in files}
    initial = Path(plan['initialization']['path'])
    if sha256(initial) != plan['initialization']['sha256']:
        raise ValueError('Native10 initialization changed.')
    import torch
    from main.training_health import model_health
    payload = torch.load(initial, map_location='cpu', weights_only=False)
    model_health(payload['model'])
    safety = load(prefix / 'safety_policy.json')
    gpu = query()
    if (gpu['temperature_C'] > safety['preflight_maximum_temperature_C']
            or gpu['free_MiB'] < safety['preflight_minimum_free_MiB'] or gpu['utilization_percent'] > 10):
        raise RuntimeError('Thermal, memory or idle gate failed.')
    if available_seconds(load(RUNS / 'compute_allocation.json')) < CAP + 60:
        raise RuntimeError('The unchanged allocation cannot admit the finite regression.')
    ac_connected()
    save(out / 'protocol.json', plan, out)
    save(out / 'safety_policy.json', safety, out)
    save(out / 'admission.json', {'status': 'ONE_NUMERICAL_REGRESSION_ADMITTED', 'GPU_before': gpu,
         'available_seconds': available_seconds(load(RUNS / 'compute_allocation.json')),
         'original_deadline_UTC': load(RUNS / 'compute_allocation.json')['absolute_deadline_UTC'],
         'CPU_gate': source(accepted / 'input_acceptance.json'), 'CPU_unit': source(unit / 'test_results.json'),
         'budget_reset': False, 'pilot20': 'NOT_ADMITTED', 'formal100': 'RESOURCE_BLOCKED'}, out)
    (out / 'source_snapshots').mkdir()
    for path in files:
        shutil.copyfile(path, out / 'source_snapshots' / str(path.relative_to(PUBLIC)).replace('\\', '__').replace('/', '__'))
    print(out, flush=True)


def worker(out):
    plan = load(out / 'protocol.json')
    if load(out / 'execution_ticket.json')['protocol_SHA256'] != sha256(out / 'protocol.json'):
        raise PermissionError('Execution ticket differs.')
    runtime(out, cuda=True)
    import cv2
    import numpy as np
    import torch
    import yaml
    from portable import renderer
    from post80_loader import read_only_sources
    from training import trainer_class, verify_input_preservation
    from main.labels import VIS_NAMES
    from main.training_health import model_health, cpu_state, finite_tree, same_tree
    from ultralytics.utils import callbacks
    renderer()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    trainer = None
    status = 'FAILED'
    try:
        for relative, expected in plan['sources'].items():
            if sha256(PUBLIC / relative) != expected:
                raise ValueError('Frozen source changed.')
        protected = [plan['initialization']]
        for rows, path_key, sha_key, labels, id_key in (
            (plan['train'], 'image_path', 'image_SHA256', plan['train_labels'], 'image_id'),
            (plan['development'], 'loader_path', 'source_image_SHA256', plan['val_labels'], 'stem')):
            for row in rows:
                for path, expected in ((row[path_key], row[sha_key]), (labels[row[id_key]], row['derived_label_SHA256'])):
                    if sha256(Path(path)) != expected:
                        raise ValueError('Frozen input changed.')
                    protected.append({'path': path, 'sha256': expected})
        write_atomic(out / 'train_images.txt', '\n'.join(row['image_path'] for row in plan['train']) + '\n', out)
        write_atomic(out / 'val_images.txt', '\n'.join(row['loader_path'] for row in plan['development']) + '\n', out)
        write_atomic(out / 'dataset.yaml', yaml.safe_dump({'train': str(out / 'train_images.txt'),
            'val': str(out / 'val_images.txt'), 'names': dict(enumerate(VIS_NAMES))}), out)
        Parent = trainer_class(plan, out)
        class Trainer(Parent):
            def build_dataset(self, *args, **kwargs):
                value = super().build_dataset(*args, **kwargs)
                return compared_dataset(value) if value.augment else value

            def preprocess_batch(self, batch):
                from ultralytics.models.yolo.detect.train import DetectionTrainer
                if self.args.multi_scale:
                    raise ValueError('The frozen regression has multi_scale disabled.')
                cpu_normalized = batch['img'].float() / 255
                expected = DetectionTrainer.preprocess_batch(self, {'img': batch['img'].clone()})['img']
                actual = super().preprocess_batch(batch)
                host = actual['img'].detach().cpu()
                if not torch.equal(actual['img'], expected):
                    raise ValueError('Actual FP32 network input differs.')
                with (out / 'tensor_checks.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'position': len(self.rows), 'NCHW': list(host.shape),
                        'FP32': True, 'SHA256': hashlib.sha256(host.numpy().tobytes()).hexdigest(),
                        'original_same_device_FP32_exact': True,
                        'CPU_GPU_normalization_max_abs_difference': float((host-cpu_normalized).abs().max())}) + '\n')
                return actual

            def batch_end(self, instance):
                super().batch_end(instance)
                self.train_loader.dataset.checked_event_count = 0
                if len(self.rows) == BATCHES:
                    raise PrefixComplete()

        original_backward = torch.autograd.backward
        def backward(*args, **kwargs):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            tick = time.perf_counter()
            value = original_backward(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            trainer.backward_count += 1
            trainer.backward_seconds = time.perf_counter() - tick
            trainer.backward_CUDA_ms = begin.elapsed_time(end)
            return value

        with patch.object(callbacks, 'add_integration_callbacks', lambda instance: None), \
             patch.object(torch.autograd, 'backward', backward), read_only_sources(out):
            trainer = Trainer(overrides=plan['args'])
            try:
                trainer.train()
            except PrefixComplete:
                pass
            else:
                raise RuntimeError('Bounded regression boundary was not reached.')
        if (len(trainer.rows) != BATCHES or trainer.forward_count != BATCHES or trainer.backward_count != BATCHES
                or trainer.rq4_optimizer_updates != 128 or trainer.ema.updates != 128):
            raise ValueError('Regression update counts differ.')
        model_health(trainer.model, gradients=True)
        model_health(trainer.ema.ema)
        payload = {'online_state': cpu_state(trainer.model), 'EMA_state': cpu_state(trainer.ema.ema),
                   'optimizer': cpu_tree(trainer.optimizer.state_dict()), 'scheduler': cpu_tree(trainer.scheduler.state_dict()),
                   'scaler': cpu_tree(trainer.scaler.state_dict())}
        reference = torch.load(plan['reference_state']['path'], map_location='cpu', weights_only=False)
        checks = {key: same_tree(value, reference[key]) for key, value in payload.items()}
        old_rows = [json.loads(line) for line in (run_path(PREFIX) / 'training_batches.jsonl').read_text().splitlines()]
        fields = ('image_ids', 'loss', 'loss_items', 'optimizer_updates', 'EMA_updates', 'accumulation', 'NCHW')
        checks['historical_training_trace_exact'] = all(all(a[key] == b[key] for key in fields)
                                                       for a, b in zip(old_rows, trainer.rows))
        checks['actual_inputs_exact_samples'] = trainer.train_loader.dataset.checked_samples == BATCHES * 2
        if not finite_tree(payload):
            raise FloatingPointError('Nonfinite engineering state.')
        temporary = out / 'engineering_state.partial'
        torch.save(payload, temporary)
        checks['FP32_save_reload_exact'] = same_tree(payload, torch.load(temporary, map_location='cpu', weights_only=False))
        temporary.rename(out / 'engineering_state.pt')
        verify_input_preservation(out, protected)
        status = 'GPU_NUMERICAL_EXACT_PASS' if all(checks.values()) else 'NUMERICAL_NONIDENTICAL'
        save(out / 'numerical_comparison.json', {'status': status, 'checks': checks,
             'reference': plan['reference_state'], 'new_state': source(out / 'engineering_state.pt'),
             'batches': len(trainer.rows), 'optimizer_updates': trainer.rq4_optimizer_updates,
             'EMA_updates': trainer.ema.updates, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
             'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'completed_epochs': 0,
             'AP': 'NOT_EVALUATED', 'training_effectiveness': 'NOT_EVALUATED'}, out)
    except BaseException:
        save(out / 'failure.json', {'traceback': traceback.format_exc(), 'automatic_retry': False}, out)
    finally:
        if trainer is not None and hasattr(trainer, 'train_loader'):
            data = trainer.train_loader.dataset
            if hasattr(data, 'close'):
                data.close()
    save(out / 'worker_result.json', {'status': status, 'worker_seconds': time.perf_counter() - started}, out)


def execute(out):
    plan = load(out / 'protocol.json')
    if (out / 'execution_ticket.json').exists() or CORRECTED_TICKET.exists():
        raise RuntimeError('One regression attempt only; no retry.')
    ac_connected()
    save(out / 'execution_ticket.json', {'protocol_SHA256': sha256(out / 'protocol.json'),
         'UTC': utc().isoformat(), 'retry': False}, out)
    save(CORRECTED_TICKET, {'RunID': out.name, 'protocol_SHA256': sha256(out / 'protocol.json'),
         'corrected_attempt_limit': 1, 'retry': False}, RUNS)
    cap = budget_start(out, CAP, 'FULL_DATA_COST')
    begin = time.perf_counter()
    error = None
    supervisor = None
    try:
        supervisor = run_owned([sys.executable, '-B', '-X', 'utf8', str(Path(__file__)), '--worker', out.name],
            out, cap, load(out / 'safety_policy.json'), dict(os.environ, CUDA_VISIBLE_DEVICES='0',
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true'))
    except BaseException:
        error = traceback.format_exc()
    result = load(out / 'worker_result.json') if (out / 'worker_result.json').exists() else {'status': 'INTERRUPTED'}
    status = result['status']
    if error or supervisor is None or supervisor['returncode'] or supervisor['stop_reason']:
        status = 'STOPPED_WITH_LIMITATIONS'
    charge = time.perf_counter() - begin
    budget_finish(out, charge, status)
    telemetry = [json.loads(line) for line in (out / 'gpu_telemetry.jsonl').read_text().splitlines()] if (out / 'gpu_telemetry.jsonl').exists() else []
    save(out / 'regression_result.json', {'status': status, 'worker': result, 'supervisor': supervisor,
         'error': error, 'charged_seconds': charge, 'budget_reset': False,
         'maximum_sampled_temperature_C': max((row['temperature_C'] for row in telemetry), default=None),
         'maximum_sampled_memory_MiB': max((row['used_MiB'] for row in telemetry), default=None),
         'available_seconds_after': available_seconds(load(RUNS / 'compute_allocation.json')),
         'full_epoch_repeated': False, 'pilot20': 'NOT_ADMITTED'}, out)
    save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT,
         'mode': 'INPUT_NUMERICAL_REGRESSION', 'status': status, 'completed_epochs': 0}, out, replace=True)
    write_atomic(out / 'diary.md', '# Optimized-input numerical regression\n\nOne bounded128-batch run from '
         'the frozen initialization. Compared actual inputs to a serial shadow and saved model/EMA/optimizer '
         'states to the historical prefix. No epoch, AP, pilot or robustness-benefit result.\n', out)
    seal(out)
    print(json.dumps({'RunID': out.name, 'status': status, 'charged_seconds': charge}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', nargs=2, metavar=('INPUT_RUN', 'UNIT_RUN'))
    parser.add_argument('--execute')
    parser.add_argument('--worker')
    args = parser.parse_args()
    if args.prepare:
        prepare(*args.prepare)
    elif args.execute:
        execute(run_path(args.execute))
    elif args.worker:
        worker(run_path(args.worker))
    else:
        parser.error('Prepare a CPU-accepted regression, or execute its single ticket.')
