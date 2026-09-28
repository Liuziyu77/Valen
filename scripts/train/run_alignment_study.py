"""Sequence validated 100k alignment, 1M alignment, and decision comparisons."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'next-generation-exp'


def status(phase, **fields):
    value = {'time_utc': datetime.now(timezone.utc).isoformat(), 'phase': phase, **fields}
    temp = EXP / 'study-status.tmp'; temp.write_text(json.dumps(value, indent=2)); temp.replace(EXP / 'study-status.json')
    print(json.dumps(value), flush=True)


def wait_files(phase, *paths):
    status(phase, waiting_for=[str(p) for p in paths])
    while not all((ROOT / p).exists() for p in paths):
        time.sleep(20)


def launch(key, kind, output, command, **fields):
    registry = EXP / 'monitor-registry' / f'{key}.json'
    if registry.exists():
        old = json.loads(registry.read_text())
        if old.get('terminal') == 'needs_attention':
            raise RuntimeError(f'{key} needs diagnosis; see monitor.jsonl')
        return registry
    name = 'valen-' + key.replace('_', '-') + '-' + datetime.now().strftime('%m%d-%H%M%S')
    result = subprocess.run(['bash', 'scripts/train/submit_experiment.sh', *command], cwd=ROOT,
                            env=dict(os.environ, VALEN_RUN_NAME=name, VALEN_GPUS='8'), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    logs = Path('next-generation-exp/jobs') / name
    job = {'name': name, 'job_id': (ROOT / logs / 'job_id').read_text().strip(), 'log_dir': str(logs),
           'kind': kind, 'output': output, 'attempt': 0, **fields}
    temp = registry.with_suffix('.tmp'); temp.write_text(json.dumps(job, indent=2)); temp.replace(registry)
    status('job_submitted', experiment=key, job=job['job_id'])
    return registry


def wait_job(registry, completion):
    while not (ROOT / completion).exists():
        job = json.loads(registry.read_text())
        if job.get('terminal') == 'needs_attention':
            raise RuntimeError(f"Job {job['job_id']} needs diagnosis")
        time.sleep(20)


def pretrain(label, initialize=None):
    config = f'next-generation-exp/configs/alignment_{label}.json'
    output = f'next-generation-exp/alignment_{label}'
    command = ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8', '--max_restarts=0',
               '-m', 'valen.pretrain', '--config', config]
    if initialize:
        command.extend(['--initialize', initialize])
    registry = launch(f'alignment_{label}', 'pretrain', output, command, config=config)
    wait_job(registry, f'{output}/complete.json')
    completion = json.loads((ROOT / output / 'complete.json').read_text())
    if not completion['training_budget_complete']:
        raise RuntimeError(f'{label} did not complete its configured training budget')
    return output


def sft(label, shared):
    output = f'next-generation-exp/{label}'
    command = ['scripts/train/run_sft_experiment.py', '--label', label, '--shared', shared]
    return launch(label, 'sft_pipeline', output, command, label=label, shared=shared)


def main():
    wait_files('waiting_for_pilot_data_and_checks', 'data/train_caption/train_100k.jsonl',
               'data/eval_caption/validation.jsonl', 'next-generation-exp/gpu_validation_passed.json')
    tests = (EXP / 'full-tests.log').read_text()
    passed = re.search(r'(\d+) passed', tests)
    if passed is None or int(passed[1]) < 113 or re.search(r'\d+ (failed|errors?)', tests):
        raise RuntimeError('Full regression tests have not passed')
    pilot = pretrain('100k')
    evaluations = [json.loads(line) for line in (ROOT / pilot / 'evaluation.jsonl').read_text().splitlines()]
    initial = next(r for r in evaluations if r['label'] == 'initial')
    final = next(r for r in reversed(evaluations) if r['label'] == 'final_comparable')
    valid = final['loss'] < initial['loss'] and max(final['image_to_text_recall_at_1'], final['text_to_image_recall_at_1']) > 3 / final['samples']
    gate = {'initial': initial, 'final_comparable': final, 'passed': valid,
            'rule': 'loss decreases and at least one Recall@1 exceeds three times random chance on the same fixed pool'}
    (EXP / 'pilot_gate.json').write_text(json.dumps(gate, indent=2))
    if not valid:
        raise RuntimeError('Pilot alignment quality gate failed; inspect objectives/data before 1M')
    pilot_sft = sft('sft_alignment_100k', f'{pilot}/shared')
    wait_job(pilot_sft, 'next-generation-exp/sft_alignment_100k/complete.json')
    wait_files('waiting_for_baseline_and_1m_data', 'next-generation-exp/sft_baseline/complete.json',
               'data/train_caption/train_1m.jsonl')
    # Preserve all downstream comparison metrics. The alignment quality gate and
    # end-to-end checks govern scale-up; no claim that pilot SFT must beat baseline.
    status('pilot_sft_complete_starting_1m')
    main_output = pretrain('1m', f'{pilot}/latest.pt')
    final_sft = sft('sft_alignment_1m', f'{main_output}/shared')
    wait_job(final_sft, 'next-generation-exp/sft_alignment_1m/complete.json')
    status('training_and_validation_complete')
    subprocess.run([str(ROOT / '.venv/bin/python'), 'scripts/eval/report_alignment_study.py'], cwd=ROOT, check=True)
    (EXP / 'STUDY_TRAINING_COMPLETE').touch()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        status('needs_attention', error=str(e))
        raise
