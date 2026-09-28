"""Check registered experiment jobs every 30 minutes and resume failed training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RJOB = '/mnt/shared-storage-user/liuziyu/miniconda3/envs/lmm_xc/bin/rjob'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=int, default=1800)
    args = parser.parse_args()
    root = ROOT / 'next-generation-exp'
    registry = root / 'monitor-registry'
    registry.mkdir(exist_ok=True)
    while not (root / 'STOP_MONITOR').exists():
        for path in sorted(registry.glob('*.json')):
            job = json.loads(path.read_text())
            if job.get('terminal'):
                continue
            logs = ROOT / job['log_dir']
            status = logs / 'status.log'
            result = {'time_utc': datetime.now(timezone.utc).isoformat(), 'job': job['job_id'], 'name': job['name']}
            failed = False
            completion = ROOT / job.get('output', '') / 'complete.json'
            completion_ready = completion.exists() and (not job.get('prepare_caption') or
                                                        (ROOT / 'data/train_caption/train_1m.jsonl').exists())
            if status.exists():
                result['exit_status'] = status.read_text().strip()
                failed = result['exit_status'] != 'EXIT_CODE=0'
                if not failed and job['kind'] in ('pretrain', 'sft_pipeline') and not completion_ready:
                    failed = True
                    result['failure_reason'] = 'Process exited without its training completion marker'
                if not failed:
                    job['terminal'] = 'succeeded'
            else:
                try:
                    check = subprocess.run([RJOB, 'list', job['job_id']], capture_output=True, text=True, timeout=45)
                    result['scheduler'] = (check.stdout + check.stderr)[-5000:]
                    failed = any(word in result['scheduler'] for word in (': Failed', ': FAILED', ': Terminated', ': Stopped'))
                    if ': Succeeded' in result['scheduler'] and job['kind'] in ('pretrain', 'sft_pipeline'):
                        failed = not completion_ready
                        if not failed:
                            job['terminal'] = 'succeeded'
                except subprocess.TimeoutExpired:
                    result['scheduler'] = 'status query timed out; training was not interrupted'
            metrics = ROOT / job.get('output', '') / 'metrics.jsonl'
            if job['kind'] == 'sft_pipeline':
                candidates = list((ROOT / job['output']).glob('*/metrics.jsonl'))
                if candidates:
                    metrics = max(candidates, key=lambda p: p.stat().st_mtime)
            if metrics.is_file():
                result['metrics_file'] = str(metrics.relative_to(ROOT))
                result['metrics_age_seconds'] = round(time.time() - metrics.stat().st_mtime)
                with metrics.open('rb') as f:
                    f.seek(max(0, metrics.stat().st_size - 8192))
                    lines = f.read().splitlines()
                if lines:
                    try:
                        result['latest_metrics'] = json.loads(lines[-1])
                    except ValueError:
                        pass
            if failed:
                attempt = job.get('attempt', 0)
                checkpoint = ROOT / job.get('output', '') / ('latest.pt' if job['kind'] == 'pretrain' else 'latest/checkpoint.pt')
                resumable = checkpoint.exists() if job['kind'] in ('pretrain', 'sft') else job['kind'] == 'sft_pipeline'
                if resumable and attempt < 2:
                    # Atomic checkpoint writers ensure latest is either old or complete.
                    name = job['name'].split('-recovery-')[0] + f'-recovery-{attempt + 1}'
                    module = 'valen.pretrain' if job['kind'] == 'pretrain' else 'valen.train'
                    resume = checkpoint if job['kind'] == 'pretrain' else checkpoint.parent
                    command = ['bash', 'scripts/train/submit_experiment.sh', '-m', 'torch.distributed.run',
                               '--standalone', '--nproc_per_node=8', '--max_restarts=0', '-m', module,
                               '--config', job.get('config', ''), '--resume', str(resume)]
                    if job['kind'] == 'sft_pipeline':
                        pipeline = 'scripts/train/sft_with_caption_preparation.py' if job.get('prepare_caption') else 'scripts/train/run_sft_experiment.py'
                        command = ['bash', 'scripts/train/submit_experiment.sh', pipeline,
                                   '--label', job['label']]
                        if job.get('shared'):
                            command.extend(['--shared', job['shared']])
                    try:
                        submitted = subprocess.run(command, cwd=ROOT, env=dict(os.environ, VALEN_RUN_NAME=name, VALEN_GPUS='8',
                                                   VALEN_CAPTION_SOURCE_MOUNT='1' if job.get('prepare_caption') else '0'),
                                                   capture_output=True, text=True, timeout=120)
                        result['recovery_submission'] = (submitted.stdout + submitted.stderr)[-5000:]
                        new_logs = Path('next-generation-exp/jobs') / name
                        id_path = ROOT / new_logs / 'job_id'
                        if submitted.returncode == 0 and id_path.exists():
                            job.update(name=name, job_id=id_path.read_text().strip(), log_dir=str(new_logs), attempt=attempt + 1)
                        else:
                            job['terminal'] = 'needs_attention'
                    except subprocess.TimeoutExpired:
                        job['terminal'] = 'needs_attention'; result['recovery_submission'] = 'submission timed out; inspect scheduler before retry'
                else:
                    job['terminal'] = 'needs_attention'
            temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(job, indent=2)); temp.replace(path)
            with (root / 'monitor.jsonl').open('a') as f:
                f.write(json.dumps(result) + '\n')
            print(json.dumps(result), flush=True)
        deadline = time.monotonic() + args.interval
        while time.monotonic() < deadline and not (root / 'STOP_MONITOR').exists():
            time.sleep(min(10, max(.01, deadline - time.monotonic())))


if __name__ == '__main__':
    main()
