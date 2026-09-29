import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def monitor(tmp_path, monkeypatch, job, exit_code):
    spec = importlib.util.spec_from_file_location('experiment_monitor', Path(__file__).resolve().parents[2] / 'scripts/train/watch_experiments.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ROOT = tmp_path
    root = tmp_path / 'next-generation-exp'; registry = root / 'monitor-registry'
    registry.mkdir(parents=True)
    path = registry / 'test.json'; path.write_text(json.dumps(job))
    logs = tmp_path / job['log_dir']; logs.mkdir(parents=True)
    (logs / 'status.log').write_text(f'EXIT_CODE={exit_code}')
    monkeypatch.setattr('sys.argv', ['watch', '--interval', '0'])
    def finish_iteration():
        (root / 'STOP_MONITOR').touch()
        return 0
    module.time = SimpleNamespace(monotonic=finish_iteration)
    return module, path


def test_zero_exit_without_completion_is_not_success(tmp_path, monkeypatch):
    job = {'name': 'test', 'job_id': 'test-1', 'log_dir': 'logs', 'kind': 'pretrain', 'output': 'output', 'attempt': 2}
    module, path = monitor(tmp_path, monkeypatch, job, 0)
    module.main()
    assert json.loads(path.read_text())['terminal'] == 'needs_attention'


def test_recovery_retains_caption_sidecar_and_source_mount(tmp_path, monkeypatch):
    job = {'name': 'test', 'job_id': 'test-1', 'log_dir': 'logs', 'kind': 'sft_pipeline',
           'output': 'output', 'attempt': 0, 'label': 'sft_alignment_100k', 'shared': 'shared', 'prepare_caption': True}
    module, path = monitor(tmp_path, monkeypatch, job, 1)
    calls = []
    def submit(command, **kwargs):
        calls.append((command, kwargs))
        logs = tmp_path / 'next-generation-exp/jobs' / kwargs['env']['VALEN_RUN_NAME']
        logs.mkdir(parents=True); (logs / 'job_id').write_text('recovered-job')
        return SimpleNamespace(stdout='', stderr='', returncode=0)
    monkeypatch.setattr(module.subprocess, 'run', submit)
    module.main()
    restored = json.loads(path.read_text())
    assert restored['job_id'] == 'recovered-job' and restored['attempt'] == 1
    assert 'scripts/train/sft_with_caption_preparation.py' in calls[0][0]
    assert calls[0][1]['env']['VALEN_CAPTION_SOURCE_MOUNT'] == '1'


def test_finished_sft_does_not_hide_interrupted_caption_preparation(tmp_path, monkeypatch):
    job = {'name': 'test', 'job_id': 'test-1', 'log_dir': 'logs', 'kind': 'sft_pipeline',
           'output': 'output', 'attempt': 2, 'prepare_caption': True}
    module, path = monitor(tmp_path, monkeypatch, job, 0)
    (tmp_path / 'output').mkdir(); (tmp_path / 'output/complete.json').write_text('{}')
    module.main()
    assert json.loads(path.read_text())['terminal'] == 'needs_attention'


def test_caption_completion_uses_registered_output_root(tmp_path, monkeypatch):
    job = {'name': 'test', 'job_id': 'test-1', 'log_dir': 'logs', 'kind': 'sft_pipeline',
           'output': 'output', 'attempt': 2, 'prepare_caption': True, 'caption_root': 'custom-caption'}
    module, path = monitor(tmp_path, monkeypatch, job, 0)
    (tmp_path / 'output').mkdir(); (tmp_path / 'output/complete.json').write_text('{}')
    caption = tmp_path / 'custom-caption/train_caption'; caption.mkdir(parents=True)
    (caption / 'train_1m.jsonl').touch()
    module.main()
    assert json.loads(path.read_text())['terminal'] == 'succeeded'
