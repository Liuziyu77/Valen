"""本地模拟提交，确保不依赖真实集群。 / Exercise submission without contacting a cluster."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('script', ['submit.sh', 'submit_experiment.sh'])
def test_cluster_settings_and_paths_are_passed_as_single_arguments(tmp_path, script):
    project = tmp_path / 'project with spaces'
    scripts = project / 'scripts/train'
    scripts.mkdir(parents=True)
    for name in (script, 'cluster_env.sh'):
        shutil.copy2(ROOT / 'scripts/train' / name, scripts / name)
    fake = tmp_path / 'fake rjob'
    fake.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                    'open(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n'
                    'print("created rjob_name: test-job")\n')
    fake.chmod(0o755)
    capture = tmp_path / 'args.json'
    config = project / '.env.cluster'
    config.write_text('export RJOB_NAMESPACE="${RJOB_NAMESPACE:-local-namespace}"\n'
                      'export RJOB_CHARGED_GROUP="test-quota"\n'
                      'export RJOB_IMAGE="test-image"\n'
                      'export RJOB_MOUNTS="first:/data with spaces\nsecond:/models"\n'
                      'export VALEN_CAPTION_ROOT="/captions with spaces"\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('RJOB_', 'VALEN_'))}
    env.update(RJOB_BIN=str(fake), RJOB_NAMESPACE='override-namespace', VALEN_GPUS='2',
               VALEN_RUN_NAME='test', CAPTURE=str(capture), VALEN_CAPTION_SOURCE_MOUNT='1',
               RJOB_CAPTION_MOUNT='third:/source indices', VALEN_PYTHON='/python env/bin/python')
    run = subprocess.run(['bash', str(scripts / script), 'config with spaces.json'], env=env,
                         text=True, capture_output=True, timeout=15)
    assert run.returncode == 0, run.stderr
    args = json.loads(capture.read_text())
    assert '--namespace=override-namespace' in args
    assert '--gpu=2' in args
    assert '--mount=first:/data with spaces' in args
    assert '--mount=third:/source indices' in args
    assert 'VALEN_CAPTION_ROOT=/captions with spaces' in args
    assert 'VALEN_PYTHON=/python env/bin/python' in args
    assert args[-1] == 'config with spaces.json'
    log_root = 'artifacts' if script == 'submit.sh' else 'next-generation-exp'
    assert (project / log_root / 'jobs/test/job_id').read_text().strip() == 'test-job'


def test_missing_cluster_settings_fail_before_submission(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('RJOB_', 'VALEN_'))}
    empty = tmp_path / 'empty.env'; empty.write_text('')
    env.update(VALEN_CLUSTER_CONFIG=str(empty), VALEN_RUN_NAME='test')
    run = subprocess.run(['bash', str(ROOT / 'scripts/train/submit_experiment.sh'), '--help'],
                         env=env, text=True, capture_output=True, timeout=15)
    assert run.returncode != 0
    assert 'RJOB_NAMESPACE' in run.stderr
