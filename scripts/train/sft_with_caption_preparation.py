"""Share spare CPUs on an SFT node with resumable caption preparation."""
import argparse
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--shared')
    args = parser.parse_args()
    children = []
    try:
        if not Path('data/train_caption/train_1m.jsonl').exists():
            log = Path('next-generation-exp/data-preparation-cluster.log').open('a')
            children.append(subprocess.Popen([sys.executable, 'scripts/data/prepare_caption.py',
                '--workers', '128', '--write-workers', '16', '--credentials-config',
                '/mnt/shared-storage-user/liuziyu/petreloss.conf'], stdout=log, stderr=subprocess.STDOUT))
        command = [sys.executable, 'scripts/train/run_sft_experiment.py', '--label', args.label]
        if args.shared:
            command.extend(['--shared', args.shared])
        children.append(subprocess.Popen(command))
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError('Caption preparation or SFT failed; inspect their logs')
            time.sleep(5)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError('Child process failed')
    finally:
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
        for child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == '__main__':
    main()
