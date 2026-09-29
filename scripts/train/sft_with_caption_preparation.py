"""Share spare CPUs on an SFT node with resumable caption preparation."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--shared')
    parser.add_argument('--source-root', default=os.environ.get('VALEN_CAPTION_SOURCE_ROOT'))
    parser.add_argument('--caption-root', default=os.environ.get('VALEN_CAPTION_ROOT', 'data'))
    parser.add_argument('--credentials-config', default=os.environ.get('VALEN_CREDENTIALS_CONFIG', str(Path.home() / 'petreloss.conf')))
    args = parser.parse_args()
    caption_ready = (Path(args.caption_root) / 'train_caption/train_1m.jsonl').exists()
    if not caption_ready and not args.source_root:
        parser.error('--source-root or VALEN_CAPTION_SOURCE_ROOT is required for caption preparation')
    children = []
    try:
        if not caption_ready:
            Path('next-generation-exp').mkdir(parents=True, exist_ok=True)
            log = Path('next-generation-exp/data-preparation-cluster.log').open('a')
            children.append(subprocess.Popen([sys.executable, 'scripts/data/prepare_caption.py',
                '--workers', '128', '--write-workers', '16', '--credentials-config', args.credentials_config,
                '--source-root', args.source_root, '--caption-root', args.caption_root], stdout=log, stderr=subprocess.STDOUT))
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
