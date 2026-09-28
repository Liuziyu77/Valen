"""Eight-GPU resume, shared export, refrozen SFT, unfreezing and evaluation smoke."""
import json
from pathlib import Path
import subprocess
import sys

from valen.configuration import flatten_config


def launch(module, *args):
    subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
                    '--max_restarts=0', '-m', module, *args], check=True)


def main():
    root = Path('next-generation-exp')
    launch('valen.pretrain', '--config', str(root / 'configs/alignment_smoke_resume.json'),
           '--resume', str(root / 'alignment_gpu_smoke/latest.pt'))
    config = flatten_config(json.loads(Path('configs/train/dual_encoder/modernbert_dinov3b16/sft_warmup.json').read_text()))
    config.update(data='data/train_v1/smoke_dual_pretrained.jsonl', epochs=1, max_steps=1, save_every=1,
                  pretrained_shared_path=str(root / 'alignment_gpu_smoke/shared'), fusion_lr=2e-5)
    previous = None
    for stage in ('warmup', 'vision_top'):
        config.update(stage=stage, output=str(root / f'sft_transfer_smoke_{stage}'))
        path = root / f'configs/sft_transfer_smoke_{stage}.json'; path.write_text(json.dumps(config, indent=2))
        launch('valen.train', '--config', str(path), *(['--initialize', previous] if previous else []))
        previous = str(Path(config['output']) / 'latest')
    launch('valen.evaluate', '--checkpoint', previous, '--data', config['data'],
           '--output', str(root / 'sft_transfer_smoke_evaluation'))
    (root / 'gpu_validation_passed.json').write_text(json.dumps({'alignment_steps': 8, 'world_size': 8,
        'pretrained_encoders': True, 'resume': True, 'shared_transfer': True, 'sft_stages': ['warmup', 'vision_top'],
        'evaluation': str(root / 'sft_transfer_smoke_evaluation/metrics.json')}, indent=2))


if __name__ == '__main__':
    main()
