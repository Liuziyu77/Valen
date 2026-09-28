"""Run the same decision SFT budget for the baseline and pretrained backbones."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from valen.configuration import flatten_config


def evaluation_ready(output):
    try:
        report = json.loads((output / 'metrics.json').read_text())
        with (output / 'predictions.jsonl').open() as stream:
            predictions = [json.loads(line) for line in stream]
        keys = {(r['record_index'], r['qid']) for r in predictions}
        return len(predictions) == len(keys) == report['questions']
    except (OSError, ValueError, KeyError):
        return False


def launch(module, *args):
    subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
                    '--max_restarts=0', '-m', module, *args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--shared')
    args = parser.parse_args()
    root = Path('next-generation-exp') / args.label; root.mkdir(parents=True, exist_ok=True)
    base = flatten_config(json.loads(Path('configs/train/dual_encoder/modernbert_dinov3b16/sft_warmup.json').read_text()))
    base.update(epochs=1, max_steps=1000000, tokens_per_step=8192, save_every=100, state_max_length=1024,
                question_max_length=2048, candidate_max_length=128, fusion_lr=2e-5)
    if args.shared:
        base['pretrained_shared_path'] = args.shared
    previous = None
    for stage in ('warmup', 'text', 'vision_top'):
        output = root / stage; output.mkdir(exist_ok=True)
        config = dict(base, stage=stage, data=f'data/train_v1/experiment_{stage}.jsonl', output=str(output))
        path = root / f'{stage}.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('Existing experiment configuration changed')
        path.write_text(json.dumps(config, indent=2))
        if not (output / 'training_complete.json').exists():
            if (output / 'latest/checkpoint.pt').exists():
                launch('valen.train', '--config', str(path), '--resume', str(output / 'latest'))
            elif previous:
                launch('valen.train', '--config', str(path), '--initialize', previous)
            else:
                launch('valen.train', '--config', str(path))
            (output / 'training_complete.json').write_text(json.dumps({'stage': stage, 'data': config['data']}))
        evaluation = output / 'evaluation'
        if not evaluation_ready(evaluation):
            launch('valen.evaluate', '--checkpoint', str(output / 'latest'), '--data', 'data/eval_v1/train.jsonl',
                   '--output', str(evaluation))
        previous = str(output / 'latest')
    ablation_data = Path('next-generation-exp/ablation/shuffled_images.jsonl')
    ablation_output = root / 'vision_top/evaluation_shuffled_images'
    if ablation_data.exists() and not evaluation_ready(ablation_output):
        launch('valen.evaluate', '--checkpoint', previous, '--data', str(ablation_data), '--output', str(ablation_output))
    if args.label == 'sft_alignment_1m':
        fast = Path('next-generation-exp/parallel_decision_benchmark_bf16.json')
        if not fast.exists() or not json.loads(fast.read_text()).get('measurement_complete'):
            subprocess.run([sys.executable, 'scripts/eval/benchmark_parallel_decisions.py',
                            '--checkpoint', previous, '--output', str(fast), '--dtype', 'bf16', '--report-only'], check=True)
        benchmark = Path('next-generation-exp/parallel_decision_benchmark.json')
        if not benchmark.exists() or not json.loads(benchmark.read_text()).get('passed'):
            subprocess.run([sys.executable, 'scripts/eval/benchmark_parallel_decisions.py',
                            '--checkpoint', previous, '--output', str(benchmark), '--dtype', 'fp32'], check=True)
    results = {stage: json.loads((root / stage / 'evaluation/metrics.json').read_text())
               for stage in ('warmup', 'text', 'vision_top')}
    (root / 'results.json').write_text(json.dumps(results, indent=2))
    completion = root / 'complete.tmp'
    completion.write_text(json.dumps({'shared': args.shared, 'training_records': 100000,
                                     'evaluation_records': 5000, 'stages': list(results)}))
    completion.replace(root / 'complete.json')


if __name__ == '__main__':
    main()
