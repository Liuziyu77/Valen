"""Measure one-image parallel requests; synthetic combinations are not accuracy tests."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import statistics
import time

import torch

from valen.evaluation.inference import predict
from valen.modeling.factory import build_compiler, build_model, normalize_model_config
from valen.training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data/eval_v1/train.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--dtype', choices=['bf16', 'fp32'])
    parser.add_argument('--report-only', action='store_true', help='Record numerical drift without failing its tolerance check')
    args = parser.parse_args()
    config = normalize_model_config(json.loads((Path(args.checkpoint) / 'config.json').read_text()))
    config.update(device='cuda:0', gradient_checkpointing=False)
    if args.dtype:
        config['dtype'] = args.dtype
    # Explicitly disable reduced-mantissa FP32 kernels for the isolation check.
    # CUDA defaults and launcher settings can otherwise alter this comparison.
    if config['dtype'] == 'fp32':
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.cuda.set_device(0)
    model = build_model(config); load_checkpoint(args.checkpoint, model); model.eval()
    compiler = build_compiler(config, Path(args.data).parent)
    with Path(args.data).open() as stream:
        records = [json.loads(line) for line in stream]
    pool = [(i, q) for i, record in enumerate(records) for q in record['request']['questions'].values()]
    chosen = [next(i for i, (_, q) in enumerate(pool) if q['type'] == kind) for kind in ('choice', 'noul', 'score')]
    chosen += random.Random(42).sample([i for i in range(len(pool)) if i not in chosen], 253)
    questions = [pool[i][1] for i in chosen]
    row = deepcopy(records[0]); row['targets'] = {}
    measurements = []
    full = None
    for n in (1, 16, 64, 256):
        row['request']['questions'] = {f'q{i}': deepcopy(q) for i, q in enumerate(questions[:n])}
        started = time.perf_counter(); compiled = compiler.compile(row)
        compile_seconds = time.perf_counter() - started
        for _ in range(2):
            predict(model, compiled)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(5):
            torch.cuda.synchronize(); started = time.perf_counter()
            result = predict(model, compiled)
            torch.cuda.synchronize(); times.append(time.perf_counter() - started)
        measurements.append({'questions': n, 'candidates': sum(len(q.keys) for q in compiled.questions),
            'types': dict(Counter(q['type'] for q in questions[:n])), 'compile_seconds': compile_seconds,
            'forward_median_seconds': statistics.median(times), 'forward_runs_seconds': times,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'logical_tokens': compiled.logical_tokens,
            'longest_instruction_tokens': max(map(len, compiled.instruction_tokens))})
        if n == 256:
            full = result['answers']
    differences = []
    for i in (0, 1, 2, 128, 255):
        row['request']['questions'] = {f'q{i}': deepcopy(questions[i])}
        alone = predict(model, compiler.compile(row))['answers'][f'q{i}']
        together = full[f'q{i}']
        if alone['type'] == 'noul':
            differences.append(abs(alone['noul'] - together['noul']))
        else:
            differences.extend(abs(p - together['probabilities'][k]) for k, p in alone['probabilities'].items())
    result = {'checkpoint': args.checkpoint, 'gpu': torch.cuda.get_device_name(), 'image_size': config['image_size'],
              'dtype': config['dtype'],
              'float32_matmul_precision': torch.get_float32_matmul_precision(),
              'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
              'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
              'request_profile': 'One image with mixed questions sampled from different records; performance/independence only, not accuracy',
              'measurements': measurements, 'max_independence_probability_difference': max(differences)}
    result['independence_tolerance'] = 1e-4 if config['dtype'] == 'fp32' else .01
    result['measurement_complete'] = len(full) == 256
    result['passed'] = result['measurement_complete'] and max(differences) < result['independence_tolerance']
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(output)
    assert result['measurement_complete'], result
    if not args.report_only:
        assert result['passed'], result
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
