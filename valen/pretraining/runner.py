"""图文对齐训练与断点恢复。 / Image-caption alignment training with checkpoint resume."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from valen.modeling.dual_encoder.encoders.modernbert import ModernBertAdapter
from valen.modeling.dual_encoder.encoders.dinov3 import DINOv3Adapter
from valen.modeling.dual_encoder.transfer import export_shared, file_hash
from .data import CaptionDataset
from .model import AlignmentModel


def set_stage(model, stage, config):
    """按阶段解冻编码器顶部层。 / Unfreeze top encoder layers by stage."""
    core = model.core
    core.text_encoder.requires_grad_(False); core.vision_encoder.requires_grad_(False)
    if stage in ('text', 'vision_top'):
        ModernBertAdapter.unfreeze(core.text_encoder, config.get('text_unfreeze_layers', 6))
    if stage == 'vision_top':
        DINOv3Adapter.unfreeze(core.vision_encoder, config.get('vision_unfreeze_layers', 2))


def optimizer_groups(model, config):
    groups = {'new': [], 'text': [], 'vision': []}
    for name, p in model.named_parameters():
        if p.requires_grad:
            key = 'text' if name.startswith('core.text_encoder.') else 'vision' if name.startswith('core.vision_encoder.') else 'new'
            groups[key].append(p)
    rates = {'new': config.get('alignment_lr', 1e-4), 'text': config.get('text_lr', 1e-5), 'vision': config.get('vision_lr', 2e-6)}
    return [{'name': k, 'params': p, 'lr': rates[k], 'initial_lr': rates[k]} for k, p in groups.items() if p]


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'python': random.getstate(), 'numpy': np.random.get_state()}


def restore_rng(state):
    torch.set_rng_state(state['torch']); random.setstate(state['python']); np.random.set_state(state['numpy'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def move(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, dataset, config, device, limit=None):
    """固定评估样本；检索使用整个评估池。 / Evaluate a fixed subset; retrieval uses the full evaluated pool."""
    limit = min(limit or len(dataset), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, limit, dtype=int).tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=config.get('batch_size', 16),
                        collate_fn=dataset.collate, num_workers=config.get('eval_workers', 2),
                        generator=torch.Generator().manual_seed(917), pin_memory=device.type == 'cuda')
    was_training = model.training; model.eval()
    stats, images, texts, groups, count = {}, [], [], [], 0
    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
        for batch_index, batch in enumerate(loader):
            if len(batch['group_ids']) < 2:
                continue
            batch = move(batch, device); n = len(batch['group_ids'])
            torch.manual_seed(1234 + batch_index)
            loss, terms = model(batch, distributed=False)
            terms['loss'] = loss.detach()
            with model.core._amp():
                h, v, text, image = model.features(batch)
                # 固定错配用于对照难负例。 / Fixed mismatches complement hard negatives.
                positive = model.fuse(h, batch['attention_mask'], v, batch['visual_mask'], batch['pixel_values'])[:, 0]
                negative = model.fuse(h.roll(1, 0), batch['attention_mask'].roll(1, 0), v,
                                      batch['visual_mask'], batch['pixel_values'])[:, 0]
                random_logits = model.itm_head(torch.cat((positive, negative)))
                random_labels = torch.cat((torch.ones(n, device=device), torch.zeros(n, device=device))).long()
                terms['random_negative_itm_accuracy'] = (random_logits.argmax(-1) == random_labels).float().mean()
            images.append(image.cpu()); texts.append(text.cpu()); groups.append(batch['group_ids'].cpu())
            changed = dict(batch, pixel_values=batch['pixel_values'].roll(1, 0), visual_mask=batch['visual_mask'].roll(1, 0))
            # 对照实验使用相同遮盖。 / Reuse identical masks for the ablation.
            torch.manual_seed(1234 + batch_index)
            _, wrong = model(changed, distributed=False)
            terms['shuffled_image_mlm'] = wrong['mlm']; terms['shuffled_image_mlm_accuracy'] = wrong['mlm_accuracy']
            for key, value in terms.items():
                stats[key] = stats.get(key, 0.) + float(value) * n
            count += n
    image, text, ids = torch.cat(images), torch.cat(texts), torch.cat(groups)
    result = {k: v / count for k, v in stats.items()}
    for name, a, b in [('image_to_text', image, text), ('text_to_image', text, image)]:
        scores = {k: 0 for k in (1, 5, 10)}
        b = b.to(device)
        for start in range(0, len(a), 256):
            logits = a[start:start + 256].to(device) @ b.T
            top = logits.topk(min(10, len(b)), dim=-1).indices.cpu()
            correct = ids[top] == ids[start:start + len(top), None]
            for k in scores:
                scores[k] += int(correct[:, :k].any(1).sum())
        result.update({f'{name}_recall_at_{k}': v / count for k, v in scores.items()})
    result['samples'] = count
    result['mlm_image_gain'] = result['shuffled_image_mlm'] - result['mlm']
    model.train(was_training)
    return result


def run(config, resume=None, initialize=None):
    """resume 恢复完整训练状态；initialize 仅加载权重。
    resume restores full training state; initialize loads weights only.
    """
    rank, world = int(os.getenv('RANK', 0)), int(os.getenv('WORLD_SIZE', 1))
    device = torch.device(f"cuda:{os.getenv('LOCAL_RANK', 0)}" if config.get('device', 'cuda') == 'cuda' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(minutes=15))
    random.seed(config.get('seed', 42) + rank); np.random.seed(config.get('seed', 42) + rank)
    torch.manual_seed(config.get('seed', 42))
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=True)
    data = CaptionDataset(config['data'], config)
    validation = CaptionDataset(config['eval_data'], config) if rank == 0 else None
    data_hash = file_hash(config['data']); eval_hash = file_hash(config['eval_data'])
    # 先向优化器和 DDP 注册后续要解冻的参数。
    # Register future trainable parameters before optimizer/DDP setup.
    build = dict(config, device=str(device), stage='vision_top')
    model = AlignmentModel(build, data.tokenizer.mask_token_id).to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config), weight_decay=config.get('weight_decay', .01))
    progress = {'epoch': 0, 'batch': 0, 'step': 0, 'samples_seen': 0}
    payload = None
    if resume:
        payload = torch.load(resume, map_location='cpu', weights_only=False)
        allowed = {'output', 'max_steps', 'eval_every', 'save_every', 'eval_limit', 'workers', 'eval_workers'}
        if {k: v for k, v in config.items() if k not in allowed} != {k: v for k, v in payload['config'].items() if k not in allowed}:
            raise ValueError('Resume configuration changed')
        if payload['world_size'] != world or payload['data_sha256'] != data_hash or payload['eval_sha256'] != eval_hash:
            raise ValueError('Resume world size or data changed')
        if payload['base_manifest'] != model.core.base_manifest:
            raise ValueError('Resume encoder identity changed')
        model.load_state_dict(payload['model']); optimizer.load_state_dict(payload['optimizer'])
        progress = payload['progress']
    elif initialize:
        payload = torch.load(initialize, map_location='cpu', weights_only=False)
        if payload['base_manifest'] != model.core.base_manifest:
            raise ValueError('Initialization encoder identity changed')
        model.load_state_dict(payload['model'])
    wrapper = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None,
                                     find_unused_parameters=True, broadcast_buffers=False) if world > 1 else model
    if resume:
        restore_rng(payload['rng'][rank])
    else:
        torch.manual_seed(config.get('seed', 42) + rank)
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True,
                                 seed=config.get('seed', 42), drop_last=True)
    batch_size = config.get('batch_size', 16)
    steps_per_epoch = len(sampler) // batch_size
    if not steps_per_epoch:
        raise ValueError('Dataset smaller than global batch')
    total_steps = steps_per_epoch * config.get('epochs', 1)
    stop = min(total_steps, config.get('max_steps', total_steps))
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    def stage_for(step):
        # 解冻比例按总更新步数计算。 / Unfreeze fractions use total update steps.
        fraction = step / total_steps
        return 'warmup' if fraction < config.get('text_unfreeze_fraction', .1) else 'text' if fraction < config.get('vision_unfreeze_fraction', .6) else 'vision_top'
    def save():
        state = rng_state()
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, state)
        else:
            states = [state]
        if rank == 0:
            payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config,
                       'progress': dict(progress), 'rng': states, 'world_size': world,
                       'base_manifest': model.core.base_manifest, 'data_sha256': data_hash, 'eval_sha256': eval_hash}
            temp = output / 'checkpoint.tmp'; torch.save(payload, temp)
            latest = output / 'latest.pt'
            if latest.exists():
                latest.replace(output / 'previous.pt')
            temp.replace(latest)
        if world > 1:
            dist.barrier()
    def validate(label, limit):
        if world > 1:
            dist.barrier()
        if rank == 0:
            result = evaluate(model, validation, config, device, limit)
            result.update(step=progress['step'], label=label, stage=stage_for(progress['step']))
            with (output / 'evaluation.jsonl').open('a') as f:
                f.write(json.dumps(result) + '\n')
            print(json.dumps({'evaluation': result}), flush=True)
        if world > 1:
            dist.barrier()
    if rank == 0:
        (output / 'run_manifest.json').write_text(json.dumps({'config': config, 'world_size': world,
            'data_sha256': data_hash, 'eval_sha256': eval_hash, 'global_batch': batch_size * world,
            'steps_per_epoch': steps_per_epoch, 'total_steps': total_steps,
            'initialization_checkpoint': initialize, 'potential_trainable_parameters': trainable_names,
            'gpus': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}, indent=2))
    model.train(); started = time.monotonic(); session_step = progress['step']
    if progress['step'] == 0:
        validate('initial', config.get('eval_limit', 1024))
    while progress['epoch'] < config.get('epochs', 1) and progress['step'] < stop:
        sampler.set_epoch(progress['epoch'])
        loader = DataLoader(data, batch_size=batch_size, sampler=sampler, collate_fn=data.collate,
            num_workers=config.get('workers', 4), drop_last=True, pin_memory=device.type == 'cuda',
            generator=torch.Generator().manual_seed(1000 + progress['epoch']))
        for batch_index, batch in enumerate(loader):
            if batch_index < progress['batch']:
                continue
            if progress['step'] >= stop:
                break
            stage = stage_for(progress['step']); set_stage(model, stage, config)
            warmup = max(1, round(total_steps * .05))
            step = progress['step']
            multiplier = min(1., (step + 1) / warmup) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / total_steps)))
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * multiplier
            optimizer.zero_grad(set_to_none=True)
            loss, terms = wrapper(move(batch, device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite alignment loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            progress.update(step=step + 1, batch=batch_index + 1, samples_seen=progress['samples_seen'] + batch_size * world)
            metrics = torch.stack([loss.detach(), *terms.values()]).float()
            if world > 1:
                dist.all_reduce(metrics); metrics /= world
            if rank == 0:
                stats = dict(zip(['loss', *terms], metrics.tolist()))
                stats.update(**progress, stage=stage, grad_norm=float(grad), elapsed_seconds=time.monotonic() - started,
                             samples_per_second=(progress['step'] - session_step) * batch_size * world / (time.monotonic() - started),
                             peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)
                with (output / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(stats) + '\n')
                if progress['step'] % 10 == 0:
                    print(json.dumps(stats), flush=True)
            if progress['step'] % config.get('save_every', 200) == 0:
                save()
            if progress['step'] % config.get('eval_every', 400) == 0:
                validate('periodic', config.get('eval_limit', 1024))
        if progress['batch'] >= steps_per_epoch:
            progress.update(epoch=progress['epoch'] + 1, batch=0)
    save()
    if config.get('final_eval_limit', 10000) != config.get('eval_limit', 1024):
        validate('final_comparable', config.get('eval_limit', 1024))
    validate('final', config.get('final_eval_limit', 10000))
    if rank == 0:
        export_shared(model.core, output / 'shared')
        if config.get('test_data'):
            test_data = CaptionDataset(config['test_data'], config)
            test_result = evaluate(model, test_data, config, device)
            test_result.update(data=config['test_data'], data_sha256=file_hash(config['test_data']), step=progress['step'])
            (output / 'test_metrics.json').write_text(json.dumps(test_result, indent=2))
        completion = output / 'complete.tmp'
        completion.write_text(json.dumps({'progress': progress, 'training_budget_complete': progress['step'] >= total_steps}))
        completion.replace(output / 'complete.json')
    if world > 1:
        dist.barrier(); dist.destroy_process_group()
    return model, progress


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    group = p.add_mutually_exclusive_group(); group.add_argument('--resume'); group.add_argument('--initialize')
    args = p.parse_args()
    run(json.loads(Path(args.config).read_text()), args.resume, args.initialize)


if __name__ == '__main__':
    main()
