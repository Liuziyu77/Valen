"""Verify pixel-dependent learning with identical text and masks; discard weights."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from valen.modeling.factory import build_compiler, build_model, normalize_model_config
from valen.training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.manual_seed(42); torch.cuda.set_device(0)
    config = normalize_model_config(json.loads((Path(args.checkpoint) / 'config.json').read_text()))
    config.update(device='cuda:0', gradient_checkpointing=False)
    model = build_model(config); load_checkpoint(args.checkpoint, model)
    model.text_encoder.requires_grad_(False); model.vision_encoder.requires_grad_(False)
    source = Path('data/train_v1/train.jsonl')
    compiler = build_compiler(config, source.parent)
    states, groups = [], []
    with source.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record['meta']['domain'] != 'vqa' or record['group_id'] in groups:
                continue
            images = [item for message in record['request']['state']['messages'] if isinstance(message['content'], list)
                      for item in message['content'] if item['type'] == 'image_url']
            if len(images) != 1:
                continue
            row = {'group_id': record['group_id'], 'request': {
                'state': {'messages': [{'role': 'user', 'content': images}]},
                'questions': {'q': {'type': 'choice', 'instructions': 'Which assigned group does this image belong to?',
                                     'criteria': {'A': 'Group A', 'B': 'Group B'}}}}}
            state = compiler.compile(row)
            if states and not torch.equal(states[0].visual_mask, state.visual_mask):
                continue
            states.append(state); groups.append(record['group_id'])
            if len(states) == 4:
                break
    if len(states) != 4:
        raise ValueError('Could not construct four distinct images with matching padding masks')
    assert all(s.state_tokens == states[0].state_tokens and s.instruction_tokens == states[0].instruction_tokens
               and s.candidate_tokens == states[0].candidate_tokens for s in states)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    labels = torch.tensor([0, 1, 0, 1], device='cuda')
    history = []
    model.train()
    for step in range(151):
        logits = torch.stack([model(s)[0] for s in states])
        loss = F.cross_entropy(logits, labels)
        if step % 10 == 0:
            record = {'step': step, 'loss': float(loss.detach()), 'accuracy': float((logits.argmax(-1) == labels).float().mean())}
            history.append(record); print(json.dumps(record), flush=True)
        if step == 150:
            break
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
    Path(args.output).write_text(json.dumps({'checkpoint': args.checkpoint, 'training_images': 4,
        'identical_text_and_padding_mask': True, 'groups': groups, 'history': history,
        'note': 'Training-path diagnostic on four training images with arbitrary balanced labels; weights discarded'}, indent=2))


if __name__ == '__main__':
    main()
