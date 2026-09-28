"""Isolate ITM optimization on cached training examples; never export weights."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from valen.pretraining.data import CaptionDataset
from valen.pretraining.model import AlignmentModel


def main():
    torch.manual_seed(42)
    payload = torch.load('next-generation-exp/alignment_100k/latest.pt', map_location='cpu', weights_only=False)
    config = dict(payload['config'], device='cuda:0', stage='warmup')
    data = CaptionDataset(config['data'], config)
    model = AlignmentModel(config, data.tokenizer.mask_token_id).cuda()
    model.load_state_dict(payload['model']); model.eval(); model.requires_grad_(False)
    batch = {k: v.cuda() for k, v in data.collate([data[i] for i in range(64)]).items()}
    with torch.no_grad(), model.core._amp():
        h, v, text, image = model.features(batch)
    n = len(h)
    probabilities = (image @ text.T / model.log_temperature.exp().clamp(.01, .5)).masked_fill(torch.eye(n, device='cuda', dtype=torch.bool), -torch.inf).softmax(-1)
    original_fusion = {k: t.clone() for k, t in model.core.fusion.state_dict().items()}
    original_head = {k: t.clone() for k, t in model.itm_head.state_dict().items()}
    model.core.fusion.requires_grad_(True); model.itm_head.requires_grad_(True)
    records = []
    for policy in ('hard', 'random'):
        torch.manual_seed(42)
        model.core.fusion.load_state_dict(original_fusion); model.itm_head.load_state_dict(original_head)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        for step in range(301):
            negative = (torch.multinomial(probabilities, 1).squeeze(1) if policy == 'hard' else
                        torch.arange(n, device='cuda').roll(int(torch.randint(1, n, ()).item())))
            with model.core._amp():
                positive = model.fuse(h, batch['attention_mask'], v, batch['visual_mask'], batch['pixel_values'])[:, 0]
                wrong = model.fuse(h[negative], batch['attention_mask'][negative], v, batch['visual_mask'], batch['pixel_values'])[:, 0]
                logits = model.itm_head(torch.cat((positive, wrong))).float()
                labels = torch.cat((torch.ones(n, device='cuda'), torch.zeros(n, device='cuda'))).long()
                loss = F.cross_entropy(logits, labels)
            if step % 25 == 0:
                row = {'negative_policy': policy, 'step': step, 'loss': float(loss.detach()),
                       'accuracy': float((logits.argmax(-1) == labels).float().mean())}
                records.append(row); print(json.dumps(row), flush=True)
            if step == 300:
                break
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
    Path('next-generation-exp/alignment_100k/itm_overfit_diagnostic.json').write_text(json.dumps(
        {'purpose': 'Training-path diagnostic only, not held-out evaluation; weights discarded', 'training_samples': 64, 'history': records}, indent=2))


if __name__ == '__main__':
    main()
