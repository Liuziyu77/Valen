"""Evaluate a completed alignment checkpoint on one GPU."""
import argparse
import json
from pathlib import Path

import torch

from valen.pretraining.data import CaptionDataset
from valen.pretraining.model import AlignmentModel
from valen.pretraining.runner import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=1024)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = dict(payload['config'], device='cuda:0', stage='vision_top')
    dataset = CaptionDataset(args.data, config)
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    model = AlignmentModel(config, dataset.tokenizer.mask_token_id).to(device)
    model.load_state_dict(payload['model'])
    result = evaluate(model, dataset, config, device, args.limit)
    result.update(checkpoint=args.checkpoint, data=args.data, step=payload['progress']['step'])
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
