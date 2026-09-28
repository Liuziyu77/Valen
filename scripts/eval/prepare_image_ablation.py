"""Keep decision questions/labels fixed and permute images within each domain."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random


def image_item(record):
    state = record['request']['state']
    found = [item for message in state['messages'] if isinstance(message['content'], list)
             for item in message['content'] if item['type'] == 'image_url']
    if len(found) != 1:
        raise ValueError('This ablation requires exactly one image per record')
    return found[0]


def main():
    source = Path('data/eval_v1/train.jsonl')
    with source.open() as stream:
        records = [json.loads(line) for line in stream]
    domains = defaultdict(dict)
    for record in records:
        domain = record['meta']['domain']
        path = (source.parent / image_item(record)['image_url']['url']).resolve(strict=True)
        domains[domain][record['group_id']] = path
    swaps = {}
    rng = random.Random(42)
    for domain, images in sorted(domains.items()):
        groups = sorted(images); rng.shuffle(groups)
        if len(groups) < 2:
            raise ValueError('Need at least two image groups per domain')
        for i, group in enumerate(groups):
            replacement = images[groups[(i + 1) % len(groups)]]
            if replacement == images[group]:
                raise ValueError('Image permutation has a fixed point')
            swaps[domain, group] = replacement
    output = Path('next-generation-exp/ablation'); output.mkdir(exist_ok=True)
    destination = output / 'shuffled_images.jsonl'
    with destination.open('w') as stream:
        for original in records:
            record = deepcopy(original)
            replacement = swaps[record['meta']['domain'], record['group_id']]
            image_item(record)['image_url']['url'] = str(replacement)
            # group_id continues to identify the original evaluation unit for
            # paired comparisons. This file must never be used for training.
            record['meta']['ablation'] = 'within-domain image permutation; original questions and labels'
            record.pop('assets', None)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    (output / 'manifest.json').write_text(json.dumps({'source': str(source), 'output': str(destination),
        'records': len(records), 'seed': 42, 'image_groups_per_domain': {k: len(v) for k, v in domains.items()},
        'note': 'Input ablation only; no answer accuracy claim for the substituted image itself'}, indent=2))
    print('Prepared', len(records), 'paired ablation records')


if __name__ == '__main__':
    main()
