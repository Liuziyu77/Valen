"""Prepare resumable local caption datasets without copying the source indices."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import configparser
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import random
import re
import time

import boto3
from botocore.config import Config
import numpy as np
from PIL import Image, ImageOps
from transformers import AutoTokenizer


SOURCES = {'cc12m': .4, 'laion_coco': .4, 'coyo700m_new': .15, 'sbu': .05}


def read_journal(path):
    # str.splitlines also splits Unicode separators inside JSON strings.
    # JSONL records are separated by physical newlines only.
    # Bytes also allow recovery when interruption cuts a UTF-8 code point.
    with Path(path).open('rb') as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError as error:
                if f.read().strip():
                    raise ValueError(f'Malformed non-final journal record {number}: {path}') from error
                break  # Only a partial final record can result from interruption.


def dhash(image):
    a = np.asarray(image.convert('L').resize((9, 8)))
    return int.from_bytes(np.packbits(a[:, 1:] > a[:, :-1]).tobytes(), 'big')


class NearDuplicates:
    def __init__(self):
        self.bands = [defaultdict(list) for _ in range(4)]

    def contains(self, value):
        return any((other ^ value).bit_count() <= 3 for i in range(4)
                   for other in self.bands[i][(value >> (16 * i)) & 65535])

    def add(self, value):
        for i in range(4):
            self.bands[i][(value >> (16 * i)) & 65535].append(value)


def candidates(root, source, seed):
    rng = random.Random(seed)
    files = sorted((root / source).glob('*.jsonl'))
    if not files:
        raise FileNotFoundError(source)
    while True:
        path = rng.choice(files)
        with path.open('rb') as f:
            f.seek(rng.randrange(max(1, path.stat().st_size - 65536)))
            f.readline()
            for _ in range(512):
                line = f.readline()
                if not line:
                    break
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                row['_source'] = source
                yield row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=1000000)
    p.add_argument('--eval-count', type=int, default=20000)
    p.add_argument('--workers', type=int, default=64)
    p.add_argument('--write-workers', type=int, default=16)
    p.add_argument('--credentials-config', default=os.environ.get('VALEN_CREDENTIALS_CONFIG', str(Path.home() / 'petreloss.conf')))
    p.add_argument('--source-root', default=os.environ.get('VALEN_CAPTION_SOURCE_ROOT'),
                   help='Caption index directory; or set VALEN_CAPTION_SOURCE_ROOT')
    p.add_argument('--caption-root', default=os.environ.get('VALEN_CAPTION_ROOT', 'data'))
    p.add_argument('--output', default='next-generation-exp/data-preparation')
    args = p.parse_args()
    if not args.source_root:
        p.error('--source-root or VALEN_CAPTION_SOURCE_ROOT is required')
    if args.count <= 0 or args.count % 20 or args.eval_count <= 0 or args.eval_count % 40:
        p.error('--count must be a positive multiple of 20; --eval-count a positive multiple of 40')
    logging.disable(logging.CRITICAL)
    try:
        from PIL import AvifImagePlugin
        # The outer pool supplies parallelism. One decoder must not spawn a
        # thread for every CPU on a 64-core training node.
        AvifImagePlugin.DEFAULT_MAX_THREADS = 1
    except ImportError:
        pass
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    roots = {'train': Path(args.caption_root) / 'train_caption', 'eval': Path(args.caption_root) / 'eval_caption'}
    for root in roots.values():
        (root / 'assets').mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained('models/ModernBERT-base', local_files_only=True)
    cfg = configparser.ConfigParser(); cfg.read(args.credentials_config)
    clients = {}
    for name in ('langchao', 'langchao2'):
        c = cfg[name]
        clients[name] = boto3.client('s3', endpoint_url=c['host_base'], aws_access_key_id=c['access_key'],
            aws_secret_access_key=c['secret_key'], config=Config(proxies={}, max_pool_connections=args.workers,
            connect_timeout=5, read_timeout=15, retries={'max_attempts': 2}, s3={'addressing_style': 'path'}))
    seen, urls, near = set(), set(), NearDuplicates()
    blocked_paths = {}
    decision_groups = {}
    for split in ('train_v1', 'eval_v1'):
        groups = set()
        for line in (Path('data') / split / 'train.jsonl').open():
            r = json.loads(line); groups.add(r['group_id'])
            for a in r.get('assets', []):
                seen.add(a['sha256'])
                if split == 'eval_v1':
                    blocked_paths[a['sha256']] = Path('data') / split / a['path']
        decision_groups[split] = groups
    if decision_groups['train_v1'] & decision_groups['eval_v1']:
        raise ValueError('Decision train/eval group overlap')
    def fingerprint(path):
        with Image.open(path) as im:
            return dhash(ImageOps.exif_transpose(im).convert('RGB'))
    with ThreadPoolExecutor(max_workers=16) as pool:
        for value in pool.map(fingerprint, blocked_paths.values()):
            near.add(value)
    accepted = {s: Counter() for s in roots}
    rejection = Counter()
    records = {s: [] for s in roots}
    handles = {}
    for split, root in roots.items():
        path = root / 'all.jsonl'
        if path.exists():
            # A crash may leave a partial final line. Rewrite only the valid prefix.
            for r in read_journal(path):
                if not (root / r['image']).is_file():
                    raise ValueError('Cached image missing')
                records[split].append(r); accepted[split][r['source']] += 1
                seen.add(r['sha256']); urls.add(r['uri']); near.add(int(r['dhash'], 16))
            temp = path.with_suffix('.recovered.tmp')
            temp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records[split]))
            temp.replace(path)
        handles[split] = path.open('a', buffering=1)
    def write_image(path, raw):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp'); temp.write_bytes(raw); temp.replace(path)
    streams = {s: candidates(Path(args.source_root), s, 42 + i) for i, s in enumerate(SOURCES)}
    def fetch(row):
        try:
            uri = row['image']; cluster, address = uri.split(':s3://', 1); bucket, key = address.split('/', 1)
            response = clients[cluster].get_object(Bucket=bucket, Key=key)
            with response['Body'] as f:
                raw = f.read(16 * 1024 * 1024 + 1)
            if not raw or len(raw) > 16 * 1024 * 1024:
                return None, 'empty_or_oversize'
            digest = hashlib.sha256(raw).hexdigest()
            with Image.open(io.BytesIO(raw)) as im:
                if im.width * im.height > 25000000 or min(im.size) < 64 or max(im.size) / min(im.size) > 6:
                    return None, 'dimensions'
                image = ImageOps.exif_transpose(im).convert('RGB'); image.load()
                if np.asarray(image.resize((32, 32))).std() < 5:
                    return None, 'blank_image'
                size, phash = image.size, dhash(image)
            return (row, raw, digest, size, phash), None
        except Exception as e:
            return None, type(e).__name__
    def publish():
        for split, rows in records.items():
            if split == 'eval' and len(rows) >= args.eval_count:
                for name, subset in [('validation', rows[::2]), ('test', rows[1::2])]:
                    path = roots[split] / (name + '.jsonl')
                    if not path.exists():
                        temp = path.with_suffix('.tmp'); temp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in subset)); temp.replace(path)
            if split == 'train':
                sizes = sorted({min(100000, args.count), args.count})
                for n in sizes:
                    name = {100000: 'train_100k', 1000000: 'train_1m', 3000000: 'train_3m'}.get(n, f'train_{n}')
                    path = roots[split] / (name + '.jsonl')
                    if len(rows) >= n and not path.exists():
                        # Select balanced prefixes per source, then shuffle reproducibly.
                        used, subset = Counter(), []
                        for r in rows:
                            if used[r['source']] < round(n * SOURCES[r['source']]):
                                subset.append(r); used[r['source']] += 1
                        if len(subset) == n:
                            random.Random(42).shuffle(subset)
                            temp = path.with_suffix('.tmp'); temp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in subset)); temp.replace(path)
        status = {'accepted': {s: dict(c) for s, c in accepted.items()}, 'rejected': dict(rejection),
                  'elapsed_seconds': time.monotonic() - started, 'source_sampling': 'random shard/byte windows; not uniform record sampling',
                  'dedup': 'SHA256 and dHash Hamming distance <=3, including decision evaluation images',
                  'caption_policy': 'Laion-COCO BLIP caption when present; first caption elsewhere',
                  'train_target': args.count, 'eval_target': args.eval_count}
        temp = out / 'status.tmp'; temp.write_text(json.dumps(status, indent=2)); temp.replace(out / 'status.json')
        print(json.dumps(status), flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, ThreadPoolExecutor(max_workers=args.write_workers) as writers:
        # Finish the evaluation set and balanced 100k before expanding to 1M.
        for split, total in [('eval', args.eval_count), ('train', min(100000, args.count)), ('train', args.count)]:
            while any(accepted[split][s] < round(total * w) for s, w in SOURCES.items()):
                batch = []
                for source, weight in SOURCES.items():
                    remaining = round(total * weight) - accepted[split][source]
                    if remaining <= 0:
                        continue
                    for _ in range(min(256, remaining)):
                        for attempt in range(1000):
                            row = next(streams[source]); captions = row.get('caption', [])
                            caption = captions[1] if source == 'laion_coco' and len(captions) > 1 else (captions[0] if captions else '')
                            if not isinstance(caption, str) or not 4 <= len(caption.split()) <= 90:
                                rejection['caption_words'] += 1; continue
                            letters = [c for c in caption if c.isalpha()]
                            if not letters or sum(c.isascii() for c in letters) / len(letters) < .95 or 'http' in caption.lower():
                                rejection['language_or_url'] += 1; continue
                            if len(tokenizer.encode(caption)) > 128:
                                rejection['caption_tokens'] += 1; continue
                            if row['image'] in urls:
                                rejection['duplicate_uri'] += 1; continue
                            urls.add(row['image']); row['_caption'] = caption; batch.append(row); break
                if not batch:
                    raise RuntimeError('No candidate progress')
                pending_writes = []
                for value, reason in pool.map(fetch, batch):
                    if reason:
                        rejection[reason] += 1; continue
                    row, raw, digest, size, phash = value
                    if digest in seen or near.contains(phash):
                        rejection['duplicate_or_eval_overlap'] += 1; continue
                    if accepted[split][row['_source']] >= round(total * SOURCES[row['_source']]):
                        continue
                    relative = f'assets/{digest[:3]}/{digest}.img'; path = roots[split] / relative
                    record = {'image': relative, 'caption': row['_caption'], 'group_id': digest, 'sha256': digest,
                              'dhash': f'{phash:016x}', 'source': row['_source'], 'uri': row['image'],
                              'width': size[0], 'height': size[1]}
                    pending_writes.append((writers.submit(write_image, path, raw), record))
                    accepted[split][record['source']] += 1
                    seen.add(digest); near.add(phash)
                # Keep the journal in deterministic candidate order, and commit
                # a record only after its atomic image write has succeeded.
                for future, record in pending_writes:
                    future.result()
                    handles[split].write(json.dumps(record, ensure_ascii=False) + '\n')
                    records[split].append(record)
                publish()
    for f in handles.values():
        f.close()
    publish()


if __name__ == '__main__':
    main()
