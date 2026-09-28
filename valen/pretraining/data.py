import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoTokenizer
from valen.data.compilers.dual_encoder import ParallelCompiler


class CaptionDataset(Dataset):
    """读取图文对，复用决策模型的图片预处理。
    Load caption pairs with the decision model’s image preprocessing.
    """
    def __init__(self, path, config):
        path = Path(path)
        self.root = path.parent
        self.rows = []
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                self.rows.append((row['image'], row['caption'], int(row['group_id'][:15], 16)))
        self.tokenizer = AutoTokenizer.from_pretrained(config['text_model_path'], local_files_only=True)
        processor = AutoImageProcessor.from_pretrained(config['vision_model_path'], local_files_only=True)
        self.compiler = ParallelCompiler(self.tokenizer, processor, image_size=config.get('image_size', 384))
        self.max_length = config.get('caption_max_length', 128)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, caption, group = self.rows[index]
        ids = self.tokenizer.encode(caption, add_special_tokens=True)
        if len(ids) > self.max_length:
            raise ValueError(f'Caption exceeds budget: {path}')
        pixels, visual_mask = self.compiler._image(self.root / path)
        return {'ids': ids, 'pixels': pixels.squeeze(0), 'visual_mask': visual_mask.squeeze(0), 'group': group}

    def collate(self, rows):
        length = max(len(r['ids']) for r in rows)
        ids = torch.full((len(rows), length), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, row in enumerate(rows):
            ids[i, :len(row['ids'])] = torch.tensor(row['ids']); mask[i, :len(row['ids'])] = True
        eligible = mask.clone()
        for token in self.tokenizer.all_special_ids:
            eligible &= ids != token
        return {'input_ids': ids, 'attention_mask': mask, 'eligible': eligible,
                'pixel_values': torch.stack([r['pixels'] for r in rows]),
                'visual_mask': torch.stack([r['visual_mask'] for r in rows]),
                'group_ids': torch.tensor([r['group'] for r in rows])}


def mask_tokens(ids, eligible, mask_id, vocab_size, probability=.15):
    """MLM 遮盖：80% MASK、10% 随机、10% 保留。
    MLM masking: 80% MASK, 10% random, 10% unchanged.
    """
    if mask_id is None:
        raise ValueError('MLM requires a mask token')
    if not eligible.any(1).all():
        raise ValueError('Caption has no maskable tokens')
    selected = (torch.rand(ids.shape, device=ids.device) < probability) & eligible
    for i in (~selected.any(1)).nonzero().flatten():
        choices = eligible[i].nonzero().flatten()
        selected[i, choices[torch.randint(len(choices), (), device=ids.device)]] = True
    replacement = torch.rand(ids.shape, device=ids.device)
    masked = ids.clone()
    masked[selected & (replacement < .8)] = mask_id
    random_positions = selected & (replacement >= .8) & (replacement < .9)
    masked[random_positions] = torch.randint(vocab_size, (int(random_positions.sum()),), device=ids.device)
    labels = ids.clone().masked_fill(~selected, -100)
    return masked, labels
