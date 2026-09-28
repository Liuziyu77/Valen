from copy import deepcopy
import json
from pathlib import Path
import random

from PIL import Image
import pytest
import torch
from transformers import AutoTokenizer

from test_dual_encoder import tiny_encoders, record
from valen.modeling.factory import build_model
from valen.modeling.dual_encoder.transfer import export_shared, is_shared
from valen.pretraining.data import CaptionDataset, mask_tokens
from valen.pretraining.model import AlignmentModel, contrastive_loss
from valen.pretraining.runner import run, set_stage
from valen.training.checkpoint import load_checkpoint, save_checkpoint


@pytest.fixture
def alignment(tmp_path):
    torch.set_num_threads(1)
    config = tiny_encoders(tmp_path / 'encoders')
    tokenizer = AutoTokenizer.from_pretrained(config['text_model_path'], mask_token='[UNK]')
    tokenizer.save_pretrained(config['text_model_path'])
    rows = []
    for i in range(8):
        path = tmp_path / f'{i}.png'
        Image.new('RGB', (32, 24), 'red' if i % 2 else 'blue').save(path)
        rows.append({'image': path.name, 'caption': 'red blue green' if i % 2 else 'blue red green', 'group_id': f'{i:064x}'})
    path = tmp_path / 'captions.jsonl'; path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    # Distinct group IDs in the first 15 hexadecimal positions.
    for i, row in enumerate(rows):
        row['group_id'] = f'{i + 1:015x}' + '0' * 49
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    config.update(data=str(path), eval_data=str(path), batch_size=2, workers=0, eval_workers=0,
                  epochs=1, eval_limit=4, final_eval_limit=4, save_every=1, eval_every=100,
                  output=str(tmp_path / 'alignment'), stage='vision_top', contrastive_dim=16)
    data = CaptionDataset(path, config)
    return config, data, tokenizer


def test_masking_preserves_labels_and_special_tokens():
    ids = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])
    eligible = torch.tensor([[False, True, True, False]] * 2)
    masked, labels = mask_tokens(ids, eligible, 3, 10, probability=1)
    assert torch.equal(labels[eligible], ids[eligible])
    assert (labels[~eligible] == -100).all()
    assert torch.equal(masked[~eligible], ids[~eligible])


def test_alignment_gradients_and_shared_fusion_equivalence(alignment):
    config, data, tokenizer = alignment
    model = AlignmentModel(config, tokenizer.mask_token_id)
    set_stage(model, 'warmup', config)
    batch = data.collate([data[0], data[1]])
    loss, terms = model(batch)
    assert torch.isfinite(loss) and {'itc', 'itm', 'mlm'} <= set(terms)
    loss.backward()
    for prefix in ['core.text_projection', 'core.vision_projection', 'core.fusion', 'image_head', 'text_head', 'itm_head', 'mlm_projection', 'mlm_head', 'mlm_decoder']:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.startswith(prefix)), prefix
    assert all(p.grad is None for p in model.core.text_encoder.parameters())
    assert all(p.grad is None for p in model.core.question_reader.parameters())
    from types import SimpleNamespace
    state = SimpleNamespace(state_tokens=data[0]['ids'], pixel_values=batch['pixel_values'][:1], visual_mask=batch['visual_mask'][:1])
    with torch.no_grad():
        expected, _ = model.core.encode_state(state)
        h = model.encode_text(batch['input_ids'][:1], batch['attention_mask'][:1])
        v = model.encode_image(batch['pixel_values'][:1])
        actual = model.fuse(h, batch['attention_mask'][:1], v, batch['visual_mask'][:1], batch['pixel_values'][:1])
    torch.testing.assert_close(actual, expected)


def test_multi_positive_contrastive_targets():
    image = torch.eye(3, requires_grad=True)
    loss, _ = contrastive_loss(image, image, torch.tensor([1, 1, 2]), .1, distributed=False)
    assert torch.isfinite(loss)
    loss.backward()
    assert image.grad is not None


def test_shared_export_refreeze_and_decision_checkpoint(alignment, tmp_path):
    config, data, tokenizer = alignment
    model = AlignmentModel(config, tokenizer.mask_token_id)
    with torch.no_grad():
        model.core.text_encoder.layers[-1].parameters().__next__().add_(.1)
    original = {k: v.clone() for k, v in model.core.state_dict().items() if is_shared(k)}
    export_shared(model.core, tmp_path / 'shared')
    sft = dict(config, stage='warmup', pretrained_shared_path=str(tmp_path / 'shared'), fusion_lr=2e-5)
    target = build_model(sft)
    for key, value in original.items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
    assert not any(p.requires_grad for p in target.text_encoder.parameters())
    optimizer = torch.optim.AdamW([p for p in target.parameters() if p.requires_grad])
    save_checkpoint(tmp_path / 'sft', target, optimizer, sft, {'step': 0}, random.Random(42))
    restored = build_model(sft); load_checkpoint(tmp_path / 'sft', restored)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def test_alignment_resume_equals_continuous_across_unfreezing(alignment, tmp_path, monkeypatch):
    config, _, _ = alignment
    monkeypatch.setenv('WORLD_SIZE', '1'); monkeypatch.setenv('RANK', '0')
    continuous, progress = run(config)
    interrupted = dict(config, output=str(tmp_path / 'interrupted'), max_steps=2)
    run(interrupted)
    resumed, restored_progress = run(dict(interrupted, max_steps=4), resume=tmp_path / 'interrupted/latest.pt')
    assert progress == restored_progress
    for key, value in continuous.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
