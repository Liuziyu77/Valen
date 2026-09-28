"""ITC、ITM、MLM 图文预训练。 / Image-caption pretraining with ITC, ITM and MLM."""
from contextlib import nullcontext
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from safetensors import safe_open
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead

from valen.modeling.dual_encoder.builder import build_dual_encoder
from valen.modeling.dual_encoder.layers import positions
from valen.modeling.dual_encoder.transfer import is_shared
from .data import mask_tokens


def gather_features(x):
    return torch.cat(all_gather(x), 0) if dist.is_initialized() and dist.get_world_size() > 1 else x


def contrastive_loss(image, text, groups, temperature, distributed=True):
    """双向 ITC，同组样本均为正例。 / Bidirectional ITC with same-group positives."""
    world = dist.get_world_size() if distributed and dist.is_initialized() else 1
    images = gather_features(image) if world > 1 else image
    texts = gather_features(text) if world > 1 else text
    if world > 1:
        collected = [torch.empty_like(groups) for _ in range(world)]
        dist.all_gather(collected, groups)
        all_groups = torch.cat(collected)
    else:
        all_groups = groups
    positives = groups[:, None] == all_groups[None, :]
    target = positives.float() / positives.sum(1, keepdim=True)
    i2t, t2i = image.float() @ texts.float().T / temperature, text.float() @ images.float().T / temperature
    loss = -((target * i2t.log_softmax(-1)).sum(-1).mean() + (target * t2i.log_softmax(-1)).sum(-1).mean()) / 2
    accuracy = (positives.gather(1, i2t.argmax(-1, keepdim=True)).float().mean()
                + positives.gather(1, t2i.argmax(-1, keepdim=True)).float().mean()) / 2
    return loss, accuracy


class AlignmentModel(nn.Module):
    """共享双编码器主干；预训练头不迁移到 SFT。
    Share the dual-encoder backbone; pretraining heads do not transfer to SFT.
    """
    def __init__(self, config, mask_token_id):
        super().__init__()
        self.core = build_dual_encoder(config)
        for name, parameter in self.core.named_parameters():
            if not is_shared(name):
                parameter.requires_grad_(False)
        d = self.core.width
        self.image_head = nn.Linear(d, config.get('contrastive_dim', 256))
        self.text_head = nn.Linear(d, config.get('contrastive_dim', 256))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(.07)))
        self.itm_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        tc = self.core.text_encoder.config
        self.mlm_projection = nn.Linear(d, tc.hidden_size)
        self.mlm_head = ModernBertPredictionHead(tc)
        self.mlm_decoder = nn.Linear(tc.hidden_size, tc.vocab_size, bias=tc.decoder_bias)
        # 独立复制，保留冻结的词嵌入。 / Copy separately to preserve frozen embeddings.
        with torch.no_grad():
            self.mlm_decoder.weight.copy_(self.core.text_encoder.get_input_embeddings().weight)
        for path in Path(config['text_model_path']).glob('*.safetensors'):
            with safe_open(path, framework='pt') as f:
                for prefix, module in [('head.', self.mlm_head), ('decoder.', self.mlm_decoder)]:
                    weights = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
                    if weights:
                        result = module.load_state_dict(weights, strict=False)
                        if result.unexpected_keys:
                            raise ValueError('Unexpected pretrained MLM weights')
        self.mask_token_id = mask_token_id

    def encode_text(self, ids, mask):
        encoder = self.core.text_encoder
        with nullcontext() if any(p.requires_grad for p in encoder.parameters()) else torch.no_grad():
            h = encoder(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
        return self.core.text_projection(h)

    def encode_image(self, pixels):
        encoder = self.core.vision_encoder
        with nullcontext() if any(p.requires_grad for p in encoder.parameters()) else torch.no_grad():
            h = encoder(pixel_values=pixels, return_dict=True).last_hidden_state
        h = torch.cat((h[:, :1], h[:, 1 + encoder.config.num_register_tokens:]), 1)
        return self.core.vision_projection(h)

    def fuse(self, text, mask, image, visual_mask, pixels):
        core = self.core
        text = text + core.modality.weight[0] + positions(text.shape[1], core.width, text.device)
        patch = core.vision_encoder.config.patch_size
        rows, cols = pixels.shape[-2] // patch, pixels.shape[-1] // patch
        y, x = torch.meshgrid(torch.linspace(-1, 1, rows, device=text.device),
                              torch.linspace(-1, 1, cols, device=text.device), indexing='ij')
        xy = core.coordinates(torch.stack((x, y), -1).reshape(1, -1, 2))
        image = image + torch.cat((core.visual_cls, xy), 1) + core.modality.weight[1]
        h, m = torch.cat((text, image), 1), torch.cat((mask, visual_mask), 1)
        for layer in core.fusion:
            h = layer(h, m)
        return h

    def features(self, batch):
        h = self.encode_text(batch['input_ids'], batch['attention_mask'])
        v = self.encode_image(batch['pixel_values'])
        mask = batch['eligible'].unsqueeze(-1)
        text = F.normalize(self.text_head((h * mask).sum(1) / mask.sum(1).clamp_min(1)).float(), dim=-1)
        image = F.normalize(self.image_head(v[:, 0]).float(), dim=-1)
        return h, v, text, image

    def forward(self, batch, distributed=True):
        with self.core._amp():
            h, v, text, image = self.features(batch)
            temperature = self.log_temperature.exp().clamp(.01, .5)
            itc, retrieval = contrastive_loss(image, text, batch['group_ids'], temperature, distributed)
            mask, vm, pixels = batch['attention_mask'], batch['visual_mask'], batch['pixel_values']
            positives = self.fuse(h, mask, v, vm, pixels)[:, 0]
            scores = (image @ text.T).detach() / temperature.detach()
            valid = batch['group_ids'][:, None] != batch['group_ids'][None, :]
            if not valid.any(1).all():
                raise ValueError('ITM requires a distinct negative image for every local sample')
            probabilities = scores.masked_fill(~valid, -float('inf')).softmax(-1)
            negative = torch.multinomial(probabilities, 1).squeeze(1)
            negatives = self.fuse(h[negative], mask[negative], v, vm, pixels)[:, 0]
            logits = self.itm_head(torch.cat((positives, negatives))).float()
            labels = torch.cat((torch.ones(len(h), device=h.device), torch.zeros(len(h), device=h.device))).long()
            itm = F.cross_entropy(logits, labels)
            masked, labels_mlm = mask_tokens(batch['input_ids'], batch['eligible'], self.mask_token_id,
                                            self.core.text_encoder.config.vocab_size)
            # 遮盖后重新编码，避免答案泄漏。 / Re-encode masked text to avoid target leakage.
            masked_h = self.encode_text(masked, mask)
            fused = self.fuse(masked_h, mask, v, vm, pixels)[:, :h.shape[1]]
            selected = labels_mlm != -100
            mlm_logits = self.mlm_decoder(self.mlm_head(self.mlm_projection(fused[selected]))).float()
            mlm = F.cross_entropy(mlm_logits, labels_mlm[selected])
            loss = itc + itm + mlm
        return loss, {'itc': itc.detach(), 'itm': itm.detach(), 'mlm': mlm.detach(),
                      'retrieval_batch_accuracy': retrieval.detach(),
                      'itm_accuracy': (logits.argmax(-1) == labels).float().mean().detach(),
                      'mlm_accuracy': (mlm_logits.argmax(-1) == labels_mlm[selected]).float().mean().detach()}
