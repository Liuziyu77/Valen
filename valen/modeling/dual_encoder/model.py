"""Shared immutable state memory with independent question/candidate readers."""
from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F


from .layers import Attention, FusionBlock, QuestionBlock, CandidateBlock, positions


class ValenDualEncoder(nn.Module):
    architecture = "dual_encoder"
    model_name = "Valen-DE-ModernBERTbase-DINOv3B16-en-v1"

    def __init__(self, text_encoder, vision_encoder, config):
        super().__init__()
        self.config = dict(config)
        if config.get("dtype", "bf16") not in {"bf16", "fp32"}:
            raise ValueError("Dual encoder dtype must be bf16 or fp32")
        for key, default in (("hidden_size", 512), ("num_heads", 8), ("intermediate_size", 2048),
                             ("fusion_layers", 2), ("question_layers", 3), ("candidate_layers", 2),
                             ("question_queries", 8), ("text_batch_size", 32), ("candidate_chunk_size", 256)):
            if type(config.get(key, default)) is not int or config.get(key, default) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        self.model_name = config.get("model_name", self.model_name)
        self.text_encoder, self.vision_encoder = text_encoder, vision_encoder
        d, heads, inner = config.get("hidden_size", 512), config.get("num_heads", 8), config.get("intermediate_size", 2048)
        if d <= 0 or heads <= 0 or d % heads or d % 2 or inner <= 0:
            raise ValueError("hidden_size must be positive/even and divisible by num_heads")
        self.width = d
        self.pad_token_id = text_encoder.config.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("Text encoder config needs pad_token_id")
        self.text_projection = nn.Linear(text_encoder.config.hidden_size, d)
        self.vision_projection = nn.Linear(vision_encoder.config.hidden_size, d)
        self.modality = nn.Embedding(2, d)
        self.coordinates = nn.Linear(2, d)
        self.visual_cls = nn.Parameter(torch.zeros(1, 1, d))
        self.fusion = nn.ModuleList([FusionBlock(d, heads, inner) for _ in range(config.get("fusion_layers", 2))])
        self.question_queries = nn.Parameter(torch.randn(1, config.get("question_queries", 8), d) / math.sqrt(d))
        self.question_pool = Attention(d, heads)
        self.question_reader = nn.ModuleList([QuestionBlock(d, heads, inner) for _ in range(config.get("question_layers", 3))])
        self.candidate_query = nn.Parameter(torch.randn(1, 1, d) / math.sqrt(d))
        self.candidate_pool = Attention(d, heads)
        self.task_type = nn.Embedding(3, d)
        self.candidate_init = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.candidate_reader = nn.ModuleList([CandidateBlock(d, heads, inner) for _ in range(config.get("candidate_layers", 2))])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.text_batch_size = config.get("text_batch_size", 32)
        self.candidate_chunk_size = config.get("candidate_chunk_size", 256)
        if min(self.text_batch_size, self.candidate_chunk_size) <= 0:
            raise ValueError("Batch and chunk sizes must be positive")
        if not self.fusion or not self.question_reader or not self.candidate_reader or self.question_queries.shape[1] < 1:
            raise ValueError("Fusion, readers and query counts must be positive")

    def _amp(self):
        device = next(self.parameters()).device
        return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and self.config.get("dtype", "bf16") == "bf16" else nullcontext()

    def _text_batch(self, sequences):
        device = next(self.parameters()).device
        length = max(map(len, sequences))
        if length > self.text_encoder.config.max_position_embeddings:
            raise ValueError("Text sequence exceeds encoder position limit")
        ids = torch.full((len(sequences), length), self.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, seq in enumerate(sequences):
            ids[i, :len(seq)] = torch.tensor(seq, device=device)
            mask[i, :len(seq)] = True
        context = nullcontext() if any(p.requires_grad for p in self.text_encoder.parameters()) else torch.no_grad()
        with context:
            hidden = self.text_encoder(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
        return self.text_projection(hidden), mask

    def _text_chunks(self, sequences, pool=None):
        # Length sorting affects execution only, never the original question/candidate order.
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        results = [None] * len(sequences)
        for start in range(0, len(order), self.text_batch_size):
            indices = order[start:start + self.text_batch_size]
            hidden, mask = self._text_batch([sequences[i] for i in indices])
            if pool is not None:
                hidden = pool(self.candidate_query.expand(len(indices), -1, -1), hidden, mask).squeeze(1)
            for row, i in enumerate(indices):
                results[i] = hidden[row] if pool is not None else hidden[row, :len(sequences[i])]
        if pool is not None:
            return torch.stack(results)
        max_length = max(map(len, sequences))
        hidden = torch.stack([F.pad(h, (0, 0, 0, max_length - h.shape[0])) for h in results])
        mask = torch.arange(max_length, device=hidden.device)[None, :] < torch.tensor(list(map(len, sequences)), device=hidden.device)[:, None]
        return hidden, mask

    def encode_state(self, state):
        with self._amp():
            text, mask = self._text_batch([state.state_tokens])
            text = text + self.modality.weight[0] + positions(text.shape[1], self.width, text.device)
            if state.pixel_values is not None:
                pixels = state.pixel_values.to(device=text.device, dtype=next(self.vision_encoder.parameters()).dtype)
                context = nullcontext() if any(p.requires_grad for p in self.vision_encoder.parameters()) else torch.no_grad()
                with context:
                    hidden = self.vision_encoder(pixel_values=pixels, return_dict=True).last_hidden_state
                registers = self.vision_encoder.config.num_register_tokens
                hidden = torch.cat((hidden[:, :1], hidden[:, 1 + registers:]), dim=1)
                vision = self.vision_projection(hidden)
                patch = self.vision_encoder.config.patch_size
                rows, cols = pixels.shape[-2] // patch, pixels.shape[-1] // patch
                if vision.shape[1] != rows * cols + 1:
                    raise ValueError("Unexpected vision token layout")
                y, x = torch.meshgrid(torch.linspace(-1, 1, rows, device=text.device),
                                      torch.linspace(-1, 1, cols, device=text.device), indexing="ij")
                xy = self.coordinates(torch.stack((x, y), -1).reshape(1, -1, 2))
                vision = vision + torch.cat((self.visual_cls, xy), 1) + self.modality.weight[1]
                text = torch.cat((text, vision), dim=1)
                mask = torch.cat((mask, state.visual_mask.to(text.device)), dim=1)
            for layer in self.fusion:
                text = layer(text, mask)
            return text, mask

    def encode_questions(self, state, memory):
        with self._amp():
            m, mmask = memory
            h, hmask = self._text_chunks(state.instruction_tokens)
            r = self.question_pool(self.question_queries.expand(len(state.questions), -1, -1), h, hmask)
            for layer in self.question_reader:
                r = layer(r, h, hmask, m, mmask)
            return r

    def score_questions(self, state, memory, evidence):
        with self._amp():
            m, mmask = memory
            candidates = self._text_chunks(state.candidate_tokens, self.candidate_pool)
            counts = torch.tensor([b - a for a, b in zip(state.candidate_offsets, state.candidate_offsets[1:])], device=m.device)
            owners = torch.arange(len(counts), device=m.device).repeat_interleave(counts)
            kinds = torch.tensor([{"choice": 0, "noul": 1, "score": 2}[q.kind] for q in state.questions], device=m.device)
            u = self.candidate_init(torch.cat((candidates, evidence.mean(1)[owners], self.task_type(kinds)[owners]), -1))
            for layer in self.candidate_reader:
                u = layer(u, evidence, owners, m, mmask, self.candidate_chunk_size)
        # Keep logits and all probability operations in FP32.
        with torch.autocast(device_type=u.device.type, enabled=False):
            logits = self.head(u.float()).squeeze(-1)
        return list(logits.split(counts.tolist()))

    def forward(self, state):
        if not state.questions:
            return []
        memory = self.encode_state(state)
        evidence = self.encode_questions(state, memory)
        return self.score_questions(state, memory, evidence)

    forward_state = forward
