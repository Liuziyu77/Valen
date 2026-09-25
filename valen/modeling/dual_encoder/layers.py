import math
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def _split(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_width).transpose(1, 2)

    def project_memory(self, memory):
        return self._split(self.k(memory)), self._split(self.v(memory))

    def forward(self, query, memory=None, mask=None, kv=None):
        k, v = self.project_memory(memory) if kv is None else kv
        allowed = mask[:, None, None, :].bool() if mask is not None else None
        x = F.scaled_dot_product_attention(self._split(self.q(query)), k, v,
                                           attn_mask=allowed, dropout_p=0.0)
        return self.out(x.transpose(1, 2).reshape(query.shape))


class FFN(nn.Sequential):
    def __init__(self, width, intermediate):
        super().__init__(nn.Linear(width, intermediate), nn.GELU(), nn.Linear(intermediate, width))


class FusionBlock(nn.Module):
    def __init__(self, width, heads, intermediate):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.attention = Attention(width, heads)
        self.ffn = FFN(width, intermediate)

    def forward(self, x, mask):
        norm = self.norms[0](x)
        x = x + self.attention(norm, norm, mask)
        return x + self.ffn(self.norms[1](x))


class QuestionBlock(nn.Module):
    def __init__(self, width, heads, intermediate):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.self_attention = Attention(width, heads)
        self.instructions = Attention(width, heads)
        self.memory = Attention(width, heads)
        self.ffn = FFN(width, intermediate)

    def forward(self, r, h, hmask, memory, mmask):
        n = self.norms[0](r)
        r = r + self.self_attention(n, n)
        r = r + self.instructions(self.norms[1](r), h, hmask)
        # Flatten query rows, never memory: one K/V projection per state/layer.
        shape = r.shape
        r = r + self.memory(self.norms[2](r).reshape(1, -1, shape[-1]), memory, mmask).reshape(shape)
        return r + self.ffn(self.norms[3](r))


class CandidateBlock(nn.Module):
    def __init__(self, width, heads, intermediate):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.question = Attention(width, heads)
        self.memory = Attention(width, heads)
        self.ffn = FFN(width, intermediate)

    def forward(self, u, r, owners, memory, mmask, chunk_size):
        kv = self.memory.project_memory(memory)
        chunks = []
        for start in range(0, len(u), chunk_size):
            x = u[start:start + chunk_size].unsqueeze(1)
            evidence = r[owners[start:start + chunk_size]]
            x = x + self.question(self.norms[0](x), evidence)
            x = x + self.memory(self.norms[1](x).transpose(0, 1), mask=mmask, kv=kv).transpose(0, 1)
            chunks.append((x + self.ffn(self.norms[2](x))).squeeze(1))
        return torch.cat(chunks)


def positions(length, width, device):
    pos = torch.arange(length, device=device).float()[:, None]
    frequencies = torch.exp(torch.arange(0, width, 2, device=device).float() * (-math.log(10000.0) / width))
    features = pos * frequencies[None, :]
    return torch.stack((features.sin(), features.cos()), -1).flatten(-2)[:, :width]
