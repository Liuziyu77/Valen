"""可替换的候选评分头。 / Interchangeable, candidate-wise decision heads."""
import math

import torch
from torch import nn


def head_signature(config):
    """用于构建与恢复校验。 / Canonical architecture settings for loading."""
    kind = config.get("head_type", "bilinear")
    defaults = {
        "bilinear": {"projection_dim": 256},
        "mlp": {"head_hidden_dim": 512, "head_bottleneck_dim": 128},
        "role_mlp": {"head_width": 256, "head_hidden_dim": 1024},
        "mixer": {"head_width": 256, "head_layers": 2,
                  "head_token_hidden_dim": 16, "head_channel_hidden_dim": 1024},
    }
    if kind not in defaults:
        raise ValueError(f"Unknown Qwen head_type: {kind!r}")
    values = {key: config.get(key, value) for key, value in defaults[kind].items()}
    if any(type(value) is not int or value <= 0 for value in values.values()):
        raise ValueError("Decision head dimensions must be positive integers")
    return {"head_type": kind, **values}


class DecisionHead(nn.Module):
    """原有参数名保持兼容。 / Preserve legacy bilinear parameter names."""
    def __init__(self, hidden_size, projection_dim=256):
        super().__init__()
        self.decision = nn.Linear(hidden_size, projection_dim, bias=False)
        self.candidate = nn.Linear(hidden_size, projection_dim, bias=False)
        self.scale = math.sqrt(projection_dim)

    def forward(self, hidden, positions, decision_position):
        d = self.decision(hidden[0, decision_position].float())
        c = self.candidate(hidden[0, positions].float())
        return (c * d).sum(-1) / self.scale

    def extract_features(self, hidden, branch):
        return hidden[:, [*branch.candidate_positions, branch.decision_position], :]

    def score_features(self, features):
        return self(features, list(range(features.shape[1] - 1)), features.shape[1] - 1)


class MLPHead(nn.Module):
    """每个候选共享一个 MLP。 / One shared MLP for arbitrary candidate counts."""
    def __init__(self, hidden_size, hidden_dim=512, bottleneck_dim=128):
        super().__init__()
        self.decision_norm = nn.LayerNorm(hidden_size)
        self.candidate_norm = nn.LayerNorm(hidden_size)
        self.network = nn.Sequential(nn.Linear(2 * hidden_size, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, bottleneck_dim), nn.GELU(),
                                     nn.Linear(bottleneck_dim, 1))

    def extract_features(self, hidden, branch):
        c = hidden[0, branch.candidate_positions]
        d = hidden[0, branch.decision_position].expand_as(c)
        return torch.stack((d, c), dim=1)

    def score_features(self, features):
        d, c = features.float().unbind(1)
        return self.network(torch.cat((self.decision_norm(d), self.candidate_norm(c)), -1)).squeeze(-1)


def role_features(hidden, branch):
    """先池化再投影，可缓存冻结特征。 / Pool before trainable projections."""
    spans = (getattr(branch, "context_span", None), getattr(branch, "instruction_span", None))
    candidates = getattr(branch, "candidate_spans", None)
    if any(span is None for span in spans) or not candidates or len(candidates) != len(branch.candidate_positions):
        raise ValueError("Role heads require compiler context, instruction and candidate spans")

    def pool(span):
        start, end = span
        if not 0 <= start < end <= hidden.shape[1]:
            raise ValueError(f"Invalid role span: {span}")
        return hidden[0, start:end].float().mean(0)

    context, instruction = (pool(span) for span in spans)
    decision = hidden[0, branch.decision_position].float()
    return torch.stack([torch.stack((context, instruction, pool(span), decision)) for span in candidates])


class RoleHead(nn.Module):
    def __init__(self, hidden_size, width):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, width)
        self.roles = nn.Parameter(torch.zeros(1, 4, width))

    def extract_features(self, hidden, branch):
        return role_features(hidden, branch)

    def project(self, features):
        return self.projection(self.input_norm(features.float())) + self.roles


class RoleMLPHead(RoleHead):
    """与 Mixer 使用相同输入。 / Control for the Mixer's richer input features."""
    def __init__(self, hidden_size, width=256, hidden_dim=1024):
        super().__init__(hidden_size, width)
        self.network = nn.Sequential(nn.Linear(4 * width, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def score_features(self, features):
        return self.network(self.project(features).flatten(1)).squeeze(-1)


class MixerBlock(nn.Module):
    def __init__(self, width, token_hidden, channel_hidden):
        super().__init__()
        self.token_norm = nn.LayerNorm(width)
        self.token_mlp = nn.Sequential(nn.Linear(4, token_hidden), nn.GELU(), nn.Linear(token_hidden, 4))
        self.channel_norm = nn.LayerNorm(width)
        self.channel_mlp = nn.Sequential(nn.Linear(width, channel_hidden), nn.GELU(), nn.Linear(channel_hidden, width))

    def forward(self, x):
        # 混合角色，不混合不同候选。 / Mix fixed roles, never candidate rows.
        x = x + self.token_mlp(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(self.channel_norm(x))


class MixerHead(RoleHead):
    def __init__(self, hidden_size, width=256, layers=2, token_hidden=16, channel_hidden=1024):
        super().__init__(hidden_size, width)
        self.blocks = nn.Sequential(*(MixerBlock(width, token_hidden, channel_hidden) for _ in range(layers)))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def score_features(self, features):
        # 第三行是候选角色。 / Read the updated candidate role.
        return self.output(self.blocks(self.project(features))[:, 2]).squeeze(-1)


def build_head(hidden_size, config):
    settings = head_signature(config)
    kind = settings.pop("head_type")
    if kind == "bilinear":
        return DecisionHead(hidden_size, settings["projection_dim"])
    if kind == "mlp":
        return MLPHead(hidden_size, settings["head_hidden_dim"], settings["head_bottleneck_dim"])
    if kind == "role_mlp":
        return RoleMLPHead(hidden_size, settings["head_width"], settings["head_hidden_dim"])
    return MixerHead(hidden_size, settings["head_width"], settings["head_layers"],
                     settings["head_token_hidden_dim"], settings["head_channel_hidden_dim"])
