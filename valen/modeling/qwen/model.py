import math
import torch
from torch import nn
from valen import MODEL_NAME


class DecisionHead(nn.Module):
    def __init__(self, hidden_size, projection_dim=256):
        super().__init__()
        self.decision = nn.Linear(hidden_size, projection_dim, bias=False)
        self.candidate = nn.Linear(hidden_size, projection_dim, bias=False)
        self.scale = math.sqrt(projection_dim)

    def forward(self, hidden, positions, decision_position):
        # Compute head and losses in FP32 even with a BF16 backbone.
        d = self.decision(hidden[0, decision_position].float())
        c = self.candidate(hidden[0, positions].float())
        return (c * d).sum(-1) / self.scale


class ValenQwen(nn.Module):
    """Qwen3.5 backbone with a shared candidate decision head."""

    model_name = MODEL_NAME
    architecture = "qwen"

    def __init__(self, backbone, projection_dim=256):
        super().__init__()
        self.backbone = backbone
        self.head = DecisionHead(backbone.config.text_config.hidden_size, projection_dim)

    def forward(self, question):
        device = next(self.backbone.parameters()).device
        outputs = []
        for branch in question.branches:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in branch.inputs.items()}
            hidden = self.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
            outputs.append(self.head(hidden, branch.candidate_positions, branch.decision_position))
        return torch.cat(outputs)

    def extract_features(self, question):
        """Keep only head readout positions, once per frozen-backbone rollout."""
        device = next(self.backbone.parameters()).device
        features = []
        for branch in question.branches:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in branch.inputs.items()}
            hidden = self.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
            features.append(hidden[:, [*branch.candidate_positions, branch.decision_position], :])
        return features

    def score_features(self, features, head=None):
        head = self.head if head is None else head
        return torch.cat([head(hidden, list(range(hidden.shape[1] - 1)), hidden.shape[1] - 1)
                          for hidden in features])
