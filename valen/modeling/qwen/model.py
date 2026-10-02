import torch
from torch import nn
from valen import MODEL_NAME
from .heads import DecisionHead, build_head


class ValenQwen(nn.Module):
    """Qwen3.5 backbone with a shared candidate decision head."""

    model_name = MODEL_NAME
    architecture = "qwen"

    def __init__(self, backbone, projection_dim=256, head_config=None):
        super().__init__()
        self.backbone = backbone
        self.head = build_head(backbone.config.text_config.hidden_size,
                               dict({"projection_dim": projection_dim}, **(head_config or {})))

    def forward(self, question):
        device = next(self.backbone.parameters()).device
        outputs = []
        for branch in question.branches:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in branch.inputs.items()}
            hidden = self.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
            outputs.append(self.head.score_features(self.head.extract_features(hidden, branch)))
        return torch.cat(outputs)

    def extract_features(self, question):
        """缓存投影前的特征。 / Cache raw readouts before trainable head layers."""
        device = next(self.backbone.parameters()).device
        features = []
        for branch in question.branches:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in branch.inputs.items()}
            hidden = self.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
            features.append(self.head.extract_features(hidden, branch))
        return features

    def score_features(self, features, head=None):
        head = self.head if head is None else head
        return torch.cat([head.score_features(hidden) for hidden in features])

    def extract_state_features(self, state):
        """Keep one backbone graph shared by every question's candidate readouts."""
        if not state.questions:
            return []
        if state.inputs is None:
            raise ValueError("Shared forward requires a shared_state compiler")
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in state.inputs.items()}
        hidden = self.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
        return [[self.head.extract_features(hidden, branch) for branch in q.branches] for q in state.questions]

    def forward_state(self, state):
        return [self.score_features(features) for features in self.extract_state_features(state)]
