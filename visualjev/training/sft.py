import torch


def question_loss(logits, target, kind, rps_weight=0.0):
    logits = logits.float()
    y = torch.as_tensor(target, dtype=torch.float32, device=logits.device)
    ce = -(y * logits.log_softmax(-1)).sum()
    rps = logits.new_zeros(())
    if kind == "score":
        rps = (logits.softmax(-1).cumsum(-1)[:-1] - y.cumsum(-1)[:-1]).square().mean()
    return ce + rps_weight * rps, {"ce": ce.detach(), "rps": rps.detach()}


class SFTObjective:
    metric_names = ("ce", "rps")
    num_iterations = 1

    def __init__(self, config):
        self.rps_weight = config.get("rps_weight", 0.0)

    def prepare(self, model, pack):
        return [[None for _ in state.questions] for state in pack]

    def loss(self, model, question, rollout):
        return question_loss(model(question), question.target, question.kind, self.rps_weight)

    def state_dict(self):
        return None
