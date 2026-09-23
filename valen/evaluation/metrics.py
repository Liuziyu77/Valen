"""Decision quality metrics shared by evaluation entry points."""
from collections import defaultdict
import torch


def question_metrics(question, logits):
    logits = logits.detach().float().cpu()
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite evaluation logits")
    target = torch.tensor(question.target, dtype=torch.float32)
    probs = logits.softmax(-1)
    predicted, expected = int(probs.argmax()), int(target.argmax())
    values = {"nll": float(-(target * logits.log_softmax(-1)).sum()),
              "brier": float((probs - target).square().sum())}
    # Accuracy is defined here only for hard labels, not soft distributions.
    if int((target == 1).sum()) == 1 and int((target != 0).sum()) == 1:
        values["accuracy"] = float(predicted == expected)
        if question.kind == "noul":
            values["true_positive"] = float(question.keys[predicted] == "true" and question.keys[expected] == "true")
            values["false_positive"] = float(question.keys[predicted] == "true" and question.keys[expected] != "true")
            values["false_negative"] = float(question.keys[predicted] != "true" and question.keys[expected] == "true")
            values["true_negative"] = float(question.keys[predicted] != "true" and question.keys[expected] != "true")
    if question.kind == "score":
        levels = torch.arange(len(probs), dtype=torch.float32)
        values["expected_score_mae"] = float(((probs * levels).sum() - (target * levels).sum()).abs())
        values["rps"] = float((probs.cumsum(-1)[:-1] - target.cumsum(-1)[:-1]).square().mean())
    return values


def summarize(rows):
    buckets = defaultdict(list)
    for row in rows:
        for key in ("overall", f"task/{row['task']}", f"modality/{row['modality']}",
                    f"task_modality/{row['task']}/{row['modality']}", f"language/{row['language']}",
                    f"domain/{row.get('domain', 'unknown')}"):
            buckets[key].append(row["metrics"])
    result = {}
    confusion = {"true_positive", "false_positive", "false_negative", "true_negative"}
    for key, values in sorted(buckets.items()):
        names = set().union(*(v.keys() for v in values))
        aggregate = {"questions": len(values), "metric_counts": {}}
        for name in sorted(names - confusion):
            entries = [v[name] for v in values if name in v]
            aggregate[name] = sum(entries) / len(entries)
            aggregate["metric_counts"][name] = len(entries)
        if key == "task/noul" or key.startswith("task_modality/noul/"):
            counts = {name: sum(v.get(name, 0) for v in values) for name in confusion}
            tp, fp, fn, tn = (counts[n] for n in ("true_positive", "false_positive", "false_negative", "true_negative"))
            positive = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            negative = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
            aggregate.update(confusion=counts, macro_f1=(positive + negative) / 2)
        result[key] = aggregate
    return result


