# Vendored unchanged from fstandhartinger/jevbench, revision
# bb05a335bc809e61b20c0f745d25499a82b326fc. See LICENSE in this directory.
"""Metrics for JevBench v1.

Conventions (documented in README):
  * Accuracy: argmax for classification families (policy/intent/extraction/
    routing/adequacy); ordinal uses expected value, scored both by rounded-EV
    accuracy and MAE.
  * Brier: multi-class sum over the exact label set, sum_k (p_k - y_k)^2.
    For binary questions (noul) we use the 2-class convention:
    brier = (p_yes - y)^2 + (p_no - (1 - y))^2, i.e. 2 * (p_yes - y)^2.
    This makes binary Brier directly comparable to the multi-class sum.
  * ECE: top-label confidence, 10 equal-width bins.
  * Latency: p50/p95 across all attempts; failures reported separately.
  * Paraphrase consistency: pairs where BOTH items are answered and BOTH are
    correct, over the count of scorable pairs. Items with expected=None are
    not scorable and excluded from the numerator.
  * No invented calibration: invalid distributions count as incorrect and as
    schema failures; they never get synthesized probabilities.
"""

from __future__ import annotations

import math
from collections import defaultdict


def macro_accuracy(per_family: dict) -> float:
    """Unweighted mean of per-family accuracies (families with >=1 scorable)."""
    vals = [f["accuracy"] for f in per_family.values() if f["n_scorable"] > 0]
    return sum(vals) / len(vals) if vals else 0.0


def brier_score(probs: dict, expected_label: str, labels: list) -> float:
    """Multi-class Brier sum over the exact label set.

    Binary 2-class convention: pass labels ["no", "yes"] with probs over both
    keys -> (p_yes - y)^2 + (p_no - (1 - y))^2 = 2 * (p_yes - y)^2.
    """
    if len(labels) == 2:
        # 2-class convention: (p_yes - y)^2 + (p_no - (1 - y))^2
        if expected_label not in labels:
            raise KeyError(f"expected label {expected_label!r} missing from probs")
        p_yes = probs.get(expected_label)
        if p_yes is None:
            raise KeyError(f"expected label {expected_label!r} missing from probs")
        no_label = labels[0] if labels[1] == expected_label else labels[1]
        p_no = probs.get(no_label, 1.0 - p_yes)
        return (p_yes - 1.0) ** 2 + (p_no - 0.0) ** 2
    total = 0.0
    for lab in labels:
        target = 1.0 if lab == expected_label else 0.0
        total += (float(probs.get(lab, 0.0)) - target) ** 2
    return total


def ece_top_label(pairs: list, n_bins: int = 10) -> dict:
    """Expected calibration error over (confidence, correct) pairs.

    10 equal-width bins on confidence in [0,1]. Returns dict with ece and
    per-bin stats (bin edges, n, mean confidence, accuracy).
    """
    bins = [
        {"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": 0, "conf_sum": 0.0, "correct": 0}
        for i in range(n_bins)
    ]
    for conf, correct in pairs:
        conf = min(max(float(conf), 0.0), 1.0)
        idx = min(int(conf * n_bins), n_bins - 1)
        b = bins[idx]
        b["n"] += 1
        b["conf_sum"] += conf
        b["correct"] += 1 if correct else 0
    n_total = sum(b["n"] for b in bins)
    ece = 0.0
    for b in bins:
        if b["n"]:
            acc = b["correct"] / b["n"]
            mean_conf = b["conf_sum"] / b["n"]
            ece += (b["n"] / n_total) * abs(acc - mean_conf)
    return {
        "ece": ece,
        "n": n_total,
        "bins": [
            {
                "lo": b["lo"],
                "hi": b["hi"],
                "n": b["n"],
                "mean_confidence": (b["conf_sum"] / b["n"]) if b["n"] else None,
                "accuracy": (b["correct"] / b["n"]) if b["n"] else None,
            }
            for b in bins
        ],
    }


def ordinal_mae(pairs: list) -> float:
    """Mean absolute error over (expected_level:int, predicted_ev:float)."""
    if not pairs:
        return None
    return sum(abs(e - p) for e, p in pairs) / len(pairs)


def percentile(values: list, q: float) -> float:
    if not values:
        return None
    vals = sorted(values)
    k = (len(vals) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)


def latency_summary(seconds: list) -> dict:
    ok = [v for v in seconds if v is not None]
    return {
        "n": len(ok),
        "p50_s": percentile(ok, 0.5),
        "p95_s": percentile(ok, 0.95),
    }


def paraphrase_consistency(results_by_id: dict, tasks_by_id: dict) -> dict:
    """Both-correct consistency over paraphrase pairs.

    A pair is countable when both members have a valid schema outcome.
    expected=None members make the pair unscorable (excluded from numerator,
    reported in counts).
    """
    groups = defaultdict(list)
    for tid, t in tasks_by_id.items():
        if t.group:
            groups[t.group].append(tid)
    n_pairs = 0
    n_both_answered = 0
    n_both_correct = 0
    n_unscorable = 0
    for g, ids in sorted(groups.items()):
        if len(ids) < 2:
            continue
        n_pairs += 1
        outcomes = [results_by_id.get(i) for i in ids]
        if any(o is None or not o.get("valid") for o in outcomes):
            continue
        n_both_answered += 1
        if any(o.get("correct") is None for o in outcomes):
            n_unscorable += 1
            continue
        if all(o.get("correct") for o in outcomes):
            n_both_correct += 1
    return {
        "pairs": n_pairs,
        "both_answered_valid": n_both_answered,
        "both_correct": n_both_correct,
        "unscorable_expected_none": n_unscorable,
        "consistency": (n_both_correct / n_both_answered) if n_both_answered else None,
    }
