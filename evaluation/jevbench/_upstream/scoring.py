# Vendored unchanged from fstandhartinger/jevbench, revision
# bb05a335bc809e61b20c0f745d25499a82b326fc. See LICENSE in this directory.
"""Distribution validation, argmax and ordinal expected-value scoring.

All scoring is deterministic and pure. Malformed distributions fail closed:
they are invalid and count as incorrect; callers must not invent calibration.
"""

from __future__ import annotations

import math

SUM_TOL = 1e-3
# v1 froze SUM_TOL = 1e-3 before the run. The run then showed what that
# actually measures: several models round their probabilities to three
# decimals, so a nine-option answer lands on 0.999 or 1.005 and would be
# thrown out for arithmetic rather than for judgement. RENORM_TOL is the band
# inside which a distribution is treated as a rounded one - rescaled to sum to
# 1 and scored normally. Outside it, the answer is still invalid and still
# counts as wrong. Both tolerances are reported for every model, so the
# pre-registered strict number stays visible next to the headline.
RENORM_TOL = 2e-2


class InvalidDistribution(ValueError):
    pass


def validate_probs(probs: dict, labels: list, sum_tol: float = SUM_TOL) -> dict:
    """Validate a probability map against the exact label set.

    Keys must match labels exactly (no missing, no extra). Values must be
    finite floats in [0, 1] and sum to 1 within sum_tol. Returns a sanitized
    copy with plain float values.
    """
    if not isinstance(probs, dict):
        raise InvalidDistribution("probs is not a dict")
    want = set(labels)
    got = set(probs.keys())
    if got != want:
        raise InvalidDistribution(
            f"label keys mismatch: missing={sorted(want - got)} extra={sorted(got - want)}"
        )
    out = {}
    total = 0.0
    for k, v in probs.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise InvalidDistribution(f"prob[{k!r}] is not a number: {v!r}")
        f = float(v)
        if not math.isfinite(f):
            raise InvalidDistribution(f"prob[{k!r}] is not finite")
        if f < 0.0 or f > 1.0:
            raise InvalidDistribution(f"prob[{k!r}] out of [0,1]: {f}")
        out[str(k)] = f
        total += f
    if abs(total - 1.0) > sum_tol:
        raise InvalidDistribution(f"probs sum to {total}, tolerance {sum_tol}")
    return out


def argmax_label(probs: dict) -> str:
    """Deterministic argmax: ties broken by lexicographically smallest label."""
    best, best_p = None, -1.0
    for k in sorted(probs.keys()):
        if probs[k] > best_p:
            best, best_p = k, probs[k]
    return best


def expected_value(probs: dict) -> float:
    """Expected value over ordinal score levels keyed by string indices."""
    return sum(float(k) * v for k, v in probs.items())


def top_label_confidence(probs: dict) -> float:
    return max(probs.values()) if probs else 0.0


def score_task(probs: dict, task) -> dict:
    """Score one distribution against a canonical task.

    Returns: valid (inside RENORM_TOL, the headline rule), strict_valid
    (inside SUM_TOL, the pre-registered rule), renormalized, correct (None
    when expected is None), predicted, ordinal_ev (score family only).
    Distributions outside RENORM_TOL are invalid and count as wrong; they are
    never repaired into a distribution.
    """
    try:
        clean = validate_probs(probs, task.labels)
        strict_valid, renormalized = True, False
    except InvalidDistribution as strict_error:
        try:
            clean = validate_probs(probs, task.labels, sum_tol=RENORM_TOL)
        except InvalidDistribution:
            return {"valid": False, "strict_valid": False, "renormalized": False,
                    "error": str(strict_error), "correct": False, "predicted": None}
        total = sum(clean.values())
        if total <= 0:
            return {"valid": False, "strict_valid": False, "renormalized": False,
                    "error": "probabilities sum to zero", "correct": False,
                    "predicted": None}
        clean = {k: v / total for k, v in clean.items()}
        strict_valid, renormalized = False, True

    # `probs` is the distribution every downstream metric uses: the model's own
    # numbers, rescaled only when they were inside the rounding band.
    base = {"valid": True, "strict_valid": strict_valid,
            "renormalized": renormalized, "probs": clean}
    if task.expected is None:
        pred = argmax_label(clean) if task.question["type"] != "score" else None
        out = {**base, "correct": None, "predicted": pred}
        if task.question["type"] == "score":
            out["ordinal_ev"] = expected_value(clean)
        return out

    out = dict(base)
    if task.question["type"] == "score":
        ev = expected_value(clean)
        out["ordinal_ev"] = ev
        out["predicted"] = argmax_label(clean)
        out["correct"] = out["predicted"] == str(task.expected)
    else:
        pred = argmax_label(clean)
        out["predicted"] = pred
        out["correct"] = pred == task.expected
    return out


def score_label(label, task) -> dict:
    """Score a label-only answer (a system with no distribution, e.g. Needle 3).

    Valid when the label is in the exact label set. No probabilities are
    produced here or anywhere downstream, so calibration metrics skip it.
    An abstention (label None) is a valid-but-empty answer that counts wrong.
    """
    ok = label is not None and str(label) in [str(x) for x in task.labels]
    out = {"valid": ok, "strict_valid": ok, "renormalized": False, "probs": None,
           "predicted": str(label) if ok else None}
    if task.expected is None:
        out["correct"] = None
    else:
        out["correct"] = ok and str(label) == str(task.expected)
    if not ok:
        out["error"] = "abstained" if label is None else "label outside the label set"
    return out
