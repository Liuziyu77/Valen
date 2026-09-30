"""Pinned upstream metrics plus public-subset and exact-distribution diagnostics."""
import math

from ._upstream.metrics import ece_top_label, latency_summary
from ._upstream.summarize import metric, summarize
from ._upstream.scoring import argmax_label


def summarize_run(tasks, records, tiers):
    report = summarize(tasks, records)
    report["per_tier"] = {tier: metric([t for t in tasks if tiers[t.id] == tier], records)
                          for tier in sorted(set(tiers[t.id] for t in tasks))}
    report["per_type"] = {kind: metric([t for t in tasks if t.question["type"] == kind], records)
                          for kind in sorted(set(t.question["type"] for t in tasks))}
    for name in ("compile_seconds", "forward_seconds", "answer_seconds"):
        values = [r["timing"][name] for r in records
                  if r.get("ok") and r.get("timing", {}).get(name) is not None]
        report[name] = {**latency_summary(values), "mean_s": sum(values) / len(values) if values else None}
    values = [r["latency_s"] for r in records]
    report["latency"]["mean_s"] = sum(values) / len(values) if values else None
    by_id = {r["task_id"]: r for r in records}
    gold_tasks = [t for t in tasks if t.provenance.get("gold_probs") and
                  t.expected is not None and not t.provenance.get("exclude_reason")]
    tvds, briers = [], []
    for task in gold_tasks:
        row = by_id.get(task.id)
        if row and row.get("valid"):
            gold, predicted = task.provenance["gold_probs"], row["probs"]
            tvds.append(sum(abs(predicted[k] - gold[k]) for k in task.labels) / 2)
            briers.append(sum((predicted[k] - gold[k]) ** 2 for k in task.labels))
    report["gold_distribution"] = {"n_planned": len(gold_tasks), "n_valid": len(tvds),
                                   "mean_tvd": sum(tvds) / len(tvds) if tvds else None,
                                   "brier_mean": sum(briers) / len(briers) if briers else None}
    # This diagnostic uses the actual public option counts, never the full
    # leaderboard's published baselines (which include inaccessible items).
    corrected = {}
    for tier, result in report["per_tier"].items():
        scorable = [t for t in tasks if tiers[t.id] == tier and t.expected is not None
                    and not t.provenance.get("exclude_reason")]
        chance = sum(1 / len(t.labels) for t in scorable) / len(scorable) if scorable else None
        accuracy = result["accuracy"]
        corrected[tier] = {"uniform_chance": chance,
                           "accuracy_above_chance": max(0., (accuracy - chance) / (1 - chance))
                           if accuracy is not None and chance is not None else None}
    report["chance_corrected_public_tiers"] = corrected
    hard = [t for t in tasks if tiers[t.id] == "hard"]
    pairs = [(max(by_id[t.id]["probs"].values()), argmax_label(by_id[t.id]["probs"]) == str(t.expected))
             for t in hard if t.expected is not None and not t.provenance.get("exclude_reason")
             and t.id in by_id and by_id[t.id].get("valid")]
    report["hard_ece"] = ece_top_label(pairs) if pairs else None
    report["scope"] = "public_subset"
    report["official_leaderboard_score"] = None
    report["score_note"] = "Public tasks only; sealed tasks and a measured serving tariff are unavailable."
    # Additional diagnostic; failures remain in upstream accuracy denominators.
    losses = []
    for task in tasks:
        row = by_id.get(task.id)
        if row and row.get("valid") and task.expected is not None and not task.provenance.get("exclude_reason"):
            losses.append(-math.log(max(row["probs"][str(task.expected)], 1e-12)))
    report["nll"] = {"n": len(losses), "mean": sum(losses) / len(losses) if losses else None,
                     "probability_floor": 1e-12}
    return report
