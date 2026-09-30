"""Audit and compare different decision model families on identical public tasks."""
import argparse
from itertools import combinations
import json
from pathlib import Path

from .baselines import BASELINE_MAPPING_VERSION, native_probabilities, native_request
from .dataset import DEFAULT_ROOT, MAPPING_VERSION, file_hash, load_dataset, write_json
from .metrics import summarize_run
from ._upstream.scoring import score_task
from ._upstream.summarize import metric


def compare_models(runs, dataset, output):
    tasks, data_manifest = load_dataset(dataset)
    task_ids = {t.id for t in tasks}
    rows, manifests, predictions, audits = [], [], [], []
    for path in map(Path, runs):
        report = json.loads((path / "metrics.json").read_text())
        manifest = json.loads((path / "run_manifest.json").read_text())
        records = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
        by_id = {r["task_id"]: r for r in records}
        if (not (path / "complete.json").is_file() or not report["complete"] or report["smoke_subset"]
                or set(by_id) != task_ids or len(records) != len(task_ids)
                or manifest["planned_tasks"] != len(task_ids) or manifest["dataset"] != data_manifest):
            raise ValueError(f"Incomplete, duplicate or mismatched dataset: {path}")
        if manifest["mapping_version"] not in {MAPPING_VERSION, BASELINE_MAPPING_VERSION}:
            raise ValueError(f"Unsupported input mapping: {path}")
        ordered = [by_id[t.id] for t in tasks]
        for task, record in zip(tasks, ordered):
            if record["expected"] != task.expected:
                raise ValueError(f"Expected answer mismatch: {path}/{task.id}")
            if "request" in record and record["request"] != native_request(task):
                raise ValueError(f"Native input evidence/order mismatch: {path}/{task.id}")
            if record.get("ok"):
                if "response" in record and native_probabilities(task, record["response"]) != record["probs_as_returned"]:
                    raise ValueError(f"Native answer mismatch: {path}/{task.id}")
                score = score_task(record["probs_as_returned"], task)
                if any(record.get(key) != value for key, value in score.items()):
                    raise ValueError(f"Saved scoring mismatch: {path}/{task.id}")
        recomputed = summarize_run(tasks, ordered, data_manifest["tiers"])
        if any(report.get(key) != value for key, value in recomputed.items()):
            raise ValueError(f"Saved aggregate metrics mismatch: {path}")
        rows.append({**report, "run": str(path.resolve())})
        manifests.append(manifest)
        predictions.append(by_id)
        audits.append({"run": str(path.resolve()), "prediction_sha256": file_hash(path / "predictions.jsonl"),
                       "metrics_sha256": file_hash(path / "metrics.json"), "rescored": len(tasks)})
    if not rows or len({r["model"] for r in rows}) != len(rows):
        raise ValueError("Provide distinct model runs")
    keys = ("benchmark_revision", "dataset", "task_ids_sha256", "warmup_per_type", "hardware", "code_sha256", "versions")
    conditions = lambda m: {k: m[k] for k in keys}
    if any(conditions(m) != conditions(manifests[0]) for m in manifests[1:]):
        raise ValueError("Different dataset, hardware, evaluator source, warmup or library versions")
    paired = []
    for i, j in combinations(range(len(rows)), 2):
        paired.append({"first": rows[i]["model"], "second": rows[j]["model"],
                       "first_only_correct": sum(bool(predictions[i][t.id]["correct"]) and not predictions[j][t.id]["correct"] for t in tasks),
                       "second_only_correct": sum(bool(predictions[j][t.id]["correct"]) and not predictions[i][t.id]["correct"] for t in tasks)})
    truncated = set().union(*(set(r.get("truncation", {}).get("task_ids", [])) for r in rows))
    retained_tasks = [t for t in tasks if t.id not in truncated]
    result = {"scope": "public_subset", "benchmark_revision": rows[0]["benchmark_revision"],
              "hardware": rows[0]["hardware"], "versions": manifests[0]["versions"], "tasks": len(tasks),
              "models": rows, "paired_outcomes": paired, "audit": audits,
              "state_untruncated_subset": {"tasks": len(retained_tasks), "excluded_ids": sorted(truncated),
                                           "models": {r["model"]: metric(retained_tasks, list(p.values()))
                                                      for r, p in zip(rows, predictions)}},
              "official_leaderboard_score": None}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", result)
    def fmt(number, factor=1, digits=4):
        return "—" if number is None else f"{number * factor:.{digits}f}"
    lines = ["# JevBench：Valen、Laya 与 Intern-Decision", "",
             f"同一块 {result['hardware']}，每个模型评测 231 道公开题（Easy 48 / Standard 72 / Hard 111）。", "",
             "| 模型 | 正确数 | 总准确率 | Easy | Standard | Hard | Brier ↓ | ECE ↓ | 平均耗时 ms |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['n_correct']}/{r['n_scorable']} | {fmt(r['accuracy'],100,2)}% | "
                     + " | ".join(fmt(r["per_tier"][tier]["accuracy"],100,2)+"%" for tier in ("easy","standard","hard"))
                     + f" | {fmt(r['brier_mean'])} | {fmt(r['ece']['ece'])} | {fmt(r['latency']['mean_s'],1000,2)} |")
    lines += ["", "## 题型", "", "| 模型 | Choice | Noul | Score |", "| --- | ---: | ---: | ---: |"]
    for r in rows:
        lines.append("| " + r["model"] + " | " + " | ".join(fmt(r["per_type"][k]["accuracy"],100,2)+"%" for k in ("choice","noul","score")) + " |")
    lines += ["", "## Intern-Decision 原始概率（T=1）", "",
              "与默认校准共用一次前向计算；准确率不变。温度来自所下载模型的官方 inference.py，没有使用本评测集拟合。", "",
              "| 模型 | 默认 T | T=1 Brier | 默认 Brier | T=1 ECE | 默认 ECE |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        if r.get("adapter") == "intern_hf":
            raw = json.loads((Path(r["run"]) / "uncalibrated_metrics.json").read_text())
            raw_records = [json.loads(line) for line in (Path(r["run"]) / "uncalibrated_predictions.jsonl").read_text().splitlines()]
            raw_recomputed = summarize_run(tasks, raw_records, data_manifest["tiers"])
            if any(raw.get(k) != v for k, v in raw_recomputed.items()) or raw["accuracy"] != r["accuracy"]:
                raise ValueError("Uncalibrated replay metrics/accuracy mismatch")
            lines.append(f"| {r['model']} | {fmt(r['runtime_config']['temperature'])} | {fmt(raw['brier_mean'])} | "
                         f"{fmt(r['brier_mean'])} | {fmt(raw['ece']['ece'])} | {fmt(r['ece']['ece'])} |")
    lines += ["", "## 输入长度与测量范围", "",
              "Laya 使用英文主模型和原生 512-token 预算；其运行时也会裁剪问题/选项描述。"
              "以下只剔除运行时明确报告 state 截断的题目，不能保证问题/选项完整。", "",
              f"运行时报告 state 截断 {len(truncated)} 题；共同保留 {len(retained_tasks)} 题。", "",
              "| 模型 | 保留题准确率 |", "| --- | ---: |"]
    for r in rows:
        lines.append(f"| {r['model']} | {fmt(result['state_untruncated_subset']['models'][r['model']]['accuracy'],100,2)}% |")
    lines += ["", "延迟为逐题串行的本地请求耗时：包含原生输入编译、GPU 前向和概率转换，GPU 已同步；"
              "不计模型加载、warmup、评分和网络传输。所有模型使用相同库版本，每种题型各预热一次。"
              "各模型保留原生模板、选项语义、输入预算和默认概率处理。", "",
              "Valen 使用四个 joint/latest checkpoint，基座均为 Qwen3.5-2B，温度 T=1；"
              "bilinear 沿用较早的 warmup recipe，差异不能只归因于决策头结构。"
              "Laya 使用发布权重附带的温度；Intern-Decision 使用权重附带的 HF 接口、BF16、SDPA 和默认温度。"
              "本次库版本见 comparison.json；执行环境与作者发布表格可能不同。", "",
              "准确率使用官方 scorer 的 argmax（含 Score），失败计入准确率分母。"
              "总 Brier/ECE 覆盖 231 题；作者常用的 Hard-only 指标另存 JSON per_tier.hard/hard_ece。"
              "10 道有 gold_probs 的 Hard 题额外记录 TVD。没有私有题、judge tier 或服务价格，未计算官方榜单综合分。", "",
              "每题原始输入/响应/概率、校验过的模型版本与权重哈希、逐题评分及聚合复核均已保存。"]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compare_models(args.runs, args.dataset, args.output)


if __name__ == "__main__":
    main()
