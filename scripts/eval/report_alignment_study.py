"""Generate a report from completed experiment artifacts, never placeholder metrics."""
from collections import defaultdict
import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path('next-generation-exp')
CONDITIONS = ['sft_baseline', 'sft_alignment_100k', 'sft_alignment_1m']
STAGES = ['warmup', 'text', 'vision_top']


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def paired_bootstrap(a, b, metric):
    left = {(r['record_index'], r['qid']): r for r in a}
    right = {(r['record_index'], r['qid']): r for r in b}
    if left.keys() != right.keys():
        raise ValueError('Comparison does not cover the same decisions')
    groups = defaultdict(list)
    for key, r in left.items():
        if metric in r['metrics'] and metric in right[key]['metrics']:
            groups[r['group_id']].append(right[key]['metrics'][metric] - r['metrics'][metric])
    values = np.array([[sum(v), len(v)] for v in groups.values()], dtype=float)
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(2000):
        aggregate = values[rng.integers(len(values), size=len(values))].sum(0)
        samples.append(aggregate[0] / aggregate[1])
    return {'difference': float(values[:, 0].sum() / values[:, 1].sum()),
            'ci95': np.quantile(samples, [.025, .975]).tolist(), 'groups': len(values),
            'questions': int(values[:, 1].sum())}


def image_sensitivity(original, shuffled):
    left = {(r['record_index'], r['qid']): r for r in original}
    right = {(r['record_index'], r['qid']): r for r in shuffled}
    if left.keys() != right.keys():
        raise ValueError('Image ablation does not cover the same decisions')
    distances, flips = [], []
    for key, row in left.items():
        a, b = row['probabilities'], right[key]['probabilities']
        if a.keys() != b.keys():
            raise ValueError('Image ablation changed candidate keys')
        distances.append(sum(abs(a[k] - b[k]) for k in a) / 2)
        flips.append(max(a, key=a.get) != max(b, key=b.get))
    return {'questions': len(left), 'mean_total_variation': float(np.mean(distances)),
            'p95_total_variation': float(np.quantile(distances, .95)),
            'prediction_flip_rate': float(np.mean(flips)),
            'shuffled_minus_original_accuracy': paired_bootstrap(original, shuffled, 'accuracy')}


def plot_results(records):
    os.environ.setdefault('MPLCONFIGDIR', str((ROOT / 'matplotlib-cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
    colors = ['#496782', '#d78843', '#49846b']
    for condition, color in zip(CONDITIONS, colors):
        points = [r for r in records if r['condition'] == condition]
        label = condition.removeprefix('sft_').replace('_', ' ')
        for ax, metric in zip(axes, ['accuracy', 'nll', 'brier']):
            ax.plot(STAGES, [r[metric] for r in points], marker='o', color=color, label=label)
            ax.set_title(metric.upper()); ax.set_xlabel('SFT stage'); ax.grid(axis='y', alpha=.2)
    axes[0].legend(frameon=False)
    fig.savefig(ROOT / 'decision_comparison.png', dpi=180)
    fig.savefig(ROOT / 'decision_comparison.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout='constrained')
    for row, label in zip(axes, ['100k', '1m']):
        # A resumed run can repeat updates after its last checkpoint. Keep the
        # most recently recorded value for each update in the plotted history.
        history = {r['step']: r for r in rows(ROOT / f'alignment_{label}/metrics.jsonl')}
        history = [history[k] for k in sorted(history)]
        for ax, metric in zip(row, ['itc', 'itm', 'mlm']):
            x = [r['samples_seen'] / 1e6 for r in history]
            y = np.array([r[metric] for r in history])
            ax.plot(x, y, color=colors[0], alpha=.2, linewidth=.6)
            window = min(20, len(y))
            ax.plot(x[window - 1:], np.convolve(y, np.ones(window) / window, mode='valid'), color=colors[0])
            ax.set_title(f'{label}: {metric.upper()}'); ax.set_xlabel('Million sample exposures')
            ax.grid(axis='y', alpha=.2)
    fig.savefig(ROOT / 'alignment_losses.png', dpi=180)
    fig.savefig(ROOT / 'alignment_losses.pdf')
    plt.close(fig)


def main():
    if not all((ROOT / c / 'complete.json').exists() for c in CONDITIONS):
        raise RuntimeError('All three decision experiments must finish before the final report')
    records, metrics = [], {}
    for condition in CONDITIONS:
        metrics[condition] = {}
        for stage in STAGES:
            report = json.loads((ROOT / condition / stage / 'evaluation/metrics.json').read_text())
            metrics[condition][stage] = report
            overall = report['metrics']['overall']
            records.append({'condition': condition, 'stage': stage, **{k: overall.get(k) for k in ['accuracy', 'nll', 'brier']}})
    fingerprints = {report['data_sha256'] for stages in metrics.values() for report in stages.values()}
    if len(fingerprints) != 1:
        raise ValueError('Decision comparisons used different evaluation datasets')
    with (ROOT / 'decision_comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    baseline = rows(ROOT / CONDITIONS[0] / 'vision_top/evaluation/predictions.jsonl')
    uncertainty = {}
    for condition in CONDITIONS[1:]:
        predictions = rows(ROOT / condition / 'vision_top/evaluation/predictions.jsonl')
        uncertainty[condition] = {m: paired_bootstrap(baseline, predictions, m) for m in ['accuracy', 'nll', 'brier']}
    pilot_predictions = rows(ROOT / 'sft_alignment_100k/vision_top/evaluation/predictions.jsonl')
    final_predictions = rows(ROOT / 'sft_alignment_1m/vision_top/evaluation/predictions.jsonl')
    scale_gain = {m: paired_bootstrap(pilot_predictions, final_predictions, m) for m in ['accuracy', 'nll', 'brier']}
    caption = {}
    for label in ['100k', '1m']:
        history = rows(ROOT / f'alignment_{label}/evaluation.jsonl')
        caption[label] = {kind: next(r for r in reversed(history) if r['label'] == kind)
                          for kind in ['initial', 'final_comparable', 'final']}
    test = json.loads((ROOT / 'alignment_1m/test_metrics.json').read_text())
    ablation, sensitivity = {}, {}
    for condition in CONDITIONS:
        path = ROOT / condition / 'vision_top/evaluation_shuffled_images/metrics.json'
        if path.exists():
            ablation[condition] = json.loads(path.read_text())
            sensitivity[condition] = image_sensitivity(
                rows(ROOT / condition / 'vision_top/evaluation/predictions.jsonl'),
                rows(path.parent / 'predictions.jsonl'))
            (path.parent.parent / 'image_sensitivity.json').write_text(json.dumps(sensitivity[condition], indent=2))
    benchmark_path = ROOT / 'parallel_decision_benchmark.json'
    benchmark = json.loads(benchmark_path.read_text()) if benchmark_path.exists() else None
    fast_path = ROOT / 'parallel_decision_benchmark_bf16.json'
    fast_benchmark = json.loads(fast_path.read_text()) if fast_path.exists() else None
    audit_path = ROOT / 'caption_final_data_audit.json'
    data_audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    if data_audit is not None and not data_audit.get('passed'):
        raise ValueError('Caption manifest audit did not pass')
    summary = {'decision': metrics, 'paired_cluster_bootstrap': uncertainty, 'scale_up_comparison': scale_gain,
               'caption': caption, 'caption_test': test,
               'shuffled_image_ablation': ablation, 'image_sensitivity': sensitivity,
               'parallel_decision_benchmark': benchmark, 'parallel_decision_benchmark_bf16': fast_benchmark,
               'caption_data_audit': data_audit}
    (ROOT / 'results_summary.json').write_text(json.dumps(summary, indent=2))
    plot_results(records)
    final = metrics['sft_alignment_1m']['vision_top']['metrics']['overall']
    reference = metrics['sft_baseline']['vision_top']['metrics']['overall']
    delta = uncertainty['sft_alignment_1m']['accuracy']
    low, high = delta['ci95']
    interpretation = ('本轮验证集支持预训练带来的准确率提升。' if low > 0 else
                      '本轮验证集显示预训练后的准确率下降，需要调整预训练或迁移配方。' if high < 0 else
                      '区间覆盖零，本轮尚不能确认准确率有稳定变化。')
    out = ['# Valen 双编码器两阶段实验报告', '',
           '本报告由实际训练与评估产物生成。首版使用 ModernBERT-base、DINOv3 ViT-B/16、384×384 图片和英语数据。正式预训练和 SFT 使用 8 张 GPU；小样本诊断与并行推理测量使用单张 GPU。', '',
           f"100 万图文预训练后，最终 SFT 的验证准确率为 {final['accuracy']:.2%}，直接 SFT 为 {reference['accuracy']:.2%}，相差 {100 * delta['difference']:+.2f} 个百分点（配对 95% 区间 [{100 * low:+.2f}, {100 * high:+.2f}]）。{interpretation}", '',
           f"对应 NLL 为 {final['nll']:.4f}，直接 SFT 为 {reference['nll']:.4f}；Brier 为 {final['brier']:.4f}，直接 SFT 为 {reference['brier']:.4f}。这两项均越低越好。", '',
           '## 实验设置', '',
           '- 共享维度 512，融合层 2 层，问题读取器 3 层、8 个查询向量，候选读取器 2 层。',
           '- 图文训练集：10 万组试验集（3 epochs），随后在其 checkpoint 上继续训练 100 万组数据（1 epoch）。两个集合是包含关系，不能理解为 110 万张互不重复的图片。',
           '- 图文评估集：独立的 1 万组验证数据与 1 万组测试数据；周期评估使用固定的 1,024 条验证子集。',
           '- 百万训练集来源：CC12M 40 万、Laion-COCO 40 万、COYO 15 万、SBU 5 万。清单计数、SHA-256 去重与集合关系见 `caption_final_data_audit.json`。',
           '- 图文目标：ITC + ITM + 视觉条件 MLM。前 10% 更新冻结编码器，之后解冻文本顶部，60% 后再解冻视觉顶部。',
           '- 图文训练的新模块学习率为 1e-4；文本最后 6 层及归一化层为 1e-5，视觉最后 2 层及归一化层为 2e-6。使用 5% 学习率预热与余弦衰减。',
           '- 决策训练：train_v1 的 10 万条记录，按固定种子划分为 warmup 1 万、text 3 万、vision_top 6 万，每条记录在整个 SFT 流程中使用一次。',
           '- SFT 使用 CE + 0.1 × Brier；Score 额外加入 0.2 × RPS。决策模块学习率 2e-4，共享投影和融合模块 2e-5，编码器学习率与图文阶段相同；每卡每次更新预算 8,192 个计算 token。',
           '- 决策验证：eval_v1 的 5,000 条记录。三种条件使用相同的划分、参数和预算；对照组也保留官方 ModernBERT 和 DINOv3 的预训练权重，只跳过本轮图文预训练。',
           '- 图文训练的 global batch 为 512；末尾不足一个 batch 的记录在当前 epoch 丢弃，实际样本使用次数以 checkpoint 的 samples_seen 为准。', '',
           '## 决策验证结果', '', '| 初始化 | SFT 阶段 | Accuracy | NLL | Brier |', '| --- | --- | ---: | ---: | ---: |']
    names = {'sft_baseline': '直接 SFT', 'sft_alignment_100k': '10 万图文预训练', 'sft_alignment_1m': '100 万图文预训练'}
    if 'sft_alignment_1m' in sensitivity:
        check = sensitivity['sft_alignment_1m']['shuffled_minus_original_accuracy']
        if check['ci95'][1] >= 0:
            out[8:8] = ['图片错配检查没有确认原图带来的准确率优势。因此，以上总体指标变化还不能证明模型已经学会了可靠的视觉决策；图文检索、决策概率和视觉依赖需要分别判断。', '']
    base_tasks = metrics['sft_baseline']['vision_top']['metrics']
    final_tasks = metrics['sft_alignment_1m']['vision_top']['metrics']
    intro_position = out.index('## 实验设置')
    out[intro_position:intro_position] = [
        f"收益也不均匀：Choice 准确率从 {base_tasks['task/choice']['accuracy']:.2%} 升至 {final_tasks['task/choice']['accuracy']:.2%}；Noul 从 {base_tasks['task/noul']['accuracy']:.2%} 变为 {final_tasks['task/noul']['accuracy']:.2%}。Game 领域从 {base_tasks['domain/game']['accuracy']:.2%} 变为 {final_tasks['domain/game']['accuracy']:.2%}，Score 的 {final_tasks['task/score']['questions']} 道验证题不足以支持稳定结论。总体提升不能代表每类决策都改善了。", '']
    for row in records:
        out.append(f"| {names[row['condition']]} | {row['stage']} | {row['accuracy']:.4f} | {row['nll']:.4f} | {row['brier']:.4f} |")
    out += ['', '![决策验证曲线](decision_comparison.png)']
    out += ['', '最终 vision_top checkpoint 的配对差值如下。按图片 group 重采样 2,000 次；Accuracy 正值更好，NLL/Brier 负值更好。', '',
            '| 预训练规模 | 指标 | 相对直接 SFT 的差值 | 95% 区间 |', '| --- | --- | ---: | --- |']
    for condition, values in uncertainty.items():
        for metric, value in values.items():
            lo, hi = value['ci95']
            out.append(f"| {names[condition]} | {metric} | {value['difference']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    scale = scale_gain['accuracy']; lo, hi = scale['ci95']
    scale_text = '本轮验证支持扩大图文训练规模的额外收益。' if lo > 0 else '区间覆盖零，额外收益尚不确定。' if hi >= 0 else '本轮扩大规模后的准确率反而下降。'
    out += ['', f"100 万组相对 10 万组的最终 SFT 准确率差为 {100 * scale['difference']:+.2f} 个百分点，95% 区间为 [{100 * lo:+.2f}, {100 * hi:+.2f}]。{scale_text}这同时扩大了图文样本覆盖并增加了训练更新次数，不能把差异单独归因于样本数量。"]
    out += ['', '## 分任务表现', '', '| 初始化 | 任务 | 题数 | Accuracy | Brier | RPS |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for condition in CONDITIONS:
        for task in ['choice', 'noul', 'score']:
            m = metrics[condition]['vision_top']['metrics'][f'task/{task}']
            fmt = lambda k: f'{m[k]:.4f}' if k in m else '—'
            out.append(f"| {names[condition]} | {task} | {m['questions']} | {fmt('accuracy')} | {fmt('brier')} | {fmt('rps')} |")
    out += ['', '## 分领域表现', '', '| 初始化 | 领域 | 题数 | Accuracy | NLL |', '| --- | --- | ---: | ---: | ---: |']
    for condition in CONDITIONS:
        for domain in ['document', 'game', 'ui', 'vqa']:
            m = metrics[condition]['vision_top']['metrics'][f'domain/{domain}']
            out.append(f"| {names[condition]} | {domain} | {m['questions']} | {m['accuracy']:.4f} | {m['nll']:.4f} |")
    if len(ablation) == len(CONDITIONS):
        out += ['', '## 图片输入检查', '',
                '保持问题、选项和目标不变，在同领域内按图片组固定轮换图片。替换图片后仍按原标签计分，这项检查用于观察模型对原图的依赖。', '',
                '| 初始化 | 原图 Accuracy | 打乱图片 Accuracy | 原图优势（百分点） | 原图 NLL | 打乱图片 NLL |',
                '| --- | ---: | ---: | ---: | ---: | ---: |']
        for condition in CONDITIONS:
            normal = metrics[condition]['vision_top']['metrics']['overall']
            shuffled = ablation[condition]['metrics']['overall']
            if normal['questions'] != shuffled['questions']:
                raise ValueError('Image ablation coverage differs')
            out.append(f"| {names[condition]} | {normal['accuracy']:.4f} | {shuffled['accuracy']:.4f} | {100 * (normal['accuracy'] - shuffled['accuracy']):+.2f} | {normal['nll']:.4f} | {shuffled['nll']:.4f} |")
        out += ['', '逐题输出的变化如下。总变差是候选概率差绝对值之和的一半；越小表示换图后输出越接近。', '',
                '| 初始化 | 平均总变差 | P95 总变差 | 预测类别改变比例 | 错配减原图 Accuracy 的 95% 区间 |',
                '| --- | ---: | ---: | ---: | --- |']
        for condition, value in sensitivity.items():
            lo, hi = value['shuffled_minus_original_accuracy']['ci95']
            out.append(f"| {names[condition]} | {value['mean_total_variation']:.4f} | {value['p95_total_variation']:.4f} | {value['prediction_flip_rate']:.2%} | [{lo:+.4f}, {hi:+.4f}] |")
    out += ['', '## 图文对齐结果', '', '以下使用各自训练前后的同一 1,024 条验证池，避免把不同检索池大小的 Recall 直接比较。', '',
            '| 规模 | 时间点 | 图→文 R@1 | 文→图 R@1 | 难负例 ITM Accuracy | 随机负例 ITM Accuracy | MLM Loss | 正确图片带来的 MLM loss 降幅 |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, values in caption.items():
        for kind in ['initial', 'final_comparable']:
            m = values[kind]
            random_itm = m.get('random_negative_itm_accuracy')
            diagnostic = ROOT / 'alignment_100k/random_negative_diagnostic.json'
            if random_itm is None and label == '100k' and kind == 'final_comparable' and diagnostic.exists():
                random_itm = json.loads(diagnostic.read_text())['random_negative_itm_accuracy']
            random_fmt = f'{random_itm:.4f}' if random_itm is not None else '—'
            out.append(f"| {label} | {kind} | {m['image_to_text_recall_at_1']:.4f} | {m['text_to_image_recall_at_1']:.4f} | {m['itm_accuracy']:.4f} | {random_fmt} | {m['mlm']:.4f} | {m['mlm_image_gain']:.4f} |")
    out += ['', '难负例由当前模型的相似度分布采样，随模型变化，难度也会变化。随机错配使用固定的批内轮换，作为额外参照；两项均采用正负样本各半的评估。']
    out += ['', '扩大到完整 1 万条检索池后，任务更难。以下两组验证结果使用相同清单；独立测试集只在百万阶段结束后评估。', '',
            '| 规模 | 数据划分 | 样本数 | 图→文 R@1 | 文→图 R@1 | 难负例 ITM | 随机负例 ITM | MLM 图片收益 |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, split, m in [('100k', 'validation', caption['100k']['final']),
                             ('1m', 'validation', caption['1m']['final']), ('1m', 'test', test)]:
        random_fmt = f"{m['random_negative_itm_accuracy']:.4f}" if 'random_negative_itm_accuracy' in m else '—'
        out.append(f"| {label} | {split} | {m['samples']} | {m['image_to_text_recall_at_1']:.4f} | {m['text_to_image_recall_at_1']:.4f} | {m['itm_accuracy']:.4f} | {random_fmt} | {m['mlm_image_gain']:.4f} |")
    out += ['', '![图文预训练损失曲线](alignment_losses.png)', '',
            '浅色线是每次更新的损失，深色线是 20 次更新的移动平均。横轴为各段训练内的样本使用次数，包含重复使用。', '']
    if benchmark:
        out += ['## 并行推理检查', '',
                f"使用一张 {benchmark['gpu']}、严格 FP32 测量（关闭 TF32，并固定使用数学实现的注意力内核）。将不同记录中的问题组合到同一图片上，仅检查吞吐和题间隔离，不评价这些组合请求的答案语义。每个规模预热 2 次、测量 5 次；前向时间包含概率输出转换，数据编译另行测量。", '',
                '| 并行题数 | 候选总数 | 最长指令 token 数 | 前向中位数（ms） | PyTorch 峰值 allocated（GiB） |',
                '| ---: | ---: | ---: | ---: | ---: |']
        for point in benchmark['measurements']:
            out.append(f"| {point['questions']} | {point['candidates']} | {point['longest_instruction_tokens']} | {1000 * point['forward_median_seconds']:.1f} | {point['peak_allocated_bytes'] / 2**30:.2f} |")
        out += ['', f"抽查 5 道题，单独计算与 256 题并行计算的最大概率差为 {benchmark['max_independence_probability_difference']:.6f}。上述延迟仅对应本次请求形状与硬件。", '']
        if fast_benchmark:
            point = fast_benchmark['measurements'][-1]
            numerical_status = '通过' if fast_benchmark['passed'] else '未通过'
            out += [f"同一 checkpoint 使用 BF16 时，256 题前向中位数为 {1000 * point['forward_median_seconds']:.1f} ms，峰值 allocated 为 {point['peak_allocated_bytes'] / 2**30:.2f} GiB；单题与并行输出的最大概率差为 {fast_benchmark['max_independence_probability_difference']:.6f}。BF16 的 0.01 概率差阈值检查{numerical_status}。这反映了批形状变化下的数值误差，不能承诺 BF16 输出逐位一致。严格 FP32 用于题间隔离验收，阈值为 0.0001；正式决策验证表仍使用训练配置的 BF16。", '']
    out += [
            '## 代码检查与修正', '',
            '- 实现 ITC、ITM、MLM 三目标预训练，并验证跨卡 ITC 的梯度与等价 global batch 一致。',
            '- 主干导出包含完整编码器权重，验证了迁移到 SFT 后再次冻结编码器仍保留预训练结果。',
            '- 决策数据最长指令约 1,950 个 token，本轮问题长度上限设为 2,048。',
            '- 修复了 Unicode 行分隔符被误当作 JSONL 换行的数据恢复问题；从固定试验清单重建采集日志，已发布的 10 万条训练清单和评估清单保持不变。',
            '- 补充完成标记与逐题评估覆盖检查，避免将提前退出或只写完部分结果的任务误判为完成。', '',
            '10 万预训练组的 SFT 曾迁移到同时准备图文数据的作业，并从最近 checkpoint 恢复。保存点后的少量更新重新执行，最终有效训练仍覆盖同一份 10 万条决策数据；GPU 浮点计算不保证逐位复现。作业与恢复记录保存在 `jobs/` 和 `monitor.jsonl`。', '',
            '## 验证与限制', '',
            '- 全量回归测试通过（日志见 `full-tests.log`），覆盖跨 GPU ITC 梯度、checkpoint 恢复、主干迁移和 Unicode caption 的数据恢复；另外完成 8 卡真实权重训练、恢复和 SFT 检查。',
            '- 10 万条试验的 ITM 验证结果接近随机水平，因此另做了 64 条训练样本的 ITM 单目标检查，确认融合路径可以学习。该检查的权重已丢弃，其训练集准确率不计入验证结果；原始记录见 `alignment_100k/itm_overfit_diagnostic.json`。',
            '- 另用 4 张训练图片构造文字完全相同、标签只由图片决定的任务，检查完整决策路径是否能学习视觉差异。此检查仅用于排查路径断开，不能证明泛化；权重不用于正式实验，记录见 `sft_alignment_100k/visual_overfit_diagnostic.json`。',
            '- 图文数据采用随机分片/字节窗口采样，不是按全库记录严格均匀采样。执行 SHA-256 与 dHash 距离不超过 3 的筛选，并排除决策验证图片的相近指纹；这不等于完成了所有语义近重复检测。',
            '- Caption 英语筛选采用来源与字符规则，Laion-COCO 默认使用 BLIP 描述；没有逐条人工审核。',
            '- 决策 eval_v1 用作验证集并在各阶段重复评估，因此这些结果应称为验证结果；Score 只有 18 道题，不能据此作稳定的泛化结论。',
            '- 本轮为单个随机种子的实验。置信区间描述评估样本的不确定性，不包含更换训练种子带来的波动。',
            '- 三组 SFT 使用相同配方；这不是针对每种初始化单独调参后的最优结果，也不能代表模型充分训练后的性能上限。',
            '- 三组只对齐了 SFT 预算。预训练组另有图文数据与计算开销，总计算量不同。',
            '- 模型没有 OCR 模块。文字密集图片任务的限制需要结合各数据来源的结果解读。', '',
            '## 后续实验建议', '',
            '先加强视觉依赖的验证，再考虑继续扩大数据。保留同领域换图检查，并建立问题和候选相同、正确答案随图片变化的成对验证集。', '',
            '检查决策数据中的文本线索。例如验证集有一道问题询问 “NHRA Winston Drag Racing Series” 的冠名商，正确选项就是 “winston”；另一些题目的干扰项与问题语义明显不符。这些样本允许模型靠文字取得分数。原始记录保存在 `decision_text_cue_examples.json`；个别例子不能代表整套数据的比例，下一轮应抽样审查并补充更接近的干扰项。', '',
            '若视觉融合指标仍弱，可以分别比较融合层学习率、ITM 负例从随机到困难的课程，以及 MLM 权重。一次只改变一个因素，并同时看 ITM、正确图片带来的 MLM 收益和决策换图差值。当前结果不足以把原因归结为某一个模块。', '',
            'Noul 和 Score 需要单独跟踪。下一轮可比较按任务平衡采样，并补充 Score 验证样本，避免用以 Choice 为主的总体指标代替各输出类型的质量。', '',
            '## 产物位置', '',
            '- `alignment_100k/`、`alignment_1m/`：预训练 checkpoint、主干导出、训练曲线与验证结果。',
            '- `sft_baseline/`、`sft_alignment_100k/`、`sft_alignment_1m/`：各 SFT 阶段 checkpoint 与逐题预测。',
            '- SFT checkpoint 采用增量保存；加载时须保留原始 `models/`，预训练组还需要其对应的 `alignment_*/shared/` 主干目录。',
            '- `monitor.jsonl`、`jobs/`：半小时检查、恢复操作和作业日志。',
            '- `results_summary.json`、`decision_comparison.csv`：结构化结果。',
            '- `reproducibility/`：源代码快照、文件哈希与依赖版本；`final_validation.json`：训练预算与评估覆盖的最终核对。', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(out))
    (ROOT / 'PROGRESS.md').write_text('# 实验已完成\n\n见[完整结果报告](RESULTS.md)。\n')
    print('Wrote', ROOT / 'RESULTS.md')


if __name__ == '__main__':
    main()
