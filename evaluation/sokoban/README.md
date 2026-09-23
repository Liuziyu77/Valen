# Sokoban 数据与完整游戏评测

在仓库根目录执行以下命令，并激活已安装项目依赖的环境。环境、求解器、生成器、数据校验和参考策略仅需 CPU；模型策略另需已准备的模型和 checkpoint。

## 代码

| 模块 | 用途 |
| --- | --- |
| `env.py`、`render.py` | 确定性规则、符号状态和截图渲染 |
| `solver.py`、`generator.py` | 精确搜索、最优动作集合及关卡生成 |
| `dataset.py`、`build_dataset.py` | Visual-Jev 记录格式及训练/测试数据生成 |
| `validate_dataset.py` | 标签、轨迹、媒体哈希与跨划分去重校验 |
| `evaluate.py` | 完整游戏评测和四种策略适配器 |
| `compare.py` | 完整分片合并、两组策略结果的配对比较 |
| `replay.py` | 轨迹核验、GIF 回放和手动游玩 |

## 生成与校验数据

先用小规模数据验证流程：

```bash
python -m evaluation.sokoban.build_dataset \
  --train-count 40 --test-count 4 --workers 1 --seed 123

python -m evaluation.sokoban.validate_dataset
```

默认输出到 `data/train_sokoban/` 和 `data/eval_sokoban/`。训练数量必须是 20 的正整数倍，测试数量需为正整数。不带数量参数时生成 30,000 条训练记录和 200 个测试关卡。生成器拒绝覆盖非空目录；正式生成时使用新的 `--train-dir` 和 `--eval-dir`。

训练记录来自最优轨迹、偏离后可解和失败纠正三类状态，配比为 80/15/5；每张截图对应一道下一动作 Choice，多种同样最优的动作均分标签概率。纠正样本展示失败前仍可解的状态，不为已经死锁的状态编造恢复动作。

训练和测试按墙体与目标布局的旋转/镜像规范化指纹隔离。生成器通过逆向多源 BFS 获取精确移动距离；搜索预算外的状态标记为未知。独立校验重新计算最优动作集合、回放来源轨迹并检查跨划分去重。只有校验成功后才删除 `BUILDING` 标记。

测试集包含 `levels.jsonl`、`reference.jsonl` 和 `manifest.json`。模型只接收截图、图例、规则和候选动作；符号状态、最优路径和参考标签留在评测侧。

## 运行完整游戏

```bash
# CPU 参考策略检查环境和评测流程，不代表视觉模型成绩。
python -m evaluation.sokoban.evaluate \
  --policy oracle --output evaluation/sokoban/results/oracle

python -m evaluation.sokoban.evaluate \
  --policy random --seed 0 --output evaluation/sokoban/results/random

# 替换为需要评测的已训练 checkpoint。
python -m evaluation.sokoban.evaluate \
  --policy visualjev --checkpoint output/sft_warmup/latest \
  --output evaluation/sokoban/results/visualjev

# 使用与决策头相同的基础模型及图片预处理配置。
python -m evaluation.sokoban.evaluate \
  --policy qwen --model-path models/Qwen3.5-0.8B \
  --media-config output/sft_warmup/latest/config.json \
  --output evaluation/sokoban/results/qwen
```

每次运行使用新的输出目录。自定义数据路径用 `--eval-dir` 指定。模型示例中的 smoke checkpoint 只适合检查调用流程；正式实验需使用预先训练并确定选择规则的 checkpoint。

每关从初态开始，默认生成的关卡最多执行 200 步。每局一次尝试，不自动重开、纠错或屏蔽非法动作。Visual-Jev 选择概率最大的动作；Qwen 使用无 thinking 的贪心生成，要求输出一个方向词。格式错误或推理异常计为该局失败。

`--limit` 仅用于接口短测，结果标记为 `smoke_subset`，不能作为完整评估合并。`--wall-timeout` 默认 1,200 秒，在模型调用之间及动作执行前检查，不会抢占正在执行的模型调用。

`--memoize` 可复用完全相同图片、规则和候选顺序下的确定性模型输出；每个进程首次缓存命中会重新推理核验。默认不启用，随机和 oracle 策略不支持。跨策略比较需保持缓存设置一致。

## 分片、合并与比较

大数据集可分片运行；每个进程加载一个模型，GPU 分配由调用方控制。以下展示 CPU 参考策略的两个完整分片：

```bash
python -m evaluation.sokoban.evaluate --policy oracle \
  --num-shards 2 --shard-index 0 \
  --output evaluation/sokoban/results/oracle_sharded/shard_0
python -m evaluation.sokoban.evaluate --policy oracle \
  --num-shards 2 --shard-index 1 \
  --output evaluation/sokoban/results/oracle_sharded/shard_1

python -m evaluation.sokoban.compare merge \
  --directory evaluation/sokoban/results/oracle_sharded

python -m evaluation.sokoban.compare compare \
  --first evaluation/sokoban/results/visualjev \
  --second evaluation/sokoban/results/qwen \
  --output evaluation/sokoban/results/comparison
```

分片目录必须命名为 `shard_0`、`shard_1` 等，全部结束后才能合并。工具检查数据版本、分片设置与关卡覆盖，拒绝缺片、重复、错配和短测子集。比较支持完整单次运行或包含全部分片的目录；若输入是分片目录，会先在该目录写出汇总文件。

比较输出 `comparison.json` 和 `report.md`，使用实际关卡数和策略身份，包含通关率差、配对通关情况及双侧精确 McNemar 检验。数据、图片预处理、缓存、超时和 runner 版本需一致。

## 输出与回放

评测输出 `config.json`、逐关 `episodes.jsonl`、汇总 `metrics.json` 和每关轨迹目录。主指标是通关率及 Wilson 95% 区间；同时报告无效动作率、成功局路径效率、重复状态和已证明静态死锁率。静态死锁率是下界，不能据此断言未检测到的状态都可解。

```bash
python -m evaluation.sokoban.replay \
  --level-id sokoban-test-000 \
  --trajectory evaluation/sokoban/results/oracle/sokoban-test-000/trajectory.jsonl \
  --output evaluation/sokoban/results/oracle/game_000.gif

# 手动游玩：w/a/s/d 移动，q 退出。
python -m evaluation.sokoban.replay \
  --level-id sokoban-test-000 --output output/sokoban_manual.png
```

回放逐步核对状态与动作，再输出 GIF。更换数据集时用 `--levels` 指定对应的 `levels.jsonl`；评测加 `--save-frames` 可以保存每步 PNG。

## 验证与迁移范围

```bash
python -m pytest -q tests/test_sokoban.py
```

测试覆盖规则、精确距离、多最优动作、数据去重、错误标签/媒体检测、完整 oracle 游戏、分片合并与通用比较。没有随代码迁入历史数据、结果、模型权重或集群配置。

开发目录里的 `build_eval_v2.py`、`select_simple.py`、`summarize_subset.py`、`diagnose_training.py`、`preview.py`、`export_gifs.py` 和六个集群 Shell 脚本未迁入；前者分别用于额外测试集变体、特定实验筛选/汇总、训练诊断和展示，后者依赖维护者的调度环境。通用 GIF 能力由 `replay.py` 提供。
