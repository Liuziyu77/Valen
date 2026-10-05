# game-v4 与 arcade-v2 游戏评测

五类游戏为 Maze、FrozenLake、Sokoban、LaneRacer 和 ParkourRunner。
game-v4 的 Sokoban 与原有 `evaluation.sokoban` 的 `sokoban-v1` 分别报告。
环境快照来自生成这两套数据的 VisionJev 代码，位于 `_upstream/`；保留 Apache-2.0 许可和文件 SHA。

## 单步动作评测

先用 `valen.evaluate` 生成候选概率，再统计所有带标签状态：

```bash
torchrun --nproc_per_node=8 -m valen.evaluate \
  --checkpoint output/rl/latest --data data/game_eval.jsonl --output output/game_steps
python -m evaluation.game.score --data data/game_eval.jsonl \
  --predictions output/game_steps/predictions.jsonl --revision game-v4 \
  --output output/game_steps/game_metrics.json
```

`optimal_action_accuracy` 表示预测动作是否属于标签中概率大于零的最优动作集合。
等长最优动作全部接受，不能把 soft label 当作唯一答案。
同时报告最优动作概率质量、Brier、交叉熵和包含多个最优动作的样本比例。

## 完整游戏评测

```bash
torchrun --nproc_per_node=8 -m evaluation.game.evaluate \
  --checkpoint output/rl/latest \
  --datasets data/game_v4_eval.jsonl data/arcade_v2_eval.jsonl \
  --legacy-eval-dir data/eval_sokoban --output output/game_episodes
```

使用 eval JSONL 中所有地图的初始状态；重建后核验图片和规则与发布数据一致。
发布时去重可能删去初始图片；这时先复现最早保留的轨迹状态并核验，再重置到地图起点。
只评测 `split=eval` 的地图，不使用训练地图。模型每步只看到图片、规则与候选动作。
结果包含成功率、成功局步数相对最短解、最优动作命中率和实际耗时。
固定策略仅按可见图片和完整请求缓存重复状态；缓存命中数、真实调用数单独报告。
新 Sokoban 标为 `game-v4/sokoban`，老版本标为 `sokoban-v1/sokoban`。
`--legacy-eval-dir` 可省略；`--oracle --device cpu` 用于核验环境和评测器。

此评测是多步环境交互。训练中的 RLCD 则根据带标签的单步决策采样动作，
优化置信度奖励、PPO clip、固定参考 KL 与 Brier，不包含在线地图 rollout 训练。
