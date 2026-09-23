# 数据格式

训练、推理和评估读取 UTF-8 JSONL，每行一条记录。完整可运行样例在 [data/smoke](../data/smoke/)。

```json
{
  "group_id": "traffic-light-001",
  "request": {
    "state": "The traffic light is green.",
    "questions": {
      "go": {
        "type": "noul",
        "instructions": "Only a green light permits crossing. Is crossing permitted?"
      }
    }
  },
  "targets": {
    "go": {"probabilities": {"true": 1.0, "false": 0.0}}
  },
  "assets": []
}
```

## 字段

| 字段 | 约定 |
| --- | --- |
| `group_id` | 必填，标识同一素材或相关样本组；构建训练/验证/测试划分时按组隔离，并检查重复媒体。读取器不会自动替你划分。 |
| `request.state` | 纯文本字符串，或 `{"messages": [...]}` 对象。 |
| `request.questions` | 非空映射，键是问题 ID，值包含 `type`、非空 `instructions` 和适用的 `criteria`。 |
| `targets` | 可选的问题 ID 到标签的映射；推理可省略。标签和问题 ID 不进入提示词。 |
| `assets` | 可选媒体清单，每项至少包含 `path`、`sha256`；提供哈希时会核验对应输入媒体。 |
| `meta` | 可选审计信息，不进入提示词；评估可按 `domain`、`modality`、`language_bucket` 汇总。 |

Choice 的 `criteria` 是 1–255 个候选名到非空描述的映射。Noul 不需要 `criteria`，标签键固定为 `true` 和 `false`。Score 的 `criteria` 是 2–10 个有序描述组成的列表，标签键为字符串 `"0"`、`"1"` 等。

`probabilities` 必须完整覆盖候选键，所有值有限且非负，总和为 1。支持硬标签和软概率标签。缺标签的问题在训练和评估时跳过，推理时仍返回答案；软标签计入概率指标，但不计入硬标签准确率。

## 图片和视频

将 `state` 替换为消息对象，例如：

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "判断图片中的交通灯状态。"},
      {"type": "image_url", "image_url": {"url": "assets/light.png"}}
    ]
  }]
}
```

视频使用 `{"type": "video_url", "video_url": {"url": "assets/clip.mp4"}}`。媒体路径相对于 JSONL 文件所在目录，也支持本地绝对路径；远程 URL 必须先下载到本地。消息角色支持 `user`、`assistant`、`system`。

媒体处理参数在训练配置的 `media_kwargs` 中设置，随 checkpoint 保存。编译器记录媒体哈希、视觉 token 数及视频处理器提供的采样信息。

## 样例与正式数据

随仓库提供的六条合成记录覆盖文本、图片、视频及三类输出，仅用于检查流程。`text.jsonl` 是 `train.jsonl` 的子集，不能把两者作为独立训练/测试划分。生成方式见[合成数据说明](../data/smoke/README.md)。正式评估应使用独立的有标签数据，并报告数据来源与划分方法。
