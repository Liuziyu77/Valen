"""Pad language rows and concatenate visual patches. / 文本补齐，视觉 patch 拼接。"""
import torch
from torch.nn.utils.rnn import pad_sequence


SEQUENCE_KEYS = {"input_ids", "attention_mask", "mm_token_type_ids"}
VISUAL_KEYS = {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}


def collate_branches(branches, pad_token_id=0):
    if not branches:
        raise ValueError("Cannot collate an empty branch batch")
    inputs = [branch.inputs for branch in branches]
    unknown = set().union(*(row.keys() for row in inputs)) - SEQUENCE_KEYS - VISUAL_KEYS
    if unknown:
        raise ValueError(f"Unsupported batched Qwen inputs: {sorted(unknown)}")
    if any(row["input_ids"].ndim != 2 or row["input_ids"].shape[0] != 1 for row in inputs):
        raise ValueError("Each compiled branch must contain one language row")
    result = {}
    for key in SEQUENCE_KEYS:
        if key != "input_ids" and not any(key in row for row in inputs):
            continue
        rows = []
        for row in inputs:
            value = row.get(key)
            if value is None:
                value = (torch.ones_like(row["input_ids"]) if key == "attention_mask"
                         else torch.zeros_like(row["input_ids"]))
            if value.shape != row["input_ids"].shape:
                raise ValueError(f"{key} must match the branch input_ids shape")
            rows.append(value[0])
        # Right padding keeps all compiler readout positions and spans unchanged.
        # 右侧补齐，候选位置和角色区间无需偏移。
        result[key] = pad_sequence(rows, batch_first=True,
                                   padding_value=pad_token_id if key == "input_ids" else 0)
    if "attention_mask" not in result:
        result["attention_mask"] = pad_sequence([torch.ones_like(r["input_ids"][0]) for r in inputs],
                                                batch_first=True, padding_value=0)
    for key in VISUAL_KEYS:
        values = [row[key] for row in inputs if key in row]
        if values:
            result[key] = torch.cat(values, dim=0)
    return result


def branch_batches(questions, max_tokens):
    """Limit padded tokens per backbone forward, keeping Score branch order."""
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("microbatch_max_tokens must be a positive integer")
    batch, longest = [], 0
    for question_index, question in enumerate(questions):
        if not question.branches:
            raise ValueError("A Qwen question requires at least one branch")
        for branch in question.branches:
            length = branch.inputs["input_ids"].shape[1]
            if batch and max(longest, length) * (len(batch) + 1) > max_tokens:
                yield batch
                batch, longest = [], 0
            batch.append((question_index, branch))
            longest = max(longest, length)
    if batch:
        yield batch
