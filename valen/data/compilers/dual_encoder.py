"""Independent state, instruction and candidate sequences for dual encoders."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from ..schema import target_distribution
from ..types import QuestionSpec


def description(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    raise ValueError("Descriptions must be text, objects or arrays")


def parallel_candidates(question):
    if not description(question.get("instructions")).strip():
        raise ValueError("instructions must not be empty")
    kind, criteria = question.get("type"), question.get("criteria")
    if kind == "noul":
        if criteria is None:
            criteria = {"true": "The stated condition is true.", "false": "The stated condition is false."}
        if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
            raise ValueError("Noul criteria must define true and false")
        pairs = [(k, description(criteria[k])) for k in ("true", "false")]
    elif kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
            raise ValueError("Choice needs 1–255 named criteria")
        pairs = [(k, description(v) if v is not None else "") for k, v in criteria.items()]
        pairs = [(k, v if v.strip() else k) for k, v in pairs]
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("Score needs 2–10 descriptions")
        pairs = [(str(i), description(v)) for i, v in enumerate(criteria)]
    else:
        raise ValueError(f"Unsupported question type: {kind}")
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v.strip() for k, v in pairs):
        raise ValueError("Candidate names and descriptions must not be empty")
    return pairs


ParallelQuestion = QuestionSpec


@dataclass
class ParallelState:
    questions: list
    state_tokens: list
    instruction_tokens: list
    candidate_tokens: list
    candidate_offsets: list
    pixel_values: torch.Tensor | None
    visual_mask: torch.Tensor | None
    logical_tokens: int
    compute_tokens: int
    media: list


class ParallelCompiler:
    def __init__(self, tokenizer, image_processor, media_root=".", *, image_size=384, patch_size=16,
                 state_max_length=1024, question_max_length=128, candidate_max_length=64, max_questions=256):
        self.tokenizer, self.image_processor = tokenizer, image_processor
        self.media_root = Path(media_root)
        self.image_size, self.patch_size = image_size, patch_size
        self.limits = (state_max_length, question_max_length, candidate_max_length)
        self.max_questions = max_questions
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size:
            raise ValueError("image_size must be a positive multiple of patch_size")
        if any(type(x) is not int or x <= 0 for x in (*self.limits, max_questions)):
            raise ValueError("Sequence and question budgets must be positive integers")
        if max_questions > 256:
            raise ValueError("At most 256 questions are supported")
        if tokenizer.pad_token_id is None:
            raise ValueError("Text encoder tokenizer needs a pad token")

    def _tokens(self, text, limit, role):
        ids = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
        if not ids or len(ids) > limit:
            raise ValueError(f"{role}: {len(ids)} tokens exceed budget={limit} or empty sequence; no truncation")
        return ids

    def _state(self, state):
        messages = [{"role": "user", "content": state}] if isinstance(state, str) else state["messages"]
        text, media = [], []
        for message in messages:
            role = message["role"]
            if role not in {"user", "assistant", "system"}:
                raise ValueError("Unsupported state role")
            # Roles and image position are serialized as ordinary text, never tokenizer control IDs.
            text.append(f"{role}:\n")
            content = message["content"]
            content = [{"type": "text", "text": content}] if isinstance(content, str) else content
            for item in content:
                if item["type"] == "text":
                    text.append(item["text"])
                elif item["type"] == "image_url":
                    url = item["image_url"]["url"]
                    if "://" in url:
                        raise ValueError("Materialize images locally before compiling")
                    path = (self.media_root / url).resolve(strict=True)
                    media.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "type": "image_url"})
                    text.append("\n[image]\n")
                else:
                    raise ValueError("Dual encoder supports text and a single image, not video")
            text.append("\n")
        if len(media) > 1:
            raise ValueError("Dual encoder supports at most one image per state")
        return "".join(text), media

    def _image(self, path):
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            size, patch = self.image_size, self.patch_size
            ratio = size / max(image.size)
            width, height = max(1, round(image.width * ratio)), max(1, round(image.height * ratio))
            image = image.resize((width, height), resample=self.image_processor.resample)
            mean = np.array(self.image_processor.image_mean, dtype=np.float32)
            std = np.array(self.image_processor.image_std, dtype=np.float32)
            canvas = Image.new("RGB", (size, size), tuple(np.rint(mean * 255).astype(int)))
            left, top = (size - width) // 2, (size - height) // 2
            canvas.paste(image, (left, top))
            pixels = (np.asarray(canvas, dtype=np.float32) / 255.0 - mean) / std
        axis = torch.arange(size // patch) * patch
        valid_x = (axis < left + width) & (axis + patch > left)
        valid_y = (axis < top + height) & (axis + patch > top)
        valid = torch.cat([torch.ones(1, dtype=torch.bool), (valid_y[:, None] & valid_x[None, :]).flatten()])
        return torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0), valid.unsqueeze(0)

    def compile(self, record, rng=None, labeled_only=False):
        items = record["request"]["questions"]
        if not isinstance(items, dict) or not 1 <= len(items) <= self.max_questions:
            raise ValueError(f"Expected 1–{self.max_questions} questions")
        selected = []
        for qid, question in items.items():
            pairs = parallel_candidates(question)
            target = record.get("targets", {}).get(qid)
            if labeled_only and target_distribution(target, [k for k, _ in pairs]) is None:
                continue
            if rng is not None and question["type"] == "choice":
                rng.shuffle(pairs)
            selected.append((qid, question, pairs, target))
        if not selected:
            return ParallelState([], [], [], [], [0], None, None, 0, 0, [])
        state_text, media = self._state(record["request"]["state"])
        expected = {str((self.media_root / a["path"]).resolve()): a["sha256"] for a in record.get("assets", [])}
        for entry in media:
            if entry["path"] in expected and expected[entry["path"]] != entry["sha256"]:
                raise ValueError(f"Media hash mismatch: {entry['path']}")
        state_ids = self._tokens(state_text, self.limits[0], "state")
        questions, instructions, candidates, offsets = [], [], [], [0]
        for qid, q, pairs, target in selected:
            keys = [k for k, _ in pairs]
            instructions.append(self._tokens(description(q["instructions"]), self.limits[1], "instructions"))
            for key, value in pairs:
                text = json.dumps({"name": key, "description": value}, ensure_ascii=False) if q["type"] == "choice" else value
                candidates.append(self._tokens(text, self.limits[2], "candidate"))
            offsets.append(len(candidates))
            questions.append(ParallelQuestion(qid, q["type"], keys, [v for _, v in pairs],
                                              target_distribution(target, keys), not media))
        pixels, mask = self._image(media[0]["path"]) if media else (None, None)
        tokens = len(state_ids) + sum(map(len, instructions)) + sum(map(len, candidates))
        # Letterbox patches still run through DINO even when readers mask them.
        # This budget counts sequence positions, not measured FLOPs or padding in text batches.
        compute = tokens + (mask.numel() if mask is not None else 0)
        return ParallelState(questions, state_ids, instructions, candidates, offsets, pixels, mask, tokens, compute, media)


def build_parallel_compiler(config, media_root="."):
    from transformers import AutoTokenizer, AutoImageProcessor, AutoConfig
    from valen.configuration import flatten_config
    config = flatten_config(config)
    text = config["text_model_path"]
    vision = config["vision_model_path"]
    return ParallelCompiler(AutoTokenizer.from_pretrained(text, local_files_only=True),
                            AutoImageProcessor.from_pretrained(vision, local_files_only=True), media_root,
                            image_size=config.get("image_size", 384),
                            patch_size=AutoConfig.from_pretrained(vision, local_files_only=True).patch_size,
                            state_max_length=config.get("state_max_length", 1024),
                            question_max_length=config.get("question_max_length", 128),
                            candidate_max_length=config.get("candidate_max_length", 64),
                            max_questions=config.get("max_questions", 256))
