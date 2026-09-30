# Vendored unchanged from fstandhartinger/jevbench, revision
# bb05a335bc809e61b20c0f745d25499a82b326fc. See LICENSE in this directory.
"""Canonical JevBench task records.

One record per decision. Every record has:
  id          unique string
  family      one of jevbench.FAMILIES
  state       the content the model evaluates (string or structured object).
              Never contains the expected label.
  question    typed question spec: {"type": noul|choice|score,
              "instructions": str, "criteria": map|list|None}
  labels      ordered list of exact label strings the model may output.
              For noul this is ["no", "yes"]; for score it is the level
              indices as strings ["0", "1", ...].
  expected    ground truth for scoring: a label string for classification,
              "yes"/"no" for noul, integer level index for score.
              None means unmeasured ground truth -> excluded from headline
              accuracy but kept for coverage/reporting.
  split       "public" | "private"
  group       paraphrase group id (paired items share a group),
              None when the item has no paired paraphrase.
  provenance  dict: source, source_id, license/attribution, imported_at,
              exclude_reason (None or str -> excluded from headline),
              notes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

QUESTION_TYPES = ("noul", "choice", "score")


@dataclass
class Task:
    id: str
    family: str
    state: Any
    question: dict
    labels: list
    expected: Any
    split: str
    group: Optional[str] = None
    provenance: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.question.get("type") not in QUESTION_TYPES:
            raise ValueError(f"{self.id}: bad question type {self.question.get('type')!r}")
        if self.split not in ("public", "private"):
            raise ValueError(f"{self.id}: bad split {self.split!r}")
        if not self.labels:
            raise ValueError(f"{self.id}: empty labels")
        if self.expected is not None:
            if self.question["type"] == "score":
                if not isinstance(self.expected, int):
                    raise ValueError(f"{self.id}: score expected must be int level index")
                if str(self.expected) not in self.labels:
                    raise ValueError(f"{self.id}: expected level {self.expected} not in labels")
            else:
                if self.expected not in self.labels:
                    raise ValueError(f"{self.id}: expected {self.expected!r} not in labels")
        # No label leakage via a mechanically injected answer field.
        if isinstance(self.state, dict):
            for banned in ("expected", "label", "ground_truth", "answer_key"):
                if banned in self.state:
                    raise ValueError(f"{self.id}: state contains banned key {banned!r}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_dict(d: dict) -> "Task":
        t = Task(
            id=d["id"],
            family=d["family"],
            state=d["state"],
            question=d["question"],
            labels=d["labels"],
            expected=d.get("expected"),
            split=d["split"],
            group=d.get("group"),
            provenance=d.get("provenance", {}),
        )
        t.validate()
        return t


def load_jsonl(path: str) -> list:
    tasks = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(Task.from_dict(json.loads(line)))
    return tasks


def write_jsonl(tasks: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for t in tasks:
            t.validate()
            fh.write(t.to_json() + "\n")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_hash(tasks: list) -> str:
    """Deterministic hash over canonical task records (order-independent)."""
    h = hashlib.sha256()
    for blob in sorted(t.to_json() for t in tasks):
        h.update(blob.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()
