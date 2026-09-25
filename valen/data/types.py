"""Model-independent decision metadata; encoded tensors stay with each compiler."""
from dataclasses import dataclass


@dataclass
class QuestionSpec:
    qid: str
    kind: str
    keys: list
    descriptions: list
    target: list | None
    is_text: bool
