"""Download a frozen public JevBench snapshot and map its tasks to Valen."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import urlopen

from valen.data.schema import validate_record
from ._upstream.tasks import Task, dataset_hash

REPOSITORY = "https://github.com/fstandhartinger/jevbench"
REVISION = "bb05a335bc809e61b20c0f745d25499a82b326fc"
# 支持共享数据目录，默认使用项目内路径。 / Allow shared data with a portable default.
DEFAULT_ROOT = Path(os.environ.get("VALEN_JEVBENCH_DATA", "data/jevbench"))
DEFAULT_FILES = ("easy", "original", "hard")
TIERS = {"easy": "easy", "original": "standard", "hard": "hard"}
EXPECTED = {
    "easy": (48, "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b"),
    "original": (72, "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180"),
    "hard": (111, "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb"),
}
MAPPING_VERSION = 1


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def download(output=DEFAULT_ROOT):
    """Fetch only public datasets and their provenance, without installing upstream."""
    output = Path(output)
    files = {"public/" + name + ".jsonl": "datasets/public/" + name + ".jsonl" for name in DEFAULT_FILES}
    files.update({"manifest.json": "datasets/manifest.json", "LICENSE": "LICENSE"})
    base = "https://raw.githubusercontent.com/fstandhartinger/jevbench/" + REVISION + "/"
    for relative, remote in files.items():
        path = output / relative
        if path.exists():
            # Verify an existing snapshot against the pinned remote before accepting it.
            existing = path.read_bytes()
        else:
            existing = None
        for attempt in range(3):
            try:
                with urlopen(base + remote, timeout=60) as response:
                    content = response.read()
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        if existing is not None and existing != content:
            raise ValueError(f"Refusing to overwrite a different snapshot: {path}")
        if path.suffix == ".jsonl":
            name = path.stem
            if hashlib.sha256(content).hexdigest() != EXPECTED[name][1]:
                raise ValueError(f"Downloaded data checksum mismatch: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if existing is None:
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_bytes(content)
            temp.replace(path)
    tasks, metadata = load_dataset(output)
    metadata.update(repository=REPOSITORY, revision=REVISION, public_only=True)
    write_json(output / "source.json", metadata)
    return tasks, metadata


def load_dataset(root, files=DEFAULT_FILES):
    root = Path(root)
    if not files or len(set(files)) != len(files) or set(files) - set(DEFAULT_FILES):
        raise ValueError(f"Select unique public files from {DEFAULT_FILES}")
    tasks, tiers, sources = [], {}, {}
    for name in files:
        path = root / "public" / (name + ".jsonl")
        digest = file_hash(path)
        if digest != EXPECTED[name][1]:
            raise ValueError(f"Frozen JevBench checksum mismatch: {path}")
        current = []
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    task = Task.from_dict(json.loads(line))
                    if task.split != "public":
                        raise ValueError("Only public tasks are supported")
                    to_record(task)  # Validate candidate mappings before loading a model.
                    current.append(task)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
        if len(current) != EXPECTED[name][0]:
            raise ValueError(f"Unexpected task count: {path}")
        for task in current:
            if task.id in tiers:
                raise ValueError(f"Duplicate task ID: {task.id}")
            tiers[task.id] = TIERS[name]
        tasks.extend(current)
        sources[name] = {"sha256": digest, "tasks": len(current)}
    return tasks, {"repository": REPOSITORY, "revision": REVISION,
                   "public_only": True, "files": sources, "tasks": len(tasks),
                   "canonical_sha256": dataset_hash(tasks), "tiers": tiers}


def to_record(task):
    """Keep answers and provenance outside the prompt; preserve criteria insertion order."""
    question = deepcopy(task.question)
    kind = question["type"]
    if kind == "noul":
        if task.labels != ["no", "yes"]:
            raise ValueError("JevBench Noul requires ordered labels [no, yes]")
        criteria = question.get("criteria")
        if criteria is not None:
            if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
                raise ValueError("Noul criteria must define true and false")
            # Qwen's Noul candidates are fixed, so carry the task-specific rubric
            # in the instruction rather than silently discarding its semantics.
            question["instructions"] += ("\nTrue / yes criterion: " + criteria["true"]
                                          + "\nFalse / no criterion: " + criteria["false"])
        question.pop("criteria", None)
    elif kind == "choice":
        if set(question["criteria"]) != set(task.labels):
            raise ValueError("Choice labels differ from criteria")
    elif task.labels != [str(i) for i in range(len(question["criteria"]))]:
        raise ValueError("Score labels must follow contiguous level order")
    state = task.state
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return validate_record({"group_id": task.group or task.id,
                            "request": {"state": state, "questions": {"decision": question}}})


def probabilities(task, response):
    answer = response["answers"]["decision"]
    if answer["type"] != task.question["type"]:
        raise ValueError("Native answer type mismatch")
    if answer["type"] == "noul":
        value = answer["noul"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Noul probability must be numeric")
        return {"no": 1.0 - value, "yes": value}
    return answer["probabilities"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    _, metadata = download(args.output)
    print(json.dumps({"output": str(args.output), "tasks": metadata["tasks"], "revision": REVISION}))


if __name__ == "__main__":
    main()
