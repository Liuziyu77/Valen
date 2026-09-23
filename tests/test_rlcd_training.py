"""Exercise the actual training loop and checkpoint path with a small CPU model."""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from visionjev.training import runner
from visionjev.training.checkpoint import save_checkpoint
from visionjev.training.distributed import close


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(2, 3)
        self.lora_targets = []

    def forward(self, question):
        return self.head(torch.tensor(question.features))[:len(question.target)]


class TinyCompiler:
    def __init__(self, *args):
        pass

    def compile(self, record, rng, **kwargs):
        questions = []
        for name, question in record["request"]["questions"].items():
            target = list(record["targets"][name]["probabilities"].values())
            questions.append(SimpleNamespace(qid=name, kind=question["type"], target=target,
                                            features=[float(record["request"]["state"]), rng.random()], is_text=True))
        return SimpleNamespace(questions=questions, compute_tokens=1, media=[])


def prepare_files(root):
    root = Path(root)
    records = []
    for i in range(3):
        records.append({"request": {"state": str(i + 1), "questions": {
            "choice": {"type": "choice", "instructions": "Pick a letter", "criteria": {"a": "A", "b": "B", "c": "C"}},
            "noul": {"type": "noul", "instructions": "Is it valid?"},
            "score": {"type": "score", "instructions": "Rate it", "criteria": ["low", "medium", "high"]},
        }}, "targets": {
            "choice": {"probabilities": {"a": 1., "b": 0., "c": 0.}},
            "noul": {"probabilities": {"true": 0., "false": 1.}},
            "score": {"probabilities": {"0": .2, "1": .6, "2": .2}},
        }, "group_id": str(i), "assets": []})
    data = root / "train.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in records))
    torch.manual_seed(7)
    model = TinyModel()
    save_checkpoint(root / "sft" / "latest", model, torch.optim.AdamW(model.parameters()),
                    {"model_path": str(root / "base"), "stage": "warmup"}, {"step": 1}, random.Random(7))
    return dict(model_path=str(root / "base"), data=str(data), stage="warmup", method="rlcd", device="cpu",
                epochs=2, max_steps=2, tokens_per_step=1, save_every=1, head_lr=.02,
                rlcd={"group_size": 16, "num_iterations": 2, "beta": .1})


def read_payload(path):
    return torch.load(Path(path) / "latest/checkpoint.pt", map_location="cpu", weights_only=False)


def compare_runs(root, config):
    root = Path(root)
    checkpoint = root / "sft" / "latest"
    runner.run(dict(config, output=str(root / "continuous")), initialize=checkpoint)
    runner.run(dict(config, output=str(root / "resumed"), max_steps=1), initialize=checkpoint)
    model, _, _ = runner.run(dict(config, output=str(root / "resumed")), resume=root / "resumed" / "latest")
    if dist.is_initialized():
        import hashlib
        fingerprint = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, fingerprint)
        assert len(set(gathered)) == 1
    a, b = read_payload(root / "continuous"), read_payload(root / "resumed")
    for name in a["weights"]:
        torch.testing.assert_close(a["weights"][name], b["weights"][name], atol=0, rtol=0)
    assert a["progress"] == b["progress"]
    assert a["progress"]["step"] == 2 and a["progress"]["optimizer_steps"] == 4
    for key, state in a["optimizer"]["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][key][field], atol=0, rtol=0)
    original = read_payload(root / "sft")["weights"]
    assert any(not torch.equal(value, b["weights"][name]) for name, value in original.items())
    for name, value in original.items():
        torch.testing.assert_close(value, b["training_state"]["reference_weights"][name], atol=0, rtol=0)
    for x, y in zip(a["distributed"]["rank_states"], b["distributed"]["rank_states"]):
        assert x["rng"] == y["rng"] and x["progress"] == y["progress"]
        assert torch.equal(x["torch_rng"], y["torch_rng"])
    metrics = [json.loads(line) for line in (root / "resumed/metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 2 and all(row["method"] == "rlcd" for row in metrics)
    assert all(row["questions"] == 3 * row["states"] for row in metrics)
    return metrics


def test_rlcd_three_types_resume_matches_continuous_and_keeps_original_reference(tmp_path, monkeypatch):
    from transformers import AutoProcessor
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(runner, "build_model", lambda config: TinyModel())
    monkeypatch.setattr(runner, "Compiler", TinyCompiler)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: None)
    config = prepare_files(tmp_path)
    compare_runs(tmp_path, config)


def _distributed_worker(rank, root, config):
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    runner.build_model = lambda config: TinyModel()
    runner.Compiler = TinyCompiler
    AutoProcessor.from_pretrained = lambda *args, **kwargs: None
    try:
        metrics = compare_runs(root, config)
        assert any(r["states"] == 0 for row in metrics for r in row["ranks"])
        states = read_payload(Path(root) / "resumed")["distributed"]["rank_states"]
        assert not torch.equal(states[0]["torch_rng"], states[1]["torch_rng"])
    finally:
        close()


def test_rlcd_two_rank_idle_tail_and_resume(tmp_path):
    config = prepare_files(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(config))
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
                            env=environment, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    root = sys.argv[1]
    config = json.loads((Path(root) / "config.json").read_text())
    mp.spawn(_distributed_worker, args=(root, config), nprocs=2, join=True)
