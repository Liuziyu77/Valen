import json
from types import SimpleNamespace

import pytest
import torch

from visionjev.training.runner import normalize_config, run


@pytest.mark.parametrize("legacy", [{"kd_weight": .2}, {"teacher_checkpoint": "old_teacher"}])
def test_removed_training_mode_fails_before_model_or_device_loading(legacy):
    with pytest.raises(ValueError, match="Teacher distillation was removed"):
        run(dict(device="not_a_device", **legacy))


def test_sft_step_and_resume_checkpoint_with_disabled_legacy_fields(tmp_path, monkeypatch):
    from transformers import AutoProcessor
    from visionjev.training import runner as train

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(2, 2)
            self.lora_targets = []

        def forward(self, question):
            return self.head(torch.tensor([1., 2.]))

    class TinyCompiler:
        def __init__(self, *args):
            pass

        def compile(self, *args, **kwargs):
            question = SimpleNamespace(qid="q", kind="noul", target=[0., 1.], is_text=True)
            return SimpleNamespace(questions=[question], compute_tokens=1, media=[])

    monkeypatch.setattr(train, "build_model", lambda config: TinyModel())
    monkeypatch.setattr(train, "Compiler", TinyCompiler)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: None)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"request": {"state": "example", "questions": {
        "q": {"type": "noul", "instructions": "Is the example valid?"}}},
        "targets": {"q": {"probabilities": {"true": 0., "false": 1.}}},
        "group_id": "example", "assets": []}) + "\n")
    config = dict(model_path=str(tmp_path / "base"), data=str(data), output=str(tmp_path / "run"),
                  stage="warmup", device="cpu", epochs=2, max_steps=1, tokens_per_step=1, save_every=1)
    _, _, progress = run(config)
    assert progress["step"] == 1
    checkpoint = tmp_path / "run" / "latest"
    payload = torch.load(checkpoint / "checkpoint.pt", weights_only=False)
    payload["config"].update(kd_weight=0., kd_temperature=2.)
    manifest_path = tmp_path / "run" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(initialization_checkpoint="original-sft/latest", initialization_sha256="original-sha")
    manifest_path.write_text(json.dumps(manifest))
    # Emulate the metadata in already-delivered, pure-SFT checkpoints.
    torch.save(payload, checkpoint / "checkpoint.pt")
    _, _, progress = run(dict(config, max_steps=2), resume=checkpoint)
    assert progress["step"] == 2 and progress["epoch"] == 2
    metrics = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 2 and all(row["loss"] == row["ce"] for row in metrics)
    assert metrics[1]["elapsed_seconds"] > metrics[0]["elapsed_seconds"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["initialization_checkpoint"] == "original-sft/latest"
    assert manifest["initialization_sha256"] == "original-sha"
    assert manifest["resume_step"] == 1
    assert all("kd" not in row for row in metrics)
    saved = torch.load(checkpoint / "checkpoint.pt", weights_only=False)
    assert saved["config"] == normalize_config(saved["config"])
    assert saved["optimizer"]["state"][0]["step"] == 2
