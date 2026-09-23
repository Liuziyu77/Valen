"""Store only trainable deltas, optimizer and RNG; original backbone stays pinned."""
import json
import random
from pathlib import Path
import torch


def capture_rank_state(progress, rng):
    return {"progress": progress, "rng": rng.getstate(), "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng_current": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def save_checkpoint(path, model, optimizer, config, progress, rng, rank_states=None, training_state=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    delta = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable}
    manifest_path = Path(config["model_path"]) / "visionjev_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    payload = {"weights": delta, "optimizer": optimizer.state_dict(), "config": config,
               **capture_rank_state(progress, rng),
               "base_manifest": manifest}
    if rank_states is not None:
        payload["distributed"] = {"world_size": len(rank_states), "rank_states": rank_states}
    if training_state is not None:
        payload["training_state"] = training_state
    temp = path / "checkpoint.tmp"
    torch.save(payload, temp)
    temp.replace(path / "checkpoint.pt")
    (path / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path, model, optimizer=None, rng=None, strict=True, rank=0, world_size=1):
    # Only load checkpoints produced by this project and from trusted local storage.
    payload = torch.load(Path(path) / "checkpoint.pt", map_location="cpu", weights_only=False)
    if optimizer is not None or rng is not None:
        saved_world_size = payload.get("distributed", {}).get("world_size", 1)
        if saved_world_size != world_size:
            raise ValueError(f"Resume requires world_size={saved_world_size}, got {world_size}; use --initialize to change GPU count")
        if "distributed" in payload:
            payload.update(payload["distributed"]["rank_states"][rank])
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    actual = set(payload["weights"])
    if strict and expected != actual:
        raise ValueError(f"Trainable checkpoint mismatch: missing={expected-actual}, unexpected={actual-expected}")
    if not strict and actual - expected:
        raise ValueError("Initialization would freeze previously changed parameters; use a stage with a superset of trainable parameters")
    base = payload.get("base_manifest")
    if base:
        current = getattr(model, "base_manifest", None)
        if not current or current["revision"] != base["revision"] or current.get("weight_sha256") != base.get("weight_sha256"):
            raise ValueError("Base model revision differs from checkpoint")
    result = model.load_state_dict(payload["weights"], strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unknown checkpoint weights: {result.unexpected_keys}")
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if rng is not None:
        rng.setstate(payload["rng"])
        random.setstate(payload["python_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if torch.cuda.is_available():
            if payload.get("cuda_rng_current") is not None:
                torch.cuda.set_rng_state(payload["cuda_rng_current"])
            elif payload.get("cuda_rng"):
                torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload
