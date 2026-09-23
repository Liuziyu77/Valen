"""Read a base-model manifest across package directory renames."""
import json
from pathlib import Path


def read_base_manifest(model_path):
    root = Path(model_path)
    preferred = root / "visualjev_manifest.json"
    if preferred.exists():
        return json.loads(preferred.read_text(encoding="utf-8"))
    # A downloaded model may retain its earlier manifest filename. Accept a
    # single manifest with the expected shape, but never guess among several.
    candidates = []
    for path in root.glob("*_manifest.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("revision"), str) and isinstance(value.get("weight_sha256"), dict):
            candidates.append(value)
    if len(candidates) > 1:
        raise ValueError("Multiple base-model manifests; keep only one or create visualjev_manifest.json")
    return candidates[0] if candidates else None
