import json

import pytest

from visualjev.modeling.manifest import read_base_manifest


def test_manifest_reader_keeps_existing_snapshot_metadata(tmp_path):
    manifest = {"revision": "base-revision", "weight_sha256": {"model.safetensors": "digest"}}
    (tmp_path / "original_manifest.json").write_text(json.dumps(manifest))
    assert read_base_manifest(tmp_path) == manifest

    preferred = {"revision": "new-revision", "weight_sha256": {}}
    (tmp_path / "visualjev_manifest.json").write_text(json.dumps(preferred))
    assert read_base_manifest(tmp_path) == preferred


def test_manifest_reader_rejects_ambiguous_existing_files(tmp_path):
    manifest = {"revision": "base-revision", "weight_sha256": {}}
    for name in ("first_manifest.json", "second_manifest.json"):
        (tmp_path / name).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Multiple base-model manifests"):
        read_base_manifest(tmp_path)
