"""Download an immutable official snapshot and record dependency/media defaults."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/Qwen3.5-0.8B")
    parser.add_argument("--revision", default="2fc06364715b967f1860aea9cf38778875588b17")
    parser.add_argument("--verify-local", action="store_true", help="Validate already downloaded files against Hub SHA256")
    args = parser.parse_args()
    repo = "Qwen/Qwen3.5-0.8B"
    revision = HfApi().model_info(repo, revision=args.revision).sha
    path = Path(args.output) if args.verify_local else Path(snapshot_download(repo, revision=revision, local_dir=args.output))
    import hashlib
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    hashes = {}
    for item in info.siblings:
        if item.rfilename.endswith(".safetensors"):
            file = path / item.rfilename
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            hashes[item.rfilename] = digest.hexdigest()
            if item.lfs and (file.stat().st_size != item.lfs.size or digest.hexdigest() != item.lfs.sha256):
                raise ValueError(f"Weight checksum mismatch: {file}")
    from transformers import AutoProcessor, AutoConfig
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if config.model_type != "qwen3_5":
        raise ValueError("Expected Qwen3.5 multimodal backbone")
    versions = {}
    for name in ("torch", "torchvision", "transformers", "peft", "huggingface-hub", "av"):
        versions[name] = importlib.metadata.version(name)
    manifest = {"repo": repo, "revision": revision, "versions": versions, "weight_sha256": hashes,
                "processor": type(processor).__name__, "hidden_size": config.text_config.hidden_size,
                "media_defaults": {name: json.loads((path / name).read_text(encoding="utf-8")) for name in
                                   ("preprocessor_config.json", "video_preprocessor_config.json")}}
    (path / "visualjev_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
