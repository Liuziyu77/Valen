"""Download pinned encoder snapshots and verify all safetensors against Hub hashes."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def prepare(repo, output, revision="main"):
    api = HfApi()
    info = api.model_info(repo, revision=revision, files_metadata=True)
    root = Path(snapshot_download(repo, revision=info.sha, local_dir=output,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
                                 max_workers=4))
    hashes = {}
    for entry in info.siblings:
        if not entry.rfilename.endswith(".safetensors"):
            continue
        path = root / entry.rfilename
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        hashes[entry.rfilename] = digest.hexdigest()
        if entry.lfs and (digest.hexdigest() != entry.lfs.sha256 or path.stat().st_size != entry.lfs.size):
            raise ValueError(f"Weight checksum mismatch: {path}")
    if not hashes:
        raise ValueError(f"No safetensors weights in {repo}")
    # Pin preprocessing and tokenizer files as well as weights.
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()
             if p.is_file() and p.suffix in {".json", ".txt", ".model"} and p.name != "valen_manifest.json"}
    manifest = {"repo": repo, "revision": info.sha, "weight_sha256": hashes, "files_sha256": files}
    (root / "valen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"repo": repo, "revision": info.sha, "path": str(root)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--vision-model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--text-output", default="models/ModernBERT-base")
    parser.add_argument("--vision-output", default="models/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--text-revision", default="main")
    parser.add_argument("--vision-revision", default="main")
    args = parser.parse_args()
    for repo, output, revision in [(args.text_model, args.text_output, args.text_revision),
                                   (args.vision_model, args.vision_output, args.vision_revision)]:
        prepare(repo, output, revision)


if __name__ == "__main__":
    main()
