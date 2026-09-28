import hashlib
import json
from pathlib import Path
import torch

# 仅迁移共享主干。 / Transfer shared-backbone parameters only.
SHARED = ('text_encoder.', 'vision_encoder.', 'text_projection.', 'vision_projection.',
          'modality.', 'coordinates.', 'visual_cls', 'fusion.')
SHAPE_KEYS = ('hidden_size', 'num_heads', 'intermediate_size', 'fusion_layers')


def is_shared(name):
    return name.startswith(SHARED)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def export_shared(model, path):
    """导出完整共享主干，包含冻结权重。 / Export the full backbone, including frozen weights."""
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if is_shared(k)}
    temp = path / 'shared.tmp'; torch.save(weights, temp); temp.replace(path / 'shared.pt')
    manifest = {'format': 'valen-shared-v1', 'sha256': file_hash(path / 'shared.pt'),
                'base_manifest': model.base_manifest,
                'shape': {k: model.config[k] for k in SHAPE_KEYS}, 'keys': sorted(weights)}
    temp = path / 'manifest.tmp'; temp.write_text(json.dumps(manifest, indent=2)); temp.replace(path / 'manifest.json')
    return manifest


def load_shared(model, path):
    """校验来源与完整性后加载主干。 / Validate provenance and integrity before loading."""
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest['format'] != 'valen-shared-v1' or file_hash(path / 'shared.pt') != manifest['sha256']:
        raise ValueError('Shared backbone checksum or format mismatch')
    if manifest['base_manifest'] != model.base_manifest:
        raise ValueError('Shared backbone base manifest mismatch')
    if any(model.config[k] != v for k, v in manifest['shape'].items()):
        raise ValueError('Shared backbone architecture mismatch')
    weights = torch.load(path / 'shared.pt', map_location='cpu', weights_only=True)
    expected = {k for k in model.state_dict() if is_shared(k)}
    if set(weights) != expected or set(manifest['keys']) != expected:
        raise ValueError('Incomplete shared backbone')
    model.load_state_dict(weights, strict=False)
    model.base_manifest = {**model.base_manifest, 'pretraining': manifest}
