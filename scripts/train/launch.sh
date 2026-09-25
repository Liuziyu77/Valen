#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PWD/artifacts/hf-cache}"
export PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
VALEN_PYTHON="${VALEN_PYTHON:-${VJ_PYTHON:-$PWD/.venv/bin/python}}"
VALEN_GPUS="${VALEN_GPUS:-${VJ_GPUS:-1}}"
VALEN_CONFIG="${1:-configs/train/qwen/sft_joint.json}"
if (( $# > 0 )); then shift; fi
exec "$VALEN_PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node="$VALEN_GPUS" --max_restarts=0 \
  -m valen.train --config "$VALEN_CONFIG" "$@"
