#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export HF_HOME="${HF_HOME:-$PWD/artifacts/hf-cache}"
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
VJ_GPUS="${VJ_GPUS:-2}"
VJ_PYTHON="${VJ_PYTHON:-python}"
VJ_CONFIG="${1:-configs/train/sft_joint.json}"
if (( $# > 0 )); then shift; fi
exec "$VJ_PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node="$VJ_GPUS" --max_restarts=0 \
    -m valen.train --config "$VJ_CONFIG" "$@"
