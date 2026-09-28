#!/usr/bin/env bash
set -euo pipefail
VALEN_LOG_DIR="$1"
shift
exec > >(tee "$VALEN_LOG_DIR/console.log") 2>&1
trap 'VALEN_EXIT=$?; printf "EXIT_CODE=%s\n" "$VALEN_EXIT" > "$VALEN_LOG_DIR/status.log"' EXIT
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/next-generation-exp/hf-cache"
export PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi
"$PWD/.venv/bin/python" "$@"
