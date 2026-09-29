#!/usr/bin/env bash
set -euo pipefail
VALEN_LOG_DIR="$1"
shift
exec > >(tee "$VALEN_LOG_DIR/console.log") 2>&1
trap 'VALEN_EXIT=$?; printf "EXIT_CODE=%s\n" "$VALEN_EXIT" > "$VALEN_LOG_DIR/status.log"' EXIT
cd -- "${VALEN_CODE_ROOT:-$(dirname -- "${BASH_SOURCE[0]}")/../..}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${VALEN_EXPERIMENT_ROOT:-$PWD/next-generation-exp}/hf-cache"
export PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
nvidia-smi
"${VALEN_PYTHON:-$PWD/.venv/bin/python}" "$@"
