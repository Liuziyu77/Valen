#!/usr/bin/env bash
set -euo pipefail
VALEN_LOG_DIR="$1"
shift
exec > >(tee "$VALEN_LOG_DIR/console.log") 2>&1
trap 'VALEN_EXIT=$?; printf "EXIT_CODE=%s\n" "$VALEN_EXIT" > "$VALEN_LOG_DIR/status.log"' EXIT
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PWD/artifacts/hf-cache}"
export PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
VALEN_PYTHON="${VALEN_PYTHON:-$PWD/.venv/bin/python}"
VALEN_TASK="${1:-configs/train/qwen/sft_joint.json}"
if (( $# > 0 )); then shift; fi
nvidia-smi
if [[ "$VALEN_TASK" == smoke ]]; then
  VALEN_ARCH="${1:?Specify smoke architecture: qwen or dual_encoder}"
  shift
  case "$VALEN_ARCH" in
    qwen|dual_encoder) "$VALEN_PYTHON" "scripts/smoke/$VALEN_ARCH/gpu_smoke.py" "$@" ;;
    *) printf 'Unsupported smoke architecture: %s\n' "$VALEN_ARCH" >&2; exit 2 ;;
  esac
else
  bash scripts/train/launch.sh "$VALEN_TASK" "$@"
fi
