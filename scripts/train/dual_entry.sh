#!/usr/bin/env bash
set -euo pipefail
VALEN_LOG_DIR="$1"
shift
VALEN_MODE="${1:-smoke}"
if (( $# > 0 )); then shift; fi
VALEN_ENTRY="$(dirname -- "${BASH_SOURCE[0]}")/entry.sh"
case "$VALEN_MODE" in
  smoke) exec bash "$VALEN_ENTRY" "$VALEN_LOG_DIR" smoke dual_encoder "$@" ;;
  sft|rlcd) exec bash "$VALEN_ENTRY" "$VALEN_LOG_DIR" "configs/train/dual_encoder/modernbert_dinov3b16/${VALEN_MODE}_warmup.json" "$@" ;;
  *) printf 'Expected mode: smoke, sft, rlcd\n' >&2; exit 2 ;;
esac
