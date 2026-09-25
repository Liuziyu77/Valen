#!/usr/bin/env bash
set -euo pipefail
VALEN_MODE="${1:-smoke}"
if (( $# > 0 )); then shift; fi
VALEN_SUBMIT="$(dirname -- "${BASH_SOURCE[0]}")/submit.sh"
case "$VALEN_MODE" in
  smoke) exec bash "$VALEN_SUBMIT" smoke dual_encoder "$@" ;;
  sft|rlcd) exec bash "$VALEN_SUBMIT" "configs/train/dual_encoder/modernbert_dinov3b16/${VALEN_MODE}_warmup.json" "$@" ;;
  *) printf 'Expected mode: smoke, sft, rlcd\n' >&2; exit 2 ;;
esac
