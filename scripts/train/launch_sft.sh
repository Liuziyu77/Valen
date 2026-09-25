#!/usr/bin/env bash
set -euo pipefail
export VALEN_GPUS="${VALEN_GPUS:-${VJ_GPUS:-2}}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/launch.sh" "$@"
