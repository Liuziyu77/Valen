#!/usr/bin/env bash
set -euo pipefail
VALEN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RJOB_BIN="${RJOB_BIN:-/mnt/shared-storage-user/liuziyu/miniconda3/envs/lmm_xc/bin/rjob}"
VALEN_NAME="${VALEN_RUN_NAME:?Set a unique VALEN_RUN_NAME}"
VALEN_GPUS="${VALEN_GPUS:-8}"
VALEN_LOG_DIR="$VALEN_ROOT/next-generation-exp/jobs/$VALEN_NAME"
mkdir -p "$VALEN_LOG_DIR"
VALEN_EXTRA_MOUNTS=()
if [[ "${VALEN_CAPTION_SOURCE_MOUNT:-0}" == 1 ]]; then
  VALEN_EXTRA_MOUNTS+=(--mount=gpfs://gpfs1/mllmexp:/mnt/shared-storage-user/mllmexp)
fi
"$RJOB_BIN" submit --name="$VALEN_NAME" --gpu="$VALEN_GPUS" --cpu="$((VALEN_GPUS * 8))" --memory="$((VALEN_GPUS * 32768))" \
  --charged-group="${RJOB_CHARGED_GROUP:-llmmultimodal_gpu_pool}" --namespace="${RJOB_NAMESPACE:-ailab-llmmultimodal}" \
  --private-machine=group --mount=gpfs://gpfs1/liuziyu:/mnt/shared-storage-user/liuziyu \
  "${VALEN_EXTRA_MOUNTS[@]}" \
  --image=registry.h.pjlab.org.cn/ailab/pytorch:2.7.0-cuda12.8.1-py3.12-ubuntu24.04 \
  --restart-policy=never --auto-restart=false -P 1 --host-network=false \
  -e VALEN_GPUS="$VALEN_GPUS" \
  -- bash "$VALEN_ROOT/scripts/train/experiment_entry.sh" "$VALEN_LOG_DIR" "$@" \
  2>&1 | tee "$VALEN_LOG_DIR/submit.log"
VALEN_ID="$(awk '/created rjob_name:/ {print $NF}' "$VALEN_LOG_DIR/submit.log" | tail -n 1)"
test -n "$VALEN_ID"
printf '%s\n' "$VALEN_ID" > "$VALEN_LOG_DIR/job_id"
printf 'Job: %s\nLogs: %s\n' "$VALEN_ID" "$VALEN_LOG_DIR"
