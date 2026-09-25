#!/usr/bin/env bash
set -euo pipefail
VALEN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RJOB_BIN="${RJOB_BIN:-/mnt/shared-storage-user/liuziyu/miniconda3/envs/lmm_xc/bin/rjob}"
VALEN_NAMESPACE="${RJOB_NAMESPACE:-ailab-llmmultimodal}"
VALEN_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-llmmultimodal_gpu_pool}"
VALEN_GPUS="${VALEN_GPUS:-1}"
VALEN_TIMEOUT="${VALEN_TIMEOUT:-1800}"
VALEN_UUID="$(tr -d '-' < /proc/sys/kernel/random/uuid)"
VALEN_NAME="${VALEN_RUN_NAME:-valen-train-$(date -u +%Y%m%d-%H%M%S)-${VALEN_UUID:0:8}}"
VALEN_LOG_DIR="$VALEN_ROOT/artifacts/jobs/$VALEN_NAME"
mkdir -p "$VALEN_LOG_DIR"
# Default to one GPU; larger runs can explicitly set VALEN_GPUS.
"$RJOB_BIN" submit --name="$VALEN_NAME" --gpu="$VALEN_GPUS" --cpu="$((VALEN_GPUS * 4))" --memory="$((VALEN_GPUS * 49152))" \
  --charged-group="$VALEN_CHARGED_GROUP" --namespace="$VALEN_NAMESPACE" --private-machine=group \
  --mount=gpfs://gpfs1/liuziyu:/mnt/shared-storage-user/liuziyu \
  --image=registry.h.pjlab.org.cn/ailab/pytorch:2.7.0-cuda12.8.1-py3.12-ubuntu24.04 \
  --restart-policy=never --auto-restart=false -P 1 --host-network=false \
  -e VALEN_GPUS="$VALEN_GPUS" \
  -- timeout "$VALEN_TIMEOUT" bash "$VALEN_ROOT/scripts/train/entry.sh" "$VALEN_LOG_DIR" "$@" \
  2>&1 | tee "$VALEN_LOG_DIR/submit.log"
VALEN_ID="$(awk '/created rjob_name:/ {print $NF}' "$VALEN_LOG_DIR/submit.log" | tail -n 1)"
test -n "$VALEN_ID"
printf '%s\n' "$VALEN_ID" > "$VALEN_LOG_DIR/job_id"
printf 'Job: %s\nLogs: %s\n' "$VALEN_ID" "$VALEN_LOG_DIR"
