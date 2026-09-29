#!/usr/bin/env bash
# 本机配置不提交到 Git。 / Keep machine-specific settings outside version control.
VALEN_CLUSTER_CONFIG="${VALEN_CLUSTER_CONFIG:-$VALEN_ROOT/.env.cluster}"
if [[ -f "$VALEN_CLUSTER_CONFIG" ]]; then
  source "$VALEN_CLUSTER_CONFIG"
elif [[ "$VALEN_CLUSTER_CONFIG" != "$VALEN_ROOT/.env.cluster" ]]; then
  printf 'Cluster config not found: %s\n' "$VALEN_CLUSTER_CONFIG" >&2
  exit 2
fi
RJOB_BIN="${RJOB_BIN:-rjob}"
: "${RJOB_NAMESPACE:?Set RJOB_NAMESPACE in the environment or .env.cluster}"
: "${RJOB_CHARGED_GROUP:?Set RJOB_CHARGED_GROUP in the environment or .env.cluster}"
: "${RJOB_IMAGE:?Set RJOB_IMAGE in the environment or .env.cluster}"
RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE:-group}"

# 每行一个挂载，保留路径中的空格。 / One mount per line, preserving spaces.
VALEN_MOUNT_ARGS=()
while IFS= read -r VALEN_MOUNT; do
  [[ -z "$VALEN_MOUNT" ]] || VALEN_MOUNT_ARGS+=("--mount=$VALEN_MOUNT")
done <<< "${RJOB_MOUNTS:-}"
if [[ "${VALEN_CAPTION_SOURCE_MOUNT:-0}" == 1 ]]; then
  : "${RJOB_CAPTION_MOUNT:?Set RJOB_CAPTION_MOUNT for caption source storage}"
  VALEN_MOUNT_ARGS+=("--mount=$RJOB_CAPTION_MOUNT")
fi

VALEN_WORKER_ENV=()
for VALEN_ENV_NAME in VALEN_PYTHON VALEN_CAPTION_SOURCE_ROOT VALEN_CAPTION_ROOT VALEN_CREDENTIALS_CONFIG; do
  if [[ -n "${!VALEN_ENV_NAME:-}" ]]; then
    VALEN_WORKER_ENV+=(-e "$VALEN_ENV_NAME=${!VALEN_ENV_NAME}")
  fi
done
