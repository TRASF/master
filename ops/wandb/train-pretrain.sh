#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUNTIME_ROOT="${WINGBEAT_RUNTIME_ROOT:-$REPO_ROOT/runtime}"
DATASET_DIR="${WINGBEAT_DATASET_DIR:-$REPO_ROOT/dataset/MSB/Indoor}"
PROFILE="${WINGBEAT_PROFILE:-$REPO_ROOT/configs/profiles/local.yaml}"
ENABLE_WANDB="${WINGBEAT_ENABLE_WANDB:-true}"
CACHE_DIR="${WINGBEAT_CACHE_DIR:-$RUNTIME_ROOT/dataset/.tf_cache}"
CAPTURE_LOG="${WINGBEAT_CAPTURE_CONSOLE_LOG:-false}"

BASE_CONFIG="${WINGBEAT_BASE_CONFIG:-$REPO_ROOT/configs/base.yaml}"
MODEL_CONFIG="${WINGBEAT_MODEL_CONFIG:-$REPO_ROOT/configs/models/mossong_plus.yaml}"
EXPERIMENT_CONFIG="${WINGBEAT_EXPERIMENT_CONFIG:-$REPO_ROOT/configs/experiments/pretrain.yaml}"

for path in "$BASE_CONFIG" "$MODEL_CONFIG" "$EXPERIMENT_CONFIG" "$PROFILE"; do
  test -f "$path" || {
    echo "Required configuration file not found: $path" >&2
    exit 1
  }
done

test -d "$DATASET_DIR" || {
  echo "Dataset directory not found: $DATASET_DIR" >&2
  echo "Set WINGBEAT_DATASET_DIR to the mounted training dataset." >&2
  exit 1
}

mkdir -p \
  "$RUNTIME_ROOT/configs" \
  "$CACHE_DIR" \
  "$RUNTIME_ROOT/logs" \
  "$RUNTIME_ROOT/models" \
  "$RUNTIME_ROOT/wandb"

STAMP="$(date +%Y%m%d-%H%M%S)"
RESOLVED="$RUNTIME_ROOT/configs/pretrain-$STAMP.yaml"
LOG="$RUNTIME_ROOT/logs/pretrain-$STAMP.log"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DIR="$RUNTIME_ROOT/wandb"
export WINGBEAT_CACHE_DIR="$CACHE_DIR"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

MANIFEST_ARGS=()
if [[ -n "${WINGBEAT_DATASET_MANIFEST:-}" || -n "${WINGBEAT_DATASET_MANIFEST_SHA256:-}" ]]; then
  test -f "${WINGBEAT_DATASET_MANIFEST:-}" || {
    echo "WINGBEAT_DATASET_MANIFEST must name an existing manifest." >&2
    exit 1
  }
  test -n "${WINGBEAT_DATASET_MANIFEST_SHA256:-}" || {
    echo "WINGBEAT_DATASET_MANIFEST_SHA256 is required with a manifest." >&2
    exit 1
  }
  python -c \
    'from wingbeat_ml.ops.preflight import manifest_identity, require_manifest_checksum; import os; require_manifest_checksum(os.environ["WINGBEAT_DATASET_MANIFEST_SHA256"], manifest_identity(os.environ["WINGBEAT_DATASET_MANIFEST"]))'
  MANIFEST_ARGS=(--set "dataset.manifest_sha256=$WINGBEAT_DATASET_MANIFEST_SHA256")
fi

cd "$REPO_ROOT"
python -m wingbeat_ml config resolve \
  --base "$BASE_CONFIG" \
  --model "$MODEL_CONFIG" \
  --experiment "$EXPERIMENT_CONFIG" \
  --profile "$PROFILE" \
  --set "dataset.train_dir=$DATASET_DIR" \
  --set "wandb.enabled=$ENABLE_WANDB" \
  "${MANIFEST_ARGS[@]}" \
  --output "$RESOLVED"

cd "$RUNTIME_ROOT"
TRAIN_COMMAND=(
  python -m wingbeat_ml.pipelines.pretrain
  --defaults_path "$RESOLVED"
  --model_cfg_path "$MODEL_CONFIG"
)
if [[ "$CAPTURE_LOG" == "true" ]]; then
  "${TRAIN_COMMAND[@]}" 2>&1 | tee "$LOG"
else
  "${TRAIN_COMMAND[@]}"
fi

echo "Resolved configuration: $RESOLVED"
if [[ "$CAPTURE_LOG" == "true" ]]; then
  echo "Training log: $LOG"
fi
echo "Runtime root: $RUNTIME_ROOT"
