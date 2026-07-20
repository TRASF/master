#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUNTIME_ROOT="${WINGBEAT_RUNTIME_ROOT:-/media/miru4090s/New Volume2/wingbeat_ml}"
DATASET_DIR="${WINGBEAT_DATASET_DIR:-/app/dataset/MSB/Indoor}"
PROFILE="${WINGBEAT_PROFILE:-$REPO_ROOT/configs/profiles/local.yaml}"
ENABLE_WANDB="${WINGBEAT_ENABLE_WANDB:-true}"

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
  "$RUNTIME_ROOT/dataset/.tf_cache" \
  "$RUNTIME_ROOT/logs" \
  "$RUNTIME_ROOT/models" \
  "$RUNTIME_ROOT/wandb"

STAMP="$(date +%Y%m%d-%H%M%S)"
RESOLVED="$RUNTIME_ROOT/configs/pretrain-$STAMP.yaml"
LOG="$RUNTIME_ROOT/logs/pretrain-$STAMP.log"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DIR="$RUNTIME_ROOT/wandb"

cd "$REPO_ROOT"
python -m wingbeat_ml config resolve \
  --base "$BASE_CONFIG" \
  --model "$MODEL_CONFIG" \
  --experiment "$EXPERIMENT_CONFIG" \
  --profile "$PROFILE" \
  --set "dataset.train_dir=$DATASET_DIR" \
  --set "wandb.enabled=$ENABLE_WANDB" \
  --output "$RESOLVED"

cd "$RUNTIME_ROOT"
python -m wingbeat_ml.pipelines.pretrain \
  --defaults_path "$RESOLVED" \
  --model_cfg_path "$MODEL_CONFIG" \
  2>&1 | tee "$LOG"

echo "Resolved configuration: $RESOLVED"
echo "Training log: $LOG"
echo "Runtime root: $RUNTIME_ROOT"
