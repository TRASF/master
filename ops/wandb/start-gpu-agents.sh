#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export WINGBEAT_LAUNCH_QUEUE="${WINGBEAT_LAUNCH_QUEUE:-wingbeat-training}"
export WINGBEAT_RUNTIME_ROOT="${WINGBEAT_RUNTIME_ROOT:?Set WINGBEAT_RUNTIME_ROOT}"
export WINGBEAT_DATASET_DIR="${WINGBEAT_DATASET_DIR:?Set WINGBEAT_DATASET_DIR}"
export WINGBEAT_CACHE_DIR="${WINGBEAT_CACHE_DIR:-$WINGBEAT_RUNTIME_ROOT/dataset/.tf_cache}"

exec python "$SCRIPT_DIR/gpu-agent-supervisor.py" "$@"
