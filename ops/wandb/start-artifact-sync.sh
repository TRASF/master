#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
STAGING_ROOT="${WINGBEAT_STAGING_ROOT:-${WINGBEAT_RUNTIME_ROOT:?Set WINGBEAT_STAGING_ROOT}}"
DESTINATION_ROOT="${WINGBEAT_ARTIFACT_DESTINATION:?Set WINGBEAT_ARTIFACT_DESTINATION}"
POLL_SECONDS="${WINGBEAT_ARTIFACT_POLL_SECONDS:-30}"

COMMAND=(
  python -m wingbeat_ml.ops.artifact_handoff sync
  --staging-root "$STAGING_ROOT"
  --destination-root "$DESTINATION_ROOT"
  --poll-seconds "$POLL_SECONDS"
  --watch
)
if [[ -n "${WINGBEAT_ARTIFACT_OWNER:-}" ]]; then
  COMMAND+=(--owner "$WINGBEAT_ARTIFACT_OWNER")
fi

exec "${COMMAND[@]}"
