#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [[ -f .gitnexus/run.cjs ]]; then
    node .gitnexus/run.cjs analyze "$@"
else
    npx gitnexus analyze "$@"
fi

remove_generated_block() {
    local file="$1"
    local start_marker="$2"
    local end_marker="$3"
    local temp_file

    [[ -f "$file" ]] || return 0

    temp_file="$(mktemp)"

    awk \
        -v start_marker="$start_marker" \
        -v end_marker="$end_marker" '
        index($0, start_marker) {
            skipping = 1
            next
        }

        index($0, end_marker) {
            skipping = 0
            next
        }

        !skipping {
            print
        }
    ' "$file" > "$temp_file"

    mv "$temp_file" "$file"
}

remove_generated_block \
    AGENTS.md \
    '<!-- gitnexus:start -->' \
    '<!-- gitnexus:end -->'

remove_generated_block \
    AGENTS.md \
    '<!-- BEGIN sqz-agents-guidance' \
    '<!-- END sqz-agents-guidance -->'

# Keep the canonical policy file stable after generated-block removal.
python3 - <<'PY_NORMALIZE'
from pathlib import Path

path = Path("AGENTS.md")
if path.exists():
    path.write_text(path.read_text().rstrip() + "\n")
PY_NORMALIZE

printf '@AGENTS.md\n' > CLAUDE.md

echo "GitNexus refreshed; agent instructions normalized."
