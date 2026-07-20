"""Persistence for machine-readable quality reports."""

import json
from pathlib import Path


def write_quality_report(report, output_path):
    """Atomically write a quality-gate report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = ["write_quality_report"]
