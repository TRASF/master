"""Portable model-lineage manifests."""

import hashlib
import json
from numbers import Real
from pathlib import Path


def sha256_file(path):
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_lineage(
    *,
    metrics,
    model_path=None,
    config_sha256=None,
    dataset_sha256=None,
    git_commit=None,
    source_artifact=None,
):
    """Build a stable, JSON-compatible lineage manifest."""
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a mapping")

    normalized_metrics = {}
    for name, value in sorted(metrics.items()):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"lineage metric {name!r} must be numeric"
            )
        normalized_metrics[name] = float(value)

    manifest = {
        "schema_version": 1,
        "metrics": normalized_metrics,
    }

    if model_path:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"model file not found: {model_path}"
            )
        manifest["model"] = {
            "name": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    sources = {
        "config_sha256": config_sha256,
        "dataset_sha256": dataset_sha256,
        "git_commit": git_commit,
        "source_artifact": source_artifact,
    }
    sources = {
        key: value
        for key, value in sources.items()
        if value is not None
    }

    if sources:
        manifest["sources"] = sources

    return manifest


def write_lineage(manifest, output_path):
    """Atomically write a lineage manifest as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "build_lineage",
    "sha256_file",
    "write_lineage",
]
