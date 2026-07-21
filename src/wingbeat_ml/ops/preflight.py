"""Host-local checks performed before Launch agents consume work."""

import os
from pathlib import Path


def require_manifest_checksum(expected, actual):
    expected = str(expected or "").strip().casefold()
    actual = str(actual or "").strip().casefold()
    if not expected:
        raise RuntimeError("Expected dataset manifest checksum is not configured")
    if expected != actual:
        raise RuntimeError(
            "Dataset manifest mismatch: "
            f"expected {expected}, found {actual or '<missing>'}"
        )


def manifest_identity(path):
    from wingbeat_ml.data.manifest import load_manifest, manifest_sha256

    return manifest_sha256(load_manifest(path))


def require_writable_directory(path, name):
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if not os.access(directory, os.W_OK):
        raise RuntimeError(f"{name} is not writable: {directory}")
    return directory


def run_host_preflight(*, dataset_dir, runtime_root, cache_dir, manifest_path=None,
                       expected_manifest_sha256=None):
    dataset = Path(dataset_dir).expanduser().resolve()
    if not dataset.is_dir():
        raise RuntimeError(f"Dataset directory not found: {dataset}")
    runtime = require_writable_directory(runtime_root, "Runtime root")
    cache = require_writable_directory(cache_dir, "Cache directory")

    actual = None
    if manifest_path or expected_manifest_sha256:
        if not manifest_path:
            raise RuntimeError("Dataset manifest path is required")
        actual = manifest_identity(manifest_path)
        require_manifest_checksum(expected_manifest_sha256, actual)

    return {
        "dataset_dir": str(dataset),
        "runtime_root": str(runtime),
        "cache_dir": str(cache),
        "manifest_sha256": actual,
    }


__all__ = [
    "manifest_identity",
    "require_manifest_checksum",
    "require_writable_directory",
    "run_host_preflight",
]
