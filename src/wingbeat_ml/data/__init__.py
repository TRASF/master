"""Lazy public data API that avoids audio imports during lightweight tools."""

from importlib import import_module


_EXPORTS = {
    "DataLoader": "loading",
    "DatasetFileLoader": "loading",
    "DatasetSplits": "splits",
    "FileLoader": "audio",
    "generate_manifest": "manifest",
    "load_audio": "audio",
    "load_file": "audio",
    "load_manifest": "manifest",
    "load_raw": "audio",
    "manifest_sha256": "manifest",
    "preprocess_audio": "audio",
    "resample_audio": "audio",
    "segment_audio": "audio",
    "split_files": "splits",
    "to_mono": "audio",
    "verify_manifest": "manifest",
    "write_manifest": "manifest",
}


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(f"wingbeat_ml.data.{module_name}"), name)

__all__ = sorted(_EXPORTS)
