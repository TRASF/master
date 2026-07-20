"""Legacy compatibility imports; use wingbeat_ml.data.audio."""

from wingbeat_ml.data.audio import (
    FileLoader,
    load_file,
    resample_audio,
    to_mono,
)

__all__ = ["FileLoader", "load_file", "resample_audio", "to_mono"]
