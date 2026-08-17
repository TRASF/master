"""STFT spectrogram visualizer.

Delegates signal analysis computation to wingbeat_ml.analysis.signal.spectrum.
"""

from __future__ import annotations

from wingbeat_ml.analysis.signal.spectrum import (
    analyze_harmonics,
    compute_spectrogram,
)

__all__ = [
    "compute_spectrogram",
    "analyze_harmonics",
]
