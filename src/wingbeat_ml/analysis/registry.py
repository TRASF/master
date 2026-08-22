"""Analysis module registry."""

from __future__ import annotations

from typing import Any
from wingbeat_ml.registry import Registry
from wingbeat_ml.analysis.signal.spectrum import (
    analyze_harmonics,
    compute_psd,
    compute_spectrogram,
)
from wingbeat_ml.analysis.model.signal_gradcam import SignalGradCamAnalyzer
from wingbeat_ml.analysis.edge.complexity import (
    analyze_edge_complexity,
    compute_receptive_field,
)

ANALYZERS = Registry[Any]("analyzer")
ANALYZERS.register("signal_psd", compute_psd)
ANALYZERS.register("signal_spectrogram", compute_spectrogram)
ANALYZERS.register("signal_harmonics", analyze_harmonics)
ANALYZERS.register("model_signal_gradcam", SignalGradCamAnalyzer)
ANALYZERS.register("edge_complexity", analyze_edge_complexity)
ANALYZERS.register("edge_receptive_field", compute_receptive_field)


def register_analyzer(name: str):
    return ANALYZERS.register(name)


__all__ = [
    "ANALYZERS",
    "register_analyzer",
]
