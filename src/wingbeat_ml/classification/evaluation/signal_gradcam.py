"""Compatibility shim for SignalGrad-CAM.

Canonical implementation lives in wingbeat_ml.analysis.model.signal_gradcam.
This module re-exports components to maintain backward compatibility for external callers.
"""

from __future__ import annotations

from wingbeat_ml.analysis.model.signal_gradcam import (
    CamDiagnostics,
    CamExplanation,
    CamExtractionError,
    CamValidationResult,
    ExplanationSample,
    GradCamConfig,
    GradientDiagnostics,
    PredictionResult,
    SignalGradCamAnalyzer,
    _class_id_from_target,
    _ensure_signal_shape,
    collect_real_samples_by_class,
    extract_cam_array_explicit,
    validate_cam,
)

__all__ = [
    "CamExtractionError",
    "GradCamConfig",
    "PredictionResult",
    "CamDiagnostics",
    "CamValidationResult",
    "ExplanationSample",
    "GradientDiagnostics",
    "CamExplanation",
    "SignalGradCamAnalyzer",
    "validate_cam",
    "collect_real_samples_by_class",
    "_class_id_from_target",
    "_ensure_signal_shape",
    "extract_cam_array_explicit",
]
