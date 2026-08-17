"""Compatibility shim for standard Grad-CAM.

Canonical implementation lives in wingbeat_ml.analysis.model.gradcam.
This module re-exports components to maintain backward compatibility for external callers.
"""

from __future__ import annotations

from wingbeat_ml.analysis.model.gradcam import (
    GradCamResult,
    aggregate_raw_cams,
    compute_gradcam,
    find_last_conv_layer,
)

__all__ = [
    "GradCamResult",
    "find_last_conv_layer",
    "compute_gradcam",
    "aggregate_raw_cams",
]
