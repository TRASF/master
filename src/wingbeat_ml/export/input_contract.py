"""Compatibility shim for input contracts.

Canonical implementation lives in wingbeat_ml.deployment.contracts.
This module re-exports functions and classes to maintain backward compatibility.
"""

from __future__ import annotations

from wingbeat_ml.deployment.contracts import (
    DeploymentInputContract,
    ModelContract,
    dequantize_int8_to_float,
    preprocess_audio_canonical,
    quantize_float_to_int8,
    resolve_deployment_shape,
)

__all__ = [
    "DeploymentInputContract",
    "ModelContract",
    "quantize_float_to_int8",
    "dequantize_int8_to_float",
    "preprocess_audio_canonical",
    "resolve_deployment_shape",
]
