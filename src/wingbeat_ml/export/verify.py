"""Compatibility shim for deployment verification.

Canonical implementation lives in wingbeat_ml.deployment.verify.
This module re-exports functions to maintain backward compatibility.
"""

from __future__ import annotations

from wingbeat_ml.deployment.verify import (
    compare_model_pair_agreement,
    compute_metrics,
    evaluate_keras_input_qdq_model,
    evaluate_keras_model,
    evaluate_tflite_model,
    predict_keras_dataset,
    predict_keras_input_qdq_dataset,
    predict_tflite_dataset,
    verify_tflite_against_keras,
)

__all__ = [
    "predict_keras_dataset",
    "predict_tflite_dataset",
    "predict_keras_input_qdq_dataset",
    "compute_metrics",
    "evaluate_keras_model",
    "evaluate_tflite_model",
    "evaluate_keras_input_qdq_model",
    "compare_model_pair_agreement",
    "verify_tflite_against_keras",
]
