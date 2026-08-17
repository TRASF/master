"""Compatibility shim for TFLite export operations.

Canonical implementation lives in wingbeat_ml.deployment.tflite.
This module re-exports functions to maintain backward compatibility.
"""

from __future__ import annotations

from wingbeat_ml.deployment.tflite import (
    convert_dynamic_range_tflite,
    convert_float_tflite,
    convert_full_int8_tflite,
    convert_int8_tflite,
    convert_int16x8_tflite_experiment,
    create_representative_dataset,
    dump_tflite_analyzer,
    ensure_dir,
    inspect_tflite_io,
    make_representative_dataset,
    run_quantization_debugger,
    save_tflite_model,
)

__all__ = [
    "ensure_dir",
    "save_tflite_model",
    "make_representative_dataset",
    "create_representative_dataset",
    "convert_float_tflite",
    "convert_dynamic_range_tflite",
    "convert_full_int8_tflite",
    "convert_int8_tflite",
    "convert_int16x8_tflite_experiment",
    "inspect_tflite_io",
    "dump_tflite_analyzer",
    "run_quantization_debugger",
]
