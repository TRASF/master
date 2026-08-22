"""Deployment domain module."""

from wingbeat_ml.deployment.tflite import (
    convert_float_tflite,
    convert_full_int8_tflite,
    convert_dynamic_range_tflite,
    make_representative_dataset,
    inspect_tflite_io,
)
from wingbeat_ml.deployment.bundle import create_export_bundle
from wingbeat_ml.deployment.contracts import (
    DeploymentInputContract,
    DeploymentOutputContract,
    DeploymentArtifact,
    ModelContract,
)
from wingbeat_ml.deployment.verify import verify_tflite_against_keras
from wingbeat_ml.deployment.registry import EXPORTERS, register_exporter
from wingbeat_ml.deployment.runtime.tflite import TFLitePredictor, FastTFLiteModel

convert_int8_tflite = convert_full_int8_tflite

__all__ = [
    "convert_float_tflite",
    "convert_full_int8_tflite",
    "convert_int8_tflite",
    "convert_dynamic_range_tflite",
    "make_representative_dataset",
    "inspect_tflite_io",
    "create_export_bundle",
    "DeploymentInputContract",
    "DeploymentOutputContract",
    "DeploymentArtifact",
    "ModelContract",
    "verify_tflite_against_keras",
    "EXPORTERS",
    "register_exporter",
    "TFLitePredictor",
    "FastTFLiteModel",
]
