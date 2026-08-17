"""Compatibility shim for deployment bundle operations.

Canonical implementation lives in wingbeat_ml.deployment.bundle.
This module re-exports functions to maintain backward compatibility.
"""

from __future__ import annotations

from wingbeat_ml.deployment.bundle import (
    create_export_bundle,
    export_input_quantization_header,
    export_ota_config_json,
    export_tflite_to_c_header,
    write_esp32_readme,
)

__all__ = [
    "export_input_quantization_header",
    "export_ota_config_json",
    "export_tflite_to_c_header",
    "write_esp32_readme",
    "create_export_bundle",
]
