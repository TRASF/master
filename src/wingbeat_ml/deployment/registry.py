"""Deployment exporters registry."""

from __future__ import annotations

from typing import Any
from wingbeat_ml.registry import Registry
from wingbeat_ml.deployment.tflite import convert_float_tflite, convert_int8_tflite
from wingbeat_ml.deployment.bundle import create_export_bundle

EXPORTERS = Registry[Any]("exporter")
EXPORTERS.register("float_tflite", convert_float_tflite)
EXPORTERS.register("int8_tflite", convert_int8_tflite)
EXPORTERS.register("bundle", create_export_bundle)


def register_exporter(name: str):
    return EXPORTERS.register(name)


__all__ = [
    "EXPORTERS",
    "register_exporter",
]
