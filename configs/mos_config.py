"""Compatibility wrapper for canonical runtime configuration helpers."""

from wingbeat_ml.config.runtime import (
    apply_reproducibility_environment,
    configure_training_runtime,
    generate_experiment_name,
    load_config,
    resolve_class_weights,
    resolve_experiment_paths,
)
from wingbeat_ml.config.schema import AppConfig, validate_config


def normalize_config(cfg):
    """Compatibility wrapper returning validated AppConfig."""
    return validate_config(cfg)


__all__ = [
    "AppConfig",
    "apply_reproducibility_environment",
    "configure_training_runtime",
    "generate_experiment_name",
    "load_config",
    "normalize_config",
    "resolve_class_weights",
    "resolve_experiment_paths",
]
