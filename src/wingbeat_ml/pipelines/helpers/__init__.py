"""Focused coordination helpers for canonical pipelines."""

from importlib import import_module


_EXPORTS = {
    "SupervisedComponents": "components",
    "TrainingRunContext": "runtime",
    "build_dataset_bundle": "components",
    "build_model_component": "components",
    "build_supervised_components": "components",
    "evaluate_training_run": "reporting",
    "find_project_root": "configuration",
    "load_pipeline_configuration": "configuration",
    "make_epoch_printer": "reporting",
    "prepare_default_pilot": "configuration",
    "prepare_export_runtime": "runtime",
    "prepare_training_run": "runtime",
    "validate_pipeline_configuration": "configuration",
}


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(
        f"wingbeat_ml.pipelines.helpers.{module_name}"
    )
    return getattr(module, name)


__all__ = sorted(_EXPORTS)
