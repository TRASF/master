"""Assembly of existing domain components for canonical pipelines."""

from dataclasses import dataclass
import json
from pathlib import Path
import time


@dataclass(frozen=True)
class SupervisedComponents:
    dataset_builder: object
    train_dataset: object
    validation_dataset: object
    test_dataset: object
    model: object
    loss_fn: object
    evaluator: object
    class_weights: object


def build_dataset_bundle(config, *, return_builder=False):
    """Build configured train, validation, and test datasets."""
    from wingbeat_ml.data.dataset import build_datasets

    dataset = config["dataset"]
    return build_datasets(
        dataset["train_dir"],
        config,
        val_dir=dataset["val_dir"],
        test_dir=dataset["test_dir"],
        return_builder=return_builder,
    )


def build_model_component(config, model_config, *, batch_size=None):
    """Build the configured model through the canonical registry."""
    from wingbeat_ml.registry import build_model

    arguments = {}
    if batch_size is not None:
        arguments["batch_size"] = batch_size
    return build_model(config, model_config, **arguments)


def _synchronize_loss_activation(config):
    activation = config["model"]["output_activation"]
    config["loss"]["from_logits"] = activation is None


def build_supervised_components(
    config,
    model_config,
    *,
    show_class_counts=False,
):
    """Build the common dataset, model, loss, and evaluation stack."""
    from wingbeat_ml.evaluation import ModelEvaluator
    from wingbeat_ml.pipelines.train import resolve_training_class_weights
    from wingbeat_ml.training import LossFactory

    console = str(config.get("logging", {}).get("console", "normal"))
    if console != "quiet":
        print("Setting up datasets...")
    dataset_started = time.perf_counter()
    builder, train, validation, test = build_dataset_bundle(
        config,
        return_builder=True,
    )
    from wingbeat_ml.data.cache import consume_cache_events
    config.setdefault("resolved_timing", {})[
        "dataset_setup_seconds"
    ] = time.perf_counter() - dataset_started
    config["resolved_cache_events"] = consume_cache_events()

    if console != "quiet":
        print("Building model...")
    model_started = time.perf_counter()
    model = build_model_component(config, model_config)
    config.setdefault("resolved_timing", {})[
        "model_build_seconds"
    ] = time.perf_counter() - model_started
    if config.get("logging", {}).get("model_summary", False):
        model.summary()

    class_weights = resolve_training_class_weights(
        config,
        builder,
        show_counts=show_class_counts,
    )
    _synchronize_loss_activation(config)
    loss_fn = LossFactory.get_loss(config)
    evaluator = ModelEvaluator(model, config["classes"], loss_fn)

    if config.get("wandb", {}).get("enabled", False):
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update(
                    {
                        "resolved_timing": config["resolved_timing"],
                        "resolved_cache_events": config["resolved_cache_events"],
                    },
                    allow_val_change=True,
                )
        except ImportError:
            pass

    resolved_run = config.get("resolved_run", {})
    save_path = resolved_run.get("save_path")
    if save_path:
        metadata_path = Path(save_path).parent / "run_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(config, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(metadata_path)

    return SupervisedComponents(
        dataset_builder=builder,
        train_dataset=train,
        validation_dataset=validation,
        test_dataset=test,
        model=model,
        loss_fn=loss_fn,
        evaluator=evaluator,
        class_weights=class_weights,
    )


__all__ = [
    "SupervisedComponents",
    "build_dataset_bundle",
    "build_model_component",
    "build_supervised_components",
]
