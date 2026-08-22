"""Assembly of existing domain components for canonical pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union

from wingbeat_ml.config.schema import AppConfig
from wingbeat_ml.data.bundle import DatasetBundle


@dataclass(frozen=True)
class SupervisedComponents:
    dataset_builder: Any
    train_dataset: Any
    validation_dataset: Any
    test_dataset: Any
    model: Any
    loss_fn: Any
    evaluator: Any
    class_weights: Any
    bundle: Optional[DatasetBundle] = None


def build_dataset_bundle(config: AppConfig, *, return_builder: bool = False):
    """Build configured train, validation, and test datasets."""
    from wingbeat_ml.data.dataset import build_datasets

    return build_datasets(
        config.dataset.train_dir,
        config,
        val_dir=config.dataset.val_dir,
        test_dir=config.dataset.test_dir,
        return_builder=return_builder,
    )


def build_model_component(config: AppConfig, model_config: Any, *, batch_size: Optional[int] = None):
    """Build the configured model through the canonical registry."""
    from wingbeat_ml.registry import build_model

    arguments = {}
    if batch_size is not None:
        arguments["batch_size"] = batch_size
    return build_model(config, model_config, **arguments)


def build_warmup_dataset(builder: Any, config: AppConfig):
    """Build a no-augmentation train dataset for warmup epochs."""
    return builder._create_pipeline(
        builder.train_paths,
        builder.train_labels,
        augment=False,
        batch_size=config.train.batch_size,
        shuffle=config.train.shuffle,
        one_hot=True,
    )


def build_supervised_components(
    config: AppConfig,
    model_config: Any,
    *,
    show_class_counts: bool = False,
) -> SupervisedComponents:
    """Build the common dataset, model, loss, and evaluation stack."""
    from wingbeat_ml.classification.evaluation import ModelEvaluator
    from wingbeat_ml.classification.pipelines.train import resolve_training_class_weights
    from wingbeat_ml.classification.training import build_loss

    console = config.logging.console
    if console != "quiet":
        print("Setting up datasets...")
    dataset_started = time.perf_counter()
    builder, train, validation, test = build_dataset_bundle(
        config,
        return_builder=True,
    )
    from wingbeat_ml.data.cache import consume_cache_events

    dataset_setup_seconds = time.perf_counter() - dataset_started
    cache_events = consume_cache_events()
    cache_identities = sorted({event["key"] for event in cache_events})

    timing_dict = dict(getattr(config, "resolved_timing", None) or {})
    timing_dict["dataset_setup_seconds"] = dataset_setup_seconds

    provenance_dict = dict(getattr(config, "resolved_provenance", None) or {})
    provenance_dict["cache_identity"] = cache_identities

    if console != "quiet":
        print(f"Resolved cache identity: {cache_identities}")
        print("Building model...")

    model_started = time.perf_counter()
    model = build_model_component(config, model_config)
    timing_dict["model_build_seconds"] = time.perf_counter() - model_started

    if config.logging.model_summary:
        model.summary()

    class_weights = resolve_training_class_weights(
        config,
        builder,
        show_counts=show_class_counts,
    )

    from_logits = config.model.output_activation is None
    loss_fn = build_loss(config.loss, from_logits=from_logits)
    evaluator = ModelEvaluator(model, config.classes, loss_fn)

    if config.wandb.enabled:
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update(
                    {
                        "resolved_timing": timing_dict,
                        "resolved_cache_events": cache_events,
                        "resolved.cache_identity": cache_identities,
                    },
                    allow_val_change=True,
                )
        except ImportError:
            pass

    resolved_run = getattr(config, "resolved_run", None) or {}
    save_path = resolved_run.get("save_path")
    if save_path:
        metadata_path = Path(save_path).parent / "run_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(metadata_path)

    bundle = DatasetBundle(
        train=train,
        validation=validation,
        test=test,
    )
    return SupervisedComponents(
        dataset_builder=builder,
        train_dataset=train,
        validation_dataset=validation,
        test_dataset=test,
        model=model,
        loss_fn=loss_fn,
        evaluator=evaluator,
        class_weights=class_weights,
        bundle=bundle,
    )


__all__ = [
    "build_warmup_dataset",
    "DatasetBundle",
    "SupervisedComponents",
    "build_dataset_bundle",
    "build_model_component",
    "build_supervised_components",
]
