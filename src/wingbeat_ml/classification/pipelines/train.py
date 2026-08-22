"""Shared training orchestration used by every training mode."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import tensorflow as tf

from wingbeat_ml.config.runtime import resolve_class_weights
from wingbeat_ml.config.schema import AppConfig
from wingbeat_ml.classification.training import (
    Trainer,
    build_callbacks,
    build_loss,
    build_optimizer,
    build_strategy,
)
from wingbeat_ml.classification.training.strategies.supervised import SupervisedStrategy


def _optimizer_learning_rate(optimizer: Any) -> Any:
    learning_rate = getattr(optimizer, "learning_rate", None)
    if learning_rate is not None:
        return learning_rate
    return optimizer.inner_optimizer.learning_rate


def _normalize_mode(mode: str) -> str:
    normalized = mode.strip().casefold().replace("-", "_")

    aliases = {
        "finetune": "fine_tune",
        "linearprobe": "linear_probe",
    }
    normalized = aliases.get(normalized, normalized)

    allowed = {"pretrain", "linear_probe", "fine_tune"}
    if normalized not in allowed:
        raise ValueError(
            f"Unsupported training mode {mode!r}; "
            f"expected one of {sorted(allowed)}"
        )
    return normalized


def configure_trainable_layers(model: Any, mode: str) -> str:
    """Apply the trainability policy for a training mode."""
    normalized = _normalize_mode(mode)

    if normalized == "linear_probe":
        if not hasattr(model, "layers") or not model.layers:
            raise ValueError("Linear probing requires a model with layers")

        for layer in model.layers[:-1]:
            layer.trainable = False
        model.layers[-1].trainable = True
    else:
        for layer in getattr(model, "layers", []):
            layer.trainable = True

    return normalized


def resolve_training_class_weights(
    config: Any,
    dataset_builder: Any,
    *,
    show_counts: bool = False,
) -> Optional[np.ndarray]:
    """Resolve class weights once and record them in the run config."""
    from wingbeat_ml.config.schema import validate_config

    app_cfg = validate_config(config)
    enabled, weights = resolve_class_weights(
        app_cfg.class_weights,
        dataset_builder.class_weights,
        app_cfg.num_classes,
        labels_dict=app_cfg.labels,
    )

    console = app_cfg.logging.console
    if not enabled:
        if show_counts and console != "quiet":
            print("Class weights disabled.")
        return None

    estimated_counts = getattr(dataset_builder, "class_counts", None)
    if isinstance(estimated_counts, (list, tuple, np.ndarray)):
        counts = np.asarray(estimated_counts, dtype=np.float32)
    else:
        counts = np.bincount(
            dataset_builder.train_labels,
            minlength=app_cfg.num_classes,
        )
    if show_counts and console != "quiet":
        print(f"Training class counts: {counts.tolist()}")
    if console != "quiet":
        print(f"Using class weights: {np.round(weights, 3).tolist()}")

    if isinstance(config, dict):
        raw_cfg = config
        raw_cfg["resolved_class_weights"] = weights.tolist()

    if app_cfg.wandb.enabled:
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update(
                    {
                        "resolved_class_counts": counts.tolist(),
                        "resolved_class_weights": weights.tolist(),
                    },
                    allow_val_change=True,
                )
        except ImportError:
            pass
    return weights


def build_training_components(
    model: Any,
    train_dataset: Any,
    config: Any,
    *,
    class_weights: Optional[Any] = None,
    save_path: Optional[str] = None,
) -> Tuple[Trainer, Any, Any, Dict[str, Any], str]:
    """Build the shared trainer, optimizer, loss and callbacks."""
    from wingbeat_ml.config.schema import validate_config

    app_cfg = validate_config(config)
    mode = configure_trainable_layers(
        model,
        app_cfg.training_mode,
    )

    output_act = app_cfg.model.output_activation
    from_logits = output_act is None

    optimizer = build_optimizer(app_cfg.optimizer)
    if tf.keras.mixed_precision.global_policy().compute_dtype == "float16":
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    loss_fn = build_loss(app_cfg.loss, from_logits=from_logits)
    perf = app_cfg.performance
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        train_dataset,
        class_weights=class_weights,
        steps_per_call=perf.steps_per_call,
        jit_compile=perf.jit_compile,
        profiler=perf.profiler.model_dump() if hasattr(perf.profiler, "model_dump") else perf.profiler,
        profiler_logdir=(
            Path(save_path).parent / "profiler"
            if save_path
            else None
        ),
    )

    callbacks_cfg = app_cfg.callbacks
    callbacks = build_callbacks(
        app_cfg,
        optimizer,
        model,
        save_path,
    )

    return trainer, optimizer, loss_fn, callbacks, mode


def run_training(
    model: Any,
    train_dataset: Any,
    config: Any,
    *,
    evaluate_epoch: Optional[Callable[[], Dict[str, float]]] = None,
    on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
    class_weights: Optional[Any] = None,
    save_path: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Run the shared epoch loop via Trainer.fit() and return its metric history."""
    from wingbeat_ml.config.schema import validate_config

    app_cfg = validate_config(config)
    trainer, optimizer, _, callbacks, _ = build_training_components(
        model,
        train_dataset,
        app_cfg,
        class_weights=class_weights,
        save_path=save_path,
    )
    strategy = SupervisedStrategy(
        None, None, None, None,
        _trainer=trainer,
        evaluate_fn=evaluate_epoch,
    )

    result = trainer.fit(
        model=model,
        train_dataset=train_dataset,
        epochs=app_cfg.train.epochs,
        strategy=strategy,
        callbacks=callbacks,
        evaluate_epoch=evaluate_epoch,
        on_epoch_end=on_epoch_end,
        save_path=save_path,
        config=app_cfg,
    )

    history_len = len(next(iter(result.history.values()))) if result.history else 0
    return [{k: v[i] for k, v in result.history.items()} for i in range(history_len)]


__all__ = [
    "build_strategy",
    "build_training_components",
    "configure_trainable_layers",
    "resolve_training_class_weights",
    "run_training",
]
