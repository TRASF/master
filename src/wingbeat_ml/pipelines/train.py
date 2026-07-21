"""Shared training orchestration used by every training mode."""

from __future__ import annotations

import copy
from pathlib import Path
import time
from collections.abc import Callable, Mapping

import numpy as np
import tensorflow as tf

from wingbeat_ml.config.runtime import resolve_class_weights
from wingbeat_ml.training import (
    CallbackFactory,
    LossFactory,
    OptimizerFactory,
    Trainer,
)


def _optimizer_learning_rate(optimizer):
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


def configure_trainable_layers(model, mode: str) -> str:
    """Apply the trainability policy for a training mode."""
    normalized = _normalize_mode(mode)

    if normalized == "linear_probe":
        if not model.layers:
            raise ValueError("Linear probing requires a model with layers")

        for layer in model.layers[:-1]:
            layer.trainable = False
        model.layers[-1].trainable = True
    else:
        for layer in model.layers:
            layer.trainable = True

    return normalized


def resolve_training_class_weights(
    config: dict,
    dataset_builder,
    *,
    show_counts: bool = False,
):
    """Resolve class weights once and record them in the run config."""
    enabled, weights = resolve_class_weights(
        config["class_weights"],
        dataset_builder.class_weights,
        config["num_classes"],
        labels_dict=config["labels"],
    )

    console = str(config.get("logging", {}).get("console", "normal"))
    if not enabled:
        if show_counts and console != "quiet":
            print("Class weights disabled.")
        config["resolved_class_weights"] = None
        return None

    estimated_counts = getattr(dataset_builder, "class_counts", None)
    if isinstance(estimated_counts, (list, tuple, np.ndarray)):
        counts = np.asarray(estimated_counts, dtype=np.float32)
    else:
        counts = np.bincount(
            dataset_builder.train_labels,
            minlength=config["num_classes"],
        )
    if show_counts and console != "quiet":
        print(f"Training class counts: {counts.tolist()}")
    if console != "quiet":
        print(f"Using class weights: {np.round(weights, 3).tolist()}")
    config["resolved_class_counts"] = counts.tolist()
    config["resolved_class_weights"] = weights.tolist()
    if config.get("wandb", {}).get("enabled", False):
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
    model,
    train_dataset,
    config: Mapping[str, object],
    *,
    class_weights=None,
    save_path: str | None = None,
):
    """Build the shared trainer, optimizer, loss and callbacks."""
    mode = configure_trainable_layers(
        model,
        str(config["training_mode"]),
    )

    resolved = copy.deepcopy(dict(config))
    model_cfg = resolved.get("model", {})
    loss_cfg = resolved.setdefault("loss", {})

    if model_cfg.get("output_activation") == "softmax":
        loss_cfg["from_logits"] = False
    elif model_cfg.get("output_activation") is None:
        loss_cfg["from_logits"] = True

    optimizer = OptimizerFactory.get_optimizer(resolved)
    if tf.keras.mixed_precision.global_policy().compute_dtype == "float16":
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    loss_fn = LossFactory.get_loss(resolved)
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        train_dataset,
        class_weights=class_weights,
        steps_per_call=int(
            resolved.get("performance", {}).get("steps_per_call", 20)
        ),
        jit_compile=bool(
            resolved.get("performance", {}).get("jit_compile", False)
        ),
        profiler=resolved.get("performance", {}).get("profiler", {}),
        profiler_logdir=(
            Path(save_path).parent / "profiler"
            if save_path
            else None
        ),
    )

    callback_cfg = resolved.get("callbacks", {})
    needs_checkpoint_path = bool(
        callback_cfg.get("model_checkpoint")
        or (
            callback_cfg.get("early_stopping", {}) or {}
        ).get("restore_best_weights")
    )
    if needs_checkpoint_path and not save_path:
        raise ValueError(
            "save_path is required when checkpoint callbacks are enabled"
        )

    callbacks = CallbackFactory.get_callbacks(
        resolved,
        optimizer,
        model,
        save_path,
    )

    return trainer, optimizer, loss_fn, callbacks, mode


def run_training(
    model,
    train_dataset,
    config: Mapping[str, object],
    *,
    evaluate_epoch: Callable[[], Mapping[str, float]] | None = None,
    on_epoch_end: Callable[[int, Mapping[str, float]], None] | None = None,
    class_weights=None,
    save_path: str | None = None,
) -> list[dict[str, float]]:
    """Run the shared epoch loop and return its metric history."""
    trainer, optimizer, _, callbacks, _ = build_training_components(
        model,
        train_dataset,
        config,
        class_weights=class_weights,
        save_path=save_path,
    )

    epochs = int(config["train"]["epochs"])
    history: list[dict[str, float]] = []
    console = str(config.get("logging", {}).get("console", "normal"))
    jsonl_logger = None
    if config.get("logging", {}).get("jsonl", True) and save_path:
        from wingbeat_ml.pipelines.helpers.reporting import JsonlMetricLogger
        jsonl_logger = JsonlMetricLogger(
            Path(save_path).parent / "metrics.jsonl"
        )

    for epoch in range(epochs):
        started = time.perf_counter()
        train_metrics = trainer.train_epoch()
        train_duration = time.perf_counter() - started

        logs = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "learning_rate": float(
                tf.keras.backend.get_value(
                    _optimizer_learning_rate(optimizer)
                )
            ),
            "epoch_duration_seconds": train_duration,
            "train_duration_seconds": train_duration,
            "global_step": train_metrics["global_step"],
            "steps_per_epoch": train_metrics["batches"],
            "steps_per_call": trainer.steps_per_call,
        }

        for key, value in train_metrics.items():
            logs.setdefault(f"train_{key}", value)

        if evaluate_epoch is not None:
            validation_started = time.perf_counter()
            validation_values = evaluate_epoch()
            logs["validation_duration_seconds"] = (
                time.perf_counter() - validation_started
            )
            for key, value in validation_values.items():
                name = key if key.startswith("val_") else f"val_{key}"
                logs[name] = value

        callback_started = time.perf_counter()
        checkpoint_started = time.perf_counter()
        checkpoint = callbacks.get("model_checkpoint")
        if checkpoint is not None:
            saved = checkpoint.save(model, logs)
            if saved and save_path:
                monitor = getattr(checkpoint, "monitor", None)
                monitor_name = getattr(monitor, "monitor", "val_score")
                monitor_value = float(logs.get(monitor_name, 0.0))
                if console != "quiet":
                    print(
                        f"  --> Saved best weights to {save_path} "
                        f"({monitor_name}={monitor_value:.4f})"
                    )
        logs["checkpoint_duration_seconds"] = (
            time.perf_counter() - checkpoint_started
        )

        reduce_lr = callbacks.get("reduce_lr_on_plateau")
        if reduce_lr is not None:
            reduce_lr.on_epoch_end(logs)

        cosine = callbacks.get("cosine_annealing")
        if cosine is not None:
            cosine.on_epoch_end(logs)

        wandb_logger = callbacks.get("wandb_logger")
        logging_started = time.perf_counter()
        if wandb_logger is not None:
            wandb_logger.on_epoch_end(logs)
        logs["tracking_duration_seconds"] = (
            time.perf_counter() - logging_started
        )
        logs["callback_duration_seconds"] = (
            time.perf_counter() - callback_started
        )
        logs["epoch_total_duration_seconds"] = (
            time.perf_counter() - started
        )

        if jsonl_logger is not None:
            jsonl_logger.log(logs)
        history.append(dict(logs))

        if on_epoch_end is not None:
            on_epoch_end(epoch, logs)

        early_stopping = callbacks.get("early_stopping")
        if early_stopping is not None and early_stopping.check(
            logs,
            epoch=epoch,
        ):
            if console != "quiet":
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    return history


__all__ = [
    "build_training_components",
    "configure_trainable_layers",
    "resolve_training_class_weights",
    "run_training",
]
