"""Shared training orchestration used by every training mode."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping

import tensorflow as tf

from wingbeat_ml.training import (
    CallbackFactory,
    LossFactory,
    OptimizerFactory,
    Trainer,
)


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
        str(config.get("training_mode", "pretrain")),
    )

    resolved = copy.deepcopy(dict(config))
    model_cfg = resolved.get("model", {})
    loss_cfg = resolved.setdefault("loss", {})

    if model_cfg.get("output_activation") == "softmax":
        loss_cfg["from_logits"] = False
    elif model_cfg.get("output_activation") is None:
        loss_cfg["from_logits"] = True

    optimizer = OptimizerFactory.get_optimizer(resolved)
    loss_fn = LossFactory.get_loss(resolved)
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        train_dataset,
        class_weights=class_weights,
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

    epochs = int(config.get("train", {}).get("epochs", 1))
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        started = time.perf_counter()
        train_metrics = trainer.train_epoch()
        duration = time.perf_counter() - started

        logs = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "learning_rate": float(
                tf.keras.backend.get_value(optimizer.learning_rate)
            ),
            "epoch_duration_seconds": duration,
        }

        if evaluate_epoch is not None:
            for key, value in evaluate_epoch().items():
                name = key if key.startswith("val_") else f"val_{key}"
                logs[name] = value

        history.append(dict(logs))

        checkpoint = callbacks.get("model_checkpoint")
        if checkpoint is not None:
            checkpoint.save(model, logs)

        reduce_lr = callbacks.get("reduce_lr_on_plateau")
        if reduce_lr is not None:
            reduce_lr.on_epoch_end(logs)

        cosine = callbacks.get("cosine_annealing")
        if cosine is not None:
            cosine.on_epoch_end(logs)

        wandb_logger = callbacks.get("wandb_logger")
        if wandb_logger is not None:
            wandb_logger.on_epoch_end(logs)

        early_stopping = callbacks.get("early_stopping")
        if early_stopping is not None and early_stopping.check(
            logs,
            epoch=epoch,
        ):
            break

    return history


__all__ = [
    "build_training_components",
    "configure_trainable_layers",
    "run_training",
]
