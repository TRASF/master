"""TensorFlow Semi-Supervised Learning (SSL) pipeline for FixMatch and FlexMatch."""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, Optional, Union

import tensorflow as tf

from wingbeat_ml.config import load_config, validate_config
from wingbeat_ml.data.ssl_dataset import build_ssl_datasets
from wingbeat_ml.pipelines.helpers.configuration import load_pipeline_configuration
from wingbeat_ml.registry import build_model
from wingbeat_ml.training.tf_ssl_losses import (
    TFFlexMatchLoss,
    evaluate_tf_domain_performance,
    train_tf_fixmatch_step,
    train_tf_flexmatch_step,
)


def run_tf_ssl_pipeline(
    config: Any = None,
    method: Optional[str] = None,
    epochs: Optional[int] = None,
    train_samples_per_class: Optional[int] = None,
    val_samples_per_class: Optional[int] = None,
    test_samples_per_class: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run canonical TensorFlow Semi-Supervised Learning pipeline (FixMatch / FlexMatch).

    Loads sample-limited Indoor (Source / Labeled) and Outdoor (Target / Unlabeled) datasets
    and evaluates algorithm performance on both domains.
    """
    if config is None:
        app_cfg, model_config = load_pipeline_configuration(
            defaults_path="configs/defaults.yaml",
            model_config_path="configs/models/mossong_plus.yaml",
        )
    else:
        app_cfg = validate_config(config)
        model_config = app_cfg.model.model_dump()
    ssl_cfg = app_cfg.ssl

    method = (method or ssl_cfg.method).lower().strip()
    if method not in ("fixmatch", "flexmatch"):
        raise ValueError(f"Unsupported SSL method {method!r}. Expected 'fixmatch' or 'flexmatch'.")

    # Override sample limits if explicitly supplied
    overrides = {}
    if epochs is not None:
        overrides["train.epochs"] = epochs
    if train_samples_per_class is not None:
        overrides["ssl.train_samples_per_class"] = train_samples_per_class
    if val_samples_per_class is not None:
        overrides["ssl.val_samples_per_class"] = val_samples_per_class
    if test_samples_per_class is not None:
        overrides["ssl.test_samples_per_class"] = test_samples_per_class

    if overrides:
        train_updates = {}
        ssl_updates = {}
        if "train.epochs" in overrides:
            train_updates["epochs"] = overrides["train.epochs"]
        if "ssl.train_samples_per_class" in overrides:
            ssl_updates["train_samples_per_class"] = overrides["ssl.train_samples_per_class"]
        if "ssl.val_samples_per_class" in overrides:
            ssl_updates["val_samples_per_class"] = overrides["ssl.val_samples_per_class"]
        if "ssl.test_samples_per_class" in overrides:
            ssl_updates["test_samples_per_class"] = overrides["ssl.test_samples_per_class"]

        new_train = app_cfg.train.model_copy(update=train_updates)
        new_ssl = app_cfg.ssl.model_copy(update=ssl_updates)
        app_cfg = app_cfg.model_copy(update={"train": new_train, "ssl": new_ssl})

    # 1. Build TensorFlow model
    model = build_model(app_cfg, model_config)

    # 2. Build sample-limited SSL datasets
    datasets = build_ssl_datasets(app_cfg)
    train_ds = datasets["train_ds"]
    source_val_ds = datasets["source_val_ds"]
    target_val_ds = datasets["target_val_ds"]
    counts = datasets["counts"]

    if verbose:
        print(f"\n=== TensorFlow {method.upper()} SSL Pipeline ===")
        print(f"Source (Indoor) train samples: {counts['source_train']} | val: {counts['source_val']}")
        print(f"Target (Outdoor) train samples: {counts['target_train']} | val: {counts['target_val']}")

    # 3. Setup optimizer & loss modules
    optimizer = tf.keras.optimizers.Adam(learning_rate=app_cfg.optimizer.learning_rate)
    flex_layer = None
    if method == "flexmatch":
        flex_layer = TFFlexMatchLoss(
            num_classes=app_cfg.num_classes,
            tau=app_cfg.ssl.tau,
            lambda_u=app_cfg.ssl.lambda_u,
            mapping=app_cfg.ssl.mapping,
        )

    # 4. Training Loop
    total_epochs = app_cfg.train.epochs
    history = []

    for epoch in range(total_epochs):
        t0 = time.time()
        steps = 0
        total_loss, total_s, total_u, total_mask = 0.0, 0.0, 0.0, 0.0

        for (x_l, y_l), (x_u_w, x_u_s) in train_ds:
            if method == "fixmatch":
                step_res = train_tf_fixmatch_step(
                    model=model,
                    optimizer=optimizer,
                    x_l=x_l,
                    y_l=y_l,
                    x_u_w=x_u_w,
                    x_u_s=x_u_s,
                    tau=app_cfg.ssl.tau,
                    lambda_u=app_cfg.ssl.lambda_u,
                )
            else:
                step_res = train_tf_flexmatch_step(
                    model=model,
                    flexmatch_layer=flex_layer,
                    optimizer=optimizer,
                    x_l=x_l,
                    y_l=y_l,
                    x_u_w=x_u_w,
                    x_u_s=x_u_s,
                )

            total_loss += step_res["total_loss"]
            total_s += step_res["loss_s"]
            total_u += step_res["loss_u"]
            total_mask += step_res["mask_ratio"]
            steps += 1

        duration = time.time() - t0
        avg_loss = total_loss / max(steps, 1)
        avg_s = total_s / max(steps, 1)
        avg_u = total_u / max(steps, 1)
        avg_mask = total_mask / max(steps, 1)

        val_eval = evaluate_tf_domain_performance(model, source_val_ds, target_val_ds)

        log_entry = {
            "epoch": epoch + 1,
            "duration_sec": round(duration, 3),
            "train_loss": round(avg_loss, 4),
            "train_loss_s": round(avg_s, 4),
            "train_loss_u": round(avg_u, 4),
            "mask_ratio": round(avg_mask, 4),
            "val_source_accuracy": round(val_eval["source_accuracy"], 4),
            "val_target_accuracy": round(val_eval["target_accuracy"], 4),
        }
        history.append(log_entry)

        if verbose:
            print(
                f"Epoch {epoch+1}/{total_epochs} | "
                f"Loss: {log_entry['train_loss']:.4f} (s: {log_entry['train_loss_s']:.4f}, u: {log_entry['train_loss_u']:.4f}) | "
                f"Mask Ratio: {log_entry['mask_ratio']:.4f} | "
                f"Indoor Acc: {log_entry['val_source_accuracy']:.4f} | "
                f"Outdoor Acc: {log_entry['val_target_accuracy']:.4f}"
            )

    # 5. Final Evaluation on Test Sets
    final_eval = evaluate_tf_domain_performance(
        model,
        datasets["source_test_ds"],
        datasets["target_test_ds"],
    )

    return {
        "status": "success",
        "method": method,
        "epochs": total_epochs,
        "history": history,
        "final_evaluation": final_eval,
        "sample_counts": counts,
    }


def main(args=None):
    parser = argparse.ArgumentParser(description="Run TensorFlow FixMatch/FlexMatch SSL Pipeline")
    parser.add_argument("--method", type=str, default="fixmatch", choices=["fixmatch", "flexmatch"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-samples-per-class", type=int, default=100)
    parser.add_argument("--val-samples-per-class", type=int, default=50)

    parsed, _ = parser.parse_known_args(args)
    res = run_tf_ssl_pipeline(
        method=parsed.method,
        epochs=parsed.epochs,
        train_samples_per_class=parsed.train_samples_per_class,
        val_samples_per_class=parsed.val_samples_per_class,
    )
    print(f"SSL Complete. Final Outdoor Test Accuracy: {res['final_evaluation']['target_accuracy']:.4f}")


if __name__ == "__main__":
    main()
