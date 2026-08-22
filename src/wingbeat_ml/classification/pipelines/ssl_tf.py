"""TensorFlow Semi-Supervised Learning (SSL) pipeline supporting multi-domain label-efficiency benchmark.

Supported arms:
  - 'fixmatch': FixMatch cross-domain SSL
  - 'flexmatch': FlexMatch SSL with Curriculum Pseudo-Labeling (CPL)
  - 'supervised_small': Supervised control trained strictly on the budgeted labeled subset
  - 'full_supervised': Upper-bound control trained on full supervised dataset
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional, Union

import tensorflow as tf

from wingbeat_ml.config import load_config, validate_config
from wingbeat_ml.data.ssl_dataset import build_ssl_datasets
from wingbeat_ml.classification.pipelines.helpers.configuration import load_pipeline_configuration
from wingbeat_ml.registry import build_model
from wingbeat_ml.classification.training.tf_ssl_losses import (
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
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run canonical TensorFlow Semi-Supervised Learning pipeline or baseline control arm.
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
    valid_methods = ("fixmatch", "flexmatch", "supervised_small", "full_supervised")
    if method not in valid_methods:
        raise ValueError(f"Unsupported method {method!r}. Expected one of {valid_methods}.")

    # Override sample limits if explicitly supplied
    overrides = {}
    if epochs is not None:
        overrides["train.epochs"] = epochs
    if train_samples_per_class is not None:
        overrides["ssl.train_samples_per_class"] = train_samples_per_class
        overrides["ssl.labeled_samples_per_class"] = train_samples_per_class
    if val_samples_per_class is not None:
        overrides["ssl.val_samples_per_class"] = val_samples_per_class
    if test_samples_per_class is not None:
        overrides["ssl.test_samples_per_class"] = test_samples_per_class

    if method == "full_supervised":
        overrides["ssl.labeled_samples_per_class"] = None
        overrides["ssl.train_samples_per_class"] = None

    if overrides:
        train_updates = {}
        ssl_updates = {}
        if "train.epochs" in overrides:
            train_updates["epochs"] = overrides["train.epochs"]
        for k in ("train_samples_per_class", "labeled_samples_per_class", "val_samples_per_class", "test_samples_per_class"):
            if f"ssl.{k}" in overrides:
                ssl_updates[k] = overrides[f"ssl.{k}"]

        new_train = app_cfg.train.model_copy(update=train_updates)
        new_ssl = app_cfg.ssl.model_copy(update=ssl_updates)
        app_cfg = app_cfg.model_copy(update={"train": new_train, "ssl": new_ssl})

    # 1. Build TensorFlow model
    model = build_model(app_cfg, model_config)

    # 2. Build multi-domain SSL / baseline datasets
    datasets = build_ssl_datasets(app_cfg)
    train_zipped_ds = datasets["train_ds"]
    labeled_train_ds = datasets["labeled_train_ds"]
    indoor_val_ds = datasets["validation"]["indoor"]
    outdoor_val_ds = datasets["validation"]["outdoor"]
    counts = datasets["counts"]
    manifests = datasets["manifests"]

    if verbose:
        print(f"\n=== TensorFlow {method.upper()} Benchmark Arm ===")
        print(f"Labeled train samples: {counts['labeled_train']} | Unlabeled train samples: {counts['unlabeled_train']}")
        print(f"Indoor val: {counts['indoor_val']} | Outdoor val: {counts['outdoor_val']}")

    # 3. Setup optimizer & loss modules
    optimizer = tf.keras.optimizers.Adam(learning_rate=app_cfg.optimizer.learning_rate)
    if tf.keras.mixed_precision.global_policy().compute_dtype == "float16":
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    flex_layer = None
    if method == "flexmatch":
        flex_layer = TFFlexMatchLoss(
            num_classes=app_cfg.num_classes,
            tau=app_cfg.ssl.tau,
            lambda_u=app_cfg.ssl.lambda_u,
            mapping=app_cfg.ssl.mapping,
        )

    # 4. Training Execution via Trainer.fit()
    from wingbeat_ml.classification.training.trainer import Trainer
    from wingbeat_ml.classification.training.strategies.supervised import SupervisedStrategy
    from wingbeat_ml.classification.training.strategies.ssl_tf import FixMatchStrategy, FlexMatchStrategy

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    if method in ("supervised_small", "full_supervised"):
        strategy = SupervisedStrategy(
            model=model, optimizer=optimizer, loss_fn=loss_fn, config=app_cfg, train_dataset=labeled_train_ds
        )
        train_ds = labeled_train_ds
    elif method == "fixmatch":
        strategy = FixMatchStrategy(model=model, optimizer=optimizer, config=app_cfg)
        train_ds = train_zipped_ds
    elif method == "flexmatch":
        strategy = FlexMatchStrategy(model=model, optimizer=optimizer, config=app_cfg)
        train_ds = train_zipped_ds

    def _evaluate_epoch_ssl():
        val_eval = evaluate_tf_domain_performance(
            model, indoor_val_ds, outdoor_val_ds, num_classes=app_cfg.num_classes
        )
        return {
            "indoor_acc": val_eval["indoor_macro_f1"],
            "outdoor_acc": val_eval["outdoor_macro_f1"],
            "worst_domain_macro_f1": val_eval["worst_domain_macro_f1"],
            "mean_domain_macro_f1": val_eval["mean_domain_macro_f1"],
        }

    trainer = Trainer(model, optimizer, loss_fn, train_ds)
    training_result = trainer.fit(
        model=model,
        train_dataset=train_ds,
        epochs=app_cfg.train.epochs,
        strategy=strategy,
        evaluate_epoch=_evaluate_epoch_ssl,
        config=app_cfg,
    )

    total_epochs = app_cfg.train.epochs
    history = []
    hist_dict = training_result.history
    steps_count = len(hist_dict.get("epoch", []))
    for ep in range(steps_count):
        log_entry = {
            "epoch": ep + 1,
            "duration_sec": round(float(hist_dict.get("epoch_duration_seconds", [0.0]*steps_count)[ep]), 3),
            "train_loss": round(float(hist_dict.get("train_loss", [0.0]*steps_count)[ep]), 4),
            "train_loss_s": round(float(hist_dict.get("train_loss_s", hist_dict.get("train_loss", [0.0]*steps_count))[ep]), 4),
            "train_loss_u": round(float(hist_dict.get("train_loss_u", [0.0]*steps_count)[ep]), 4),
            "mask_ratio": round(float(hist_dict.get("train_mask_ratio", hist_dict.get("mask_ratio", [0.0]*steps_count))[ep]), 4),
            "indoor_acc": round(float(hist_dict.get("val_indoor_acc", hist_dict.get("indoor_acc", [0.0]*steps_count))[ep]), 4),
            "outdoor_acc": round(float(hist_dict.get("val_outdoor_acc", hist_dict.get("outdoor_acc", [0.0]*steps_count))[ep]), 4),
            "worst_domain_macro_f1": round(float(hist_dict.get("val_worst_domain_macro_f1", hist_dict.get("worst_domain_macro_f1", [0.0]*steps_count))[ep]), 4),
            "mean_domain_macro_f1": round(float(hist_dict.get("val_mean_domain_macro_f1", hist_dict.get("mean_domain_macro_f1", [0.0]*steps_count))[ep]), 4),
        }
        history.append(log_entry)
        if verbose:
            print(
                f"Epoch {ep+1}/{total_epochs} | "
                f"Loss: {log_entry['train_loss']:.4f} | "
                f"Indoor Macro-F1: {log_entry['indoor_acc']:.4f} | "
                f"Outdoor Macro-F1: {log_entry['outdoor_acc']:.4f} | "
                f"Worst Domain F1: {log_entry['worst_domain_macro_f1']:.4f}"
            )

    # 5. Final Evaluation on Indoor and Outdoor Test Sets
    final_eval = evaluate_tf_domain_performance(
        model,
        datasets["test"]["indoor"],
        datasets["test"]["outdoor"],
        num_classes=app_cfg.num_classes,
    )

    results = {
        "status": "success",
        "method": method,
        "epochs": total_epochs,
        "subset_seed": app_cfg.ssl.subset_seed,
        "history": history,
        "final_evaluation": final_eval,
        "sample_counts": counts,
        "manifests": manifests,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, f"manifest_{method}_{app_cfg.ssl.subset_seed}.json")
        with open(manifest_path, "w") as f:
            json.dump({"manifests": manifests, "counts": counts}, f, indent=2)

    return results


def main(args=None):
    parser = argparse.ArgumentParser(description="Run TensorFlow SSL and Label-Efficiency Benchmark Arm")
    parser.add_argument(
        "--method",
        type=str,
        default="fixmatch",
        choices=["fixmatch", "flexmatch", "supervised_small", "full_supervised"],
    )
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
    print("\nBenchmark Execution Complete:")
    print(f"Arm: {res['method']} | Worst Domain Macro-F1: {res['final_evaluation']['worst_domain_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
