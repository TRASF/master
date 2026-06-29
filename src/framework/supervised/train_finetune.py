import os
import random
import numpy as np
from configs.mos_config import load_config, normalize_config, apply_reproducibility_environment, resolve_class_weights, generate_experiment_name, resolve_experiment_paths
import tensorflow as tf


def _mean_report_metric(report, classes, token, metric_name):
    values = [
        float(report[class_name][metric_name])
        for class_name in classes
        if token in class_name and class_name in report
    ]
    return float(np.mean(values)) if values else 0.0


def _log_test_report_metrics(wandb, report, metrics, classes):
    log_dict = {}
    class_f1 = []
    rows = []

    for class_name in classes:
        class_metrics = report.get(class_name)
        if not class_metrics:
            continue

        precision = float(class_metrics.get("precision", 0.0))
        recall = float(class_metrics.get("recall", 0.0))
        f1 = float(class_metrics.get("f1-score", 0.0))
        support = float(class_metrics.get("support", 0.0))
        class_f1.append(f1)
        rows.append([class_name, precision, recall, f1, support])

        log_dict[f"test_class_precision/{class_name}"] = precision
        log_dict[f"test_class_recall/{class_name}"] = recall
        log_dict[f"test_class_f1/{class_name}"] = f1
        log_dict[f"test_class_support/{class_name}"] = support

    if class_f1:
        log_dict["test/min_class_f1"] = float(np.min(class_f1))
        log_dict["test/max_class_f1"] = float(np.max(class_f1))
        log_dict["test/class_f1_range"] = float(np.max(class_f1) - np.min(class_f1))

    log_dict["test/macro_weighted_f1_gap"] = float(
        metrics.get("weighted_f1", 0.0) - metrics.get("macro_f1", 0.0)
    )
    log_dict["test_male_f1"] = _mean_report_metric(report, classes, "Male", "f1-score")
    log_dict["test_male_precision"] = _mean_report_metric(report, classes, "Male", "precision")
    log_dict["test_male_recall"] = _mean_report_metric(report, classes, "Male", "recall")
    log_dict["test_female_f1"] = _mean_report_metric(report, classes, "Female", "f1-score")
    log_dict["test_female_precision"] = _mean_report_metric(report, classes, "Female", "precision")
    log_dict["test_female_recall"] = _mean_report_metric(report, classes, "Female", "recall")

    if rows:
        metrics_table = wandb.Table(
            columns=["class", "precision", "recall", "f1", "support"],
            data=rows,
        )
        log_dict["test/per_class_metrics_table"] = metrics_table
        log_dict["test/per_class_f1_bar"] = wandb.plot.bar(
            metrics_table,
            "class",
            "f1",
            title="Test F1 by class",
        )
        log_dict["test/per_class_support_bar"] = wandb.plot.bar(
            metrics_table,
            "class",
            "support",
            title="Test support by class",
        )

        try:
            import matplotlib.pyplot as plt

            class_names = [r[0] for r in rows]
            precision = [r[1] for r in rows]
            recall = [r[2] for r in rows]
            f1 = [r[3] for r in rows]
            x = np.arange(len(class_names))
            width = 0.26

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.bar(x - width, precision, width, label="Precision")
            ax.bar(x, recall, width, label="Recall")
            ax.bar(x + width, f1, width, label="F1")
            ax.axhline(metrics.get("macro_f1", 0.0), color="black", linestyle="--", linewidth=1, label="Macro F1")
            ax.axhline(metrics.get("weighted_f1", 0.0), color="gray", linestyle=":", linewidth=1, label="Weighted F1")
            ax.set_ylim(0.0, 1.05)
            ax.set_ylabel("Score")
            ax.set_title("Test precision, recall, and F1 by class")
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.02))
            fig.tight_layout()
            log_dict["test/per_class_precision_recall_f1_grouped"] = wandb.Image(fig)
            plt.close(fig)
        except Exception as e:
            print(f"Failed to log grouped per-class metric plot to wandb: {e}")

    wandb.log(log_dict)


def _log_class_support_tables(wandb, ds_builder, classes):
    split_labels = {
        "train_files": ds_builder.train_labels,
        "val_files": ds_builder.val_labels,
        "test_files": ds_builder.test_labels,
    }
    rows = []
    num_classes = len(classes)

    for split_name, labels in split_labels.items():
        if labels is None:
            continue
        counts = np.bincount(labels, minlength=num_classes)
        for class_idx, count in enumerate(counts):
            rows.append([split_name, classes[class_idx], int(count)])

    table = wandb.Table(columns=["split", "class", "support"], data=rows)
    wandb.log({"data/class_support_by_file": table})

    for split_name, labels in split_labels.items():
        if labels is None:
            continue
        counts = np.bincount(labels, minlength=num_classes)
        split_table = wandb.Table(
            columns=["class", "support"],
            data=[[classes[i], int(counts[i])] for i in range(num_classes)],
        )
        wandb.log({
            f"data/{split_name}_support_bar": wandb.plot.bar(
                split_table,
                "class",
                "support",
                title=f"{split_name} class support",
            )
        })


def _log_confusion_matrices(wandb, confusion_matrix, classes):
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.array(confusion_matrix)

    def plot_matrix(matrix, title, fmt):
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.set_title(title)
        fig.tight_layout()
        return fig

    raw_fig = plot_matrix(cm, "Final Confusion Matrix", "d")
    wandb.log({"test/confusion_matrix_raw": wandb.Image(raw_fig)})
    plt.close(raw_fig)

    row_sums = cm.sum(axis=1, keepdims=True)
    row_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    row_fig = plot_matrix(row_norm, "Final Confusion Matrix - Row Normalized", ".2f")
    wandb.log({"test/confusion_matrix_recall_normalized": wandb.Image(row_fig)})
    plt.close(row_fig)

    col_sums = cm.sum(axis=0, keepdims=True)
    col_norm = np.divide(cm, col_sums, out=np.zeros_like(cm, dtype=float), where=col_sums != 0)
    col_fig = plot_matrix(col_norm, "Final Confusion Matrix - Column Normalized", ".2f")
    wandb.log({"test/confusion_matrix_precision_normalized": wandb.Image(col_fig)})
    plt.close(col_fig)


def _log_prediction_table(wandb, evaluator, dataset, split_name, cfg):
    import csv
    import os

    wandb_cfg = cfg.get("wandb", {})
    max_rows = wandb_cfg.get("prediction_table_max_rows", 5000)
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows <= 0:
            max_rows = None

    diagnostics = evaluator.collect_prediction_diagnostics(
        dataset,
        split_name=split_name,
        max_rows=max_rows,
        sample_rate=cfg["audio"]["sample_rate"],
        include_audio=bool(wandb_cfg.get("log_prediction_audio", False)),
        wandb_module=wandb,
    )

    row_count = len(diagnostics["data"])
    total_count = int(len(diagnostics["y_true"]))
    table = wandb.Table(columns=diagnostics["columns"], data=diagnostics["data"])

    log_payload = {
        f"{split_name}/prediction_table": table,
        f"{split_name}_prediction_table": table,
        f"{split_name}/prediction_table_rows": row_count,
        f"{split_name}/prediction_total_rows": total_count,
    }

    try:
        csv_columns = [c for c in diagnostics["columns"] if c != "audio"]
        csv_indices = [diagnostics["columns"].index(c) for c in csv_columns]
        csv_path = os.path.join(wandb.run.dir, f"{split_name}_prediction_table.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_columns)
            for row in diagnostics["data"]:
                writer.writerow([row[i] for i in csv_indices])
        wandb.save(csv_path, policy="now")
    except Exception as e:
        print(f"Failed to save {split_name} prediction CSV for wandb: {e}")

    error_conf = diagnostics["confidence"][diagnostics["y_true"] != diagnostics["y_pred"]]
    correct_conf = diagnostics["confidence"][diagnostics["y_true"] == diagnostics["y_pred"]]
    confidence_rows = [["correct", float(v)] for v in correct_conf] + [["error", float(v)] for v in error_conf]
    if confidence_rows:
        conf_table = wandb.Table(columns=["outcome", "confidence"], data=confidence_rows)
        log_payload[f"{split_name}/confidence_by_outcome"] = conf_table
        log_payload[f"{split_name}_confidence_by_outcome"] = conf_table

    wandb.log(log_payload)
    return diagnostics


def train_finetune(defaults_path="configs/defaults.yaml",
                   model_cfg_path="configs/model.yaml",
                   pretrained_weights="models/supervised_mossongplus/best_model.weights.h5",
                   save_path=None,
                   results_dir=None):

    # 1. Load and Normalize Configurations
    defaults_raw = load_config(defaults_path)
    cfg = normalize_config(defaults_raw)
    model_cfg = load_config(model_cfg_path)

    # 2. Dynamic Experiment Naming & Path Resolution
    exp_name = generate_experiment_name(cfg, mode="FT")
    resolved_paths = resolve_experiment_paths(cfg, exp_name)
    if save_path is None:
        save_path = resolved_paths["save_path"]
    if results_dir is None:
        results_dir = resolved_paths["results_dir"]
        
    print(f"Experiment Name: {exp_name}")
    print(f"Saving weights to: {save_path}")
    print(f"Saving results to: {results_dir}")

    apply_reproducibility_environment(cfg["reproducibility"])

    # Ensure classification output activation is softmax if not set
    if cfg["model"].get("output_activation") is None:
        cfg["model"]["output_activation"] = "softmax"

    # Ensure consistency between activation and from_logits
    if cfg["model"]["output_activation"] == "softmax":
        cfg["loss"]["from_logits"] = False
    else:
        cfg["loss"]["from_logits"] = True

    # Set seeds for reproducibility
    if cfg["reproducibility"]["enabled"]:
        seed = cfg["reproducibility"]["seed"]
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print(f"Reproducibility enabled. Seed: {seed}")

    if cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb
            wandb.init(project=cfg["wandb"].get("project", "MosSongPlus"), config=cfg)

            # Allow W&B Sweep to overwrite config
            for k, v in wandb.config.items():
                if "." in k:
                    parts = k.split(".")
                    if len(parts) == 2 and parts[0] in cfg:
                        cfg[parts[0]][parts[1]] = v
                    elif len(parts) == 3 and parts[0] in cfg and parts[1] in cfg[parts[0]]:
                        cfg[parts[0]][parts[1]][parts[2]] = v

            # Re-generate name after any W&B sweep overrides
            exp_name = generate_experiment_name(cfg, mode="FT")
            wandb.run.name = exp_name
            # Re-resolve paths to match sweep parameters
            resolved_paths = resolve_experiment_paths(cfg, exp_name)
            save_path = resolved_paths["save_path"]
            results_dir = resolved_paths["results_dir"]
        except ImportError:
            print("WandB is enabled in config but 'wandb' package is not installed.")

    from src.framework.supervised.dataset import SupervisedDataset
    from src.framework.supervised.train_step import Train
    from src.evaluation.evaluate import ModelEvaluator
    from model.mossongplus import MosSongPlusModel
    from src.framework.optimizer import OptimizerFactory
    from src.framework.loss import LossFactory
    from src.framework.callbacks import CallbackFactory

    # 2. Setup Dataset
    print("Setting up datasets...")
    ds_builder = SupervisedDataset(
        dataset_dir=cfg["dataset"].get("train_dir") or cfg["dataset"]["indoor"],
        val_dir=cfg["dataset"]["val_dir"],
        test_dir=cfg["dataset"]["test_dir"],
        sample_rate=cfg["audio"]["sample_rate"],
        segment_length=cfg["audio"]["segment_length"],
        classes=cfg["classes"],
        noise_dirs=cfg["augment"]["noise_banks"],
        augment_cfg=cfg["augment"],
        seed=cfg["reproducibility"]["seed"],
        deterministic=cfg["reproducibility"]["deterministic_data"],
        nomos_index=cfg["nomos_index"]
    )

    train_ds, val_ds, test_ds = ds_builder.build(
        split=cfg["dataset"]["split_list"],
        batch_size=cfg["train"]["batch_size"],
        shuffle=cfg["train"]["shuffle"]
    )

    # Warmup dataset builder (with high augmentation probability)
    import copy
    warmup_augment_cfg = copy.deepcopy(cfg["augment"])
    warmup_p = cfg["train"].get("warmup_augment_p", 1.0)

    # Force warmup augmentation probabilities for active augmentations
    for key in ["noise_overlay", "random_gain"]:
        if key in warmup_augment_cfg and isinstance(warmup_augment_cfg[key], dict):
            # Only force if the augmentation is valid (e.g. noise banks exist for noise_overlay)
            if key == "noise_overlay" and not warmup_augment_cfg.get("noise_banks"):
                continue
            warmup_augment_cfg[key]["p"] = warmup_p

    print(f"Setting up warmup dataset (forcing augmentation p={warmup_p})...")
    ds_builder_warmup = SupervisedDataset(
        dataset_dir=cfg["dataset"].get("train_dir") or cfg["dataset"]["indoor"],
        val_dir=cfg["dataset"]["val_dir"],
        test_dir=cfg["dataset"]["test_dir"],
        sample_rate=cfg["audio"]["sample_rate"],
        segment_length=cfg["audio"]["segment_length"],
        classes=cfg["classes"],
        noise_dirs=cfg["augment"]["noise_banks"],
        augment_cfg=warmup_augment_cfg,
        seed=cfg["reproducibility"]["seed"],
        deterministic=cfg["reproducibility"]["deterministic_data"],
        nomos_index=cfg["nomos_index"]
    )

    train_ds_warmup, _, _ = ds_builder_warmup.build(
        split=cfg["dataset"]["split_list"],
        batch_size=cfg["train"]["batch_size"],
        shuffle=cfg["train"]["shuffle"]
    )

    # 3. Build Model
    print("Building model...")
    model_builder = MosSongPlusModel(model_cfg)
    model = model_builder.build(
        input_shape=(cfg["audio"]["segment_length"], 1),
        output_units=cfg["num_classes"],
        output_activation=cfg["model"]["output_activation"]
    )

    # LOAD PRETRAINED WEIGHTS
    if os.path.exists(pretrained_weights):
        print(f"Loading pre-trained contrastive weights from {pretrained_weights}...")
        model.load_weights(pretrained_weights)
    else:
        print(f"WARNING: Pre-trained weights not found at {pretrained_weights}! Training from scratch.")

    model.summary()

    # 4. Resolve Class Weights
    class_weights_enabled, class_weights = resolve_class_weights(
        cfg["class_weights"],
        ds_builder.class_weights,
        cfg["num_classes"],
    )

    if class_weights_enabled:
        print(f"Using class weights: {np.round(class_weights, 3).tolist()}")
        cfg["resolved_class_weights"] = class_weights.tolist()
    else:
        cfg["resolved_class_weights"] = None

    # 5. Setup Optimizer and Loss
    loss_fn = LossFactory.get_loss(cfg)
    evaluator = ModelEvaluator(model, cfg["classes"], loss_fn)

    epochs = cfg["train"]["epochs"]
    warmup_epochs = cfg["train"].get("warmup_epochs", 15)

    # -------------------------------------------------------------
    # PHASE 1: LINEAR PROBING (Warmup Dense Head)
    # -------------------------------------------------------------
    print(f"\n--- PHASE 1: Warming up Dense Head for {warmup_epochs} epochs ---")
    for layer in model.layers[:-1]:
        layer.trainable = False
    model.layers[-1].trainable = True

    optimizer_phase1 = OptimizerFactory.get_optimizer(cfg)
    trainer_phase1 = Train(model, optimizer_phase1, loss_fn, train_ds_warmup, class_weights=class_weights if class_weights_enabled else None)

    for epoch in range(warmup_epochs):
        train_metrics = trainer_phase1.train_epoch()
        val_metrics = evaluator.evaluate_epoch(val_ds)
        print(f"Warmup Epoch {epoch+1}/{warmup_epochs} - loss: {train_metrics['loss']:.4f} - acc: {train_metrics['accuracy']:.4f} | "
              f"val_loss: {val_metrics['loss']:.4f} - val_acc: {val_metrics['accuracy']:.4f}")

        if cfg.get("wandb", {}).get("enabled", False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "warmup_epoch": epoch,
                        "warmup_train_loss": train_metrics["loss"],
                        "warmup_train_accuracy": train_metrics["accuracy"],
                        "warmup_val_loss": val_metrics["loss"],
                        "warmup_val_accuracy": val_metrics["accuracy"]
                    })
            except ImportError:
                pass

    # -------------------------------------------------------------
    # PHASE 2: FULL FINE-TUNING
    # -------------------------------------------------------------
    print(f"\n--- PHASE 2: Full Fine-Tuning ---")
    for layer in model.layers:
        layer.trainable = True

    # Lower the learning rate drastically for fine-tuning
    cfg["optimizer"]["learning_rate"] = 1e-3
    optimizer_phase2 = OptimizerFactory.get_optimizer(cfg)

    # Extract validation batch for activation logging
    val_x = None
    try:
        val_x, _ = next(iter(val_ds))
    except Exception as e:
        print(f"Warning: Could not extract sample validation batch for activation logging: {e}")

    trainer_phase2 = Train(model, optimizer_phase2, loss_fn, train_ds, class_weights=class_weights if class_weights_enabled else None)
    callbacks = CallbackFactory.get_callbacks(cfg, optimizer_phase2, model, save_path, val_x=val_x)

    print(f"Starting full fine-tuning for {epochs} epochs...")

    import time
    for epoch in range(epochs):
        # --- Train ---
        start_time = time.time()
        train_metrics = trainer_phase2.train_epoch()
        epoch_time = time.time() - start_time

        # --- Eval ---
        val_metrics = evaluator.evaluate_epoch(val_ds)

        print(f"Epoch {epoch+1}/{epochs} - loss: {train_metrics['loss']:.4f} - acc: {train_metrics['accuracy']:.4f} | "
              f"val_loss: {val_metrics['loss']:.4f} - val_acc: {val_metrics['accuracy']:.4f} | "
              f"val_f1: {val_metrics['macro_f1']:.3f} | "
              f"Time: {epoch_time:.1f}s")

        # --- Manual Callback Execution ---
        callback_values = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "learning_rate": float(tf.keras.backend.get_value(optimizer_phase2.learning_rate)),
            "epoch_duration_seconds": epoch_time,
        }

        # Populate all val_metrics dynamically
        for k, v in val_metrics.items():
            key = f"val_{k}" if not k.startswith("val_") else k
            callback_values[key] = v

        # Checkpoint
        if 'model_checkpoint' in callbacks:
            cb = callbacks['model_checkpoint']
            if cb.save(model, callback_values):
                print(f"  --> Saved best weights to {save_path} (val_macro_f1={callback_values['val_macro_f1']:.4f})")

        # Reduce LR
        if 'reduce_lr_on_plateau' in callbacks:
            callbacks['reduce_lr_on_plateau'].on_epoch_end(callback_values)

        # Cosine Annealing
        if 'cosine_annealing' in callbacks:
            callbacks['cosine_annealing'].on_epoch_end(callback_values)

        # Wandb
        if 'wandb_logger' in callbacks:
            callbacks['wandb_logger'].on_epoch_end(callback_values)

        # Early Stopping
        if 'early_stopping' in callbacks:
            if callbacks['early_stopping'].check(callback_values):
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

    # 9. Final Evaluation on Test Set
    print("\nTraining complete. Running final evaluation on test set...")
    if os.path.exists(save_path):
        model.load_weights(save_path)
    test_results = evaluator.evaluate_final_test(test_ds, save_dir=results_dir, return_predictions=True)
    print(f"Final Test Accuracy: {test_results['metrics']['accuracy']:.4f}")
    print(f"Final Test Macro F1: {test_results['metrics']['macro_f1']:.4f}")
    print(f"Confusion Matrix: Test Accuracy: {test_results['metrics']['accuracy']:.4f} | Macro F1: {test_results['metrics']['macro_f1']:.4f}")
    print(np.array(test_results["confusion_matrix"]))

    print("\nClassification Report:")
    for label, metrics in test_results["report"].items():
        if label in cfg["classes"]:
            print(f"Class {label:20} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1-score']:.4f}")

    # File-level evaluation and diagnostic tracking
    print("\nRunning file-level evaluation on test set...")
    file_results = evaluator.evaluate_files(
        file_paths=ds_builder.test_paths,
        labels=ds_builder.test_labels,
        load_fn=ds_builder.data_loader.load_file,
        augmentor=ds_builder.augmentor,
        batch_size=cfg["train"]["batch_size"],
        save_dir=results_dir
    )
    print(f"Final File-level Test Accuracy: {file_results['metrics']['accuracy']:.4f}")
    print(f"Final File-level Test Macro F1: {file_results['metrics']['macro_f1']:.4f}")

    print("\nRunning file-level evaluation on training set...")
    train_file_results = evaluator.evaluate_files(
        file_paths=ds_builder.train_paths,
        labels=ds_builder.train_labels,
        load_fn=ds_builder.data_loader.load_file,
        augmentor=ds_builder.augmentor,
        batch_size=cfg["train"]["batch_size"],
        save_dir=results_dir,
        filename="train_file_level_results.yaml"
    )
    print(f"Final File-level Train Accuracy: {train_file_results['metrics']['accuracy']:.4f}")
    print(f"Final File-level Train Macro F1: {train_file_results['metrics']['macro_f1']:.4f}")

    if cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb
            if wandb.run is not None:
                # Log final test metrics and diagnosis-ready artifacts
                wandb.log({f"test/{k}": v for k, v in test_results['metrics'].items()})
                _log_test_report_metrics(wandb, test_results["report"], test_results["metrics"], cfg["classes"])
                _log_class_support_tables(wandb, ds_builder, cfg["classes"])

                # Log test file-level diagnostics table
                try:
                    if "file_diagnostics" in file_results:
                        diags = file_results["file_diagnostics"]
                        columns = [
                            "file_name", "true_label", "pred_label", "is_correct", 
                            "loss", "confidence", "total_segments", "correct_segments", 
                            "segment_accuracy", "incorrect_segments"
                        ]
                        data = [
                            [
                                d["file_name"], d["true_label"], d["pred_label"], d["is_correct"],
                                d["loss"], d["confidence"], d["total_segments"], d["correct_segments"],
                                d["segment_accuracy"], str(d["incorrect_segments"])
                            ]
                            for d in diags
                        ]
                        table = wandb.Table(columns=columns, data=data)
                        wandb.log({
                            "test/file_level_diagnostics_table": table,
                            "test_file_level_accuracy": file_results["metrics"]["accuracy"],
                            "test_file_level_macro_f1": file_results["metrics"]["macro_f1"]
                        })
                except Exception as e:
                    print(f"Failed to log test file level diagnostics to wandb: {e}")

                # Log train file-level diagnostics table
                try:
                    if "file_diagnostics" in train_file_results:
                        diags = train_file_results["file_diagnostics"]
                        columns = [
                            "file_name", "true_label", "pred_label", "is_correct", 
                            "loss", "confidence", "total_segments", "correct_segments", 
                            "segment_accuracy", "incorrect_segments"
                        ]
                        data = [
                            [
                                d["file_name"], d["true_label"], d["pred_label"], d["is_correct"],
                                d["loss"], d["confidence"], d["total_segments"], d["correct_segments"],
                                d["segment_accuracy"], str(d["incorrect_segments"])
                            ]
                            for d in diags
                        ]
                        table = wandb.Table(columns=columns, data=data)
                        wandb.log({
                            "train/file_level_diagnostics_table": table,
                            "train_file_level_accuracy": train_file_results["metrics"]["accuracy"],
                            "train_file_level_macro_f1": train_file_results["metrics"]["macro_f1"]
                        })
                except Exception as e:
                    print(f"Failed to log train file level diagnostics to wandb: {e}")

                try:
                    _log_confusion_matrices(wandb, test_results["confusion_matrix"], cfg["classes"])
                    if "y_true" in test_results and "y_pred" in test_results:
                        wandb.log({
                            "test/confusion_matrix_interactive": wandb.plot.confusion_matrix(
                                y_true=test_results["y_true"],
                                preds=test_results["y_pred"],
                                class_names=cfg["classes"],
                            )
                        })
                except Exception as e:
                    print(f"Failed to log diagnostic confusion matrices to wandb: {e}")

                try:
                    _log_prediction_table(wandb, evaluator, val_ds, "val", cfg)
                    _log_prediction_table(wandb, evaluator, test_ds, "test", cfg)
                except Exception as e:
                    print(f"Failed to log prediction diagnostics to wandb: {e}")

                # Upload the best model weights to WandB Artifacts (Model Registry)
                if os.path.exists(save_path):
                    try:
                        print(f"Uploading {save_path} to WandB Artifacts...")
                        artifact = wandb.Artifact(
                            name='mossongplus-finetuned',
                            type='model',
                            metadata={'accuracy': test_results['metrics']['accuracy'], 'macro_f1': test_results['metrics']['macro_f1']}
                        )
                        artifact.add_file(save_path)
                        wandb.log_artifact(artifact, aliases=['best', 'latest'])
                    except Exception as e:
                        print(f"Failed to upload model artifact to wandb: {e}")

                # Try to log confusion matrix as a wandb custom chart
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(test_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                                xticklabels=cfg["classes"], yticklabels=cfg["classes"])
                    plt.ylabel('Actual')
                    plt.xlabel('Predicted')
                    plt.title(f"Final Confusion Matrix: Test Accuracy: {test_results['metrics']['accuracy']:.4f} | Macro F1: {test_results['metrics']['macro_f1']:.4f}")
                    wandb.log({"confusion_matrix_plot": wandb.Image(fig)})
                    plt.close(fig)
                except Exception as e:
                    print(f"Failed to plot confusion matrix for wandb: {e}")

                wandb.finish()
        except ImportError:
            pass

if __name__ == "__main__":
    train_finetune()
