import os
import sys
# Add project root to sys.path so we can import configs/ from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import random
import numpy as np
from configs.mos_config import load_config, normalize_config, apply_reproducibility_environment, resolve_class_weights, generate_experiment_name, resolve_experiment_paths


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

    # W&B workspace panels can be finicky with logged Table objects. Save a CSV too,
    # so the diagnosis rows are always available from the run files/artifacts.
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


def train_supervised(defaults_path="configs/defaults.yaml",
                     model_cfg_path="configs/model.yaml",
                     save_path=None,
                     results_dir=None):

    # 1. Load and Normalize Configurations
    defaults_raw = load_config(defaults_path)
    cfg = normalize_config(defaults_raw)
    model_cfg = load_config(model_cfg_path)

    # 2. Dynamic Experiment Naming & Path Resolution
    exp_name = generate_experiment_name(cfg, mode="Pretrain")
    resolved_paths = resolve_experiment_paths(cfg, exp_name)
    if save_path is None:
        save_path = resolved_paths["save_path"]
    if results_dir is None:
        results_dir = resolved_paths["results_dir"]

    print(f"Experiment Name: {exp_name}")
    print(f"Saving weights to: {save_path}")
    print(f"Saving results to: {results_dir}")

    apply_reproducibility_environment(cfg["reproducibility"])

    import tensorflow as tf
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Dynamic GPU memory allocation enabled for {len(gpus)} GPU(s).")
    except Exception as e:
        print(f"Failed to configure dynamic GPU memory allocation: {e}")

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
            wandb_cfg = cfg.get("wandb", {})
            wandb.init(
                project=wandb_cfg.get("project", "MosSongPlus"),
                config=cfg,
                group=wandb_cfg.get("group"),
                tags=wandb_cfg.get("tags"),
                job_type=wandb_cfg.get("job_type"),
            )

            # Allow W&B Sweep to overwrite config
            for k, v in wandb.config.items():
                if "." in k:
                    parts = k.split(".")
                    if len(parts) == 2 and parts[0] in cfg:
                        cfg[parts[0]][parts[1]] = v
                    elif len(parts) == 3 and parts[0] in cfg and parts[1] in cfg[parts[0]]:
                        cfg[parts[0]][parts[1]][parts[2]] = v

            # Re-apply reproducibility seeds in case they were overridden by sweep
            if cfg["reproducibility"]["enabled"]:
                apply_reproducibility_environment(cfg["reproducibility"])
                seed = cfg["reproducibility"]["seed"]
                random.seed(seed)
                np.random.seed(seed)
                tf.random.set_seed(seed)
                print(f"Sweep override: Re-applied reproducibility seed: {seed}")

            # Re-generate name after any W&B sweep overrides
            base_exp_name = generate_experiment_name(cfg, mode="Pretrain")
            hpf_p = cfg.get("augment", {}).get("high_pass", {}).get("p", 0.0)
            seed = cfg.get("reproducibility", {}).get("seed", cfg.get("train", {}).get("seed", "seed"))
            task = cfg.get("wandb", {}).get("group") or f"{cfg.get('num_classes', 'n')}class"
            exp_name = f"{task}_{base_exp_name}_hpf{hpf_p}_seed{seed}"
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
        nomos_index=cfg["nomos_index"],
        labels_dict=cfg["labels"]
    )

    train_ds, val_ds, test_ds = ds_builder.build(
        split=cfg["dataset"]["split_list"],
        batch_size=cfg["train"]["batch_size"],
        shuffle=cfg["train"]["shuffle"]
    )

    # Dynamic Batch Normalization layer overrides and L2 Regularization from W&B Sweep
    if "model" in cfg:
        import keras
        model_overrides = cfg["model"]
        bn_momentum = model_overrides.get("bn_momentum")
        dense_l2 = model_overrides.get("dense_l2")
        conv_l2 = model_overrides.get("conv_l2")

        layers = model_cfg.get("model", {}).get("mossongplus", {}).get("layers", [])
        conv_idx = 1
        dense_idx = 1
        for layer in layers:
            l_type = layer.get("type")
            if l_type == "conv1d":
                opt_key = f"bn_conv{conv_idx}"
                if opt_key in model_overrides:
                    enabled = bool(model_overrides[opt_key])
                    if enabled and bn_momentum is not None:
                        layer["batch_norm"] = {"momentum": float(bn_momentum)}
                    else:
                        layer["batch_norm"] = enabled
                    print(f"[Dynamic BN Config] Overrode {l_type} layer {conv_idx} batch_norm to {layer['batch_norm']}")
                elif layer.get("batch_norm") and bn_momentum is not None:
                    layer["batch_norm"] = {"momentum": float(bn_momentum)}
                    print(f"[Dynamic BN Config] Updated {l_type} layer {conv_idx} batch_norm momentum to {bn_momentum}")

                if conv_l2 is not None and float(conv_l2) > 0:
                    layer["kernel_regularizer"] = keras.regularizers.l2(float(conv_l2))
                    print(f"[Dynamic L2 Config] Added L2 ({conv_l2}) regularization to {l_type} layer {conv_idx}")
                conv_idx += 1
            elif l_type == "dense":
                opt_key = f"bn_dense{dense_idx}"
                if opt_key in model_overrides:
                    enabled = bool(model_overrides[opt_key])
                    if enabled and bn_momentum is not None:
                        layer["batch_norm"] = {"momentum": float(bn_momentum)}
                    else:
                        layer["batch_norm"] = enabled
                    print(f"[Dynamic BN Config] Overrode {l_type} layer {dense_idx} batch_norm to {layer['batch_norm']}")
                elif layer.get("batch_norm") and bn_momentum is not None:
                    layer["batch_norm"] = {"momentum": float(bn_momentum)}
                    print(f"[Dynamic BN Config] Updated {l_type} layer {dense_idx} batch_norm momentum to {bn_momentum}")

                if dense_l2 is not None and float(dense_l2) > 0:
                    layer["kernel_regularizer"] = keras.regularizers.l2(float(dense_l2))
                    print(f"[Dynamic L2 Config] Added L2 ({dense_l2}) regularization to {l_type} layer {dense_idx}")
                dense_idx += 1

    # 3. Build Model
    print("Building model...")
    model_builder = MosSongPlusModel(model_cfg)
    model = model_builder.build(
        input_shape=(cfg["audio"]["segment_length"], 1),
        output_units=cfg["num_classes"],
        output_activation=cfg["model"]["output_activation"]
    )
    model.summary()

    # 4. Resolve Class Weights
    class_weights_enabled, class_weights = resolve_class_weights(
        cfg["class_weights"],
        ds_builder.class_weights,
        cfg["num_classes"],
        labels_dict=cfg["labels"]
    )

    if class_weights_enabled:
        print(f"Training class counts: {np.bincount(ds_builder.train_labels, minlength=cfg['num_classes']).tolist()}")
        print(f"Using class weights: {np.round(class_weights, 3).tolist()}")
        cfg["resolved_class_weights"] = class_weights.tolist()
    else:
        print("Class weights disabled.")
        cfg["resolved_class_weights"] = None

    # 5. Setup Optimizer and Loss
    optimizer = OptimizerFactory.get_optimizer(cfg)

    # Ensure consistency between activation and from_logits
    if cfg["model"]["output_activation"] is None:
        cfg["loss"]["from_logits"] = True
    elif cfg["model"]["output_activation"] == "softmax":
        cfg["loss"]["from_logits"] = False

    loss_fn = LossFactory.get_loss(cfg)

    # 6. Initialize Train and Evaluator
    print(f"Output activation: {cfg['model']['output_activation']}")
    trainer = Train(
        model,
        optimizer,
        loss_fn,
        train_ds,
        class_weights=class_weights if class_weights_enabled else None,
    )
    evaluator = ModelEvaluator(model, cfg["classes"], loss_fn)

    # 7. Setup Callbacks
    val_x = None
    try:
        val_x, _ = next(iter(val_ds))
    except Exception as e:
        print(f"Warning: Could not extract sample validation batch for activation logging: {e}")

    callbacks = CallbackFactory.get_callbacks(cfg, optimizer, model, save_path, val_x=val_x)

    # 8. Training Loop
    epochs = cfg["train"]["epochs"]
    print(f"\nStarting training for {epochs} epochs...")

    import time
    for epoch in range(epochs):
        # --- Train ---
        start_time = time.time()
        train_metrics = trainer.train_epoch()
        epoch_time = time.time() - start_time

        # --- Eval ---
        val_metrics = evaluator.evaluate_epoch(val_ds)
        val_male_f1 = val_metrics.get("male_f1", 0.0)
        val_female_f1 = val_metrics.get("female_f1", 0.0)

        # New: Precision and Recall groups
        val_m_prec = val_metrics.get("male_prec", 0.0)
        val_m_rec = val_metrics.get("male_rec", 0.0)
        val_f_prec = val_metrics.get("female_prec", 0.0)
        val_f_rec = val_metrics.get("female_rec", 0.0)
        examples_per_second = train_metrics["examples"] / epoch_time
        batches_per_second = train_metrics["batches"] / epoch_time

        # Print metrics with detailed P/R for both sexes
        print(f"Epoch {epoch+1}/{epochs} - loss: {train_metrics['loss']:.4f} - acc: {train_metrics['accuracy']:.4f} | "
              f"val_loss: {val_metrics['loss']:.4f} - val_acc: {val_metrics['accuracy']:.4f} | "
              f"val_f1: {val_metrics['macro_f1']:.3f} | "
              f"Female (P:{val_f_prec:.2f}, R:{val_f_rec:.2f}, F1:{val_female_f1:.2f}) | "
              f"Male (P:{val_m_prec:.2f}, R:{val_m_rec:.2f}, F1:{val_male_f1:.2f}) | "
              f"Time: {epoch_time:.2f}s | "
              f"Batches: {train_metrics['batches']} | "
              f"Examples: {train_metrics['examples']} | "
              f"Throughput: {examples_per_second:.0f} examples/s")

        # --- Manual Callback Execution ---
        callback_values = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "learning_rate": float(tf.keras.backend.get_value(optimizer.learning_rate)),
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
                # Robust monitor name retrieval
                if hasattr(cb, "monitor") and hasattr(cb.monitor, "monitor"):
                    m_name = cb.monitor.monitor
                elif hasattr(cb, "monitors"):
                    m_name = cb.monitors[0]
                else:
                    m_name = "val_score"

                print(f"  --> Saved best weights to {save_path} ({m_name}={callback_values.get(m_name, 0):.4f})")

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
                            name='mossongplus-pretrained',
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults_path", type=str, default="configs/defaults.yaml")
    parser.add_argument("--model_cfg_path", type=str, default="configs/model.yaml")
    args, unknown = parser.parse_known_args()

    train_supervised(
        defaults_path=args.defaults_path,
        model_cfg_path=args.model_cfg_path
    )
