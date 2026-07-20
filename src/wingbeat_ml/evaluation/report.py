"""Evaluation reporting and optional W&B visualization."""

import os
import csv
import numpy as np

def _mean_report_metric(report, classes, token, metric_name):
    values = [
        float(report[class_name][metric_name])
        for class_name in classes
        if token in class_name and class_name in report
    ]
    return float(np.mean(values)) if values else 0.0


def log_test_report_metrics(wandb, report, metrics, classes, log_grouped_plot=False):
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

    if rows:
        metrics_table = wandb.Table(
            columns=["class", "precision", "recall", "f1", "support"],
            data=rows,
        )
        # Always log the one per-class metrics table
        log_dict["test/per_class_metrics_table"] = metrics_table

        if log_grouped_plot:
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

    # Add sex-specific and class metrics range to log dict if detailed logging is on
    if log_grouped_plot:
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

    wandb.log(log_dict)


def log_class_support_tables(wandb, ds_builder, classes):
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


def log_confusion_matrices(wandb, confusion_matrix, classes, log_all_variants=False):
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

    # Always log the one main confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Final Confusion Matrix")
    wandb.log({"confusion_matrix_plot": wandb.Image(fig)})
    plt.close(fig)

    if log_all_variants:
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


def log_prediction_table(wandb, evaluator, dataset, split_name, cfg):
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


def report_results(model, test_results, file_results, train_file_results, cfg, ds_builder,
                   save_path, results_dir, artifact_name, val_ds=None, test_ds=None, evaluator=None):
    """
    Consolidated reporting function shared across training scripts.
    Logs standard metrics, a metrics table, one confusion matrix, and one model artifact.
    Optional detailed diagnostics can be enabled via configs.
    """
    # Print metrics to stdout
    print(f"Final Test Accuracy: {test_results['metrics']['accuracy']:.4f}")
    print(f"Final Test Macro F1: {test_results['metrics']['macro_f1']:.4f}")
    print(f"Confusion Matrix: Test Accuracy: {test_results['metrics']['accuracy']:.4f} | Macro F1: {test_results['metrics']['macro_f1']:.4f}")
    print(np.array(test_results["confusion_matrix"]))

    print("\nClassification Report:")
    for label, metrics in test_results["report"].items():
        if label in cfg["classes"]:
            print(f"Class {label:20} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1-score']:.4f}")

    print(f"Final File-level Test Accuracy: {file_results['metrics']['accuracy']:.4f}")
    print(f"Final File-level Test Macro F1: {file_results['metrics']['macro_f1']:.4f}")
    print(f"Final File-level Train Accuracy: {train_file_results['metrics']['accuracy']:.4f}")
    print(f"Final File-level Train Macro F1: {train_file_results['metrics']['macro_f1']:.4f}")

    # W&B Logging
    if cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({f"test/{k}": v for k, v in test_results['metrics'].items()})

                # Check for detailed diagnostics flag
                wandb_cfg = cfg.get("wandb", {})
                log_detailed = bool(wandb_cfg.get("log_detailed_diagnostics", False))

                # Log one metrics table (per-class metrics table)
                log_test_report_metrics(wandb, test_results["report"], test_results["metrics"],
                                         cfg["classes"], log_grouped_plot=log_detailed)

                if log_detailed:
                    log_class_support_tables(wandb, ds_builder, cfg["classes"])

                    # Log file-level diagnostics tables
                    for label_prefix, results_dict in [("test", file_results), ("train", train_file_results)]:
                        try:
                            if "file_diagnostics" in results_dict:
                                diags = results_dict["file_diagnostics"]
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
                                    f"{label_prefix}/file_level_diagnostics_table": table,
                                    f"{label_prefix}_file_level_accuracy": results_dict["metrics"]["accuracy"],
                                    f"{label_prefix}_file_level_macro_f1": results_dict["metrics"]["macro_f1"]
                                })
                        except Exception as e:
                            print(f"Failed to log {label_prefix} file level diagnostics to wandb: {e}")

                # Log one confusion matrix
                try:
                    log_confusion_matrices(wandb, test_results["confusion_matrix"], cfg["classes"],
                                           log_all_variants=log_detailed)
                except Exception as e:
                    print(f"Failed to log confusion matrix: {e}")

                # Log optional prediction tables
                if log_detailed and val_ds is not None and test_ds is not None and evaluator is not None:
                    try:
                        log_prediction_table(wandb, evaluator, val_ds, "val", cfg)
                        log_prediction_table(wandb, evaluator, test_ds, "test", cfg)
                    except Exception as e:
                        print(f"Failed to log prediction diagnostics: {e}")

                # Log model artifact (one model artifact)
                if os.path.exists(save_path):
                    try:
                        print(f"Uploading {save_path} to WandB Artifacts...")
                        artifact = wandb.Artifact(
                            name=artifact_name,
                            type='model',
                            metadata={'accuracy': test_results['metrics']['accuracy'], 'macro_f1': test_results['metrics']['macro_f1']}
                        )
                        artifact.add_file(save_path)
                        wandb.log_artifact(artifact, aliases=['best', 'latest'])
                    except Exception as e:
                        print(f"Failed to upload model artifact to wandb: {e}")

                wandb.finish()
        except ImportError:
            pass
