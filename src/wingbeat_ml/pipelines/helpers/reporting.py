"""Low-overhead console, JSONL, and evaluation coordination."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

from wingbeat_ml.config.schema import AppConfig


class JsonlMetricLogger:
    """Append epoch metrics without rewriting an existing history."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _json_value(value: Any) -> Any:
        item = getattr(value, "item", None)
        if callable(item):
            return item()
        return value

    def log(self, values: Dict[str, Any]) -> None:
        record = {
            key: self._json_value(value)
            for key, value in values.items()
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def make_epoch_printer(config: Any, *, detailed: bool = False) -> Callable[[int, Dict[str, Any]], None]:
    """Return the shared console formatter for training epochs."""
    from wingbeat_ml.config.schema import validate_config

    app_cfg = validate_config(config)
    epochs = app_cfg.train.epochs
    console = app_cfg.logging.console
    interval = app_cfg.logging.epoch_interval

    def print_epoch(epoch: int, logs: Dict[str, Any]) -> None:
        if console == "quiet":
            return
        is_last = epoch + 1 >= epochs
        if not is_last and (epoch + 1) % interval != 0:
            return
        duration = logs["epoch_duration_seconds"]
        message = (
            f"Epoch {epoch + 1}/{epochs} - "
            f"loss: {logs['train_loss']:.4f} - "
            f"acc: {logs['train_accuracy']:.4f} | "
            f"val_loss: {logs['val_loss']:.4f} - "
            f"val_acc: {logs['val_accuracy']:.4f} | "
            f"val_f1: {logs['val_macro_f1']:.3f}"
        )
        if detailed and console == "verbose":
            examples = logs.get("train_examples", 0)
            throughput = examples / duration if duration else 0.0
            message += (
                f" | Female (P:{logs.get('val_female_prec', 0.0):.2f}, "
                f"R:{logs.get('val_female_rec', 0.0):.2f}, "
                f"F1:{logs.get('val_female_f1', 0.0):.2f}) | "
                f"Male (P:{logs.get('val_male_prec', 0.0):.2f}, "
                f"R:{logs.get('val_male_rec', 0.0):.2f}, "
                f"F1:{logs.get('val_male_f1', 0.0):.2f}) | "
                f"Time: {duration:.2f}s | "
                f"Batches: {logs.get('train_batches', 0)} | "
                f"Examples: {examples} | "
                f"Throughput: {throughput:.0f} examples/s"
            )
        else:
            examples = logs.get("train_examples", 0)
            throughput = examples / duration if duration else 0.0
            message += (
                f" | Time: {duration:.2f}s | "
                f"Step: {logs.get('global_step', 0)} | "
                f"Throughput: {throughput:.0f} examples/s"
            )
        print(message)

    return print_epoch


def evaluate_training_run(
    *,
    model: Any,
    evaluator: Any,
    dataset_builder: Any,
    config: Any,
    checkpoint_path: str,
    results_dir: str,
    artifact_name: str,
    validation_dataset: Any,
    test_dataset: Any,
) -> None:
    """Evaluate and report one completed training run."""
    from wingbeat_ml.config.schema import validate_config
    from wingbeat_ml.evaluation import report_results

    app_cfg = validate_config(config)
    console = app_cfg.logging.console
    if console != "quiet":
        print("\nTraining complete. Running final evaluation on test set...")
    if Path(checkpoint_path).exists():
        model.load_weights(checkpoint_path)

    test_results = evaluator.evaluate_final_test(
        test_dataset,
        save_dir=results_dir,
        return_predictions=True,
    )
    file_results = None
    train_file_results = None
    file_enabled = app_cfg.evaluation.file_level.enabled

    if file_enabled:
        common_file_args = {
            "load_fn": dataset_builder.data_loader.load_file,
            "augmentor": dataset_builder.augmentor,
            "batch_size": app_cfg.train.batch_size,
            "save_dir": results_dir,
        }
        if console != "quiet":
            print("\nRunning file-level evaluation on test set...")
        file_results = evaluator.evaluate_files(
            file_paths=dataset_builder.test_paths,
            labels=dataset_builder.test_labels,
            **common_file_args,
        )

        if console != "quiet":
            print("\nRunning file-level evaluation on training set...")
        train_file_results = evaluator.evaluate_files(
            file_paths=dataset_builder.train_paths,
            labels=dataset_builder.train_labels,
            filename="train_file_level_results.yaml",
            **common_file_args,
        )

    report_results(
        model=model,
        test_results=test_results,
        file_results=file_results,
        train_file_results=train_file_results,
        cfg=config,
        ds_builder=dataset_builder,
        save_path=checkpoint_path,
        results_dir=results_dir,
        artifact_name=artifact_name,
        val_ds=validation_dataset,
        test_ds=test_dataset,
        evaluator=evaluator,
    )


__all__ = [
    "JsonlMetricLogger",
    "evaluate_training_run",
    "make_epoch_printer",
]
