"""Shared evaluation and report coordination for completed runs."""

from pathlib import Path

def make_epoch_printer(config, *, detailed=False):
    """Return the shared console formatter for training epochs."""
    epochs = config["train"]["epochs"]

    def print_epoch(epoch, logs):
        duration = logs["epoch_duration_seconds"]
        message = (
            f"Epoch {epoch + 1}/{epochs} - "
            f"loss: {logs['train_loss']:.4f} - "
            f"acc: {logs['train_accuracy']:.4f} | "
            f"val_loss: {logs['val_loss']:.4f} - "
            f"val_acc: {logs['val_accuracy']:.4f} | "
            f"val_f1: {logs['val_macro_f1']:.3f}"
        )
        if detailed:
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
            message += f" | Time: {duration:.1f}s"
        print(message)

    return print_epoch


def evaluate_training_run(
    *,
    model,
    evaluator,
    dataset_builder,
    config,
    checkpoint_path,
    results_dir,
    artifact_name,
    validation_dataset,
    test_dataset,
):
    """Evaluate and report one completed training run."""
    from wingbeat_ml.evaluation import report_results

    print("\nTraining complete. Running final evaluation on test set...")
    if Path(checkpoint_path).exists():
        model.load_weights(checkpoint_path)

    test_results = evaluator.evaluate_final_test(
        test_dataset,
        save_dir=results_dir,
        return_predictions=True,
    )
    common_file_args = {
        "load_fn": dataset_builder.data_loader.load_file,
        "augmentor": dataset_builder.augmentor,
        "batch_size": config["train"]["batch_size"],
        "save_dir": results_dir,
    }

    print("\nRunning file-level evaluation on test set...")
    file_results = evaluator.evaluate_files(
        file_paths=dataset_builder.test_paths,
        labels=dataset_builder.test_labels,
        **common_file_args,
    )

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


__all__ = ["evaluate_training_run", "make_epoch_printer"]
