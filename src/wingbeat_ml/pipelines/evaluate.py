"""High-level model evaluation pipeline."""

from pathlib import Path

from wingbeat_ml.evaluation import ModelEvaluator, report_results


def evaluate_model(
    model,
    dataset,
    classes,
    *,
    loss_fn=None,
    save_dir=None,
    return_predictions=False,
):
    """Evaluate a model and return the canonical result mapping."""
    evaluator = ModelEvaluator(model, list(classes), loss_fn)
    return evaluator.evaluate_final_test(
        dataset,
        save_dir=save_dir,
        return_predictions=return_predictions,
    )


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

__all__ = ["evaluate_model", "evaluate_training_run"]
