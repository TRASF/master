"""Canonical fine-tuning pipeline."""
from __future__ import annotations

from pathlib import Path

from wingbeat_ml.pipelines.train import run_training
from wingbeat_ml.pipelines.helpers import (
    build_supervised_components,
    evaluate_training_run,
    load_pipeline_configuration,
    make_epoch_printer,
    prepare_training_run,
)

def run_checkpoint_training(
    mode,
    *,
    defaults_path="configs/defaults.yaml",
    model_cfg_path="configs/model.yaml",
    pretrained_weights=None,
    save_path=None,
    results_dir=None,
):
    """Run linear probing or fine-tuning from a shared checkpoint path."""

    config, model_config = load_pipeline_configuration(
        defaults_path,
        model_cfg_path,
    )
    config["training_mode"] = mode
    if config["model"]["output_activation"] is None:
        config["model"]["output_activation"] = "softmax"

    run_label = "LP" if mode == "linear_probe" else "FT"
    artifact_name = (
        "mossongplus-linearprobe"
        if mode == "linear_probe"
        else "mossongplus-finetuned"
    )
    run = prepare_training_run(
        config,
        mode=run_label,
        save_path=save_path,
        results_dir=results_dir,
    )
    components = build_supervised_components(config, model_config)

    weights = Path(
        pretrained_weights
        or config["model"].get("pretrained_weights")
        or config["model"]["checkpoint"]
    )
    if weights.exists():
        print(f"Loading pre-trained weights from {weights}...")
        components.model.load_weights(weights)
    else:
        print(f"WARNING: Pre-trained weights not found at {weights}! Training from scratch.")

    run_training(
        components.model,
        components.train_dataset,
        config,
        evaluate_epoch=lambda: components.evaluator.evaluate_epoch(
            components.validation_dataset
        ),
        on_epoch_end=make_epoch_printer(config),
        class_weights=components.class_weights,
        save_path=run.save_path,
    )
    evaluate_training_run(
        model=components.model,
        evaluator=components.evaluator,
        dataset_builder=components.dataset_builder,
        config=config,
        checkpoint_path=run.save_path,
        results_dir=run.results_dir,
        artifact_name=artifact_name,
        validation_dataset=components.validation_dataset,
        test_dataset=components.test_dataset,
    )

def train_finetune(
    defaults_path="configs/defaults.yaml",
    model_cfg_path="configs/model.yaml",
    pretrained_weights=None,
    save_path=None,
    results_dir=None,
):
    """Fine-tune all model layers with the canonical training runner."""
    return run_checkpoint_training(
        "fine_tune",
        defaults_path=defaults_path,
        model_cfg_path=model_cfg_path,
        pretrained_weights=pretrained_weights,
        save_path=save_path,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    train_finetune()
