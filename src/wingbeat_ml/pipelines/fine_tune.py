"""Canonical fine-tuning pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from wingbeat_ml.pipelines.helpers import (
    build_supervised_components,
    evaluate_training_run,
    load_pipeline_configuration,
    make_epoch_printer,
    prepare_training_run,
)
from wingbeat_ml.pipelines.train import run_training


def train_finetune(
    defaults_path: Union[str, Path] = "configs/defaults.yaml",
    model_cfg_path: Union[str, Path] = "configs/model.yaml",
    pretrained_weights: Optional[Union[str, Path]] = None,
    save_path: Optional[str] = None,
    results_dir: Optional[str] = None,
):
    """Fine-tune all model layers with the canonical training runner."""
    config, model_config = load_pipeline_configuration(
        defaults_path,
        model_cfg_path,
    )
    output_act = config.model.output_activation or "softmax"
    model_cfg = config.model.model_copy(update={"output_activation": output_act})
    config = config.model_copy(update={"training_mode": "fine_tune", "model": model_cfg})

    run = prepare_training_run(
        config,
        mode="FT",
        save_path=save_path,
        results_dir=results_dir,
    )
    components = build_supervised_components(config, model_config)

    weights_target = (
        pretrained_weights
        or config.model.pretrained_weights
        or config.model.checkpoint
    )
    weights = Path(weights_target) if weights_target else Path("non_existent_weights")
    if weights.exists():
        print(f"Loading pre-trained contrastive weights from {weights}...")
        components.model.load_weights(weights)
    else:
        print(
            f"WARNING: Pre-trained weights not found at {weights}! "
            "Training from scratch."
        )

    epochs = config.train.epochs
    print("\n--- Full Fine-Tuning ---")
    print(f"Starting full fine-tuning for {epochs} epochs...")
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
        artifact_name="mossongplus-finetuned",
        validation_dataset=components.validation_dataset,
        test_dataset=components.test_dataset,
    )


if __name__ == "__main__":
    train_finetune()
