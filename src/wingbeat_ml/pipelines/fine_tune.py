"""Canonical fine-tuning pipeline."""

import copy
from pathlib import Path

from wingbeat_ml.pipelines.helpers import (
    build_dataset_bundle,
    build_supervised_components,
    evaluate_training_run,
    load_pipeline_configuration,
    make_epoch_printer,
    prepare_training_run,
)
from wingbeat_ml.pipelines.train import run_training
from wingbeat_ml.training import OptimizerFactory, Trainer


def _build_warmup_dataset(config):
    warmup = copy.deepcopy(config)
    probability = config["train"]["warmup_augment_p"]

    for name in ("noise_overlay", "random_gain"):
        settings = warmup["augment"].get(name)
        if not isinstance(settings, dict):
            continue
        if name == "noise_overlay" and not warmup["augment"]["noise_banks"]:
            continue
        settings["p"] = probability

    print(
        "Setting up warmup dataset "
        f"(forcing augmentation p={probability})..."
    )
    train, _, _ = build_dataset_bundle(warmup)
    return train


def _run_warmup(components, warmup_dataset, config, tracking_run):
    warmup_epochs = config["train"]["warmup_epochs"]
    print(
        "\n--- PHASE 1: Warming up Dense Head for "
        f"{warmup_epochs} epochs ---"
    )

    for layer in components.model.layers[:-1]:
        layer.trainable = False
    components.model.layers[-1].trainable = True

    trainer = Trainer(
        components.model,
        OptimizerFactory.get_optimizer(config),
        components.loss_fn,
        warmup_dataset,
        class_weights=components.class_weights,
    )

    for epoch in range(warmup_epochs):
        train_metrics = trainer.train_epoch()
        validation = components.evaluator.evaluate_epoch(
            components.validation_dataset
        )
        print(
            f"Warmup Epoch {epoch + 1}/{warmup_epochs} - "
            f"loss: {train_metrics['loss']:.4f} - "
            f"acc: {train_metrics['accuracy']:.4f} | "
            f"val_loss: {validation['loss']:.4f} - "
            f"val_acc: {validation['accuracy']:.4f}"
        )
        if tracking_run is not None:
            tracking_run.log({
                "warmup_epoch": epoch,
                "warmup_train_loss": train_metrics["loss"],
                "warmup_train_accuracy": train_metrics["accuracy"],
                "warmup_val_loss": validation["loss"],
                "warmup_val_accuracy": validation["accuracy"],
            })


def train_finetune(
    defaults_path="configs/defaults.yaml",
    model_cfg_path="configs/model.yaml",
    pretrained_weights=None,
    save_path=None,
    results_dir=None,
):
    """Warm up the classification head, then fine-tune the full model."""
    config, model_config = load_pipeline_configuration(
        defaults_path,
        model_cfg_path,
    )
    config["training_mode"] = "fine_tune"
    if config["model"]["output_activation"] is None:
        config["model"]["output_activation"] = "softmax"

    run = prepare_training_run(
        config,
        mode="FT",
        save_path=save_path,
        results_dir=results_dir,
    )
    components = build_supervised_components(config, model_config)
    warmup_dataset = _build_warmup_dataset(config)

    weights = Path(
        pretrained_weights
        or config["model"].get("pretrained_weights")
        or config["model"]["checkpoint"]
    )
    if weights.exists():
        print(f"Loading pre-trained contrastive weights from {weights}...")
        components.model.load_weights(weights)
    else:
        print(
            f"WARNING: Pre-trained weights not found at {weights}! "
            "Training from scratch."
        )

    _run_warmup(
        components,
        warmup_dataset,
        config,
        run.tracking_run,
    )

    epochs = config["train"]["epochs"]
    print("\n--- PHASE 2: Full Fine-Tuning ---")
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
